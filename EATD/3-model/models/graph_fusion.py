import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import os

def _env(k, d):
    return os.environ.get(k, d)
def _envi(k, d):
    return int(os.environ.get(k, d))
def _envb(k, d=0):
    return int(os.environ.get(k, d)) != 0

# ----------------------------------------------------------------------------
# Configurable EATD graph model.  Default flags reproduce the EATD baseline
# (graph_fusion.py) exactly; flags morph it toward the paper's SEGA structure.
#
#   READOUT   : dual (baseline, concat[text_summary,audio_summary])
#               text  (single text_summary node, paper-style readout)
#               vsum  (single virtual summary node aggregating text nodes, paper)
#               vsum_a(virtual summary aggregating text AND audio_summary -> rescue)
#   PROXY     : 0/1  insert proxy node per round (audio->proxy->text); paper noise-suppression
#   DIRECTED  : 0/1  make audio->proxy->text and text->summary edges directed (paper DAG)
#   LGAT      : number of GAT layers (default 2, paper value)
#   NODE_LN   : 0/1  LayerNorm on node features before GAT
#   AUDIO_BN  : 0/1  BatchNorm on raw audio (default 1, baseline)
#   DROP_EMO  : 0/1  remove emotion(7)+gender structural nodes
#   NOAUDIO   : 0/1  zero out audio (multimodal ablation)
#   SUMM_SELF : 0/1  give the virtual summary a self loop (default 1)
# ----------------------------------------------------------------------------

class GraphFusion(nn.Module):
    def __init__(self, config, text_model=None, audio_model=None):
        super(GraphFusion, self).__init__()

        self.text_embed_size = config['text_embed_size']
        self.audio_embed_size = config['audio_embed_size']
        self.text_hidden_dims = config['text_hidden_dims']
        self.audio_hidden_dims = config['audio_hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.bidirectional = config['bidirectional']

        # --- flags ---
        self.f_readout  = _env('READOUT', 'dual')
        self.f_proxy    = _envb('PROXY', 0)
        self.f_directed = _envb('DIRECTED', 0)
        self.f_lgat     = _envi('LGAT', 2)
        self.f_node_ln  = _envb('NODE_LN', 0)
        self.f_audio_bn = _envb('AUDIO_BN', 1)
        self.f_drop_emo = _envb('DROP_EMO', 0)
        self.f_summ_self= _envb('SUMM_SELF', 1)
        self.f_summ_reads_audio = _envb('SUMM_READS_AUDIO', 0)
        self.f_resid    = _envb('RESID', 0)
        self.f_gat_ln   = _envb('GAT_LN', 0)

        H = self.text_hidden_dims  # 256

        self.text_attention = nn.Sequential(nn.Linear(H, 1), nn.Tanh())
        self.text_bilstm = nn.GRU(self.text_embed_size, H // 2, num_layers=self.rnn_layers,
                                  dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)

        self.audio_attention = nn.Sequential(nn.Linear(self.audio_hidden_dims, 1), nn.Tanh())
        self.audio_bilstm = nn.GRU(self.audio_embed_size, self.audio_hidden_dims // 2, num_layers=self.rnn_layers,
                                   dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)

        self.question_attention = nn.Sequential(nn.Linear(H, 1), nn.Tanh())
        self.question_bilstm = nn.GRU(self.text_embed_size, H // 2, num_layers=self.rnn_layers,
                                      dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)

        self.emotion_lookup = nn.Embedding(7, 256)
        self.gender_lookup = nn.Embedding(2, 256)

        self.bn = nn.BatchNorm1d(self.audio_embed_size)

        if self.f_node_ln:
            self.node_ln = nn.LayerNorm(256)

        # readout dimension
        n_read = {'dual': 2, 'text': 1, 'vsum': 1, 'vsum_a': 1}[self.f_readout]
        self.classifier = nn.Sequential(
            nn.Linear(256 * n_read, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, self.num_classes, bias=False),
        )

        self.gats = nn.ModuleList([GAT(256, 256) for _ in range(max(2, self.f_lgat))])
        if self.f_gat_ln:
            self.gat_lns = nn.ModuleList([nn.LayerNorm(256) for _ in range(max(2, self.f_lgat))])

    # ------------------------------------------------------------------
    def _node_layout(self):
        names = ['text0','text1','text2','audio0','audio1','audio2',
                 'question0','question1','question2']
        if self.f_proxy:
            names += ['proxy0','proxy1','proxy2']
        names += ['tsum','asum','qsum']
        if not self.f_drop_emo:
            names += ['emo0','emo1','emo2','emo3','emo4','emo5','emo6','gender']
        if self.f_readout in ('vsum','vsum_a'):
            names += ['vsummary']
        return names

    def _static_edges(self):
        # (dst, src, kind) ; kind 'bi' sets both directions, 'dir' only dst<-src
        E = []
        for i in range(3):
            E.append(('text%d'%i, 'question%d'%i, 'bi'))
        if self.f_proxy:
            k = 'dir' if self.f_directed else 'bi'
            for i in range(3):
                E.append(('proxy%d'%i, 'audio%d'%i, k))   # proxy aggregates audio
                E.append(('text%d'%i,  'proxy%d'%i, k))   # text aggregates proxy
        else:
            for i in range(3):
                E.append(('audio%d'%i, 'question%d'%i, 'bi'))  # baseline: audio<->question
        for i in range(3):
            E.append(('question%d'%i, 'qsum', 'bi'))
            E.append(('text%d'%i,  'tsum', 'bi'))
            E.append(('audio%d'%i, 'asum', 'bi'))
        if not self.f_drop_emo:
            E.append(('asum', 'gender', 'bi'))
        if self.f_readout in ('vsum','vsum_a'):
            k = 'dir' if self.f_directed else 'bi'
            for i in range(3):
                E.append(('vsummary', 'text%d'%i, k))   # summary aggregates text (paper)
            if self.f_readout == 'vsum_a' or self.f_summ_reads_audio:
                E.append(('vsummary', 'asum', k))       # rescue: summary also sees audio
        return E

    def _build_adj(self, emotion_feat, batch_size, names, idx):
        N = len(names)
        A = torch.zeros(N, N)
        for (dst, src, kind) in self._static_edges():
            A[idx[dst], idx[src]] = 1.0
            if kind == 'bi':
                A[idx[src], idx[dst]] = 1.0
        # self loops
        for k in range(N):
            if names[k] == 'vsummary' and not self.f_summ_self:
                continue
            A[k, k] = 1.0
        A = A.cuda()
        adj = A.unsqueeze(0).repeat(batch_size, 1, 1).clone()
        if not self.f_drop_emo:
            for b in range(batch_size):
                ep = emotion_feat[b]  # [3,7]
                for r in range(3):
                    for c in range(7):
                        w = ep[r, c]
                        adj[b, idx['text%d'%r], idx['emo%d'%c]] = w
                        adj[b, idx['emo%d'%c], idx['text%d'%r]] = w  # symmetric, as baseline
        return adj

    def forward(self, text_feat_mixup, audio_feat_mixup, question_feat_mixup,
                gender_feat_mixup, emotion_feat_mixup, process=None):

        if process == 'train':
            sample_num = text_feat_mixup.shape[1]
            output_mixup, text_contra_feat, audio_contra_feat = [], [], []
        elif process == 'test':
            sample_num = 1

        for i in range(sample_num):
            if process == 'train':
                text_feat = text_feat_mixup[:, i]
                audio_feat = audio_feat_mixup[:, i]
                question_feat = question_feat_mixup[:, i]
                gender_feat = gender_feat_mixup[:, i]
                emotion_feat = emotion_feat_mixup[:, i]
            else:
                text_feat = text_feat_mixup
                audio_feat = audio_feat_mixup
                question_feat = question_feat_mixup
                gender_feat = gender_feat_mixup
                emotion_feat = emotion_feat_mixup

            batch_size = text_feat.shape[0]

            text_feat = self.gather_feature(text_feat, self.text_bilstm, self.text_attention, 'text')
            audio_feat = self.gather_feature(audio_feat, self.audio_bilstm, self.audio_attention, 'audio')
            if os.environ.get('NOAUDIO'):
                audio_feat = audio_feat * 0.0
            question_feat = self.gather_feature(question_feat, self.question_bilstm, self.question_attention, 'text')

            emotion_idx = torch.tensor([0,1,2,3,4,5,6], dtype=torch.int64)
            emotion_embedding = self.emotion_lookup(emotion_idx.cuda())
            emotion_embeddings = emotion_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
            gender_embeddings = self.gender_lookup(gender_feat.long())
            if gender_embeddings.dim() == 3:
                gender_embeddings = gender_embeddings[:, 0, :]
            gender_embeddings = gender_embeddings.reshape(batch_size, 256)  # [b,256]

            text_summary_feat = torch.mean(text_feat, 1)
            audio_summary_feat = torch.mean(audio_feat, 1)
            question_summary_feat = torch.mean(question_feat, 1)

            names = self._node_layout()
            idx = {n: k for k, n in enumerate(names)}

            feat = {}
            for j in range(3):
                feat['text%d'%j] = text_feat[:, j, :]
                feat['audio%d'%j] = audio_feat[:, j, :]
                feat['question%d'%j] = question_feat[:, j, :]
                if self.f_proxy:
                    feat['proxy%d'%j] = text_feat[:, j, :]  # paper: proxy initialized from t_i
            feat['tsum'] = text_summary_feat
            feat['asum'] = audio_summary_feat
            feat['qsum'] = question_summary_feat
            if not self.f_drop_emo:
                for c in range(7):
                    feat['emo%d'%c] = emotion_embeddings[:, c, :]
                feat['gender'] = gender_embeddings
            if self.f_readout in ('vsum','vsum_a'):
                feat['vsummary'] = text_summary_feat  # paper: init = average of t_i

            concat_feat = torch.stack([feat[n] for n in names], dim=1)  # [b,N,256]
            if self.f_node_ln:
                concat_feat = self.node_ln(concat_feat)

            adj = self._build_adj(emotion_feat, batch_size, names, idx)

            h = concat_feat
            for g in range(self.f_lgat):
                hg = self.gats[g](h, adj)
                if self.f_resid:
                    hg = hg + h
                if self.f_gat_ln:
                    hg = self.gat_lns[g](hg)
                h = hg
            graph_feat2 = h

            if self.f_readout == 'dual':
                read = torch.cat([graph_feat2[:, idx['tsum'], :], graph_feat2[:, idx['asum'], :]], -1)
                text_pnn_out = graph_feat2[:, idx['tsum'], :]
                audio_pnn_out = graph_feat2[:, idx['asum'], :]
            elif self.f_readout == 'text':
                read = graph_feat2[:, idx['tsum'], :]
                text_pnn_out = read; audio_pnn_out = read
            else:  # vsum / vsum_a
                read = graph_feat2[:, idx['vsummary'], :]
                text_pnn_out = read; audio_pnn_out = read

            output = self.classifier(read)

            if process == 'train':
                output_mixup.append(output.unsqueeze(1))
                text_contra_feat.append(text_pnn_out.unsqueeze(1))
                audio_contra_feat.append(audio_pnn_out.unsqueeze(1))

        if process == 'train':
            return torch.cat(output_mixup, 1), torch.cat(text_contra_feat, 1), torch.cat(audio_contra_feat, 1)
        else:
            return output, text_pnn_out, audio_pnn_out

    def gather_feature(self, x, lstm_func, att_func, flag):
        bs, cate_num, seq_len, embed_size = x.shape
        x = x.reshape(-1, seq_len, embed_size)
        x_mask = (1 - torch.eq(x[:, :, 0], torch.tensor(0.)).float()).unsqueeze(-1)
        if flag == 'audio' and self.f_audio_bn:
            x = self.bn(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        x = x * x_mask
        x, _ = lstm_func(x)
        seq_att = self.softmask(att_func(x).squeeze(-1), x_mask.squeeze(-1)).unsqueeze(-1)
        gather_x = torch.matmul(seq_att.permute(0, 2, 1), x).squeeze(1)
        gather_x = gather_x.reshape(bs, cate_num, -1)
        return gather_x

    def softmask(self, score, x_mask):
        score_exp = torch.exp(score) * x_mask
        score_sum = torch.sum(score_exp, -1, keepdim=True)
        return score_exp / (score_sum + 1e-6)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.3, alpha=0.01, nheads=8):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid // nheads, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=1).view(B, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = attention * adj
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime) if self.concat else h_prime
