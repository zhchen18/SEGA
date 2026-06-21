"""
SEGA-lite for DAIC-WOZ — keeps the paper's "shadow" but leans on tricks.

Paper shadow kept:
  * per-modality sequence encoder (BiGRU over QA turns)
  * a per-turn PROXY node that distills the other modalities into the text path
  * a structural-element GAT graph + a SUMMARY node used as readout

Tricks (the part that actually moves the number):
  * features standardized offline (done in features/prepare_data.py)
  * input LayerNorm per modality
  * gated fusion whose gates init at 0  -> at init the model is EXACTLY text-only,
    so multimodal can only help, never collapse (directly fixes the prior failure)
  * optional gated concat readout of modality summaries (EATD-style, the variant that
    actually let audio reach the prediction)
  * optional gender side-info node

`fusion` switch selects the regime:
  'text'   : text path only (baseline)
  'proxy'  : modalities injected only via the per-turn proxy, summary-node readout
  'concat' : text graph + gated concat of {audio,video,question} summaries
  'both'   : proxy + concat
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------------------------- GAT (dense) -----------------------------------
class GraphAttentionLayer(nn.Module):
    """Efficient additive GAT: e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j), no N^2xF tensor."""
    def __init__(self, in_features, out_features, dropout, alpha=0.01, concat=True):
        super().__init__()
        self.dropout, self.out_features, self.concat = dropout, out_features, concat
        self.W = nn.Parameter(torch.empty(in_features, out_features)); nn.init.xavier_uniform_(self.W, gain=1.414)
        self.a_src = nn.Parameter(torch.empty(out_features, 1)); nn.init.xavier_uniform_(self.a_src, gain=1.414)
        self.a_dst = nn.Parameter(torch.empty(out_features, 1)); nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h_in, adj):
        h = torch.matmul(h_in, self.W)                       # B,N,F
        e = self.leakyrelu((h @ self.a_src) + (h @ self.a_dst).transpose(1, 2))  # B,N,N
        e = e.masked_fill(adj <= 0, -9e15)
        att = F.softmax(e, dim=2)
        att = F.dropout(att, self.dropout, training=self.training) * adj
        h_prime = torch.matmul(att, h)
        return F.elu(h_prime) if self.concat else h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.3, nheads=8):
        super().__init__()
        self.dropout = dropout
        self.heads = nn.ModuleList([GraphAttentionLayer(nfeat, nhid // nheads, dropout) for _ in range(nheads)])

    def forward(self, x, adj):
        x = torch.cat([h(x, adj) for h in self.heads], dim=2)
        return F.dropout(x, self.dropout, training=self.training)


# ----------------------------- encoder ---------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hid, ln_input=True, rnn_layers=1, drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim) if ln_input else nn.Identity()
        self.drop = nn.Dropout(drop)
        self.gru = nn.GRU(in_dim, hid // 2, rnn_layers, batch_first=True, bidirectional=True)

    def forward(self, x, length):
        x = self.drop(self.ln(x))
        packed = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out                                            # B, Tmax_in_batch, hid


# ----------------------------- model -----------------------------------------
class SegaLite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        H = cfg["hidden"]
        self.fusion = cfg["fusion"]
        d = cfg["dims"]
        ln = cfg.get("ln_input", True)
        edrop = cfg.get("enc_dropout", 0.0)
        self.use_emotion = cfg.get("use_emotion", False)
        d = {**d, "emotion": cfg.get("emotion_dim", 4)}
        # which non-text modalities feed the fusion (proxy / concat / late)
        side = list(cfg.get("mm_mods", ["question", "audio", "video"]))
        if self.use_emotion and "emotion" not in side:
            side = side + ["emotion"]
        self.side = side
        self.mods = ["text"] + side
        self.enc = nn.ModuleDict({
            m: ModalityEncoder(d[m], H, ln_input=ln, drop=edrop) for m in self.mods
        })
        # proxy: project each side modality to H, gated-add to text (gates init 0)
        self.proxy_proj = nn.ModuleDict({m: nn.Linear(H, H) for m in self.side})
        gi = cfg.get("gate_init", 0.0)
        self.proxy_gate = nn.ParameterDict({m: nn.Parameter(torch.full((H,), gi)) for m in self.side})
        self.proxy_ln = nn.LayerNorm(H)

        self.gat1 = GAT(H, H, dropout=cfg.get("gat_dropout", 0.3))
        self.gat2 = GAT(H, H, dropout=cfg.get("gat_dropout", 0.3))

        # concat-fusion gates (scalar per modality, init 0)
        self.cat_gate = nn.ParameterDict({m: nn.Parameter(torch.zeros(1)) for m in self.side})

        # readout width
        self.use_gender = cfg.get("use_gender", False)
        gdim = cfg.get("gender_dim", 16)
        if self.use_gender:
            self.gender_emb = nn.Embedding(2, gdim)
        cdrop, C = cfg.get("cls_dropout", 0.5), cfg["num_classes"]
        # optional bottleneck: compress each modality summary (denoise noisy a/v)
        self.mm_bn = int(cfg.get("mm_bottleneck", 0))
        if self.mm_bn:
            self.bn_proj = nn.ModuleDict({m: nn.Linear(H, self.mm_bn) for m in self.side})
        msz = self.mm_bn if self.mm_bn else H
        mm_w = len(self.side) * msz + (gdim if self.use_gender else 0)

        if self.fusion in ("late", "boost", "boost_graph"):
            # text head + gated multimodal head. 'late' = joint CE; 'boost'/'boost_graph' =
            # mm trained on detached text logits (text path can never be polluted).
            # 'boost_graph': mm comes from a paper-style DIRECTED proxy graph (in-graph fusion).
            self.fc_text = nn.Sequential(nn.Linear(H, H), nn.ReLU())
            self.cls_text = nn.Sequential(nn.Dropout(cdrop), nn.Linear(H, C))
            # boost_graph mm input = graph summary (H) + gender; late/boost mm input = mm_w
            # (mm_w already includes gender, so do NOT add gdim again).
            fc_in = (H + (gdim if self.use_gender else 0)) if self.fusion == "boost_graph" else mm_w
            self.fc_mm = nn.Sequential(nn.Linear(fc_in, H), nn.ReLU())
            self.cls_mm = nn.Sequential(nn.Dropout(cdrop), nn.Linear(H, C))
            self.g_mm = nn.Parameter(torch.zeros(1))
        if self.fusion == "boost_graph":
            # directed multimodal graph: q/a/v(/emo) -> proxy(text-init) -> text -> summary
            self.gat_mm = nn.ModuleList([GAT(H, H, dropout=cfg.get("gat_dropout", 0.3))
                                         for _ in range(int(cfg.get("mm_gat_layers", 3)))])
            self.proxy_init_ln = nn.LayerNorm(H)
        if self.fusion not in ("late", "boost", "boost_graph"):
            # single-head early-fusion modes (text / concat / both / proxy) only.
            # NB: must NOT build this for late/boost/boost_graph — the unused head would
            # consume RNG at init and shift the training dropout sequence.
            width = H
            if self.fusion in ("concat", "both"):
                width += len(self.side) * H
            if self.use_gender:
                width += gdim
            self.fc = nn.Sequential(nn.Linear(width, H), nn.ReLU())
            self.cls = nn.Sequential(nn.Dropout(cdrop), nn.Linear(H, C))

    @staticmethod
    def _mask_len(text):
        mask = (text.abs().sum(-1) > 0).float()              # B,T
        return mask, mask.sum(1).long().clamp(min=1)

    def _summary(self, hid, mask):
        m = mask[:, :hid.shape[1]].unsqueeze(-1)
        return (hid * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def _adj(self, B, N, device):
        # EATD-style: per-turn nodes 0..N-2, summary node N-1; summary<->all + self loops
        adj = torch.eye(N, device=device)
        adj[N - 1, :] = 1.0
        adj[:, N - 1] = 1.0
        return adj.unsqueeze(0).expand(B, N, N)

    def _mm_adj(self, B, T, nm, mask, device):
        # node blocks order: [text, *side, proxy] each length T, then 1 summary node.
        # directed (adj[i,j]=1 => node i aggregates from node j):
        #   proxy_i <- text_i, side_m,i ;  text_i <- proxy_i ;  summary <- text_i ; self loops
        N = (nm + 1) * T + 1
        i = torch.arange(T, device=device)
        adj = torch.zeros(N, N, device=device)
        text_idx, proxy_idx, summ = i, nm * T + i, N - 1
        adj[proxy_idx, text_idx] = 1.0
        for b in range(1, nm):                       # side modality blocks
            adj[proxy_idx, b * T + i] = 1.0
        adj[text_idx, proxy_idx] = 1.0
        adj[summ, text_idx] = 1.0
        adj[torch.arange(N), torch.arange(N)] = 1.0
        adj = adj.unsqueeze(0).expand(B, N, N)
        nmask = torch.cat([mask.repeat(1, nm + 1), torch.ones(B, 1, device=device)], dim=1)
        adj = adj * nmask.unsqueeze(1) * nmask.unsqueeze(2)
        return adj + torch.eye(N, device=device).unsqueeze(0)   # self-loops survive masking

    def _mm_graph(self, H, mask):
        mods = ["text"] + self.side
        proxy = self.proxy_init_ln(H["text"])                   # paper: text-initialized proxy
        node_turns = torch.cat([H[m] for m in mods] + [proxy], dim=1)
        summ0 = self._summary(H["text"], mask).unsqueeze(1)
        nodes = torch.cat([node_turns, summ0], dim=1)
        adj = self._mm_adj(nodes.shape[0], H["text"].shape[1], len(mods), mask, nodes.device)
        for gat in self.gat_mm:
            nodes = gat(nodes, adj)
        return nodes[:, -1, :]                                  # summary node (3 hops from a/v/q)

    def forward(self, batch):
        mask, length = self._mask_len(batch["text"])
        H = {m: self.enc[m](batch[m], length) for m in self.mods}
        Tb = H["text"].shape[1]
        mask = mask[:, :Tb]

        # ---- per-turn carrier: text, optionally + gated proxy distillation -----
        carrier = H["text"]
        if self.fusion in ("proxy", "both"):
            inj = 0
            for m in self.side:
                inj = inj + self.proxy_gate[m] * torch.tanh(self.proxy_proj[m](H[m]))
            carrier = self.proxy_ln(H["text"] + inj)

        # ---- structural graph over carrier turns + summary node ----------------
        summ = self._summary(carrier, mask).unsqueeze(1)      # B,1,H
        nodes = torch.cat([carrier, summ], dim=1)             # B,T+1,H
        adj = self._adj(nodes.shape[0], nodes.shape[1], nodes.device)
        # mask padded turn nodes out of the graph (keep summary col/row)
        nmask = torch.cat([mask, torch.ones(mask.shape[0], 1, device=mask.device)], dim=1)
        adj = adj * nmask.unsqueeze(1) * nmask.unsqueeze(2)
        adj = adj + torch.eye(nodes.shape[1], device=adj.device).unsqueeze(0)  # keep self loops for padded
        g = self.gat2(self.gat1(nodes, adj), adj)
        readout = g[:, -1, :]                                 # summary node

        # ---- boost_graph: mm residual from a paper-style DIRECTED proxy graph --
        if self.fusion == "boost_graph":
            text_feat = self.fc_text(readout)
            text_logit = self.cls_text(text_feat)
            mm_vec = self._mm_graph(H, mask)
            if self.use_gender:
                mm_vec = torch.cat([mm_vec, self.gender_emb(batch["gender"])], dim=-1)
            mm_logit = self.g_mm * self.cls_mm(self.fc_mm(mm_vec))
            return text_feat, text_logit + mm_logit, {"text_logit": text_logit, "mm_logit": mm_logit}

        # ---- late / boost fusion: text head + gated mm residual head -----------
        if self.fusion in ("late", "boost"):
            text_feat = self.fc_text(readout)
            text_logit = self.cls_text(text_feat)
            mm = []
            for m in self.side:
                s = self._summary(H[m], mask)
                if self.mm_bn:
                    s = torch.tanh(self.bn_proj[m](s))
                mm.append(self.cat_gate[m] * s)
            if self.use_gender:
                mm.append(self.gender_emb(batch["gender"]))
            mm_logit = self.g_mm * self.cls_mm(self.fc_mm(torch.cat(mm, dim=-1)))
            aux = {"text_logit": text_logit, "mm_logit": mm_logit} if self.fusion == "boost" else {}
            return text_feat, text_logit + mm_logit, aux

        # ---- early/mid fusion: single head over concatenated features ----------
        feats = [readout]
        if self.fusion in ("concat", "both"):
            for m in self.side:
                feats.append(self.cat_gate[m] * self._summary(H[m], mask))
        if self.use_gender:
            feats.append(self.gender_emb(batch["gender"]))
        feat = self.fc(torch.cat(feats, dim=-1))
        return feat, self.cls(feat), {}
