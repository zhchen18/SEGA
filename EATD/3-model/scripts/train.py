"""
EATD SEGA / SEGA++ trainer (unified).

One invocation trains the 3-fold cross-validation for a single seed and reports the
per-fold and mean best-dev Macro-F1 (the paper's protocol: best epoch per fold,
averaged over folds).  Two run modes, selected by the `AUG` env var:

  AUG=none   SEGA    -- real data only.
  AUG=llm    SEGA++  -- paper section 3.3 "A Helping Hand from LLMs": the LLM
                        paraphrase of each answer is mixed into training via manifold
                        mixup with same-class samples (MIXMODE/MIXSYN/MIXA/AUGW below).

Backbone flags (PROXY/DIRECTED/READOUT/LGAT) are read by models/graph_fusion.py and are
set by run_sega.sh / run_sega++.sh to the structural-element graph configuration.

If SAVEDIR (+ SAVETAG) is set, the best-dev model of every fold is bundled into one
checkpoint  SAVEDIR/SAVETAG_s<seed>.pt  that scripts/eval_ckpt.py can load to reproduce
the seed's metrics without retraining.
"""
import os, sys, random, itertools, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from models.graph_fusion import GraphFusion

DATA = os.path.join(ROOT, "data", "EATD-Features")
FOLDSPLIT = os.path.join(ROOT, "data", "train_test_fold_split.txt")


def _envf(k, d): return float(os.environ.get(k, d))
def _envi(k, d): return int(os.environ.get(k, d))
def _envs(k, d): return os.environ.get(k, d)


# -------- augmentation / regularization knobs (active only when AUG=llm) --------
AUG      = _envs('AUG', 'none')               # none | llm
MIXMODE  = _envs('MIXMODE', 'same')           # realsyn | same | both
MIXSYN   = _envi('MIXSYN', 1)                 # same-branch partner endpoint = its LLM paraphrase
MIXA     = _envf('MIXA', 0.1)                 # Beta(MIXA,MIXA) mixup coefficient
AUGW     = _envf('AUGW', 1.0)                 # CE weight of the mixed views
AUDIOSYN = _envi('AUDIOSYN', 1)              # also mix the synthetic audio endpoint
LS       = _envf('LS', 0.0)                   # label smoothing


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
def _stack_views(views):
    keys = ['text', 'audio', 'question', 'gender', 'emotion']
    out = {k: np.stack([v[k] for v in views], axis=1) for k in keys}        # [N,S,...]
    Y = np.stack([v['y'] for v in views], axis=1)                           # [N,S]
    W = np.stack([v['w'] for v in views], axis=1)                           # [N,S]
    return out['text'], out['audio'], out['question'], out['gender'], out['emotion'], Y, W


def build_views(train_data):
    """Real view (+ paper section 3.3 LLM-mixup views when AUG=llm).  Returns stacked
    [N,S,...] arrays, soft labels Y [N,S] and per-(sample,view) CE weights W [N,S].
    AUG=none uses no RNG here, reproducing the SEGA baseline exactly."""
    text   = train_data['text_feature']
    audio  = train_data['audio_feature']
    quest  = train_data['question_feature']
    gender = train_data['gender_feature']
    emo    = train_data['emotion_feature']
    label  = train_data['target']
    syntext  = train_data['syntext_feature']
    synaudio = train_data['synaudio_feature']
    N = label.shape[0]
    ones = np.ones(N, dtype=np.float32)

    real = dict(text=text, audio=audio, question=quest, gender=gender, emotion=emo,
                y=label.astype(np.float32), w=ones.copy())
    views = [real]

    if AUG == 'llm':
        depr = np.where(label == 1.)[0]
        ctrl = np.where(label == 0.)[0]
        # (a) real <-> LLM-paraphrase interpolation (label preserving)
        if MIXMODE in ('realsyn', 'both'):
            a = np.random.beta(MIXA, MIXA, size=(N, 1, 1, 1)).astype(np.float32)
            mt = text * a + syntext * (1 - a)
            ma = audio * a + (synaudio if AUDIOSYN else audio) * (1 - a)
            views.append(dict(text=mt, audio=ma, question=quest, gender=gender, emotion=emo,
                              y=label.astype(np.float32), w=ones.copy() * AUGW))
        # (b) within-class manifold mixup; with MIXSYN the same-class partner endpoint is
        #     its LLM paraphrase -> injects the synthetic data as diverse manifold endpoints.
        if MIXMODE in ('same', 'both'):
            part = np.array([np.random.choice(depr if label[i] == 1. else ctrl) for i in range(N)])
            a = np.random.beta(MIXA, MIXA, size=(N, 1, 1, 1)).astype(np.float32)
            pt = syntext[part] if MIXSYN else text[part]
            pa = (synaudio if AUDIOSYN else audio)[part] if MIXSYN else audio[part]
            mt = text * a + pt * (1 - a)
            ma = audio * a + pa * (1 - a)
            views.append(dict(text=mt, audio=ma, question=quest, gender=gender, emotion=emo,
                              y=label.astype(np.float32), w=ones.copy() * AUGW))

    return _stack_views(views)


def soft_ce(prob, y_scalar, w):
    """prob [M,2], y_scalar [M] in [0,1], w [M]; label-smoothed soft cross-entropy."""
    y1 = y_scalar
    if LS > 0:
        y1 = y1 * (1 - LS) + 0.5 * LS
    y_bin = torch.stack([1. - y1, y1], dim=-1)
    logp = torch.log(prob + 1e-8)
    ce = -(y_bin * logp).sum(-1)
    return (w * ce).sum() / (w.sum() + 1e-8)


# ---------------------------------------------------------------------------
def get_eatd_mm_data(idxs, raw_data, split):
    (rt, ra, rq, rg, rst, rsa, re_, ry) = raw_data
    text, audio, target, quest, gender = [], [], [], [], []
    syntext, synaudio, emotion = [], [], []
    split2samples = {'train': [0, 1, 2, 3, 4, 5], 'test': [0, 1, 4, 5]}
    for idx in idxs:
        if ry[idx] == 0:
            text.append(rt[idx]); audio.append(ra[idx]); target.append(ry[idx])
            quest.append(rq[idx]); gender.append(rg[idx])
            syntext.append(rst[idx]); synaudio.append(rsa[idx]); emotion.append(re_[idx])
        else:
            tp = itertools.permutations(rt[idx], rt[idx].shape[0])
            ap = itertools.permutations(ra[idx], ra[idx].shape[0])
            for k, (tper, aper) in enumerate(zip(tp, ap)):
                if k in split2samples[split]:
                    text.append(np.array(list(tper))); audio.append(np.array(list(aper)))
                    target.append(ry[idx]); quest.append(rq[idx]); gender.append(rg[idx])
                    syntext.append(rst[idx]); synaudio.append(rsa[idx]); emotion.append(re_[idx])
    return {
        'text_feature': np.array(text), 'audio_feature': np.array(audio),
        'question_feature': np.array(quest), 'gender_feature': np.array(gender),
        'emotion_feature': np.array(emotion), 'syntext_feature': np.array(syntext),
        'synaudio_feature': np.array(synaudio), 'target': np.array(target),
    }


def load_dataset(path):
    def L(n): return np.load(f'{path}/{n}')
    tr_text  = L('train_text_simple_roberta_768.npy')
    tr_audio = L('train_audio_new_features_b_3_len_88.npz')['arr_0']
    tr_quest = L('train_questions_b_3_len_768.npz')['arr_0']
    tr_gen   = L('train_gender_b.npz')['arr_0']
    tr_y     = L('train_labels.npz')['arr_0']
    tr_st    = L('train_syntext_roberta_768.npy')
    tr_sa    = L('train_synaudio_opensmile_88.npz')['arr_0']
    tr_emo   = L('train_emotion_probs_b_3_7.npz')['arr_0']
    va_text  = L('valid_text_simple_roberta_768.npy')
    va_audio = L('valid_audio_new_features_b_3_len_88.npz')['arr_0']
    va_quest = L('valid_questions_b_3_len_768.npz')['arr_0']
    va_gen   = L('valid_gender_b.npz')['arr_0']
    va_y     = L('valid_labels.npz')['arr_0']
    va_st    = L('valid_syntext_roberta_768.npy')
    va_sa    = L('valid_synaudio_opensmile_88.npz')['arr_0']
    va_emo   = L('valid_emotion_probs_b_3_7.npz')['arr_0']
    text  = np.concatenate([tr_text, va_text], 0)
    audio = np.concatenate([tr_audio, va_audio], 0)
    quest = np.concatenate([tr_quest, va_quest], 0)
    gen   = np.concatenate([tr_gen, va_gen], 0)
    y     = np.concatenate([tr_y, va_y], 0).astype(np.float64)
    st    = np.concatenate([tr_st, va_st], 0)
    sa    = np.concatenate([tr_sa, va_sa], 0)
    emo   = np.concatenate([tr_emo, va_emo], 0)
    # synthetic audio shares the real audio's frame length (zero-pad the tail).
    pad = np.zeros_like(audio)[:, :, :audio.shape[2] - sa.shape[2], :]
    sa = np.concatenate([sa, pad], axis=2)
    return (text, audio, quest, gen, st, sa, emo, y)


# ---------------------------------------------------------------------------
def make_config():
    return {
        'num_classes': 2, 'dropout': 0.3, 'rnn_layers': 1,
        'audio_embed_size': 88, 'text_embed_size': 768,
        'batch_size': 8, 'epochs': _envi('EPOCHS', 100), 'learning_rate': 2e-5,
        'audio_hidden_dims': 256, 'text_hidden_dims': 256,
        'bidirectional': True, 'cuda': True, 'lambda': 1e-5,
    }


def train_fold(fold, raw_data, config, criterion):
    f = open(FOLDSPLIT, 'r', encoding='utf-8')
    folds = [list(map(int, line.strip().split())) for line in f.readlines()]
    train_idx = folds[fold - 1]
    test_idx = [i for i in range(len(raw_data[-1])) if i not in train_idx]
    train_data = get_eatd_mm_data(train_idx, raw_data, 'train')
    test_data = get_eatd_mm_data(test_idx, raw_data, 'test')

    model = GraphFusion(config, None, None).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)

    best = dict(macro=-1, prec=0, rec=0, acc=0, epoch=-1)
    best_state = None
    for ep in range(config['epochs']):
        # shuffle
        idx = np.random.permutation(len(train_data['text_feature']))
        for k in train_data:
            train_data[k] = train_data[k][idx]
        # ---- train ----
        model.train()
        Xt, Xa, Xq, Xg, Xe, Yt, Wt = build_views(train_data)
        n = len(Xt); bs = config['batch_size']
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            xt = torch.from_numpy(Xt[sl]).float().cuda(); xa = torch.from_numpy(Xa[sl]).float().cuda()
            xq = torch.from_numpy(Xq[sl]).float().cuda(); xg = torch.from_numpy(Xg[sl]).float().cuda()
            xe = torch.from_numpy(Xe[sl]).float().cuda()
            y = torch.from_numpy(Yt[sl]).float().cuda(); w = torch.from_numpy(Wt[sl]).float().cuda()
            optimizer.zero_grad()
            out, _, _ = model(xt, xa, xq, xg, xe, 'train')
            prob = torch.softmax(out, -1)
            loss = soft_ce(prob.reshape(-1, 2), y.reshape(-1), w.reshape(-1))
            loss.backward(); optimizer.step()
        # ---- eval ----
        model.eval()
        with torch.no_grad():
            out, _, _ = model(
                torch.from_numpy(test_data['text_feature']).float().cuda(),
                torch.from_numpy(test_data['audio_feature']).float().cuda(),
                torch.from_numpy(test_data['question_feature']).float().cuda(),
                torch.from_numpy(test_data['gender_feature']).float().cuda(),
                torch.from_numpy(test_data['emotion_feature']).float().cuda(), 'test')
            yp = np.argmax(out.cpu().numpy(), -1); yt = test_data['target']
            p, r, fsc, _ = precision_recall_fscore_support(yt, yp, zero_division=0)
            macro = float(np.mean(fsc))
            if macro >= best['macro']:
                best = dict(macro=macro, prec=float(np.mean(p)), rec=float(np.mean(r)),
                            acc=float(accuracy_score(yt, yp)), epoch=ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best, best_state


if __name__ == '__main__':
    seed = _envi('SEED', 0)
    set_seed(seed)
    torch.set_num_threads(_envi('NT', 8))
    config = make_config()
    criterion = nn.CrossEntropyLoss()
    raw_data = load_dataset(DATA)

    env_flags = {k: os.environ.get(k, '') for k in
                 ['PROXY', 'DIRECTED', 'READOUT', 'LGAT', 'NODE_LN', 'AUDIO_BN', 'SUMM_SELF']}
    aug_cfg = dict(AUG=AUG, MIXMODE=MIXMODE, MIXSYN=MIXSYN, MIXA=MIXA, AUGW=AUGW, LS=LS)

    folds = [int(x) for x in _envs('FOLDS', '1 2 3').split()]
    macros, bundle = [], {}
    for fold in folds:
        set_seed(seed)                      # identical RNG state per fold (matches baseline)
        best, state = train_fold(fold, raw_data, config, criterion)
        macros.append(best['macro'])
        bundle[fold] = dict(state_dict=state, best=best)
        print(f"  fold{fold}: Macro-F1 {best['macro']*100:.2f}  (P {best['prec']*100:.2f} "
              f"R {best['rec']*100:.2f} acc {best['acc']*100:.2f}) @ep{best['epoch']}", flush=True)

    mean = float(np.mean(macros))
    print(f"[AUG={AUG} seed={seed}] BEST macro={mean*100:.2f}  folds=[{' '.join(f'{m*100:.2f}' for m in macros)}]",
          flush=True)

    savedir = os.environ.get('SAVEDIR', '')
    if savedir:
        tag = os.environ.get('SAVETAG', 'model')
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, f"{tag}_s{seed}.pt")
        torch.save(dict(folds=bundle, env=env_flags, aug=aug_cfg, config=config,
                        seed=seed, macro_mean=mean), path)
        print(f"  saved checkpoint -> {path}", flush=True)
