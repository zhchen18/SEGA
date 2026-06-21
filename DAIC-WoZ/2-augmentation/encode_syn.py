# -*- coding: utf-8 -*-
"""
Encode synthetic DAIC answers ([SEP]-separated variants) to per-QA-turn GloVe-300 using
the SAME corpus word2id + glove_embedding.npy as the original text features. OOV skipped.
Location-relative (works inside DAIC-WoZ/2-augmentation/). Parametrized by env:
  SYNFILE  (default syn_answers.txt)   -- the synthetic file per participant folder
  OUTSUF   (default '')                 -- output suffix: aug/text_syn{OUTSUF}_{split}.npz
"""
import os, re, ast, json, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))           # .../DAIC-WoZ/2-augmentation
NSEGA = os.path.abspath(os.path.join(HERE, "..", ".."))      # .../NAACL24-SEGA
DW    = f"{NSEGA}/DAIC-WOZ"; EMB = f"{DW}/Embedding"; GP = f"{DW}/utils/Glove_Preprocess"
OUT   = f"{HERE}/aug"; os.makedirs(OUT, exist_ok=True)
META  = f"{HERE}/../3-model/data/meta.json"
SYNFILE = os.environ.get("SYNFILE", "syn_answers.txt")
OUTSUF  = os.environ.get("OUTSUF", "")

word2id = ast.literal_eval(open(f"{GP}/word2id.txt", encoding="utf-8").read())
glove = np.load(f"{GP}/glove_embedding.npy")
meta = json.load(open(META))
MAXQA = meta["max_qa"]; D = 300; NVAR = 3
PUNCT = re.compile(r"[.,!?;:\"'`()\[\]{}…—\-]")

def vec(sent, stat):
    toks = PUNCT.sub(" ", sent.lower()).split(); embs = []
    for t in toks:
        stat[0] += 1
        if t in word2id: embs.append(glove[word2id[t]])
        else: stat[1] += 1
    return np.mean(embs, 0).astype(np.float32) if embs else np.zeros(D, np.float32)

def build(split):
    ids = pd.read_csv(f"{DW}/utils/{split}_split_Depression_AVEC2017.csv")["Participant_ID"].tolist()
    n = len(ids); out = np.zeros((NVAR, n, MAXQA, D), np.float32); stat = [0, 0]
    for i, pid in enumerate(ids):
        p = f"{EMB}/merged_{pid}_TRANSCRIPT_embedding"
        ans = open(f"{p}/answers.txt", encoding="utf-8").read().splitlines()
        syn = open(f"{p}/{SYNFILE}", encoding="utf-8").read().splitlines()
        assert len(ans) == len(syn), f"{pid}: {len(ans)} vs {len(syn)}"
        for t in range(min(len(ans), MAXQA)):
            variants = [v.strip() for v in syn[t].split("[SEP]") if v.strip()] or [ans[t]]
            for k in range(NVAR):
                out[k, i, t] = vec(variants[k % len(variants)], stat)
    print(f"{split}: n={n} OOV={stat[1]/max(stat[0],1)*100:.2f}%")
    return out

tr = build("train"); dev = build("dev")
np.savez(f"{OUT}/text_syn{OUTSUF}_train.npz", text_v0=tr[0], text_v1=tr[1], text_v2=tr[2])
np.savez(f"{OUT}/text_syn{OUTSUF}_dev.npz", text_v0=dev[0], text_v1=dev[1], text_v2=dev[2])
print("saved", f"text_syn{OUTSUF}_*.npz", tr.shape, dev.shape)
