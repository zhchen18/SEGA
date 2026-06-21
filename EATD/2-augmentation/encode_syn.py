# -*- coding: utf-8 -*-
"""
Encode the GPT-synthesized answers into the SAME text space as the original EATD text:
chinese-roberta-wwm-ext token embeddings, pad/truncate to 128 -> (N, 3, 128, 768),
matching {train,valid}_text_simple_roberta_768.npy.  Writes the synthetic text features
consumed by SEGA++.

  CORPUS=./EATD-Corpus OUT=../3-model/data/EATD-Features python encode_syn.py
"""
import os, numpy as np, torch
from transformers import BertTokenizer, BertModel

M = 'hfl/chinese-roberta-wwm-ext'        # the confirmed EATD text encoder
CORP = os.environ.get('CORPUS', './EATD-Corpus')
OUT = os.environ.get('OUT', '../3-model/data/EATD-Features')
POLAR = ['negative', 'positive', 'neutral']
MAXLEN = 128

tok = BertTokenizer.from_pretrained(M)
mdl = BertModel.from_pretrained(M).eval()


def enc(text):
    ids = tok.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        return mdl(ids).last_hidden_state.squeeze(0).numpy()       # [L,768]


def build(prefix):
    idxs = sorted(int(d.split('_')[1]) for d in os.listdir(CORP) if d.startswith(prefix + '_'))
    feats = []
    for i in idxs:
        seg = []
        for pol in POLAR:
            txt = open(f'{CORP}/{prefix}_{i}/{pol}_synthetic.txt', encoding='utf-8').read().strip()
            h = enc(txt)[:MAXLEN]
            h = np.pad(h, ((0, MAXLEN - h.shape[0]), (0, 0)), mode='constant')
            seg.append(h)
        feats.append(np.array(seg))
    return np.array(feats).astype(np.float32)                       # [N,3,128,768]


tr, va = build('t'), build('v')
print('train', tr.shape, 'valid', va.shape, flush=True)
os.makedirs(OUT, exist_ok=True)
np.save(f'{OUT}/train_syntext_roberta_768.npy', tr)
np.save(f'{OUT}/valid_syntext_roberta_768.npy', va)
print('saved synthetic text features ->', OUT, flush=True)
