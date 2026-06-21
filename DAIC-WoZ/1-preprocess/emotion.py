"""
Extra side-information: per-QA-turn EMOTION/sentiment of the participant's answer.

DAIC-WOZ has no per-turn emotion labels, so we derive a cheap 4-d emotion vector
(VADER neg/neu/pos/compound) from each answer's text. Rows align 1:1 with the GloVe
feature rows (answers.txt has one line per QA turn, same order). Standardized on TRAIN.

Run:  python emotion.py
"""
import os, json
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.abspath(os.path.join(ROOT, "..", "NAACL24-SEGA", "DAIC-WOZ"))
EMB = os.path.join(SRC, "Embedding")
SPLIT = os.path.join(SRC, "utils")
OUT = os.path.join(ROOT, "data")

sia = SentimentIntensityAnalyzer()
meta = json.load(open(os.path.join(OUT, "meta.json")))
MAXQA = meta["max_qa"]


def turn_emotions(pid):
    lines = open(os.path.join(EMB, f"merged_{pid}_TRANSCRIPT_embedding", "answers.txt"),
                 encoding="utf-8").read().splitlines()
    out = []
    for s in lines:
        d = sia.polarity_scores(s)
        out.append([d["neg"], d["neu"], d["pos"], d["compound"]])
    return np.array(out, dtype=np.float64)


def main():
    ids = {s: pd.read_csv(os.path.join(SPLIT, f"{s}_split_Depression_AVEC2017.csv"))["Participant_ID"].tolist()
           for s in ["train", "dev"]}
    raw = {s: [turn_emotions(p) for p in ids[s]] for s in ids}
    allt = np.concatenate(raw["train"], axis=0)
    mean, std = allt.mean(0), allt.std(0); std[std < 1e-6] = 1.0
    for s in ids:
        T = np.zeros((len(raw[s]), MAXQA, 4), dtype=np.float32)
        for i, x in enumerate(raw[s]):
            T[i, :len(x)] = ((x - mean) / std).astype(np.float32)
        np.savez(os.path.join(OUT, f"emotion_{s}.npz"), emotion=T)
    print("saved emotion tensors; train mean/std:", mean.round(3), std.round(3))


if __name__ == "__main__":
    main()
