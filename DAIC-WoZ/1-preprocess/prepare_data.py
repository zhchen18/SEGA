"""
DAIC-EXPLORE — Feature processing pipeline.

Reads the per-QA-turn features that were extracted from the raw DAIC-WOZ corpus
(GloVe-300 text/question, COVAREP+FORMANT-79 audio, CLNF-378 + HOG-4464 video),
applies *offline, train-set-fitted, per-feature standardization* per modality, pads
to a common turn length, and caches train/dev tensors as a single .npz.

This is the crucial step the earlier reproduction under-exploited: the raw modalities
live on wildly different scales (text row-L2 ~4, audio ~6e3, clnf ~9e3), which makes
any joint fusion collapse. Standardizing each feature to zero-mean/unit-std on the TRAIN
turns brings every modality onto a comparable scale before the model ever sees it.

Run:  python prepare_data.py
"""
import os, json
import numpy as np
import pandas as pd

# ---- paths -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                     # DAIC-EXPLORE
SRC  = os.path.abspath(os.path.join(ROOT, "..", "NAACL24-SEGA", "DAIC-WOZ"))
EMB  = os.path.join(SRC, "Embedding")
SPLIT = os.path.join(SRC, "utils")
OUT  = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

MODALITIES = {
    # name : (list of npy files to concat per turn, dim)
    "question": (["question_glove_emb.npy"], 300),
    "text":     (["text_glove_emb.npy"],     300),
    "audio":    (["audio_covarep_emb.npy"],  79),
    "video":    (["video_clnf_emb.npy", "video_hog_emb.npy"], 4842),
}

def load_turns(pid, files):
    arrs = [np.load(os.path.join(EMB, f"merged_{pid}_TRANSCRIPT_embedding", f)) for f in files]
    return np.concatenate(arrs, axis=-1).astype(np.float64)

def load_label(pid):
    p = os.path.join(EMB, f"merged_{pid}_TRANSCRIPT_embedding", "label.txt")
    with open(p, encoding="utf-8") as f:
        return int(f.readline().strip().split("\t")[1])

def main():
    tr_df = pd.read_csv(os.path.join(SPLIT, "train_split_Depression_AVEC2017.csv"))
    dev_df = pd.read_csv(os.path.join(SPLIT, "dev_split_Depression_AVEC2017.csv"))
    tr_ids = tr_df["Participant_ID"].tolist()
    dev_ids = dev_df["Participant_ID"].tolist()
    gender = {"train": tr_df["Gender"].to_numpy(np.int64), "dev": dev_df["Gender"].to_numpy(np.int64)}
    pids = {"train": np.array(tr_ids), "dev": np.array(dev_ids)}

    # ---- load raw per-turn features ------------------------------------------
    raw = {}  # split -> modality -> list[ (T, d) ]
    labels = {}
    for split, ids in [("train", tr_ids), ("dev", dev_ids)]:
        raw[split] = {m: [] for m in MODALITIES}
        labels[split] = []
        for pid in ids:
            for m, (files, _) in MODALITIES.items():
                raw[split][m].append(load_turns(pid, files))
            labels[split].append(load_label(pid))

    # ---- fit per-feature standardizer on TRAIN turns -------------------------
    # nan_to_num first (HOG/CLNF can contain nan from empty windows)
    scalers = {}
    for m in MODALITIES:
        allturns = np.concatenate([np.nan_to_num(x) for x in raw["train"][m]], axis=0)  # (sum_T, d)
        mean = allturns.mean(0)
        std = allturns.std(0)
        std[std < 1e-6] = 1.0   # dead features -> leave as (x-mean)
        scalers[m] = (mean, std)

    def standardize(x, m, clip=8.0):
        mean, std = scalers[m]
        z = (np.nan_to_num(x) - mean) / std
        if clip:
            z = np.clip(z, -clip, clip)
        return z.astype(np.float32)

    # ---- pad to common turn length & assemble tensors ------------------------
    max_qa = max(len(x) for split in raw for m in ["text"] for x in raw[split][m])
    print(f"max QA turns = {max_qa}")

    def assemble(split, normed):
        out = {}
        n = len(labels[split])
        for m, (_, d) in MODALITIES.items():
            T = np.zeros((n, max_qa, d), dtype=np.float32)
            for i, x in enumerate(raw[split][m]):
                xx = standardize(x, m) if normed else np.nan_to_num(x).astype(np.float32)
                T[i, :len(xx)] = xx
            out[m] = T
        out["label"] = np.array(labels[split], dtype=np.int64)
        out["gender"] = gender[split]
        out["pid"] = pids[split]
        return out

    for normed in (True, False):
        tag = "norm" if normed else "raw"
        for split in ("train", "dev"):
            d = assemble(split, normed)
            np.savez(os.path.join(OUT, f"{split}_{tag}.npz"), **d)
        print(f"saved {tag} tensors")

    # save scaler stats for reference
    np.savez(os.path.join(OUT, "scalers.npz"),
             **{f"{m}_mean": scalers[m][0] for m in MODALITIES},
             **{f"{m}_std": scalers[m][1] for m in MODALITIES})
    meta = {"max_qa": int(max_qa),
            "dims": {m: MODALITIES[m][1] for m in MODALITIES},
            "train_n": len(labels["train"]), "dev_n": len(labels["dev"]),
            "train_pos": int(sum(labels["train"])), "dev_pos": int(sum(labels["dev"]))}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("meta:", meta)

    # sanity: report row-L2 norms before/after norm on dev
    print("\n-- mean row-L2 norm (dev, first sample) --")
    for m in MODALITIES:
        x = raw["dev"][m][0]
        raw_n = np.mean(np.linalg.norm(np.nan_to_num(x), axis=-1))
        z = standardize(x, m)
        z_n = np.mean(np.linalg.norm(z, axis=-1))
        print(f"  {m:9s}: raw {raw_n:10.2f}  ->  norm {z_n:7.2f}")

if __name__ == "__main__":
    main()
