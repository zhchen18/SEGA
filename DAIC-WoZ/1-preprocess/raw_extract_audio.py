"""
Provenance / verification: re-extract per-QA-turn COVAREP audio features straight from
the RAW DAIC-WOZ corpus (the xxx_P.zip files) and check they reproduce the pre-extracted
audio_covarep_emb.npy that the rest of the pipeline consumes.

This proves the per-turn feature tensors are genuinely grounded in the raw corpus (the
same recipe as the paper code: merge consecutive same-speaker turns -> pair Ellie->Participant
with a >=3-token gate -> mean COVAREP+FORMANT over each answer's [start-0.01, stop+0.01] window).

Run:  python raw_extract_audio.py 300
"""
import os, sys, io, zipfile
import numpy as np
import pandas as pd

# Path to the raw DAIC-WOZ corpus (the folder of xxx_P.zip files). Override via env:
#   DAIC_RAW=/path/to/189SAMPLES python raw_extract_audio.py 302
RAW = os.environ.get("DAIC_RAW", r"D:\本地Coding\2026-06-14-抑郁语音合成\抑郁检测-DAIC-WOZ\189SAMPLES")
HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.abspath(os.path.join(HERE, "..", "..", "NAACL24-SEGA", "DAIC-WOZ", "Embedding"))


def merge_same_speaker(df):
    rows, cur = [], None
    for _, r in df.iterrows():
        if cur and r["speaker"] == cur[2]:
            cur[1] = r["stop_time"]; cur[3].append(str(r["value"]))
        else:
            if cur: rows.append((cur[0], cur[1], cur[2], ". ".join(cur[3]) + "."))
            cur = [r["start_time"], r["stop_time"], r["speaker"], [str(r["value"])]]
    if cur: rows.append((cur[0], cur[1], cur[2], ". ".join(cur[3]) + "."))
    return pd.DataFrame(rows, columns=["start_time", "stop_time", "speaker", "value"])


def extract(pid):
    zf = zipfile.ZipFile(os.path.join(RAW, f"{pid}_P.zip"))
    tr = pd.read_csv(io.BytesIO(zf.read(f"{pid}_TRANSCRIPT.csv")), sep="\t")
    cov = pd.read_csv(io.BytesIO(zf.read(f"{pid}_COVAREP.csv")), header=None).values
    form = pd.read_csv(io.BytesIO(zf.read(f"{pid}_FORMANT.csv")), header=None).values
    n = min(len(cov), len(form))
    feat = np.concatenate([cov[:n], form[:n]], axis=1)            # 74 + 5 = 79, 100 Hz
    t = np.arange(0, len(feat) / 100, 1 / 100)

    m = merge_same_speaker(tr)
    out = []
    for i in range(len(m) - 1):
        if m.iloc[i]["speaker"] == "Ellie" and m.iloc[i + 1]["speaker"] == "Participant":
            q, a = str(m.iloc[i]["value"]), str(m.iloc[i + 1]["value"])
            if len(q.split()) >= 3 or len(a.split()) >= 3:        # token gate (whitespace approx of spaCy)
                s = float(m.iloc[i + 1]["start_time"]) - 0.01
                e = float(m.iloc[i + 1]["stop_time"]) + 0.01
                idx = np.where((t >= s) & (t <= e))[0]
                out.append(np.nan_to_num(feat[idx].mean(0)))
    return np.array(out)


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    mine = extract(pid)
    ref = np.load(os.path.join(EMB, f"merged_{pid}_TRANSCRIPT_embedding", "audio_covarep_emb.npy"))
    print(f"pid {pid}: my turns {mine.shape}, reference {ref.shape}")
    k = min(len(mine), len(ref))
    if k:
        # row-wise cosine between my extraction and the reference (aligned turns)
        a, b = mine[:k], ref[:k]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        print(f"  aligned {k} turns | mean row-cos {cos.mean():.4f} | "
              f"max abs diff {np.abs(a - b).max():.3e} | exact-shape-match {mine.shape == ref.shape}")


if __name__ == "__main__":
    main()
