"""
Load a saved best-dev checkpoint and reproduce its DAIC-WOZ dev-set metrics.

  python eval_ckpt.py ../checkpoints/segapp_run1.pt
  python eval_ckpt.py ../checkpoints/sega_run1.pt

Prints Macro-F1 / Control-F1 / Depressed-F1 / accuracy; these match the BEST line the
training run reported for that seed (best-dev-epoch model).
"""
import os, sys, json
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from models.sega import SegaLite

DATA = os.path.join(ROOT, "data")


def to_tensors(d):
    out = {}
    for k in ["question", "text", "audio", "video", "gender", "label"]:
        t = torch.from_numpy(d[k]); out[k] = t.float() if d[k].dtype.kind == "f" else t.long()
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "checkpoints", "segapp_run1.pt")
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = "norm" if int(cfg.get("norm", 0)) else "raw"
    z = np.load(os.path.join(DATA, f"dev_{tag}.npz"))
    dev = to_tensors({k: z[k] for k in z.files})
    if cfg.get("use_emotion"):
        dev["emotion"] = torch.from_numpy(np.load(os.path.join(DATA, "emotion_dev.npz"))["emotion"]).float()

    model = SegaLite(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        b = {k: v.to(device) for k, v in dev.items()}
        _, out, _ = model(b)
        yp = out.argmax(-1).cpu().numpy(); yt = dev["label"].numpy()
    _, _, f1, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    macro, c, d, acc = np.mean(f1) * 100, f1[0] * 100, f1[1] * 100, accuracy_score(yt, yp) * 100
    rep = ck.get("best", {})
    print(f"{os.path.basename(path)}: "
          f"Macro-F1 {macro:.2f}  Control {c:.2f}  Depressed {d:.2f}  acc {acc:.2f}")
    if rep:
        print(f"   (training-reported best: Macro {rep['macro']*100:.2f} @ep{rep['epoch']}) "
              f"-> {'MATCH' if abs(macro - rep['macro']*100) < 0.01 else 'DIFF'}")


if __name__ == "__main__":
    main()
