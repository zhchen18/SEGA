"""
Load a bundled best-dev checkpoint and reproduce its EATD 3-fold Macro-F1 (no training).

  python eval_ckpt.py ../checkpoints/segapp_best.pt
  python eval_ckpt.py ../checkpoints/sega_best.pt

Each checkpoint bundles the best-dev model of all 3 folds for one run.  The script
rebuilds every fold's held-out test set from the bundled features + fold split, runs the
saved model, and prints per-fold and mean Macro-F1 -- matching the BEST line that
training reported for that run.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import train as T          # reuse load_dataset / get_eatd_mm_data / GraphFusion / fold split


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "checkpoints", "segapp_best.pt")
    ck = torch.load(path, map_location="cpu")

    # restore the backbone structural flags this checkpoint was trained with
    for k, v in ck["env"].items():
        if v != "":
            os.environ[k] = str(v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dict(ck["config"]); config["cuda"] = (device == "cuda")
    raw_data = T.load_dataset(T.DATA)
    folds = [list(map(int, line.strip().split()))
             for line in open(T.FOLDSPLIT, encoding="utf-8").read().splitlines()]

    macros = []
    for fold, fc in sorted(ck["folds"].items()):
        train_idx = folds[fold - 1]
        test_idx = [i for i in range(len(raw_data[-1])) if i not in train_idx]
        test = T.get_eatd_mm_data(test_idx, raw_data, "test")

        model = T.GraphFusion(config, None, None)
        model = model.to(device)
        model.load_state_dict(fc["state_dict"])
        model.eval()
        with torch.no_grad():
            args = [torch.from_numpy(test[k]).float().to(device) for k in
                    ["text_feature", "audio_feature", "question_feature",
                     "gender_feature", "emotion_feature"]]
            out, _, _ = model(*args, "test")
            yp = np.argmax(out.cpu().numpy(), -1); yt = test["target"]
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, zero_division=0)
        macro = float(np.mean(f1)); macros.append(macro)
        print(f"  fold{fold}: Macro-F1 {macro*100:.2f}  "
              f"Control {f1[0]*100:.2f}  Depressed {f1[1]*100:.2f}  acc {accuracy_score(yt,yp)*100:.2f}")

    mean = float(np.mean(macros))
    rep = ck.get("macro_mean", None)
    line = f"{os.path.basename(path)}: 3-fold mean Macro-F1 {mean*100:.2f}"
    if rep is not None:
        line += f"   (training-reported {rep*100:.2f} -> {'MATCH' if abs(mean-rep)<1e-4 else 'DIFF'})"
    print(line)


if __name__ == "__main__":
    main()
