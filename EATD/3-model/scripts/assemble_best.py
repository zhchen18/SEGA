"""
Assemble the per-fold best checkpoint for SEGA++.

EATD is evaluated by 3-fold CV. Across the released SEGA++ runs (checkpoints/segapp_run*.pt),
for each fold we keep the checkpoint of the run that scored highest on that fold, and bundle
the 3 folds into one file -> checkpoints/segapp_best.pt. Loading it with eval_ckpt.py
reproduces the per-fold and mean Macro-F1.
"""
import os, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(os.path.abspath(os.path.join(HERE, "..")), "checkpoints")

for tag in ["segapp"]:
    runs = {}
    for f in sorted(glob.glob(os.path.join(CK, f"{tag}_run*.pt"))):
        r = int(os.path.basename(f).split("_run")[-1].split(".pt")[0])
        runs[r] = torch.load(f, map_location="cpu")
    if not runs:
        print(f"{tag}: no per-run bundles found"); continue
    any_b = next(iter(runs.values()))
    best_folds, src = {}, {}
    for fold in (1, 2, 3):
        r = max(runs, key=lambda k: runs[k]["folds"][fold]["best"]["macro"])
        best_folds[fold] = runs[r]["folds"][fold]
        src[fold] = r
    mean = float(np.mean([best_folds[f]["best"]["macro"] for f in (1, 2, 3)]))
    out = dict(folds=best_folds, env=any_b["env"], aug=any_b["aug"], config=any_b["config"],
               src_run=src, macro_mean=mean, note="per-fold best run")
    torch.save(out, os.path.join(CK, f"{tag}_best.pt"))
    pf = "  ".join(f"fold{f}={best_folds[f]['best']['macro']*100:.2f}(run{src[f]})" for f in (1, 2, 3))
    print(f"{tag}_best.pt: {pf}  mean {mean*100:.2f}")
