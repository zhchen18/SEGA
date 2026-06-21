"""
DAIC-SEGA++ training harness = DAIC-RELEASE harness + paper section 3.3 LLM augmentation.

Reproduces the RELEASE baseline exactly when aug=none. Adds:
  aug=double  nvar=K        : add K LLM-rephrased copies of each train interview
                             (text swapped, all other modalities + label identical).
  aug=replace augprob=P     : each batch, with prob P swap text -> a random synthetic
                             variant (size-preserving regularizer).
  aug=consist cw=W          : original forward + a synthetic-text forward; add W * MSE
                             between the two prob distributions (consistency reg).
  focal=1 gamma=G           : focal cross-entropy instead of plain CE.
plus the RELEASE regularizers mixup=1 malpha=.. and ls=.. (label smoothing).
Synthetic GloVe features come from ../aug/text_syn_train.npz (3 variants).
"""
import os, sys, json, time, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from models.sega import SegaLite

DATA = os.path.join(ROOT, "data")
AUG = os.path.join(ROOT, "aug")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(split, tag):
    z = np.load(os.path.join(DATA, f"{split}_{tag}.npz"))
    return {k: z[k] for k in z.files}


def to_tensors(d):
    keys = ["question", "text", "audio", "video", "gender", "label"]
    out = {}
    for k in keys:
        t = torch.from_numpy(d[k])
        out[k] = t.float() if d[k].dtype.kind == "f" else t.long()
    return out


def contrastive_loss(feat, label):
    sim = torch.exp(torch.cosine_similarity(feat.unsqueeze(1), feat.unsqueeze(0), dim=-1))
    sim = torch.triu(sim, diagonal=1)
    denom = sim.sum()
    num = feat.new_tensor(0.0)
    for l in label.unique():
        idx = (label == l).nonzero(as_tuple=True)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                num = num + sim[idx[i], idx[j]]
    return -torch.log((num + 1e-4) / (denom + 1e-4)) / len(label)


def evaluate(model, dev, device):
    model.eval()
    with torch.no_grad():
        b = {k: v.to(device) for k, v in dev.items()}
        _, out, _ = model(b)
        yp = out.argmax(-1).cpu().numpy()
        yt = dev["label"].numpy()
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    return dict(acc=accuracy_score(yt, yp), macro=float(np.mean(f1)),
                control=float(f1[0]), depressed=float(f1[1]))


def main():
    cfg = dict(fusion="text", norm=1, seed=0, epochs=200, lr=1e-4, batch=8, hidden=256,
               ln_input=1, use_gender=0, gate_init=0.0, enc_dropout=0.0, gat_dropout=0.3,
               cls_dropout=0.5, contrastive=1, wd=1e-5, num_classes=2, quiet=1,
               mixup=0, malpha=0.2, ls=0.0,
               aug="none", nvar=1, augprob=0.5, cw=1.0, focal=0, gamma=2.0, synsrc="",
               save="")
    for a in sys.argv[1:]:
        k, v = a.split("=")
        cfg[k] = type(cfg[k])(v) if k in cfg else v
    if "mm_mods" in cfg and isinstance(cfg["mm_mods"], str):
        cfg["mm_mods"] = cfg["mm_mods"].split(",")
    tag = "norm" if int(cfg["norm"]) else "raw"
    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(os.path.join(DATA, "meta.json")))
    cfg["dims"] = meta["dims"]
    for k in ["ln_input", "use_gender", "contrastive"]:
        cfg[k] = bool(int(cfg[k]))

    cfg["use_emotion"] = bool(int(cfg.get("use_emotion", 0))) or ("mm_mods" in cfg and "emotion" in cfg["mm_mods"])
    tr = to_tensors(load_split("train", tag))
    dev = to_tensors(load_split("dev", tag))
    keys = ["question", "text", "audio", "video", "gender", "label"]
    if cfg["use_emotion"]:
        for split, d in [("train", tr), ("dev", dev)]:
            d["emotion"] = torch.from_numpy(np.load(os.path.join(DATA, f"emotion_{split}.npz"))["emotion"]).float()
        keys = keys + ["emotion"]

    # ---- LLM augmentation: load synthetic text variants (aligned to train rows) ----
    aug = cfg["aug"]; nvar = int(cfg["nvar"])
    syn = None
    if aug != "none":
        z = np.load(os.path.join(AUG, f"text_syn{cfg.get('synsrc','')}_train.npz"))
        syn = torch.stack([torch.from_numpy(z[f"text_v{k}"]).float() for k in range(3)])  # (3,n,T,300)
        Tt = tr["text"].shape[1]
        if syn.shape[2] != Tt:
            if syn.shape[2] > Tt:
                syn = syn[:, :, :Tt]
            else:
                pad = torch.zeros(3, syn.shape[1], Tt - syn.shape[2], syn.shape[3])
                syn = torch.cat([syn, pad], dim=2)

    if aug == "double":
        parts = {k: [tr[k]] for k in keys}
        for k in range(nvar):
            for key in keys:
                parts[key].append(syn[k] if key == "text" else tr[key].clone())
        tr = {key: torch.cat(parts[key], dim=0) for key in keys}
        syn = None  # consumed
    elif aug == "balance":
        # class-balanced oversampling: add synthetic copies of the MINORITY class until
        # both classes are equal. Synthetic text + original other-modalities + same label.
        lab = tr["label"]
        cnt = [int((lab == c).sum()) for c in (0, 1)]
        minc = 0 if cnt[0] < cnt[1] else 1
        need = abs(cnt[0] - cnt[1])
        pool = (lab == minc).nonzero(as_tuple=True)[0].tolist()
        parts = {k: [tr[k]] for k in keys}
        for j in range(need):
            si = pool[j % len(pool)]; kv = (j // len(pool)) % 3
            for key in keys:
                v = syn[kv][si] if key == "text" else tr[key][si]
                parts[key].append(v.unsqueeze(0))
        tr = {key: torch.cat(parts[key], dim=0) for key in keys}
        print(f"balance: {cnt} -> minority={minc} added={need}", flush=True)
        syn = None
    tr_ds = TensorDataset(*[tr[k] for k in keys], torch.arange(tr["label"].shape[0]))
    loader = DataLoader(tr_ds, batch_size=int(cfg["batch"]), shuffle=True)

    model = SegaLite(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))

    ls = float(cfg.get("ls", 0.0)); use_focal = int(cfg.get("focal", 0)); gamma = float(cfg.get("gamma", 2.0))
    ce_plain = nn.CrossEntropyLoss(label_smoothing=ls)

    def ce(logit, label):
        if not use_focal:
            return ce_plain(logit, label)
        logp = F.log_softmax(logit, -1)
        if ls > 0:
            nll = -((1 - ls) * logp.gather(1, label[:, None]).squeeze(1) + ls * logp.mean(1))
        else:
            nll = -logp.gather(1, label[:, None]).squeeze(1)
        pt = logp.exp().gather(1, label[:, None]).squeeze(1)
        return ((1 - pt) ** gamma * nll).mean()

    use_mixup = int(cfg.get("mixup", 0)); malpha = float(cfg.get("malpha", 0.2))
    cw = float(cfg.get("cw", 1.0)); augprob = float(cfg.get("augprob", 0.5))
    fcols = [k for k in keys if k not in ("gender", "label")]
    if syn is not None:
        syn = syn.to(device)

    def head_loss(out, aux, label):
        if "text_logit" in aux:
            tl, ml = aux["text_logit"], aux["mm_logit"]
            return ce(tl, label) + ce(tl.detach() + ml, label)
        return ce(out, label)

    best = dict(macro=-1, control=0, depressed=0, acc=0, epoch=-1)
    best_state = None
    t0 = time.time()
    for ep in range(int(cfg["epochs"])):
        model.train()
        for batch in loader:
            cols = batch[:-1]; idx = batch[-1]
            b = dict(zip(keys, cols))
            b = {k: v.to(device) for k, v in b.items()}
            idx = idx.to(device)
            opt.zero_grad()

            if aug == "replace" and b["label"].shape[0] >= 1:
                ksel = torch.randint(0, 3, (idx.shape[0],), device=device)
                mask = (torch.rand(idx.shape[0], device=device) < augprob)
                newtext = b["text"].clone()
                for vi in range(3):
                    sel = mask & (ksel == vi)
                    if sel.any():
                        newtext[sel] = syn[vi][idx[sel]]
                b["text"] = newtext

            if use_mixup and b["label"].shape[0] > 1:
                lam = float(np.random.beta(malpha, malpha))
                perm = torch.randperm(b["label"].shape[0], device=device)
                bm = dict(b)
                for c in fcols:
                    bm[c] = lam * b[c] + (1 - lam) * b[c][perm]
                feat, out, aux = model(bm)
                ya, yb = b["label"], b["label"][perm]
                loss = lam * head_loss(out, aux, ya) + (1 - lam) * head_loss(out, aux, yb)
            else:
                feat, out, aux = model(b)
                loss = head_loss(out, aux, b["label"])
                if cfg["contrastive"]:
                    loss = loss + contrastive_loss(feat, b["label"])

            if aug == "consist" and cw > 0:
                kk = torch.randint(0, 3, (1,)).item()
                bs = dict(b); bs["text"] = syn[kk][idx]
                _, out_s, aux_s = model(bs)
                loss = loss + head_loss(out_s, aux_s, b["label"])
                p1 = F.softmax(out, -1); p2 = F.softmax(out_s, -1)
                loss = loss + cw * F.mse_loss(p1, p2)

            loss.backward(); opt.step()
        m = evaluate(model, dev, device)
        if m["macro"] > best["macro"]:
            best = {**m, "epoch": ep}
            if cfg.get("save"):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not int(cfg["quiet"]):
            print(f"ep{ep:3d} macro={m['macro']*100:.2f} (best {best['macro']*100:.2f}@{best['epoch']})")
    dt = time.time() - t0
    print(f"[{cfg['fusion']:6s} norm={tag} aug={aug} nvar={nvar} mixup={use_mixup} ls={ls} "
          f"focal={use_focal} seed={cfg['seed']}] BEST macro={best['macro']*100:.2f} "
          f"(C={best['control']*100:.2f} D={best['depressed']*100:.2f} acc={best['acc']*100:.2f}) "
          f"@ep{best['epoch']}  [{dt:.0f}s, {dt/int(cfg['epochs'])*1000:.0f}ms/ep]")
    if cfg.get("save") and best_state is not None:
        ckpt_cfg = {k: cfg[k] for k in (
            "fusion", "norm", "hidden", "ln_input", "use_gender", "gate_init",
            "enc_dropout", "gat_dropout", "cls_dropout", "num_classes", "dims",
            "use_emotion", "mm_mods") if k in cfg}
        os.makedirs(os.path.dirname(os.path.abspath(cfg["save"])), exist_ok=True)
        torch.save({"state_dict": best_state, "cfg": ckpt_cfg, "best": best,
                    "seed": int(cfg["seed"])}, cfg["save"])
        print(f"  saved checkpoint -> {cfg['save']}")
    return best


if __name__ == "__main__":
    main()
