# 2-augmentation — paper §3.3 "A Helping Hand from LLMs"

Generates and encodes the LLM text augmentation that powers **SEGA++**, following the
paper's recipe (GPT-3.5 principle-based rephrasing).

### `gpt_augment_eatd.py` — paraphrase generation (GPT-3.5)
For each participant answer (the three polarity segments: negative / positive / neutral),
GPT-3.5 produces a paraphrase under the paper's **five principles** (content integrity,
colloquial naturalness, respect/appropriateness, length consistency, tolerate disfluency).
EATD answers are Chinese, so the rephrasing is done **in Chinese** (the corpus language) —
no translation step. Output is written to `{polarity}_synthetic.txt` per subject folder.
Resumable. Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`) to run.

```bash
OPENAI_API_KEY=... CORPUS=./EATD-Corpus python gpt_augment_eatd.py
```

### `encode_syn.py` — synthetic text → per-turn features
Encodes the paraphrases with the **same** text encoder as the original EATD text
(`hfl/chinese-roberta-wwm-ext`, pad/truncate to 128 → `(N,3,128,768)`), so original and
synthetic text share one vector space. Writes `train_syntext_roberta_768.npy` /
`valid_syntext_roberta_768.npy` into `../3-model/data/EATD-Features/`.

```bash
CORPUS=./EATD-Corpus OUT=../3-model/data/EATD-Features python encode_syn.py
```

### Cached features
The encoded synthetic text features are **already bundled** in
`../3-model/data/EATD-Features/` (`{train,valid}_syntext_roberta_768.npy`), together
with the synthetic-audio features (`{train,valid}_synaudio_opensmile_88.npz`), so
`3-model/run_sega++.sh` reproduces SEGA++ with **no API access required**. This folder is
only needed to regenerate the augmentation from scratch.

> The synthetic audio for SEGA++ is produced by the preprocessing stage (frame-level
> perturbation of the openSMILE eGeMAPS sequence); see `1-preprocess/README.md`.
