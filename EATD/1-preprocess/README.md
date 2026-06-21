# 1-preprocess — raw EATD corpus → per-segment feature tensors

> **You do NOT need this stage to run the model.** The final feature tensors are already
> bundled in `../3-model/data/EATD-Features/` (and the synthetic-text features in
> `../2-augmentation/aug/`), so `3-model/run_sega.sh` / `run_sega++.sh` reproduce everything
> **without the raw corpus**. This folder documents how the features are extracted from
> scratch.

## The corpus
**EATD-Corpus** (Shen et al., 2022): 162 student volunteers, each an audio interview with
**three emotional segments** — *positive / neutral / negative* — plus the transcript and an
SDS depression score (subjects with raw SDS ≥ 53 are labelled depressed). Evaluated by
**3-fold cross-validation** (`../3-model/data/train_test_fold_split.txt`).

## Features (paper-aligned)

| modality | extractor | per subject |
|---|---|---|
| **text** | pre-trained **Chinese BERT** (`hfl/chinese-roberta-wwm-ext`); the answer of each of the 3 segments is tokenized (original Chinese, no translation) and pad/truncated to 128 → token embeddings | `(3, 128, 768)` |
| **question** | the same Chinese-BERT encoder applied to the standard segment prompts | `(3, 128, 768)` |
| **audio** | **openSMILE** eGeMAPS low-level descriptors over each segment's speech, per-frame | `(3, F, 88)` |
| **emotion** | per-segment emotion distribution (7-way) used as structural side-nodes | `(3, 7)` |
| **gender** | subject gender, used as a structural side-node | `(1,)` |

The paper uses pre-trained Chinese BERT as the EATD word embedder and openSMILE acoustic
features; the structural-element graph then runs over these per-segment nodes.

## Scripts
- `EATD_BERT_text_feature.py` — encode each segment's answer / question text with
  `hfl/chinese-roberta-wwm-ext` → text & question tensors.
- `extract_eatd_feature.py` — openSMILE eGeMAPS acoustic features per segment.
- `annotate_eatd_auxiliary_info.py` — assemble per-segment question prompts + auxiliary
  (gender / emotion) info.
- `EATD-SBERT-Extraction.py` — sentence-level text variant (kept for provenance).

> These are the original research-extraction scripts, kept verbatim. They expect a local
> copy of the EATD corpus under `./EATD-Corpus` and an external acoustic-feature toolkit
> (openSMILE / MSA-FET) on the path; edit the hard-coded paths at the top of each script to
> your environment before running. The bundled `../3-model/data/EATD-Features/*.npy|*.npz`
> are the output of this stage and were verified (text-encoder row-cosine 1.0000 against the
> released EATD features).
