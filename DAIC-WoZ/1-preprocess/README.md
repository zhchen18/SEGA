# 1-preprocess — raw DAIC-WOZ → per-QA-turn feature tensors

> **You do NOT need this stage to run the model.** The final feature tensors are already
> bundled in `../3-model/data/` (and the synthetic features in `../2-augmentation/aug/`),
> so `3-model/run_sega.sh` / `run_sega++.sh` reproduce everything **without downloading
> DAIC-WOZ**. This folder is for re-extracting the features **from scratch** if you want to.

## What "from scratch" needs (external)
- The **DAIC-WOZ** corpus (request from the dataset authors). It already ships, per participant,
  the **COVAREP + FORMANT** audio features and **CLNF + HOG** video features, plus the
  transcripts — so audio/video are *read*, not re-extracted from raw wav/video.
- **spaCy** model `en_core_web_lg` (`python -m spacy download en_core_web_lg`).
- **`glove.840B.300d.txt`** (Stanford GloVe) — only to *build the word→vector table*.
- The **AVEC2017** `train/dev/test_split_Depression_AVEC2017.csv` splits (ship with DAIC-WOZ).

## Pipeline (run in this order)

| step | script | in → out | needs |
|---|---|---|---|
| 1 | `daicwoz_data_process.py` | raw transcripts/COVAREP/CLNF/HOG → per-turn `*.npy` per participant under `DAIC-WOZ/Embedding/merged_<id>_TRANSCRIPT_embedding/` (text/question GloVe, audio_covarep, video_clnf/hog, answers.txt, questions.txt) | spaCy `en_core_web_lg`, the corpus, the GloVe table from step 2 |
| 2 | `glove_vocab_preprocess.py` | corpus vocab + `glove.840B.300d.txt` → `DAIC-WOZ/utils/Glove_Preprocess/{word2id.txt, glove_embedding.npy}` (the text/question GloVe-300 lookup) | `glove.840B.300d.txt` |
| 3 | `emotion.py` | per-turn answers → per-turn VADER emotion (neg/neu/pos/compound) → `emotion_{train,dev}.npz` | `vaderSentiment` |
| 4 | `prepare_data.py` | the per-turn `*.npy` (+ AVEC2017 splits) → train-fit per-feature standardize + pad → **`data/{train,dev}_{raw,norm}.npz` + `meta.json`** | numpy/pandas |
| — | `raw_extract_audio.py` | provenance check: re-derives per-turn features from the raw `_P.zip` corpus and verifies row-cosine 0.999–1.000 vs the cached `*.npy` | the raw `189SAMPLES` corpus |

The model trains on the **raw** (un-standardized) tensors + per-modality input LayerNorm, i.e.
`{train,dev}_raw.npz`. After step 4, copy `data/*` into `../3-model/data/`.

> Notes: `daicwoz_data_process.py` and `glove_vocab_preprocess.py` are the **dataset authors' original
> extraction scripts**, kept verbatim; they use the corpus' own relative paths (`../Corpus`,
> `../Embedding`, `./Glove_Preprocess`) and a couple of absolute Windows paths in the
> CLNF/HOG functions — set those to your local DAIC-WOZ corpus before running. The
> already-built `word2id.txt` + `glove_embedding.npy` live under `DAIC-WOZ/utils/Glove_Preprocess/`,
> so if you keep those you can skip step 2 (and won't need `glove.840B.300d.txt`).
>
> The **augmented** text is featurized in `../2-augmentation/encode_syn_any.py` (synthetic
> paraphrase → GloVe-300), which **reuses the very same `word2id`/`glove_embedding` from step 2**
> so original and synthetic text share one vector space.
