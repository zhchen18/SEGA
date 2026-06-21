# SEGA: Depression Detection with LLM-Empowered Structural Element Graph

Thanks for your patience with the wait. I have been quite busy these days, and now that I finally
have some time, I have retrieved and organized this code from a large backup and released it here.
This is the reference implementation of **SEGA** (structural-element graph) and its **SEGA++**
extension (the paper's §3.3 "A Helping Hand from LLMs" text augmentation) for depression
detection, on two corpora: **DAIC-WOZ** and **EATD**. For each corpus, **SEGA** (no augmentation)
and **SEGA++** (LLM augmentation) are the two run modes of one trainer.

## Structure

```
SEGA-GITHUB/
├── DAIC-WoZ/
│   ├── 1-preprocess/    raw DAIC-WOZ  -> per-turn GloVe / COVAREP / CLNF features
│   ├── 2-augmentation/  GPT-3.5 paraphrase augmentation + cached synthetic features
│   └── 3-model/         SEGA backbone, unified trainer, bundled data, checkpoints, logs
└── EATD/
    ├── 1-preprocess/    raw EATD      -> per-segment Chinese-BERT / openSMILE features
    ├── 2-augmentation/  GPT-3.5 paraphrase augmentation + cached synthetic features
    └── 3-model/         SEGA backbone, unified trainer, bundled data, checkpoints, logs
```

## Requirements

Python 3.10+, `torch` (tested 2.0.1+cu118), `numpy`, `pandas`, `scikit-learn`; EATD text
re-encoding additionally needs `transformers`. A single GPU suffices (CPU also works).

## Run

Because the raw DAIC-WOZ and EATD corpora are under copyright restrictions, we cannot redistribute
them directly. Instead, we bundle the corresponding feature vectors under each `3-model/`, which can
be used for training out of the box. If you want to start from the raw data, please apply for the
datasets yourself and follow the processing steps in `1-preprocess/`.

```bash
# DAIC-WOZ
cd DAIC-WoZ/3-model
GPU=0 bash run_sega.sh        # SEGA   (no augmentation)
GPU=0 bash run_sega++.sh      # SEGA++ (LLM augmentation)

# EATD
cd EATD/3-model
GPU=0 bash run_sega.sh
GPU=0 bash run_sega++.sh
```

## Checkpoints

Bundled best checkpoints reproduce the reported numbers directly (no training):

```bash
# DAIC-WOZ  (dev Macro-F1)
cd DAIC-WoZ/3-model
python scripts/eval_ckpt.py checkpoints/sega_run1.pt      # -> 86.74
python scripts/eval_ckpt.py checkpoints/segapp_run1.pt    # -> 90.67

# EATD  (3-fold CV Macro-F1)
cd EATD/3-model
python scripts/eval_ckpt.py checkpoints/sega_best.pt      # -> 81.06
python scripts/eval_ckpt.py checkpoints/segapp_best.pt    # -> 82.24
```

| dataset | metric | SEGA | SEGA++ |
|---|---|---|---|
| DAIC-WOZ | dev Macro-F1 (3-seed avg) | 85.00 | **88.39** |
| EATD | 3-fold CV Macro-F1 | 81.06 | **82.24** |

## Citation

If you find this code and data useful, please cite:

```bibtex
@inproceedings{chen2024depression,
  title={Depression detection in clinical interviews with LLM-empowered structural element graph},
  author={Chen, Zhuang and Deng, Jiawen and Zhou, Jinfeng and Wu, Jincenzi and Qian, Tieyun and Huang, Minlie},
  booktitle={Proceedings of the 2024 conference of the north american chapter of the association for computational linguistics: Human language technologies (volume 1: Long papers)},
  pages={8181--8194},
  year={2024}
}
```

## Misc

- The DAIC-WOZ data is very large and fairly complex to process. The transcripts of two
  patients, **451** and **458**, contain errors, so I listened to the recordings myself and
  fixed them by hand. My corrected versions are provided at
  `DAIC-WoZ/1-preprocess/CORRECT_451_TRANSCRIPT.csv` and `CORRECT_458_TRANSCRIPT.csv`.

- Depression detection in the lab setting has very little data, so the results swing a lot from
  run to run. On top of that, different data-processing choices can change the performance by a
  lot, and previous work usually does not explain these details. I tried about 20-30
  combinations — different encoders, different ways to align the modalities, different feature
  granularities, and so on — before settling on the final setup. This also made it very hard to
  retrieve the right code: I had more than 100 sub-repos with all kinds of variants. Thanks to the
  coding agents, I was finally able to sort it all out and release it. Even so, it still took me
  three full days. Thank you for your patience; I know it has been a long wait.

- SEGA was once rejected by IJCAI 2023 and AAAI 2024, both for reasons that did not make much
  sense. At IJCAI 2023 we submitted to the AI for Social Good track, and the reviewers thought
  depression detection had nothing to do with social good and raised ethical concerns (even
  though we used open data with proper authorization), so it was rejected outright. At AAAI 2024,
  the reviewers gave us low scores and asked us to run experiments that were already shown in the
  paper, and it was rejected again. In short, SEGA has been a very hard journey, and I am very
  grateful to the responsible reviewers at NAACL 2024.
