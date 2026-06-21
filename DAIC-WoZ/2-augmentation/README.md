# 2-augmentation — paper §3.3 "A Helping Hand from LLMs"

Generates and encodes the LLM text augmentation that powers **SEGA++**, following the paper's
augmentation recipe (GPT-3.5 principle-based rephrasing).

### `gpt_augment_daic.py` — paraphrase generation (GPT-3.5)
For each participant answer turn, GPT-3.5 produces **3 paraphrases** under the paper's **five
principles** (content integrity, colloquial naturalness, respect/appropriateness, length
consistency, tolerate disfluency), written `[SEP]`-joined to `syn_answers.txt` per participant
folder. Chunked, parallel, resumable. Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`,
`OPENAI_MODEL`) to run.

### `encode_syn.py` — synthetic text → per-turn GloVe-300
Encodes the paraphrases into the **same** GloVe-300 space as the original text (the corpus
`word2id` + `glove_embedding.npy`; OOV ≈ 2%, synthetic-vs-original row-cosine ≈ 0.82 — genuinely
different, semantically faithful). Writes `aug/text_syn_{train,dev}.npz` (3 variants each).

### Cached features (`aug/`)
- `text_syn_{train,dev}.npz` — the LLM paraphrase features consumed by the model
  (3 variants per turn), **bundled** so reproduction needs no API access.

`3-model/scripts/train.py` reads these via the `3-model/aug` symlink. During training each
interview's text is, with probability `augprob`, swapped for a random paraphrase — a
size-preserving way to mix the synthetic data into training (see the top-level README).

### Regenerate (optional)
```bash
OPENAI_API_KEY=... python gpt_augment_daic.py                 # -> syn_answers.txt (resumable)
SYNFILE=syn_answers.txt python encode_syn.py                  # -> aug/text_syn_*.npz
```
