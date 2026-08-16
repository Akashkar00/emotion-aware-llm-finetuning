"""Empirically justify the `max_length` used in `emotion_llm.data.tokenize_dataset`.

Phase 2's word-count stats (median 12, p95 24 words) are on whitespace-split
words, not subword tokens. Subword tokenization (WordPiece for BERT, BPE for
RoBERTa/DeBERTa) can inflate that count, especially on GoEmotions' anonymized
placeholder tokens (e.g. `[NAME]`). This script measures actual truncation
rates at a few candidate max_length values across the three Phase 4 baseline
tokenizers, so the default in data.py is chosen from evidence, not guessed.
"""

import numpy as np
from transformers import AutoTokenizer

from emotion_llm.data import build_clean_dataset

CANDIDATE_MODELS = ["bert-base-uncased", "roberta-base", "microsoft/deberta-v3-base"]
CANDIDATE_LENGTHS = [16, 32, 48, 64]

print("Loading + cleaning GoEmotions...")
ds = build_clean_dataset()
texts = ds["train"]["clean_text"]
print(f"{len(texts)} train examples\n")

for model_name in CANDIDATE_MODELS:
    print(f"=== {model_name} ===")
    tok = AutoTokenizer.from_pretrained(model_name)
    lengths = np.array([len(tok.encode(t, add_special_tokens=True)) for t in texts])
    print(f"  token length: mean={lengths.mean():.1f} median={np.median(lengths):.0f} "
          f"p95={np.percentile(lengths, 95):.0f} p99={np.percentile(lengths, 99):.0f} max={lengths.max()}")
    for max_len in CANDIDATE_LENGTHS:
        truncated_pct = (lengths > max_len).mean() * 100
        print(f"  max_length={max_len:>3}: {truncated_pct:.2f}% of examples truncated")
    print()
