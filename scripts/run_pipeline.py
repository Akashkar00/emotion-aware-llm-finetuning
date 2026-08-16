"""Phase 3 — run the full data engineering pipeline end to end.

Cleans GoEmotions, attaches multi-hot label vectors, tokenizes with each of
the three Phase 4 baseline tokenizers, computes class weights for the
imbalance found in Phase 2, and saves everything to data/processed/ (git-
ignored — regenerate with this script rather than committing the artifacts).
"""

from pathlib import Path

from transformers import AutoTokenizer

from emotion_llm.data import (
    NUM_LABELS,
    build_clean_dataset,
    compute_class_weights,
    to_label_matrix,
    tokenize_dataset,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
BASELINE_TOKENIZERS = ["bert-base-uncased", "roberta-base", "microsoft/deberta-v3-base"]
MAX_LENGTH = 64

print("Building clean dataset (text cleaning + multi-hot label encoding)...")
ds = build_clean_dataset()
for split in ds:
    print(f"  {split}: {len(ds[split])} examples")

print("\nExample transformation:")
ex = ds["train"][0]
print(f"  raw text     : {ex['text']!r}")
print(f"  clean_text   : {ex['clean_text']!r}")
print(f"  label ids    : {ex['labels']}")
print(f"  label_vector : sum={sum(ex['label_vector'])} (len={len(ex['label_vector'])})")

print("\nComputing class weights from train split (addresses Phase 2's 184.7x imbalance)...")
train_labels = to_label_matrix(ds["train"])
weights = compute_class_weights(train_labels)
from emotion_llm.data import LABEL_NAMES

weight_table = sorted(zip(LABEL_NAMES, weights), key=lambda x: -x[1])
print("  Top 5 highest-weighted (rarest) classes:")
for name, w in weight_table[:5]:
    print(f"    {name:<15} weight={w:.3f}")
print("  Top 5 lowest-weighted (most frequent) classes:")
for name, w in weight_table[-5:]:
    print(f"    {name:<15} weight={w:.3f}")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np

np.save(PROCESSED_DIR / "class_weights.npy", weights)
print(f"\nSaved class weights to {PROCESSED_DIR / 'class_weights.npy'}")

for model_name in BASELINE_TOKENIZERS:
    print(f"\nTokenizing with {model_name} (max_length={MAX_LENGTH})...")
    tok = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tok, max_length=MAX_LENGTH)
    out_dir = PROCESSED_DIR / model_name.replace("/", "__")
    tokenized.save_to_disk(str(out_dir))
    print(f"  saved to {out_dir}")

print("\nPipeline complete.")
print(f"NUM_LABELS = {NUM_LABELS}")
