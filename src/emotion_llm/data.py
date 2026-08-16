"""Phase 3 — reusable GoEmotions data engineering pipeline.

Shared by the encoder baselines (Phase 4), the instruction-tuning dataset
builder (Phase 5), and the LoRA/QLoRA training scripts (Phases 6-7), so this
is the single place text cleaning, label encoding, and tokenization logic
live. Every downstream phase imports from here rather than re-implementing
it, so a change to (say) the cleaning rules only has to happen once.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"

# GoEmotions is a fixed, published label set — hardcoding order here (instead
# of only ever reading it off the loaded dataset) means label index 3 means
# "annoyance" everywhere in the codebase, even in code paths that never load
# the raw dataset (e.g. the inference pipeline in Phase 10/11).
LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
NUM_LABELS = len(LABEL_NAMES)


def load_raw() -> DatasetDict:
    """Load the official GoEmotions train/validation/test splits.

    We deliberately keep Google's original split rather than re-shuffling
    and re-splitting: (1) it's what every published GoEmotions baseline
    reports against, so our numbers stay comparable, and (2) re-splitting
    randomly risks leaking near-duplicate Reddit threads (same conversation,
    multiple comments) across train/test, which a fresh random split
    wouldn't know to avoid but the curators' split already accounts for.
    """
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    assert ds["train"].features["labels"].feature.names == LABEL_NAMES, (
        "GoEmotions label order changed upstream — LABEL_NAMES in this file "
        "must be updated to match, or every trained model's label indices "
        "silently become wrong."
    )
    return ds


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Minimal, conservative cleaning.

    GoEmotions text is already anonymized by the curators (named entities
    replaced with tokens like ``[NAME]``, ``[RELIGION]``) — we keep those
    tokens as-is rather than stripping them, since they're meaningful
    placeholders, not noise. We also don't lowercase or strip punctuation:
    case and punctuation carry emotional signal ("this is fine." vs.
    "THIS IS FINE!!!") that a subword tokenizer can use, and the encoder
    baselines' own tokenizers (e.g. bert-base-uncased) handle casing
    normalization internally when relevant — duplicating that here would be
    redundant and could disagree with what a given tokenizer expects.
    """
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Label transformation
# ---------------------------------------------------------------------------
def multihot(label_ids: list[int], num_labels: int = NUM_LABELS) -> list[float]:
    """Convert a list of label indices into a multi-hot float vector.

    Float (not int) because this feeds directly into BCEWithLogitsLoss for
    both the encoder baselines and the LLM classification head — no cast
    needed downstream.
    """
    vec = [0.0] * num_labels
    for label_id in label_ids:
        vec[label_id] = 1.0
    return vec


def compute_class_weights(label_matrix: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights for BCEWithLogitsLoss(pos_weight=...).

    Phase 2 found a 184.7x imbalance ratio (neutral: 14,219 vs. grief: 77
    in train). Unweighted BCE would let the model minimize loss by mostly
    ignoring rare classes like grief/pride/relief — this is what makes that
    failure mode addressable at training time instead of discovered only
    after Phase 9's per-class evaluation.

    Weight formula: n_samples / (n_labels * class_count), the standard
    sklearn-style inverse-frequency weighting, clipped so a class that
    happens to have zero positives in a given subset doesn't produce inf.
    """
    n_samples, n_labels = label_matrix.shape
    counts = label_matrix.sum(axis=0)
    counts = np.clip(counts, 1, None)
    weights = n_samples / (n_labels * counts)
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------
def build_clean_dataset() -> DatasetDict:
    """Load raw GoEmotions and attach `clean_text` + `label_vector` columns."""
    ds = load_raw()

    def _process(example):
        example["clean_text"] = clean_text(example["text"])
        example["label_vector"] = multihot(example["labels"])
        return example

    return ds.map(_process)


def tokenize_dataset(ds: DatasetDict, tokenizer, max_length: int = 64) -> DatasetDict:
    """Tokenize `clean_text` with a given HF tokenizer.

    max_length=64 default: Phase 2 measured a 24-word p95 on raw text.
    Subword tokenization inflates word count (see
    `scripts/check_tokenizer_coverage.py` for the empirical measurement
    across the three Phase 4 baseline tokenizers) — 64 subword tokens covers
    that with headroom without wasting compute on unnecessary padding.
    """

    def _tok(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return ds.map(_tok, batched=True)


@dataclass
class ProcessedSplit:
    label_matrix: np.ndarray
    texts: list[str]


def to_label_matrix(ds: Dataset) -> np.ndarray:
    return np.array(ds["label_vector"], dtype=np.float32)
