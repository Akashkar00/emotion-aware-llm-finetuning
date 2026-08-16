"""Unit tests for the Phase 3 data pipeline.

`test_official_split_sizes` and `test_label_order_matches_upstream` hit the
network (they load the real dataset) — everything else is pure-function and
runs offline in milliseconds.
"""

import numpy as np
import pytest

from emotion_llm.data import (
    LABEL_NAMES,
    NUM_LABELS,
    clean_text,
    compute_class_weights,
    load_raw,
    multihot,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("hello    world\n\n") == "hello world"


def test_clean_text_unescapes_html_entities():
    assert clean_text("Tom &amp; Jerry") == "Tom & Jerry"


def test_clean_text_keeps_anonymization_placeholders():
    # GoEmotions replaces named entities with tokens like [NAME] — these are
    # meaningful, not noise, and must survive cleaning unchanged.
    assert clean_text("thanks [NAME], appreciate it") == "thanks [NAME], appreciate it"


def test_multihot_sets_correct_indices():
    vec = multihot([1, 3], num_labels=5)
    assert vec == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_multihot_empty_labels():
    assert multihot([], num_labels=4) == [0.0, 0.0, 0.0, 0.0]


def test_compute_class_weights_rare_class_gets_higher_weight():
    # 100 examples, label 0 present in all of them, label 1 present in 1.
    matrix = np.zeros((100, 2), dtype=np.float32)
    matrix[:, 0] = 1.0
    matrix[0, 1] = 1.0
    weights = compute_class_weights(matrix)
    assert weights[1] > weights[0], "the rarer class must get a larger weight"


def test_compute_class_weights_shape():
    matrix = np.random.randint(0, 2, size=(50, NUM_LABELS)).astype(np.float32)
    weights = compute_class_weights(matrix)
    assert weights.shape == (NUM_LABELS,)


@pytest.mark.network
def test_official_split_sizes():
    ds = load_raw()
    assert len(ds["train"]) == 43410
    assert len(ds["validation"]) == 5426
    assert len(ds["test"]) == 5427


@pytest.mark.network
def test_label_order_matches_upstream():
    ds = load_raw()
    assert ds["train"].features["labels"].feature.names == LABEL_NAMES
