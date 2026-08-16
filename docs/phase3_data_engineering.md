# Phase 3 — Data Engineering Pipeline

**Dependency audit:** cleaning → label encoding → tokenization are a real
chain — tokenization reads `clean_text`, which reads the raw dataset. No
independent sub-tasks worth a fan-out here (per graph-orchestrator's own
scope rule: a genuine chain isn't dressed up as a DAG), so this phase is one
reusable module (`src/emotion_llm/data.py`) exercised by two scripts, not a
set of parallel dispatches.

Code: `src/emotion_llm/data.py` (pipeline), `scripts/run_pipeline.py`
(end-to-end runner), `scripts/check_tokenizer_coverage.py` (empirical
`max_length` justification), `tests/test_data_pipeline.py` (9 tests, all
passing).

## Cleaning

`clean_text()` is deliberately minimal: HTML-entity unescaping (`&amp;` →
`&`) and whitespace collapsing, nothing else.

**Why each choice, not just what it does:**

- **No lowercasing.** `bert-base-uncased`'s own tokenizer already lowercases
  internally; `roberta-base` and `deberta-v3-base` are case-sensitive by
  design and rely on case as signal ("this is fine." vs. "THIS IS FINE!!!"
  carry different emotional weight). Lowercasing in the shared pipeline
  would silently help one baseline and hurt two others — the tokenizer is
  the right layer for that decision, not the shared cleaning step.
- **No punctuation stripping.** Repeated punctuation and emphasis
  (`!!!`, `...`) are emotionally informative on Reddit text specifically —
  stripping it would remove signal the model needs for exactly this task.
- **Anonymization placeholders kept as-is.** GoEmotions' curators already
  replaced named entities with tokens like `[NAME]`, `[RELIGION]` before
  release — these are meaningful placeholders, not artifacts to clean up.
  Verified by test: `test_clean_text_keeps_anonymization_placeholders`.

## Tokenization

`max_length=64`, chosen from evidence, not a guess. Phase 2 measured word
counts (median 12, p95 24), but subword tokenization inflates that —
`scripts/check_tokenizer_coverage.py` measured actual subword-token length
across all three Phase 4 baseline tokenizers on the full 43,410-example
train split:

| Tokenizer | mean tokens | p95 | p99 | truncated @ 48 | truncated @ 64 |
|---|---|---|---|---|---|
| bert-base-uncased | 19.2 | 34 | 38 | 0.03% | 0.01% |
| roberta-base | 18.8 | 33 | 37 | 0.06% | 0.02% |
| microsoft/deberta-v3-base | 18.7 | 33 | 36 | 0.04% | 0.01% |

All three tokenizers agree closely (unsurprising — they're tokenizing the
same short, informal English text). `max_length=64` truncates at most 1 in
5,000 examples across all three, at a modest, fixed compute cost per batch —
the right trade-off for this dataset. This also gives headroom for the
instruction-formatted prompts built in Phase 5, which wrap the raw text in a
template and will run longer than the raw text alone.

## Label transformation

`multihot()` converts GoEmotions' native `list[int]` label format into a
28-dim float vector (float, not int, so it plugs directly into
`BCEWithLogitsLoss` without a cast in Phase 4/6). Verified against known
inputs (`test_multihot_sets_correct_indices`, `test_multihot_empty_labels`).

`compute_class_weights()` implements standard inverse-frequency weighting
(`n_samples / (n_labels * class_count)`), directly responding to Phase 2's
184.7× imbalance finding. Computed on the real train split:

| Rank | Label | Weight |
|---|---|---|
| highest | grief | 20.135 |
| | pride | 13.967 |
| | relief | 10.133 |
| | nervousness | 9.453 |
| | embarrassment | 5.117 |
| ... | ... | ... |
| | annoyance | 0.628 |
| | gratitude | 0.582 |
| | approval | 0.528 |
| | admiration | 0.375 |
| lowest | neutral | 0.109 |

This is the mechanism, not just a number: passed as `pos_weight` to
`BCEWithLogitsLoss` in Phase 4, a false negative on `grief` costs ~20× what
a false negative on `neutral` costs during training — directly counteracting
the imbalance instead of letting the optimizer ignore rare classes to
minimize aggregate loss. Saved to `data/processed/class_weights.npy` so
Phase 4's training scripts load it rather than recomputing it.

## Train / validation / test split

**Decision: keep GoEmotions' official split (43,410 / 5,426 / 5,427)
unchanged rather than re-splitting.** Two reasons, both concrete:

1. **Comparability.** Every published GoEmotions baseline (including the
   original paper's own BERT results) reports against this exact split.
   Re-splitting would make Phase 4's baseline comparison table numbers
   incomparable to the literature, undermining the "know where you stand"
   value of having baselines at all.
2. **Leakage risk.** The official split was curated with awareness of
   Reddit thread/comment relationships; a naive random re-split risks
   putting near-duplicate or same-thread comments on both sides of the
   train/test boundary, inflating test performance in a way that wouldn't
   reproduce on genuinely new data. This is exactly the kind of "hidden
   edge" the graph-orchestrator dependency-audit habit is meant to catch —
   two dataset rows that look independent (different comments) but aren't,
   because they came from the same conversation.

Verified by test: `test_official_split_sizes` (network test, asserts the
loaded dataset matches 43410/5426/5427 exactly) and
`test_label_order_matches_upstream` (guards against GoEmotions changing
label order upstream and silently desyncing every trained model's label
indices).

## Pipeline output

Running `scripts/run_pipeline.py` end-to-end produces, under
`data/processed/` (git-ignored — regenerate with the script rather than
committing binary artifacts):

- `class_weights.npy` — the 28-dim weight vector above.
- `bert-base-uncased/`, `roberta-base/`, `microsoft__deberta-v3-base/` —
  tokenized `DatasetDict`s (train/validation/test), ready to feed directly
  into Phase 4's training loops with no further preprocessing.

Example transformation, verified against real data:

```
raw text     : "My favourite food is anything I didn't have to cook myself."
clean_text   : "My favourite food is anything I didn't have to cook myself."
label ids    : [27]
label_vector : sum=1.0 (len=28)
```

## Test coverage

9/9 tests passing (`uv run pytest tests/ -v`): cleaning behavior (whitespace,
HTML entities, anonymization placeholders), label encoding correctness
(including the empty-labels edge case), class-weight correctness (rare class
must outweigh frequent class — a regression here would silently undo the
imbalance mitigation), and two network tests confirming the pipeline still
agrees with the upstream dataset's split sizes and label order.

---

**Next up (Phase 4):** baseline experiments — train BERT-base, RoBERTa-base,
and DeBERTa-v3-base on the tokenized data produced here, evaluate with
accuracy/precision/recall/macro-F1/weighted-F1, and identify the strongest
baseline. This phase has genuine fan-out (the three models are fully
independent training runs) — will use graph-orchestrator to dispatch them
as parallel subagents rather than training sequentially.

Ready for the next phase?
