# Project Status — Emotion-Aware LLM Fine-Tuning

Last updated: 2026-08-17. This is the handoff doc for continuing the project
on Google Colab (or any CUDA machine) — read this first before doing
anything else.

## Why the move to Colab

This machine is an Apple M1 with 8GB unified memory. Two real constraints
showed up doing Phase 4 locally:

1. **`uv` and the Homebrew Python on this machine are legacy x86_64
   binaries running under Rosetta**, even though the hardware is arm64. This
   was fixed for local dev (the project venv now points at a native arm64
   Python via `/opt/anaconda3/bin/python3.12`, so `torch` installs with MPS
   support — see `pyproject.toml` / `.python-version`), but this is a
   workaround, not a real fix to the machine's global toolchain.
2. **Even with MPS working, training throughput is too slow for this
   project's scope.** A calibration run (`bert-base-uncased`, batch 16, MPS)
   measured roughly 0.2–3 iterations/sec depending on warmup state — at the
   low end, one epoch over the full 43,410-example train split would take
   on the order of **hours per model**, and Phase 4 alone needs 3 models ×
   several epochs, before Phases 6–8 (LoRA, QLoRA, hyperparameter sweeps —
   the actual point of this project) even start, on models 2–20x larger
   than the encoder baselines. That's not viable in 8GB of unified memory
   shared with the OS regardless of throughput.

**Decision: move all GPU-bound training (Phase 4 onward) to Google Colab.**
Local machine stays for data work, EDA, pipeline code, and writing — the
things Phases 1–3 already proved work fine here.

## What's done (Phases 1–3, complete, committed)

| Phase | Status | Key artifacts |
|---|---|---|
| 1 — Research & Planning | ✅ Done | `docs/phase1_research_and_planning.md` |
| 2 — Dataset Deep Dive (EDA) | ✅ Done | `docs/phase2_dataset_deep_dive.md`, plots/CSVs in `docs/assets/` |
| 3 — Data Engineering Pipeline | ✅ Done | `src/emotion_llm/data.py`, `scripts/run_pipeline.py`, `tests/test_data_pipeline.py` (9/9 passing) |

Findings that carry forward and matter for every remaining phase:

- 54,263 examples (43,410/5,426/5,427 train/val/test), 28 labels (27
  emotions + neutral), official split kept as-is (don't re-split — see
  Phase 3 doc for why).
- **184.7× class imbalance** (`neutral`: 14,219 vs. `grief`: 77). Class
  weights are precomputed in `emotion_llm.data.compute_class_weights()` and
  used as `BCEWithLogitsLoss(pos_weight=...)` — don't train unweighted.
- `max_length=64` tokens covers >99.98% of examples (empirically measured
  across BERT/RoBERTa/DeBERTa-v3 tokenizers — see
  `scripts/check_tokenizer_coverage.py`).
- Text cleaning is intentionally minimal (HTML unescape + whitespace only —
  no lowercasing/punctuation stripping). Don't add more without re-reading
  the reasoning in `docs/phase3_data_engineering.md` — it's deliberate.

## What's in progress (Phase 4)

**Harness is built and correctness-verified, full training is not yet run.**

- `scripts/train_baseline.py` — trains one encoder baseline
  (`bert-base-uncased` / `roberta-base` / `microsoft/deberta-v3-base`),
  applies the Phase 3 class weights, evaluates with subset accuracy +
  precision/recall/F1 (macro, weighted, micro) on both validation and test.
- **Smoke-tested successfully** on a 320-example slice (1 epoch, ran
  end-to-end, produced sane-shaped output) — this proved the harness is
  correct, not that the model is trained. That smoke-test output was
  deleted; don't treat any `results/` you find locally as real numbers
  unless it came from a full run.
- **Not yet run at full scale.** This is the first thing to do on Colab.

## How to resume on Colab

1. Push this repo to GitHub (or upload the folder), `git clone` it in a
   Colab notebook. `git init` already ran locally; commits exist through
   Phase 3 plus the Phase 4 harness — check `git log` for exact state.
2. In the first Colab cell:
   ```
   !pip install -r requirements.txt
   ```
   `requirements.txt` was exported from the local `uv.lock` via
   `uv export --no-hashes --no-dev -o requirements.txt` — it pins the same
   versions verified locally (transformers 5.15.0, accelerate 1.14.0, etc.),
   so `torch==2.13.0` will resolve to a CUDA-enabled Linux wheel
   automatically on Colab's GPU runtime (Runtime → Change runtime type →
   GPU, before installing).
3. Regenerate the tokenized data (not committed — it's derived, git-ignored
   by design):
   ```
   !python scripts/run_pipeline.py
   ```
4. Run all three baselines — sequentially is fine on a single Colab GPU for
   the same reason it had to be sequential locally (shared device), but a
   Colab GPU will be dramatically faster per run than the M1 was:
   ```
   !python scripts/train_baseline.py --model_name bert-base-uncased
   !python scripts/train_baseline.py --model_name roberta-base
   !python scripts/train_baseline.py --model_name microsoft/deberta-v3-base
   ```
   Default is 3 epochs, batch size 16 — adjust `--batch_size` up
   (Colab GPUs have far more than 8GB) if you want faster iteration; a T4
   (16GB) should comfortably handle batch 32–64 for these model sizes at
   `max_length=64`.
5. Results land in `results/phase4_baselines/<model>/metrics.json`. Pull
   those back down (or keep working in Colab / Drive-mounted storage) to
   write up `docs/phase4_baseline_experiments.md` — the comparison table,
   strongest-baseline identification, and per-class discussion the phase
   plan calls for.

## What's left (Phases 4 finish through 12)

| Phase | Needs GPU? | What it produces |
|---|---|---|
| 4 — Baseline Experiments | Yes (fast on any Colab GPU) | Comparison table across BERT/RoBERTa/DeBERTa-v3, strongest baseline identified |
| 5 — Instruction Dataset Creation | No — pure data work, could go back to local | GoEmotions reformatted as instruction/JSON-output examples for the LLM phases |
| 6 — LoRA Fine-Tuning | Yes, meaningfully | LoRA adapter on Phi-3 Mini (per Phase 1's recommendation) |
| 7 — QLoRA Fine-Tuning | Yes, meaningfully | 4-bit NF4 QLoRA run + full-FT vs LoRA vs QLoRA comparison |
| 8 — Hyperparameter Experiments | Yes, most GPU-hours of any phase | Sweep over LR/epochs/batch/rank/alpha/warmup — this is the one to run as parallel Colab sessions or a scripted sweep, not by hand |
| 9 — Evaluation Framework | No (once models exist) | Confusion matrices, PR/ROC curves, failure analysis |
| 10 — Model Optimization | Partially | Quantized inference, latency/throughput benchmarks |
| 11 — Emotion-Aware Response Generation | No | Small wrapper app — deliberately kept minimal per the project's own rules (80% fine-tuning, 20% application) |
| 12 — Research Documentation | No | Final report, resume bullets, README |

## Known gotchas (don't rediscover these on Colab)

- `microsoft/deberta-v3-base`'s tokenizer needs `sentencepiece` **and**
  `protobuf` installed explicitly, or `AutoTokenizer.from_pretrained` fails
  with a tiktoken-fallback error. Already in `requirements.txt`.
- This `transformers` version (5.15.0) removed `TrainingArguments`'
  `use_mps_device` and `warmup_ratio` args — `train_baseline.py` already
  works around both (device auto-detection, `warmup_steps` computed
  manually). If Colab resolves a different transformers version, re-check
  the `TrainingArguments` signature before assuming those args exist.
- Set `HF_TOKEN` (or accept the rate-limit warning) — anonymous HF Hub
  requests work but are slower/rate-limited.

---

Ready to continue whenever — pick up at "How to resume on Colab" above.
