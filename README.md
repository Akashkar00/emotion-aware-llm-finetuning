# Emotion-Aware LLM: Fine-Tuning Large Language Models for Multi-Label Emotion Understanding

Research-grade project demonstrating LoRA / QLoRA fine-tuning of open LLMs
for multi-label emotion classification on GoEmotions, benchmarked against
encoder-only baselines (BERT, RoBERTa, DeBERTa-v3).

Status: Phase 3 (Data Engineering Pipeline) complete. See `docs/` for phase reports.

## Setup

```bash
uv sync
uv run pytest tests/ -v          # 9 tests, ~9s (2 require network)
uv run python scripts/run_pipeline.py   # regenerate data/processed/
```

## Phases
- [x] Phase 1 — Research & Planning
- [x] Phase 2 — Dataset Deep Dive (EDA)
- [x] Phase 3 — Data Engineering Pipeline
- [ ] Phase 4 — Baseline Experiments
- [ ] Phase 5 — Instruction Dataset Creation
- [ ] Phase 6 — LoRA Fine-Tuning
- [ ] Phase 7 — QLoRA Fine-Tuning
- [ ] Phase 8 — Hyperparameter Experiments
- [ ] Phase 9 — Evaluation Framework
- [ ] Phase 10 — Model Optimization
- [ ] Phase 11 — Emotion-Aware Response Generation
- [ ] Phase 12 — Research Documentation
