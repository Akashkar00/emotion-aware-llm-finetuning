"""Phase 4 — train a single encoder baseline for multi-label emotion classification.

Usage:
    uv run python scripts/train_baseline.py --model_name bert-base-uncased
    uv run python scripts/train_baseline.py --model_name roberta-base
    uv run python scripts/train_baseline.py --model_name microsoft/deberta-v3-base

Run once per model — sequentially, not concurrently. Hardware note: this
machine is an 8GB-unified-memory Apple M1 (MPS backend). Training three
encoder models concurrently would contend for the same memory pool and GPU
cores rather than actually parallelizing, so each run is a separate
sequential process (see docs/phase4_baseline_experiments.md for the
dependency-audit reasoning).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from emotion_llm.data import LABEL_NAMES, NUM_LABELS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "phase4_baselines"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class WeightedTrainer(Trainer):
    """Applies the Phase 3 inverse-frequency class weights as BCE pos_weight."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss(
            pos_weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    subset_acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    return {
        "subset_accuracy": subset_acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    tok_dir = PROCESSED_DIR / args.model_name.replace("/", "__")
    if not tok_dir.exists():
        raise SystemExit(
            f"{tok_dir} not found — run scripts/run_pipeline.py first (Phase 3)."
        )
    ds = load_from_disk(str(tok_dir))
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(args.max_train_samples))
    print(f"train={len(ds['train'])} validation={len(ds['validation'])} test={len(ds['test'])}")

    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "label_vector"}
    for split in ds:
        drop_cols = [c for c in ds[split].column_names if c not in keep_cols]
        ds[split] = ds[split].remove_columns(drop_cols)
        ds[split] = ds[split].rename_column("label_vector", "labels")
    ds.set_format("torch")

    class_weights = torch.tensor(
        np.load(PROCESSED_DIR / "class_weights.npy"), dtype=torch.float32
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    out_dir = RESULTS_DIR / args.model_name.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, len(ds["train"]) // args.batch_size)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    print("\nEvaluating on validation split...")
    val_metrics = trainer.evaluate(ds["validation"])
    print("\nEvaluating on test split...")
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")

    result = {
        "model_name": args.model_name,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_examples": len(ds["train"]),
        "train_time_seconds": round(train_time, 1),
        "validation": {k: v for k, v in val_metrics.items()},
        "test": {k: v for k, v in test_metrics.items()},
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to {out_dir / 'metrics.json'}")
    print(f"Train time: {train_time / 60:.1f} min")
    print(f"Validation macro-F1: {val_metrics['eval_f1_macro']:.4f}")
    print(f"Test macro-F1: {test_metrics['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
