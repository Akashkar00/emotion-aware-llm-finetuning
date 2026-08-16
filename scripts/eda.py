"""Phase 2 — GoEmotions EDA.

Loads the `simplified` config of google-research-datasets/go_emotions,
computes dataset statistics, class distribution, label-imbalance analysis,
multi-label distribution, and emotion co-occurrence correlation, and writes
plots + CSVs to docs/assets/ for the Phase 2 report.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("Loading go_emotions (simplified config)...")
ds = load_dataset("google-research-datasets/go_emotions", "simplified")
label_names = ds["train"].features["labels"].feature.names
n_labels = len(label_names)
print(f"Loaded. {n_labels} labels: {label_names}")

# ---------------------------------------------------------------------------
# 1. Split sizes
# ---------------------------------------------------------------------------
split_sizes = {split: len(ds[split]) for split in ds}
print("Split sizes:", split_sizes)

# ---------------------------------------------------------------------------
# 2. Build multi-hot matrices per split + combined
# ---------------------------------------------------------------------------
def to_multihot(split_ds):
    mat = np.zeros((len(split_ds), n_labels), dtype=np.int32)
    for i, labels in enumerate(split_ds["labels"]):
        mat[i, labels] = 1
    return mat


multihot = {split: to_multihot(ds[split]) for split in ds}
combined = np.concatenate([multihot[s] for s in ds], axis=0)
combined_texts = sum([list(ds[s]["text"]) for s in ds], [])
print("Combined shape:", combined.shape)

# ---------------------------------------------------------------------------
# 3. Class distribution (train split — the one the model actually learns from)
# ---------------------------------------------------------------------------
train_counts = multihot["train"].sum(axis=0)
train_total = len(ds["train"])
dist_df = pd.DataFrame(
    {
        "label": label_names,
        "train_count": train_counts,
        "train_pct": (train_counts / train_total * 100).round(2),
    }
).sort_values("train_count", ascending=False).reset_index(drop=True)
dist_df.to_csv(ASSETS / "label_distribution_train.csv", index=False)
print("\nTrain label distribution (sorted):")
print(dist_df.to_string(index=False))

rarest = dist_df.tail(5)
most_common = dist_df.head(5)
imbalance_ratio = most_common["train_count"].iloc[0] / rarest["train_count"].iloc[-1]
print(f"\nMost common: {most_common['label'].iloc[0]} ({most_common['train_count'].iloc[0]})")
print(f"Rarest: {rarest['label'].iloc[-1]} ({rarest['train_count'].iloc[-1]})")
print(f"Imbalance ratio (most common / rarest): {imbalance_ratio:.1f}x")

# Plot: class distribution, sorted, log scale
plt.figure(figsize=(12, 8))
sns.barplot(data=dist_df, y="label", x="train_count", color="#4C72B0")
plt.xscale("log")
plt.xlabel("Count in train split (log scale)")
plt.ylabel("")
plt.title("GoEmotions — train label frequency (27 emotions + neutral)")
plt.tight_layout()
plt.savefig(ASSETS / "class_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Multi-label distribution: how many labels per example
# ---------------------------------------------------------------------------
labels_per_example = combined.sum(axis=1)
lpe_counts = pd.Series(labels_per_example).value_counts().sort_index()
lpe_pct = (lpe_counts / len(labels_per_example) * 100).round(2)
print("\nLabels per example (combined all splits):")
for k, v in lpe_counts.items():
    print(f"  {k} label(s): {v} examples ({lpe_pct[k]}%)")
print(f"Mean labels/example: {labels_per_example.mean():.3f}")
print(f"Zero-label examples: {(labels_per_example == 0).sum()}")

plt.figure(figsize=(7, 5))
sns.barplot(x=lpe_counts.index.astype(str), y=lpe_counts.values, color="#DD8452")
plt.xlabel("Number of emotion labels on the example")
plt.ylabel("Number of examples")
plt.title("GoEmotions — multi-label distribution (all splits combined)")
plt.tight_layout()
plt.savefig(ASSETS / "labels_per_example.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Emotion co-occurrence correlation (phi coefficient == Pearson on binary)
# ---------------------------------------------------------------------------
label_df = pd.DataFrame(combined, columns=label_names)
corr = label_df.corr()
corr.to_csv(ASSETS / "emotion_correlation_matrix.csv")

# Top positively-correlated pairs (excluding self and neutral, which mostly
# anti-correlates with everything by construction)
pairs = []
for i in range(n_labels):
    for j in range(i + 1, n_labels):
        pairs.append((label_names[i], label_names[j], corr.iloc[i, j]))
pairs_df = pd.DataFrame(pairs, columns=["label_a", "label_b", "correlation"])
pairs_df = pairs_df.sort_values("correlation", ascending=False)
top_pairs = pairs_df.head(15)
pairs_df.to_csv(ASSETS / "emotion_pair_correlations.csv", index=False)
print("\nTop 15 most correlated emotion pairs (co-occur together more than chance):")
print(top_pairs.to_string(index=False))

plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.3, cbar_kws={"shrink": 0.7})
plt.title("GoEmotions — emotion co-occurrence correlation (phi coefficient)")
plt.tight_layout()
plt.savefig(ASSETS / "emotion_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Text length stats (word count) — useful for tokenizer max_length choice later
# ---------------------------------------------------------------------------
word_counts = np.array([len(t.split()) for t in combined_texts])
print("\nText length (words) stats:")
print(f"  mean={word_counts.mean():.1f} median={np.median(word_counts):.0f} "
      f"p95={np.percentile(word_counts, 95):.0f} max={word_counts.max()}")

# ---------------------------------------------------------------------------
# 7. Dump a JSON summary for the report to consume
# ---------------------------------------------------------------------------
summary = {
    "split_sizes": split_sizes,
    "n_labels": n_labels,
    "label_names": label_names,
    "most_common_label": {"label": most_common["label"].iloc[0], "count": int(most_common["train_count"].iloc[0])},
    "rarest_label": {"label": rarest["label"].iloc[-1], "count": int(rarest["train_count"].iloc[-1])},
    "imbalance_ratio": round(float(imbalance_ratio), 1),
    "mean_labels_per_example": round(float(labels_per_example.mean()), 3),
    "labels_per_example_pct": {str(k): float(v) for k, v in lpe_pct.items()},
    "top_correlated_pairs": top_pairs.head(10).to_dict(orient="records"),
    "word_count_stats": {
        "mean": round(float(word_counts.mean()), 1),
        "median": float(np.median(word_counts)),
        "p95": float(np.percentile(word_counts, 95)),
        "max": int(word_counts.max()),
    },
}
with open(ASSETS / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Outputs written to", ASSETS)
