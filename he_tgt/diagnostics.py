"""Class-wise tables, cumulative confusion counts and reliability plots."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.stats import spearmanr
from .utils import read_json, write_json


def reliability(labels, probabilities, bins=15):
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = indices == index
        if mask.any():
            rows.append({"bin": index, "n": int(mask.sum()), "confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean())})
    clipped = np.clip(probabilities, 1e-15, 1.)
    entropy = -(clipped * np.log(clipped)).sum(1)
    correlation = spearmanr(entropy, ~correct).statistic
    return rows, {"ece15": sum(row["n"] / len(labels) * abs(row["confidence"] - row["accuracy"]) for row in rows),
        "brier": float(np.mean(np.sum((probabilities - np.eye(probabilities.shape[1])[labels]) ** 2, axis=1))),
        "nll": float(-np.log(clipped[np.arange(len(labels)), labels]).mean()),
        "entropy_error_spearman": float(correlation) if np.isfinite(correlation) else None}


def run(files, metadata, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    names = read_json(metadata)["label_names"]
    per_class, calibration, all_probabilities, all_labels = [], [], [], []
    reference_labels = reference_indices = None
    for file in files:
        with np.load(file, allow_pickle=False) as arr:
            labels, probabilities, indices = arr["labels"], arr["probabilities"], arr["raw_indices"]
        if reference_labels is None:
            reference_labels, reference_indices = labels.copy(), indices.copy()
        else:
            np.testing.assert_array_equal(labels, reference_labels)
            np.testing.assert_array_equal(indices, reference_indices)
        precision, recall, f1, count = precision_recall_fscore_support(labels, probabilities.argmax(1), labels=np.arange(len(names)), zero_division=0)
        for index, name in enumerate(names):
            per_class.append({"run": Path(file).stem, "class": name, "n": int(count[index]),
                              "precision": precision[index] * 100, "recall": recall[index] * 100, "f1": f1[index] * 100})
        _, row = reliability(labels, probabilities)
        calibration.append({"run": Path(file).stem, **row})
        all_probabilities.append(probabilities)
        all_labels.append(labels)
    frame = pd.DataFrame(per_class)
    frame.to_csv(output / "classwise_runs.csv", index=False)
    frame.groupby("class", sort=False)[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(output / "classwise_summary.csv")
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    labels, probabilities = np.concatenate(all_labels), np.concatenate(all_probabilities)
    counts = confusion_matrix(labels, probabilities.argmax(1), labels=np.arange(len(names)))
    write_json(output / "count_scope.json", {"graph_fits": len(files), "queries_per_fit": len(reference_labels),
        "prediction_instances": len(labels), "row_totals": counts.sum(1).tolist(), "confusion_counts": counts.tolist()})
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(counts, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Prediction count", fraction=.046, pad=.04)
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)), xticklabels=names, yticklabels=names,
           xlabel="Predicted class", ylabel="True class")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if counts[i, j] > counts.max() * .5 else "black")
    fig.savefig(output / "confusion.pdf", bbox_inches="tight")
    plt.close(fig)
    bins, _ = reliability(labels, probabilities)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color=".5", linewidth=1)
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color="#1f77b4")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy")
    fig.savefig(output / "reliability.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.predictions, args.metadata, args.output)
