"""Generate audited, publication-ready figures for the revision.

All graphics are rendered with matplotlib and exported as editable SVG/PDF
plus a high-resolution PNG for LaTeX compatibility.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})

INK = "#1F2933"
MUTED = "#52606D"
BLUE = "#3B6FB6"
BLUE_LIGHT = "#DCE8F7"
GREEN = "#2F855A"
GREEN_LIGHT = "#DDEFE4"
AMBER = "#B7791F"
AMBER_LIGHT = "#F8E8C4"
RED = "#B64B4B"
RED_LIGHT = "#F5DDDD"
GRAY_LIGHT = "#F2F4F7"


def rounded_box(ax, xy, width, height, facecolor, edgecolor=INK,
                linewidth=0.9, radius=0.025, zorder=1):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        transform=ax.transAxes, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, linewidth=1.2, style="-|>", zorder=4):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=9,
        linewidth=linewidth, color=color,
        transform=ax.transAxes, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=7, weight="normal", color=INK,
          ha="center", va="center", zorder=5, **kwargs):
    ax.text(
        x, y, text, fontsize=size, fontweight=weight, color=color,
        ha=ha, va=va, transform=ax.transAxes, zorder=zorder, **kwargs,
    )


def node(ax, x, y, color, radius=0.012, edge=INK, zorder=5):
    patch = Circle(
        (x, y), radius=radius, facecolor=color, edgecolor=edge,
        linewidth=0.7, transform=ax.transAxes, zorder=zorder,
    )
    ax.add_patch(patch)


def save_figure(fig, base):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else np.nan


def panel_label(ax, letter):
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left", color=INK)


def training_robustness_figure(expdir, outdir):
    """Show convergence, real-label scarcity, and history-length sensitivity."""
    expdir = Path(expdir)
    history = read_csv(expdir / "history.csv")
    results = read_csv(expdir / "results.csv")
    scarcity = read_csv(expdir / "scarcity_summary.csv")

    conv_id = next(
        row["run_id"] for row in results if row["experiment"] == "convergence"
    )
    conv = [row for row in history if row["run_id"] == conv_id]
    pre = [row for row in conv if row["stage"] == "pretrain"]
    fine = [row for row in conv if row["stage"] == "finetune"]
    root = [row for row in conv if row["stage"] == "root"]
    k_rows = sorted(
        (row for row in results if row["experiment"] == "history"),
        key=lambda row: int(row["history_length"]),
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.75))
    ax = axes[0, 0]
    ax.plot([int(r["epoch"]) for r in pre], [as_float(r, "loss") for r in pre],
            "o-", color=BLUE, lw=1.4, ms=3.5, label="Edge pre-training")
    ax.plot([int(r["epoch"]) for r in root], [as_float(r, "loss") for r in root],
            "s-", color=RED, lw=1.4, ms=3.2, label="Source head")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("Optimization losses decrease")
    ax.legend(loc="upper right", fontsize=6.2)
    ax.grid(axis="y", color="#D9DEE5", lw=0.5, alpha=0.8)
    panel_label(ax, "a")

    ax = axes[0, 1]
    epochs = [int(r["epoch"]) for r in fine]
    losses = [as_float(r, "loss") for r in fine]
    val_f1 = [as_float(r, "val_f1") for r in fine]
    loss_line = ax.plot(epochs, losses, "o-", color=BLUE, lw=1.4, ms=3.4,
                        label="Fine-tuning loss")[0]
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Training loss", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx()
    f1_line = ax2.plot(epochs, val_f1, "s-", color=AMBER, lw=1.4, ms=3.2,
                       label="Validation macro F1")[0]
    best = int(np.nanargmax(val_f1))
    ax2.scatter([epochs[best]], [val_f1[best]], s=34, facecolor="white",
                edgecolor=RED, linewidth=1.2, zorder=5)
    ax2.annotate(f"best: {val_f1[best]:.1f}%", (epochs[best], val_f1[best]),
                 xytext=(-5, -18), textcoords="offset points", ha="right",
                 fontsize=6.1, color=RED)
    ax2.set_ylabel("Validation macro F1 (%)", color=AMBER)
    ax2.tick_params(axis="y", labelcolor=AMBER)
    ax.set_title("Validation performance remains unstable")
    ax.set_xlim(0.7, 8.8)
    ax.text(8.12, losses[-1], "loss", color=BLUE, fontsize=6.0,
            ha="left", va="center")
    ax2.text(8.12, val_f1[-1], "validation F1", color=AMBER, fontsize=6.0,
             ha="left", va="center")
    ax.grid(axis="x", color="#E3E7EC", lw=0.5)
    panel_label(ax, "b")

    ax = axes[1, 0]
    fractions = [100 * as_float(r, "train_fraction") for r in scarcity]
    acc = [as_float(r, "accuracy_mean") for r in scarcity]
    acc_sd = [as_float(r, "accuracy_std") for r in scarcity]
    f1 = [as_float(r, "f1_mean") for r in scarcity]
    f1_sd = [as_float(r, "f1_std") for r in scarcity]
    ece = [100 * as_float(r, "ece15_mean") for r in scarcity]
    ax.errorbar(fractions, acc, yerr=acc_sd, fmt="o-", color=BLUE, lw=1.4,
                ms=3.5, capsize=2.2, label="Accuracy")
    ax.errorbar(fractions, f1, yerr=f1_sd, fmt="s-", color=RED, lw=1.4,
                ms=3.2, capsize=2.2, label="Macro F1")
    ax.plot(fractions, ece, "^-", color=AMBER, lw=1.2, ms=3.2,
            label="ECE x 100")
    ax.set_xticks(fractions, [f"{int(x)}%" for x in fractions])
    ax.set_xlabel("Labeled fraction used for classifier/source head")
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Fewer real labels harm detection and calibration")
    ax.legend(loc="upper left", fontsize=6.1, ncol=3, columnspacing=0.8)
    ax.grid(axis="y", color="#D9DEE5", lw=0.5, alpha=0.8)
    ax.text(0.02, 0.04, "mean +/- SD, 3 seeds\nunlabeled pre-training split held fixed",
            transform=ax.transAxes, fontsize=5.4, color=MUTED, va="bottom")
    panel_label(ax, "c")

    ax = axes[1, 1]
    kvals = [int(r["history_length"]) for r in k_rows]
    ax.plot(kvals, [as_float(r, "accuracy") for r in k_rows], "o-",
            color=BLUE, lw=1.4, ms=3.5, label="Accuracy")
    ax.plot(kvals, [as_float(r, "f1") for r in k_rows], "s-",
            color=RED, lw=1.4, ms=3.2, label="Macro F1")
    ax.plot(kvals, [as_float(r, "rca") for r in k_rows], "^-",
            color=GREEN, lw=1.4, ms=3.4, label="Reference Top-1")
    ax.set_xticks(kvals)
    ax.set_xlabel("Past-only history length K")
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(40, 90)
    ax.set_title("K = 3 is not uniformly optimal")
    ax.set_xlim(0.8, 5.85)
    end_x = kvals[-1] + 0.12
    ax.text(end_x, as_float(k_rows[-1], "accuracy"), "accuracy", color=BLUE,
            fontsize=5.9, va="center")
    ax.text(end_x, as_float(k_rows[-1], "rca"), "reference Top-1", color=GREEN,
            fontsize=5.9, va="center")
    ax.text(end_x, as_float(k_rows[-1], "f1"), "macro F1", color=RED,
            fontsize=5.9, va="center")
    ax.grid(axis="y", color="#D9DEE5", lw=0.5, alpha=0.8)
    ax.text(0.02, 0.05, "single seed; identical short-training budget",
            transform=ax.transAxes, fontsize=5.5, color=MUTED, ha="left")
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.1,
                        wspace=0.34, hspace=0.42)
    save_figure(fig, Path(outdir) / "hgan_trace_training_robustness")
    plt.close(fig)


def diagnostic_figure(expdir, outdir):
    """Combine the held-out confusion matrix and probability reliability audit."""
    expdir = Path(expdir)
    detail_path = expdir / "details" / (
        "fair_baselines__HGAN-Trace__s42__f1__k3__p1f3r3__dw1_ts1_de1.json"
    )
    with detail_path.open("r", encoding="utf-8") as handle:
        detail = json.load(handle)
    cm = np.asarray(json.loads(detail["confusion_matrix"]), dtype=float)
    row_pct = np.divide(cm, cm.sum(axis=1, keepdims=True),
                        out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) != 0) * 100
    labels = ["Normal", "NS", "NM", "PM", "PS", "SS"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15),
                             gridspec_kw={"width_ratios": [1.08, 0.92]})
    ax = axes[0]
    image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            if value == 0:
                text_value = "0"
            else:
                text_value = f"{value}\n{row_pct[i, j]:.1f}%"
            ax.text(j, i, text_value, ha="center", va="center", fontsize=5.5,
                    color="white" if row_pct[i, j] > 55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Held-out IGCPS failures")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row-normalized (%)")
    panel_label(ax, "a")

    ax = axes[1]
    bins = detail["calibration"]["reliability_bins"]
    confidence = np.asarray([b["confidence"] for b in bins])
    accuracy = np.asarray([b["accuracy"] for b in bins])
    counts = np.asarray([b["count"] for b in bins])
    sizes = 16 + 100 * counts / counts.max()
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.0, label="Perfect calibration")
    ax.plot(confidence, accuracy, color=RED, lw=1.2, alpha=0.8)
    ax.scatter(confidence, accuracy, s=sizes, color=RED, edgecolor="white",
               linewidth=0.6, zorder=3, label="Observed bins")
    ax.set_xlim(0.3, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Probabilities are measured, not calibrated")
    ax.grid(color="#D9DEE5", lw=0.5, alpha=0.8)
    ax.text(
        0.04, 0.96,
        f"ECE = {detail['ece15']:.3f}\nBrier = {detail['brier']:.3f}\n"
        f"NLL = {detail['nll']:.3f}\nEntropy-error rho = {detail['entropy_error_spearman']:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "#C9D0D8", "pad": 3.0},
    )
    ax.text(0.98, 0.05, "marker area represents bin count",
            transform=ax.transAxes, fontsize=5.5, color=MUTED, ha="right")
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.09, right=0.97, top=0.9, bottom=0.2, wspace=0.34)
    save_figure(fig, Path(outdir) / "hgan_trace_detection_calibration")
    plt.close(fig)


def scaling_figure(expdir, outdir):
    """Profile the shared-embedding inference path as graph size grows."""
    rows = [
        row for row in read_csv(Path(expdir) / "graph_scaling.csv")
        if row.get("mode") == "classification_plus_shared_root"
    ]
    rows.sort(key=lambda row: int(row["n_nodes"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    ax = axes[0]
    nodes = np.asarray([int(r["n_nodes"]) for r in rows])
    median = np.asarray([as_float(r, "median_ms") for r in rows])
    p95 = np.asarray([as_float(r, "p95_ms") for r in rows])
    ax.plot(nodes, median, "o-", color=BLUE, lw=1.5, ms=3.4,
            label="Detection + source ranking")
    ax.fill_between(nodes, median, p95, color=BLUE, alpha=0.13, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    ax.set_xlabel("Graph nodes")
    ax.set_ylabel("CPU latency (ms, log scale)")
    ax.set_title("One shared-encoder forward pass")
    ax.legend(loc="upper left", fontsize=6.2)
    ax.grid(which="both", color="#D9DEE5", lw=0.5, alpha=0.8)
    panel_label(ax, "a")

    ax = axes[1]
    rss = np.asarray([as_float(r, "peak_rss_mb") for r in rows])
    ax.plot(nodes, rss, "o-", color=GREEN, lw=1.5, ms=3.4)
    lower = max(0.0, float(rss.min()) - 2.0)
    ax.fill_between(nodes, lower, rss, color=GREEN_LIGHT, alpha=0.75)
    for x, y in zip(nodes, rss):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=5.8, color=GREEN)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    ax.set_xlabel("Graph nodes")
    ax.set_ylabel("Peak process RSS (MB)")
    ax.set_ylim(lower, float(rss.max()) + 2.0)
    ax.set_title("Loaded-process memory rises for larger graphs")
    ax.grid(axis="y", color="#D9DEE5", lw=0.5, alpha=0.8)
    ax.text(0.98, 0.05, "synthetic topology audit; 30 repeats per size",
            transform=ax.transAxes, fontsize=5.5, color=MUTED, ha="right")
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.22, wspace=0.3)
    save_figure(fig, Path(outdir) / "hgan_trace_graph_scaling")
    plt.close(fig)


def architecture_figure(outdir):
    """Show the executed K=1 joint HGAN-Trace pipeline."""
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.02, 0.80, 0.18, 0.14, "a", "Causally fused input", BLUE_LIGHT, BLUE),
        (0.265, 0.80, 0.20, 0.14, "b", "Typed current graph", GREEN_LIGHT, GREEN),
        (0.53, 0.80, 0.20, 0.14, "c", "Shared HGAN encoder", AMBER_LIGHT, AMBER),
        (0.795, 0.80, 0.185, 0.14, "d", "Joint task outputs", RED_LIGHT, RED),
    ]
    for x, y, w, h, panel, title, fc, ec in stages:
        rounded_box(ax, (x, y), w, h, fc, ec, linewidth=1.0, radius=0.018)
        label(ax, x + 0.012, y + h - 0.018, panel, size=8, weight="bold", ha="left", va="top")
        label(ax, x + w / 2, y + h / 2, title, size=8.2, weight="bold")

    arrow(ax, (0.205, 0.87), (0.257, 0.87), color=BLUE)
    arrow(ax, (0.47, 0.87), (0.522, 0.87), color=GREEN)
    arrow(ax, (0.735, 0.87), (0.787, 0.87), color=AMBER)

    rounded_box(ax, (0.025, 0.45), 0.17, 0.27, "white", BLUE, radius=0.015)
    label(ax, 0.04, 0.69, "Information layer", size=7.1, weight="bold", ha="left")
    label(ax, 0.04, 0.657, "real source/destination endpoints", size=5.7, color=MUTED, ha="left")
    for i in range(4):
        ax.plot([0.04, 0.18], [0.627 - i * 0.018] * 2, color=BLUE, lw=0.7,
                transform=ax.transAxes, zorder=3)
    label(ax, 0.04, 0.555, "Physical layer", size=7.1, weight="bold", ha="left")
    label(ax, 0.04, 0.523, "sensor and actuator variables", size=5.8, color=MUTED, ha="left")
    label(ax, 0.11, 0.485, "1 s right-edge network windows", size=5.8, weight="bold", color=BLUE)
    label(ax, 0.11, 0.461, "backward/as-of process alignment", size=5.3, color=MUTED)

    rounded_box(ax, (0.27, 0.45), 0.19, 0.27, "white", GREEN, radius=0.015)
    label(ax, 0.365, 0.69, "Current row only (K = 1)", size=7.1, weight="bold")
    rounded_box(ax, (0.29, 0.618), 0.065, 0.043, BLUE_LIGHT, BLUE, radius=0.007)
    label(ax, 0.3225, 0.6395, "endpoint", size=5.6, weight="bold", color=BLUE)
    rounded_box(ax, (0.375, 0.618), 0.065, 0.043, GREEN_LIGHT, GREEN, radius=0.007)
    label(ax, 0.4075, 0.6395, "process", size=5.6, weight="bold", color=GREEN)
    label(ax, 0.365, 0.579, "type and position identifiers", size=5.8, color=MUTED)
    label(ax, 0.365, 0.542, "communication + process-chain edges", size=5.5, color=MUTED)
    label(ax, 0.365, 0.495, "No interpolation, temporal-shift,\nor hand-built cross-layer edges", size=5.6, weight="bold", color=GREEN)

    rounded_box(ax, (0.535, 0.45), 0.19, 0.27, "white", AMBER, radius=0.015)
    label(ax, 0.63, 0.67, "One HGAN-Trace encoder", size=7.1, weight="bold")
    rounded_box(ax, (0.553, 0.610), 0.154, 0.036, BLUE_LIGHT, BLUE, radius=0.008)
    label(ax, 0.63, 0.628, "63 -> 48 typed projection", size=5.45, weight="bold", color=BLUE)
    rounded_box(ax, (0.553, 0.562), 0.154, 0.036, AMBER_LIGHT, AMBER, radius=0.008)
    label(ax, 0.63, 0.580, "type-specific residual adapters", size=5.15, weight="bold", color=AMBER)
    rounded_box(ax, (0.553, 0.514), 0.154, 0.036, AMBER_LIGHT, AMBER, radius=0.008)
    label(ax, 0.63, 0.532, "2 neighborhood residual blocks", size=5.15, weight="bold", color=AMBER)
    rounded_box(ax, (0.553, 0.466), 0.154, 0.036, GREEN_LIGHT, GREEN, radius=0.008)
    label(ax, 0.63, 0.484, "graph token + 2-layer Transformer", size=4.95, weight="bold", color=GREEN)

    rounded_box(ax, (0.80, 0.45), 0.175, 0.27, "white", RED, radius=0.015)
    rounded_box(ax, (0.813, 0.625), 0.149, 0.055, RED_LIGHT, RED, radius=0.008)
    label(ax, 0.8875, 0.6525, "typed graph classifier", size=5.8, weight="bold")
    rounded_box(ax, (0.813, 0.548), 0.149, 0.055, RED_LIGHT, RED, radius=0.008)
    label(ax, 0.8875, 0.5755, "node + graph root scorer", size=5.55, weight="bold")
    rounded_box(ax, (0.813, 0.471), 0.149, 0.055, "white", RED, radius=0.008)
    label(ax, 0.8875, 0.4985, "12-parameter conditional head", size=5.15, weight="bold", color=RED)

    rounded_box(ax, (0.11, 0.18), 0.78, 0.16, GRAY_LIGHT, MUTED, radius=0.015)
    label(ax, 0.13, 0.309, "Joint supervised optimization", size=7.1, weight="bold", ha="left")
    protocol = [
        (0.14, "weighted cross-entropy\nanomaly classes", BLUE),
        (0.405, "0.30 x weighted BCE\nreference nodes", GREEN),
        (0.68, "AdamW; 12 / 15 epochs\nvalidation locked", RED),
    ]
    for x, text, color in protocol:
        rounded_box(ax, (x, 0.215), 0.18, 0.065, "white", color, radius=0.007)
        label(ax, x + 0.09, 0.2475, text, size=5.8, weight="bold", color=color)

    rounded_box(ax, (0.11, 0.055), 0.78, 0.075, "white", MUTED, radius=0.012)
    label(ax, 0.13, 0.092, "Deployment boundary", size=6.5, weight="bold", ha="left")
    label(
        ax, 0.285, 0.092,
        "152,325 parameters on IGCPS; shared embeddings produce class probabilities and root-cause ranks in one pass",
        size=5.4, color=MUTED, ha="left",
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save_figure(fig, Path(outdir) / "hgan_trace_audited_architecture")
    plt.close(fig)


def final_diagnostic_figure(expdir, outdir):
    """Plot the final seed-42 confusion matrix and reliability audit."""
    expdir = Path(expdir)
    with (expdir / "hgan_conditional_calibration_20260825" / "final_results.json").open(
        "r", encoding="utf-8"
    ) as handle:
        final = json.load(handle)
    with (expdir / "final_hgan_audit_20260825" / "audit.json").open(
        "r", encoding="utf-8"
    ) as handle:
        audit = json.load(handle)

    cm = np.asarray(final["confusion_matrix"], dtype=float)
    row_pct = np.divide(
        cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm),
        where=cm.sum(axis=1, keepdims=True) != 0,
    ) * 100
    labels = ["Normal", "NS", "NM", "PM", "PS", "SS"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15),
                             gridspec_kw={"width_ratios": [1.05, 0.95]})

    ax = axes[0]
    image_obj = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{int(cm[i, j])}\n{row_pct[i, j]:.1f}%",
                ha="center", va="center", fontsize=5.7,
                color="white" if row_pct[i, j] > 55 else INK,
            )
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Temporal-tail classification")
    cbar = fig.colorbar(image_obj, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row-normalized (%)")
    panel_label(ax, "a")

    ax = axes[1]
    bins = audit["reliability_bins"]
    confidence = np.asarray([row["confidence"] for row in bins], dtype=float)
    accuracy = np.asarray([row["accuracy"] for row in bins], dtype=float)
    counts = np.asarray([row["count"] for row in bins], dtype=float)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.0, label="Perfect calibration")
    sizes = 22 + 150 * counts / max(float(counts.max()), 1.0)
    ax.scatter(confidence, accuracy, s=sizes, color=BLUE, edgecolor="white",
               linewidth=0.6, alpha=0.9, label="Observed bins")
    ax.plot(confidence, accuracy, color=BLUE, lw=1.0, alpha=0.75)
    ax.set_xlim(0.35, 1.01)
    ax.set_ylim(0.25, 1.01)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Held-out probability reliability")
    ax.grid(color="#D9DEE5", lw=0.5, alpha=0.8)
    ax.text(
        0.04, 0.96,
        f"ECE = {audit['ece15']:.3f}\nBrier = {audit['brier']:.3f}\nNLL = {audit['nll']:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D9DEE5"},
    )
    ax.legend(loc="lower right", fontsize=6.1)
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.20, wspace=0.34)
    save_figure(fig, Path(outdir) / "hgan_trace_final_diagnostic")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--expdir", type=Path)
    parser.add_argument("--architecture", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--final-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.architecture:
        architecture_figure(args.outdir)
    if args.results:
        if args.expdir is None:
            parser.error("--expdir is required with --results")
        training_robustness_figure(args.expdir, args.outdir)
        diagnostic_figure(args.expdir, args.outdir)
        scaling_figure(args.expdir, args.outdir)
    if args.final_diagnostic:
        if args.expdir is None:
            parser.error("--expdir is required with --final-diagnostic")
        final_diagnostic_figure(args.expdir, args.outdir)


if __name__ == "__main__":
    main()
