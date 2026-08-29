"""Create the implementation-aligned root-evidence figure for the revision."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "final_hgan_audit_20260825" / "root_examples.json"
OUTPUT = Path(r"C:\Users\pcsys\Desktop\ESWA\hgan_trace_root_evidence")
EXPLAIN_OUTPUT = Path(r"C:\Users\pcsys\Desktop\ESWA\hgan_trace_explainability")
TRACE_OUTPUT = Path(r"C:\Users\pcsys\Desktop\ESWA\hgan_trace_traceback_examples")


SHORT_NAMES = {
    "Process:phy_Switch status of high pressure switch": "High-pressure\nswitch",
    "Process:phy_Switch status of sub-high pressure switch": "Sub-high-pressure\nswitch",
    "Process:phy_Switch status of medium pressure skid heater": "Medium-pressure\nheater switch",
    "Process:phy_Inlet temperature of medium pressure skid heater": "Heater inlet\ntemperature",
    "Process:phy_Outlet temperature of medium pressure skid heater": "Heater outlet\ntemperature",
    "Process:phy_Outlet temperature of medium pressure skid sensor": "Sensor outlet\ntemperature",
    "Process:phy_Outlet pressure of medium pressure skid sensor": "Sensor outlet\npressure",
    "Endpoint:10.0.6.118": "Endpoint\n10.0.6.118",
    "Endpoint:10.0.6.63": "Endpoint\n10.0.6.63",
    "Endpoint:10.0.6.30": "Endpoint\n10.0.6.30",
}


def short_name(node: str) -> str:
    if node in SHORT_NAMES:
        return SHORT_NAMES[node]
    return "\n".join(textwrap.wrap(node.split(":", 1)[-1], width=18)[:2])


def draw_node(ax, x: float, y: float, item: dict, width: float = 0.22) -> None:
    is_endpoint = item["node"].startswith("Endpoint:")
    face = "#DCEAF7" if is_endpoint else "#FBE6D5"
    edge = "#2E7D32" if item["is_reference"] else "#68727D"
    box = FancyBboxPatch(
        (x - width / 2, y - 0.145),
        width,
        0.290,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.25 if item["is_reference"] else 0.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + 0.045,
        short_name(item["node"]),
        ha="center",
        va="center",
        fontsize=5.4,
        linespacing=0.95,
        color="#17212B",
    )
    ax.text(
        x,
        y - 0.098,
        f'{item["score"]:.3f}',
        ha="center",
        va="center",
        fontsize=5.5,
        color="#4B5563",
    )


def main() -> None:
    examples = json.loads(SOURCE.read_text(encoding="utf-8"))
    classes = [entry["class"] for entry in examples]
    scores = np.array(
        [[ranked["score"] for ranked in entry["top5"]] for entry in examples],
        dtype=float,
    )
    references = np.array(
        [[ranked["is_reference"] for ranked in entry["top5"]] for entry in examples],
        dtype=bool,
    )
    types = np.array(
        [
            ["E" if ranked["node"].startswith("Endpoint:") else "P" for ranked in entry["top5"]]
            for entry in examples
        ]
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.20, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.82, 1.68],
        left=0.07,
        right=0.985,
        top=0.92,
        bottom=0.12,
        wspace=0.22,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list(
        "root_score", ["#F4F6F8", "#BFD5EA", "#2C6FA8"]
    )
    image = ax_a.imshow(scores, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax_a.set_xticks(range(5), [f"Rank {i}" for i in range(1, 6)], rotation=35, ha="right")
    ax_a.set_yticks(range(len(classes)), classes)
    ax_a.set_xlabel("Ranked node position")
    ax_a.set_ylabel("Anomaly class")
    ax_a.tick_params(length=0)
    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            value = scores[row, col]
            marker = "*" if references[row, col] else ""
            color = "white" if value >= 0.62 else "#17212B"
            ax_a.text(
                col,
                row,
                f"{types[row, col]}\n{value:.3f}{marker}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=color,
                fontweight="bold" if references[row, col] else "normal",
            )
    cbar = fig.colorbar(image, ax=ax_a, fraction=0.055, pad=0.035)
    cbar.set_label("Root score", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    ax_a.set_title("a  Final-model root scores", loc="left", fontsize=8, fontweight="bold", pad=7)
    ax_a.text(
        0.0,
        -0.20,
        "* reference node; E endpoint; P process variable",
        transform=ax_a.transAxes,
        fontsize=5.8,
        color="#4B5563",
    )

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.65, 4.65)
    ax_b.axis("off")
    ax_b.set_title(
        "b  Representative investigation order",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=7,
    )
    xs = [0.20, 0.52, 0.84]
    ys = list(reversed(range(5)))
    for entry, y in zip(examples, ys):
        ax_b.text(
            0.01,
            y,
            entry["class"],
            ha="left",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
        top3 = entry["top5"][:3]
        for item, x in zip(top3, xs):
            draw_node(ax_b, x, y, item)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            arrow = FancyArrowPatch(
                (x0 + 0.115, y),
                (x1 - 0.115, y),
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=0.70,
                color="#C43C39",
                alpha=0.88,
                shrinkA=0,
                shrinkB=0,
            )
            ax_b.add_patch(arrow)

    legend_items = [
        Patch(facecolor="#DCEAF7", edgecolor="#68727D", label="Endpoint"),
        Patch(facecolor="#FBE6D5", edgecolor="#68727D", label="Process variable"),
        Patch(facecolor="white", edgecolor="#2E7D32", linewidth=1.25, label="Reference node"),
    ]
    ax_b.legend(
        handles=legend_items,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.54, -0.16),
        frameon=False,
        fontsize=5.8,
        handlelength=1.3,
        columnspacing=1.0,
    )
    ax_b.text(
        0.01,
        -0.54,
        "Thin arrows connect ranks for visual reading only; they are not learned causal edges.",
        fontsize=5.8,
        color="#4B5563",
    )

    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    explain_fig, explain_ax = plt.subplots(figsize=(3.45, 4.05))
    explain_image = explain_ax.imshow(
        scores, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto"
    )
    explain_ax.set_xticks(
        range(5), [f"Rank {i}" for i in range(1, 6)], rotation=35, ha="right"
    )
    explain_ax.set_yticks(range(len(classes)), classes)
    explain_ax.set_xlabel("Ranked node position")
    explain_ax.set_ylabel("Anomaly class")
    explain_ax.tick_params(length=0)
    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            value = scores[row, col]
            marker = "*" if references[row, col] else ""
            color = "white" if value >= 0.62 else "#17212B"
            explain_ax.text(
                col,
                row,
                f"{types[row, col]}\n{value:.3f}{marker}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=color,
                fontweight="bold" if references[row, col] else "normal",
            )
    explain_cbar = explain_fig.colorbar(
        explain_image, ax=explain_ax, fraction=0.055, pad=0.035
    )
    explain_cbar.set_label("Root score", fontsize=6.5)
    explain_cbar.ax.tick_params(labelsize=5.8, length=2)
    explain_ax.set_title(
        "Final-model root-score explanation",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=7,
    )
    explain_ax.text(
        0.0,
        -0.20,
        "* reference node; E endpoint; P process variable",
        transform=explain_ax.transAxes,
        fontsize=5.8,
        color="#4B5563",
    )
    explain_fig.savefig(EXPLAIN_OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    explain_fig.savefig(EXPLAIN_OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    explain_fig.savefig(
        EXPLAIN_OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight"
    )
    plt.close(explain_fig)

    trace_fig, trace_ax = plt.subplots(figsize=(7.20, 3.35))
    trace_ax.set_xlim(0, 1)
    trace_ax.set_ylim(-0.48, 3.40)
    trace_ax.axis("off")
    trace_ax.set_title(
        "Representative root-cause traceback results",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=7,
    )
    trace_xs = [0.18, 0.51, 0.84]
    trace_ys = [2.88, 2.16, 1.44, 0.72, 0.0]
    for entry, y in zip(examples, trace_ys):
        trace_ax.text(
            0.01,
            y,
            entry["class"],
            ha="left",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
        top3 = entry["top5"][:3]
        for item, x in zip(top3, trace_xs):
            draw_node(trace_ax, x, y, item, width=0.23)
        for x0, x1 in zip(trace_xs[:-1], trace_xs[1:]):
            trace_ax.add_patch(
                FancyArrowPatch(
                    (x0 + 0.120, y),
                    (x1 - 0.120, y),
                    arrowstyle="-|>",
                    mutation_scale=6.5,
                    linewidth=0.70,
                    color="#C43C39",
                    alpha=0.88,
                    shrinkA=0,
                    shrinkB=0,
                )
            )
    trace_ax.legend(
        handles=legend_items,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.54, -0.16),
        frameon=False,
        fontsize=5.8,
        handlelength=1.3,
        columnspacing=1.0,
    )
    trace_ax.text(
        0.01,
        -0.38,
        "Arrows connect descending ranks for visual inspection; they are not learned causal paths.",
        fontsize=5.8,
        color="#4B5563",
    )
    trace_fig.savefig(TRACE_OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    trace_fig.savefig(TRACE_OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    trace_fig.savefig(TRACE_OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(trace_fig)


if __name__ == "__main__":
    main()
