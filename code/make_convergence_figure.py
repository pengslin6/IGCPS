from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_loss(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"epoch", "loss"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.loc[:, ["epoch", "loss"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot retained final-refit joint training-loss trajectories."
    )
    parser.add_argument("--igcps", type=Path, required=True)
    parser.add_argument("--te", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    series = [
        ("IGCPS", load_loss(args.igcps), "#1f77b4", "o"),
        ("TE-CUP-SEC", load_loss(args.te), "#d95f02", "s"),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 2.55), constrained_layout=True)
    for label, frame, color, marker in series:
        ax.plot(
            frame["epoch"],
            frame["loss"],
            label=label,
            color=color,
            marker=marker,
            markersize=3.2,
            linewidth=1.45,
        )

    ax.set_xlabel("Joint-training epoch")
    ax.set_ylabel("Final-refit training loss")
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#d9d9d9", linewidth=0.55, alpha=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
