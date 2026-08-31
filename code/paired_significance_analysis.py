"""Paired seed-level comparisons against the strongest baseline per metric."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel, wilcoxon


METRICS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")


def parse_mapping(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        metric, model = value.split("=", 1)
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        mapping[metric] = model
    missing = [metric for metric in METRICS if metric not in mapping]
    if missing:
        raise ValueError(f"Missing baseline mappings: {missing}")
    return mapping


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_row(metric: str, model_name: str, main: pd.DataFrame,
               baseline: pd.DataFrame) -> dict:
    merged = main[["seed", metric]].merge(
        baseline[baseline["model"].astype(str) == model_name][["seed", metric]],
        on="seed",
        suffixes=("_main", "_baseline"),
        validate="one_to_one",
    ).sort_values("seed")
    if len(merged) < 2:
        raise ValueError(f"At least two paired runs are required for {metric}")
    main_values = merged[f"{metric}_main"].to_numpy(dtype=float)
    baseline_values = merged[f"{metric}_baseline"].to_numpy(dtype=float)
    differences = main_values - baseline_values
    n = len(differences)
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    half = float(t.ppf(0.975, n - 1) * sd_difference / math.sqrt(n))
    t_result = ttest_rel(main_values, baseline_values)
    if np.allclose(differences, 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        w_result = wilcoxon(
            main_values,
            baseline_values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        wilcoxon_statistic = float(w_result.statistic)
        wilcoxon_p = float(w_result.pvalue)
    return {
        "metric": metric,
        "baseline": model_name,
        "pairs": n,
        "main_mean": float(np.mean(main_values)),
        "main_std": float(np.std(main_values, ddof=1)),
        "baseline_mean": float(np.mean(baseline_values)),
        "baseline_std": float(np.std(baseline_values, ddof=1)),
        "mean_difference": mean_difference,
        "difference_ci95_low": mean_difference - half,
        "difference_ci95_high": mean_difference + half,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p,
        "all_main_higher": bool(np.all(differences > 0)),
        "seeds": merged["seed"].astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--main-runs", type=Path, required=True)
    parser.add_argument("--baseline-runs", type=Path, required=True)
    parser.add_argument("--mapping", nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    main_runs = pd.read_csv(args.main_runs)
    baseline_runs = pd.read_csv(args.baseline_runs)
    mapping = parse_mapping(args.mapping)
    rows = [
        paired_row(metric, mapping[metric], main_runs, baseline_runs)
        for metric in METRICS
    ]
    adjusted = holm_adjust([row["paired_t_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["paired_t_p_holm"] = float(value)
        row["paired_t_significant_0_05_holm"] = bool(value < 0.05)
        row["dataset"] = args.dataset
    frame = pd.DataFrame(rows)
    frame.to_csv(args.outdir / "paired_statistics.csv", index=False)
    payload = {
        "dataset": args.dataset,
        "primary_test": "two-sided paired Student t-test with Holm correction across six metrics",
        "sensitivity_test": "two-sided Wilcoxon signed-rank test",
        "mapping": mapping,
        "rows": rows,
    }
    (args.outdir / "paired_statistics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
