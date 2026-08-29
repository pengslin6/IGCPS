"""Validation-only label-scarcity audit for the final K=1 HGAN-Trace."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

import graphsage_bias_calibration as calibration
import hgan_conditional_calibration as conditional
import reviewer_experiments as exp
import single_hgan_joint_experiment as joint
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
METRICS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5", "ece15")


def stratified_subset(data: dict, fraction: float, seed: int) -> dict:
    labels = np.asarray(data["y"])
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in sorted(np.unique(labels).tolist()):
        indices = np.where(labels == class_id)[0]
        count = max(2, int(math.ceil(len(indices) * fraction)))
        chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
        selected.extend(chosen.tolist())
    return tr.subset_data(data, np.asarray(sorted(selected), dtype=int))


def classwise_half(data: dict, first: bool) -> dict:
    return temporal.classwise_time_slice(data, 0.0, 0.5) if first else temporal.classwise_time_slice(data, 0.5, 1.0)


def run_once(protocol, candidate, fraction: float, seed: int, outdir: Path) -> dict:
    train = stratified_subset(protocol.train, fraction, seed)
    early_val = classwise_half(protocol.val, True)
    late_val = classwise_half(protocol.val, False)
    builder = conditional.make_builder(protocol.raw, candidate)
    adjacency = joint.static_adjacency(builder, candidate)
    train_cache = joint.build_cache(train, builder)
    early_cache = joint.build_cache(early_val, builder)
    late_cache = joint.build_cache(late_val, builder)

    model, history, _ = joint.train_model(
        candidate,
        train_cache,
        early_cache,
        early_val,
        builder,
        adjacency,
        fixed_epochs=12,
        seed=seed,
    )
    early_probs, _ = conditional.collect(model, early_cache, adjacency)
    late_probs, root_scores = conditional.collect(model, late_cache, adjacency)
    early_labels = early_cache["labels"].numpy()
    late_labels = late_cache["labels"].numpy()
    scaler, classifier = conditional.fit_conditional(
        early_probs, early_labels, np.arange(len(early_labels)), 1.0
    )
    calibrated_early = conditional.apply_conditional(
        early_probs, scaler, classifier
    )
    reference = calibrated_early.argmax(axis=1)
    margin = 0.0
    for delta in np.linspace(0.0, 12.0, 601):
        adjusted = calibration.apply_normal_nm_margin(calibrated_early, float(delta))
        if np.array_equal(adjusted.argmax(axis=1), reference):
            margin = float(delta)
        else:
            break
    calibrated_late = conditional.apply_conditional(late_probs, scaler, classifier)
    calibrated_late = calibration.apply_normal_nm_margin(calibrated_late, margin)
    detection = calibration.metrics(late_labels, calibrated_late)
    predictions = calibrated_late.argmax(axis=1)
    confidence = exp.calibration_metrics(late_labels, predictions, calibrated_late)
    root = tr.compute_strict_traceback_metrics(root_scores, late_val, builder)
    row = {
        "fraction": fraction,
        "seed": seed,
        "labeled_train_samples": int(len(train["y"])),
        "late_validation_samples": int(len(late_val["y"])),
        "accuracy": detection["accuracy"],
        "f1": detection["f1"],
        "auc": detection["auc"],
        "top1": root["rca"],
        "mrr": root["mrr"],
        "ndcg5": root["ndcg"],
        "ece15": confidence["ece15"],
        "fixed_epochs": 12,
        "conditional_C": 1.0,
        "normal_nm_margin": margin,
        "test_used": False,
    }
    pd.DataFrame(history).to_csv(
        outdir / f"history__f{fraction:g}__seed{seed}.csv", index=False
    )
    return row


def summarize(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for fraction, group in frame.groupby("fraction", sort=True):
        row = {
            "fraction": float(fraction),
            "labeled_train_samples_mean": float(group["labeled_train_samples"].mean()),
            "runs": int(len(group)),
        }
        multiplier = float(t.ppf(0.975, len(group) - 1)) if len(group) > 1 else 0.0
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sd
            row[f"{metric}_ci95_low"] = mean - multiplier * sd / math.sqrt(len(values))
            row[f"{metric}_ci95_high"] = mean + multiplier * sd / math.sqrt(len(values))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_causal_v2.csv")
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "single_hgan_joint_causal_rootfix_validation_20260825" / "validation_lock.json",
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "hgan_current_scarcity_20260825"
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    joint.HISTORY_LENGTH = 1
    joint.USE_TEMPORAL = False
    joint.RECENCY_STRENGTH = 1.0
    joint.NORMAL_CLASS_WEIGHT = 1.0
    protocol = exp.load_protocol(args.csv)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    candidate = joint.Candidate(**lock["selected_candidate"])

    result_path = args.outdir / "runs.csv"
    completed = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    rows = completed.to_dict("records")
    complete_keys = {
        (float(row["fraction"]), int(row["seed"])) for row in rows
    }
    total = len(args.fractions) * len(args.seeds)
    position = len(complete_keys)
    for fraction in args.fractions:
        for seed in args.seeds:
            key = (float(fraction), int(seed))
            if key in complete_keys:
                print(f"Reuse fraction={fraction:g}, seed={seed}", flush=True)
                continue
            position += 1
            print(
                f"SCARCITY RUN {position}/{total}: fraction={fraction:g}, seed={seed}",
                flush=True,
            )
            row = run_once(protocol, candidate, fraction, seed, args.outdir)
            rows.append(row)
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(
                f"COMPLETE fraction={fraction:g}, seed={seed}: "
                f"F1={row['f1']:.2f}, Top1={row['top1']:.2f}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values(["fraction", "seed"])
    frame.to_csv(result_path, index=False)
    report = {
        "model": "final K=1 single-encoder HGAN-Trace",
        "candidate": asdict(candidate),
        "evaluation": "late half of the chronological validation partition",
        "test_used": False,
        "conditional_head": "fixed C=1; fitted on early validation only",
        "summary": summarize(frame),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
