"""Five-seed audit of the locked DA-TGT on causal fusion data."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t

import graphsage_bias_calibration as calibration
import hgan_conditional_calibration as conditional
import reviewer_experiments as exp
import single_hgan_joint_experiment as joint
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
METRICS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")


def train_base(candidate, train_data, validation_data, builder, seed: int,
               epochs: int):
    adjacency = joint.static_adjacency(builder, candidate)
    train_cache = joint.build_cache(train_data, builder)
    validation_cache = joint.build_cache(validation_data, builder)
    model, history, _ = joint.train_model(
        candidate,
        train_cache,
        validation_cache,
        validation_data,
        builder,
        adjacency,
        fixed_epochs=epochs,
        seed=seed,
    )
    return model, history, validation_cache, adjacency


def fit_lock(model, cache: dict, adjacency: torch.Tensor) -> dict:
    labels = cache["labels"].numpy()
    probs, _ = conditional.collect(model, cache, adjacency)
    early, late = calibration.classwise_halves(labels)
    candidates = []
    for c_value in calibration.C_VALUES:
        scaler, classifier = conditional.fit_conditional(
            probs, labels, early, c_value
        )
        result = calibration.metrics(
            labels[late],
            conditional.apply_conditional(probs[late], scaler, classifier),
        )
        candidates.append({"C": float(c_value), **result})
    selected = max(
        candidates,
        key=lambda row: (row["accuracy"], row["f1"], row["auc"], -row["C"]),
    )
    scaler, classifier = conditional.fit_conditional(
        probs, labels, np.arange(len(labels)), selected["C"]
    )
    calibrated = conditional.apply_conditional(probs, scaler, classifier)
    reference = calibrated.argmax(axis=1)
    margin = 0.0
    for delta in np.linspace(0.0, 12.0, 601):
        adjusted = calibration.apply_normal_nm_margin(calibrated, float(delta))
        if np.array_equal(adjusted.argmax(axis=1), reference):
            margin = float(delta)
        else:
            break
    calibrated = calibration.apply_normal_nm_margin(calibrated, margin)
    return {
        "selection_data": "class-wise early/late split within 60--80% validation",
        "test_used_for_selection": False,
        "selected_C": float(selected["C"]),
        "normal_nm_log_odds_margin": margin,
        "rolling_validation": selected,
        "full_validation": calibration.metrics(labels, calibrated),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
    }


def summarize(frame: pd.DataFrame) -> dict:
    output = {}
    n = len(frame)
    multiplier = float(t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        half = multiplier * sd / math.sqrt(n) if n > 1 else 0.0
        output[metric] = {
            "mean": mean,
            "std": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return output


def robust_validation_epoch(history_path: Path) -> int:
    frame = pd.read_csv(history_path)
    frame = frame[pd.to_numeric(frame["epoch"], errors="coerce").notna()].copy()
    for column in (
        "epoch", "full_accuracy", "full_f1", "full_auc", "full_top1",
        "late_f1", "late_auc", "late_top1",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["robust_f1"] = frame[["full_f1", "late_f1"]].min(axis=1)
    frame["robust_auc"] = frame[["full_auc", "late_auc"]].min(axis=1)
    frame["robust_top1"] = frame[["full_top1", "late_top1"]].min(axis=1)
    selected = frame.sort_values(
        ["robust_f1", "robust_auc", "robust_top1", "full_accuracy", "epoch"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    return int(selected["epoch"])


def run_seed(seed: int, protocol, development: dict, test: dict,
             candidate, epochs: int, outdir: Path,
             use_calibration: bool = True) -> dict:
    builder = conditional.make_builder(protocol.raw, candidate)
    started = time.perf_counter()
    lock = None
    if use_calibration:
        validation_model, validation_history, validation_cache, adjacency = train_base(
            candidate, protocol.train, protocol.val, builder, seed, epochs
        )
        lock = fit_lock(validation_model, validation_cache, adjacency)
        pd.DataFrame(validation_history).to_csv(
            outdir / f"validation_history__seed{seed}.csv", index=False
        )
        (outdir / f"calibration_lock__seed{seed}.json").write_text(
            json.dumps(lock, indent=2), encoding="utf-8"
        )

    final_model, final_history, _, adjacency = train_base(
        candidate, development, development, builder, seed, epochs
    )
    test_cache = joint.build_cache(test, builder)
    raw_probs, root_scores = conditional.collect(final_model, test_cache, adjacency)
    probs = (
        conditional.locked_probabilities(raw_probs, lock)
        if lock is not None else raw_probs
    )
    labels = test_cache["labels"].numpy()
    result = calibration.metrics(labels, probs)
    root = tr.compute_strict_traceback_metrics(root_scores, test, builder)
    result.update({
        "top1": root["rca"],
        "mrr": root["mrr"],
        "ndcg5": root["ndcg"],
        "mean_reference_rank": root["apd"],
        "seed": seed,
        "fixed_epochs": epochs,
        "calibration_enabled": bool(lock is not None),
        "normal_nm_log_odds_margin": (
            lock["normal_nm_log_odds_margin"] if lock is not None else 0.0
        ),
        "selected_C": lock["selected_C"] if lock is not None else None,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_used_test": False,
    })
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "candidate": asdict(candidate),
            "fixed_epochs": epochs,
            "calibration_lock": lock,
        },
        outdir / f"HGAN-Trace__seed{seed}.pt",
    )
    pd.DataFrame(final_history).to_csv(
        outdir / f"final_history__seed{seed}.csv", index=False
    )
    (outdir / f"result__seed{seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_causal_v2.csv")
    parser.add_argument(
        "--lock", type=Path,
        default=ROOT / "single_hgan_joint_causal_rootfix_validation_20260825"
        / "validation_lock.json",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "hgan_causal_multiseed_20260825",
    )
    parser.add_argument(
        "--epoch-source-dir", type=Path, default=None,
        help="Optional validation histories used for robust per-seed epoch locking",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--train-cap-per-class", type=int, default=0)
    parser.add_argument("--validation-cap-per-class", type=int, default=0)
    parser.add_argument("--test-cap-per-class", type=int, default=0)
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Evaluate the raw DA-TGT probabilities, as in the TE-CUP-SEC protocol.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    joint.HISTORY_LENGTH = 1
    joint.USE_TEMPORAL = False
    source_lock = json.loads(args.lock.read_text(encoding="utf-8"))
    joint.RECENCY_STRENGTH = float(source_lock.get("recency_strength", 1.0))
    joint.NORMAL_CLASS_WEIGHT = float(source_lock.get("normal_class_weight", 1.0))
    protocol = exp.cap_protocol(
        exp.load_protocol(args.csv),
        max(0, args.train_cap_per_class),
        max(0, args.validation_cap_per_class),
        max(0, args.test_cap_per_class),
        seed=42,
    )
    development, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    development = tr.cap_split_per_class(
        development,
        max(0, args.train_cap_per_class),
        seed=42,
        split_name="Development",
    )
    test = tr.cap_split_per_class(
        test,
        max(0, args.test_cap_per_class),
        seed=44,
        split_name="Test",
    )
    candidate = joint.Candidate(**source_lock["selected_candidate"])
    epochs = int(source_lock["selected_epoch"])

    rows = []
    epochs_by_seed = {}
    for position, seed in enumerate(args.seeds, start=1):
        seed_epochs = epochs
        if args.epoch_source_dir is not None:
            seed_epochs = robust_validation_epoch(
                args.epoch_source_dir / f"validation_history__seed{seed}.csv"
            )
        epochs_by_seed[str(seed)] = int(seed_epochs)
        result_path = args.outdir / f"result__seed{seed}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"Seed {seed} already complete; reusing result", flush=True)
        else:
            print(
                f"Seed {seed} ({position}/{len(args.seeds)}): validation calibration",
                flush=True,
            )
            result = run_seed(
                seed, protocol, development, test, candidate, seed_epochs,
                args.outdir, use_calibration=not args.no_calibration,
            )
        rows.append(result)
        print(
            f"Seed {seed} complete: Acc/F1/AUC="
            f"{result['accuracy']:.2f}/{result['f1']:.2f}/{result['auc']:.2f}; "
            f"Top1/MRR/NDCG5={result['top1']:.2f}/{result['mrr']:.2f}/"
            f"{result['ndcg5']:.2f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(args.outdir / "runs.csv", index=False)

    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(args.outdir / "runs.csv", index=False)
    summary = {
        "seeds": list(args.seeds),
        "candidate": asdict(candidate),
        "fixed_epochs_by_seed": epochs_by_seed,
        "epoch_selection": (
            "maximize minimum full/late validation F1, then minimum AUC and "
            "Top-1, then full-validation accuracy"
            if args.epoch_source_dir is not None
            else "fixed from the seed-42 validation lock"
        ),
        "history_length": 1,
        "temporal_history_enabled": False,
        "calibration_parameters": 0 if args.no_calibration else 12,
        "train_cap_per_class": max(0, args.train_cap_per_class),
        "validation_cap_per_class": max(0, args.validation_cap_per_class),
        "test_cap_per_class": max(0, args.test_cap_per_class),
        "recency_strength": joint.RECENCY_STRENGTH,
        "selection_used_test": False,
        "summary": summarize(frame),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
