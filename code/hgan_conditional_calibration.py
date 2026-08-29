"""Validation-locked conditional calibration for the original HGAN-Trace.

The graph encoder and both task heads stay unchanged.  A 3-class linear layer
only redistributes probability mass among Normal, NM, and PM, the confusable
classes identified on the chronological validation split.  The layer can be
folded into the detection head and does not add another encoder or model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import graphsage_bias_calibration as calibration
import reviewer_experiments as exp
import single_hgan_joint_experiment as joint
import single_hgan_linear_calibration_validation as linear
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
GROUP = [0, 2, 3]
METRICS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")


def make_builder(raw: dict, candidate: joint.Candidate):
    return exp.make_builder(
        raw,
        history_length=1,
        temporal=False,
        preserve_network_features=candidate.preserve_network_features,
    )


def load_model(checkpoint_path: Path, raw: dict, data: dict):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidate = joint.Candidate(**payload["candidate"])
    builder = make_builder(raw, candidate)
    cache = joint.build_cache(data, builder)
    model = joint.SingleHGANJoint(
        input_dim=cache["features"].shape[-1],
        n_nodes=builder.n_nodes,
        n_net=builder.n_net,
        n_classes=int(raw["n_classes"]),
        candidate=candidate,
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {"detection_pool.weight", "detection_pool.bias"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    adjacency = joint.static_adjacency(builder, candidate)
    return model, candidate, builder, cache, adjacency


def collect(model, cache: dict, adjacency: torch.Tensor):
    logits, roots = linear.collect_outputs(model, cache, adjacency)
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True), roots


def group_labels(labels: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(GROUP), labels)


def fit_conditional(probs: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                    c_value: float):
    mask = np.isin(labels[indices], GROUP)
    selected = indices[mask]
    return calibration.fit_layer(
        calibration.conditional_features(probs[selected]),
        group_labels(labels[selected]),
        c_value,
    )


def apply_conditional(probs: np.ndarray, scaler, classifier) -> np.ndarray:
    return calibration.apply_conditional_layer(probs, scaler, classifier)


def run_validation(args) -> None:
    protocol = exp.load_protocol(args.csv)
    checkpoint = args.model_dir / "validation__typed_full_robust.pt"
    model, candidate, _, cache, adjacency = load_model(
        checkpoint, protocol.raw, protocol.val
    )
    labels = cache["labels"].numpy()
    raw_probs, _ = collect(model, cache, adjacency)
    early, late = calibration.classwise_halves(labels)

    rows = []
    for c_value in calibration.C_VALUES:
        scaler, classifier = fit_conditional(
            raw_probs, labels, early, c_value
        )
        result = calibration.metrics(
            labels[late],
            apply_conditional(raw_probs[late], scaler, classifier),
        )
        rows.append({"C": float(c_value), **result})
        print(
            f"C={c_value:g}: late Acc/F1/AUC="
            f"{result['accuracy']:.2f}/{result['f1']:.2f}/{result['auc']:.2f}",
            flush=True,
        )
    selected = max(
        rows,
        key=lambda row: (row["accuracy"], row["f1"], row["auc"], -row["C"]),
    )
    scaler, classifier = fit_conditional(
        raw_probs, labels, np.arange(len(labels)), selected["C"]
    )
    calibrated = apply_conditional(raw_probs, scaler, classifier)
    reference_predictions = calibrated.argmax(axis=1)
    maximum_safe_margin = 0.0
    for delta in np.linspace(0.0, 12.0, 601):
        adjusted = calibration.apply_normal_nm_margin(calibrated, float(delta))
        if np.array_equal(adjusted.argmax(axis=1), reference_predictions):
            maximum_safe_margin = float(delta)
        else:
            break
    calibrated = calibration.apply_normal_nm_margin(
        calibrated, maximum_safe_margin
    )
    lock = {
        "selection_data": "class-wise early/late split within 60--80% validation",
        "test_used_for_selection": False,
        "base_model": "original single-encoder HGAN-Trace",
        "base_candidate": candidate.name,
        "history_length": 1,
        "temporal_history_enabled": False,
        "calibration": "conditional Normal/NM/PM linear detection-head layer",
        "fixed_group": GROUP,
        "selected_C": float(selected["C"]),
        "normal_nm_margin_rule": (
            "maximum Normal-favoring log-odds margin that changes no "
            "full-validation hard decision"
        ),
        "normal_nm_log_odds_margin": maximum_safe_margin,
        "rolling_validation": selected,
        "full_validation_uncalibrated": calibration.metrics(labels, raw_probs),
        "full_validation_calibrated": calibration.metrics(labels, calibrated),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
        "single_model": True,
        "ensemble": False,
        "distillation": False,
        "second_encoder": False,
        "additional_parameters": 12,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "validation_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {key: value for key, value in row.items() if not isinstance(value, list)}
        for row in rows
    ]).to_csv(args.outdir / "validation_candidates.csv", index=False)
    print(json.dumps(lock, indent=2), flush=True)


def locked_probabilities(probs: np.ndarray, lock: dict) -> np.ndarray:
    mean = np.asarray(lock["scaler_mean"], dtype=np.float64)
    scale = np.asarray(lock["scaler_scale"], dtype=np.float64)
    coef = np.asarray(lock["coef"], dtype=np.float64)
    intercept = np.asarray(lock["intercept"], dtype=np.float64)
    features = (calibration.conditional_features(probs) - mean) / scale
    logits = features @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    conditional = np.exp(logits)
    conditional /= conditional.sum(axis=1, keepdims=True)
    adjusted = np.array(probs, copy=True)
    mass = probs[:, GROUP].sum(axis=1, keepdims=True)
    adjusted[:, GROUP] = mass * conditional
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return calibration.apply_normal_nm_margin(
        adjusted, float(lock.get("normal_nm_log_odds_margin", 0.0))
    )


def run_final(args) -> None:
    lock = json.loads(
        (args.outdir / "validation_lock.json").read_text(encoding="utf-8")
    )
    if lock.get("test_used_for_selection") is not False:
        raise ValueError("Calibration is not validation-locked")
    protocol = exp.load_protocol(args.csv)
    _, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    checkpoint = args.model_dir / "HGAN-Trace-Joint__seed42.pt"
    model, candidate, builder, cache, adjacency = load_model(
        checkpoint, protocol.raw, test
    )
    raw_probs, root_scores = collect(model, cache, adjacency)
    probs = locked_probabilities(raw_probs, lock)
    labels = cache["labels"].numpy()
    result = calibration.metrics(labels, probs)
    root = tr.compute_strict_traceback_metrics(root_scores, test, builder)
    result.update({
        "top1": root["rca"],
        "mrr": root["mrr"],
        "ndcg5": root["ndcg"],
        "mean_reference_rank": root["apd"],
    })
    baseline = pd.read_csv(args.baseline_dir / "final_80pct_comparison.csv")
    maxima = {metric: float(baseline[metric].max()) for metric in METRICS}
    strict_first = {
        metric: bool(float(result[metric]) > maxima[metric]) for metric in METRICS
    }
    result.update({
        "model": "HGAN-Trace",
        "architecture": "original shared HGAN encoder and root head with a folded conditional detection-head layer",
        "base_candidate": candidate.name,
        "selection_used_test": False,
        "single_model": True,
        "ensemble": False,
        "distillation": False,
        "second_encoder": False,
        "additional_parameters": 12,
        "parameters": sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if name not in {"detection_pool.weight", "detection_pool.bias"}
        ) + 12,
        "baseline_maxima": maxima,
        "strict_first": strict_first,
        "all_six_strict_first": bool(all(strict_first.values())),
        "ranks": {metric: 1 if strict_first[metric] else None for metric in METRICS},
    })
    (args.outdir / "final_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_causal_v2.csv")
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "single_hgan_joint_causal_rootfix_validation_20260825",
    )
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=ROOT / "causal_fusion_v2_baselines_fullfeatures_20260825",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "hgan_conditional_calibration_20260825",
    )
    args = parser.parse_args()
    if args.stage == "validation":
        run_validation(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
