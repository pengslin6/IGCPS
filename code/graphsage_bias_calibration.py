"""Validation-locked linear calibration for the stable shared encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize

import reviewer_experiments as exp
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
CONDITIONAL_GROUP = [0, 2, 3]


def build_model(builder, n_classes: int):
    return exp.build_model("GraphSAGE-AE", builder, n_classes)


def probabilities(model, data: dict, builder) -> np.ndarray:
    _, _, probs = exp.evaluate_classification(
        model, data, builder, torch.device("cpu")
    )
    return np.asarray(probs, dtype=np.float64)


def log_features(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, 1e-8, 1.0))


def fit_layer(features: np.ndarray, labels: np.ndarray, c_value: float):
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=3000,
        random_state=42,
    ).fit(scaler.transform(features), labels)
    return scaler, classifier


def apply_layer(features: np.ndarray, scaler, classifier) -> np.ndarray:
    return classifier.predict_proba(scaler.transform(features))


def metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    prediction = probs.argmax(axis=1)
    encoded = label_binarize(labels, classes=range(probs.shape[1]))
    per_class_auc = [
        roc_auc_score(encoded[:, class_id], probs[:, class_id]) * 100
        for class_id in range(probs.shape[1])
    ]
    return {
        "accuracy": accuracy_score(labels, prediction) * 100,
        "f1": f1_score(
            labels, prediction, average="macro", zero_division=0
        ) * 100,
        "auc": float(np.mean(per_class_auc)),
        "per_class_auc": per_class_auc,
        "confusion_matrix": confusion_matrix(labels, prediction).tolist(),
    }


def classwise_halves(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    early = []
    late = []
    for class_id in np.unique(labels):
        indices = np.where(labels == class_id)[0]
        split = max(1, len(indices) // 2)
        early.extend(indices[:split].tolist())
        late.extend(indices[split:].tolist())
    return np.asarray(sorted(early)), np.asarray(sorted(late))


def run_validation(args) -> None:
    protocol = exp.load_protocol(args.csv)
    builder = exp.make_builder(
        protocol.raw,
        history_length=1,
        temporal=False,
        preserve_network_features=True,
    )
    model = build_model(builder, protocol.raw["n_classes"])
    checkpoint = torch.load(
        args.baseline_dir / "baseline_checkpoints"
        / "validation__GraphSAGE-AE__recency1__current__s42__p1f8.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    labels = np.asarray(protocol.val["y"], dtype=np.int64)
    features = log_features(probabilities(model, protocol.val, builder))
    early, late = classwise_halves(labels)

    candidates = []
    for c_value in C_VALUES:
        scaler, classifier = fit_layer(
            features[early], labels[early], c_value
        )
        result = metrics(
            labels[late], apply_layer(features[late], scaler, classifier)
        )
        candidates.append({"C": c_value, **result})
        print(
            f"C={c_value:g}: late Acc/F1/AUC="
            f"{result['accuracy']:.2f}/{result['f1']:.2f}/{result['auc']:.2f}"
        )
    selected = max(
        candidates,
        key=lambda row: (row["accuracy"], row["f1"], row["auc"], -row["C"]),
    )
    scaler, classifier = fit_layer(features, labels, selected["C"])
    calibrated = metrics(labels, apply_layer(features, scaler, classifier))
    uncalibrated = metrics(labels, np.exp(features))
    lock = {
        "selection_data": "class-wise early/late split within 60--80% validation",
        "test_used_for_selection": False,
        "history_length": 1,
        "temporal_history_enabled": False,
        "selected_C": float(selected["C"]),
        "rolling_validation": selected,
        "full_validation_uncalibrated": uncalibrated,
        "full_validation_calibrated": calibrated,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "calibration_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {
            key: value for key, value in row.items()
            if not isinstance(value, (list, dict))
        }
        for row in candidates
    ]).to_csv(args.outdir / "calibration_candidates.csv", index=False)
    print(json.dumps(lock, indent=2))


def locked_probabilities(features: np.ndarray, lock: dict) -> np.ndarray:
    mean = np.asarray(lock["scaler_mean"], dtype=np.float64)
    scale = np.asarray(lock["scaler_scale"], dtype=np.float64)
    coef = np.asarray(lock["coef"], dtype=np.float64)
    intercept = np.asarray(lock["intercept"], dtype=np.float64)
    transformed = (features - mean) / scale
    logits = transformed @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def conditional_features(probs: np.ndarray) -> np.ndarray:
    group = probs[:, CONDITIONAL_GROUP]
    group = group / group.sum(axis=1, keepdims=True).clip(min=1e-8)
    return log_features(group)


def apply_conditional_layer(probs: np.ndarray, scaler, classifier) -> np.ndarray:
    calibrated = np.array(probs, copy=True)
    mass = probs[:, CONDITIONAL_GROUP].sum(axis=1, keepdims=True)
    conditional = classifier.predict_proba(
        scaler.transform(conditional_features(probs))
    )
    calibrated[:, CONDITIONAL_GROUP] = mass * conditional
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def apply_normal_nm_margin(probs: np.ndarray, delta: float) -> np.ndarray:
    adjusted = np.array(probs, copy=True)
    pair_mass = probs[:, 0] + probs[:, 2]
    log_odds = np.log(np.clip(probs[:, 0], 1e-12, 1.0)) - np.log(
        np.clip(probs[:, 2], 1e-12, 1.0)
    )
    normal_share = 1.0 / (1.0 + np.exp(-(log_odds + float(delta))))
    adjusted[:, 0] = pair_mass * normal_share
    adjusted[:, 2] = pair_mass * (1.0 - normal_share)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def locked_conditional_probabilities(probs: np.ndarray, lock: dict) -> np.ndarray:
    mean = np.asarray(lock["scaler_mean"], dtype=np.float64)
    scale = np.asarray(lock["scaler_scale"], dtype=np.float64)
    coef = np.asarray(lock["coef"], dtype=np.float64)
    intercept = np.asarray(lock["intercept"], dtype=np.float64)
    transformed = (conditional_features(probs) - mean) / scale
    logits = transformed @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    conditional = np.exp(logits)
    conditional /= conditional.sum(axis=1, keepdims=True)
    calibrated = np.array(probs, copy=True)
    mass = probs[:, CONDITIONAL_GROUP].sum(axis=1, keepdims=True)
    calibrated[:, CONDITIONAL_GROUP] = mass * conditional
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    return apply_normal_nm_margin(
        calibrated, float(lock.get("normal_nm_log_odds_margin", 0.0))
    )


def run_group_validation(args) -> None:
    protocol = exp.load_protocol(args.csv)
    builder = exp.make_builder(
        protocol.raw, history_length=1, temporal=False,
        preserve_network_features=True,
    )
    model = build_model(builder, protocol.raw["n_classes"])
    checkpoint = torch.load(
        args.baseline_dir / "baseline_checkpoints"
        / "validation__GraphSAGE-AE__recency1__current__s42__p1f8.pt",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    labels = np.asarray(protocol.val["y"], dtype=np.int64)
    raw_probs = probabilities(model, protocol.val, builder)
    group_mask = np.isin(labels, CONDITIONAL_GROUP)
    group_labels = np.searchsorted(
        np.asarray(CONDITIONAL_GROUP), labels[group_mask]
    )
    group_features = conditional_features(raw_probs[group_mask])
    early, late = classwise_halves(group_labels)
    candidates = []
    for c_value in C_VALUES:
        scaler, classifier = fit_layer(
            group_features[early], group_labels[early], c_value
        )
        result = metrics(
            group_labels[late],
            classifier.predict_proba(scaler.transform(group_features[late])),
        )
        candidates.append({"C": c_value, **result})
        print(
            f"group C={c_value:g}: late Acc/F1/AUC="
            f"{result['accuracy']:.2f}/{result['f1']:.2f}/{result['auc']:.2f}"
        )
    selected = max(
        candidates,
        key=lambda row: (row["accuracy"], row["f1"], row["auc"], -row["C"]),
    )
    fit_indices = (
        late if args.mode == "group_recent"
        else np.arange(len(group_labels), dtype=int)
    )
    scaler, classifier = fit_layer(
        group_features[fit_indices], group_labels[fit_indices], selected["C"]
    )
    calibrated_probs = apply_conditional_layer(
        raw_probs, scaler, classifier
    )
    normal_nm_margin = 0.0
    maximum_safe_margin = 0.0
    if args.mode in {"group_margin", "group_margin_max"}:
        reference_predictions = calibrated_probs.argmax(axis=1)
        for delta in np.linspace(0.0, 12.0, 601):
            adjusted = apply_normal_nm_margin(calibrated_probs, float(delta))
            if np.array_equal(adjusted.argmax(axis=1), reference_predictions):
                maximum_safe_margin = float(delta)
            else:
                break
        normal_nm_margin = maximum_safe_margin * (
            1.0 if args.mode == "group_margin_max" else 0.5
        )
        calibrated_probs = apply_normal_nm_margin(
            calibrated_probs, normal_nm_margin
        )
    lock = {
        "selection_data": "class-wise early/late split within 60--80% validation",
        "test_used_for_selection": False,
        "calibration": "conditional Normal/NM/PM linear layer",
        "fit_window": (
            "late half of validation" if args.mode == "group_recent"
            else "full validation"
        ),
        "fixed_group": CONDITIONAL_GROUP,
        "selected_C": float(selected["C"]),
        "maximum_safe_normal_nm_margin": maximum_safe_margin,
        "normal_nm_log_odds_margin": normal_nm_margin,
        "rolling_group_validation": selected,
        "full_validation_uncalibrated": metrics(labels, raw_probs),
        "full_validation_calibrated": metrics(labels, calibrated_probs),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    lock_name = {
        "group_recent": "recent_group_calibration_lock.json",
        "group_margin": "margin_group_calibration_lock.json",
        "group_margin_max": "max_margin_group_calibration_lock.json",
    }.get(args.mode, "group_calibration_lock.json")
    (args.outdir / lock_name).write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(json.dumps(lock, indent=2))


def run_group_final(args) -> None:
    lock_name = {
        "group_recent": "recent_group_calibration_lock.json",
        "group_margin": "margin_group_calibration_lock.json",
        "group_margin_max": "max_margin_group_calibration_lock.json",
    }.get(args.mode, "group_calibration_lock.json")
    lock = json.loads((args.outdir / lock_name).read_text(encoding="utf-8"))
    if lock.get("test_used_for_selection") is not False:
        raise ValueError("Conditional calibration is not validation-locked")
    protocol = exp.load_protocol(args.csv)
    _, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    builder = exp.make_builder(
        protocol.raw, history_length=1, temporal=False,
        preserve_network_features=True,
    )
    model = build_model(builder, protocol.raw["n_classes"])
    checkpoint = torch.load(
        args.baseline_dir / "final_80pct__GraphSAGE-AE.pt",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    labels = np.asarray(test["y"], dtype=np.int64)
    raw_probs = probabilities(model, test, builder)
    calibrated_probs = locked_conditional_probabilities(raw_probs, lock)
    result = {
        "model": "DA-TGT conditionally calibrated detector",
        "selection_used_test": False,
        "uncalibrated": metrics(labels, raw_probs),
        "calibrated": metrics(labels, calibrated_probs),
    }
    result_name = {
        "group_recent": "recent_group_calibrated_final.json",
        "group_margin": "margin_group_calibrated_final.json",
        "group_margin_max": "max_margin_group_calibrated_final.json",
    }.get(args.mode, "group_calibrated_final.json")
    (args.outdir / result_name).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def run_final(args) -> None:
    lock = json.loads(
        (args.outdir / "calibration_lock.json").read_text(encoding="utf-8")
    )
    if lock.get("test_used_for_selection") is not False:
        raise ValueError("Calibration is not validation-locked")
    protocol = exp.load_protocol(args.csv)
    _, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    builder = exp.make_builder(
        protocol.raw,
        history_length=1,
        temporal=False,
        preserve_network_features=True,
    )
    model = build_model(builder, protocol.raw["n_classes"])
    checkpoint = torch.load(
        args.baseline_dir / "final_80pct__GraphSAGE-AE.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    labels = np.asarray(test["y"], dtype=np.int64)
    raw_probs = probabilities(model, test, builder)
    calibrated_probs = locked_probabilities(log_features(raw_probs), lock)
    result = {
        "model": "DA-TGT calibrated detector",
        "selection_used_test": False,
        "uncalibrated": metrics(labels, raw_probs),
        "calibrated": metrics(labels, calibrated_probs),
    }
    (args.outdir / "calibrated_final.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validation", "final"], required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "full", "group", "group_recent", "group_margin",
            "group_margin_max",
        ],
        default="group",
    )
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_causal_v2.csv")
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=ROOT / "causal_fusion_v2_baselines_fullfeatures_20260825",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "graphsage_calibration_20260825",
    )
    args = parser.parse_args()
    if args.stage == "validation" and args.mode.startswith("group"):
        run_group_validation(args)
    elif args.stage == "final" and args.mode.startswith("group"):
        run_group_final(args)
    elif args.stage == "validation":
        run_validation(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
