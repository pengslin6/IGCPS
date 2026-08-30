"""Cross-fitted linear calibration for the original single HGAN model.

The graph encoder and both task heads are unchanged.  A single 6x6 linear
probability calibration is learned from two development-only rolling folds.
Candidate blend strengths are evaluated in the opposite fold.  The held-out
80--100% tail is accessed only by the explicit final phase after locking.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

import reviewer_experiments as exp
import single_hgan_joint_experiment as joint
import single_hgan_shared_gat_experiment as shared
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "single_hgan_linear_calibration_20260825"
BASE_OUT = ROOT / "single_hgan_joint_20260824"
SEED = 42
FIXED_EPOCHS = 10
BLENDS = (0.0, 0.10, 0.20, 0.35, 0.50)
CALIBRATOR_KINDS = ("logistic", "balanced_logistic")


def probabilities_metrics(labels: np.ndarray,
                          probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    encoded = label_binarize(labels, classes=np.arange(probabilities.shape[1]))
    per_class_auc = [
        roc_auc_score(encoded[:, class_id], probabilities[:, class_id]) * 100
        for class_id in range(probabilities.shape[1])
    ]
    return {
        "accuracy": accuracy_score(labels, predictions) * 100,
        "f1": f1_score(labels, predictions, average="macro", zero_division=0) * 100,
        "auc": float(np.mean(per_class_auc)),
        "per_class_auc": per_class_auc,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def collect_outputs(model, cache: dict, adjacency: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    dataset = TensorDataset(cache["features"], cache["labels"])
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    logits = []
    roots = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            output = model(features, adjacency)
            logits.append(output["anomaly_logits"].cpu().numpy())
            roots.append(torch.sigmoid(output["root_logits"]).cpu().numpy())
    return np.concatenate(logits), np.concatenate(roots)


def load_model(checkpoint_path: Path, builder, cache: dict):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidate = joint.Candidate(**payload["candidate"])
    model = joint.DATGTJoint(
        input_dim=cache["features"].shape[-1],
        n_nodes=builder.n_nodes,
        n_net=builder.n_net,
        n_classes=int(cache["labels"].max().item() + 1),
        candidate=candidate,
    )
    model.load_state_dict(payload["state_dict"])
    return model, candidate


def fit_fold_model(raw: dict, train_end: float, validation_end: float,
                   fold_name: str) -> dict:
    candidate = joint.Candidate(
        name="typed_full_robust",
        preserve_network_features=True,
        model_dim=48,
        n_heads=4,
        transformer_layers=2,
        dropout=0.25,
        input_noise=0.02,
        root_weight=0.30,
        normal_root_weight=0.03,
        learning_rate=7e-4,
        weight_decay=5e-4,
        epochs=18,
    )
    train, validation = shared.rolling_fold(raw, train_end, validation_end)
    builder = joint.make_builder(raw, candidate)
    adjacency = joint.static_adjacency(builder)
    train_cache = joint.build_cache(train, builder)
    validation_cache = joint.build_cache(validation, builder)
    checkpoint_path = OUT / f"base__{fold_name}.pt"
    if checkpoint_path.exists():
        model, _ = load_model(checkpoint_path, builder, train_cache)
        history = []
    else:
        model, history, _ = joint.train_model(
            candidate,
            train_cache,
            validation_cache,
            validation,
            builder,
            adjacency,
            fixed_epochs=FIXED_EPOCHS,
            seed=SEED,
        )
        torch.save({
            "state_dict": model.state_dict(),
            "candidate": asdict(candidate),
            "fixed_epochs": FIXED_EPOCHS,
        }, checkpoint_path)
        pd.DataFrame(history).to_csv(
            OUT / f"base_history__{fold_name}.csv", index=False
        )
    logits, _ = collect_outputs(model, validation_cache, adjacency)
    labels = validation_cache["labels"].numpy()
    np.savez_compressed(
        OUT / f"oof__{fold_name}.npz", logits=logits, labels=labels
    )
    return {"name": fold_name, "logits": logits, "labels": labels}


def fit_calibrator(kind: str, logits: np.ndarray,
                   labels: np.ndarray) -> LogisticRegression:
    class_weight = "balanced" if kind == "balanced_logistic" else None
    model = LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        max_iter=2000,
        solver="lbfgs",
        random_state=SEED,
    )
    model.fit(logits, labels)
    return model


def calibrated_probabilities(logits: np.ndarray,
                             calibrator: LogisticRegression | None,
                             blend: float) -> np.ndarray:
    if calibrator is None or blend <= 0:
        return softmax(logits, axis=1)
    calibrated = np.log(np.clip(calibrator.predict_proba(logits), 1e-12, 1.0))
    return softmax((1.0 - blend) * logits + blend * calibrated, axis=1)


def run_selection() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = exp.load_protocol(ROOT / "sr_com_new.csv")
    folds = [
        fit_fold_model(protocol.raw, 0.4, 0.6, "40_60"),
        fit_fold_model(protocol.raw, 0.6, 0.8, "60_80"),
    ]
    rows = []
    for kind in CALIBRATOR_KINDS:
        for source_index, target_index in ((0, 1), (1, 0)):
            source = folds[source_index]
            target = folds[target_index]
            calibrator = fit_calibrator(kind, source["logits"], source["labels"])
            for blend in BLENDS:
                metrics = probabilities_metrics(
                    target["labels"],
                    calibrated_probabilities(target["logits"], calibrator, blend),
                )
                rows.append({
                    "kind": kind,
                    "blend": float(blend),
                    "fit_fold": source["name"],
                    "evaluation_fold": target["name"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "crossfit_candidates.csv", index=False)
    summary = frame.groupby(["kind", "blend"], as_index=False).agg(
        worst_f1=("f1", "min"),
        mean_f1=("f1", "mean"),
        worst_accuracy=("accuracy", "min"),
        mean_accuracy=("accuracy", "mean"),
        worst_auc=("auc", "min"),
        mean_auc=("auc", "mean"),
    )
    identity = summary[summary["blend"] == 0.0].iloc[0]
    eligible = summary[
        (summary["worst_f1"] >= float(identity["worst_f1"]) - 0.5)
        & (summary["mean_f1"] >= float(identity["mean_f1"]) - 0.5)
        & (summary["mean_accuracy"] >= float(identity["mean_accuracy"]) - 0.5)
    ].copy()
    selected = eligible.sort_values(
        ["mean_auc", "worst_auc", "mean_f1", "blend"],
        ascending=[False, False, False, True],
    ).iloc[0]
    combined_logits = np.concatenate([fold["logits"] for fold in folds])
    combined_labels = np.concatenate([fold["labels"] for fold in folds])
    calibrator = fit_calibrator(
        str(selected["kind"]), combined_logits, combined_labels
    )
    joblib.dump(calibrator, OUT / "locked_calibrator.joblib")
    payload = {
        "selection_data": [
            "0--40% train / 40--60% OOF outputs",
            "0--60% train / 60--80% OOF outputs",
        ],
        "held_out_80_100_used_for_selection": False,
        "base_model": "single shared DA-TGT",
        "fixed_base_epochs": FIXED_EPOCHS,
        "candidate_kinds": list(CALIBRATOR_KINDS),
        "candidate_blends": list(BLENDS),
        "selection_rule": (
            "maximize cross-fold mean AUC subject to worst/mean F1 and mean "
            "accuracy being within 0.5 point of identity"
        ),
        "identity": {key: float(identity[key]) for key in summary.columns
                     if key not in {"kind"}},
        "selected": {
            key: (str(selected[key]) if key == "kind" else float(selected[key]))
            for key in summary.columns
        },
        "single_model": True,
        "ensemble": False,
        "teacher": False,
        "calibration_parameters": 42,
    }
    (OUT / "validation_lock.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    summary.to_csv(OUT / "crossfit_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def run_final() -> None:
    lock = json.loads((OUT / "validation_lock.json").read_text(encoding="utf-8"))
    calibrator = joblib.load(OUT / "locked_calibrator.joblib")
    blend = float(lock["selected"]["blend"])
    protocol = exp.load_protocol(ROOT / "sr_com_new.csv")
    development, test = temporal.build_development_test(protocol.raw, 0.8)
    checkpoint = torch.load(
        BASE_OUT / "HGAN-Trace-Joint__seed42.pt",
        map_location="cpu",
        weights_only=False,
    )
    candidate = joint.Candidate(**checkpoint["candidate"])
    builder = joint.make_builder(protocol.raw, candidate)
    adjacency = joint.static_adjacency(builder)
    development_cache = joint.build_cache(development, builder)
    test_cache = joint.build_cache(test, builder)
    model, _ = load_model(
        BASE_OUT / "HGAN-Trace-Joint__seed42.pt", builder, development_cache
    )
    logits, root_scores = collect_outputs(model, test_cache, adjacency)
    probabilities = calibrated_probabilities(logits, calibrator, blend)
    result = probabilities_metrics(test_cache["labels"].numpy(), probabilities)
    root = tr.compute_strict_traceback_metrics(root_scores, test, builder)
    result.update({
        "top1": root["rca"],
        "mrr": root["mrr"],
        "ndcg5": root["ndcg"],
        "mean_reference_rank": root["apd"],
    })
    baseline = pd.read_csv(
        ROOT / "temporal_robust_20260823" / "final_80pct_comparison.csv"
    )
    metric_names = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")
    maxima = {metric: float(baseline[metric].max()) for metric in metric_names}
    strict_first = {
        metric: bool(float(result[metric]) > maxima[metric])
        for metric in metric_names
    }
    result.update({
        "model": "DA-TGT-LinearCal",
        "base_checkpoint": str(BASE_OUT / "HGAN-Trace-Joint__seed42.pt"),
        "calibrator_kind": lock["selected"]["kind"],
        "calibrator_blend": blend,
        "selection_used_test": False,
        "single_model": True,
        "ensemble": False,
        "teacher": False,
        "baseline_maxima": maxima,
        "strict_first": strict_first,
        "all_six_strict_first": bool(all(strict_first.values())),
    })
    (OUT / "final_result__seed42.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("selection", "final"), required=True)
    args = parser.parse_args()
    if args.phase == "selection":
        run_selection()
    else:
        run_final()


if __name__ == "__main__":
    main()
