"""Validation-locked temporal-drift experiments for DA-TGT.

The search phases use only the original 60% training and 20% validation
partitions. After the recency strength and epoch count are locked, the final
phases refit on the first 80% of each class and evaluate the last 20% once.
Strong baselines receive the same temporal weighting and source head.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

import reviewer_experiments as exp
import trace as tr


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "temporal_robust_20260823"
BUILDER_HISTORY_LENGTH = 3
BUILDER_TEMPORAL = True
TRAIN_CAP_PER_CLASS = 0
VALIDATION_CAP_PER_CLASS = 0
TEST_CAP_PER_CLASS = 0
RECENCY_STRENGTHS = [0.0, 0.5, 1.0, 2.0]
STRONG_BASELINES = [
    "GCN-AE", "GAT-AE", "VGAE", "GraphSAGE-AE", "IIoT-GNN",
    "EE-GCN", "STGaAN", "STCI", "DT-GNN",
]
HGAN_CONFIG = {
    "hgan_variant": "typed_dsgc",
    "use_cross_attention": True,
    "use_type_adapters": True,
    "use_dynamic_edge_weights": True,
    "encoder_lr_scale": 0.5,
    "class_weight_power": 0.5,
    "apply_class_boosts": False,
    "oversample": True,
    "label_smoothing": 0.0,
}


def write_row(path: Path, row: dict, key: str = "run_id") -> None:
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame[frame[key].astype(str) != str(row[key])]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(path, index=False)


def classwise_time_slice(data: dict, start: float, end: float) -> dict:
    selected = []
    y = np.asarray(data["y"])
    for cls_id in sorted(np.unique(y).tolist()):
        cls_idx = np.where(y == cls_id)[0]
        lo = int(np.floor(len(cls_idx) * start))
        hi = int(np.ceil(len(cls_idx) * end))
        hi = max(lo + 1, min(hi, len(cls_idx)))
        selected.extend(cls_idx[lo:hi].tolist())
    return tr.subset_data(data, np.asarray(sorted(selected), dtype=int))


def build_development_test(raw: dict, dev_ratio: float = 0.8) -> tuple[dict, dict]:
    dev_indices = []
    test_indices = []
    y = np.asarray(raw["y"])
    for cls_id in sorted(np.unique(y).tolist()):
        cls_idx = np.where(y == cls_id)[0]
        split = max(1, min(int(np.floor(len(cls_idx) * dev_ratio)), len(cls_idx) - 1))
        dev_indices.extend(cls_idx[:split].tolist())
        test_indices.extend(cls_idx[split:].tolist())

    development = tr.subset_data(raw, np.asarray(sorted(dev_indices), dtype=int))
    test = tr.subset_data(raw, np.asarray(sorted(test_indices), dtype=int))
    scaler_phy = StandardScaler().fit(development["X_phy"])
    scaler_net = StandardScaler().fit(development["X_net_feat"])
    scaler_control = None
    if development.get("X_control") is not None and development["X_control"].shape[1] > 0:
        scaler_control = StandardScaler().fit(development["X_control"])
    for split in (development, test):
        split["X_phy"] = scaler_phy.transform(split["X_phy"]).astype(np.float32)
        split["X_net_feat"] = scaler_net.transform(split["X_net_feat"]).astype(np.float32)
        if scaler_control is not None:
            split["X_control"] = scaler_control.transform(split["X_control"]).astype(np.float32)
    development["_val_data"] = development
    development["_test_data"] = test
    return development, test


def build_hgan(builder, n_classes: int):
    return exp.build_model(
        "DA-TGT", builder, n_classes,
        use_cross_attention=HGAN_CONFIG["use_cross_attention"],
        use_type_adapters=HGAN_CONFIG["use_type_adapters"],
        use_dynamic_edge_weights=HGAN_CONFIG["use_dynamic_edge_weights"],
        revised_hgan=True,
        hgan_variant=HGAN_CONFIG["hgan_variant"],
    )


def fit_model(model, train: dict, selection: dict, builder, device,
              model_name: str, epochs: int, recency: float,
              restore_best: bool = True):
    model = tr.pretrain_self_supervised(
        model, train, builder, device,
        epochs=1, lr=0.002,
        use_dynamic_edges=False, use_cross_layer=False,
        use_phy_chain=True, use_net_edges=True,
        verbose=True, model_name=model_name,
    )
    model = tr.finetune_supervised(
        model, train, builder, device,
        epochs=epochs, lr=0.002, freeze_encoder=False,
        use_dynamic_edges=False, use_cross_layer=False,
        use_phy_chain=True, use_net_edges=True,
        verbose=True, model_name=model_name,
        encoder_lr_scale=HGAN_CONFIG["encoder_lr_scale"],
        class_weight_power=HGAN_CONFIG["class_weight_power"],
        apply_class_boosts=HGAN_CONFIG["apply_class_boosts"],
        oversample=HGAN_CONFIG["oversample"],
        label_smoothing=HGAN_CONFIG["label_smoothing"],
        temporal_recency_strength=recency,
        selection_data=selection,
        restore_best=restore_best,
        early_stopping_patience=max(epochs + 1, 15),
    )
    return model


def validation_metrics(model, head, protocol, builder, device) -> dict:
    early = classwise_time_slice(protocol.val, 0.0, 0.5)
    late = classwise_time_slice(protocol.val, 0.5, 1.0)
    full_cls, _, _ = exp.evaluate_classification(model, protocol.val, builder, device)
    early_cls, _, _ = exp.evaluate_classification(model, early, builder, device)
    late_cls, _, _ = exp.evaluate_classification(model, late, builder, device)
    full_root = exp.evaluate_shared_root(model, head, protocol.val, builder, device)
    late_root = exp.evaluate_shared_root(model, head, late, builder, device)
    return {
        "accuracy": full_cls["accuracy"],
        "precision": full_cls["precision"],
        "recall": full_cls["recall"],
        "f1": full_cls["f1"],
        "balanced_accuracy": full_cls["balanced_accuracy"],
        "mcc": full_cls["mcc"],
        "auc": full_cls["auc"],
        "pr_auc": full_cls["pr_auc"],
        "early_f1": early_cls["f1"],
        "early_balanced_accuracy": early_cls["balanced_accuracy"],
        "late_f1": late_cls["f1"],
        "late_balanced_accuracy": late_cls["balanced_accuracy"],
        "top1": full_root["rca"],
        "mrr": full_root["mrr"],
        "ndcg5": full_root["ndcg"],
        "late_top1": late_root["rca"],
        "late_mrr": late_root["mrr"],
    }


def run_hgan_candidate(protocol, outdir: Path, recency: float, stage: str,
                       epochs: int, force: bool = False) -> dict:
    run_id = f"{stage}__HGAN__recency{recency:g}__s42__p1f{epochs}"
    result_path = outdir / "recency_validation.csv"
    if result_path.exists() and not force:
        frame = pd.read_csv(result_path)
        matched = frame[frame["run_id"].astype(str) == run_id]
        if len(matched):
            row = matched.iloc[-1].to_dict()
            print(f"[validation cache] {run_id}: F1={row['f1']:.2f}, late F1={row['late_f1']:.2f}")
            return row

    tr.set_seed(42)
    device = torch.device("cpu")
    builder = exp.make_builder(
        protocol.raw,
        history_length=BUILDER_HISTORY_LENGTH,
        temporal=BUILDER_TEMPORAL,
        preserve_network_features=True,
    )
    model = build_hgan(builder, protocol.raw["n_classes"]).to(device)
    head = exp.SharedRootHead(latent_dim=32).to(device)
    log_dir = outdir / "logs"
    checkpoint_dir = outdir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with (log_dir / f"{run_id}.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(exp.Tee(sys.stdout, stream)):
            model = fit_model(
                model, protocol.train, protocol.val, builder, device,
                "DA-TGT", epochs, recency, restore_best=True,
            )
            root_epochs = 1 if stage == "screen" else 3
            root_history = exp.train_shared_root_head(
                model, head, protocol.train, builder, device,
                epochs=root_epochs, lr=0.002,
            )

    metrics = validation_metrics(model, head, protocol, builder, device)
    checkpoint = checkpoint_dir / f"{run_id}.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "root_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
        "recency_strength": recency,
        "best_epoch": int(model.finetune_best_epoch),
        "history": model.finetune_history,
        "stage": stage,
    }, checkpoint)
    row = {
        "run_id": run_id,
        "stage": stage,
        "model": "DA-TGT",
        "recency_strength": recency,
        "best_epoch": int(model.finetune_best_epoch),
        **metrics,
        "root_final_loss": root_history[-1],
        "train_seconds": time.perf_counter() - start_time,
        "checkpoint": str(checkpoint),
    }
    write_row(result_path, row)
    print(
        f"VALIDATION {run_id}: F1={row['f1']:.2f}, late F1={row['late_f1']:.2f}, "
        f"Top1={row['top1']:.2f}, late Top1={row['late_top1']:.2f}"
    )
    return row


def robust_rank(row: dict) -> tuple:
    return (
        min(float(row["f1"]), float(row["late_f1"])),
        0.5 * (float(row["f1"]) + float(row["late_f1"])),
        min(float(row["top1"]), float(row["late_top1"])),
        float(row["mcc"]),
    )


def recency_search(protocol, outdir: Path, force: bool = False) -> None:
    screens = [
        run_hgan_candidate(protocol, outdir, strength, "screen", 3, force)
        for strength in RECENCY_STRENGTHS
    ]
    top_strengths = [
        float(row["recency_strength"])
        for row in sorted(screens, key=robust_rank, reverse=True)[:2]
    ]
    confirms = [
        run_hgan_candidate(protocol, outdir, strength, "confirm", 8, force)
        for strength in top_strengths
    ]
    selected = sorted(confirms, key=robust_rank, reverse=True)[0]
    lock = {
        "selection_data": "original 60--80% chronological validation only",
        "test_used_for_selection": False,
        "architecture": HGAN_CONFIG,
        "candidate_recency_strengths": RECENCY_STRENGTHS,
        "selection_rule": [
            "maximize minimum of full-validation and late-validation macro F1",
            "then their mean",
            "then minimum of full/late Top-1",
            "then MCC",
        ],
        "selected_run_id": selected["run_id"],
        "selected_recency_strength": float(selected["recency_strength"]),
        "selected_epoch": int(selected["best_epoch"]),
        "selected_validation_metrics": {
            key: float(selected[key]) for key in [
                "accuracy", "precision", "recall", "f1", "balanced_accuracy",
                "mcc", "auc", "pr_auc", "early_f1", "late_f1",
                "top1", "mrr", "ndcg5", "late_top1", "late_mrr",
            ]
        },
    }
    (outdir / "temporal_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print("\nTEMPORAL CONFIGURATION LOCKED")
    print(json.dumps(lock, indent=2))


def run_baseline_validation(protocol, outdir: Path, model_name: str,
                            force: bool = False) -> dict:
    lock = json.loads((outdir / "temporal_lock.json").read_text(encoding="utf-8"))
    recency = float(lock["selected_recency_strength"])
    temporal_tag = f"h{BUILDER_HISTORY_LENGTH}" if BUILDER_TEMPORAL else "current"
    run_id = (
        f"validation__{model_name}__recency{recency:g}__{temporal_tag}__s42__p1f8"
    )
    result_path = outdir / "baseline_validation.csv"
    if result_path.exists() and not force:
        frame = pd.read_csv(result_path)
        matched = frame[frame["run_id"].astype(str) == run_id]
        if len(matched):
            row = matched.iloc[-1].to_dict()
            print(f"[validation cache] {run_id}: F1={row['f1']:.2f}, late F1={row['late_f1']:.2f}")
            return row

    tr.set_seed(42)
    device = torch.device("cpu")
    builder = exp.make_builder(
        protocol.raw,
        history_length=BUILDER_HISTORY_LENGTH,
        temporal=BUILDER_TEMPORAL,
        preserve_network_features=True,
    )
    model = exp.build_model(model_name, builder, protocol.raw["n_classes"]).to(device)
    head = exp.SharedRootHead(latent_dim=32).to(device)
    log_dir = outdir / "baseline_logs"
    checkpoint_dir = outdir / "baseline_checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with (log_dir / f"{run_id}.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(exp.Tee(sys.stdout, stream)):
            model = fit_model(
                model, protocol.train, protocol.val, builder, device,
                model_name, 8, recency, restore_best=True,
            )
            root_history = exp.train_shared_root_head(
                model, head, protocol.train, builder, device, epochs=3, lr=0.002,
            )
    metrics = validation_metrics(model, head, protocol, builder, device)
    checkpoint = checkpoint_dir / f"{run_id}.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "root_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
        "recency_strength": recency,
        "best_epoch": int(model.finetune_best_epoch),
        "history": model.finetune_history,
        "model_name": model_name,
    }, checkpoint)
    row = {
        "run_id": run_id,
        "model": model_name,
        "recency_strength": recency,
        "best_epoch": int(model.finetune_best_epoch),
        **metrics,
        "root_final_loss": root_history[-1],
        "train_seconds": time.perf_counter() - start_time,
        "checkpoint": str(checkpoint),
    }
    write_row(result_path, row)
    print(
        f"VALIDATION {model_name}: F1={row['f1']:.2f}, late F1={row['late_f1']:.2f}, "
        f"Top1={row['top1']:.2f}"
    )
    return row


def per_class_metrics(y, pred, label_names) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(len(label_names)), zero_division=0,
    )
    return [
        {
            "class": str(label_names[i]),
            "precision": float(precision[i] * 100),
            "recall": float(recall[i] * 100),
            "f1": float(f1[i] * 100),
            "support": int(support[i]),
        }
        for i in range(len(label_names))
    ]


def final_refit(protocol, outdir: Path, model_name: str) -> dict:
    safe_name = model_name.replace("/", "_")
    marker = outdir / f"final_80pct_refit__{safe_name}.json"
    if marker.exists():
        result = json.loads(marker.read_text(encoding="utf-8"))
        print(f"[final cache] {model_name}: F1={result['f1']:.2f}, Top1={result['top1']:.2f}")
        return result

    lock = json.loads((outdir / "temporal_lock.json").read_text(encoding="utf-8"))
    recency = float(lock["selected_recency_strength"])
    if model_name == "DA-TGT":
        epochs = int(lock["selected_epoch"])
    else:
        frame = pd.read_csv(outdir / "baseline_validation.csv")
        selected = frame[frame["model"].astype(str) == model_name]
        if selected.empty:
            raise RuntimeError(f"Run baseline-validation for {model_name} first.")
        epochs = int(selected.iloc[-1]["best_epoch"])

    development, test = build_development_test(protocol.raw, dev_ratio=0.8)
    development = tr.cap_split_per_class(
        development, TRAIN_CAP_PER_CLASS, seed=42, split_name="Development"
    )
    test = tr.cap_split_per_class(
        test, TEST_CAP_PER_CLASS, seed=44, split_name="Test"
    )
    tr.set_seed(42)
    device = torch.device("cpu")
    builder = exp.make_builder(
        protocol.raw,
        history_length=BUILDER_HISTORY_LENGTH,
        temporal=BUILDER_TEMPORAL,
        preserve_network_features=True,
    )
    if model_name == "DA-TGT":
        model = build_hgan(builder, protocol.raw["n_classes"]).to(device)
    else:
        model = exp.build_model(model_name, builder, protocol.raw["n_classes"]).to(device)
    head = exp.SharedRootHead(latent_dim=32).to(device)
    log_dir = outdir / "final_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with (log_dir / f"{safe_name}.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(exp.Tee(sys.stdout, stream)):
            model = fit_model(
                model, development, development, builder, device,
                model_name, epochs, recency, restore_best=False,
            )
            root_history = exp.train_shared_root_head(
                model, head, development, builder, device, epochs=3, lr=0.002,
            )

    cls, pred, probs = exp.evaluate_classification(model, test, builder, device)
    root = exp.evaluate_shared_root(model, head, test, builder, device)
    calibration = exp.calibration_metrics(test["y"], pred, probs)
    result = {
        "run_id": f"final80__{model_name}__recency{recency:g}__s42__p1f{epochs}r3",
        "model": model_name,
        "split": "class-wise chronological first 80% refit / last 20% evaluation",
        "selection_used_test": False,
        "recency_strength": recency,
        "fixed_finetune_epochs": epochs,
        "accuracy": cls["accuracy"],
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "balanced_accuracy": cls["balanced_accuracy"],
        "mcc": cls["mcc"],
        "auc": cls["auc"],
        "pr_auc": cls["pr_auc"],
        "top1": root["rca"],
        "mrr": root["mrr"],
        "ndcg5": root["ndcg"],
        "mean_reference_rank": root["apd"],
        "ece15": calibration["ece15"],
        "brier": calibration["brier"],
        "nll": calibration["nll"],
        "entropy_error_spearman": calibration["entropy_error_spearman"],
        "classification_ms": cls["inference_time_ms"],
        "root_ms": root["inference_time_ms"],
        "train_seconds": time.perf_counter() - start_time,
        "root_final_loss": root_history[-1],
        "confusion_matrix": cls["confusion_matrix"].tolist(),
        "per_class": per_class_metrics(test["y"], pred, test["label_names"]),
        "reliability_bins": calibration["reliability_bins"],
    }
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    scalar = {
        key: value for key, value in result.items()
        if key not in {"confusion_matrix", "per_class", "reliability_bins"}
    }
    write_row(outdir / "final_80pct_comparison.csv", scalar)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "root_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
        "model_name": model_name,
        "recency_strength": recency,
        "fixed_finetune_epochs": epochs,
    }, outdir / f"final_80pct__{safe_name}.pt")
    print("\nFINAL 80% REFIT RESULT")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    global BUILDER_HISTORY_LENGTH, BUILDER_TEMPORAL
    global TRAIN_CAP_PER_CLASS, VALIDATION_CAP_PER_CLASS, TEST_CAP_PER_CLASS
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_new.csv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--phase",
        choices=["search", "baseline-validation", "final-hgan", "final-baselines"],
        required=True,
    )
    parser.add_argument("--models", nargs="+", default=STRONG_BASELINES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--history-length", type=int, default=BUILDER_HISTORY_LENGTH)
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--train-cap-per-class", type=int, default=0)
    parser.add_argument("--validation-cap-per-class", type=int, default=0)
    parser.add_argument("--test-cap-per-class", type=int, default=0)
    args = parser.parse_args()
    if args.history_length < 1:
        parser.error("--history-length must be at least 1")
    BUILDER_HISTORY_LENGTH = args.history_length
    BUILDER_TEMPORAL = not args.current_only
    TRAIN_CAP_PER_CLASS = max(0, args.train_cap_per_class)
    VALIDATION_CAP_PER_CLASS = max(0, args.validation_cap_per_class)
    TEST_CAP_PER_CLASS = max(0, args.test_cap_per_class)
    args.outdir.mkdir(parents=True, exist_ok=True)
    protocol = exp.cap_protocol(
        exp.load_protocol(args.csv),
        TRAIN_CAP_PER_CLASS,
        VALIDATION_CAP_PER_CLASS,
        TEST_CAP_PER_CLASS,
        seed=42,
    )

    if args.phase == "search":
        recency_search(protocol, args.outdir, args.force)
    elif args.phase == "baseline-validation":
        for model_name in args.models:
            run_baseline_validation(protocol, args.outdir, model_name, args.force)
    elif args.phase == "final-hgan":
        final_refit(protocol, args.outdir, "DA-TGT")
    else:
        for model_name in args.models:
            final_refit(protocol, args.outdir, model_name)


if __name__ == "__main__":
    main()
