"""Compare the validation-locked HGAN configuration with strong baselines.

The held-out test split is never evaluated here.  Every baseline uses the same
training partition, temporal features, optimization schedule, class balancing,
and supervised root-localization head as the locked HGAN candidate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

import reviewer_experiments as exp
import trace as tr


ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH = ROOT / "hgan_framework_search_20260823_v3"
DEFAULT_MODELS = ["GAT-AE", "VGAE", "GraphSAGE-AE"]


def train_and_validate(protocol, outdir: Path, model_name: str,
                       config: dict, force: bool = False) -> dict:
    result_path = outdir / "validation_baselines.csv"
    run_id = f"validation__{model_name}__s42__p1f8r3"
    if result_path.exists() and not force:
        old = pd.read_csv(result_path)
        matched = old[old["run_id"].astype(str) == run_id]
        if len(matched):
            row = matched.iloc[-1].to_dict()
            print(
                f"[validation cache] {model_name}: F1={row['f1']:.2f}, "
                f"Top1={row['top1']:.2f}"
            )
            return row

    print("\n" + "=" * 90)
    print(f"VALIDATION-ONLY BASELINE {model_name}")
    print("=" * 90)
    tr.set_seed(42)
    device = torch.device("cpu")
    builder = exp.make_builder(protocol.raw, history_length=3, temporal=True)
    model = exp.build_model(
        model_name, builder, protocol.raw["n_classes"]
    ).to(device)
    head = exp.SharedRootHead(latent_dim=32).to(device)
    log_dir = outdir / "baseline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (log_dir / f"{model_name}.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(exp.Tee(sys.stdout, stream)):
            model = tr.pretrain_self_supervised(
                model, protocol.train, builder, device,
                epochs=1, lr=0.002,
                use_dynamic_edges=False, use_cross_layer=False,
                use_phy_chain=True, use_net_edges=True,
                verbose=True, model_name=model_name,
            )
            model = tr.finetune_supervised(
                model, protocol.train, builder, device,
                epochs=8, lr=0.002, freeze_encoder=False,
                use_dynamic_edges=False, use_cross_layer=False,
                use_phy_chain=True, use_net_edges=True,
                verbose=True, model_name=model_name,
                encoder_lr_scale=float(config["encoder_lr_scale"]),
                class_weight_power=float(config["class_weight_power"]),
                apply_class_boosts=bool(config["apply_class_boosts"]),
                oversample=bool(config["oversample"]),
                label_smoothing=float(config["label_smoothing"]),
            )
            exp.train_shared_root_head(
                model, head, protocol.train, builder, device,
                epochs=3, lr=0.002,
            )

    cls, _, _ = exp.evaluate_classification(model, protocol.val, builder, device)
    root = exp.evaluate_shared_root(model, head, protocol.val, builder, device)
    row = {
        "run_id": run_id,
        "model": model_name,
        "seed": 42,
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
        "params_k": sum(p.numel() for p in model.parameters()) / 1000.0,
        "train_seconds": time.perf_counter() - start,
    }
    if result_path.exists():
        frame = pd.read_csv(result_path)
        frame = frame[frame["run_id"].astype(str) != run_id]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(result_path, index=False)
    print(
        f"VALIDATION RESULT {model_name}: Acc={row['accuracy']:.2f}, "
        f"F1={row['f1']:.2f}, BAcc={row['balanced_accuracy']:.2f}, "
        f"Top1={row['top1']:.2f}, MRR={row['mrr']:.2f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_new.csv")
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    lock = json.loads(
        (args.search_dir / "locked_hgan_config.json").read_text(encoding="utf-8")
    )
    config = next(
        item for item in lock["candidate_registry"]
        if item["name"] == lock["selected_candidate"]
    )
    protocol = exp.load_protocol(args.csv)
    for model_name in args.models:
        train_and_validate(protocol, args.search_dir, model_name, config, args.force)


if __name__ == "__main__":
    main()
