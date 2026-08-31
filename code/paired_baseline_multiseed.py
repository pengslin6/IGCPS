"""Repeated final-protocol runs for the strongest controlled baseline adapters."""

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
import torch.nn.functional as F
from scipy.stats import t

import reviewer_experiments as exp
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
METRICS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")


def summarize(frame: pd.DataFrame) -> dict:
    output = {}
    for model_name, model_rows in frame.groupby("model"):
        model_summary = {}
        n = len(model_rows)
        multiplier = float(t.ppf(0.975, n - 1)) if n > 1 else float("nan")
        for metric in METRICS:
            values = model_rows[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
            half = multiplier * sd / np.sqrt(n) if n > 1 else 0.0
            model_summary[metric] = {
                "mean": mean,
                "std": sd,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        output[str(model_name)] = model_summary
    return output


def fixed_epochs(protocol_dir: Path, model_name: str) -> int:
    frame = pd.read_csv(protocol_dir / "baseline_validation.csv")
    selected = frame[frame["model"].astype(str) == model_name]
    if selected.empty:
        raise ValueError(f"No validation-locked epoch found for {model_name}")
    return int(selected.iloc[-1]["best_epoch"])


def train_shared_root_head_cached(
    model,
    head,
    data: dict,
    builder,
    device,
    epochs: int = 3,
    lr: float = 0.002,
) -> list[float]:
    """Train the frozen-encoder root head using one deterministic embedding cache."""
    anomaly_indices = [
        i for i, label in enumerate(data["y"])
        if int(label) > 0 and tr.infer_true_root_nodes(i, data, builder)
    ]
    adjacency = exp.static_adj(builder, device)
    model.eval()
    cached = []
    with torch.no_grad():
        for sample_index in anomaly_indices:
            features = builder.get_node_features_with_history(
                sample_index,
                data["X_net_feat"],
                data["X_phy"],
                data["src_ips"],
                data["dst_ips"],
            )
            x = torch.tensor(features, dtype=torch.float32, device=device)
            embedding = model.encode(x, adjacency).detach()
            target = tr.build_root_target(sample_index, data, builder, device)
            weights = torch.ones_like(target)
            weights[target > 0] = max(
                5.0,
                builder.n_nodes / max(target.sum().item(), 1.0),
            )
            cached.append((embedding, target, weights))

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    order = np.arange(len(cached))
    head.train()
    history = []
    for epoch in range(epochs):
        np.random.shuffle(order)
        total_loss = 0.0
        for cache_index in order:
            embedding, target, weights = cached[int(cache_index)]
            scores = head(embedding).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(scores, target, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        average = total_loss / max(len(cached), 1)
        history.append(average)
        print(
            f"     [Shared root cached] Epoch {epoch + 1}/{epochs}: "
            f"Loss={average:.6f}",
            flush=True,
        )
    head.eval()
    return history


def run_seed(
    model_name: str,
    seed: int,
    protocol,
    development: dict,
    epoch_monitor: dict,
    test: dict,
    epochs: int,
    recency: float,
    outdir: Path,
) -> dict:
    safe_name = model_name.replace("/", "_")
    result_path = outdir / f"result__{safe_name}__seed{seed}.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    tr.set_seed(seed)
    device = torch.device("cpu")
    builder = exp.make_builder(
        protocol.raw,
        history_length=1,
        temporal=False,
        preserve_network_features=True,
    )
    model = exp.build_model(model_name, builder, protocol.raw["n_classes"]).to(device)
    head = exp.SharedRootHead(latent_dim=32).to(device)
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (log_dir / f"{safe_name}__seed{seed}.log").open(
        "w", encoding="utf-8"
    ) as stream:
        with contextlib.redirect_stdout(exp.Tee(sys.stdout, stream)):
            model = temporal.fit_model(
                model,
                development,
                epoch_monitor,
                builder,
                device,
                model_name,
                epochs,
                recency,
                restore_best=False,
            )
            root_history = train_shared_root_head_cached(
                model,
                head,
                development,
                builder,
                device,
                epochs=3,
                lr=0.002,
            )

    classification, _, _ = exp.evaluate_classification(
        model, test, builder, device
    )
    root = exp.evaluate_shared_root(model, head, test, builder, device)
    result = {
        "model": model_name,
        "seed": int(seed),
        "fixed_epochs": int(epochs),
        "root_epochs": 3,
        "recency_strength": float(recency),
        "accuracy": float(classification["accuracy"]),
        "f1": float(classification["f1"]),
        "auc": float(classification["auc"]),
        "top1": float(root["rca"]),
        "mrr": float(root["mrr"]),
        "ndcg5": float(root["ndcg"]),
        "elapsed_seconds": time.perf_counter() - started,
        "selection_used_test": False,
        "history_length": 1,
        "temporal_history_enabled": False,
        "epoch_monitor_samples": int(len(epoch_monitor["y"])),
        "root_embedding_cache_enabled": True,
        "root_final_loss": float(root_history[-1]),
    }
    history = pd.DataFrame(getattr(model, "finetune_history", []))
    if not history.empty:
        history.to_csv(
            outdir / f"history__{safe_name}__seed{seed}.csv", index=False
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--train-cap-per-class", type=int, default=0)
    parser.add_argument("--validation-cap-per-class", type=int, default=0)
    parser.add_argument("--test-cap-per-class", type=int, default=0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    lock = json.loads(
        (args.protocol_dir / "temporal_lock.json").read_text(encoding="utf-8")
    )
    recency = float(lock["selected_recency_strength"])
    temporal.BUILDER_HISTORY_LENGTH = 1
    temporal.BUILDER_TEMPORAL = False
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
    # With restore_best=False, per-epoch selection metrics do not affect training.
    # A fixed compact monitor preserves logging while avoiding a full development
    # forward pass after every epoch; final metrics still use the complete test cap.
    epoch_monitor = tr.cap_split_per_class(
        development,
        8,
        seed=43,
        split_name="Epoch monitor",
    )

    rows = []
    for model_name in args.models:
        epochs = fixed_epochs(args.protocol_dir, model_name)
        for position, seed in enumerate(args.seeds, start=1):
            print(
                f"{model_name} seed {seed} ({position}/{len(args.seeds)}), "
                f"fixed epochs={epochs}",
                flush=True,
            )
            result = run_seed(
                model_name,
                seed,
                protocol,
                development,
                epoch_monitor,
                test,
                epochs,
                recency,
                args.outdir,
            )
            rows.append(result)
            print(
                f"complete: Acc/F1/AUC={result['accuracy']:.2f}/"
                f"{result['f1']:.2f}/{result['auc']:.2f}; "
                f"Top1/MRR/NDCG5={result['top1']:.2f}/"
                f"{result['mrr']:.2f}/{result['ndcg5']:.2f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(args.outdir / "runs.csv", index=False)

    frame = pd.DataFrame(rows).sort_values(["model", "seed"])
    frame.to_csv(args.outdir / "runs.csv", index=False)
    summary = {
        "models": list(args.models),
        "seeds": list(args.seeds),
        "fixed_epochs": {
            model_name: fixed_epochs(args.protocol_dir, model_name)
            for model_name in args.models
        },
        "recency_strength": recency,
        "history_length": 1,
        "temporal_history_enabled": False,
        "train_cap_per_class": max(0, args.train_cap_per_class),
        "validation_cap_per_class": max(0, args.validation_cap_per_class),
        "test_cap_per_class": max(0, args.test_cap_per_class),
        "selection_used_test": False,
        "epoch_monitor_samples": int(len(epoch_monitor["y"])),
        "epoch_monitor_affects_weights": False,
        "root_embedding_cache_enabled": True,
        "root_embedding_cache_affects_weights": False,
        "summary": summarize(frame),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
