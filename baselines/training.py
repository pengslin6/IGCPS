"""Retained comparison-model training and repeated-seed evaluation."""
from __future__ import annotations
import contextlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
from . import protocol as exp
from . import components as tr
METRICS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")
TRAINING = {"encoder_lr_scale": .5, "class_weight_power": .5,
            "apply_class_boosts": False, "oversample": True, "label_smoothing": 0.}
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
        encoder_lr_scale=TRAINING["encoder_lr_scale"],
        class_weight_power=TRAINING["class_weight_power"],
        apply_class_boosts=TRAINING["apply_class_boosts"],
        oversample=TRAINING["oversample"],
        label_smoothing=TRAINING["label_smoothing"],
        temporal_recency_strength=recency,
        selection_data=selection,
        restore_best=restore_best,
        early_stopping_patience=max(epochs + 1, 15),
    )
    return model


def summarize(frame: pd.DataFrame) -> dict:
    output = {}
    for model_name, model_rows in frame.groupby("model"):
        model_summary = {}
        n = len(model_rows)
        multiplier = float(t.ppf(0.975, n - 1)) if n > 1 else 0.0
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
    device="auto",
) -> dict:
    safe_name = model_name.replace("/", "_")
    result_path = outdir / f"result__{safe_name}__seed{seed}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("root_reference_policy") != tr.ROOT_REFERENCE_POLICY:
            raise ValueError("Saved baseline uses a different or unversioned root-reference policy; use a new output directory.")
        return result

    tr.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
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
            model = fit_model(
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

    classification, _, probabilities = exp.evaluate_classification(
        model, test, builder, device
    )
    root, root_scores = exp.evaluate_shared_root(
        model, head, test, builder, device, return_scores=True
    )
    result = {
        "model": model_name,
        "root_reference_policy": tr.ROOT_REFERENCE_POLICY,
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
        "artifacts_saved": True,
        "torch_threads": torch.get_num_threads(),
    }
    torch.save({"state_dict": model.state_dict(), "root_state_dict": head.state_dict(),
                "model": model_name, "seed": seed, "fixed_epochs": epochs,
                "root_reference_policy": tr.ROOT_REFERENCE_POLICY},
               outdir / f"checkpoint__{safe_name}__seed{seed}.pt")
    np.savez_compressed(outdir / f"predictions__{safe_name}__seed{seed}.npz",
                        probabilities=probabilities, root_scores=root_scores,
                        labels=np.asarray(test["y"]))
    history = pd.DataFrame(getattr(model, "finetune_history", []))
    if not history.empty:
        history.to_csv(
            outdir / f"history__{safe_name}__seed{seed}.csv", index=False
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
