"""Reproducible supplementary experiments for the DA-TGT revision.

The runner deliberately disables the Ours-only raw-feature calibrator in
trace.py. Detection models use the same class-wise chronological split,
training-only normalization, topology, temporal context, and class balancing.
Root-cause comparisons use an identical supervised head on frozen embeddings.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import trace as tr


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "reviewer_experiments_20260822"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class PeakRSS:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


class SharedRootHead(nn.Module):
    """The same source-localization head used for every frozen encoder."""

    def __init__(self, latent_dim=32, n_heads=4, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, embeddings):
        x = embeddings.unsqueeze(0)
        attended, _ = self.attention(x, x, x)
        return torch.sigmoid(self.scorer(attended.squeeze(0)).squeeze(-1))


@dataclass
class Protocol:
    raw: dict
    train: dict
    val: dict
    test: dict


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def load_protocol(csv_path):
    raw = tr.load_sr_com_data(str(csv_path))
    parsed_time = pd.to_datetime(raw["times"], format="mixed", errors="coerce", utc=True)
    if parsed_time.notna().all():
        order = np.argsort(parsed_time.astype("int64").to_numpy(), kind="stable")
        if not np.array_equal(order, np.arange(len(order))):
            raw = tr.subset_data(raw, order)
            print("Chronology audit: input rows were sorted by timestamp before splitting.")
    splits = tr.chronological_split_and_scale(raw, class_aware=True)
    return Protocol(raw=raw, train=splits["train"], val=splits["val"], test=splits["test"])


def cap_protocol(protocol, train_cap_per_class=0, validation_cap_per_class=0,
                 test_cap_per_class=0, seed=42):
    """Apply the same deterministic, chronology-covering class caps to a protocol."""
    train = tr.cap_split_per_class(
        protocol.train, train_cap_per_class, seed=seed, split_name="Train"
    )
    validation = tr.cap_split_per_class(
        protocol.val, validation_cap_per_class, seed=seed + 1,
        split_name="Validation",
    )
    test = tr.cap_split_per_class(
        protocol.test, test_cap_per_class, seed=seed + 2, split_name="Test"
    )
    train["_val_data"] = validation
    train["_test_data"] = test
    return Protocol(raw=protocol.raw, train=train, val=validation, test=test)


def make_builder(raw, history_length=3, temporal=True,
                 preserve_network_features=False):
    net_dim = raw["X_net_feat"].shape[1] if raw["X_net_feat"].ndim > 1 else 1
    builder = tr.HeteroGraphBuilder(
        raw["phy_cols"],
        raw["all_ips"],
        net_feat_dim=net_dim,
        phy_feat_dim=len(raw["phy_cols"]),
        preserve_net_features=preserve_network_features,
    )
    builder.temporal_window = int(history_length)
    builder.use_temporal = bool(temporal)
    builder.temporal_feat_dim = builder.node_feat_dim + 4 if temporal else builder.node_feat_dim
    return builder


def sparse_training_view(train, fraction, seed):
    if fraction >= 0.999:
        return train
    rng = np.random.default_rng(seed)
    y = np.asarray(train["y"])
    selected = []
    for cls_id in np.unique(y):
        cls_idx = np.where(y == cls_id)[0]
        n_keep = max(1, int(round(len(cls_idx) * fraction)))
        selected.extend(rng.choice(cls_idx, n_keep, replace=False).tolist())
    subset = tr.subset_data(train, np.array(sorted(selected), dtype=int))
    subset["_val_data"] = train["_val_data"]
    subset["_test_data"] = train["_test_data"]
    return subset


def build_model(name, builder, n_classes, use_dw_sep=True,
                use_temporal_shift=True, use_dynamic_edge_weights=True,
                use_cross_attention=True, use_type_adapters=True,
                revised_hgan=True, hgan_variant=None,
                readout_variant="typed"):
    node_dim = builder.temporal_feat_dim
    hidden = 64
    latent = 32
    n_nodes = builder.n_nodes
    if name == "DA-TGT":
        if hgan_variant is None:
            hgan_variant = "typed_mean" if revised_hgan else "legacy_global"
        if hgan_variant == "typed_mean":
            encoder = tr.HGANTraceEncoder(
                in_channels=node_dim,
                hidden_channels=hidden,
                latent_channels=latent,
                num_layers=2,
                n_heads=4,
                dropout=0.15,
                use_cross_attention=use_cross_attention,
                use_type_adapters=use_type_adapters,
            )
        elif hgan_variant in {"typed_dsgc", "typed_hybrid"}:
            encoder = tr.HGANTraceDSGCFusionEncoder(
                in_channels=node_dim,
                hidden_channels=hidden,
                latent_channels=latent,
                num_layers=2,
                n_heads=4,
                dropout=0.15,
                use_cross_attention=use_cross_attention,
                use_type_adapters=use_type_adapters,
                use_dynamic_edge_weights=use_dynamic_edge_weights,
                use_dw_sep=use_dw_sep,
                use_local_attention=hgan_variant == "typed_hybrid",
            )
        elif hgan_variant == "legacy_global":
            encoder = tr.HGT_Trace(
                in_channels=node_dim,
                hidden_channels=hidden,
                latent_channels=latent,
                num_layers=2,
                n_heads=4,
                dropout=0.15,
                use_cross_attn=use_cross_attention,
                use_gate=use_cross_attention,
                use_dw_sep=use_dw_sep,
                use_temporal_shift=use_temporal_shift,
                use_dynamic_edge_weights=use_dynamic_edge_weights,
            )
        else:
            raise ValueError(f"Unknown HGAN encoder variant: {hgan_variant}")
        detection_encoder = None
        if readout_variant == "dual_encoder":
            detection_encoder = tr.GAT_AE_Custom(
                node_dim, hidden // 4, latent,
                num_layers=2, heads=4, dropout=0.3,
            )
        return tr.EnhancedTracebackSystem(
            encoder, n_nodes, builder.n_net, builder.n_phy,
            node_dim, latent, hidden_dim=hidden, n_classes=n_classes,
            readout_variant=readout_variant,
            net_feature_dim=builder.net_feat_dim,
            net_raw_offset=builder.net_raw_offset,
            detection_encoder=detection_encoder,
        )

    constructors = {
        "GCN-AE": lambda: tr.GCN_AE_Custom(node_dim, hidden, latent, num_layers=2, dropout=0.3),
        "GAT-AE": lambda: tr.GAT_AE_Custom(node_dim, hidden // 4, latent, num_layers=2, heads=4, dropout=0.3),
        "VGAE": lambda: tr.VGAE_Custom(node_dim, hidden, latent, num_layers=2, dropout=0.3),
        "GraphSAGE-AE": lambda: tr.GraphSAGE_AE_Custom(node_dim, hidden, latent, num_layers=2, dropout=0.3),
        "IIoT-GNN": lambda: tr.IIoT_GNN_Custom(node_dim, hidden, latent, num_layers=3, dropout=0.3),
        "EE-GCN": lambda: tr.EE_GCN_Custom(
            node_dim, hidden, latent, num_layers=2, dropout=0.3,
            use_edge_feat=True, use_edge_attn=True,
        ),
        "STGaAN": lambda: tr.STGaAN_Custom(
            node_dim, hidden, latent, num_layers=2, n_heads=4, dropout=0.3,
            use_temporal=True, use_spatial=True, use_fusion=True,
        ),
        "STCI": lambda: tr.STCI_Custom(
            node_dim, hidden, latent, num_layers=2, n_heads=4, dropout=0.3,
        ),
        "DT-GNN": lambda: tr.DTGNN_Custom(
            node_dim, hidden, latent, num_layers=2, dropout=0.3,
            n_phy_nodes=builder.n_phy,
        ),
    }
    encoder = constructors[name]()
    return tr.BaselineGNNClassifier(
        encoder, n_nodes, latent, hidden_dim=hidden, n_classes=n_classes
    )


def static_adj(builder, device):
    adj = builder.build_static_adjacency(
        use_cross_layer=False, use_phy_chain=True, use_net_edges=True
    )
    return torch.tensor(adj, dtype=torch.float32, device=device)


def train_shared_root_head(model, head, data, builder, device, epochs=2, lr=0.002):
    anomaly_indices = [
        i for i, label in enumerate(data["y"])
        if int(label) > 0 and tr.infer_true_root_nodes(i, data, builder)
    ]
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    adj = static_adj(builder, device)
    model.eval()
    head.train()
    history = []
    for epoch in range(epochs):
        np.random.shuffle(anomaly_indices)
        total_loss = 0.0
        for t in anomaly_indices:
            feats = builder.get_node_features_with_history(
                t, data["X_net_feat"], data["X_phy"], data["src_ips"], data["dst_ips"]
            )
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            target = tr.build_root_target(t, data, builder, device)
            with torch.no_grad():
                z = model.encode(x, adj)
            scores = head(z.detach()).clamp(1e-6, 1 - 1e-6)
            weights = torch.ones_like(target)
            weights[target > 0] = max(5.0, builder.n_nodes / max(target.sum().item(), 1.0))
            loss = F.binary_cross_entropy(scores, target, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(len(anomaly_indices), 1)
        history.append(avg)
        print(f"     [Shared root] Epoch {epoch + 1}/{epochs}: Loss={avg:.6f}")
    head.eval()
    return history


def evaluate_classification(model, data, builder, device):
    model.eval()
    adj = static_adj(builder, device)
    predictions = []
    probabilities = []
    start = time.perf_counter()
    with torch.no_grad():
        for t in range(len(data["y"])):
            feats = builder.get_node_features_with_history(
                t, data["X_net_feat"], data["X_phy"], data["src_ips"], data["dst_ips"]
            )
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            out = model(x, adj, return_traceback=False)
            logits = out["anomaly_logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions.append(int(np.argmax(probs)))
            probabilities.append(probs)
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(data["y"]), 1)
    y = np.asarray(data["y"], dtype=int)
    pred = np.asarray(predictions, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    n_classes = int(data.get("n_classes", probs.shape[1]))
    y_bin = label_binarize(y, classes=range(n_classes))
    try:
        auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr") * 100
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_bin, probs, average="macro") * 100
    except ValueError:
        pr_auc = float("nan")

    cm = confusion_matrix(y, pred, labels=range(n_classes))
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    per_class_fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) > 0)
    per_class_fnr = np.divide(fn, fn + tp, out=np.zeros_like(fn), where=(fn + tp) > 0)
    per_class_specificity = 1.0 - per_class_fpr
    metrics = {
        "accuracy": accuracy_score(y, pred) * 100,
        "precision": precision_score(y, pred, average="macro", zero_division=0) * 100,
        "recall": recall_score(y, pred, average="macro", zero_division=0) * 100,
        "f1": f1_score(y, pred, average="macro", zero_division=0) * 100,
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0) * 100,
        "balanced_accuracy": balanced_accuracy_score(y, pred) * 100,
        "mcc": matthews_corrcoef(y, pred) * 100,
        "auc": auc,
        "pr_auc": pr_auc,
        "fpr": per_class_fpr.mean() * 100,
        "fnr": per_class_fnr.mean() * 100,
        "specificity": per_class_specificity.mean() * 100,
        "per_class_fpr": per_class_fpr * 100,
        "per_class_fnr": per_class_fnr * 100,
        "inference_time_ms": elapsed_ms,
        "confusion_matrix": cm,
    }
    return metrics, pred, probs


def evaluate_shared_root(model, head, data, builder, device):
    model.eval()
    head.eval()
    adj = static_adj(builder, device)
    all_scores = []
    start = time.perf_counter()
    with torch.no_grad():
        for t in range(len(data["y"])):
            feats = builder.get_node_features_with_history(
                t, data["X_net_feat"], data["X_phy"], data["src_ips"], data["dst_ips"]
            )
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            z = model.encode(x, adj)
            all_scores.append(head(z).cpu().numpy())
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(data["y"]), 1)
    metrics = tr.compute_strict_traceback_metrics(np.asarray(all_scores), data, builder)
    metrics["inference_time_ms"] = elapsed_ms
    return metrics


def calibration_metrics(y, pred, probs, bins=15):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    confidence = probs.max(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    reliability = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
            reliability.append({
                "lower": float(lo),
                "upper": float(hi),
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "confidence": float(confidence[mask].mean()),
            })
    onehot = np.eye(probs.shape[1])[y]
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))
    nll = -np.mean(np.log(probs[np.arange(len(y)), y]))
    entropy = -np.sum(probs * np.log(probs), axis=1)
    error = 1.0 - correct
    try:
        from scipy.stats import spearmanr
        entropy_error = float(spearmanr(entropy, error).statistic)
    except Exception:
        entropy_error = float(np.corrcoef(entropy, error)[0, 1])
    return {
        "ece15": ece,
        "brier": brier,
        "nll": nll,
        "entropy_error_spearman": entropy_error,
        "mean_confidence": confidence.mean(),
        "reliability_bins": reliability,
    }


def auxiliary_uncertainty_metrics(model, data, builder, device, pred):
    if not isinstance(model, tr.EnhancedTracebackSystem):
        return {}
    model.eval()
    adj = static_adj(builder, device)
    uncertainty = []
    with torch.no_grad():
        for t in range(len(data["y"])):
            feats = builder.get_node_features_with_history(
                t, data["X_net_feat"], data["X_phy"], data["src_ips"], data["dst_ips"]
            )
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            out = model(x, adj, return_traceback=True)
            uncertainty.append(float(out["uncertainty"].mean().cpu()))
    uncertainty = np.asarray(uncertainty)
    error = (np.asarray(pred) != np.asarray(data["y"])).astype(int)
    try:
        from scipy.stats import spearmanr
        corr = float(spearmanr(uncertainty, error).statistic)
    except Exception:
        corr = float(np.corrcoef(uncertainty, error)[0, 1])
    try:
        error_auc = roc_auc_score(error, uncertainty)
    except ValueError:
        error_auc = float("nan")
    return {
        "aux_uncertainty_error_spearman": corr,
        "aux_uncertainty_error_auc": error_auc,
        "aux_uncertainty_correct_mean": float(uncertainty[error == 0].mean()),
        "aux_uncertainty_error_mean": float(uncertainty[error == 1].mean()),
    }


def parse_training_history(log_path, run_id):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in re.finditer(r"Pretrain Epoch (\d+)/(\d+): Edge Loss=([0-9.eE+-]+)", text):
        rows.append({"run_id": run_id, "stage": "pretrain", "epoch": int(match.group(1)), "loss": float(match.group(3))})
    for match in re.finditer(r"Finetune Epoch (\d+)/(\d+): Loss=([0-9.eE+-]+), Acc=([0-9.eE+-]+)%, F1=([0-9.eE+-]+)%", text):
        rows.append({
            "run_id": run_id, "stage": "finetune", "epoch": int(match.group(1)),
            "loss": float(match.group(3)), "val_accuracy": float(match.group(4)),
            "val_f1": float(match.group(5)),
        })
    return rows


def upsert_csv(path, row, key="run_id"):
    path = Path(path)
    if path.exists():
        frame = pd.read_csv(path)
        if key in frame and row[key] in set(frame[key].astype(str)):
            frame = frame[frame[key].astype(str) != str(row[key])]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(path, index=False)


def run_one(protocol, outdir, experiment, model_name, seed, pretrain_epochs,
            finetune_epochs, root_epochs=2, train_fraction=1.0,
            history_length=3, temporal=True, use_dw_sep=True,
            use_temporal_shift=True, use_dynamic_edge_weights=True,
            use_cross_attention=True, use_type_adapters=True,
            revised_hgan=True, encoder_lr_scale=0.1,
            class_weight_power=1.0, apply_class_boosts=True,
            oversample=True, label_smoothing=0.0,
             early_stopping_patience=15,
             measure_aux=False, pretrain_full=False, evaluate_root=True,
             hgan_variant=None, temporal_recency_strength=0.0, force=False):
    variant = f"dw{int(use_dw_sep)}_ts{int(use_temporal_shift)}_de{int(use_dynamic_edge_weights)}"
    if model_name == "DA-TGT":
        variant += (
            f"_rev{int(revised_hgan)}_ca{int(use_cross_attention)}"
            f"_ta{int(use_type_adapters)}_els{encoder_lr_scale:g}"
            f"_cwp{class_weight_power:g}_cb{int(apply_class_boosts)}"
            f"_os{int(oversample)}_ls{label_smoothing:g}"
        )
        if hgan_variant is not None:
            variant += f"_hv{hgan_variant}"
        if temporal_recency_strength:
            variant += f"_tr{temporal_recency_strength:g}"
    if pretrain_full:
        variant += "_ufull"
    run_id = (
        f"{experiment}__{model_name.replace(' ', '_')}__s{seed}__f{train_fraction:g}"
        f"__k{history_length}__p{pretrain_epochs}f{finetune_epochs}r{root_epochs}__{variant}"
    )
    result_path = outdir / "results.csv"
    if result_path.exists() and not force:
        old = pd.read_csv(result_path)
        if run_id in set(old.get("run_id", pd.Series(dtype=str)).astype(str)):
            print(f"[skip] {run_id}")
            return None

    print("\n" + "=" * 90)
    print(f"RUN {run_id}")
    print("=" * 90)
    tr.set_seed(seed)
    builder = make_builder(protocol.raw, history_length=history_length, temporal=temporal)
    train = sparse_training_view(protocol.train, train_fraction, seed)
    pretrain_data = protocol.train if pretrain_full else train
    model = build_model(
        model_name, builder, protocol.raw["n_classes"],
        use_dw_sep=use_dw_sep,
        use_temporal_shift=use_temporal_shift,
        use_dynamic_edge_weights=use_dynamic_edge_weights,
        use_cross_attention=use_cross_attention,
        use_type_adapters=use_type_adapters,
        revised_hgan=revised_hgan,
        hgan_variant=hgan_variant,
    ).to(torch.device("cpu"))
    device = torch.device("cpu")
    log_path = outdir / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    head = SharedRootHead(latent_dim=32).to(device)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_stream)):
            with PeakRSS() as memory:
                model = tr.pretrain_self_supervised(
                    model, pretrain_data, builder, device,
                    epochs=pretrain_epochs, lr=0.002,
                    use_dynamic_edges=False, use_cross_layer=False,
                    use_phy_chain=True, use_net_edges=True,
                    verbose=True, model_name=model_name,
                )
                model = tr.finetune_supervised(
                    model, train, builder, device,
                    epochs=finetune_epochs, lr=0.002,
                    freeze_encoder=False,
                    use_dynamic_edges=False, use_cross_layer=False,
                    use_phy_chain=True, use_net_edges=True,
                    verbose=True, model_name=model_name,
                    encoder_lr_scale=encoder_lr_scale,
                    class_weight_power=class_weight_power,
                    apply_class_boosts=apply_class_boosts,
                    oversample=oversample,
                    label_smoothing=label_smoothing,
                    early_stopping_patience=early_stopping_patience,
                    temporal_recency_strength=temporal_recency_strength,
                )
                root_history = train_shared_root_head(
                    model, head, train, builder, device,
                    epochs=root_epochs, lr=0.002,
                ) if evaluate_root else []
    train_seconds = time.perf_counter() - start

    cls, pred, probs = evaluate_classification(model, protocol.test, builder, device)
    root = evaluate_shared_root(model, head, protocol.test, builder, device) if evaluate_root else {
        "rca": float("nan"), "mrr": float("nan"), "ndcg": float("nan"),
        "apd": float("nan"), "inference_time_ms": float("nan"),
    }
    cal = calibration_metrics(protocol.test["y"], pred, probs)
    aux = auxiliary_uncertainty_metrics(model, protocol.test, builder, device, pred) if measure_aux else {}
    params = sum(p.numel() for p in model.parameters())
    row = {
        "run_id": run_id,
        "experiment": experiment,
        "model": model_name,
        "seed": seed,
        "train_fraction": train_fraction,
        "train_samples": len(train["y"]),
        "pretrain_samples": len(pretrain_data["y"]),
        "pretrain_full_unlabeled": pretrain_full,
        "history_length": history_length,
        "temporal": temporal,
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "root_epochs": root_epochs,
        "evaluate_root": evaluate_root,
        "use_dw_sep": use_dw_sep,
        "use_temporal_shift": use_temporal_shift,
        "use_dynamic_edge_weights": use_dynamic_edge_weights,
        "use_cross_attention": use_cross_attention,
        "use_type_adapters": use_type_adapters,
        "revised_hgan": revised_hgan,
        "hgan_variant": hgan_variant,
        "encoder_lr_scale": encoder_lr_scale,
        "class_weight_power": class_weight_power,
        "apply_class_boosts": apply_class_boosts,
        "oversample": oversample,
        "label_smoothing": label_smoothing,
        "early_stopping_patience": early_stopping_patience,
        "accuracy": cls["accuracy"],
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "f1_weighted": cls["f1_weighted"],
        "balanced_accuracy": cls["balanced_accuracy"],
        "mcc": cls["mcc"],
        "auc": cls["auc"],
        "pr_auc": cls["pr_auc"],
        "fpr": cls["fpr"],
        "fnr": cls["fnr"],
        "specificity": cls["specificity"],
        "rca": root["rca"],
        "mrr": root["mrr"],
        "ndcg5": root["ndcg"],
        "apd": root["apd"],
        "ece15": cal["ece15"],
        "brier": cal["brier"],
        "nll": cal["nll"],
        "entropy_error_spearman": cal["entropy_error_spearman"],
        "mean_confidence": cal["mean_confidence"],
        "train_seconds": train_seconds,
        "peak_rss_mb": memory.peak_rss / 1024 ** 2,
        "peak_rss_delta_mb": (memory.peak_rss - memory.start_rss) / 1024 ** 2,
        "inference_time_ms": cls["inference_time_ms"],
        "root_inference_time_ms": root["inference_time_ms"],
        "two_pass_total_time_ms": cls["inference_time_ms"] + root["inference_time_ms"],
        "params_k": params / 1000,
        "model_size_mb": params * 4 / 1024 ** 2,
        "root_final_loss": root_history[-1] if root_history else float("nan"),
        "confusion_matrix": json.dumps(cls["confusion_matrix"].tolist()),
        "per_class_fpr": json.dumps(cls["per_class_fpr"].tolist()),
        "per_class_fnr": json.dumps(cls["per_class_fnr"].tolist()),
        **aux,
    }
    upsert_csv(result_path, row)
    histories = parse_training_history(log_path, run_id)
    histories.extend(
        {"run_id": run_id, "stage": "root", "epoch": i + 1, "loss": loss}
        for i, loss in enumerate(root_history)
    )
    for item in histories:
        upsert_csv(outdir / "history.csv", item, key="run_id_stage_epoch") if False else None
    if histories:
        hist_path = outdir / "history.csv"
        old = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["run_id"].astype(str) != run_id]
        pd.concat([old, pd.DataFrame(histories)], ignore_index=True).to_csv(hist_path, index=False)
    (outdir / "details").mkdir(parents=True, exist_ok=True)
    detail = dict(row)
    detail["calibration"] = cal
    detail["root_history"] = root_history
    (outdir / "details" / f"{run_id}.json").write_text(
        json.dumps(detail, indent=2, default=jsonable), encoding="utf-8"
    )
    print(
        f"RESULT {model_name}: Acc={row['accuracy']:.2f}, F1={row['f1']:.2f}, "
        f"RCA={row['rca']:.2f}, MRR={row['mrr']:.2f}, "
        f"ECE={row['ece15']:.4f}, train={train_seconds / 60:.2f} min"
    )
    return model, head, builder, row


def summarize_group(outdir, experiment, group_cols):
    frame = pd.read_csv(outdir / "results.csv")
    frame = frame[frame["experiment"] == experiment]
    metrics = ["accuracy", "f1", "auc", "rca", "mrr", "ndcg5", "ece15", "brier", "train_seconds"]
    rows = []
    for keys, part in frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(part)
        for metric in metrics:
            values = part[metric].dropna().astype(float).to_numpy()
            if not len(values):
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            try:
                from scipy.stats import t
                half = float(t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
            except Exception:
                half = 1.96 * std / math.sqrt(max(len(values), 1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_half"] = half
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / f"{experiment}_summary.csv", index=False)


def paired_core_tests(outdir):
    frame = pd.read_csv(outdir / "results.csv")
    frame = frame[frame["experiment"] == "core"]
    ours = frame[frame["model"] == "DA-TGT"].set_index("seed")
    base = frame[frame["model"] == "VGAE"].set_index("seed")
    common = sorted(set(ours.index) & set(base.index))
    output = {"n_pairs": len(common), "tests": {}}
    for metric in ["accuracy", "f1", "rca", "mrr"]:
        a = ours.loc[common, metric].astype(float).to_numpy()
        b = base.loc[common, metric].astype(float).to_numpy()
        entry = {"mean_difference": float(np.mean(a - b))}
        try:
            from scipy.stats import ttest_rel, wilcoxon
            entry["paired_t_p"] = float(ttest_rel(a, b).pvalue)
            entry["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
        except Exception as exc:
            entry["error"] = str(exc)
        output["tests"][metric] = entry
    (outdir / "core_paired_tests.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


def graph_scaling(outdir, repeats=30):
    device = torch.device("cpu")
    rows = []
    for n_nodes in [16, 32, 64, 128, 256]:
        tr.set_seed(42)
        n_net = max(2, n_nodes // 4)
        n_phy = n_nodes - n_net
        encoder = tr.HGANTraceDSGCFusionEncoder(
            12, 64, 32, num_layers=2, n_heads=4, dropout=0.15,
            use_cross_attention=True, use_type_adapters=True,
            use_dynamic_edge_weights=True, use_dw_sep=True,
        )
        model = tr.EnhancedTracebackSystem(
            encoder, n_nodes, n_net, n_phy, 12, 32, hidden_dim=64, n_classes=6
        ).to(device).eval()
        head = SharedRootHead(latent_dim=32).to(device).eval()
        x = torch.randn(n_nodes, 12, device=device)
        adj = torch.zeros(n_nodes, n_nodes, device=device)
        for i in range(n_net - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1
        for i in range(n_net, n_nodes - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1
        with torch.no_grad():
            for _ in range(5):
                model(x, adj, return_traceback=False)
                head(model.encode(x, adj))
            for mode in ["classification", "classification_plus_shared_root"]:
                times = []
                with PeakRSS() as memory:
                    for _ in range(repeats):
                        start = time.perf_counter()
                        output = model(x, adj, return_traceback=False)
                        if mode == "classification_plus_shared_root":
                            head(output["embeddings"])
                        times.append((time.perf_counter() - start) * 1000)
                rows.append({
                    "n_nodes": n_nodes,
                    "mode": mode,
                    "median_ms": float(np.median(times)),
                    "p95_ms": float(np.percentile(times, 95)),
                    "peak_rss_mb": memory.peak_rss / 1024 ** 2,
                    "repeats": repeats,
                })
        print(f"Scaling N={n_nodes}: completed")
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "graph_scaling.csv", index=False)
    slopes = []
    for mode, part in frame.groupby("mode"):
        slope = np.polyfit(np.log(part["n_nodes"]), np.log(part["median_ms"]), 1)[0]
        slopes.append({"mode": mode, "log_log_slope": float(slope)})
    pd.DataFrame(slopes).to_csv(outdir / "graph_scaling_slopes.csv", index=False)


def tsm_identity_check(protocol, outdir, n_samples=128):
    """Verify the inactive TSM branch with identical weights and inputs."""
    tr.set_seed(42)
    device = torch.device("cpu")
    builder = make_builder(protocol.raw, history_length=3, temporal=True)
    model = build_model(
        "DA-TGT", builder, protocol.raw["n_classes"],
        use_dw_sep=True, use_temporal_shift=True, use_dynamic_edge_weights=True,
    ).to(device).eval()
    adj = static_adj(builder, device)
    max_logit_diff = 0.0
    max_embedding_diff = 0.0
    changed_predictions = 0
    checked = min(int(n_samples), len(protocol.test["y"]))
    with torch.no_grad():
        for t in range(checked):
            feats = builder.get_node_features_with_history(
                t, protocol.test["X_net_feat"], protocol.test["X_phy"],
                protocol.test["src_ips"], protocol.test["dst_ips"],
            )
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            model.gnn_encoder.use_temporal_shift = True
            enabled = model(x, adj, return_traceback=False)
            model.gnn_encoder.use_temporal_shift = False
            disabled = model(x, adj, return_traceback=False)
            logit_diff = (enabled["anomaly_logits"] - disabled["anomaly_logits"]).abs().max().item()
            embedding_diff = (enabled["embeddings"] - disabled["embeddings"]).abs().max().item()
            max_logit_diff = max(max_logit_diff, logit_diff)
            max_embedding_diff = max(max_embedding_diff, embedding_diff)
            changed_predictions += int(
                enabled["anomaly_logits"].argmax().item()
                != disabled["anomaly_logits"].argmax().item()
            )
    result = {
        "samples": checked,
        "cross_layer_edges": int((adj[:builder.n_net, builder.n_net:] > 0).sum().item()),
        "max_abs_logit_difference": max_logit_diff,
        "max_abs_embedding_difference": max_embedding_diff,
        "changed_predictions": changed_predictions,
        "interpretation": "TSM is functionally inactive when encoder cross-layer edges are disabled.",
    }
    (outdir / "tsm_identity_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"TSM identity check: {result}")


def run_phase(args, protocol, outdir):
    common = dict(protocol=protocol, outdir=outdir, force=args.force)
    if args.phase == "pilot":
        run_one(**common, experiment="pilot", model_name="DA-TGT", seed=42,
                pretrain_epochs=1, finetune_epochs=1, root_epochs=1, measure_aux=True)
    elif args.phase == "core":
        for seed in [11, 22, 33, 44, 55]:
            for model in ["DA-TGT", "VGAE"]:
                run_one(**common, experiment="core", model_name=model, seed=seed,
                        pretrain_epochs=1, finetune_epochs=3, root_epochs=1)
        summarize_group(outdir, "core", ["model"])
        paired_core_tests(outdir)
    elif args.phase == "scarcity":
        for fraction in [0.10, 0.25, 0.50, 1.00]:
            for seed in [11, 22, 33]:
                run_one(**common, experiment="scarcity", model_name="DA-TGT", seed=seed,
                        pretrain_epochs=1, finetune_epochs=2, root_epochs=1,
                        train_fraction=fraction, pretrain_full=True)
        summarize_group(outdir, "scarcity", ["train_fraction"])
    elif args.phase == "convergence":
        run_one(**common, experiment="convergence", model_name="DA-TGT", seed=42,
                pretrain_epochs=5, finetune_epochs=8, root_epochs=3, measure_aux=True)
    elif args.phase == "sensitivity":
        tsm_identity_check(protocol, outdir)
        for history in [1, 2, 3, 5]:
            run_one(**common, experiment="history", model_name="DA-TGT", seed=42,
                    pretrain_epochs=1, finetune_epochs=2, root_epochs=1,
                    history_length=history)
        variants = [
            ("no_dwsep", False, True, True),
            ("no_dynamic_edge", True, True, False),
        ]
        for label, dw, ts, de in variants:
            run_one(**common, experiment=f"ablation_{label}", model_name="DA-TGT", seed=42,
                    pretrain_epochs=1, finetune_epochs=2, root_epochs=1,
                    use_dw_sep=dw, use_temporal_shift=ts, use_dynamic_edge_weights=de)
    elif args.phase == "fair":
        models = [
            "DA-TGT", "GCN-AE", "GAT-AE", "VGAE", "GraphSAGE-AE",
            "IIoT-GNN", "EE-GCN", "STGaAN", "STCI", "DT-GNN",
        ]
        for model in models:
            run_one(**common, experiment="fair_baselines", model_name=model, seed=42,
                    pretrain_epochs=1, finetune_epochs=3, root_epochs=3)
    elif args.phase == "locked_baselines":
        # The protocol is copied from the validation-locked HGAN configuration.
        # DA-TGT itself is not rerun here because its held-out test result is
        # guarded by hgan_validation_search.py.
        models = [
            "GraphSAGE-AE", "GCN-AE", "VGAE", "GAT-AE", "IIoT-GNN",
            "EE-GCN", "STGaAN", "STCI", "DT-GNN",
        ]
        for model in models:
            run_one(
                **common, experiment="locked_baselines", model_name=model, seed=42,
                pretrain_epochs=1, finetune_epochs=8, root_epochs=3,
                encoder_lr_scale=0.5, class_weight_power=0.5,
                apply_class_boosts=False, oversample=True, label_smoothing=0.0,
            )
    elif args.phase == "locked_paired":
        for seed in [11, 22, 33, 44, 55]:
            for model in ["DA-TGT", "GAT-AE"]:
                run_one(
                    **common, experiment="locked_paired", model_name=model, seed=seed,
                    pretrain_epochs=1, finetune_epochs=8, root_epochs=3,
                    encoder_lr_scale=0.5, class_weight_power=0.5,
                    apply_class_boosts=False, oversample=True, label_smoothing=0.0,
                    early_stopping_patience=3,
                )
        summarize_group(outdir, "locked_paired", ["model"])
    elif args.phase == "scaling":
        graph_scaling(outdir)
    elif args.phase == "te_audit":
        for model in ["DA-TGT", "VGAE"]:
            run_one(**common, experiment="te_audit", model_name=model, seed=42,
                    pretrain_epochs=1, finetune_epochs=3, root_epochs=0,
                    evaluate_root=False, encoder_lr_scale=0.5,
                    class_weight_power=0.5, apply_class_boosts=False,
                    oversample=True, label_smoothing=0.0,
                    temporal_recency_strength=1.0,
                    hgan_variant="typed_dsgc" if model == "DA-TGT" else None)
    else:
        raise ValueError(args.phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_new.csv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--phase", required=True,
        choices=["pilot", "core", "scarcity", "convergence", "sensitivity", "fair", "locked_baselines", "locked_paired", "scaling", "te_audit"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": sys.version,
        "torch": torch.__version__,
        "device": "CPU",
        "cpu_count": psutil.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 1024 ** 3,
        "csv": str(args.csv.resolve()),
        "protocol": "class-wise chronological 60/20/20; train-only scaling; no Ours-only raw-feature calibrator",
    }
    env_name = f"environment_{re.sub(r'[^A-Za-z0-9_.-]+', '_', args.csv.stem)}.json"
    (args.outdir / env_name).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.phase == "scaling":
        graph_scaling(args.outdir)
        return
    protocol = load_protocol(args.csv)
    run_phase(args, protocol, args.outdir)


if __name__ == "__main__":
    main()
