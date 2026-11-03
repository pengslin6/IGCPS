"""Common data split, comparison-model construction and evaluation."""
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef)
from sklearn.preprocessing import label_binarize
from . import components as tr
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


def build_model(name, builder, n_classes):
    node_dim = builder.temporal_feat_dim
    hidden = 64
    latent = 32
    n_nodes = builder.n_nodes
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


def evaluate_shared_root(model, head, data, builder, device, return_scores=False):
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
    all_scores = np.asarray(all_scores)
    metrics = tr.compute_strict_traceback_metrics(all_scores, data, builder)
    metrics["inference_time_ms"] = elapsed_ms
    return (metrics, all_scores) if return_scores else metrics


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
