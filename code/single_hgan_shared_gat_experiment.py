"""Validation-locked shared HGAN for detection and root localization.

This experiment keeps one typed graph encoder and two task heads.  The only
architectural ablation is whether one gated bidirectional cyber-physical
fusion step is active.  Candidate selection uses chronological development
folds; the final 20% test interval is evaluated only after the configuration
and epoch count are locked.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import reviewer_experiments as exp
import single_hgan_joint_experiment as joint
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "single_hgan_shared_gat_20260824"
SCREEN_SEEDS = [11, 22, 33]
FINAL_SEEDS = [11, 22, 33, 44, 55]


@dataclass(frozen=True)
class Candidate:
    name: str
    preserve_network_features: bool = True
    model_dim: int = 48
    n_heads: int = 4
    local_layers: int = 2
    dropout: float = 0.10
    use_cross_fusion: bool = True
    root_weight: float = 0.24
    normal_root_weight: float = 0.02
    learning_rate: float = 8e-4
    weight_decay: float = 5e-4
    epochs: int = 14


CANDIDATES = [
    Candidate(name="shared_gat_no_fusion", use_cross_fusion=False),
    Candidate(name="shared_gat_cp_fusion", use_cross_fusion=True),
]


def set_seed(seed: int) -> None:
    joint.set_seed(seed)


def scale_fold(train: dict, validation: dict) -> tuple[dict, dict]:
    """Fit every scaler on the earlier interval only."""
    train = copy.deepcopy(train)
    validation = copy.deepcopy(validation)
    phy_scaler = StandardScaler().fit(train["X_phy"])
    net_scaler = StandardScaler().fit(train["X_net_feat"])
    control_scaler = None
    if train.get("X_control") is not None and train["X_control"].shape[1] > 0:
        control_scaler = StandardScaler().fit(train["X_control"])
    for split in (train, validation):
        split["X_phy"] = phy_scaler.transform(split["X_phy"]).astype(np.float32)
        split["X_net_feat"] = net_scaler.transform(split["X_net_feat"]).astype(np.float32)
        if control_scaler is not None:
            split["X_control"] = control_scaler.transform(
                split["X_control"]
            ).astype(np.float32)
    return train, validation


def rolling_fold(raw: dict, train_end: float,
                 validation_end: float) -> tuple[dict, dict]:
    train = temporal.classwise_time_slice(raw, 0.0, train_end)
    validation = temporal.classwise_time_slice(raw, train_end, validation_end)
    return scale_fold(train, validation)


class MaskedGraphAttentionBlock(nn.Module):
    """Multi-head graph attention restricted to the supplied typed graph."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads:
            raise ValueError("model_dim must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, hidden: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        batch, nodes, dim = hidden.shape
        normalized = self.norm1(hidden)
        qkv = self.qkv(normalized).reshape(
            batch, nodes, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        mask = adjacency.gt(0).view(1, 1, nodes, nodes)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        message = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, nodes, dim
        )
        hidden = hidden + self.residual_dropout(self.out(message))
        hidden = hidden + self.residual_dropout(self.ffn(self.norm2(hidden)))
        return hidden


class TypedCrossFusion(nn.Module):
    """One active, gated, bidirectional cyber-physical fusion operation."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.net_from_phy = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.phy_from_net = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.net_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.phy_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.net_norm = nn.LayerNorm(dim)
        self.phy_norm = nn.LayerNorm(dim)

    def forward(self, net: torch.Tensor,
                phy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net_message, _ = self.net_from_phy(net, phy, phy, need_weights=False)
        phy_message, _ = self.phy_from_net(phy, net, net, need_weights=False)
        net_gate = self.net_gate(torch.cat([net, net_message], dim=-1))
        phy_gate = self.phy_gate(torch.cat([phy, phy_message], dim=-1))
        return (
            self.net_norm(net + net_gate * net_message),
            self.phy_norm(phy + phy_gate * phy_message),
        )


class SharedTypedGAT(nn.Module):
    """One shared typed graph encoder with detection and root-ranking heads."""

    def __init__(self, input_dim: int, n_nodes: int, n_net: int,
                 n_classes: int, candidate: Candidate):
        super().__init__()
        dim = int(candidate.model_dim)
        self.n_nodes = int(n_nodes)
        self.n_net = int(n_net)
        self.use_cross_fusion = bool(candidate.use_cross_fusion)

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.type_embedding = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.net_adapter = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.phy_adapter = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.type_norm = nn.LayerNorm(dim)
        self.local_blocks = nn.ModuleList([
            MaskedGraphAttentionBlock(dim, candidate.n_heads, candidate.dropout)
            for _ in range(candidate.local_layers)
        ])
        if self.use_cross_fusion:
            self.cross_fusion = TypedCrossFusion(
                dim, candidate.n_heads, candidate.dropout
            )
        self.output_norm = nn.LayerNorm(dim)

        self.net_pool = nn.Linear(dim, 1)
        self.phy_pool = nn.Linear(dim, 1)
        self.root_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(candidate.dropout),
            nn.Linear(dim, 1),
        )
        self.detection_norm = nn.LayerNorm(dim * 8)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 8, dim * 3),
            nn.LayerNorm(dim * 3),
            nn.GELU(),
            nn.Dropout(candidate.dropout),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(candidate.dropout / 2),
            nn.Linear(dim, n_classes),
        )

    @staticmethod
    def attention_pool(nodes: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
        weights = torch.softmax(scorer(nodes).squeeze(-1), dim=1)
        return torch.sum(weights.unsqueeze(-1) * nodes, dim=1)

    def forward(self, features: torch.Tensor,
                adjacency: torch.Tensor) -> dict:
        hidden = self.input_projection(features)
        net = hidden[:, :self.n_net] + self.type_embedding[0]
        phy = hidden[:, self.n_net:] + self.type_embedding[1]
        net = net + self.net_adapter(net)
        phy = phy + self.phy_adapter(phy)
        hidden = self.type_norm(torch.cat([net, phy], dim=1))
        for block in self.local_blocks:
            hidden = block(hidden, adjacency)

        net = hidden[:, :self.n_net]
        phy = hidden[:, self.n_net:]
        if self.use_cross_fusion:
            net, phy = self.cross_fusion(net, phy)
        nodes = self.output_norm(torch.cat([net, phy], dim=1))

        net_repr = self.attention_pool(net, self.net_pool)
        phy_repr = self.attention_pool(phy, self.phy_pool)
        graph_context = torch.cat([net.mean(dim=1), phy.mean(dim=1)], dim=-1)
        graph_context = graph_context.reshape(features.shape[0], 2, -1).mean(dim=1)
        root_logits = self.root_scorer(torch.cat([
            nodes,
            graph_context.unsqueeze(1).expand(-1, self.n_nodes, -1),
        ], dim=-1)).squeeze(-1)

        # The source-weighted representation is part of the same computation
        # graph, so both objectives shape the shared encoder and root scorer.
        source_weights = torch.softmax(root_logits, dim=1)
        source_repr = torch.sum(source_weights.unsqueeze(-1) * nodes, dim=1)
        net_max = net.max(dim=1).values
        phy_max = phy.max(dim=1).values
        discrepancy = torch.abs(net_repr - phy_repr)
        agreement = net_repr * phy_repr
        detection = torch.cat([
            net_repr, phy_repr, net_max, phy_max,
            graph_context, source_repr, discrepancy, agreement,
        ], dim=-1)
        anomaly_logits = self.classifier(self.detection_norm(detection))
        return {
            "anomaly_logits": anomaly_logits,
            "root_logits": root_logits,
            "embeddings": nodes,
        }


def joint_loss(output: dict, labels: torch.Tensor, roots: torch.Tensor,
               root_mask: torch.Tensor, recency: torch.Tensor,
               weights: torch.Tensor, candidate: Candidate) -> tuple[torch.Tensor, dict]:
    classification = F.cross_entropy(
        output["anomaly_logits"], labels, weight=weights, reduction="none"
    )
    classification = torch.mean(classification * recency)

    log_ranking = F.log_softmax(output["root_logits"], dim=1)
    target_distribution = roots / roots.sum(dim=1, keepdim=True).clamp_min(1.0)
    rank_per_sample = -(target_distribution * log_ranking).sum(dim=1)
    anomaly_denominator = root_mask.sum().clamp_min(1.0)
    root_ranking = torch.sum(rank_per_sample * root_mask) / anomaly_denominator

    normal_mask = 1.0 - root_mask
    normal_per_sample = F.softplus(output["root_logits"]).mean(dim=1)
    normal_root = torch.sum(normal_per_sample * normal_mask)
    normal_root = normal_root / normal_mask.sum().clamp_min(1.0)
    root_loss = root_ranking + candidate.normal_root_weight * normal_root
    total = classification + candidate.root_weight * root_loss
    return total, {
        "classification": float(classification.detach()),
        "root_ranking": float(root_ranking.detach()),
        "normal_root": float(normal_root.detach()),
    }


def train_model(candidate: Candidate, train_cache: dict,
                validation_cache: dict, validation_data: dict, builder,
                adjacency: torch.Tensor, seed: int,
                fixed_epochs: int | None = None) -> tuple[nn.Module, list[dict], int]:
    set_seed(seed)
    model = SharedTypedGAT(
        input_dim=train_cache["features"].shape[-1],
        n_nodes=builder.n_nodes,
        n_net=builder.n_net,
        n_classes=int(train_cache["labels"].max().item() + 1),
        candidate=candidate,
    )
    weights = joint.class_weights(train_cache["labels"])
    dataset = TensorDataset(
        train_cache["features"], train_cache["labels"],
        train_cache["root_targets"], train_cache["root_mask"],
        train_cache["recency"],
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=96, shuffle=True, generator=generator,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    epochs = int(fixed_epochs or candidate.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=candidate.learning_rate * 0.1
    )

    late_indices = joint.classwise_late_indices(validation_cache["labels"].numpy())
    late_cache = joint.slice_cache(validation_cache, late_indices)
    late_data = tr.subset_data(validation_data, late_indices)
    best_state = None
    best_epoch = epochs
    best_key = None
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for features, labels, roots, root_mask, recency in loader:
            output = model(features, adjacency)
            loss, _ = joint_loss(
                output, labels, roots, root_mask, recency, weights, candidate
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        scheduler.step()

        full = joint.evaluate(
            model, validation_cache, validation_data, builder, adjacency
        )
        late = joint.evaluate(model, late_cache, late_data, builder, adjacency)
        selection_key = (
            min(full["f1"], late["f1"]),
            min(full["auc"], late["auc"]),
            min(full["top1"], late["top1"]),
            full["accuracy"],
        )
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / len(dataset),
            **{f"full_{key}": value for key, value in full.items()
               if key not in {"confusion_matrix", "per_class_auc"}},
            **{f"late_{key}": value for key, value in late.items()
               if key not in {"confusion_matrix", "per_class_auc"}},
        }
        history.append(row)
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch + 1
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"{candidate.name} seed {seed} epoch {epoch + 1:02d}/{epochs}: "
            f"loss={row['loss']:.4f}, full F1/AUC/Top1="
            f"{full['f1']:.2f}/{full['auc']:.2f}/{full['top1']:.2f}, "
            f"late={late['f1']:.2f}/{late['auc']:.2f}/{late['top1']:.2f}",
            flush=True,
        )
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def screen() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = exp.load_protocol(ROOT / "sr_com_new.csv")
    folds = [
        ("train40_val20", *rolling_fold(protocol.raw, 0.4, 0.6), [42]),
        ("train60_val20", *rolling_fold(protocol.raw, 0.6, 0.8), SCREEN_SEEDS),
    ]
    rows = []
    for candidate in CANDIDATES:
        builder = joint.make_builder(protocol.raw, candidate)
        adjacency = joint.static_adjacency(builder)
        for fold_name, train_data, validation_data, seeds in folds:
            train_cache = joint.build_cache(train_data, builder)
            validation_cache = joint.build_cache(validation_data, builder)
            for seed in seeds:
                print(f"\nSCREEN {candidate.name} / {fold_name} / seed {seed}")
                model, history, best_epoch = train_model(
                    candidate, train_cache, validation_cache, validation_data,
                    builder, adjacency, seed,
                )
                result = joint.evaluate(
                    model, validation_cache, validation_data, builder, adjacency
                )
                late_indices = joint.classwise_late_indices(
                    validation_cache["labels"].numpy()
                )
                late_result = joint.evaluate(
                    model,
                    joint.slice_cache(validation_cache, late_indices),
                    tr.subset_data(validation_data, late_indices),
                    builder,
                    adjacency,
                )
                run_id = f"{candidate.name}__{fold_name}__seed{seed}"
                pd.DataFrame(history).to_csv(
                    OUT / f"history__{run_id}.csv", index=False
                )
                torch.save({
                    "state_dict": model.state_dict(),
                    "candidate": asdict(candidate),
                    "fold": fold_name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                }, OUT / f"screen__{run_id}.pt")
                row = {
                    "run_id": run_id,
                    "candidate": candidate.name,
                    "use_cross_fusion": candidate.use_cross_fusion,
                    "fold": fold_name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    **{key: value for key, value in result.items()
                       if np.isscalar(value)},
                    **{f"late_{key}": value for key, value in late_result.items()
                       if np.isscalar(value)},
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT / "screen_runs.csv", index=False)

    frame = pd.DataFrame(rows)
    summaries = []
    for candidate in CANDIDATES:
        part = frame[frame["candidate"] == candidate.name]
        summaries.append({
            "candidate": candidate.name,
            "use_cross_fusion": candidate.use_cross_fusion,
            "worst_f1": float(np.minimum(part["f1"], part["late_f1"]).min()),
            "mean_f1": float(part["f1"].mean()),
            "worst_auc": float(np.minimum(part["auc"], part["late_auc"]).min()),
            "mean_auc": float(part["auc"].mean()),
            "worst_top1": float(np.minimum(part["top1"], part["late_top1"]).min()),
            "mean_top1": float(part["top1"].mean()),
            "median_epoch": int(round(float(part["best_epoch"].median()))),
        })
    summary = pd.DataFrame(summaries).sort_values(
        ["worst_f1", "mean_f1", "worst_auc", "worst_top1"],
        ascending=False,
    )
    summary.to_csv(OUT / "screen_summary.csv", index=False)
    selected_name = str(summary.iloc[0]["candidate"])
    selected = next(c for c in CANDIDATES if c.name == selected_name)
    fixed_epochs = max(1, int(summary.iloc[0]["median_epoch"]))
    lock = {
        "selection_data": (
            "class-wise chronological 0--40/40--60 and 0--60/60--80 "
            "development folds; scalers fitted on each earlier interval"
        ),
        "selection_used_final_test": False,
        "selection_rule": [
            "maximize worst full/late macro F1",
            "then mean macro F1",
            "then worst macro AUC",
            "then worst Top-1 root localization",
        ],
        "screen_seeds": {"train40_val20": [42], "train60_val20": SCREEN_SEEDS},
        "selected_candidate": asdict(selected),
        "fixed_epochs": fixed_epochs,
        "screen_summary": summary.to_dict(orient="records"),
    }
    (OUT / "validation_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print("\nVALIDATION CONFIGURATION LOCKED")
    print(json.dumps(lock, indent=2), flush=True)


def final() -> None:
    lock_path = OUT / "validation_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("Run --phase screen before the final test.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    candidate = Candidate(**lock["selected_candidate"])
    fixed_epochs = int(lock["fixed_epochs"])
    protocol = exp.load_protocol(ROOT / "sr_com_new.csv")
    development, test_data = temporal.build_development_test(protocol.raw, 0.8)
    builder = joint.make_builder(protocol.raw, candidate)
    adjacency = joint.static_adjacency(builder)
    development_cache = joint.build_cache(development, builder)
    test_cache = joint.build_cache(test_data, builder)
    baseline = pd.read_csv(
        ROOT / "temporal_robust_20260823" / "final_80pct_comparison.csv"
    )
    rows = []
    for seed in FINAL_SEEDS:
        print(f"\nFINAL SHARED MODEL / seed {seed}")
        model, history, _ = train_model(
            candidate, development_cache, development_cache, development,
            builder, adjacency, seed, fixed_epochs=fixed_epochs,
        )
        result = joint.evaluate(
            model, test_cache, test_data, builder, adjacency
        )
        ranks = {
            metric: int(1 + np.sum(
                baseline[metric].to_numpy(dtype=float) > float(result[metric])
            ))
            for metric in ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")
        }
        row = {
            "model": "HGAN-Trace-Shared",
            "seed": seed,
            "fixed_epochs": fixed_epochs,
            "selection_used_test": False,
            **{key: value for key, value in result.items()
               if np.isscalar(value)},
            **{f"rank_{key}": value for key, value in ranks.items()},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "final_runs.csv", index=False)
        pd.DataFrame(history).to_csv(
            OUT / f"final_history__seed{seed}.csv", index=False
        )
        torch.save({
            "state_dict": model.state_dict(),
            "candidate": asdict(candidate),
            "fixed_epochs": fixed_epochs,
            "seed": seed,
        }, OUT / f"HGAN-Trace-Shared__seed{seed}.pt")
        (OUT / f"final_result__seed{seed}.json").write_text(
            json.dumps({**row, "ranks": ranks,
                        "per_class_auc": result["per_class_auc"],
                        "confusion_matrix": result["confusion_matrix"]}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({
            "seed": seed,
            **{metric: result[metric] for metric in
               ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")},
            "ranks": ranks,
        }, indent=2), flush=True)

    frame = pd.DataFrame(rows)
    critical = float(t.ppf(0.975, df=len(frame) - 1))
    summary = {}
    for metric in ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5"):
        values = frame[metric].to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        half_width = critical * std / math.sqrt(len(values))
        summary[metric] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "min": float(values.min()),
            "max": float(values.max()),
            "rank_of_mean": int(1 + np.sum(
                baseline[metric].to_numpy(dtype=float) > mean
            )),
        }
    payload = {
        "model": "HGAN-Trace-Shared",
        "single_shared_encoder": True,
        "ensemble": False,
        "candidate": asdict(candidate),
        "fixed_epochs": fixed_epochs,
        "seeds": FINAL_SEEDS,
        "selection_used_test": False,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "summary": summary,
    }
    (OUT / "final_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("\nFINAL FIVE-SEED SUMMARY")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "final", "all"],
                        default="all")
    args = parser.parse_args()
    if args.phase in {"screen", "all"}:
        screen()
    if args.phase in {"final", "all"}:
        final()


if __name__ == "__main__":
    main()
