"""Validation-locked single-model HGAN detection and source-localization study.

The model has one shared typed graph encoder. A graph token produces class
logits, while a node head produces source scores. There is no detector
ensemble, checkpoint voting, or probability fusion.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

import reviewer_experiments as exp
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "single_hgan_joint_20260824"
DATA_PATH = ROOT / "sr_com_new.csv"
BASELINE_PATH = ROOT / "temporal_robust_20260823" / "final_80pct_comparison.csv"
HISTORY_LENGTH = 3
USE_TEMPORAL = True
SEED = 42
TRAIN_CAP_PER_CLASS = 0
VALIDATION_CAP_PER_CLASS = 0
TEST_CAP_PER_CLASS = 0
RECENCY_STRENGTH = 1.0
NORMAL_CLASS_WEIGHT = 1.0


@dataclass(frozen=True)
class Candidate:
    name: str
    preserve_network_features: bool
    model_dim: int
    n_heads: int
    transformer_layers: int
    dropout: float
    input_noise: float
    root_weight: float
    normal_root_weight: float
    learning_rate: float
    weight_decay: float
    epochs: int
    auc_margin_weight: float = 0.0
    auc_margin: float = 0.5
    ovr_bce_weight: float = 0.0
    use_input_residual: bool = False
    use_local_detection_readout: bool = False
    use_simple_detection_readout: bool = False
    decouple_detection_root: bool = False
    post_root_epochs: int = 0
    use_stable_graph_encoder: bool = False
    stable_no_self_neighbors: bool = False
    use_enhanced_detection_head: bool = False
    oversample: bool = False
    encoder_lr_scale: float = 1.0
    use_warm_restarts: bool = False
    detection_before_type_adapters: bool = False
    pretrain_epochs: int = 0
    use_hierarchical_classifier: bool = False
    min_selection_epoch: int = 1
    use_raw_context_residual: bool = False


CANDIDATES = [
    Candidate(
        name="typed_compact",
        preserve_network_features=False,
        model_dim=48,
        n_heads=4,
        transformer_layers=2,
        dropout=0.20,
        input_noise=0.01,
        root_weight=0.25,
        normal_root_weight=0.02,
        learning_rate=8e-4,
        weight_decay=2e-4,
        epochs=18,
    ),
    Candidate(
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
    ),
    Candidate(
        name="typed_full_discriminative",
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
        auc_margin_weight=0.05,
        auc_margin=0.5,
        ovr_bce_weight=0.15,
    ),
    Candidate(
        name="typed_full_residual",
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
        use_input_residual=True,
    ),
    Candidate(
        name="typed_full_residual_balanced",
        preserve_network_features=True,
        model_dim=48,
        n_heads=4,
        transformer_layers=2,
        dropout=0.20,
        input_noise=0.01,
        root_weight=0.15,
        normal_root_weight=0.02,
        learning_rate=8e-4,
        weight_decay=2e-4,
        epochs=18,
        use_input_residual=True,
    ),
    Candidate(
        name="typed_full_local_readout",
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
        use_local_detection_readout=True,
    ),
    Candidate(
        name="typed_full_local_balanced",
        preserve_network_features=True,
        model_dim=48,
        n_heads=4,
        transformer_layers=2,
        dropout=0.20,
        input_noise=0.01,
        root_weight=0.15,
        normal_root_weight=0.02,
        learning_rate=8e-4,
        weight_decay=2e-4,
        epochs=18,
        use_local_detection_readout=True,
    ),
    Candidate(
        name="typed_full_stable_readout",
        preserve_network_features=True,
        model_dim=48,
        n_heads=4,
        transformer_layers=2,
        dropout=0.30,
        input_noise=0.02,
        root_weight=0.0,
        normal_root_weight=0.03,
        learning_rate=7e-4,
        weight_decay=5e-4,
        epochs=18,
        use_local_detection_readout=True,
        use_simple_detection_readout=True,
        decouple_detection_root=True,
        post_root_epochs=5,
    ),
    Candidate(
        name="typed_full_stable_light",
        preserve_network_features=True,
        model_dim=48,
        n_heads=4,
        transformer_layers=2,
        dropout=0.20,
        input_noise=0.01,
        root_weight=0.0,
        normal_root_weight=0.02,
        learning_rate=8e-4,
        weight_decay=2e-4,
        epochs=18,
        use_local_detection_readout=True,
        use_simple_detection_readout=True,
        decouple_detection_root=True,
        post_root_epochs=5,
    ),
    Candidate(
        name="typed_stable_graph",
        preserve_network_features=True,
        model_dim=32,
        n_heads=4,
        transformer_layers=2,
        dropout=0.30,
        input_noise=0.0,
        root_weight=0.0,
        normal_root_weight=0.02,
        learning_rate=1e-3,
        weight_decay=2e-4,
        epochs=18,
        use_local_detection_readout=True,
        use_simple_detection_readout=True,
        decouple_detection_root=True,
        post_root_epochs=5,
        use_stable_graph_encoder=True,
    ),
    Candidate(
        name="typed_stable_graph_balanced",
        preserve_network_features=True,
        model_dim=32,
        n_heads=4,
        transformer_layers=2,
        dropout=0.30,
        input_noise=0.0,
        root_weight=0.0,
        normal_root_weight=0.02,
        learning_rate=2e-3,
        weight_decay=1e-4,
        epochs=10,
        use_local_detection_readout=True,
        use_simple_detection_readout=True,
        decouple_detection_root=True,
        post_root_epochs=5,
        use_stable_graph_encoder=True,
        stable_no_self_neighbors=True,
        use_enhanced_detection_head=True,
        oversample=True,
        encoder_lr_scale=0.5,
        use_warm_restarts=True,
    ),
    Candidate(
        name="typed_stable_graph_shared",
        preserve_network_features=True,
        model_dim=32,
        n_heads=4,
        transformer_layers=2,
        dropout=0.30,
        input_noise=0.0,
        root_weight=0.0,
        normal_root_weight=0.02,
        learning_rate=2e-3,
        weight_decay=1e-4,
        epochs=10,
        use_local_detection_readout=True,
        use_simple_detection_readout=True,
        decouple_detection_root=True,
        post_root_epochs=5,
        use_stable_graph_encoder=True,
        stable_no_self_neighbors=True,
        use_enhanced_detection_head=True,
        oversample=True,
        encoder_lr_scale=0.5,
        use_warm_restarts=True,
        detection_before_type_adapters=True,
        pretrain_epochs=1,
    ),
    Candidate(
        name="typed_stable_graph_hierarchical",
        preserve_network_features=True,
        model_dim=32,
        n_heads=4,
        transformer_layers=2,
        dropout=0.30,
        input_noise=0.0,
        root_weight=0.0,
        normal_root_weight=0.02,
        learning_rate=2e-3,
        weight_decay=1e-4,
        epochs=10,
        use_local_detection_readout=True,
        use_simple_detection_readout=True,
        decouple_detection_root=True,
        post_root_epochs=5,
        use_stable_graph_encoder=True,
        stable_no_self_neighbors=True,
        use_enhanced_detection_head=True,
        oversample=True,
        encoder_lr_scale=0.5,
        use_warm_restarts=True,
        detection_before_type_adapters=True,
        pretrain_epochs=1,
        use_hierarchical_classifier=True,
        min_selection_epoch=7,
    ),
    Candidate(
        name="typed_stable_graph_context_hierarchy",
        preserve_network_features=True,
        model_dim=32,
        n_heads=4,
        transformer_layers=2,
        dropout=0.30,
        input_noise=0.0,
        root_weight=0.0,
        normal_root_weight=0.02,
        learning_rate=2e-3,
        weight_decay=1e-4,
        epochs=10,
        use_local_detection_readout=True,
        use_simple_detection_readout=True,
        decouple_detection_root=True,
        post_root_epochs=5,
        use_stable_graph_encoder=True,
        stable_no_self_neighbors=True,
        use_enhanced_detection_head=True,
        oversample=True,
        encoder_lr_scale=0.5,
        use_warm_restarts=True,
        detection_before_type_adapters=True,
        pretrain_epochs=1,
        use_hierarchical_classifier=True,
        min_selection_epoch=7,
        use_raw_context_residual=True,
    ),
]


def set_seed(seed: int) -> None:
    tr.set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_builder(raw: dict, candidate: Candidate):
    return exp.make_builder(
        raw,
        history_length=HISTORY_LENGTH,
        temporal=USE_TEMPORAL,
        preserve_network_features=candidate.preserve_network_features,
    )


def static_adjacency(builder, candidate: Candidate) -> torch.Tensor:
    adjacency = builder.build_static_adjacency(
        use_cross_layer=False,
        use_phy_chain=True,
        use_net_edges=True,
    )
    if not candidate.stable_no_self_neighbors:
        adjacency = adjacency + np.eye(builder.n_nodes, dtype=np.float32)
    degree = adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
    return torch.tensor(adjacency / degree, dtype=torch.float32)


def build_cache(data: dict, builder) -> dict:
    features = []
    root_targets = np.zeros((len(data["y"]), builder.n_nodes), dtype=np.float32)
    root_mask = np.zeros(len(data["y"]), dtype=np.float32)
    for sample_idx in range(len(data["y"])):
        node_features = builder.get_node_features_with_history(
            sample_idx,
            data["X_net_feat"],
            data["X_phy"],
            data["src_ips"],
            data["dst_ips"],
        )
        features.append(node_features)
        roots = tr.infer_true_root_nodes(sample_idx, data, builder)
        if int(data["y"][sample_idx]) > 0 and roots:
            valid_roots = [idx for idx in roots if 0 <= int(idx) < builder.n_nodes]
            root_targets[sample_idx, valid_roots] = 1.0
            root_mask[sample_idx] = 1.0

    labels = np.asarray(data["y"], dtype=np.int64)
    recency = np.ones(len(labels), dtype=np.float32)
    for class_id in np.unique(labels):
        indices = np.where(labels == class_id)[0]
        if len(indices) > 1:
            progress = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
            weights = np.exp(RECENCY_STRENGTH * (progress - 1.0))
            recency[indices] = weights / max(float(weights.mean()), 1e-8)

    return {
        "features": torch.tensor(np.asarray(features), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "root_targets": torch.tensor(root_targets, dtype=torch.float32),
        "root_mask": torch.tensor(root_mask, dtype=torch.float32),
        "recency": torch.tensor(recency, dtype=torch.float32),
    }


class TypedLocalBlock(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor = torch.einsum("ij,bjd->bid", adjacency, hidden)
        update = self.update(torch.cat([hidden, neighbor], dim=-1))
        return self.norm(hidden + update)


class SingleHGANJoint(nn.Module):
    """One typed graph encoder with classification and source-scoring heads."""

    def __init__(self, input_dim: int, n_nodes: int, n_net: int,
                 n_classes: int, candidate: Candidate):
        super().__init__()
        dim = candidate.model_dim
        self.n_nodes = int(n_nodes)
        self.n_net = int(n_net)
        self.n_phy = self.n_nodes - self.n_net
        self.net_raw_offset = 6
        self.net_feature_dim = max(int(input_dim) - self.net_raw_offset, 0)
        self.input_noise = float(candidate.input_noise)
        self.dropout = float(candidate.dropout)
        self.use_input_residual = bool(candidate.use_input_residual)
        self.use_local_detection_readout = bool(
            candidate.use_local_detection_readout
        )
        self.use_simple_detection_readout = bool(
            candidate.use_simple_detection_readout
        )
        self.decouple_detection_root = bool(candidate.decouple_detection_root)
        self.use_stable_graph_encoder = bool(candidate.use_stable_graph_encoder)
        self.use_enhanced_detection_head = bool(
            candidate.use_enhanced_detection_head
        )
        self.detection_before_type_adapters = bool(
            candidate.detection_before_type_adapters
        )
        self.use_hierarchical_classifier = bool(
            candidate.use_hierarchical_classifier
        )
        self.use_raw_context_residual = bool(
            candidate.use_raw_context_residual
        )
        if self.use_hierarchical_classifier and int(n_classes) != 6:
            raise ValueError("The IGCPS hierarchy requires exactly six classes")

        if self.use_stable_graph_encoder:
            stable_hidden = dim * 2
            self.stable_conv1 = nn.Linear(input_dim * 2, stable_hidden)
            self.stable_conv2 = nn.Linear(stable_hidden * 2, dim)
            self.input_projection = None
        else:
            self.stable_conv1 = None
            self.stable_conv2 = None
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
        self.type_embedding = nn.Parameter(torch.zeros(2, dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, n_nodes, dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

        self.net_adapter = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.phy_adapter = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.adapter_norm = nn.LayerNorm(dim)
        self.local_blocks = nn.ModuleList([] if self.use_stable_graph_encoder else [
            TypedLocalBlock(dim, candidate.dropout),
            TypedLocalBlock(dim, candidate.dropout),
        ])

        self.graph_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.graph_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=candidate.n_heads,
            dim_feedforward=dim * 3,
            dropout=candidate.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=candidate.transformer_layers,
            norm=nn.LayerNorm(dim),
        )

        self.root_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(candidate.dropout),
            nn.Linear(dim, 1),
        )
        self.net_pool = nn.Linear(dim, 1)
        self.phy_pool = nn.Linear(dim, 1)
        self.detection_pool = nn.Linear(dim, 1)
        if self.use_enhanced_detection_head:
            self.detection_enhance = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.20),
            )
            self.detection_graph_pool = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh(),
                nn.Linear(dim, 1),
            )
        else:
            self.detection_enhance = None
            self.detection_graph_pool = None
        if self.use_simple_detection_readout:
            detection_width = 1
        else:
            detection_width = 11 if self.use_input_residual else 7
        if self.use_enhanced_detection_head:
            self.detection_norm = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(dim * detection_width, dim * 4),
                nn.LayerNorm(dim * 4),
                nn.GELU(),
                nn.Dropout(candidate.dropout),
                nn.Linear(dim * 4, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(dim * 2, n_classes),
            )
        else:
            self.detection_norm = nn.LayerNorm(dim * detection_width)
            self.classifier = nn.Sequential(
                nn.Linear(dim * detection_width, dim * 3),
                nn.LayerNorm(dim * 3),
                nn.GELU(),
                nn.Dropout(candidate.dropout),
                nn.Linear(dim * 3, dim),
                nn.GELU(),
                nn.Dropout(candidate.dropout / 2),
                nn.Linear(dim, n_classes),
            )
        if self.use_hierarchical_classifier:
            context_dim = (
                self.net_feature_dim + self.n_phy + 2 * self.n_net
                if self.use_raw_context_residual else 0
            )
            hierarchy_input_dim = dim * 7 + context_dim
            self.hierarchy_norm = nn.LayerNorm(hierarchy_input_dim)
            self.classification_projector = nn.Sequential(
                nn.Linear(hierarchy_input_dim, dim * 3),
                nn.LayerNorm(dim * 3),
                nn.GELU(),
                nn.Dropout(candidate.dropout),
                nn.Linear(dim * 3, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            self.anomaly_gate = nn.Linear(dim, 1)
            self.origin_gate = nn.Linear(dim, 1)
            self.physical_type_output = nn.Linear(dim, 2)
            self.cyber_type_output = nn.Linear(dim, 3)
        else:
            self.hierarchy_norm = None
            self.classification_projector = None
            self.anomaly_gate = None
            self.origin_gate = None
            self.physical_type_output = None
            self.cyber_type_output = None

    @staticmethod
    def _attention_pool(nodes: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
        weights = torch.softmax(scorer(nodes).squeeze(-1), dim=1)
        return torch.sum(weights.unsqueeze(-1) * nodes, dim=1)

    def encode_stable(self, features: torch.Tensor,
                      adjacency: torch.Tensor) -> torch.Tensor:
        if not self.use_stable_graph_encoder:
            raise RuntimeError("Stable encoding is unavailable for this candidate")
        neighbor = torch.einsum("ij,bjf->bif", adjacency, features)
        hidden = F.relu(self.stable_conv1(
            torch.cat([features, neighbor], dim=-1)
        ))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        neighbor = torch.einsum("ij,bjd->bid", adjacency, hidden)
        return self.stable_conv2(torch.cat([hidden, neighbor], dim=-1))

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> dict:
        if self.training and self.input_noise > 0:
            features = features + torch.randn_like(features) * self.input_noise
        if self.use_stable_graph_encoder:
            hidden = self.encode_stable(features, adjacency)
            input_hidden = hidden
        else:
            hidden = self.input_projection(features)
            input_hidden = hidden
        hidden = hidden + self.position_embedding
        net = hidden[:, :self.n_net] + self.type_embedding[0]
        phy = hidden[:, self.n_net:] + self.type_embedding[1]
        net = net + self.net_adapter(net)
        phy = phy + self.phy_adapter(phy)
        hidden = self.adapter_norm(torch.cat([net, phy], dim=1))
        for block in self.local_blocks:
            hidden = block(hidden, adjacency)
        local_nodes = hidden

        token = self.graph_token.expand(hidden.shape[0], -1, -1)
        encoded = self.global_encoder(torch.cat([token, hidden], dim=1))
        graph = encoded[:, 0]
        nodes = encoded[:, 1:]
        graph_context = graph.unsqueeze(1).expand(-1, self.n_nodes, -1)
        root_logits = self.root_scorer(
            torch.cat([nodes, graph_context], dim=-1)
        ).squeeze(-1)

        if self.detection_before_type_adapters:
            detection_nodes = input_hidden
        else:
            detection_nodes = local_nodes if self.use_local_detection_readout else nodes
        if self.use_local_detection_readout:
            graph = detection_nodes.mean(dim=1)
        if self.use_enhanced_detection_head:
            detection_nodes = detection_nodes + self.detection_enhance(
                detection_nodes
            )
        if self.use_hierarchical_classifier:
            net_nodes = detection_nodes[:, :self.n_net]
            phy_nodes = detection_nodes[:, self.n_net:]
            graph_repr = self._attention_pool(
                detection_nodes, self.detection_graph_pool
            ) + 0.3 * detection_nodes.max(dim=1).values
            net_repr = self._attention_pool(net_nodes, self.net_pool)
            phy_repr = self._attention_pool(phy_nodes, self.phy_pool)
            net_max = net_nodes.max(dim=1).values
            phy_max = phy_nodes.max(dim=1).values
            discrepancy = torch.abs(net_repr - phy_repr)
            interaction = net_repr * phy_repr
            fused = torch.cat(
                [
                    graph_repr, net_repr, phy_repr, net_max, phy_max,
                    discrepancy, interaction,
                ],
                dim=-1,
            )
            if self.use_raw_context_residual:
                role = features[:, :self.n_net, 4]
                active = (torch.abs(role) > 0.5).to(features.dtype).unsqueeze(-1)
                raw_end = self.net_raw_offset + self.net_feature_dim
                net_raw = features[
                    :, :self.n_net, self.net_raw_offset:raw_end
                ]
                active_count = active.sum(dim=1).clamp_min(1.0)
                net_context = (net_raw * active).sum(dim=1) / active_count
                phy_context = features[
                    :, self.n_net:self.n_net + self.n_phy, 0
                ]
                source_identity = torch.clamp(role, min=0.0, max=1.0)
                destination_identity = torch.clamp(-role, min=0.0, max=1.0)
                context = torch.cat(
                    [
                        net_context, phy_context,
                        source_identity, destination_identity,
                    ],
                    dim=-1,
                )
                fused = torch.cat([fused, context], dim=-1)
            classification_embedding = self.classification_projector(
                self.hierarchy_norm(fused)
            )
            anomaly_logit = self.anomaly_gate(
                classification_embedding
            ).squeeze(-1)
            origin_logit = self.origin_gate(
                classification_embedding
            ).squeeze(-1)
            physical_types = F.log_softmax(
                self.physical_type_output(classification_embedding), dim=-1
            )
            cyber_types = F.log_softmax(
                self.cyber_type_output(classification_embedding), dim=-1
            )
            normal_log = F.logsigmoid(-anomaly_logit).unsqueeze(-1)
            anomaly_log = F.logsigmoid(anomaly_logit).unsqueeze(-1)
            physical_log = F.logsigmoid(origin_logit).unsqueeze(-1)
            cyber_log = F.logsigmoid(-origin_logit).unsqueeze(-1)
            class_log_probabilities = torch.cat(
                [
                    normal_log,
                    anomaly_log + physical_log + physical_types,
                    anomaly_log + cyber_log + cyber_types,
                ],
                dim=-1,
            )
            return {
                "anomaly_logits": class_log_probabilities,
                "root_logits": root_logits,
                "embeddings": nodes,
                "classification_embedding": classification_embedding,
            }
        if self.use_simple_detection_readout:
            pool = (
                self.detection_graph_pool
                if self.use_enhanced_detection_head else self.detection_pool
            )
            pooled = self._attention_pool(detection_nodes, pool)
            detection = pooled + 0.3 * detection_nodes.max(dim=1).values
            class_logits = self.classifier(self.detection_norm(detection))
            return {
                "anomaly_logits": class_logits,
                "root_logits": root_logits,
                "embeddings": nodes,
            }
        net_nodes = detection_nodes[:, :self.n_net]
        phy_nodes = detection_nodes[:, self.n_net:]
        net_repr = self._attention_pool(net_nodes, self.net_pool)
        phy_repr = self._attention_pool(phy_nodes, self.phy_pool)
        net_max = net_nodes.max(dim=1).values
        phy_max = phy_nodes.max(dim=1).values
        if self.decouple_detection_root:
            source_weights = torch.softmax(
                self.detection_pool(detection_nodes).squeeze(-1), dim=1
            )
        else:
            source_weights = torch.softmax(root_logits.detach(), dim=1)
        source_repr = torch.sum(
            source_weights.unsqueeze(-1) * detection_nodes, dim=1
        )
        discrepancy = torch.abs(net_repr - phy_repr)
        detection_parts = [
            graph, net_repr, phy_repr, net_max, phy_max, source_repr, discrepancy,
        ]
        if self.use_input_residual:
            input_net = input_hidden[:, :self.n_net]
            input_phy = input_hidden[:, self.n_net:]
            detection_parts.extend([
                input_net.mean(dim=1),
                input_phy.mean(dim=1),
                input_net.max(dim=1).values,
                input_phy.max(dim=1).values,
            ])
        detection = torch.cat(detection_parts, dim=-1)
        class_logits = self.classifier(self.detection_norm(detection))
        return {
            "anomaly_logits": class_logits,
            "root_logits": root_logits,
            "embeddings": nodes,
        }


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels)
    weights = torch.sqrt(counts.max().float() / counts.clamp_min(1).float())
    return weights / weights.min()


def macro_auc_margin_loss(logits: torch.Tensor, labels: torch.Tensor,
                          margin: float = 0.5) -> torch.Tensor:
    """Equal-class pairwise ranking surrogate for macro one-vs-rest AUC."""
    losses = []
    for class_id in range(logits.shape[1]):
        positive = logits[labels == class_id, class_id]
        negative = logits[labels != class_id, class_id]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        differences = positive.unsqueeze(1) - negative.unsqueeze(0)
        losses.append(F.softplus(float(margin) - differences).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def macro_ovr_bce_loss(logits: torch.Tensor,
                       labels: torch.Tensor) -> torch.Tensor:
    """Balance positive and negative BCE terms independently for every class."""
    losses = []
    for class_id in range(logits.shape[1]):
        targets = (labels == class_id).to(logits.dtype)
        positive = targets > 0
        negative = ~positive
        if not torch.any(positive) or not torch.any(negative):
            continue
        per_sample = F.binary_cross_entropy_with_logits(
            logits[:, class_id], targets, reduction="none"
        )
        losses.append(
            0.5 * per_sample[positive].mean()
            + 0.5 * per_sample[negative].mean()
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def root_objective(root_logits: torch.Tensor, roots: torch.Tensor,
                   root_mask: torch.Tensor, normal_weight: float) -> torch.Tensor:
    node_weights = torch.ones_like(roots)
    positive_counts = roots.sum(dim=1, keepdim=True).clamp_min(1.0)
    node_weights = torch.where(
        roots > 0,
        torch.maximum(
            torch.full_like(roots, 5.0),
            torch.full_like(roots, roots.shape[1]) / positive_counts,
        ),
        node_weights,
    )
    per_sample = F.binary_cross_entropy_with_logits(
        root_logits, roots, weight=node_weights, reduction="none"
    ).mean(dim=1)
    anomaly_denominator = root_mask.sum().clamp_min(1.0)
    anomaly_root = torch.sum(per_sample * root_mask) / anomaly_denominator
    normal_mask = 1.0 - root_mask
    normal_denominator = normal_mask.sum().clamp_min(1.0)
    normal_root = torch.sum(per_sample * normal_mask) / normal_denominator
    return anomaly_root + float(normal_weight) * normal_root


def compute_loss(output: dict, labels: torch.Tensor, roots: torch.Tensor,
                 root_mask: torch.Tensor, recency: torch.Tensor,
                 weights: torch.Tensor, candidate: Candidate) -> tuple[torch.Tensor, dict]:
    classification = F.cross_entropy(
        output["anomaly_logits"], labels, weight=weights, reduction="none"
    )
    classification = torch.mean(classification * recency)

    root_loss = root_objective(
        output["root_logits"], roots, root_mask, candidate.normal_root_weight
    )
    auc_margin = macro_auc_margin_loss(
        output["anomaly_logits"], labels, candidate.auc_margin
    )
    ovr_bce = macro_ovr_bce_loss(output["anomaly_logits"], labels)
    total = (
        classification
        + candidate.root_weight * root_loss
        + candidate.auc_margin_weight * auc_margin
        + candidate.ovr_bce_weight * ovr_bce
    )
    return total, {
        "classification": float(classification.detach()),
        "root": float(root_loss.detach()),
        "auc_margin": float(auc_margin.detach()),
        "ovr_bce": float(ovr_bce.detach()),
    }


def evaluate(model: nn.Module, cache: dict, source_data: dict,
             builder, adjacency: torch.Tensor, batch_size: int = 128) -> dict:
    dataset = TensorDataset(
        cache["features"], cache["labels"], cache["root_targets"],
        cache["root_mask"], cache["recency"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities = []
    predictions = []
    root_scores = []
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for features, _, _, _, _ in loader:
            output = model(features, adjacency)
            probs = torch.softmax(output["anomaly_logits"], dim=-1)
            probabilities.append(probs.cpu().numpy())
            predictions.append(probs.argmax(dim=-1).cpu().numpy())
            root_scores.append(torch.sigmoid(output["root_logits"]).cpu().numpy())
    elapsed = (time.perf_counter() - started) * 1000 / max(len(dataset), 1)
    probabilities = np.concatenate(probabilities)
    predictions = np.concatenate(predictions)
    root_scores = np.concatenate(root_scores)
    labels = cache["labels"].numpy()
    encoded = label_binarize(labels, classes=range(probabilities.shape[1]))
    per_class_auc = [
        roc_auc_score(encoded[:, class_id], probabilities[:, class_id]) * 100
        for class_id in range(probabilities.shape[1])
    ]
    auc = float(np.mean(per_class_auc))
    root = tr.compute_strict_traceback_metrics(root_scores, source_data, builder)
    return {
        "accuracy": accuracy_score(labels, predictions) * 100,
        "f1": f1_score(labels, predictions, average="macro", zero_division=0) * 100,
        "auc": auc,
        "per_class_auc": per_class_auc,
        "min_class_auc": float(min(per_class_auc)),
        "top1": root["rca"],
        "mrr": root["mrr"],
        "ndcg5": root["ndcg"],
        "mean_reference_rank": root["apd"],
        "inference_ms": elapsed,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def classwise_late_indices(labels: np.ndarray) -> np.ndarray:
    selected = []
    for class_id in np.unique(labels):
        indices = np.where(labels == class_id)[0]
        selected.extend(indices[len(indices) // 2:].tolist())
    return np.asarray(sorted(selected), dtype=int)


def slice_cache(cache: dict, indices: np.ndarray) -> dict:
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return {key: value.index_select(0, tensor_indices) for key, value in cache.items()}


def train_post_root_scorer(model: SingleHGANJoint, cache: dict,
                           adjacency: torch.Tensor, candidate: Candidate,
                           seed: int) -> list[float]:
    if candidate.post_root_epochs <= 0:
        return []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.root_scorer.parameters():
        parameter.requires_grad_(True)
    dataset = TensorDataset(
        cache["features"], cache["root_targets"], cache["root_mask"]
    )
    generator = torch.Generator().manual_seed(seed + 101)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True, generator=generator, drop_last=False
    )
    optimizer = torch.optim.AdamW(
        model.root_scorer.parameters(), lr=2e-3, weight_decay=1e-4
    )
    history = []
    model.eval()
    for epoch in range(int(candidate.post_root_epochs)):
        model.root_scorer.train()
        total = 0.0
        for features, roots, root_mask in loader:
            output = model(features, adjacency)
            loss = root_objective(
                output["root_logits"], roots, root_mask,
                candidate.normal_root_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.root_scorer.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
        average = total / max(len(dataset), 1)
        history.append(average)
        print(
            f"{candidate.name} root epoch {epoch + 1:02d}/"
            f"{candidate.post_root_epochs}: loss={average:.4f}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()
    return history


def pretrain_stable_encoder(model: SingleHGANJoint, cache: dict, builder,
                            adjacency: torch.Tensor, candidate: Candidate,
                            seed: int) -> list[float]:
    if candidate.pretrain_epochs <= 0:
        return []
    target = torch.tensor(
        builder.build_static_adjacency(
            use_cross_layer=False, use_phy_chain=True, use_net_edges=True,
        ),
        dtype=torch.float32,
    )
    dataset = TensorDataset(cache["features"])
    generator = torch.Generator().manual_seed(seed + 17)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True, generator=generator,
        drop_last=False,
    )
    parameters = [
        *model.stable_conv1.parameters(), *model.stable_conv2.parameters()
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=candidate.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(candidate.pretrain_epochs), 1)
    )
    history = []
    for epoch in range(int(candidate.pretrain_epochs)):
        model.train()
        total = 0.0
        for (features,) in loader:
            embeddings = model.encode_stable(features, adjacency)
            reconstructed = torch.sigmoid(torch.bmm(
                embeddings, embeddings.transpose(1, 2)
            ))
            loss = F.binary_cross_entropy(
                reconstructed,
                target.unsqueeze(0).expand(len(features), -1, -1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
        scheduler.step()
        average = total / max(len(dataset), 1)
        history.append(average)
        print(
            f"{candidate.name} pretrain epoch {epoch + 1:02d}/"
            f"{candidate.pretrain_epochs}: edge_loss={average:.4f}"
        )
    return history


def train_model(candidate: Candidate, train_cache: dict, validation_cache: dict,
                validation_data: dict, builder, adjacency: torch.Tensor,
                fixed_epochs: int | None = None,
                seed: int = SEED) -> tuple[nn.Module, list[dict], int]:
    set_seed(seed)
    model = SingleHGANJoint(
        input_dim=train_cache["features"].shape[-1],
        n_nodes=builder.n_nodes,
        n_net=builder.n_net,
        n_classes=int(train_cache["labels"].max().item() + 1),
        candidate=candidate,
    )
    pretrain_history = pretrain_stable_encoder(
        model, train_cache, builder, adjacency, candidate, seed
    )
    weights = class_weights(train_cache["labels"])
    if len(weights) > 0:
        weights[0] *= NORMAL_CLASS_WEIGHT
    training_indices = np.arange(len(train_cache["labels"]), dtype=np.int64)
    if candidate.oversample:
        labels_np = train_cache["labels"].numpy()
        class_counts = np.bincount(labels_np)
        expanded = []
        minority_target = 200 if len(labels_np) < 1000 else 1000
        for class_id, count in enumerate(class_counts):
            class_indices = np.where(labels_np == class_id)[0]
            if count < minority_target:
                repeats = max(1, minority_target // max(int(count), 1))
                expanded.extend(np.repeat(class_indices, repeats).tolist())
            else:
                expanded.extend(class_indices.tolist())
        training_indices = np.asarray(expanded, dtype=np.int64)
        sampled_counts = np.bincount(
            labels_np[training_indices], minlength=len(class_counts)
        )
        print(
            f"{candidate.name}: oversampled {len(labels_np)} -> "
            f"{len(training_indices)} samples; counts={sampled_counts.tolist()}"
        )
    tensor_indices = torch.tensor(training_indices, dtype=torch.long)
    dataset = TensorDataset(
        train_cache["features"].index_select(0, tensor_indices),
        train_cache["labels"].index_select(0, tensor_indices),
        train_cache["root_targets"].index_select(0, tensor_indices),
        train_cache["root_mask"].index_select(0, tensor_indices),
        train_cache["recency"].index_select(0, tensor_indices),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True, generator=generator, drop_last=False
    )
    detection_terms = (
        "classifier", "detection_norm", "detection_pool",
        "detection_enhance", "detection_graph_pool", "net_pool", "phy_pool",
        "hierarchy_norm", "classification_projector", "anomaly_gate",
        "origin_gate", "physical_type_output", "cyber_type_output",
    )
    encoder_parameters = []
    detection_parameters = []
    for name, parameter in model.named_parameters():
        if any(term in name for term in detection_terms):
            detection_parameters.append(parameter)
        else:
            encoder_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": candidate.learning_rate * candidate.encoder_lr_scale,
            },
            {"params": detection_parameters, "lr": candidate.learning_rate},
        ],
        weight_decay=candidate.weight_decay,
    )
    epochs = int(fixed_epochs or candidate.epochs)
    if candidate.use_warm_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(candidate.epochs), 1),
            eta_min=candidate.learning_rate * 0.1,
        )
    late_indices = classwise_late_indices(validation_cache["labels"].numpy())
    late_cache = slice_cache(validation_cache, late_indices)
    late_data = tr.subset_data(validation_data, late_indices)

    best_state = None
    best_epoch = epochs
    best_key = None
    history = [
        {
            "stage": "pretrain",
            "pretrain_epochs": int(candidate.pretrain_epochs),
            "pretrain_final_loss": float(pretrain_history[-1]),
        }
    ] if pretrain_history else []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for features, labels, roots, root_mask, recency in loader:
            output = model(features, adjacency)
            loss, _ = compute_loss(
                output, labels, roots, root_mask, recency,
                weights, candidate,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        scheduler.step()

        full = evaluate(model, validation_cache, validation_data, builder, adjacency)
        late = evaluate(model, late_cache, late_data, builder, adjacency)
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / len(dataset),
            **{f"full_{key}": value for key, value in full.items()
               if key != "confusion_matrix"},
            **{f"late_{key}": value for key, value in late.items()
               if key != "confusion_matrix"},
        }
        history.append(row)
        if candidate.post_root_epochs > 0:
            selection_key = (
                full["accuracy"], full["f1"], full["auc"], late["f1"]
            )
        else:
            selection_key = (
                full["accuracy"],
                full["f1"],
                full["auc"],
                full["top1"],
                full["mrr"],
                full["ndcg5"],
                late["f1"],
            )
        eligible = fixed_epochs is not None or epoch + 1 >= max(
            int(candidate.min_selection_epoch), 1
        )
        if eligible and (best_key is None or selection_key > best_key):
            best_key = selection_key
            best_epoch = epoch + 1
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"{candidate.name} epoch {epoch + 1:02d}/{epochs}: "
            f"loss={row['loss']:.4f}, full F1/AUC/Top1="
            f"{full['f1']:.2f}/{full['auc']:.2f}/{full['top1']:.2f}, "
            f"late={late['f1']:.2f}/{late['auc']:.2f}/{late['top1']:.2f}"
        )

    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    root_history = train_post_root_scorer(
        model, train_cache, adjacency, candidate, seed
    )
    if root_history:
        history.append({
            "stage": "post_root",
            "root_epochs": int(candidate.post_root_epochs),
            "root_final_loss": float(root_history[-1]),
        })
    return model, history, best_epoch


def run_validation() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = exp.cap_protocol(
        exp.load_protocol(DATA_PATH),
        TRAIN_CAP_PER_CLASS,
        VALIDATION_CAP_PER_CLASS,
        TEST_CAP_PER_CLASS,
        seed=SEED,
    )
    rows = []
    candidate_results = []
    for candidate in CANDIDATES:
        print(f"\nValidation candidate: {candidate.name}")
        builder = make_builder(protocol.raw, candidate)
        adjacency = static_adjacency(builder, candidate)
        train_cache = build_cache(protocol.train, builder)
        validation_cache = build_cache(protocol.val, builder)
        model, history, best_epoch = train_model(
            candidate, train_cache, validation_cache, protocol.val,
            builder, adjacency,
        )
        result = evaluate(model, validation_cache, protocol.val, builder, adjacency)
        late_indices = classwise_late_indices(protocol.val["y"])
        late_result = evaluate(
            model,
            slice_cache(validation_cache, late_indices),
            tr.subset_data(protocol.val, late_indices),
            builder,
            adjacency,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "candidate": asdict(candidate),
                "best_epoch": best_epoch,
            },
            OUT / f"validation__{candidate.name}.pt",
        )
        pd.DataFrame(history).to_csv(
            OUT / f"validation_history__{candidate.name}.csv", index=False
        )
        record = {
            "candidate": candidate.name,
            "best_epoch": best_epoch,
            **{f"full_{key}": value for key, value in result.items()
               if key != "confusion_matrix"},
            **{f"late_{key}": value for key, value in late_result.items()
               if key != "confusion_matrix"},
        }
        rows.append(record)
        candidate_results.append((candidate, best_epoch, result, late_result))

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "validation_candidates.csv", index=False)
    selected = max(
        candidate_results,
        key=lambda item: (
            item[2]["accuracy"],
            item[2]["f1"],
            item[2]["auc"],
            item[2]["top1"],
            item[2]["mrr"],
            item[2]["ndcg5"],
            item[3]["f1"],
        ),
    )
    candidate, best_epoch, result, late_result = selected
    lock = {
        "selection_data": "class-wise chronological 60--80% validation only",
        "test_used_for_selection": False,
        "selection_rule": [
            "maximize full-validation accuracy",
            "then full-validation macro F1 and macro one-vs-rest AUC",
            "then full-validation Top-1, MRR, and NDCG@5",
            "then late-validation macro F1 as a stability tie-breaker",
        ],
        "selected_candidate": asdict(candidate),
        "selected_epoch": best_epoch,
        "data_path": str(DATA_PATH.resolve()),
        "history_length": HISTORY_LENGTH,
        "temporal_history_enabled": USE_TEMPORAL,
        "train_cap_per_class": TRAIN_CAP_PER_CLASS,
        "validation_cap_per_class": VALIDATION_CAP_PER_CLASS,
        "test_cap_per_class": TEST_CAP_PER_CLASS,
        "recency_strength": RECENCY_STRENGTH,
        "normal_class_weight": NORMAL_CLASS_WEIGHT,
        "validation": result,
        "late_validation": late_result,
    }
    (OUT / "validation_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(json.dumps(lock, indent=2))


def run_final() -> None:
    lock_path = OUT / "validation_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("Run validation before final evaluation.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    candidate = Candidate(**lock["selected_candidate"])
    fixed_epochs = int(lock["selected_epoch"])
    if str(DATA_PATH.resolve()) != lock.get("data_path", str(DATA_PATH.resolve())):
        raise ValueError("Final-stage CSV differs from the validation-locked CSV")
    if bool(USE_TEMPORAL) != bool(lock.get("temporal_history_enabled", USE_TEMPORAL)):
        raise ValueError("Final-stage temporal mode differs from validation")
    for key, value in {
        "train_cap_per_class": TRAIN_CAP_PER_CLASS,
        "validation_cap_per_class": VALIDATION_CAP_PER_CLASS,
        "test_cap_per_class": TEST_CAP_PER_CLASS,
    }.items():
        if int(lock.get(key, value)) != int(value):
            raise ValueError(f"Final-stage {key} differs from validation")
    if not np.isclose(
        float(lock.get("recency_strength", RECENCY_STRENGTH)),
        float(RECENCY_STRENGTH),
    ):
        raise ValueError("Final-stage recency_strength differs from validation")
    if not np.isclose(
        float(lock.get("normal_class_weight", NORMAL_CLASS_WEIGHT)),
        float(NORMAL_CLASS_WEIGHT),
    ):
        raise ValueError("Final-stage normal_class_weight differs from validation")
    protocol = exp.load_protocol(DATA_PATH)
    development, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    development = tr.cap_split_per_class(
        development, TRAIN_CAP_PER_CLASS, seed=SEED, split_name="Development"
    )
    test = tr.cap_split_per_class(
        test, TEST_CAP_PER_CLASS, seed=SEED + 2, split_name="Test"
    )
    builder = make_builder(protocol.raw, candidate)
    adjacency = static_adjacency(builder, candidate)
    development_cache = build_cache(development, builder)
    test_cache = build_cache(test, builder)
    model, history, _ = train_model(
        candidate,
        development_cache,
        development_cache,
        development,
        builder,
        adjacency,
        fixed_epochs=fixed_epochs,
    )
    result = evaluate(model, test_cache, test, builder, adjacency)
    result["model"] = "HGAN-Trace-Joint"
    result["split"] = "class-wise chronological first 80% refit / last 20% evaluation"
    result["selection_used_test"] = False
    result["candidate"] = asdict(candidate)
    result["fixed_epochs"] = fixed_epochs
    result["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    result["ranks"] = {}
    if BASELINE_PATH.exists():
        baseline = pd.read_csv(BASELINE_PATH)
        for metric in ["accuracy", "f1", "auc", "top1", "mrr", "ndcg5"]:
            result["ranks"][metric] = int(
                1 + np.sum(baseline[metric].to_numpy(dtype=float) > float(result[metric]))
            )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "candidate": asdict(candidate),
            "fixed_epochs": fixed_epochs,
        },
        OUT / "HGAN-Trace-Joint__seed42.pt",
    )
    pd.DataFrame(history).to_csv(OUT / "final_history.csv", index=False)
    (OUT / "final_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    global OUT, DATA_PATH, BASELINE_PATH, HISTORY_LENGTH, USE_TEMPORAL
    global TRAIN_CAP_PER_CLASS, VALIDATION_CAP_PER_CLASS, TEST_CAP_PER_CLASS
    global RECENCY_STRENGTH
    global NORMAL_CLASS_WEIGHT
    global CANDIDATES
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validation", "final"], default="validation")
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--outdir", type=Path, default=OUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--history-length", type=int, default=HISTORY_LENGTH)
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--train-cap-per-class", type=int, default=0)
    parser.add_argument("--validation-cap-per-class", type=int, default=0)
    parser.add_argument("--test-cap-per-class", type=int, default=0)
    parser.add_argument("--recency-strength", type=float, default=RECENCY_STRENGTH)
    parser.add_argument("--normal-class-weight", type=float, default=NORMAL_CLASS_WEIGHT)
    parser.add_argument(
        "--candidates", nargs="+",
        choices=[candidate.name for candidate in CANDIDATES],
        default=[candidate.name for candidate in CANDIDATES],
    )
    args = parser.parse_args()
    if args.history_length < 1:
        parser.error("--history-length must be at least 1")
    OUT = args.outdir
    DATA_PATH = args.csv
    BASELINE_PATH = args.baseline
    HISTORY_LENGTH = args.history_length
    USE_TEMPORAL = not args.current_only
    TRAIN_CAP_PER_CLASS = max(0, args.train_cap_per_class)
    VALIDATION_CAP_PER_CLASS = max(0, args.validation_cap_per_class)
    TEST_CAP_PER_CLASS = max(0, args.test_cap_per_class)
    RECENCY_STRENGTH = max(0.0, args.recency_strength)
    NORMAL_CLASS_WEIGHT = max(0.0, args.normal_class_weight)
    selected_candidates = set(args.candidates)
    CANDIDATES = [
        candidate for candidate in CANDIDATES
        if candidate.name in selected_candidates
    ]
    if args.stage == "validation":
        run_validation()
    else:
        run_final()


if __name__ == "__main__":
    main()
