from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

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


class DATGTJoint(nn.Module):
    """Delay-aware typed graph Transformer with joint diagnostic heads."""

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


CANDIDATE = Candidate(name="typed_full_robust",preserve_network_features=True,model_dim=48,n_heads=4,transformer_layers=2,dropout=.25,input_noise=.02,root_weight=.30,normal_root_weight=.03,learning_rate=.0007,weight_decay=.0005,epochs=18)
