"""History-Enhanced Type Graph Transformer: neural graph detector."""
import math
import torch
from torch import nn


class TypedLocalBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, hidden, adjacency):
        neighbor = torch.einsum("ij,bjd->bid", adjacency, hidden)
        return self.norm(hidden + self.update(torch.cat([hidden, neighbor], -1)))


def _preserve_seed_sequence(dim):
    # Preserve the random-stream offsets of the paper's fitted checkpoints.
    # These scratch tensors are discarded; they are not model parameters.
    for fan_in, fan_out in ((dim * 2, dim), (dim, 1)):
        nn.init.kaiming_uniform_(torch.empty(fan_out, fan_in), a=math.sqrt(5))
        nn.init.uniform_(torch.empty(fan_out), -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))


class HETGT(nn.Module):
    def __init__(self, input_dim, n_nodes, n_net, n_classes, dim=48, heads=4,
                 layers=2, dropout=.25, input_noise=.02):
        super().__init__()
        if not 0 < n_net < n_nodes:
            raise ValueError("Both network and physical nodes are required")
        self.n_nodes, self.n_net = n_nodes, n_net
        self.input_noise = input_noise
        self.input_projection = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.type_embedding = nn.Parameter(torch.zeros(2, dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, n_nodes, dim))
        nn.init.normal_(self.type_embedding, std=.02)
        nn.init.normal_(self.position_embedding, std=.02)
        self.net_adapter = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.phy_adapter = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.adapter_norm = nn.LayerNorm(dim)
        self.local_blocks = nn.ModuleList([TypedLocalBlock(dim, dropout), TypedLocalBlock(dim, dropout)])
        self.graph_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.graph_token, std=.02)
        encoder = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 3,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.global_encoder = nn.TransformerEncoder(encoder, layers, norm=nn.LayerNorm(dim))
        _preserve_seed_sequence(dim)
        self.net_pool = nn.Linear(dim, 1)
        self.phy_pool = nn.Linear(dim, 1)
        self.detection_pool = nn.Linear(dim, 1)
        self.detection_norm = nn.LayerNorm(dim * 7)
        self.classifier = nn.Sequential(nn.Linear(dim * 7, dim * 3), nn.LayerNorm(dim * 3),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(dim, n_classes))

    @staticmethod
    def _attention_pool(nodes, scorer):
        weights = scorer(nodes).squeeze(-1).softmax(dim=1)
        return torch.sum(weights.unsqueeze(-1) * nodes, dim=1)

    def forward(self, features, adjacency):
        if self.training and self.input_noise > 0:
            features = features + torch.randn_like(features) * self.input_noise
        hidden = self.input_projection(features) + self.position_embedding
        net = hidden[:, :self.n_net] + self.type_embedding[0]
        phy = hidden[:, self.n_net:] + self.type_embedding[1]
        hidden = self.adapter_norm(torch.cat([net + self.net_adapter(net), phy + self.phy_adapter(phy)], 1))
        for block in self.local_blocks:
            hidden = block(hidden, adjacency)
        encoded = self.global_encoder(torch.cat([self.graph_token.expand(len(hidden), -1, -1), hidden], 1))
        graph, nodes = encoded[:, 0], encoded[:, 1:]
        net, phy = nodes[:, :self.n_net], nodes[:, self.n_net:]
        net_pool = self._attention_pool(net, self.net_pool)
        phy_pool = self._attention_pool(phy, self.phy_pool)
        source = self._attention_pool(nodes, self.detection_pool)
        representation = torch.cat([graph, net_pool, phy_pool, net.max(1).values,
                                    phy.max(1).values, source, torch.abs(net_pool - phy_pool)], -1)
        return self.classifier(self.detection_norm(representation))
