import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .layer import LPAconv


class GAT_LPA(nn.Module):
    """
    Graph Attention Network followed by optional Label Propagation (LPA) smoothing.
    The learned edge-weight parameter allows the model to reweight all edges uniformly.
    """

    def __init__(
        self,
        in_feature: int,
        hidden: int,
        out_feature: int,
        dropout: float,
        num_edges: int,
        lpaiters: int,
        gat_heads: int,
        gatnum: int,
    ):
        super(GAT_LPA, self).__init__()

        # Single learnable scalar per edge; shared across all attention layers
        self.edge_weight = nn.Parameter(torch.ones(num_edges))

        # Build a stack of GAT layers
        gc = nn.ModuleList()
        gc.append(GATConv(in_feature, hidden, heads=gat_heads, concat=True))

        for _ in range(gatnum - 2):
            gc.append(GATConv(hidden * gat_heads, hidden, heads=gat_heads, concat=True))

        gc.append(GATConv(hidden * gat_heads, out_feature, heads=1, concat=False))
        self.gc = gc

        self.lpa = LPAconv(lpaiters)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Run stacked GAT layers with dropout and ReLU activations.
        """
        for layer in self.gc[:-1]:
            x = layer(x, edge_index, self.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.gc[-1](x, edge_index, self.edge_weight)
        return x


class MLP(nn.Module):
    """
    Simple two-layer MLP baseline for node-level embeddings.
    """

    def __init__(self, in_feature: int, hidden: int, out_feature: int, dropout: float):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden)
        self.fc2 = nn.Linear(hidden, out_feature)
        self.relu = nn.ReLU()
        self.dropout_rate = dropout

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass that expects a `data` object with an `.x` attribute.
        """
        x = data.x
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x