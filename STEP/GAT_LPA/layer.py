from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul


class LPAconv(MessagePassing):
    """
    Label Propagation-like convolution layer that repeatedly propagates
    node scores over the normalized graph structure.
    """

    def __init__(self, num_layers: int):
        super(LPAconv, self).__init__(aggr='add')
        self.num_layers = num_layers

    def forward(
        self,
        y: Tensor,
        edge_index: Adj,
        mask: Optional[Tensor] = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        y : Tensor
            Initial node scores (e.g., labels or probabilities).
        edge_index : Adj
            Graph connectivity in COO or SparseTensor form.
        mask : Optional[Tensor]
            Boolean mask indicating which nodes keep their original scores.
        edge_weight : OptTensor
            Edge weights for weighted propagation.

        Returns
        -------
        Tensor
            Node scores after num_layers rounds of propagation.
        """
        out = y
        if mask is not None:
            # Reset unlabeled nodes to zero while keeping labeled ones intact
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        # Normalize graph structure (no self-loops needed for pure propagation)
        edge_index = gcn_norm(edge_index, add_self_loops=False)

        for _ in range(self.num_layers):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """Scale neighbor messages by edge weights when provided."""
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor = None) -> Tensor:
        """Efficient propagation for SparseTensor adjacency."""
        return matmul(adj_t, x, reduce=self.aggr)