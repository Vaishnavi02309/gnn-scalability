"""Purpose: GraphSAGE model for node classification.

GraphSAGE uses message passing:
- Takes node features x
- Aggregates neighbor info using edge_index
- Outputs logits for node classification
"""

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 2, "Use at least 2 layers for this baseline."
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            """GraphSAGE message passing (core GNN equation)
            Eq- (GraphSAGE mean aggregation)"""
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.convs[-1](x, edge_index)
        return x
