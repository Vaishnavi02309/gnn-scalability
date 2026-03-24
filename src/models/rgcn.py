"""Minimal R-GCN model for relational graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_bases: int = None,
    ):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                RGCNConv(in_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
            )
        else:
            self.convs.append(
                RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
                )
            self.convs.append(
                RGCNConv(hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
            )

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x