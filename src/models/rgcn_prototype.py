import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNPrototype(nn.Module):
    """
    R-GCN backbone + fixed class prototypes.
    Final logits are cosine similarities to class embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        prototype_dim: int,
        class_prototypes: torch.Tensor,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_bases: int = None,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.dropout = dropout
        self.temperature = temperature
        self.convs = nn.ModuleList()

        self.convs.append(
            RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
            )

        self.projector = RGCNConv(
            hidden_dim, prototype_dim, num_relations=num_relations, num_bases=num_bases
        )

        self.register_buffer("class_prototypes", class_prototypes.float())

    def forward(self, x, edge_index, edge_type):
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.projector(x, edge_index, edge_type)

        x = F.normalize(x, p=2, dim=-1)
        prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)

        logits = x @ prototypes.t()
        logits = logits / self.temperature
        return logits