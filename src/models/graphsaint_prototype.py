import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAINTPrototype(nn.Module):
    """
    GraphSAINT backbone + fixed class prototypes.
    The GraphSAINT part is in the sampler/training pipeline.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        prototype_dim: int,
        class_prototypes: torch.Tensor,
        num_layers: int = 2,
        dropout: float = 0.2,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.dropout = dropout
        self.temperature = temperature
        self.convs = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.projector = SAGEConv(hidden_dim, prototype_dim)
        self.register_buffer("class_prototypes", class_prototypes.float())

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.projector(x, edge_index)

        x = F.normalize(x, p=2, dim=-1)
        prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)

        logits = x @ prototypes.t()
        logits = logits / self.temperature
        return logits