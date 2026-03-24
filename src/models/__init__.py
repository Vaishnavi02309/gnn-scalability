from src.models.graphsage import GraphSAGE
from src.models.graphsaint import GraphSAINTNet
from src.models.clustergcn import ClusterGCNNet
from src.models.rgcn import RGCNNet

def build_model(
    model_name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.2,
    **kwargs
):
    """
    Factory function to build different GNN models.

    Currently supports:
        - GraphSAGE

    Later extend:
        - GraphSAINT
        - Cluster-GCN
        - R-GCN
    """

    model_name = model_name.lower()

    if model_name == "graphsage":
        return GraphSAGE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    # future models go here
    elif model_name == "graphsaint":
        return GraphSAINTNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "clustergcn":
        return ClusterGCNNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "rgcn":
        return RGCNNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_relations=kwargs["num_relations"],
            num_layers=num_layers,
            dropout=dropout,
            num_bases=kwargs.get("num_bases", None),
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")