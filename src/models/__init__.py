from src.models.graphsage import GraphSAGE
from src.models.graphsaint import GraphSAINTNet
from src.models.clustergcn import ClusterGCNNet
from src.models.rgcn import RGCNNet

from src.models.graphsage_prototype import GraphSAGEPrototype
from src.models.graphsaint_prototype import GraphSAINTPrototype
from src.models.clustergcn_prototype import ClusterGCNPrototype
from src.models.rgcn_prototype import RGCNPrototype


def build_model(
    model_name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.2,
    **kwargs
):
    model_name = model_name.lower()

    if model_name == "graphsage":
        return GraphSAGE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

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

    elif model_name == "graphsage_prototype":
        return GraphSAGEPrototype(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            prototype_dim=kwargs["prototype_dim"],
            class_prototypes=kwargs["class_prototypes"],
            num_layers=num_layers,
            dropout=dropout,
            temperature=kwargs.get("temperature", 1.0),
        )

    elif model_name == "graphsaint_prototype":
        return GraphSAINTPrototype(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            prototype_dim=kwargs["prototype_dim"],
            class_prototypes=kwargs["class_prototypes"],
            num_layers=num_layers,
            dropout=dropout,
            temperature=kwargs.get("temperature", 1.0),
        )

    elif model_name == "clustergcn_prototype":
        return ClusterGCNPrototype(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            prototype_dim=kwargs["prototype_dim"],
            class_prototypes=kwargs["class_prototypes"],
            num_layers=num_layers,
            dropout=dropout,
            temperature=kwargs.get("temperature", 1.0),
        )

    elif model_name == "rgcn_prototype":
        return RGCNPrototype(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            prototype_dim=kwargs["prototype_dim"],
            class_prototypes=kwargs["class_prototypes"],
            num_relations=kwargs["num_relations"],
            num_layers=num_layers,
            dropout=dropout,
            num_bases=kwargs.get("num_bases", None),
            temperature=kwargs.get("temperature", 1.0),
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")