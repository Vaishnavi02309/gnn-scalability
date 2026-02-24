"""Purpose: load dataset + generate subgraphs for different graph sizes."""


from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch_geometric.datasets import FB15k_237
from torch_geometric.data import Data

@dataclass
class KGGraph:
    data: Data
    num_nodes: int

def load_fb15k237(root: str = "data/raw"):
    """
    Loads FB15k-237 via PyG.

    Depending on PyG version, the dataset may expose:
    - a single Data object (len(ds)=1), OR
    - separate train/valid/test Data objects (len(ds)=3).

    This function supports both.
    """
    ds = FB15k_237(root=root)

    # Case 1: dataset provides 3 split objects
    if len(ds) >= 3:
        return {"train": ds[0], "valid": ds[1], "test": ds[2]}

    # Case 2: dataset provides a single object
    data = ds[0]
    return {"train": data}  # valid/test not needed for our current pipeline


def build_train_graph(splits: Dict[str, Data]) -> Data:
    """Use training triples only to form the message-passing graph. uses training edges only"""
    train = splits["train"]
    data = Data(edge_index=train.edge_index, edge_type=train.edge_type)
    data.num_nodes = train.num_nodes
    return data

def make_node_subset(num_nodes: int, frac: float, seed: int) -> torch.Tensor:
    """chooses a random subset of nodes"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    k = max(10, int(num_nodes * frac))
    return perm[:k]

def subset_graph_by_nodes(data: Data, keep_nodes: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    """Induce a subgraph on keep_nodes and reindex nodes."""
    keep_nodes = keep_nodes.to(torch.long)
    old_to_new = -torch.ones(data.num_nodes, dtype=torch.long)
    old_to_new[keep_nodes] = torch.arange(keep_nodes.numel(), dtype=torch.long)

    src, dst = data.edge_index[0], data.edge_index[1]
    mask = (old_to_new[src] != -1) & (old_to_new[dst] != -1)

    sub_edge_index = torch.stack([old_to_new[src[mask]], old_to_new[dst[mask]]], dim=0)
    sub_edge_type = data.edge_type[mask]

    sub = Data(edge_index=sub_edge_index, edge_type=sub_edge_type)
    sub.num_nodes = keep_nodes.numel()
    return sub, old_to_new
