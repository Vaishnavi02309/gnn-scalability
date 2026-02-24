"""Purpose: create labels and splits.
This is where  “node classification task” is defined."""



from typing import Tuple
import torch
from torch_geometric.data import Data

def frequency_counts(data: Data) -> torch.Tensor:
    """count node frequency
    freq(e) = count as head + count as tail."""
    num_nodes = data.num_nodes
    freq = torch.zeros(num_nodes, dtype=torch.long)
    src, dst = data.edge_index[0], data.edge_index[1]
    ones_src = torch.ones_like(src, dtype=torch.long)
    ones_dst = torch.ones_like(dst, dtype=torch.long)
    freq.scatter_add_(0, src, ones_src)
    freq.scatter_add_(0, dst, ones_dst)
    return freq

def quantile_bins_3(freq: torch.Tensor) -> torch.Tensor:
    """3 bins using 33rd and 66th percentiles.
    convert counts into class 0/1/2 using quantiles """
    f = freq.to(torch.float)
    q33 = torch.quantile(f, 0.33).item()
    q66 = torch.quantile(f, 0.66).item()
    y = torch.zeros(freq.size(0), dtype=torch.long)
    y[(f > q33) & (f <= q66)] = 1
    y[f > q66] = 2
    return y

def make_node_splits(
    num_nodes: int,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Boolean masks: train/val/test."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    n_train = int(num_nodes * train_frac)
    n_val = int(num_nodes * val_frac)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool); val_mask[val_idx] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
