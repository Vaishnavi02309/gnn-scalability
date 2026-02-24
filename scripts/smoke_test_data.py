"""This tells us:
what data we need, what functions must exist, what output should look like"""

"""Purpose: sanity check. No training. Just “does data + labels work?” """


import torch
from src.utils.seed import set_seed
from src.data.fb15k237 import load_fb15k237, build_train_graph, make_node_subset, subset_graph_by_nodes
from src.data.labeling import frequency_counts, quantile_bins_3, make_node_splits

def main():
    seed = 0
    set_seed(seed)

    splits = load_fb15k237(root="data/raw")
    full = build_train_graph(splits)

    keep = make_node_subset(full.num_nodes, frac=0.25, seed=seed)
    sub, _ = subset_graph_by_nodes(full, keep)

    freq = frequency_counts(sub)
    y = quantile_bins_3(freq)
    train_mask, val_mask, test_mask = make_node_splits(sub.num_nodes, seed=seed)

    print("Nodes:", sub.num_nodes)
    print("Edges:", sub.edge_index.size(1))
    print("Label distribution (0/1/2):", torch.bincount(y).tolist())
    print("Split sizes (train/val/test):", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))

if __name__ == "__main__":
    main()
