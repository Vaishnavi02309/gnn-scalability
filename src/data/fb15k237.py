"""Utilities for loading FB15k-237 as a homogeneous PyG graph."""

from collections import deque
import random
from typing import Dict, Tuple

import torch
from torch_geometric.data import Data


def load_fb15k237_homogeneous(
    train_txt_path: str,
    feat_dim: int = 64,
) -> Tuple[Data, Dict[str, int], Dict[str, int]]:
    """
    Load FB15k-237 train triples into one homogeneous graph.

    Returns:
        data: PyG Data object with:
            - x
            - edge_index
            - edge_type
        node_id_map: maps entity string -> node index
        rel_id_map: maps relation string -> relation index
    """
    node_id_map: Dict[str, int] = {}
    rel_id_map: Dict[str, int] = {}

    edges_src = []
    edges_dst = []
    edge_types = []

    def get_node_id(node: str) -> int:
        if node not in node_id_map:
            node_id_map[node] = len(node_id_map)
        return node_id_map[node]

    def get_rel_id(rel: str) -> int:
        if rel not in rel_id_map:
            rel_id_map[rel] = len(rel_id_map)
        return rel_id_map[rel]

    with open(train_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            head, relation, tail = parts

            src = get_node_id(head)
            dst = get_node_id(tail)
            rel_id = get_rel_id(relation)

            # forward edge
            edges_src.append(src)
            edges_dst.append(dst)
            edge_types.append(rel_id)

            # reverse edge
            edges_src.append(dst)
            edges_dst.append(src)
            edge_types.append(rel_id)

    num_nodes = len(node_id_map)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    # simple learnable-free random node features
    x = torch.randn((num_nodes, feat_dim), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.num_nodes = num_nodes

    return data, node_id_map, rel_id_map


def sample_induced_subgraph(
    data: Data,
    fraction: float,
    seed: int = 0,
    num_seed_nodes: int = 50,
) -> Data:
    """
    Sample an induced connected-ish subgraph using multi-source BFS.

    This preserves:
        - x
        - edge_index
        - edge_type
        - y (if present)
        - train_mask / val_mask / test_mask (if present)
    """
    if fraction >= 1.0:
        return data

    random.seed(seed)
    num_nodes = data.num_nodes
    target_num_nodes = max(1, int(num_nodes * fraction))

    edge_index = data.edge_index.cpu()
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()

    # build adjacency list
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src_list, dst_list):
        adj[s].append(d)

    # choose seed nodes
    all_nodes = list(range(num_nodes))
    seed_nodes = random.sample(all_nodes, min(num_seed_nodes, num_nodes))

    visited = set(seed_nodes)
    queue = deque(seed_nodes)

    while queue and len(visited) < target_num_nodes:
        u = queue.popleft()
        neighbors = adj[u]
        random.shuffle(neighbors)

        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append(v)
                if len(visited) >= target_num_nodes:
                    break

    selected_nodes = sorted(list(visited))
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}

    selected_set = set(selected_nodes)

    # filter edges whose both endpoints remain in the sampled node set
    new_src = []
    new_dst = []
    new_edge_types = []

    edge_type_cpu = data.edge_type.cpu() if hasattr(data, "edge_type") else None

    for i, (s, d) in enumerate(zip(src_list, dst_list)):
        if s in selected_set and d in selected_set:
            new_src.append(old_to_new[s])
            new_dst.append(old_to_new[d])

            if edge_type_cpu is not None:
                new_edge_types.append(int(edge_type_cpu[i]))

    new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)

    sub_data = Data(
        x=data.x[selected_nodes].clone(),
        edge_index=new_edge_index,
    )
    sub_data.num_nodes = len(selected_nodes)

    if edge_type_cpu is not None:
        sub_data.edge_type = torch.tensor(new_edge_types, dtype=torch.long)

    if hasattr(data, "y"):
        sub_data.y = data.y[selected_nodes].clone()

    if hasattr(data, "train_mask"):
        sub_data.train_mask = data.train_mask[selected_nodes].clone()

    if hasattr(data, "val_mask"):
        sub_data.val_mask = data.val_mask[selected_nodes].clone()

    if hasattr(data, "test_mask"):
        sub_data.test_mask = data.test_mask[selected_nodes].clone()

    return sub_data