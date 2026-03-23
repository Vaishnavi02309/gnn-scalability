"""Purpose: load dataset and build homogeneous graph from train.txt."""

from typing import Dict
import torch
from torch_geometric.data import Data
from collections import deque




def load_train_graph(train_txt_path: str) -> Dict[str, int | torch.Tensor]:
    """
    Read train.txt and build homogeneous graph.
    
    For each triple (h, r, t):
    edges.append((h_id, t_id))
    edges.append((t_id, h_id))
    
    Returns:
        node_id_map: Dict[node_id_str] -> node_idx (0-indexed)
        edges: List of (src, dst) tuples
        num_nodes: Total number of unique nodes
    """
    node_id_map = {}
    edges = []
    next_idx = 0
    
    with open(train_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            h_str, r_str, t_str = parts[0], parts[1], parts[2]
            
            # Map node IDs
            if h_str not in node_id_map:
                node_id_map[h_str] = next_idx
                next_idx += 1
            if t_str not in node_id_map:
                node_id_map[t_str] = next_idx
                next_idx += 1
            
            h_idx = node_id_map[h_str]
            t_idx = node_id_map[t_str]
            
            # Add bidirectional edges (undirected graph)
            edges.append((h_idx, t_idx))
            edges.append((t_idx, h_idx))
    
    return node_id_map, edges, next_idx


def build_data_object(
    node_id_map: Dict[str, int],
    edges: list,
    num_nodes: int,
    feat_dim: int = 64,
) -> Data:
    """
    Build PyG Data object for node classification.
    
    Args:
        node_id_map: Dict mapping node string IDs to indices
        edges: List of (src, dst) tuples
        num_nodes: Total number of nodes
        feat_dim: Feature dimension (random features)
    
    Returns:
        PyG Data object with x, edge_index, y (unlabeled)
    """
    # Random node features
    x = torch.randn(num_nodes, feat_dim)
    
    # Build edge_index tensor and remove duplicates
    if edges:
        edge_list = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_list = torch.unique(edge_list, dim=1)  # Remove duplicate edges
    else:
        edge_list = torch.zeros((2, 0), dtype=torch.long)
    
    # Labels: -1 for unlabeled (will be filled by labeling.py)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_list, y=y)
    data.num_nodes = num_nodes
    
    return data


def load_fb15k237_homogeneous(train_txt_path: str, feat_dim: int = 64) -> Data:
    """
    Main entry point: load FB15K-237 train.txt and build homogeneous graph.
    
    Returns:
        PyG Data object ready for node classification
    """
    node_id_map, edges, num_nodes = load_train_graph(train_txt_path)
    data = build_data_object(node_id_map, edges, num_nodes, feat_dim)
    return data, node_id_map


#added new on 03/2026 ,,, creates subgraph after the splits. 

def sample_induced_subgraph(data: Data, fraction: float, seed: int = 42, num_seed_nodes: int = 10) -> Data:
    """
    Build a connected-ish induced subgraph by starting from a small random seed set
    and expanding with BFS until the target node count is reached.

    Args:
        data: Full PyG graph
        fraction: Fraction of nodes to keep, in (0, 1]
        seed: Random seed
        num_seed_nodes: Number of random BFS starting nodes

    Returns:
        A new PyG Data object containing the sampled induced subgraph.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    if fraction == 1.0:
        return data.clone()

    num_nodes = data.num_nodes
    target_num_nodes = max(2, int(num_nodes * fraction))

    # Build adjacency list once from edge_index
    src, dst = data.edge_index
    adjacency = [[] for _ in range(num_nodes)]
    for s, d in zip(src.tolist(), dst.tolist()):
        adjacency[s].append(d)

    # Choose random seed nodes
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    seed_nodes = perm[:min(num_seed_nodes, num_nodes)].tolist()

    visited = set()
    queue = deque()

    for node in seed_nodes:
        if node not in visited:
            visited.add(node)
            queue.append(node)

    # BFS expansion
    while queue and len(visited) < target_num_nodes:
        current = queue.popleft()
        for nbr in adjacency[current]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
                if len(visited) >= target_num_nodes:
                    break

    # Fallback: if graph/component is too small, fill remaining nodes randomly
    if len(visited) < target_num_nodes:
        for node in perm.tolist():
            if node not in visited:
                visited.add(node)
                if len(visited) >= target_num_nodes:
                    break

    keep_nodes = torch.tensor(sorted(visited), dtype=torch.long)

    keep_mask = torch.zeros(num_nodes, dtype=torch.bool)
    keep_mask[keep_nodes] = True

    edge_mask = keep_mask[src] & keep_mask[dst]
    sub_edge_index = data.edge_index[:, edge_mask]

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
    old_to_new[keep_nodes] = torch.arange(keep_nodes.size(0), dtype=torch.long)
    sub_edge_index = old_to_new[sub_edge_index]

    sub_data = Data(
        x=data.x[keep_nodes],
        edge_index=sub_edge_index,
        y=data.y[keep_nodes],
    )
    sub_data.num_nodes = keep_nodes.size(0)

    for attr in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, attr):
            setattr(sub_data, attr, getattr(data, attr)[keep_nodes])

    return sub_data
