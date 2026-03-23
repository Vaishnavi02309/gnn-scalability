"""Purpose: create labels and splits.
This is where  “node classification task” is defined."""



from typing import Dict, Tuple, List
from collections import defaultdict
import torch
from sklearn.model_selection import train_test_split


from collections import defaultdict
from typing import Dict, List, Tuple


def extract_domains_from_train(
    train_txt_path: str,
    top_k: int = 5,
    confidence_threshold: float = 0.6
) -> Tuple[Dict[str, int], List[str], Dict[str, int]]:
    """
    Read train.txt and assign domain labels to nodes.

    Labels are constructed in two stages:
    1. For each node, find its dominant domain and confidence.
    2. Keep only the top-K domains ranked by number of confidently labeled nodes.

    Args:
        train_txt_path: Path to train.txt
        top_k: Number of most frequent node classes to keep
        confidence_threshold: Minimum dominant-domain confidence

    Returns:
        node_to_label: Dict[node_id_str] -> label_id
        kept_nodes: List of node ID strings with valid labels
        label_to_id: Dict[domain_str] -> label_id
    """
    # node_id -> {domain -> count}
    domain_counts = defaultdict(lambda: defaultdict(int))

    # -------- Pass 1: collect domain counts per node --------
    with open(train_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            h_id_str, relation_str, t_id_str = parts[0], parts[1], parts[2]

            domain = "unknown"
            if "/" in relation_str:
                split_rel = relation_str.split("/")
                if len(split_rel) > 1 and split_rel[1]:
                    domain = split_rel[1]

            # keep your current choice: count for both head and tail
            domain_counts[h_id_str][domain] += 1
            domain_counts[t_id_str][domain] += 1

    # -------- Pass 2: assign provisional dominant domain per node --------
    provisional_labels = {}   # node_id -> (dominant_domain, confidence)
    domain_node_counts = defaultdict(int)  # domain -> number of confidently labeled nodes

    for node_id, counts in domain_counts.items():
        total_count = sum(counts.values())
        max_domain, max_count = max(counts.items(), key=lambda x: x[1])
        confidence = max_count / total_count

        # ignore low-confidence nodes and unknown domain
        if max_domain != "unknown" and confidence >= confidence_threshold:
            provisional_labels[node_id] = (max_domain, confidence)
            domain_node_counts[max_domain] += 1

    # -------- Pass 3: keep only top-K domains by node count --------
    top_domains = sorted(
        domain_node_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    label_to_id = {domain: idx for idx, (domain, _) in enumerate(top_domains)}
    top_domain_set = set(label_to_id.keys())

    node_to_label = {}
    kept_nodes = []

    for node_id, (domain, confidence) in provisional_labels.items():
        if domain in top_domain_set:
            node_to_label[node_id] = label_to_id[domain]
            kept_nodes.append(node_id)

    return node_to_label, kept_nodes, label_to_id


def make_node_splits(
    labeled_indices: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified Boolean masks: train/val/test for labeled nodes only."""
    # Convert to numpy for sklearn
    indices_np = labeled_indices.numpy()
    labels_np = labels[labeled_indices].numpy()
    
    # First split: train + (val+test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices_np, labels_np, 
        train_size=train_frac, 
        stratify=labels_np, 
        random_state=seed
    )
    
    # Second split: val/test from remaining
    val_size = val_frac / (1 - train_frac)
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels,
        train_size=val_size,
        stratify=temp_labels,
        random_state=seed
    )
    
    # Convert back to tensors
    train_idx = torch.from_numpy(train_indices)
    val_idx = torch.from_numpy(val_indices)
    test_idx = torch.from_numpy(test_indices)
    
    return train_idx, val_idx, test_idx
