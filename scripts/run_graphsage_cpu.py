"""Domain-based node classification on FB15K-237 using GraphSAGE.

This script:
1. Reads train.txt and extracts domain labels
2. Builds a homogeneous graph
3. Trains GraphSAGE for node classification
4. Reports test accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import Counter

from src.utils.seed import set_seed
from src.data.fb15k237 import load_fb15k237_homogeneous
from src.data.labeling import extract_domains_from_train, make_node_splits
from src.models.graphsage import GraphSAGE
from src.train.trainer import run_training, eval_model
from src.data.fb15k237 import sample_induced_subgraph
from src.utils.memory import get_ram_usage_mb
from src.utils.io import append_result_row
import time
# Configuration
TOP_K_DOMAINS = 5
CONFIDENCE_THRESHOLD = 0.6
FEAT_DIM = 64
EPOCHS = 20  # Reduced for faster testing
SEED = 0
DEVICE = "cpu"
LR = 0.01
WEIGHT_DECAY = 5e-4
GRAPH_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
#GRAPH_FRACTIONS = [0.25]
NUM_SEED_NODES = 50


def main():
    """Main training pipeline."""
    #set_seed(SEED)
    device = torch.device(DEVICE)
    for graph_fraction in GRAPH_FRACTIONS:
        set_seed(SEED)
        print("\n" + "=" * 70)
        print(f"Running GraphSAGE with graph fraction = {graph_fraction}")
        print("=" * 70)
        
        # Step 1: Load homogeneous graph from train.txt
        print("Loading graph from train.txt...")
        data, node_id_map = load_fb15k237_homogeneous(
            train_txt_path="data/raw/raw/train.txt",
            feat_dim=FEAT_DIM
        )
        data = data.to(device)
        
        print(f"  num_nodes: {data.num_nodes}")
        print(f"  num_edges: {data.edge_index.shape[1]}")
        
        # Step 2: Extract domain labels from train.txt
        print("\nExtracting domain labels...")
        node_to_label, kept_nodes, label_to_id = extract_domains_from_train(
            train_txt_path="data/raw/raw/train.txt",
            top_k=TOP_K_DOMAINS,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        num_classes = len(label_to_id)
        print(f"  num_classes: {num_classes}")
        print(f"  classes: {label_to_id}")
        
        # Step 3: Assign labels to nodes
        print("\nAssigning labels to nodes...")
        y = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
        
        # Convert node_id_map (string -> idx) to indices
        for node_str_id, label_id in node_to_label.items():
            if node_str_id in node_id_map:
                node_idx = node_id_map[node_str_id]
                y[node_idx] = label_id
        
        # Count labeled vs unlabeled nodes
        labeled_mask = (y != -1)
        num_labeled = labeled_mask.sum().item()
        print(f"  labeled nodes: {num_labeled} / {data.num_nodes} ({100*num_labeled/data.num_nodes:.1f}%)")
        
        # Class distribution
        class_counts = Counter()
        # for node_idx in kept_nodes:
        #     if node_idx in node_id_map.values():
        #         # Find the string ID for this index
        #         for s_id, idx in node_id_map.items():
        #             if idx == node_idx and node_idx in kept_nodes:
        #                 if s_id in node_to_label:
        #                     class_counts[label_to_id[node_to_label[s_id]]] += 1
        
        # Simpler: just count from y
        for i in range(data.num_nodes):
            if y[i] >= 0:
                class_counts[y[i].item()] += 1
        
        print(f"  class distribution: {dict(class_counts)}")
        
        # Step 4: Create train/val/test splits (stratified by class)
        print("\nCreating stratified train/val/test splits...")
        labeled_indices = torch.where(labeled_mask)[0]
        #num_labeled_nodes = labeled_indices.shape[0]
        
        # Stratified split
        train_idx, val_idx, test_idx = make_node_splits(
            labeled_indices, y, seed=SEED, train_frac=0.7, val_frac=0.1
        )
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        print(f"  train: {train_idx.shape[0]}, val: {val_idx.shape[0]}, test: {test_idx.shape[0]}")
        
        # Print class distribution in splits
        for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_labels = y[indices]
            split_counts = {}
            for i in range(num_classes):
                count = (split_labels == i).sum().item()
                if count > 0:
                    split_counts[i] = count
            print(f"  {split_name} class dist: {split_counts}")
        # addded new to create subgraph after the splits
        data.y = y
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask   
        
        print(f"\nSampling induced subgraph with fraction={graph_fraction}, seed_nodes={NUM_SEED_NODES}...")
        ram_before_subgraph = get_ram_usage_mb()

        data = sample_induced_subgraph(data, fraction=graph_fraction, seed=SEED,num_seed_nodes=NUM_SEED_NODES,)

        ram_after_subgraph = get_ram_usage_mb()

        print(f"  subgraph nodes: {data.num_nodes}")
        print(f"  subgraph edges: {data.edge_index.shape[1]}")
        print(f"  RAM before subgraph: {ram_before_subgraph:.2f} MB")
        print(f"  RAM after subgraph:  {ram_after_subgraph:.2f} MB")
        sampled_labeled = (data.y >= 0)
        num_sampled_labeled = int(sampled_labeled.sum().item())
        print(f"  sampled labeled nodes: {num_sampled_labeled} / {data.num_nodes}")

        sampled_class_counts = {}
        for i in range(num_classes):
            count = int((data.y == i).sum().item())
            if count > 0:
                sampled_class_counts[i] = count
        print(f"  sampled class distribution: {sampled_class_counts}")
        
        for split_name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
            split_counts = {}
            split_labels = data.y[mask]
            for i in range(num_classes):
                count = int((split_labels == i).sum().item())
                if count > 0:
                    split_counts[i] = count
            print(f"  sampled {split_name} class dist: {split_counts}")
        
        # Step 5: Initialize model
        print("\nInitializing GraphSAGE model...")
        model = GraphSAGE(
            in_dim=FEAT_DIM,
            hidden_dim=64,
            out_dim=num_classes,
            num_layers=2,
            dropout=0.2
        )
        model = model.to(device)
        print(f"  model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Step 6: Train model
        print("\nTraining...")
        # starts timer for runtime measurement
        train_start = time.time()
        hist = run_training(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        train_time = time.time() - train_start
        print(f"Training time: {train_time:.2f} seconds")
        
        
        # Step 7: Test
        print("\nTesting...")
        test_acc, test_f1_macro, test_f1_per_class = eval_model(model, data.x, data.edge_index, data.y, data.test_mask)
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Macro-F1: {test_f1_macro:.4f}")
        print(f"  Per-class F1: {test_f1_per_class}")
        print(f"{'='*60}")
        
        # Report label noise
        print(f"\nLABELING INFO:")
        print(f"  Pseudo-labels inferred from relation namespaces")
        print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  Top-K domains: {TOP_K_DOMAINS}")
        print(f"  Classes: {label_to_id}")
        
        # -----------------------------------------
        # Save experiment results
        # -----------------------------------------
        result_row = {
            "model": "GraphSAGE",
            "graph_fraction": graph_fraction,
            "num_nodes": int(data.num_nodes),
            "num_seed_nodes": NUM_SEED_NODES,
            "num_edges": int(data.edge_index.shape[1]),
            "ram_mb": float(ram_after_subgraph),
            "train_time_sec": float(train_time),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1_macro),
        }

        append_result_row(
            "results/graphsage_scalability.csv",
            result_row
        )

        print("\nExperiment result saved to results/graphsage_scalability.csv")


if __name__ == "__main__":
    main()
