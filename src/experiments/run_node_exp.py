from collections import Counter
from pathlib import Path
import time
import torch

from src.utils.seed import set_seed
from src.data.fb15k237 import load_fb15k237_homogeneous, sample_induced_subgraph
from src.data.labeling import extract_domains_from_train, make_node_splits
from src.models import build_model
from src.train.trainer import run_training_dispatch, eval_model, measure_inference_time, measure_inference_time_rgcn, eval_model_rgcn
from src.utils.io import append_result_row, write_history_csv
from src.utils.memory import get_ram_usage_mb, reset_peak_gpu_memory, get_peak_gpu_memory_mb


def run_node_exp(
    model_name: str,
    graph_fraction: float,
    epochs: int,
    hidden_dim: int,
    lr: float,
    batch_size=None,
    device: str = "cpu",
    save_dir: str = "results",
    train_txt_path: str = "data/raw/raw/train.txt",
    feat_dim: int = 64,
    top_k_domains: int = 5,
    confidence_threshold: float = 0.6,
    seed: int = 0,
    weight_decay: float = 5e-4,
    num_layers: int = 2,
    dropout: float = 0.2,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    num_seed_nodes: int = 50,
):
    """
    Common experiment pipeline for node classification on FB15K-237.
    """

    set_seed(seed)
    device = torch.device(device)

    print("\n" + "=" * 70)
    print(f"Running {model_name} with graph fraction = {graph_fraction}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load graph
    # ------------------------------------------------------------------
    print("Loading graph from train.txt...")
    data, node_id_map, rel_id_map = load_fb15k237_homogeneous(
        train_txt_path=train_txt_path,
        feat_dim=feat_dim
    )
    data = data.to(device)
    num_relations = len(rel_id_map)
    print(f"  num_relations: {num_relations}")
    print(f"  num_nodes: {data.num_nodes}")
    print(f"  num_edges: {data.edge_index.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Extract pseudo-labels from relation domains
    # ------------------------------------------------------------------
    print("\nExtracting domain labels...")
    node_to_label, kept_nodes, label_to_id = extract_domains_from_train(
        train_txt_path=train_txt_path,
        top_k=top_k_domains,
        confidence_threshold=confidence_threshold
    )

    num_classes = len(label_to_id)
    print(f"  num_classes: {num_classes}")
    print(f"  classes: {label_to_id}")

    # ------------------------------------------------------------------
    # 3. Assign labels to graph nodes
    # ------------------------------------------------------------------
    print("\nAssigning labels to nodes...")
    y = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)

    for node_str_id, label_id in node_to_label.items():
        if node_str_id in node_id_map:
            node_idx = node_id_map[node_str_id]
            y[node_idx] = label_id

    labeled_mask = (y != -1)
    num_labeled = int(labeled_mask.sum().item())
    print(f"  labeled nodes: {num_labeled} / {data.num_nodes} ({100 * num_labeled / data.num_nodes:.1f}%)")

    class_counts = Counter()
    for i in range(data.num_nodes):
        if y[i] >= 0:
            class_counts[int(y[i].item())] += 1
    print(f"  class distribution: {dict(class_counts)}")

    # ------------------------------------------------------------------
    # 4. Create stratified train/val/test splits
    # ------------------------------------------------------------------
    print("\nCreating stratified train/val/test splits...")
    labeled_indices = torch.where(labeled_mask)[0]

    train_idx, val_idx, test_idx = make_node_splits(
        labeled_indices=labeled_indices.cpu(),
        labels=y.cpu(),
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"  train: {train_idx.shape[0]}, val: {val_idx.shape[0]}, test: {test_idx.shape[0]}")

    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_labels = y[indices]
        split_counts = {}
        for i in range(num_classes):
            count = int((split_labels == i).sum().item())
            if count > 0:
                split_counts[i] = count
        print(f"  {split_name} class dist: {split_counts}")

    # attach labels + masks to data before subgraph sampling
    data.y = y
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # ------------------------------------------------------------------
    # 5. Sample induced subgraph
    # ------------------------------------------------------------------
    print(f"\nSampling induced subgraph with fraction={graph_fraction}, seed_nodes={num_seed_nodes}...")
    ram_before_subgraph = get_ram_usage_mb()

    data = sample_induced_subgraph(
        data,
        fraction=graph_fraction,
        seed=seed,
        num_seed_nodes=num_seed_nodes,
    )

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

    # ------------------------------------------------------------------
    # 6. Build model
    # ------------------------------------------------------------------
    print("\nInitializing model...")

    model = build_model(
        model_name=model_name,
        in_dim=feat_dim,
        hidden_dim=hidden_dim,
        out_dim=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        num_relations=num_relations,
    )

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {num_params}")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("\nTraining...")
    reset_peak_gpu_memory()
    train_start = time.time()

    history = run_training_dispatch(
        model_name=model_name,
        model=model,
        data=data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=str(device),
        batch_size=batch_size if batch_size is not None else 6000,
    )

    train_time = time.time() - train_start
    mean_epoch_time = sum(history["epoch_time_s"]) / len(history["epoch_time_s"])
    peak_gpu_memory_mb = get_peak_gpu_memory_mb()

    print(f"Training time: {train_time:.2f} seconds")
    print(f"Mean epoch time: {mean_epoch_time:.2f} seconds")
    print(f"Peak GPU memory usage during training: {peak_gpu_memory_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 8. Inference / test
    # ------------------------------------------------------------------
    print("\nTesting...")

    if model_name.lower() == "rgcn":
        inference_time = measure_inference_time_rgcn(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            repeats=1,
        )

        test_acc, test_f1_macro, test_f1_per_class = eval_model_rgcn(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            y=data.y,
            mask=data.test_mask,
        )
    else:
        inference_time = measure_inference_time(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            repeats=1,
        )

        test_acc, test_f1_macro, test_f1_per_class = eval_model(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            mask=data.test_mask,
        )

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1_macro:.4f}")
    print(f"  Per-class F1: {test_f1_per_class}")
    print(f"  Inference time: {inference_time:.4f} s")
    print(f"{'=' * 60}")

    print("\nLABELING INFO:")
    print("  Pseudo-labels inferred from relation namespaces")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Top-K domains: {top_k_domains}")
    print(f"  Classes: {label_to_id}")

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    result_row = {
        "model": model_name,
        "graph_fraction": graph_fraction,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "num_classes": int(num_classes),
        "device": str(device),
        "feat_dim": int(feat_dim),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "top_k_domains": int(top_k_domains),
        "confidence_threshold": float(confidence_threshold),
        "labeled_nodes_full_graph": int(num_labeled),
        "labeled_nodes_subgraph": int(num_sampled_labeled),
        "ram_before_subgraph_mb": float(ram_before_subgraph),
        "ram_after_subgraph_mb": float(ram_after_subgraph),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "train_time_s": float(train_time),
        "mean_epoch_time_s": float(mean_epoch_time),
        "inference_time_s": float(inference_time),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1_macro),
        "num_parameters": int(num_params),
        "num_relations": int(num_relations),
    }

    result_csv = Path(save_dir) / f"{model_name.lower()}_scalability.csv"
    append_result_row(str(result_csv), result_row)

    history_csv = Path(save_dir) / f"history_{model_name.lower()}_frac_{str(graph_fraction).replace('.', 'p')}.csv"
    write_history_csv(str(history_csv), history)

    print(f"\nSaved final result row to: {result_csv}")
    print(f"Saved training history to: {history_csv}")

    return result_row, history