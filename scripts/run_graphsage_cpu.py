""" training script- Purpose: actual experiment script. It ties everything together.
shows how model training is done
what the model expects
what trainer must provide """



import argparse
import torch
import pandas as pd

from src.utils.seed import set_seed
from src.data.fb15k237 import load_fb15k237, build_train_graph, make_node_subset, subset_graph_by_nodes
from src.data.labeling import frequency_counts, quantile_bins_3, make_node_splits
from src.models.graphsage import GraphSAGE
from src.train.trainer import run_training, eval_model

def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.25, help="Fraction of nodes to keep (e.g., 0.25, 0.5, 0.75, 1.0)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", type=str, default="results/runs/graphsage_run.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    splits = load_fb15k237(root="data/raw")
    full = build_train_graph(splits)

    if args.frac < 0.999:
        keep = make_node_subset(full.num_nodes, frac=args.frac, seed=args.seed)
        data, _ = subset_graph_by_nodes(full, keep)
    else:
        data = full

    freq = frequency_counts(data)
    y = quantile_bins_3(freq)
    train_mask, val_mask, test_mask = make_node_splits(data.num_nodes, seed=args.seed)

    # Learnable node embeddings as features
    """ Eq-initialization of node features """
    x = torch.nn.Embedding(data.num_nodes, args.emb_dim) 
    model = GraphSAGE(args.emb_dim, args.hidden_dim, out_dim=3, num_layers=2)

    # Move to device
    data = data.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    x = x.to(device)
    model = model.to(device)

    # Reset CUDA peak stats if GPU
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    hist = run_training(
        model=model,
        x=x.weight,  # embedding table as node features
        edge_index=data.edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=args.epochs
    )

    test_acc = eval_model(model, x.weight, data.edge_index, y, test_mask)
    print(f"Test accuracy: {test_acc:.3f}")

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU memory (MB): {peak_mb:.1f}")

    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(hist)
    df["test_acc_last"] = test_acc
    if peak_mb is not None:
        df["peak_gpu_mem_mb"] = peak_mb
    df.to_csv(args.out, index=False)
    print("Saved:", args.out)

if __name__ == "__main__":
    import os
    main()
