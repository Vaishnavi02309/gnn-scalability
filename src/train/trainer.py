"""Purpose: training loop (the engine). This file makes experiments consistent across models."""


import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())

"""forward → loss → backward → update"""
def train_one_epoch(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        acc = accuracy(out[train_mask], y[train_mask])
    return float(loss.item()), acc


"""compute accuracy without training"""
@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    model.eval()
    out = model(x, edge_index)
    return accuracy(out[mask], y[mask])

"""repeat epochs, print metrics"""
def run_training(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> Dict[str, list]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "epoch_time_s": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, x, edge_index, y, train_mask, optimizer)
        va_acc = eval_model(model, x, edge_index, y, val_mask)
        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["epoch_time_s"].append(dt)

        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | train_acc {tr_acc:.3f} | val_acc {va_acc:.3f} | {dt:.2f}s")
    return history
