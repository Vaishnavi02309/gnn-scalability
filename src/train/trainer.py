"""Purpose: training loop (the engine). This file makes experiments consistent across models."""

import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import ClusterData, ClusterLoader


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def macro_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute macro-averaged F1 score."""
    pred = logits.argmax(dim=-1).cpu().numpy()
    true = y.cpu().numpy()
    return float(f1_score(true, pred, average="macro"))


def per_class_f1(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> Dict[int, float]:
    """Compute F1 score for each class."""
    pred = logits.argmax(dim=-1).cpu().numpy()
    true = y.cpu().numpy()
    report = classification_report(
        true,
        pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    return {
        int(cls): float(report[str(cls)]["f1-score"])
        for cls in range(num_classes)
        if str(cls) in report
    }


def train_one_epoch(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """One full training epoch: forward -> loss -> backward -> update."""
    model.train()
    optimizer.zero_grad()

    out = model(x, edge_index)
    loss = F.cross_entropy(out[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = accuracy(out[train_mask], y[train_mask])

    return float(loss.item()), acc


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float, Dict[int, float]]:
    """Compute accuracy and F1 metrics without training."""
    model.eval()
    out = model(x, edge_index)

    acc = accuracy(out[mask], y[mask])
    f1_macro = macro_f1(out[mask], y[mask])
    f1_per_class = per_class_f1(out[mask], y[mask], num_classes=out.shape[1])

    return acc, f1_macro, f1_per_class


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
    """
    Repeat epochs, track history, and restore the best model
    based on validation accuracy.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "epoch_time_s": [],
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model=model,
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            optimizer=optimizer,
        )

        va_acc, va_f1_macro, _ = eval_model(
            model=model,
            x=x,
            edge_index=edge_index,
            y=y,
            mask=val_mask,
        )

        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1_macro"].append(va_f1_macro)
        history["epoch_time_s"].append(dt)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"loss {tr_loss:.4f} | "
            f"train_acc {tr_acc:.3f} | "
            f"val_acc {va_acc:.3f} | "
            f"val_f1 {va_f1_macro:.3f} | "
            f"{dt:.2f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def measure_inference_time(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    repeats: int = 1,
) -> float:
    """
    Measure mean inference time in seconds.
    Useful for scalability comparison.
    """
    model.eval()

    start = time.time()
    for _ in range(repeats):
        _ = model(x, edge_index)
    end = time.time()

    return float((end - start) / repeats)


def train_graphsaint(
    model: torch.nn.Module,
    data,
    epochs: int,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    batch_size: int = 6000,
    walk_length: int = 2,
    num_steps: int = 5,
    sample_coverage: int = 0,
    device: str = "cpu",
):
    """
    Train with PyG GraphSAINT random-walk sampler.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = GraphSAINTRandomWalkSampler(
        data=data.cpu(),
        batch_size=batch_size,
        walk_length=walk_length,
        num_steps=num_steps,
        sample_coverage=sample_coverage,
        shuffle=True,
        num_workers=0,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "epoch_time_s": [],
    }

    best_val_acc = 0.0
    best_state = None
    device = torch.device(device)

    model = model.to(device)
    full_x = data.x.to(device)
    full_edge_index = data.edge_index.to(device)
    full_y = data.y.to(device)
    full_val_mask = data.val_mask.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            train_mask = batch.train_mask
            if train_mask.sum() == 0:
                continue

            loss = F.cross_entropy(out[train_mask], batch.y[train_mask])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = out.argmax(dim=-1)
                epoch_correct += int((pred[train_mask] == batch.y[train_mask]).sum().item())
                epoch_count += int(train_mask.sum().item())
                epoch_loss += float(loss.item())

        train_acc = epoch_correct / epoch_count if epoch_count > 0 else 0.0
        train_loss = epoch_loss / max(1, num_steps)

        va_acc, va_f1_macro, _ = eval_model(
            model=model,
            x=full_x,
            edge_index=full_edge_index,
            y=full_y,
            mask=full_val_mask,
        )

        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(va_acc)
        history["val_f1_macro"].append(va_f1_macro)
        history["epoch_time_s"].append(dt)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_loss:.4f} | "
            f"train_acc {train_acc:.3f} | "
            f"val_acc {va_acc:.3f} | "
            f"val_f1 {va_f1_macro:.3f} | "
            f"{dt:.2f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_clustergcn(
    model,
    data,
    epochs,
    lr=0.01,
    weight_decay=5e-4,
    num_parts=50,
    batch_size=5,
    device="cpu",
):
    """
    Cluster-GCN training using graph partitioning.
    On Windows, PyG METIS partitioning is not supported.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    try:
        cluster_data = ClusterData(data.cpu(), num_parts=num_parts, recursive=False)
        loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
    except Exception as e:
        raise RuntimeError(
            "Cluster-GCN partitioning failed. On Windows, PyG METIS partitioning is not supported. "
            "Use Linux/WSL for Cluster-GCN experiments."
        ) from e

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "epoch_time_s": [],
    }

    best_val_acc = 0.0
    best_state = None

    device = torch.device(device)
    model = model.to(device)

    full_x = data.x.to(device)
    full_edge_index = data.edge_index.to(device)
    full_y = data.y.to(device)
    full_val_mask = data.val_mask.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)

            train_mask = batch.train_mask
            if train_mask.sum() == 0:
                continue

            loss = F.cross_entropy(out[train_mask], batch.y[train_mask])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = out.argmax(dim=-1)
                total_correct += int((pred[train_mask] == batch.y[train_mask]).sum().item())
                total_count += int(train_mask.sum().item())
                total_loss += float(loss.item())

        train_acc = total_correct / total_count if total_count > 0 else 0.0
        train_loss = total_loss / max(1, len(loader))

        val_acc, val_f1, _ = eval_model(
            model=model,
            x=full_x,
            edge_index=full_edge_index,
            y=full_y,
            mask=full_val_mask,
        )

        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1)
        history["epoch_time_s"].append(dt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_loss:.4f} | "
            f"train_acc {train_acc:.3f} | "
            f"val_acc {val_acc:.3f} | "
            f"val_f1 {val_f1:.3f} | "
            f"{dt:.2f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_one_epoch_rgcn(
    model,
    x,
    edge_index,
    edge_type,
    y,
    train_mask,
    optimizer,
):
    model.train()
    optimizer.zero_grad()

    out = model(x, edge_index, edge_type)
    loss = F.cross_entropy(out[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    preds = out.argmax(dim=1)
    train_acc = (preds[train_mask] == y[train_mask]).float().mean().item()

    return float(loss.item()), float(train_acc)


@torch.no_grad()
def eval_model_rgcn(model, x, edge_index, edge_type, y, mask):
    model.eval()

    out = model(x, edge_index, edge_type)
    preds = out.argmax(dim=1)

    acc = (preds[mask] == y[mask]).float().mean().item()

    y_true = y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    return acc, macro_f1, per_class_f1


def run_training_rgcn(
    model,
    data,
    epochs=20,
    lr=0.01,
    weight_decay=5e-4,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "epoch_time_s": [],
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch_rgcn(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            y=data.y,
            train_mask=data.train_mask,
            optimizer=optimizer,
        )

        val_acc, val_macro_f1, _ = eval_model_rgcn(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            y=data.y,
            mask=data.val_mask,
        )

        epoch_time = time.time() - start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_macro_f1)
        history["epoch_time_s"].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_loss:.4f} | "
            f"train_acc {train_acc:.3f} | "
            f"val_acc {val_acc:.3f} | "
            f"val_f1 {val_macro_f1:.3f} | "
            f"{epoch_time:.2f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def measure_inference_time_rgcn(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    repeats: int = 1,
) -> float:
    model.eval()

    start = time.time()
    for _ in range(repeats):
        _ = model(x, edge_index, edge_type)
    end = time.time()

    return float((end - start) / repeats)


def run_training_dispatch(
    model_name: str,
    model: torch.nn.Module,
    data,
    epochs: int,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    batch_size: int = 6000,
):
    """
    Small dispatcher so run_node_exp does not become messy.
    """
    model_name = model_name.lower()

    if model_name == "graphsage":
        return run_training(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

    elif model_name == "graphsaint":
        return train_graphsaint(
            model=model,
            data=data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            device=device,
        )

    elif model_name == "clustergcn":
        return train_clustergcn(
            model=model,
            data=data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

    elif model_name == "rgcn":
        return run_training_rgcn(
            model=model,
            data=data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Unsupported model_name for training: {model_name}")