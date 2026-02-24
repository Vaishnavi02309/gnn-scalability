# GNN Scalability (FB15k-237) — Minimal Setup

This repo is a **minimal, reproducible** setup to compare GNN scalability on **FB15k-237** with a simple **node classification** task.

## Design decisions (frozen)
- Dataset: FB15k-237 (downloaded automatically by PyTorch Geometric)
- Task: node classification
- Labels: synthetic **frequency bins** (K=3) using 33rd/66th percentiles
- Graph-size scaling: induce subgraphs at 25%, 50%, 75%, (100% optional)
- CPU-first: runs on laptop CPU; uses GPU automatically if available

## Quickstart
1) Install deps:
```bash
pip install -r requirements.txt
```

2) Smoke test (downloads dataset on first run):
```bash
python scripts/smoke_test_data.py
```

3) Run GraphSAGE baseline on CPU-friendly sizes:
```bash
python scripts/run_graphsage_cpu.py --frac 0.25 --epochs 15
```

If you have a GPU available, add:
```bash
python scripts/run_graphsage_cpu.py --device cuda --frac 1.0 --epochs 30
```
