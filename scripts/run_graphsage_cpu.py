import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.run_node_exp import run_node_exp

GRAPH_FRACTIONS = [0.25, 0.50, 0.75, 1.00]

for frac in GRAPH_FRACTIONS:
    run_node_exp(
        model_name="graphsage",
        graph_fraction=frac,
        epochs=20,
        hidden_dim=64,
        lr=0.01,
        device="cpu",
        save_dir="results",
    )