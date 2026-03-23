from ast import List
from pathlib import Path
import csv
from typing import Dict, List
from pyparsing import Dict



def append_result_row(csv_path: str, row_dict: dict) -> None:
    """
    Append a single experiment result row to a CSV file.
    If the file does not exist, write the header first.
    """
    csv_file = Path(csv_path)
    
    # Check if file exists (added on 23-03-2026)
    file_exists = csv_file.exists()
    
    # Ensure parent directory exists
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())

        # Write header only once
        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


def write_history_csv(csv_path: str, history: dict):
    """
    Save epoch-wise training history to CSV.

    history example:
        {
            "epoch": [...],
            "train_loss": [...],
            "train_acc": [...],
            "val_acc": [...],
            "val_f1_macro": [...],
            "epoch_time_s": [...]
        }
    """
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(history.keys())
    num_rows = len(history[fieldnames[0]])

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_rows):
            row = {key: history[key][i] for key in fieldnames}
            writer.writerow(row)