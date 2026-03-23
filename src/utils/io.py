from pathlib import Path
import csv


def append_result_row(csv_path: str, row: dict) -> None:
    """
    Append a single experiment result row to a CSV file.
    If the file does not exist, write the header first.
    """
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_file.exists()

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)