from collections import Counter
from pathlib import Path

def get_domain(rel: str) -> str:
    # Example rel: "/people/person/place_of_birth"
    parts = rel.split("/")
    return parts[1] if len(parts) > 1 else "unknown"

def scan_file(path: Path, domain_counter: Counter, rel_counter: Counter):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            rel_counter[r] += 1
            domain_counter[get_domain(r)] += 1

def main():
    raw_dir = Path("data/raw/raw")
    paths = [raw_dir / "train.txt", raw_dir / "valid.txt", raw_dir / "test.txt"]

    for p in paths:
        if not p.exists():
            print(f"Missing file: {p}")
            return

    domain_counter = Counter()
    rel_counter = Counter()

    for p in paths:
        scan_file(p, domain_counter, rel_counter)

    print("\nTop relation domains (by triple count):")
    for dom, cnt in domain_counter.most_common(20):
        print(f"{dom:20s} {cnt}")

    print(f"\nTotal triples scanned: {sum(rel_counter.values())}")
    print(f"Unique relations: {len(rel_counter)}")
    print(f"Unique domains: {len(domain_counter)}")

if __name__ == "__main__":
    main()
