from torch_geometric.datasets import FB15k_237

def main():
    ds = FB15k_237(root="data/raw")
    print("Dataset class:", type(ds))
    print("Dataset len:", len(ds))

    # These usually exist on PyG dataset objects:
    print("root:", getattr(ds, "root", None))
    print("raw_dir:", getattr(ds, "raw_dir", None))
    print("processed_dir:", getattr(ds, "processed_dir", None))
    print("raw_paths:", getattr(ds, "raw_paths", None))
    print("processed_paths:", getattr(ds, "processed_paths", None))

    # show what's inside raw_dir if possible
    import os
    if getattr(ds, "raw_dir", None) and os.path.isdir(ds.raw_dir):
        print("\nFiles in raw_dir:")
        for f in os.listdir(ds.raw_dir):
            print(" -", f)

if __name__ == "__main__":
    main()
