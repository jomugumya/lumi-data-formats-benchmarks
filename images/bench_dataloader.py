"""
Unified benchmark across the full container x encoding matrix:
  --container: squashfs | lmdb | hdf5
  --encoding:  raw | jpeg | png

Runs the DataLoader loop TWICE — the first pass is "cold" (data not yet in
OS page cache), the second is "warm" (likely cached from the first pass).
Both are reported, since the AI Guide's own SURF reference only reports the
warm number, but the gap between cold and warm is itself informative and
was directly responsible for a confusing result earlier in this project
(Phase 2's tabular benchmarks).

Usage:
  python3 bench_dataloader.py --container lmdb --encoding jpeg \
      --datadir data-formats/lmdb_jpeg --N 10000 --workers 7
"""
import sys
import argparse
from time import time

sys.path.append("scripts/squashfs")
sys.path.append("scripts/lmdb")
sys.path.append("scripts/hdf5")

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_dataset(container, encoding, datadir):
    if container == "squashfs":
        from raw_array_folder import RawArrayFolder
        from torchvision.datasets import ImageFolder
        if encoding == "raw":
            return _NullCtx(RawArrayFolder(datadir, transform=transform))
        else:
            # jpeg and png are both plain image files that ImageFolder decodes
            return _NullCtx(ImageFolder(datadir, transform=transform))
    elif container == "lmdb":
        from lmdb_encoding_dataset import LMDBEncodingDataset
        return LMDBEncodingDataset(datadir, transform=transform)
    elif container == "hdf5":
        if encoding == "raw":
            from hdf5_dataset import HDF5Dataset
            return HDF5Dataset(datadir, transform=transform)
        else:
            # jpeg and png bytes are both decoded by PIL, which detects the format
            from hdf5_encoded_dataset import HDF5EncodedDataset
            return HDF5EncodedDataset(datadir, transform=transform)
    else:
        raise ValueError(container)


class _NullCtx:
    """For plain ImageFolder-based datasets, which have no __enter__/__exit__."""
    def __init__(self, ds):
        self.ds = ds

    def __enter__(self):
        return self.ds

    def __exit__(self, *exc):
        return False


def run_pass(ds, batch_size, num_workers, n_samples, label):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    t0 = time()
    seen = 0
    for images, labels in loader:
        seen += images.size(0)
        if seen >= n_samples:
            break
    elapsed = time() - t0
    print(f"{label} time: {elapsed:.4f}")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", choices=["squashfs", "lmdb", "hdf5"], required=True)
    parser.add_argument("--encoding", choices=["raw", "jpeg", "png"], required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    label_prefix = f"{args.container}-{args.encoding}"

    with build_dataset(args.container, args.encoding, args.datadir) as ds:
        run_pass(ds, args.batch_size, args.workers, args.N, f"{label_prefix} cold")
        run_pass(ds, args.batch_size, args.workers, args.N, f"{label_prefix} warm")


if __name__ == "__main__":
    main()
