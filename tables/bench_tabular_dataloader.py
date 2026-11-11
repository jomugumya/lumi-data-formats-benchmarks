"""
Random-access row-batch benchmark for parquet vs csv vs hdf5, mirroring the
image-pipeline DataLoader benchmark.

  - Parquet and CSV have no native per-row random lookup. Both are loaded
    fully into memory once (a one-time cost, timed separately as "load time"),
    then __getitem__ just indexes into that in-memory table.
  - HDF5 (format="table", written by convert_tabular_formats.py) supports
    per-row random reads by row number without loading the full table into
    memory first.

TWO PASSES. Each run reads the same random rows twice:
  pass 1: first read. Before it, the script asks the kernel to forget any
          cached pages of the file (best effort; see drop_file_cache), so the
          rows have to come from storage if that request is honoured.
  pass 2: the same rows again, now served from the file cache.
For parquet and csv both passes read from memory, so they should be similar;
for hdf5 the gap between the passes is the cost of reading from storage.

Usage: python3 bench_tabular_dataloader.py --datadir data-formats-tabular -ff hdf5 -N 10000
"""
import os
import time
import random
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def drop_file_cache(path):
    """Best effort: ask the kernel to forget this file's cached pages, so the
    next read starts as cold as an unprivileged user can make it. Returns True
    if the request was accepted. It cannot clear caches on the storage servers,
    and it may be ignored by some file systems."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    except (AttributeError, OSError):
        return False


class ParquetRowDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import pyarrow.parquet as pq
        t0 = time.time()
        self.table = pq.read_table(self.path, memory_map=True)
        self.load_time = time.time() - t0
        return self

    def __exit__(self, *exc):
        return False

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        return self.table.slice(idx, 1).to_pylist()[0]


class CSVRowDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        t0 = time.time()
        self.df = pd.read_csv(self.path)
        self.load_time = time.time() - t0
        return self

    def __exit__(self, *exc):
        return False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


class HDF5RowDataset(Dataset):
    def __init__(self, path, key="taxi"):
        self.path = path
        self.key = key

    def __enter__(self):
        t0 = time.time()
        self.store = pd.HDFStore(self.path, mode="r")
        self.n = self.store.get_storer(self.key).nrows
        self.load_time = time.time() - t0  # just opens the handle, not a full load
        return self

    def __exit__(self, *exc):
        self.store.close()
        return False

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Only used as a fallback; DataLoader will prefer __getitems__ below.
        row = self.store.select(self.key, where=f"index in [{idx}]")
        return row.iloc[0].to_dict()

    def __getitems__(self, indices):
        # Select by row number. Building an "index in [...]" text query for every
        # batch was measured to be much slower than passing the row numbers.
        rows = self.store.select(self.key, where=sorted(indices))
        return [row.to_dict() for _, row in rows.iterrows()]


def _collate_passthrough(batch):
    return batch


def time_pass(ds, indices, batch_size, num_workers):
    """One pass over exactly these rows, in this order. Returns (seconds, rows read)."""
    loader = DataLoader(
        ds, batch_size=batch_size, sampler=indices,
        num_workers=num_workers, collate_fn=_collate_passthrough,
    )
    t0 = time.time()
    seen = 0
    for batch in loader:
        seen += len(batch)
    return time.time() - t0, seen


def run(fmt, datadir, num_workers, n_samples, batch_size=64, seed=None):
    if fmt == "parquet":
        path = os.path.join(datadir, "taxi.parquet")
        ds_ctx = ParquetRowDataset(path)
    elif fmt == "csv":
        path = os.path.join(datadir, "taxi.csv")
        ds_ctx = CSVRowDataset(path)
    elif fmt == "hdf5":
        path = os.path.join(datadir, "taxi.h5")
        ds_ctx = HDF5RowDataset(path)
    else:
        raise ValueError(fmt)

    if seed is None:
        seed = random.SystemRandom().randrange(1_000_000)
    dropped = drop_file_cache(path)

    with ds_ctx as ds:
        n = min(n_samples, len(ds))
        # The same rows, in the same order, for both passes, so that pass 2
        # re-reads exactly what pass 1 touched.
        indices = random.Random(seed).sample(range(len(ds)), n)
        t1, seen1 = time_pass(ds, indices, batch_size, num_workers)
        t2, seen2 = time_pass(ds, indices, batch_size, num_workers)

    print(f"# {fmt} seed={seed} rows_per_pass={seen1}/{seen2} cache_drop_requested={dropped}")
    print(f"{fmt} load time: {ds_ctx.load_time:.4f}")
    print(f"{fmt} dataloader time pass1: {t1:.4f}")
    print(f"{fmt} dataloader time pass2: {t2:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data-formats-tabular")
    parser.add_argument("-n", "--num_workers", type=int, default=0)
    parser.add_argument("-N", "--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None, help="seed for the random rows (default: random)")
    parser.add_argument("-ff", "--file_format", choices=["parquet", "csv", "hdf5"], required=True)
    args = parser.parse_args()

    run(args.file_format, args.datadir, args.num_workers, args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
