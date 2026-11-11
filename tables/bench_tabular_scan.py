"""
Analytical/scan benchmark for parquet vs csv vs hdf5.

Tests three access patterns, in increasing order of how much they should
favor a columnar format:
  1. full scan       - read the entire table
  2. column subset    - read only 2-3 of the columns
  3. filtered read    - read only rows matching a condition (predicate pushdown)

CSV cannot push down either column selection or filtering at the file-read
level, so it always pays the full-scan cost first - that gap (not a single
"CSV is slow" number) is the actual finding to document.

TWO PASSES. Each format is measured twice:
  pass 1: before every operation the script asks the kernel to forget any
          cached pages of the file (best effort, see drop_file_cache), so each
          operation pays for its own first read from storage.
  pass 2: the same three operations again, with the file now in the cache.
Without this, the first operation of a run always paid the cold read and the
later ones found the file already cached, so the three numbers were not
comparable.

Usage: python3 bench_tabular_scan.py --datadir data-formats-tabular
"""
import os
import time
import argparse
import pandas as pd

FORMATS = {
    "parquet": "taxi.parquet",
    "csv": "taxi.csv",
    "hdf5": "taxi.h5",
}


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


def bench_full_scan(fmt, path):
    t0 = time.time()
    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "hdf5":
        df = pd.read_hdf(path, key="taxi")
    return time.time() - t0, len(df)


def bench_column_subset(fmt, path, columns):
    t0 = time.time()
    if fmt == "parquet":
        df = pd.read_parquet(path, columns=columns)
    elif fmt == "csv":
        df = pd.read_csv(path, usecols=columns)
    elif fmt == "hdf5":
        df = pd.read_hdf(path, key="taxi", columns=columns)
    return time.time() - t0, len(df)


def bench_filtered(fmt, path, column, threshold):
    t0 = time.time()
    if fmt == "parquet":
        df = pd.read_parquet(path, filters=[(column, ">", threshold)])
    elif fmt == "csv":
        # No pushdown available in CSV: must read everything, then filter in memory.
        df = pd.read_csv(path)
        df = df[df[column] > threshold]
    elif fmt == "hdf5":
        df = pd.read_hdf(path, key="taxi", where=f"{column} > {threshold}")
    return time.time() - t0, len(df)


def measure(fmt, path, args, drop):
    """Runs the three operations once. With drop=True the file cache is
    cleared (best effort) before each operation."""
    operations = [
        lambda: bench_full_scan(fmt, path),
        lambda: bench_column_subset(fmt, path, args.columns),
        lambda: bench_filtered(fmt, path, args.filter_column, args.filter_threshold),
    ]
    results = []
    dropped = True
    for op in operations:
        if drop:
            dropped = drop_file_cache(path) and dropped
        results.append(op())
    return results, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data-formats-tabular")
    parser.add_argument("--columns", nargs="+", default=["trip_distance", "fare_amount"])
    parser.add_argument("--filter-column", default="passenger_count")
    parser.add_argument("--filter-threshold", type=float, default=4)
    args = parser.parse_args()

    print(f"{'format':<10} {'pass':>4} {'full scan (s)':>15} {'col subset (s)':>16} {'filtered (s)':>14} {'rows(full/sub/filt)':>22}")
    for fmt, fname in FORMATS.items():
        path = os.path.join(args.datadir, fname)
        for p in (1, 2):
            (full, cols, filt), dropped = measure(fmt, path, args, drop=(p == 1))
            rows_str = f"{full[1]}/{cols[1]}/{filt[1]}"
            print(f"{fmt:<10} {p:>4} {full[0]:15.3f} {cols[0]:16.3f} {filt[0]:14.3f} {rows_str:>22}")
            if p == 1:
                print(f"# {fmt} cache_drop_requested={dropped}")


if __name__ == "__main__":
    main()
