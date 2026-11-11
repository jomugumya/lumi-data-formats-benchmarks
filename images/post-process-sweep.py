"""
Parses the output files from the worker-count sweep (submit-worker-sweep.sh)
and prints a workers x format table of WARM dataloader times (steady-state,
repeated-read performance — the number that matters across a real many-epoch
training run), plus saves a CSV.

Run from inside 01-images/, after all sweep jobs show COMPLETED in
`sacct` or `squeue --me`.

Usage: python3 post-process-sweep.py [--dir run-scripts/simple-benchmarks/sweep-logs]
"""
import re
import csv
import glob
import argparse
from pathlib import Path

FORMATS = ["squashfs", "lmdb", "hdf5"]
FILENAME_RE = re.compile(r"sweep-(?P<format>squashfs|lmdb|hdf5)-(?P<workers>\d+)w-\d+\.out")
# FIX: the old script printed "dataloader time: X" — bench_dataloader.py
# (this project's current unified benchmark) prints "<container>-<encoding>
# warm time: X" and "... cold time: X" instead. This was never updated when
# the worker-sweep scripts were carried over, causing every job to be
# reported as failed even once the underlying benchmark succeeded.
TIME_RE = re.compile(r"warm time:\s*([\d.]+)")


def parse_time(path: Path):
    text = path.read_text(errors="ignore")
    m = TIME_RE.search(text)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir",
        default="run-scripts/simple-benchmarks/sweep-logs",
        help="Directory containing sweep-*.out files",
    )
    args = parser.parse_args()

    results = {}  # {format: {workers: time}}
    failed = []

    for f in sorted(glob.glob(f"{args.dir}/sweep-*.out")):
        path = Path(f)
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        fmt = m.group("format")
        workers = int(m.group("workers"))
        t = parse_time(path)
        if t is None:
            failed.append(path.name)
            continue
        results.setdefault(fmt, {})[workers] = t

    if failed:
        print("WARNING: no 'warm time' found (job likely errored) in:")
        for name in failed:
            print(f"  - {name}")
        print()

    worker_counts = sorted({w for fmt_results in results.values() for w in fmt_results})

    header = ["workers"] + FORMATS
    rows = []
    for w in worker_counts:
        row = [w] + [results.get(fmt, {}).get(w, "-") for fmt in FORMATS]
        rows.append(row)

    col_widths = [
        max(len(str(x)) for x in ([h] + [r[i] for r in rows]))
        for i, h in enumerate(header)
    ]

    def fmt_row(r):
        return "  ".join(str(x).rjust(col_widths[i]) for i, x in enumerate(r))

    print(fmt_row(header))
    print(fmt_row(["-" * w for w in col_widths]))
    for r in rows:
        print(fmt_row(r))

    out_csv = Path(args.dir) / "sweep-results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
