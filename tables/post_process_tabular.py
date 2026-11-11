"""
Compiles repeated bench-scan and bench-dataloader runs into results tables.

Reads bench-scan-results.log and bench-dataloader-results.log, which gain one
block per run each time `make bench-scan` / `make bench-dataloader` is called
(see the Makefile's `tee -a`). For every value it prints the MEDIAN, with the
[min-max] range over the runs, because individual runs can differ a lot on a
shared file system.

Pass 1 = first read (file cache dropped before it, best effort).
Pass 2 = the same work again with the file cached.

Usage: python3 post_process_tabular.py
"""
import re
import statistics as stats
from pathlib import Path
from collections import defaultdict

FORMATS = ["parquet", "csv", "hdf5"]

# new scan lines:  <format> <pass 1|2> <full> <cols> <filt> <rows>
SCAN_RE = re.compile(r"^(parquet|csv|hdf5)\s+([12])\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)")
# old single-pass scan lines (no pass column), only used to warn
OLD_SCAN_RE = re.compile(r"^(parquet|csv|hdf5)\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+/\d+/\d+")
LOAD_RE = re.compile(r"^(\w+) load time: ([\d.]+)", re.M)
PASS_RE = re.compile(r"^(\w+) dataloader time pass([12]): ([\d.]+)", re.M)
OLD_DL_RE = re.compile(r"^\w+ dataloader time: [\d.]+", re.M)


def cell(values):
    if not values:
        return "n/a"
    return "%.3f [%.3f-%.3f]" % (stats.median(values), min(values), max(values))


def print_table(title, header, rows):
    print("\n=== %s ===" % title)
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def process_scan():
    path = Path("bench-scan-results.log")
    if not path.exists():
        print("No bench-scan-results.log found - run `make bench-scan` at least once.")
        return
    data = defaultdict(lambda: defaultdict(list))   # (fmt, pass) -> column -> values
    rows_seen = defaultdict(set)
    old_lines = 0
    for line in path.read_text().splitlines():
        m = SCAN_RE.match(line.strip())
        if m:
            fmt, p, full, cols, filt, rows = m.groups()
            data[(fmt, p)]["full"].append(float(full))
            data[(fmt, p)]["cols"].append(float(cols))
            data[(fmt, p)]["filt"].append(float(filt))
            rows_seen[fmt].add(rows)
        elif OLD_SCAN_RE.match(line.strip()):
            old_lines += 1
    if old_lines:
        print("WARNING: %d old single-pass line(s) in %s were ignored. Archive old logs before a new set of runs." % (old_lines, path))

    for p, title in (("1", "Scanning and filtering, pass 1: first read, seconds, median [min-max]"),
                     ("2", "Scanning and filtering, pass 2: file cached, seconds, median [min-max]")):
        table = []
        for fmt in FORMATS:
            d = data.get((fmt, p))
            if not d:
                continue
            table.append([fmt, cell(d["full"]), cell(d["cols"]), cell(d["filt"]), len(d["full"])])
        print_table(title, ["format", "full scan", "column subset", "filtered", "runs"], table)
    bad = {f: r for f, r in rows_seen.items() if len(r) > 1}
    if bad:
        print("\nWARNING: the row counts differ between runs for: %s" % ", ".join(sorted(bad)))
    else:
        print("\nRows returned (full/subset/filtered): " +
              "; ".join("%s %s" % (f, list(rows_seen[f])[0]) for f in FORMATS if f in rows_seen))


def process_dataloader():
    path = Path("bench-dataloader-results.log")
    if not path.exists():
        print("No bench-dataloader-results.log found - run `make bench-dataloader` at least once.")
        return
    text = path.read_text()
    old = len(OLD_DL_RE.findall(text))
    if old:
        print("WARNING: %d old single-pass line(s) in %s were ignored. Archive old logs before a new set of runs." % (old, path))
    data = defaultdict(lambda: defaultdict(list))
    for fmt, val in LOAD_RE.findall(text):
        data[fmt]["load"].append(float(val))
    for fmt, p, val in PASS_RE.findall(text):
        data[fmt]["pass" + p].append(float(val))

    table = []
    for fmt in FORMATS:
        d = data.get(fmt)
        if not d:
            continue
        table.append([fmt, cell(d["load"]), cell(d["pass1"]), cell(d["pass2"]), len(d["pass1"])])
    print_table("Random row batches (10,000 rows), seconds, median [min-max]",
                ["format", "load time", "pass 1 (first read)", "pass 2 (cached)", "runs"], table)


def main():
    process_scan()
    process_dataloader()


if __name__ == "__main__":
    main()
