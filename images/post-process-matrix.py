"""
Aggregates bench-<dataset>-<container>-<encoding>-<jobid>.out files into a
single results table covering the full matrix.

Usage: python3 post-process-matrix.py
"""
import re
import glob
from pathlib import Path

FILENAME_RE = re.compile(r"bench-(?P<dataset>tiny|imagenette)-(?P<container>squashfs|lmdb|hdf5)-(?P<encoding>raw|jpeg|png)-\d+\.out")
TIME_RE = re.compile(r"[\w-]+ (cold|warm) time: ([\d.]+)")


def main():
    results = {}
    for f in sorted(glob.glob("bench-*.out")):
        m = FILENAME_RE.match(Path(f).name)
        if not m:
            continue
        key = (m.group("dataset"), m.group("container"), m.group("encoding"))
        text = Path(f).read_text(errors="ignore")
        matches = TIME_RE.findall(text)
        if len(matches) < 2:
            print(f"WARNING: incomplete result in {f} (job may have failed)")
            continue
        cold = next(v for k, v in matches if k == "cold")
        warm = next(v for k, v in matches if k == "warm")
        results[key] = (float(cold), float(warm))

    print(f"{'dataset':<12} {'container':<10} {'encoding':<8} {'cold (s)':>10} {'warm (s)':>10}")
    for (dataset, container, encoding), (cold, warm) in sorted(results.items()):
        print(f"{dataset:<12} {container:<10} {encoding:<8} {cold:10.3f} {warm:10.3f}")


if __name__ == "__main__":
    main()
