import re
import glob
from pathlib import Path

TIME_RE = re.compile(r"(\w+) training time: ([\d.]+)")


def main():
    print(f"{'container':<10} {'training time (s)':>18}")
    for f in sorted(glob.glob("training-imagenette-*.out")):
        text = Path(f).read_text(errors="ignore")
        m = TIME_RE.search(text)
        if not m:
            print(f"WARNING: no result found in {f} (job may have failed)")
            continue
        container, value = m.group(1), float(m.group(2))
        print(f"{container:<10} {value:18.3f}")


if __name__ == "__main__":
    main()
