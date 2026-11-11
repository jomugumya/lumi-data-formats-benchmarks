#!/bin/bash
# Submits the full worker-count sweep: 3 formats x 5 worker counts = 15 jobs.
# Run this from inside 03-file-formats/ (same directory as the Makefile),
# and only AFTER `make convert` has finished successfully for all 3 formats.
#
# Usage: ./submit-worker-sweep.sh

set -e

FORMATS=("squashfs" "lmdb" "hdf5")
WORKER_COUNTS=(1 2 4 7 14)

LOGDIR="run-scripts/simple-benchmarks/sweep-logs"
mkdir -p "$LOGDIR"

for fmt in "${FORMATS[@]}"; do
    for n in "${WORKER_COUNTS[@]}"; do
        echo "Submitting: format=$fmt workers=$n"
        sbatch \
            --output="${LOGDIR}/sweep-${fmt}-${n}w-%j.out" \
            run-worker-sweep.sh "$fmt" "$n"
    done
done

echo
echo "All jobs submitted. Track with: squeue --me"
echo "Once all are COMPLETED, run: python3 post-process-sweep.py"
