#!/bin/bash
# Downloads NYC TLC yellow taxi trip data (native Parquet from the source).
#
# IMPORTANT: run this on the LOGIN NODE, not via sbatch — LUMI compute nodes
# have no internet access, so this will hang/fail inside a Slurm job.
#
# Usage: ./download_taxi_data.sh [YYYY-MM ...]
# Example: ./download_taxi_data.sh 2024-01 2024-02

set -e
OUTDIR="data-formats-tabular/raw"
mkdir -p "$OUTDIR"

MONTHS=("$@")
if [ ${#MONTHS[@]} -eq 0 ]; then
    MONTHS=("2024-01")  # ~3M rows, a reasonable single-month starting size
fi

for m in "${MONTHS[@]}"; do
    url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_${m}.parquet"
    out="${OUTDIR}/yellow_tripdata_${m}.parquet"
    echo "Downloading $url"
    wget -q --show-progress -O "$out" "$url"
done

echo
echo "Downloaded to $OUTDIR/. Next: run convert_tabular_formats.py"
