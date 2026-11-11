#!/bin/bash
#SBATCH --job-name=comp-tiny-sweep
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:15:00
#SBATCH --account=project_xxxxxxxxx

# Worker-count sweep, rebuilt for the v2 project structure — the original
# version of this script (copied from the old project) referenced a
# benchmark script and data paths that don't exist here anymore, causing
# every job to fail immediately.
#
# Usage: sbatch run-worker-sweep.sh <squashfs|lmdb|hdf5> <num_workers>
# Always uses Tiny-ImageNet, JPEG encoding (matches the original sweep's scope).

CONTAINER=$1
NUM_WORKERS=$2

if [[ -z "$CONTAINER" || -z "$NUM_WORKERS" ]]; then
    echo "Usage: sbatch run-worker-sweep.sh <squashfs|lmdb|hdf5> <num_workers>"
    exit 1
fi

OUTDIR="data-formats/tiny-${CONTAINER}-jpeg"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

if [[ "$CONTAINER" == "squashfs" ]]; then
    srun singularity run -B ${OUTDIR}/train.squashfs:/mounted_data:image-src=/ $SIF bash -c \
        "python bench_dataloader.py --container squashfs --encoding jpeg --datadir /mounted_data --N 10000 --workers $NUM_WORKERS"
elif [[ "$CONTAINER" == "lmdb" ]]; then
    export SINGULARITYENV_PREPEND_PATH=/user-software/bin
    srun singularity run -B venv.sqsh:/user-software:image-src=/ $SIF bash -c \
        "python bench_dataloader.py --container lmdb --encoding jpeg --datadir ${OUTDIR}/train_images --N 10000 --workers $NUM_WORKERS"
elif [[ "$CONTAINER" == "hdf5" ]]; then
    srun singularity run $SIF bash -c \
        "python bench_dataloader.py --container hdf5 --encoding jpeg --datadir ${OUTDIR}/train.h5 --N 10000 --workers $NUM_WORKERS"
else
    echo "Unknown container: $CONTAINER (expected squashfs, lmdb, or hdf5)"
    exit 1
fi
