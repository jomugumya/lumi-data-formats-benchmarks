#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00

# Usage: sbatch run-bench.sh <tiny|imagenette> <squashfs|lmdb|hdf5> <raw|jpeg|png>
# Must be run from 01-images/ so relative paths resolve correctly.

DATASET=$1
CONTAINER=$2
ENCODING=$3

if [[ -z "$DATASET" || -z "$CONTAINER" || -z "$ENCODING" ]]; then
    echo "ERROR: Usage: sbatch run-bench.sh <tiny|imagenette> <squashfs|lmdb|hdf5> <raw|jpeg|png>"
    exit 1
fi

OUTDIR="data-formats/${DATASET}-${CONTAINER}-${ENCODING}"

if [[ "$CONTAINER" == "squashfs" ]]; then
    if [[ "$ENCODING" == "raw" ]]; then
        DATAPATH="/mounted_data"
        BIND="-B ${OUTDIR}/train.squashfs:${DATAPATH}:image-src=/"
    else
        DATAPATH="/mounted_data"
        BIND="-B ${OUTDIR}/train.squashfs:${DATAPATH}:image-src=/"
    fi
elif [[ "$CONTAINER" == "lmdb" ]]; then
    DATAPATH="${OUTDIR}/train_images"
    BIND="-B venv.sqsh:/user-software:image-src=/"
    export SINGULARITYENV_PREPEND_PATH=/user-software/bin
elif [[ "$CONTAINER" == "hdf5" ]]; then
    DATAPATH="${OUTDIR}/train.h5"
    BIND=""
fi

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

if [[ "$CONTAINER" == "squashfs" ]]; then
    srun singularity run $BIND $SIF bash -c \
        "python bench_dataloader.py --container squashfs --encoding $ENCODING --datadir $DATAPATH --N 10000 --workers 7"
else
    srun singularity run $BIND $SIF bash -c \
        "python bench_dataloader.py --container $CONTAINER --encoding $ENCODING --datadir $DATAPATH --N 10000 --workers 7"
fi
