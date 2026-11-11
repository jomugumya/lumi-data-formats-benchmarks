#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00
#SBATCH --output=training-imagenette-%x-%j.out

# Usage: sbatch run-training.sh <squashfs|lmdb|hdf5>
# Always uses Imagenette + JPEG encoding (the realistic default) — this
# fills the one gap where Imagenette never got a real training-loop test.

CONTAINER=$1
if [[ -z "$CONTAINER" ]]; then
    echo "ERROR: Usage: sbatch run-training.sh <squashfs|lmdb|hdf5>"
    exit 1
fi

OUTDIR="data-formats/imagenette-${CONTAINER}-jpeg"
if [[ "$CONTAINER" == "lmdb" ]]; then
    DATAPATH="${OUTDIR}/train_images"
elif [[ "$CONTAINER" == "hdf5" ]]; then
    DATAPATH="${OUTDIR}/train.h5"
else
    DATAPATH="${OUTDIR}/train.squashfs"
fi

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

mkdir -p torch_home
export TORCH_HOME=$(pwd)/torch_home

if [[ "$CONTAINER" == "squashfs" ]]; then
    srun singularity run -B ${DATAPATH}:/mounted_data:image-src=/ $SIF bash -c \
        "python compare_dataset_training.py --container squashfs --datadir /mounted_data"
elif [[ "$CONTAINER" == "lmdb" ]]; then
    export SINGULARITYENV_PREPEND_PATH=/user-software/bin
    srun singularity run -B venv.sqsh:/user-software:image-src=/ $SIF bash -c \
        "python compare_dataset_training.py --container lmdb --datadir $DATAPATH"
else
    srun singularity run $SIF bash -c \
        "python compare_dataset_training.py --container hdf5 --datadir $DATAPATH"
fi
