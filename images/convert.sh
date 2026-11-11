#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750
#SBATCH --time=00:30:00

# Usage: sbatch convert.sh <dataset> <container> <encoding>
#   dataset:  tiny (Tiny-ImageNet) | imagenette (Imagenette)
#   container: squashfs | lmdb | hdf5
#   encoding:  raw | jpeg | png

DATASET=$1
CONTAINER=$2
ENCODING=$3

if [[ -z "$DATASET" || -z "$CONTAINER" || -z "$ENCODING" ]]; then
    echo "ERROR: Usage: sbatch convert.sh <tiny|imagenette> <squashfs|lmdb|hdf5> <raw|jpeg|png>"
    exit 1
fi

if [[ "$ENCODING" != "raw" && "$ENCODING" != "jpeg" && "$ENCODING" != "png" ]]; then
    echo "ERROR: unknown encoding '$ENCODING'. Expected raw, jpeg or png."
    exit 1
fi

if [[ "$DATASET" == "tiny" ]]; then
    TRAIN_DIR="data-formats/raw/tiny-imagenet-200/train"
elif [[ "$DATASET" == "imagenette" ]]; then
    TRAIN_DIR="data-formats-large/raw/imagenette2/train"
else
    echo "ERROR: unknown dataset '$DATASET'. Expected tiny or imagenette."
    exit 1
fi

OUTDIR="data-formats/${DATASET}-${CONTAINER}-${ENCODING}"
mkdir -p "$OUTDIR"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

if [[ "$CONTAINER" == "squashfs" ]]; then
    if [[ "$ENCODING" == "raw" ]]; then
        # Decode to raw .npy arrays first, then squash the resulting folder.
        time srun singularity run $SIF bash -c \
            "python scripts/squashfs/decode_to_raw_arrays.py -i $TRAIN_DIR -o ${OUTDIR}/raw_arrays"
        time srun bash -c "mksquashfs ${OUTDIR}/raw_arrays ${OUTDIR}/train.squashfs -processors 16 -no-progress -no-xattrs -noappend -noI -noD -noF"
    elif [[ "$ENCODING" == "png" ]]; then
        # PNG: re-save every image as a lossless PNG first, then squash the folder.
        time srun singularity run $SIF bash -c \
            "python scripts/squashfs/decode_to_png.py -i $TRAIN_DIR -o ${OUTDIR}/png_images"
        time srun bash -c "mksquashfs ${OUTDIR}/png_images ${OUTDIR}/train.squashfs -processors 16 -no-progress -no-xattrs -noappend -noI -noD -noF"
    else
        # JPEG: package the original files directly, no decoding needed.
        time srun bash -c "mksquashfs $TRAIN_DIR ${OUTDIR}/train.squashfs -processors 16 -no-progress -no-xattrs -noappend -noI -noD -noF"
    fi

elif [[ "$CONTAINER" == "lmdb" ]]; then
    export SINGULARITYENV_PREPEND_PATH=/user-software/bin
    time srun singularity run -B venv.sqsh:/user-software:image-src=/ $SIF bash -c \
        "python scripts/lmdb/convert_to_lmdb_encoding.py --encoding $ENCODING -i $(dirname $TRAIN_DIR)/ -o ${OUTDIR}/"

elif [[ "$CONTAINER" == "hdf5" ]]; then
    if [[ "$ENCODING" == "raw" ]]; then
        if [[ "$DATASET" == "tiny" ]]; then
            RESIZE=64
        else
            RESIZE=256
        fi
        time srun singularity run $SIF bash -c \
            "python scripts/hdf5/convert_to_hdf5_raw.py -i $TRAIN_DIR -o ${OUTDIR}/train.h5 --resize $RESIZE"
    else
        # jpeg copies the original bytes; png re-encodes losslessly
        time srun singularity run $SIF bash -c \
            "python scripts/hdf5/convert_to_hdf5_encoded.py -i $TRAIN_DIR -o ${OUTDIR}/train.h5 --encoding $ENCODING"
    fi
fi
