#!/bin/bash
# Downloads Imagenette: 10 real classes from ImageNet, original resolution,
# no registration wall (unlike the full 1000-class ImageNet-1k, which
# requires manual account approval on image-net.org and isn't scriptable).
#
# IMPORTANT: run this on the LOGIN NODE, not via sbatch — LUMI compute nodes
# have no internet access.
#
# Usage: ./download_imagenette.sh

set -e
OUTDIR="data-formats-large/raw"
mkdir -p "$OUTDIR"

URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
TGZ="${OUTDIR}/imagenette2.tgz"

echo "Downloading $URL"
wget -q --show-progress -O "$TGZ" "$URL"

echo "Extracting..."
tar -xzf "$TGZ" -C "$OUTDIR"

echo
echo "Done. Expect:"
echo "  ${OUTDIR}/imagenette2/train/<class folders>/*.JPEG"
echo "  ${OUTDIR}/imagenette2/val/<class folders>/*.JPEG"
echo "Images are at ORIGINAL ImageNet resolution (variable sizes, unlike"
echo "Tiny-ImageNet's fixed 64x64) — this is the point of this phase."
echo "Next: run convert-imagenette.sh (sbatch), same pattern as Phase 1's convert.sh"
