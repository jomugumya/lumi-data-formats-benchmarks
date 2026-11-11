"""
Stores encoded image bytes in HDF5 using a variable-length dtype.

--encoding jpeg (default): copies the ORIGINAL file bytes verbatim, with no
    decode/re-encode round-trip, so the stored bytes equal the source files.
--encoding png: decodes each image and re-saves it as PNG (lossless) at the
    image's native resolution. The source images are JPEG, so the PNG holds the
    decoded JPEG pixels exactly: this compares storage size and decode cost,
    not image quality.

Usage:
  python3 convert_to_hdf5_encoded.py -i <dataset_root>/train -o out/train.h5 --encoding png
"""
import argparse
import io

import h5py
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

# Pillow's default (zlib level 6), stated explicitly so the result does not
# depend on a library default.
PNG_COMPRESS_LEVEL = 6


def encode_bytes(path, encoding):
    if encoding == "jpeg":
        with open(path, "rb") as f:
            return f.read()
    if encoding == "png":
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
        return buf.getvalue()
    raise ValueError(encoding)


def convert(image_folder, output_path, encoding):
    finder = ImageFolder(image_folder)
    samples = finder.samples
    num_images = len(samples)

    vlen_dtype = h5py.special_dtype(vlen=np.uint8)

    with h5py.File(output_path, "w") as f:
        images = f.create_dataset("images", (num_images,), dtype=vlen_dtype)
        labels = f.create_dataset("labels", (num_images,), dtype="i")

        for i, (path, class_idx) in enumerate(samples):
            images[i] = np.frombuffer(encode_bytes(path, encoding), dtype=np.uint8)
            labels[i] = class_idx
            if i % 500 == 0:
                print(f"Processed {i}/{num_images} images")

    print(f"Processed {num_images}/{num_images} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("--encoding", choices=["jpeg", "png"], default="jpeg")
    args = parser.parse_args()
    convert(args.input_folder, args.output_path, args.encoding)


if __name__ == "__main__":
    main()
