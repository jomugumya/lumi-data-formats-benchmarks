"""
SquashFS just packages existing files, so to test "SquashFS + PNG" we need PNG
files on disk first. This script decodes every JPEG and re-saves it as a
lossless PNG at its native resolution, preserving the class-folder structure,
so the result can be packaged with mksquashfs exactly like the original JPEG
folder (and read back with torchvision's ImageFolder).

The source images are JPEG, so each PNG holds the decoded JPEG pixels exactly:
this compares storage size and decode cost, not image quality.

Usage:
  python3 decode_to_png.py -i <dataset_root>/train -o png_images/train
"""
import argparse
import os

from PIL import Image

# Pillow's default (zlib level 6), stated explicitly so the result does not
# depend on a library default.
PNG_COMPRESS_LEVEL = 6


def convert_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for class_name in sorted(os.listdir(input_folder)):
        class_in = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_in):
            continue
        class_out = os.path.join(output_folder, class_name)
        os.makedirs(class_out, exist_ok=True)

        # os.walk, as in decode_to_raw_arrays.py, so nested layouts such as
        # Tiny-ImageNet's train/<class>/images/<file> are found too.
        for root, _dirs, files in os.walk(class_in):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                except Exception:
                    continue
                out_path = os.path.join(class_out, os.path.splitext(fname)[0] + ".png")
                img.save(out_path, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
                count += 1
                if count % 500 == 0:
                    print(f"Converted {count} images")

    print(f"Converted {count}/{count} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    args = parser.parse_args()
    convert_folder(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
