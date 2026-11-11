"""
SquashFS just packages existing files — it never decodes anything. To test
"SquashFS + raw (unencoded)", we need actual raw array files on disk first.
This script decodes every JPEG into a .npy file, preserving the original
class-folder structure, so the result can be packaged with mksquashfs
exactly like the original JPEG folder was.

Usage:
  python3 decode_to_raw_arrays.py -i <dataset_root>/train -o raw_arrays/train
"""
import argparse
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def decode_folder(input_folder, output_folder, resize=None):
    os.makedirs(output_folder, exist_ok=True)
    transform = transforms.Resize(resize) if resize else None

    count = 0
    for class_name in sorted(os.listdir(input_folder)):
        class_in = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_in):
            continue
        class_out = os.path.join(output_folder, class_name)
        os.makedirs(class_out, exist_ok=True)

        # FIX: recurse into subfolders instead of only looking at direct
        # children. Tiny-ImageNet nests one level deeper than Imagenette
        # (train/<class>/images/<file>.JPEG vs train/<class>/<file>.JPEG),
        # and a flat os.listdir() silently found nothing for Tiny-ImageNet,
        # producing an empty squashfs with 0 files instead of an error.
        # torchvision's ImageFolder handles this correctly via os.walk;
        # this now does the same.
        for root, _dirs, files in os.walk(class_in):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                except Exception:
                    continue
                if transform:
                    img = transform(img)
                arr = np.array(img)
                out_path = os.path.join(class_out, os.path.splitext(fname)[0] + ".npy")
                np.save(out_path, arr)
                count += 1
                if count % 500 == 0:
                    print(f"Decoded {count} images")

    print(f"Decoded {count}/{count} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("--resize", type=int, default=None,
                         help="Optional: resize to a fixed size before saving (e.g. 256 for Imagenette)")
    args = parser.parse_args()
    decode_folder(args.input_folder, args.output_folder, args.resize)


if __name__ == "__main__":
    main()
