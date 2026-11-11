"""
Tiny-ImageNet's val/ folder ships as val/images/*.JPEG (all 10,000 images in
one flat folder) plus val/val_annotations.txt mapping each filename to its
real class. torchvision's ImageFolder expects one subfolder PER CLASS —
given the flat structure as-is, it would silently treat all of val/ as a
single fake class rather than erroring, producing meaningless validation
accuracy in any script that actually trains against it (e.g.
run-comp-vision-transformer.sh's make bench-vit).

This script moves each image into val/<wnid>/<filename>, matching train/'s
per-class layout, so ImageFolder finds the real 200 classes correctly.

Usage: python3 reorganize_val.py -r data-formats/raw/tiny-imagenet-200/val
Safe to re-run: skips reorganizing if val/images/ is already gone (meaning
this already ran successfully).
"""
import os
import shutil
import argparse


def reorganize_val(val_root):
    images_dir = os.path.join(val_root, "images")
    annotations_path = os.path.join(val_root, "val_annotations.txt")

    if not os.path.isdir(images_dir):
        print(f"{images_dir} not found — already reorganized, or val/ is missing entirely.")
        return

    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"{annotations_path} not found — cannot reorganize without the "
            "filename-to-class mapping. Check val/ extracted correctly."
        )

    moved = 0
    with open(annotations_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            class_dir = os.path.join(val_root, wnid)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(images_dir, filename)
            dst = os.path.join(class_dir, filename)
            if os.path.exists(src):
                shutil.move(src, dst)
                moved += 1

    print(f"Moved {moved} images into per-class folders under {val_root}")

    if os.path.isdir(images_dir) and not os.listdir(images_dir):
        os.rmdir(images_dir)
        print(f"Removed now-empty {images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--val_root",
        default="data-formats/raw/tiny-imagenet-200/val",
        help="Path to the val/ folder to reorganize",
    )
    args = parser.parse_args()
    reorganize_val(args.val_root)
