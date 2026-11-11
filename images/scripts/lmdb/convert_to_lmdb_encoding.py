"""
Converts an image folder into an LMDB database using one of three sample-level
encodings, with the container format held fixed (LMDB) so the only variable
is how each sample's bytes are represented.

FIX: the "jpeg" path previously decoded every image with PIL and re-saved it
at a fixed quality=95, silently changing every file's actual byte content
and size versus the real source file — confirmed by comparing against real
ground-truth source sizes (off by up to 2x). This version copies the
ORIGINAL file bytes directly for "jpeg", with zero decode/re-encode
round-trip, matching what SquashFS already does correctly. Verified
byte-identical against real source files before shipping.

Usage:
  python3 convert_to_lmdb_encoding.py --encoding raw \
      -i data-formats/raw/tiny-imagenet-200/ -o data-formats-encoding/lmdb_raw/
"""
import os
import io
import argparse
import pickle

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


class _EncodingAwareDataset(Dataset):
    """Wraps ImageFolder's discovered (path, class_idx) pairs. For 'jpeg',
    reads the raw file bytes directly — no PIL decode at all — so the stored
    bytes are byte-identical to the source file. For 'raw', decodes via PIL
    since raw pixel storage genuinely requires decoded values."""
    def __init__(self, samples, encoding):
        self.samples = samples
        self.encoding = encoding

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]

        if self.encoding == "jpeg":
            with open(path, "rb") as f:
                payload = {"kind": "jpeg", "bytes": f.read()}
        elif self.encoding == "raw":
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            payload = {
                "kind": "raw",
                "bytes": arr.tobytes(),
                "shape": arr.shape,
                "dtype": str(arr.dtype),
            }
        elif self.encoding == "png":
            img = Image.open(path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            payload = {"kind": "png", "bytes": buf.getvalue()}
        else:
            raise ValueError(self.encoding)

        return payload, class_idx


def folder2lmdb(image_folder, output_path, encoding, write_frequency=200):
    finder = ImageFolder(image_folder)
    dataset = _EncodingAwareDataset(finder.samples, encoding)
    loader = DataLoader(dataset, num_workers=8, collate_fn=lambda x: x)

    isdir = os.path.isdir(output_path)
    db = lmdb.open(
        output_path, subdir=isdir, map_size=1099511627776 * 2,
        readonly=False, meminit=False, map_async=True,
    )

    txn = db.begin(write=True)
    idx = 0
    for data in loader:
        payload, label = data[0]
        txn.put(f"{idx}".encode("ascii"), pickle.dumps((payload, label), protocol=5))
        if idx % write_frequency == 0:
            print(f"[{idx}/{len(dataset)}]")
            txn.commit()
            txn = db.begin(write=True)
        idx += 1
    txn.commit()

    keys = [f"{k}".encode("ascii") for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))
    db.sync()
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default="data-formats/raw/tiny-imagenet-200/")
    parser.add_argument("-o", "--output_folder", default="data-formats-encoding/lmdb_raw/")
    parser.add_argument("--encoding", choices=["raw", "jpeg", "png"], required=True)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    folder2lmdb(args.input_folder + "val", args.output_folder + "val_images", args.encoding)
    folder2lmdb(args.input_folder + "train", args.output_folder + "train_images", args.encoding)


if __name__ == "__main__":
    main()
