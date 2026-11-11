"""
Loads samples written by convert_to_lmdb_encoding.py. Handles all three
encodings with one class, since each payload is self-describing (it stores
its own "kind"), so the benchmark script doesn't need three separate loader
classes.
"""
import pickle
import numpy as np
from PIL import Image
import lmdb
from torch.utils.data import Dataset


class LMDBEncodingDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __enter__(self):
        self.env = lmdb.open(
            self.path, subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
        return self

    def __exit__(self, *exc):
        self.env.close()
        return False

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            payload, label = pickle.loads(txn.get(f"{idx}".encode("ascii")))

        if payload["kind"] == "raw":
            arr = np.frombuffer(payload["bytes"], dtype=payload["dtype"]).reshape(payload["shape"])
            img = Image.fromarray(arr)
        else:  # "jpeg" or "png" — both are just encoded bytes PIL can decode directly
            import io
            img = Image.open(io.BytesIO(payload["bytes"])).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label
