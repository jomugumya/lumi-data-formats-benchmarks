import io
import h5py
from PIL import Image
from torch.utils.data import Dataset


class HDF5EncodedDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __enter__(self):
        self.file = h5py.File(self.path, "r")
        self.images = self.file["images"]
        self.labels = self.file["labels"]
        return self

    def __exit__(self, *exc):
        self.file.close()
        return False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raw_bytes = self.images[idx].tobytes()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label
