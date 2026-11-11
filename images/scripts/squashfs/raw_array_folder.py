"""
ImageFolder normally expects .jpg/.png files it can open with PIL. For the
"SquashFS + raw" case, the folder contains .npy files instead — this loader
tells ImageFolder how to read those, converting the array back into a PIL
image so the same downstream transform pipeline (Resize/CenterCrop/ToTensor)
works identically to the encoded case.
"""
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder


def npy_loader(path):
    arr = np.load(path)
    return Image.fromarray(arr)


class RawArrayFolder(ImageFolder):
    """ImageFolder configured to read .npy raw-array files instead of images."""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, loader=npy_loader,
                          is_valid_file=lambda path: path.endswith(".npy"))
