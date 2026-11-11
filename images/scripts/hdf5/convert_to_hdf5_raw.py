"""
Generic raw (fixed-shape, decoded-array) HDF5 converter — works for any
ImageFolder-style dataset, not tied to one specific dataset's dimensions.
Uses the parallelized DataLoader approach already proven ~10x faster than a
plain Python for-loop earlier in this project.

Usage:
  python3 convert_to_hdf5_raw.py -i <dataset_root>/train -o out/train.h5 --resize 256
"""
import argparse
import h5py
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def convert(image_folder, output_path, resize, num_workers=16, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(image_folder, transform=transform)
    num_images = len(dataset)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    with h5py.File(output_path, "w") as h5f:
        images = h5f.create_dataset("images", (num_images, 3, resize, resize), dtype="f")
        labels = h5f.create_dataset("labels", (num_images,), dtype="i")

        idx = 0
        for batch_imgs, batch_labels in loader:
            bsz = batch_imgs.size(0)
            images[idx:idx + bsz] = batch_imgs.numpy()
            labels[idx:idx + bsz] = batch_labels.numpy()
            idx += bsz
            print(f"Processed {idx}/{num_images} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    convert(args.input_folder, args.output_path, args.resize, args.num_workers)


if __name__ == "__main__":
    main()
