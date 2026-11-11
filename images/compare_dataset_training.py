"""
Real training-loop benchmark (forward + backward pass, not just dataloading)
for Imagenette, JPEG-encoded, across all three containers. This fills the
one genuine gap in the original study: Phase 5 (Imagenette) only ever tested
dataloader speed, never real training — unlike Tiny-ImageNet, which got
both. This answers whether the dataloader-only ranking found for Imagenette
holds once GPU compute is added, the way it flipped for Tiny-ImageNet.

Usage: python3 compare_dataset_training.py --container lmdb --datadir data-formats/imagenette-lmdb-jpeg/train_images
"""
import sys
import argparse
from time import time

sys.path.append("scripts/squashfs")
sys.path.append("scripts/lmdb")
sys.path.append("scripts/hdf5")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16

from bench_dataloader import build_dataset, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", choices=["squashfs", "lmdb", "hdf5"], required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with build_dataset(args.container, "jpeg", args.datadir) as ds:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        model = vit_b_16(weights="DEFAULT")
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)  # Imagenette: 10 classes
        model = model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        t0 = time()
        for epoch in range(args.epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        elapsed = time() - t0

    print(f"{args.container} training time: {elapsed:.4f}")


if __name__ == "__main__":
    main()
