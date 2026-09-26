"""
01_data_setup.py

Single responsibility:
    Download FashionMNIST, apply the same tensor conversion used in the
    reference notebook, split the training set into 55,000 train /
    5,000 validation images, and build train/validation/test
    DataLoaders.

No model or training code here -- this script only prepares the data.

Usage:
    python 01_data_setup.py
"""

import importlib
import subprocess
import sys

# --- Ensure required third-party packages are installed ---------------
REQUIRED_PACKAGES = {"torch": "torch", "torchvision": "torchvision"}


def ensure_packages(packages=REQUIRED_PACKAGES):
    """Import each required package, pip-installing it first if missing."""
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"'{import_name}' not found -- installing '{pip_name}'...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_packages()
# ------------------------------------------------------------------------

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, random_split

# Deterministic seed, dataset location, and split sizes, matching the
# reference notebook (55,000 train / 5,000 validation out of the
# 60,000-image FashionMNIST training set).
SEED = 42
DATA_ROOT = "datasets"
TRAIN_SIZE = 55_000
VALID_SIZE = 5_000
BATCH_SIZE = 32


def build_transform():
    """
    Same tensor conversion as the reference notebook: convert the raw
    PIL image to a tensor image, then cast to float32 and scale pixel
    values into [0, 1].
    """
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def get_datasets(root=DATA_ROOT):
    """Download (if needed) and load the FashionMNIST train and test sets."""
    transform = build_transform()

    train_and_valid_data = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform)

    return train_and_valid_data, test_data


def get_dataloaders(root=DATA_ROOT, batch_size=BATCH_SIZE, seed=SEED):
    """
    Build the train/validation/test DataLoaders for FashionMNIST.

    Returns:
        (train_loader, valid_loader, test_loader, classes)
    """
    train_and_valid_data, test_data = get_datasets(root=root)

    # Seed before the random split so it's reproducible run to run.
    torch.manual_seed(seed)
    train_data, valid_data = random_split(
        train_and_valid_data, [TRAIN_SIZE, VALID_SIZE])

    # Seed again before building the shuffling train loader, matching
    # the reference notebook's ordering of manual_seed calls.
    torch.manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, train_and_valid_data.classes


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, classes = get_dataloaders()

    # --- Dataset size checks ---
    print(f"Train dataset size:      {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(valid_loader.dataset)}")
    print(f"Test dataset size:       {len(test_loader.dataset)}")
    print(f"Classes ({len(classes)}): {classes}")

    # --- One batch's shape/dtype check ---
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nOne training batch:")
    print(f"  X_batch shape: {tuple(X_batch.shape)}, dtype: {X_batch.dtype}")
    print(f"  y_batch shape: {tuple(y_batch.shape)}, dtype: {y_batch.dtype}")
    print(f"  Pixel value range: [{X_batch.min().item():.3f}, "
          f"{X_batch.max().item():.3f}]")