"""Load FashionMNIST, apply tensor conversion, and make seeded data loaders."""
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

SEED = 42
BATCH_SIZE = 32

def setup_data(root="datasets"):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    combined = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    split_generator = torch.Generator().manual_seed(SEED)
    train_data, valid_data = torch.utils.data.random_split(
        combined, [55_000, 5_000], generator=split_generator
    )
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, generator=loader_generator
    )
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return combined, train_data, valid_data, test_data, train_loader, valid_loader, test_loader

if __name__ == "__main__":
    data = setup_data()
    print(f"Train: {len(data[1]):,}; validation: {len(data[2]):,}; test: {len(data[3]):,}")
