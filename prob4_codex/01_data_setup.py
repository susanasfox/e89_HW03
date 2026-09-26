"""Problem 4, step 1: prepare FashionMNIST for the image classifier.

Run from the HW03 folder. This file can also be pasted into one notebook cell;
the loaders and class names remain available to the following cells.
"""

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as T


SEED = 42
BATCH_SIZE = 32
DATA_ROOT = "datasets"

# Match the image conversion in the supplied chapter notebook: one-channel
# images become float32 tensors with pixel values scaled to [0, 1].
image_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root=DATA_ROOT, train=True, download=True, transform=image_transform
)
test_data = torchvision.datasets.FashionMNIST(
    root=DATA_ROOT, train=False, download=True, transform=image_transform
)

# Use a dedicated generator so the 55,000/5,000 split is reproducible
# without changing random state used by later model or training cells.
split_generator = torch.Generator().manual_seed(SEED)
train_data, valid_data = random_split(
    train_and_valid_data, [55_000, 5_000], generator=split_generator
)

# Shuffle only training data. A separate seed makes its batch order repeatable.
shuffle_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_generator
)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
class_names = train_and_valid_data.classes

print(
    f"FashionMNIST: {len(train_data):,} train, "
    f"{len(valid_data):,} validation, {len(test_data):,} test images"
)
images, labels = next(iter(train_loader))
print(f"First batch: images {tuple(images.shape)} {images.dtype}; "
      f"labels {tuple(labels.shape)} {labels.dtype}")
