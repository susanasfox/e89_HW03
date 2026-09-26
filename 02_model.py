"""
02_model.py

Single responsibility:
    Define the ImageClassifier MLP architecture from the reference
    notebook: Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear
    (10 output logits). No training or data loading here -- just the
    model definition and a reproducible instantiation.

Usage:
    python 02_model.py
"""

import importlib
import subprocess
import sys

# --- Ensure required third-party packages are installed ---------------
REQUIRED_PACKAGES = {"torch": "torch"}


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
import torch.nn as nn

# Reproducible weight initialization, matching the reference notebook's
# manual_seed(42) call immediately before instantiating the model.
SEED = 42

# Architecture hyperparameters, matching the reference notebook's
# FashionMNIST classifier (1 channel, 28x28 pixels, flattened).
N_INPUTS = 1 * 28 * 28
N_HIDDEN1 = 300
N_HIDDEN2 = 100
N_CLASSES = 10


class ImageClassifier(nn.Module):
    """
    A simple fully-connected classifier for flattened image inputs.

    Architecture:
        Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear (logits)

    The final layer outputs raw logits (no softmax), since it's intended
    to be used with nn.CrossEntropyLoss, which expects logits and applies
    log-softmax internally.
    """

    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),                        # [batch, C, H, W] -> [batch, C*H*W]
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),      # raw logits, one per class
        )

    def forward(self, X):
        return self.mlp(X)


def build_model(n_inputs=N_INPUTS, n_hidden1=N_HIDDEN1, n_hidden2=N_HIDDEN2,
                 n_classes=N_CLASSES, seed=SEED):
    """Instantiate ImageClassifier with a reproducible weight init."""
    torch.manual_seed(seed)
    return ImageClassifier(n_inputs, n_hidden1, n_hidden2, n_classes)


if __name__ == "__main__":
    model = build_model()
    print(model)

    # --- Output shape check on a dummy batch (no real data needed) ---
    dummy_batch = torch.rand(4, 1, 28, 28)  # batch of 4 fake grayscale images
    logits = model(dummy_batch)

    print(f"\nInput shape:  {tuple(dummy_batch.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")  # expected: (4, 10)
    assert logits.shape == (4, N_CLASSES), "Unexpected output shape!"
    print("Model forward pass produced the expected output shape.")

    n_params = sum(param.numel() for param in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")