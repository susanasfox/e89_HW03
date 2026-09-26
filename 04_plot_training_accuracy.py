"""
04_plot_training_accuracy.py

Single responsibility:
    Load the training history saved by 03_train_model.py and plot
    accuracy by epoch, saving the figure to a predictable filename and
    also displaying it.

Does not touch 01_data_setup.py, 02_model.py, or 03_train_model.py --
it only reads the artifacts they already produced.

Usage:
    python 04_plot_training_accuracy.py
"""

import importlib
import json
import os
import subprocess
import sys

# --- Ensure required third-party packages are installed ---------------
REQUIRED_PACKAGES = {"matplotlib": "matplotlib", "numpy": "numpy"}


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

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS_DIR = "artifacts"
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "history.json")
PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_accuracy.png")


def load_history(path=HISTORY_PATH):
    """Load the history dict saved by 03_train_model.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No history file found at '{path}'. "
            "Run 03_train_model.py first to generate it."
        )
    with open(path) as f:
        return json.load(f)


def to_percentage(values):
    """
    Normalize accuracy values to a 0-100 percentage scale, regardless
    of whether they were stored as fractions (e.g. torchmetrics'
    default 0.0-1.0 range) or already as percentages (0-100). This
    keeps the y-axis consistent no matter which convention the source
    metric used.
    """
    values = np.asarray(values, dtype=float)
    if values.size and values.max() <= 1.0:
        return values * 100.0
    return values


def plot_accuracy(history, save_path=PLOT_PATH):
    """
    Plot training accuracy by epoch (with validation accuracy shown
    alongside for context, since it's already present in the saved
    history), label both axes, add a title and a readable grid, save
    the figure, and display it.
    """
    train_acc = to_percentage(history["train_metrics"])
    n_epochs = len(train_acc)
    epochs = np.arange(1, n_epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", linestyle="--",
              label="Training accuracy")

    # Validation accuracy is already computed in history -- plot it too
    # for context, on the same 0-100% scale.
    if "valid_metrics" in history and history["valid_metrics"]:
        valid_acc = to_percentage(history["valid_metrics"])
        plt.plot(epochs, valid_acc, marker="o", linestyle="-",
                  label="Validation accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy by Epoch")
    plt.xticks(epochs)          # one tick per epoch for readability
    plt.ylim(0, 100)            # fixed, consistent percentage scale
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    print(f"Saved accuracy plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    history = load_history()
    plot_accuracy(history)