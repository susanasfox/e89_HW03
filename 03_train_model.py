"""
03_train_model.py

Single responsibility:
    Train the ImageClassifier using the reference notebook's train2()
    workflow, collecting per-epoch training loss, training accuracy,
    and validation accuracy for later plotting.

Inspection notes on the reference train2()/evaluate_tm() helpers
(ported below, parameterized on `device` instead of a global):
    - train2() ALREADY computes per-epoch training accuracy: a
      torchmetrics Accuracy metric is reset at the start of each
      epoch, updated on every training batch, then read with
      metric.compute() once the epoch's batches are done. No
      modification was needed for this requirement.
    - train2() ALREADY computes per-epoch validation accuracy too, by
      calling evaluate_tm() at the end of each epoch on valid_loader.
    - Conclusion: train2()/evaluate_tm() are suitable as-is and are
      reused unchanged (aside from the device-parameter refactor
      already made when they were first ported).

Data and model setup are pulled directly from 01_data_setup.py and
02_model.py so all three scripts stay in sync. No plotting here --
this script only trains and saves history + weights to disk.

Usage:
    python 03_train_model.py
"""

import importlib
import json
import os
import subprocess
import sys

# --- Ensure required third-party packages are installed ---------------
REQUIRED_PACKAGES = {"torch": "torch", "torchmetrics": "torchmetrics"}


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
import torchmetrics

# Filenames start with digits, so they're not valid module identifiers
# for a plain `import` statement -- load them via importlib instead.
data_setup = importlib.import_module("01_data_setup")
model_module = importlib.import_module("02_model")

SEED = 42
N_EPOCHS = 20
LEARNING_RATE = 0.1
ARTIFACTS_DIR = "artifacts"
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "history.json")
MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "model_weights.pt")


def get_device():
    """Prefer CUDA, then Apple MPS, then fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_tm(model, data_loader, metric, device):
    """
    Run the model over `data_loader` in evaluation mode and compute a
    torchmetrics metric over the whole loader.
    """
    model.eval()          # switch to eval mode: disables dropout/batchnorm training behavior
    metric.reset()        # start the metric fresh for this evaluation pass

    with torch.no_grad():  # no need to track gradients during evaluation
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # accumulate this batch's stats

    return metric.compute()  # aggregate across all batches seen


def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs, device):
    """
    Train `model` for `n_epochs`, tracking training loss and both
    training and validation metric (accuracy) values per epoch.
    """
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()  # reset before accumulating this epoch's training accuracy

        for X_batch, y_batch in train_loader:
            model.train()  # switch to train mode: enables dropout/batchnorm training behavior
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

            loss.backward()        # backpropagate
            optimizer.step()       # update weights
            optimizer.zero_grad()  # clear gradients for the next batch

            metric.update(y_pred, y_batch)  # accumulate training accuracy

        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())

        # Evaluate on the validation set at the end of the epoch.
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric, device).item()
        )

        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train acc: {history['train_metrics'][-1]:.4f}, "
              f"valid acc: {history['valid_metrics'][-1]:.4f}")

    return history


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # --- Data (from 01_data_setup.py) ---
    train_loader, valid_loader, test_loader, classes = data_setup.get_dataloaders()

    # --- Model (from 02_model.py) ---
    model = model_module.build_model(seed=SEED).to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Optimizer and metric ---
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=model_module.N_CLASSES).to(device)

    # --- Train ---
    history = train2(model, optimizer, criterion, accuracy,
                      train_loader, valid_loader, N_EPOCHS, device)

    # --- Persist results for the (not-yet-implemented) plotting script ---
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)

    print(f"\nSaved training history to {HISTORY_PATH}")
    print(f"Saved model weights to {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()