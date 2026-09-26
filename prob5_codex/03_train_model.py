"""Train/evaluate models with deterministic settings, losses, momentum and plateau scheduling."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn

SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.05
WEIGHT_DECAY = 1e-4
PATIENCE = 5

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def evaluate_model(model, loader, loss_fn, device):
    """Evaluate loss and accuracy in eval mode without gradient tracking."""
    model.eval()
    loss_sum = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total

def train_model(model, train_loader, train_eval_loader, valid_loader,
                model_name="mlp", epochs=EPOCHS, learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                output_dir="artifacts"):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9,
        weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=0.001
    )
    history = {"model": model_name, "seed": SEED, "epochs_requested": epochs,
               "learning_rate": learning_rate, "momentum": 0.9,
               "weight_decay": weight_decay, "train_loss": [],
               "train_accuracy": [], "valid_loss": [], "valid_accuracy": [],
               "learning_rates": []}
    best_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss, train_acc = evaluate_model(model, train_eval_loader, loss_fn, device)
        valid_loss, valid_acc = evaluate_model(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        print(f"{model_name} epoch {epoch + 1:02d}/{epochs}: "
              f"train loss={train_loss:.4f}, accuracy={train_acc:.4f}; "
              f"validation loss={valid_loss:.4f}, accuracy={valid_acc:.4f}; "
              f"lr={optimizer.param_groups[0]['lr']:.4g}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch + 1}; best validation loss={best_loss:.4f}.")
                break
    model.load_state_dict(best_weights)
    best_epoch = min(range(len(history["valid_loss"])), key=history["valid_loss"].__getitem__)
    history["best_epoch"] = best_epoch + 1
    history["best_valid_loss"] = history["valid_loss"][best_epoch]
    history["best_valid_accuracy"] = history["valid_accuracy"][best_epoch]
    history["final_train_loss"] = history["train_loss"][best_epoch]
    history["final_train_accuracy"] = history["train_accuracy"][best_epoch]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "history": history}, output / f"prob5_{model_name}_checkpoint.pt")
    (output / f"prob5_{model_name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, history, device
