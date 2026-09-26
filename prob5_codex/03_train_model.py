"""Train with cross-entropy and SGD, recording accuracy after each epoch."""
import json
from pathlib import Path
import torch
from torch import nn
from model_definition import ImageClassifier

EPOCHS = 10
LEARNING_RATE = 0.1

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_model(train_loader, valid_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                output_dir="artifacts"):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ImageClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = {"train_accuracy": [], "valid_accuracy": [], "epochs": epochs,
               "learning_rate": learning_rate, "seed": 42}
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_fn(model(images), labels).backward()
            optimizer.step()
        train_acc = evaluate_accuracy(model, train_loader, device)
        valid_acc = evaluate_accuracy(model, valid_loader, device)
        history["train_accuracy"].append(train_acc)
        history["valid_accuracy"].append(valid_acc)
        print(f"Epoch {epoch + 1:02d}/{epochs}: train accuracy={train_acc:.4f}; validation accuracy={valid_acc:.4f}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "prob5_model_weights.pt")
    (output / "prob5_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, history, device
