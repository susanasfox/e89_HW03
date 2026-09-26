"""Plot the recorded training accuracy by epoch."""
from pathlib import Path
import matplotlib.pyplot as plt

def plot_training_accuracy(history, output_path="artifacts/prob5_training_accuracy.png"):
    epochs = range(1, len(history["train_accuracy"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_accuracy"], marker="o", label="Training")
    ax.set(title="FashionMNIST Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.show()
    return fig
