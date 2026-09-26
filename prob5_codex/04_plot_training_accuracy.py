"""Plot training and validation accuracy by epoch for model comparison."""
from pathlib import Path
import matplotlib.pyplot as plt

def plot_training_accuracy(histories, output_path="artifacts/prob5_training_accuracy.png"):
    if isinstance(histories, dict):
        histories = [histories]
    fig, ax = plt.subplots(figsize=(9, 5))
    for history in histories:
        epochs = range(1, len(history["train_accuracy"]) + 1)
        label = history["model"].upper()
        ax.plot(epochs, history["train_accuracy"], marker=".", label=f"{label} training")
        ax.plot(epochs, history["valid_accuracy"], marker=".", linestyle="--", label=f"{label} validation")
    ax.set(title="FashionMNIST Accuracy by Epoch", xlabel="Epoch", ylabel="Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.show()
    return fig
