"""Problem 4, step 4: plot the accuracy recorded during training.

Run this after 03_train_model.py in the same notebook kernel. The plot appears
in the notebook and is also saved as an image in the artifacts folder.
"""

from pathlib import Path  # Builds a portable path for the saved figure.
import matplotlib.pyplot as plt  # Draws and displays the accuracy curves.


train_accuracy = history["train_metrics"]  # Accuracy during each training epoch.
valid_accuracy = history["valid_metrics"]  # Accuracy after each epoch.
if not train_accuracy or len(train_accuracy) != len(valid_accuracy):
    raise ValueError("Training and validation accuracy histories must match.")

# Training accuracy is measured throughout an epoch; validation is measured
# after the epoch. These x positions follow the reference notebook's plot.
train_epochs = [epoch + 0.5 for epoch in range(len(train_accuracy))]
valid_epochs = [epoch + 1 for epoch in range(len(valid_accuracy))]

fig, ax = plt.subplots(figsize=(8, 5))  # Create one readable chart.
ax.plot(train_epochs, train_accuracy, ".--", label="Training accuracy")
ax.plot(valid_epochs, valid_accuracy, ".-", label="Validation accuracy")
ax.set_xlabel("Epoch")  # Label the horizontal axis.
ax.set_ylabel("Accuracy (fraction correct)")  # Values are between 0 and 1.
ax.set_title("FashionMNIST accuracy by epoch")  # Describe the chart.
ax.set_ylim(0, 1)  # Show the full possible accuracy range.
ax.grid(True, alpha=0.3)  # Add light guide lines.
ax.legend()  # Identify the two curves.
fig.tight_layout()  # Keep labels inside the saved image.

output_dir = Path("artifacts")  # Use the assignment's artifact folder.
output_dir.mkdir(exist_ok=True)  # Create it if it is not present.
figure_path = output_dir / "prob4_training_accuracy.png"  # Avoid P1's plot.
fig.savefig(figure_path, dpi=150)  # Save the same figure shown below.
plt.show()  # Display the figure in the notebook.
print(f"Saved accuracy plot to {figure_path}")  # Report the output path.
