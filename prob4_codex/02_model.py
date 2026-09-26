"""Problem 4, step 2: define the classifier and loss from the reference."""

import torch  # Provides random seeding and device checks.
from torch import nn  # Provides neural network layers and the loss function.


class ImageClassifier(nn.Module):  # Define the same model class as the reference.
    """Map a one-channel 28x28 image to ten class logits."""

    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()  # Initialize the PyTorch module base class.
        self.mlp = nn.Sequential(  # Run these layers in the listed order.
            nn.Flatten(),  # Turn each 1x28x28 image into 784 input values.
            nn.Linear(n_inputs, n_hidden1),  # Map inputs to 300 hidden units.
            nn.ReLU(),  # Add a nonlinear activation after the first layer.
            nn.Linear(n_hidden1, n_hidden2),  # Map 300 units to 100 units.
            nn.ReLU(),  # Add a nonlinear activation after the second layer.
            nn.Linear(n_hidden2, n_classes),  # Produce ten raw class scores.
        )

    def forward(self, images):  # Define what happens during a model call.
        return self.mlp(images)  # Pass the image batch through all layers.


if torch.cuda.is_available():  # Prefer an NVIDIA GPU when one is available.
    device = "cuda"  # Use CUDA for model calculations.
elif torch.backends.mps.is_available():  # Check for Apple Silicon GPU support.
    device = "mps"  # Use Apple's Metal backend.
else:  # Fall back when no supported GPU is available.
    device = "cpu"  # Use the computer's processor.

torch.manual_seed(42)  # Make the model's initial weights reproducible.
model = ImageClassifier(  # Create the reference notebook's classifier.
    n_inputs=1 * 28 * 28,  # One grayscale image has 784 pixels.
    n_hidden1=300,  # Give the first hidden layer 300 units.
    n_hidden2=100,  # Give the second hidden layer 100 units.
    n_classes=10,  # FashionMNIST has ten clothing classes.
).to(device)  # Move the model's parameters to the selected device.
xentropy = nn.CrossEntropyLoss()  # Compare raw logits with class labels.

print(f"Model device: {device}")  # Show where the model will run.
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")  # Show model size.
