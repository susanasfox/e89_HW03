"""Define the reference MLP and a compact CNN comparison model."""
import torch.nn as nn

class ImageClassifier(nn.Module):
    """Assignment reference architecture: 784→300→100→10."""
    def __init__(self, n_inputs=784, n_hidden1=300, n_hidden2=100, n_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1), nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2), nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),
        )

    def forward(self, images):
        return self.mlp(images)

class FashionCNN(nn.Module):
    """Small CNN that retains the 2D structure of grayscale images."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(p=0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))
