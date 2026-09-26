"""Problem 4, step 3: train the classifier and record epoch accuracy.

Run this after 01_data_setup.py and 02_model.py in the same notebook kernel.
It defines the reference training helpers here so the final notebook can run
without executing earlier sections of the chapter notebook.
"""

import torch  # Provides the optimizer and gradient-free evaluation.
import torchmetrics  # Computes multiclass accuracy across whole epochs.


def evaluate_tm(model, data_loader, metric):
    """Measure accuracy on a loader without changing model weights."""
    model.eval()  # Turn off training behavior while evaluating.
    metric.reset()  # Start validation accuracy from zero correct examples.
    with torch.no_grad():  # Do not build a gradient graph for validation.
        for images, labels in data_loader:  # Read one validation batch.
            images = images.to(device)  # Move images to the model's device.
            labels = labels.to(device)  # Move labels to the same device.
            logits = model(images)  # Produce ten class scores per image.
            metric.update(logits, labels)  # Accumulate correct predictions.
    return metric.compute().item()  # Return accuracy as a Python number.


def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs):
    """Train for several epochs and retain loss and accuracy history."""
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):  # Repeat one full pass over training data.
        model.train()  # Enable training behavior for this epoch.
        metric.reset()  # Start training accuracy from zero correct examples.
        total_loss = 0.0  # Sum the loss contribution from each batch.
        total_examples = 0  # Count examples for a weighted mean loss.

        for images, labels in train_loader:  # Read one training batch.
            images = images.to(device)  # Move images to the model's device.
            labels = labels.to(device)  # Move labels to the same device.
            optimizer.zero_grad()  # Clear gradients left by the prior batch.
            logits = model(images)  # Predict raw class scores.
            loss = criterion(logits, labels)  # Compare scores with labels.
            loss.backward()  # Compute gradients for model parameters.
            optimizer.step()  # Update parameters with stochastic gradient descent.

            batch_size = labels.size(0)  # Count images in this batch.
            total_loss += loss.item() * batch_size  # Accumulate example-weighted loss.
            total_examples += batch_size  # Track all images seen so far.
            metric.update(logits.detach(), labels)  # Count training predictions.

        history["train_losses"].append(total_loss / total_examples)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric)
        )  # Measure validation accuracy after this epoch.
        print(  # Show progress and the values that the plot will use.
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"loss={history['train_losses'][-1]:.4f}, "
            f"train accuracy={history['train_metrics'][-1]:.4f}, "
            f"validation accuracy={history['valid_metrics'][-1]:.4f}"
        )
    return history  # Keep every epoch's results for later plotting.


n_epochs = 20  # Match the epoch count used by the reference notebook.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Match its optimizer.
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
history = train2(  # Train and keep the history instead of discarding it.
    model, optimizer, xentropy, accuracy, train_loader, valid_loader, n_epochs
)
