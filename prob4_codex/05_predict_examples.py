"""Problem 4, step 5: inspect classifier predictions on validation images.

Run this after the model has been trained. It follows the reference notebook's
three-image prediction example and reports the four most likely classes.
"""

import torch  # Provides gradient-free prediction and probability functions.


model.eval()  # Turn off training behavior before making predictions.
images, true_labels = next(iter(valid_loader))  # Read a validation batch.
images = images[:3].to(device)  # Select three images and move them to the model.
true_labels = true_labels[:3]  # Keep their correct labels for comparison.

with torch.no_grad():  # Predictions do not need gradients.
    logits = model(images)  # Get ten raw class scores for each image.
    probabilities = torch.softmax(logits, dim=1)  # Convert all scores to probabilities.

probabilities = probabilities.cpu()  # Move results to CPU for easy display.
predicted_labels = probabilities.argmax(dim=1)  # Pick each most likely class.
top_probabilities, top_labels = probabilities.topk(4, dim=1)  # Rank four classes.

for index in range(len(images)):  # Show one result per selected image.
    actual = class_names[true_labels[index].item()]  # Look up the true class name.
    predicted = class_names[predicted_labels[index].item()]  # Look up the prediction.
    print(f"Image {index + 1}: predicted {predicted}; actual {actual}")
    for rank in range(4):  # Show the four most likely classes and probabilities.
        class_name = class_names[top_labels[index, rank].item()]
        probability = top_probabilities[index, rank].item()
        print(f"  {rank + 1}. {class_name}: {probability:.1%}")
