"""Display several validation images with predicted and true class labels."""
import matplotlib.pyplot as plt
import torch


def show_example_predictions(model, dataset, class_names, device, count=6):
    model.eval()
    images, labels = zip(*(dataset[i] for i in range(count)))
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        predictions = model(batch).argmax(dim=1).cpu().tolist()
    fig, axes = plt.subplots(1, count, figsize=(2.2 * count, 3))
    for ax, image, label, prediction in zip(axes, images, labels, predictions):
        ax.imshow(image.squeeze(0), cmap="gray")
        ax.set_title(f"Pred: {class_names[prediction]}\nTrue: {class_names[label]}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig, predictions
