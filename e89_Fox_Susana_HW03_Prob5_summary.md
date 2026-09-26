# Codex session summary — Assignment 03, Problem 5

Built a reproducible FashionMNIST image-classifier workflow from the assignment and the “Building an Image Classifier with PyTorch” reference section. The workflow uses seed 42, converts images to scaled float tensors, splits the 60,000-image training set into 55,000 training and 5,000 validation examples, and keeps the 10,000-image test set separate. It defines the 784→300→100→10 ReLU network and trains it for 10 epochs with cross-entropy and SGD (learning rate 0.1), recording both training and validation accuracy each epoch. The notebook includes an accuracy plot and example validation predictions.

The notebook ran successfully from a fresh `e89` kernel. Final epoch accuracy was 91.72% on training data and 88.64% on validation data. The test set was not evaluated.
