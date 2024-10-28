# MNIST Classifier with PyTorch

## Dependencies

This project requires the following dependencies:

1. **NumPy**: A library for scientific computing in Python, used for numerical operations.
2. **PyTorch**: An open-source machine learning library, used for building and training the neural network model.
3. **Torchvision**: A library that provides pre-trained models and datasets for computer vision tasks, used for loading the MNIST dataset.
4. **Matplotlib**: A plotting library for Python, used for visualizing the training and testing results.

You can install these dependencies using pip:

```
pip install numpy torch torchvision matplotlib
```

## Overview

This code implements a simple neural network model for classifying handwritten digits from the MNIST dataset. The MNIST dataset is a widely-used dataset in the field of image classification, consisting of 28x28 pixel grayscale images of handwritten digits (0-9) and their corresponding labels.

The main steps of the code are:

1. **Data Preprocessing**: The MNIST dataset is loaded and preprocessed, including normalizing the pixel values and applying transformations (like converting to tensors).
2. **Model Definition**: A simple neural network model is defined, consisting of an input layer, two hidden layers with ReLU activations, and an output layer with a Sigmoid activation.
3. **Training**: The model is trained for 10 epochs using the Adam optimizer and Cross-Entropy loss function. Training and testing losses, as well as test accuracy, are tracked and stored.
4. **Visualization**: The training and testing losses, as well as the test accuracy, are plotted using Matplotlib.

The key functions in the code are:

* `train(dataloader, model, loss_fn, optimizer)`: Performs one epoch of training on the model.
* `test(dataloader, model, loss_fn)`: Evaluates the model's performance on the test dataset.
* The `Net` class defines the neural network model architecture.

Overall, this code provides a basic example of how to build and train a neural network model using PyTorch for the MNIST image classification task.
