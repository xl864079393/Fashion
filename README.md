# Fashion-MNIST Classification Project

An investigation of classification methods for Fashion-MNIST  
**Authors:** Douzi Ma (10161248), Tzu Hsuan Huang (67913387), Xiang Li (77822808)

## Summary

In this project, we classify images from the Fashion-MNIST dataset using k-Nearest Neighbors (KNN), logistic regression, feed-forward neural networks, and convolutional neural networks (CNN). Through comparative analysis, we discovered that feed-forward neural networks is the best model for this dataset, achieving a test accuracy of 90.06%.

## Data Description

The dataset consists of 70,000 28x28 color images evenly distributed across 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. There are 60,000 training images and 10,000 testing images per class. We visualized 21 random datasets with their labels to explore the dataset.

## Research Background

The research paper titled "CNN Model for Image Classification on MNIST and Fashion-MNIST Dataset" by Shivam S. et al. indicates that CNNs are a powerful architecture in deep learning, particularly for image classification. In this report, we test three different classifiers and CNN to show the accuracy of each model on the Fashion-MNIST Dataset.

## Classifiers

### k-Nearest Neighbors (KNN)

The KNN classifier identifies the K nearest samples to the one being evaluated within the training dataset based on distance metrics. The hyperparameters include K [1,2,3,4,5,6,7,8,9,10, 100, 500, 1000]. We used PyTorch for GPU acceleration and integrated the Histogram of Oriented Gradients (HOG) technique to extract pivotal image features for recognition.

### Logistic Regression

This binary classification algorithm uses a sigmoid function to map input features to a probability score between 0 and 1. The hyperparameters include regularization strength (C) [0.001, 0.005, 0.01, 0.05, 10], penalty, solver, and maximum iteration. We used `sklearn.linear_model.LogisticRegression`.

### Feedforward Neural Network

This model converts 2D image tensors into 1D vectors using a flatten layer, followed by fully-connected layers with batch normalization and dropout techniques. Hyperparameters include learning rates [0.001, 0.0005, 0.0001] and dropout rates [0.25, 0.5, 0.75]. We used PyTorch for training and evaluation.

### Convolutional Neural Networks (CNN)

CNNs use filters to extract features and max-pooling to downsample, with non-linear activation functions like ReLU for learning patterns. Hyperparameters include learning rates [0.0005, 0.001, 0.005, 0.01], varying numbers of convolutional layers from 1 to 4, filter sizes [3, 5, 7, 9], and numbers of filters specified as [[16, 32, 32], [16, 32, 64], [32, 32, 64], [32, 64, 128]].

## Experimental Setup

We partitioned the training data into 50,000 training and 10,000 validation sets. The random_state is set to 1234. Missing data in the training set was handled by dropping rows with any missingness.

## Experimental Results

The table below presents the test accuracy results obtained from the trained models:

| Model                        | Hyperparameters                   | Test Accuracy |
|------------------------------|-----------------------------------|---------------|
| k-Nearest Neighbors (KNN)    | K = 4                             | 0.897         |
| Logistic Regression          | C = 0.005                         | 0.8443        |
| Feedforward Neural Network   | Learning rate: 0.0001, Dropout: 0.25 | 0.9006        |
| Convolutional Neural Networks| Layers: 3, Filters: 32,32,64, Kernel Size: 5, Pooling Size: 2x2 | 0.8984 |

## Insights

The project highlights the significant potential of well-tuned neural network architectures in handling complex image classification tasks. The performance of feed-forward neural networks, achieving a test accuracy of 90.06%, demonstrates the capability of machine learning models to generalize from data without explicit programming for each classification category.

## Contributions

For this assignment, the workload was divided as follows:
- Xiang Li: Trained K-Nearest Neighbors (KNN)
- Tzu Hsuan Huang: Trained Logistic Regression
- Douzi Ma: Trained a Feedforward Neural Network
- Tzu Hsuan Huang: Trained and tested Convolutional Neural Networks (CNNs)
- Compilation of the final report: Xiang Li and Douzi Ma

## Appendix

### K-Nearest Neighbors (KNN)

- Regular KNN: Average loss of 0.148, accuracy of 0.867.
- KNN+HOG: Average loss of 0.112, accuracy of 0.897.

### Feedforward Neural Network

- Grid search over learning rates (0.001, 0.0005, 0.0001) and dropout rates (0.25, 0.5, 0.75) across 10 epochs.
- Optimal combination: Learning rate 0.0001, dropout rate 0.25 with validation accuracy of 89.89%.
