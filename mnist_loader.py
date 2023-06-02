'''
 This code is mostly created by chatGPT.
 What I did was modify the structure of the training and test data extracted so it can work with the nn module
'''
import torch
import numpy
from torchvision import datasets, transforms

def load_mnist():
    # Define the data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Extract the data and labels
    training_data = [(x.view(-1).tolist(), y.tolist()) for x, y in train_loader]
    test_data = [(x.view(-1).tolist(), y.tolist()) for x, y in test_loader]

    # converting training_data 
    training_inputs = [numpy.reshape(x, (784, 1)) for x, y in training_data]
    training_results = [vectorized_result(y) for x, y in training_data]
    training_data_np = list(zip(training_inputs, training_results))

    # converting test_data
    test_inputs = [numpy.reshape(x, (784, 1)) for x, y in test_data]
    test_results = [y for x, y in test_data]
    test_data_np = list(zip(test_inputs, test_results))

    return training_data_np, test_data_np

def vectorized_result(j):

    e = numpy.zeros((10, 1))
    e[j] = 1.0
    return e

