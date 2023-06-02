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
    print("pytorch test size: " + str(len(test_data)))
    test_inputs = [numpy.reshape(x, (784, 1)) for x, y in test_data]
    test_results = [y for x, y in test_data]
    test_data_np = list(zip(test_inputs, test_results))

    # converting training_data into numpy arrays
    '''
    training_inputs = [numpy.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data_np = list(zip(training_inputs, training_results))

    # converting test_data into numpy arrays
    
    test_inputs = [numpy.reshape(x, (784, 1)) for x in test_data[0]]
    test_results = list(zip(test_inputs, test_data[1]))
    test_data_np = list(zip(test_inputs, test_results))
    return training_data_np, test_data_np
    '''
    return training_data_np, test_data_np

def vectorized_result(j):

    e = numpy.zeros((10, 1))
    e[j] = 1.0
    return e

# Usage
train_data, test_data = load_mnist()

# Example printing the first sample
#print(test_data[0])
print("size of training_data: " + str(len(train_data)))
print("size of test_data: " + str(len(test_data)))
print("==================================================================================")
