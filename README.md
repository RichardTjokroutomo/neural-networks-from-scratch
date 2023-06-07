# Neural Network

This project attempts to create a multilayer perceptron used to recognize handwritten digits from scratch, 
using only NumPy and Random Python modules. (currently, the input is limited to that from MNIST dataset only)


## Description
There are 3 main methods:


> ```nn.stochastic_gradient_descent(training_data, epoch, batch_size, learning_rate)```
> ```nn.testing_nn(test_data)```
> ```nn.eval_input(input_data)```


#### Notes:
> - before running master.py, make sure to change the ```path``` variable in ```mnist_loader.py``` to the directory of where you are storing the MNIST data.

> - ```.stochastic_gradient_descent()``` returns null.
> - ```.testing_nn()``` returns 2 ```int``` values: the number of correct guess and the total number of trials. 
> - ```.eval_input()``` returns the number represented by the handwritten 28x28 pixel

>    - training_data is a ```list``` of ```tuples (x, y)```, where x is an n x 1 NumPy array, where n is the size of the input (In the sample dataset loader, n is 784). As for y, it is a 10 x 1 NumPy array filled with zeros,  except for the index of the expected output
>    - test_data is a ```list``` of ```tuples (x, y)``` where x is an n x 1 NumPy array, where n is the size of the input. As for y, it is an ```int``` value containing the expected output 
>    - input_data is an n x 1 NumPy array (in the example, n is 784)

