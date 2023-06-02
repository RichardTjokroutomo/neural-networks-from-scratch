# Neural Network

This project attempts to create a multilayer perceptron used to recognize handwritten digits from scratch, 
using only NumPy and Random Python modules.


## Description
There are 2 main methods:

> ```
> nn.stochastic_gradient_descent(training_data, epoch, batch_size, >learning_rate)
> nn.testing_nn(test_data)
>```


#### Notes:
- .stochastic_gradient_descent() returns null, while .testing_nn returns 2 ```int``` values: the number of correct guess and the total number of trials
- training_data is a ```list``` of ```tuples (x, y)```, where x is an n x 1 NumPy array, where n is the size of the input (In the sample dataset loader, n is 784). As for y, it is a 10 x 1 NumPy array filled with zeros, except for the index of the expected output
- test_data is a ```list``` of ```tuples (x, y)``` where x is an n x 1 NumPy array, where n is the size of the input. As for y, it is an ```int``` value containing the expected output 
