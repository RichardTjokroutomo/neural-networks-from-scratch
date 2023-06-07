"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

# path
path = "ROOT_PATH/data/mnist.pkl.gz"


def load_data():

    f = gzip.open(path, "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="iso-8859-1")
    f.close()
    return (training_data, test_data)

def load_mnist():
   
    tr_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    print("training_inputs type: " + str(type(training_inputs)))
    print("training_results type: " + str(type(training_results)))
    print("========================================================================")
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    #print(training_data[0])
    #print(test_data[0])
    return (training_data, test_data)

def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

