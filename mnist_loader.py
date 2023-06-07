# this code is originally made by Michael Nielsen. I only made minor modifications from it.

import pickle
import gzip
import numpy as np

# path
# example: my root path: C:/Users/Richard/Documents/-projects/project-17
path = "ROOT_PATH/data/mnist.pkl.gz"


def load_data():

    f = gzip.open(path, "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="iso-8859-1")
    f.close()
    return (training_data, validation_data, test_data)

def load_mnist():
   
    tr_d, va_d, te_d = load_data()
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


#train, test = load_mnist()
#x, y = test[0]
#print(x)

