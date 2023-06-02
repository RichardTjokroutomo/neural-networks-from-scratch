import numpy as np
import random

class nn:

    # constructor
    def __init__(self):
        self.layers = 4
        self.weights = [np.random.randn(16, 784), np.random.randn(16, 16), np.random.randn(10, 16)]
        self.biases = [np.random.randn(16, 1), np.random.randn(16, 1), np.random.randn(10, 1)]


    # CONTROLLERS
    # ============================================================================================================

    # training
    def stochastic_gradient_descent(self, training_data, epoch, batch_size, learning_rate):
        print("begin SGD")
        training_data_size = len(training_data)

        for i in range(epoch):
            print("epoch " + str(i) + " begins")
            random.shuffle(training_data)
            batches = [training_data[x:x+batch_size] for x in range(0, training_data_size, batch_size)]
            for batch in batches:
                self.process_batch(batch, learning_rate)
            print("epoch " + str(i) + " completed")
        print("SGD ends.")
        print("=====================================================================")

    # testing
    def testing_nn(self, test_data):
        test_data_size = len(test_data)
        counter = 0
        for test_input, expected_output in test_data:
            result = self.evaluate_model(test_input, expected_output)
            if result:
                counter += 1
        
        return counter, test_data_size


    # MODELS
    # ============================================================================================================

    # evaluate output
    def evaluate_model(self, test_input, expected_output):
        activations = [test_input]
        for i in range(3):
            z = np.dot(self.weights[i], activations[i]) + self.biases[i]
            a = self.activation_func(z)
            activations.append(a)
        
        output = activations[-1]
        current_largest = output[0]
        current_largest_index = 0
        for i in range(10):
            if output[i] > current_largest:
                current_largest = output[i]
                current_largest_index = i

        if current_largest_index == expected_output:
            return True
        else:
            return False
    
    # processing a single batch
    def process_batch(self, training_batch, learning_rate):
        training_batch_size = len(training_batch)
        nabla_weight = [np.zeros([16, 784]), np.zeros([16, 16]), np.zeros([10, 16])]
        nabla_bias = [np.zeros([16, 1]), np.zeros([16, 1]), np.zeros([10, 1])]

        for batch in training_batch:
            training_input = batch[0]
            expected_output = batch[1]

            n_w_d, n_b_d = self.backpropagation(training_input, expected_output) # do the backpropagation

            nabla_weight = [nw + nwd for nw, nwd in zip(nabla_weight, n_w_d)]
            nabla_bias = [nb + nbd for nb, nbd in zip(nabla_bias, n_b_d)]
        
        self.weights = [w - (learning_rate / training_batch_size * nabla_w) for w, nabla_w in zip(self.weights, nabla_weight)]
        self.biases = [b - (learning_rate / training_batch_size * nabla_b) for b, nabla_b in zip(self.biases, nabla_bias)]

    # backpropagation
    def backpropagation(self, training_input, expected_output):
        # initializing stuff...
        nabla_weights = [np.zeros([16, 784]), np.zeros([16, 16]), np.zeros([10, 16])]
        nabla_biases = [np.zeros([16, 1]), np.zeros([16, 1]), np.zeros([10, 1])]
        activations = [training_input]
        z_funcs = []

        # feedforward
        for i in range(3):
            z_func = np.dot(self.weights[i], activations[i]) + self.biases[i]
            z_funcs.append(z_func)
            activation = self.activation_func(z_func)
            activations.append(activation)
        
        # find output layer's delta func
        delta = self.cost_func_prime(activations[-1], expected_output)*self.activation_func_prime(z_funcs[-1])
        nabla_weights[-1] = np.dot(delta, activations[-2].transpose())
        nabla_biases[-1] = delta

        # backpropagate
        for i in range(2):
            current_index = -2-i;
            delta = np.dot(self.weights[current_index+1].transpose(), delta)*self.activation_func_prime(z_funcs[current_index])
            nabla_weights[current_index] = np.dot(delta, activations[current_index-1].transpose())
            nabla_biases[current_index] = delta
        
        return nabla_weights, nabla_biases
    
    # activation function
    def activation_func(self, z):
        return 1.0/(1.0+np.exp(-z))

    # activation function prime
    def activation_func_prime(self, z):
        return self.activation_func(z)*(1-self.activation_func(z))
    
    # div(cost function)
    def cost_func_prime(self, output, expected_output):
        return (output - expected_output)

# begin testing here
n = nn()
