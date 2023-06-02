import mnist_loader
import nn

training_data, test_data = mnist_loader.load_mnist()

myNetwork = nn.nn()
myNetwork.stochastic_gradient_descent(training_data, 10, 10, 3)

success, total = myNetwork.testing_nn(test_data)
print("result: " + str(success) + "/" + str(total))
