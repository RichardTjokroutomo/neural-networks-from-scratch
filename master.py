import mnist_loader
import nn

print("loading the data")
print("============================================================================")

training_data, test_data = mnist_loader.load_mnist()

print("data has been loaded.")
print("============================================================================")
print("")

# training
print("begin training the NN")
print("============================================================================")

myNetwork = nn.nn()
myNetwork.stochastic_gradient_descent(training_data, 10, 10, 3)

print("NN has been trained")
print("============================================================================")
print("")

# testing
print("testing the trained network against 10,000 test samples")
print("============================================================================")

success, total = myNetwork.testing_nn(test_data)
print("result: " + str(success) + "/" + str(total))

print("testing ended")
print("============================================================================")
print("")

# evaluate 1 input
print("now give the NN 1 input, then the NN will output the result")
print("============================================================================")

result = myNetwork.eval_input(test_data[0][0])
print("for the test input, the answer is: " + str(result))

print("test done!")
print("============================================================================")