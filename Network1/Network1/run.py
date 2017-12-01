import network
import mnist_loader
import numpy as np

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

x = np.array([[0,0],[1,1],[0,1],[1,0]])
y = np.array([[0],[0],[1],[1]])

x = [np.reshape(a, (2, 1)) for a in x]
y = [np.reshape(a, (1, 1)) for a in y]

training_data = zip(x, y)

net = network.Network([2,5,1])
net.SGD(training_data, 30, 10, 3.0)



