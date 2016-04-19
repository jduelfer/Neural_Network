"""
netowrk.py
~~~~~~~~~~

A module to implement the stochasic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropogation. It is not optimized and omits many desireable
features.
"""

import random
import numpy as np

#network object
class Network(object):
	def __init__(self, sizes):
		"""The lsit sizes contains the number of neurons in the respective
		layers of the network. For example, if the list was [2, 3, 1] then it
		would be a three layer network, with the first layer containing 2 neurons,
		the second layer containing 3 neurons, and the last layer containing 1 neuron.
		The biases and weights for the network are initially randomized, using a Gaussian
		distribution with mean 0, and variance 1. Note that the first layer is assumed to 
		be an input layer, and by convention we won't set any biases for those neurons,
		since biases are only ever used in computing the outputs from later layers."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	#creates a neural network with 2 neurons in first layer,
	#3 neurons in the second layer, and 1 in the final layer
	#net = Network([2, 3, 1])

	#it is assumed that the input is an (n, 1) numpy ndarray,
	#not an (n,) vector
	def feedforward(self, a):
		"""Return the output of the network if "a" is input"""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	#stochastic gradient descent
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		"""Train the networks using mini-batch stochastic gradient descent.
		The "training_data" is a list of tuples "(x, y)" representing the training
		inputs and the desired outputs. The other non-optimal parameters are
		self-explanatory. If test_data is provided, then the network will be 
		evaluated against the test data after each epoch, and partial progress
		printed out. This is useful for tracking process, but slows things down
		substantially."""
		#epoch: the number of epochs to train for
		#mini_batch_size: the size of mini_batches to use while sampling
		#eta: the learning rate 'n'(funky n)
		if test_data: n_test = len(test_data)
		#training data is a list of tuples(x, y) representing the training inputs
		#and corresponding desired outputs
		n = len(training_data)
		#for each epoch
		for j in xrange(epochs):
			#start by randomly shuffling the data
			random.shuffle(training_data)
			#partition the data into the appropriate sized mini_batches
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			#for each mini_batch
			for mini_batch in mini_batches:
				#apply a stingle step of gradient descent
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print "Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)


	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying gradient descent
		using backpropogation to a single mini batch. The 'mini_batch' is a list
		of tuples '(x, y)', and 'eta' is the learning rate."""
		#this function works by computing these gradients for every training
		#example in the mini_batch, and then updating self.weights and self, biases appropriately
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			#most of the work is done here, which invokes the backpropogation algorithm
			#fast way of computing the gradient of the cost function
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


	def backprop(self, x, y):
		"""Returns a tuple '(nabla_b, nabla_w)' representing the gradeint for
		the cost function C_x. 'nabla_b' and 'nabla_w' are layer-by-layer lists of
		numpy arrays, similar to self.biases and self.weigths."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		#feedforward
		activation = x
		#list to store all the activations, layer by layer
		activations = [x]
		#list to store all the z vectors, layer by layer
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		#backward pass
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural network outputs
		the correct result. Note that the neural network's output is assumed to be the 
		index of whichever neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""I don't understand this function at all"""
		return (output_activations-y)


###miscellaneous function
#when z is a vector, or numpy array, numpy automatically
#applies the function sigmoid elementwise (in vectorized form)
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function"""
	return sigmoid(z)*(1-sigmoid(z))