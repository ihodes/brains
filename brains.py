"""
brains.py

Author: Isaac Hodes
Date: May 2012
Requires: Python 2.7

"""
import math, random

# utilities 
def dot(a,b):
    return sum([i*j for (i,j) in zip(a,b)])

def g(x):
    return math.tanh(x)

def g_(x):
    return 1.0 - math.tanh(x)**2

def matrix(nodes, weights):
    return [[0.0] * weights for _ in range(nodes)]

class NeuralNetwork(object):
    """
    A three-layer complete neural network with backpropagation training.
    """
    def __init__(self, inputs, hidden, outputs):
        self.num_inputs = inputs + 1 # NB: + 1 for bias
        self.num_hidden = hidden
        self.num_outputs = outputs

        self.hidden = matrix(self.num_hidden, self.num_inputs)
        self.outputs = matrix(self.num_outputs, self.num_hidden)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialized all weights to random values between 0 and 1."""
        for node in range(self.num_hidden):
            for w in range(self.num_inputs):
                self.hidden[node][w] = random.uniform(-1.0, 1.0)

        for node in range(self.num_outputs):
            for w in range(self.num_hidden):
                self.outputs[node][w] = random.uniform(-1.0, 1.0)

    def train(self, training_set, epochs=1, alpha=0.5):
        """Trains the neural network n times on each data vector."""
        for _ in xrange(epochs):
            for x, y in training_set:
                x = x + [1]

                hidden_ins = [dot(x, weights) for weights in self.hidden]
                hidden_as = [g(s) for s in hidden_ins]

                output_ins = [dot(hidden_as, weights) for weights in self.outputs]
                output_as = [g(s) for s in output_ins]

                output_deltas = [g_(output_ins[i]) * (y[i] - output_as[i])
                                 for i in range(self.num_outputs)]

                hidden_deltas = [g_(hidden_ins[i]) * sum([self.outputs[j][i] * output_deltas[j]
                                                          for j in range(self.num_outputs)])
                                 for i in range(self.num_hidden)]

                # update hidden -> output weights
                for node in range(self.num_outputs):
                    for j in range(self.num_hidden):
                        self.outputs[node][j] = self.outputs[node][j] + alpha * hidden_as[j] * output_deltas[node]

                # update input -> hidden weights
                for node in range(self.num_hidden):
                    for j in range(self.num_inputs):
                        self.hidden[node][j] = self.hidden[node][j] + alpha * x[j] * hidden_deltas[node]

    def run(self, x):
        """Runs the neural network with the given inputs, and returns the
        outputs"""
        x = x + [1]
        hidden_as = [g(dot(x, w)) for w in self.hidden]
        output_as = [g(dot(hidden_as, w)) for w in self.outputs]

        return output_as

    def error(self, test_set):
        """Returns the test set error."""
        return sum([sum([0.5*(yi - oi)**2 for oi, yi in zip(self.run(x), y)])
                    for x,y in test_set])

    def test(self, test_set):
        for x,y in test_set:
            print x, '->', self.run(x), '\n should be -> ', y

    def __str__(self):
        return "WEIGHTS\ninputs: {}\nhidden: {}\noutputs: {}".format(len(self.hidden[0]),self.hidden, self.outputs)

def test(epochs=1000):
    # AND, OR, XOR, NOR training set
    training_set = [([1, 1], [1, 1, 0, 0]), ([1, 0], [0, 1, 1, 0]), ([0, 1], [0, 1, 1, 0]), ([0, 0], [0, 0, 0, 1])]
    test_set = training_set

    nn = NeuralNetwork(2, 3, 4)
    print "error before training: ", nn.error(test_set)
    nn.train(training_set, epochs, 0.5)
    print "error after training: ", nn.error(test_set)
    nn.test(test_set)

if __name__ == '__main__': test()
