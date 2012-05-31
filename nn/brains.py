"""
brains.py

Author: Isaac Hodes
Date: May 2012
Requires: Python 2.7

A backpropagating, 3-layer neural network. Algorithm from Norvig & Russel.

"""
import math, random
from linal import *

def _g(x): return math.tanh(x) # our step fn approx
def _g_(x): return 1.0 - math.tanh(x)**2 # derivative of _g()

class NeuralNetwork(object):
    """
    A three-layer complete neural network with backpropagation training.
    """
    def __init__(self, inputs, hidden, outputs):
        self.num_inputs = inputs + 1 # NB: + 1 for bias
        self.num_hidden = hidden
        self.num_outputs = outputs

        self.hidden = matrix(self.num_inputs, self.num_hidden)
        self.outputs = matrix(self.num_hidden, self.num_outputs)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialized all weights to random values between -1 and 1."""
        for node in xrange(self.num_hidden):
            for w in xrange(self.num_inputs):
                self.hidden[node][w] = random.uniform(-1.0, 1.0)

        for node in xrange(self.num_outputs):
            for w in xrange(self.num_hidden):
                self.outputs[node][w] = random.uniform(-1.0, 1.0)

    def backpropagate(self, training_set, alpha):
        """Trains the neural network n times on each data vector."""
        for x, y in training_set:
            x = x + [1]
            
            hidden_ins = [dot(x, weights) for weights in self.hidden]
            hidden_as = [_g(s) for s in hidden_ins]

            output_ins = [dot(hidden_as, weights) for weights in self.outputs]
            output_as = [_g(s) for s in output_ins]

            output_deltas = [_g_(output_ins[i]) * (y[i] - output_as[i])
                             for i in xrange(self.num_outputs)]

            hidden_deltas = [_g_(hidden_ins[i]) * sum([self.outputs[end_node][i] * output_deltas[end_node]
                                                       for end_node in xrange(self.num_outputs)])
                             for i in xrange(self.num_hidden)]

            # update hidden -> output weights
            for node in xrange(self.num_outputs):
                for j in xrange(self.num_hidden):
                    self.outputs[node][j] = self.outputs[node][j] + alpha * hidden_as[j] * output_deltas[node]

            # update input -> hidden weights
            for node in xrange(self.num_hidden):
                for j in xrange(self.num_inputs):
                    self.hidden[node][j] = self.hidden[node][j] + alpha * x[j] * hidden_deltas[node]

    def run(self, x):
        """Runs the neural network with the given inputs, and returns the
        outputs"""
        x = x + [1]
        hidden_as = [_g(dot(x, w)) for w in self.hidden]
        output_as = [_g(dot(hidden_as, w)) for w in self.outputs]

        return output_as

    def train(self, training_set, epochs=1000, alpha=0.2):
        for _ in xrange(epochs):
            self.backpropagate(training_set, alpha)

    def error(self, test_set):
        """Returns the test set error."""
        return sum([sum([0.5*(yi - oi)**2 for oi, yi in zip(self.run(x), y)])
                    for x,y in test_set])

    def test(self, test_set):
        for x,y in test_set:
            print x, '->', self.run(x), '\n\tshould be -> ', y
            print

    def __str__(self):
        return "WEIGHTS\ninputs: {}\nhidden: {}\noutputs: {}".format(len(self.hidden[0]),self.hidden, self.outputs)

if __name__ == '__main__':
    # AND, OR, XOR, NOR training set
    training_set = [([1, 1], [1, 1, 0, 0]), ([1, 0], [0, 1, 1, 0]), ([0, 1], [0, 1, 1, 0]), ([0, 0], [0, 0, 0, 1])]
    test_set = training_set

    nn = NeuralNetwork(2, 3, 4)
    epochs = 1000
    print "error before training: ", nn.error(test_set)
    nn.train(training_set, epochs, 0.5)
    print "error after training: ", nn.error(test_set)
    nn.test(test_set)
