"""
brains.py

Author: Isaac Hodes
Date: May 2012
Requires: Python 2.7

"""
import math, random
from linal import *

def g(x): return math.tanh(x)
def g_(x): return 1.0 - math.tanh(x)**2

class NeuralNetwork(object):
    """
    An arbitrary layer complete neural network with backpropagation training.
    """
    def __init__(self, inputs, hidden_spec, outputs):
        """Hidden spec is a list of hidden layer lengths"""
        self.L = len(hidden_spec) # number of hidden layers
        
        self.num_inputs = inputs + 1 # NB: + 1 for bias
        self.num_hidden = hidden_spec
        self.num_outputs = outputs

        self.hidden = []
        self.hidden.append(matrix(self.num_hidden[0], self.num_inputs))
        for l in range(1,self.L):
            self.hidden.append(matrix(self.num_hidden[l-1], self.num_hidden[l]))

        self.outputs = matrix(self.num_hidden[len(hidden_spec)-1], self.num_outputs)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialized all weights to random values between -1 and 1."""
        for node in range(self.num_hidden[0]):
                for w in range(self.num_inputs):
                    self.hidden[0][node][w] = random.uniform(-1.0, 1.0)
        for l in range(1,self.L):
            for node in range(self.num_hidden[l]):
                for w in range(self.num_hidden[l-1]):
                    self.hidden[l][node][w] = random.uniform(-1.0, 1.0)
        for node in range(self.num_outputs):
            for n in range(self.num_hidden[self.L-1]):
                self.outputs[node][n] = random.uniform(-1.0, 1.0)

    def backpropagate(self, training_set, epochs=100, alpha=0.5):
        """Trains the neural network n times on each data vector."""
        for x, y in training_set:
            x = x + [1]
            
            ins = x[:]
            hidden_ins = [[0.0] * num_weights for num_weights in self.num_hidden]
                
            for l in range(self.L):
                layer = self.hidden[l]
                for node_idx in range(self.num_hidden[l]):
                    weights = layer[node_idx]
                    hidden_ins[l][node_idx] = dot(ins, weights)
                ins = hidden_ins[l][:]
                    
            hidden_as = [[0.0] * num_weights for num_weights in self.num_hidden]
            for l in range(self.L):
                for node_idx in range(self.num_hidden[l]):
                    hidden_as[l][node_idx] = g(hidden_ins[l][node_idx])

            output_ins = [dot(hidden_as[self.L-1], weights) for weights in self.outputs]
            output_as = [g(s) for s in output_ins]

            output_deltas = [g_(output_ins[i]) * (y[i] - output_as[i])
                             for i in range(self.num_outputs)]

            hidden_deltas = [[0.0] * num_weights for num_weights in self.num_hidden]
            for layer_idx in range(self.L):
                l = self.L - layer_idx - 1 # going backwards
                if layer_idx == 0:  # then we're backpropping from the output layer
                    hidden_deltas[l] = [g_(hidden_ins[l][i]) * sum([self.outputs[j][i] * output_deltas[j]
                                                                    for j in range(self.num_outputs)])
                                        for i in range(self.num_hidden[l])]
                else: # we're propagating from an inner hidden layer
                    for node_idx in range(self.num_hidden[l]):
                        d = 0.0
                        for end_node_idx in range(self.num_hidden[l+1]):
                            d += self.hidden[l+1][end_node_idx][node_idx] * hidden_deltas[l+1][end_node_idx]
                        hidden_deltas[l][node_idx] = g_(hidden_ins[l][node_idx]) * d
                    
            # update hidden -> output weights
            for node in range(self.num_outputs):
                for j in range(self.num_hidden[self.L-1]):
                    ch = alpha * hidden_as[self.L-1][j] * output_deltas[node]
                    self.outputs[node][j] = self.outputs[node][j] + ch
            # update input -> hidden 
            for node in range(self.num_hidden[0]):
                for j in range(self.num_inputs):
                    ch = alpha * x[j] * hidden_deltas[0][node]
                    self.hidden[0][node][j] = self.hidden[0][node][j] + ch
            # update hidden -> hidden weights
            for l in range(1,self.L):
                for node in range(self.num_hidden[l]):
                    for j in range(self.num_hidden[l-1]):
                        ch = alpha * hidden_as[l-1][j] * hidden_deltas[l][node]
                        self.hidden[l][node][j] = self.hidden[l][node][j] + ch

    def run(self, x):
        """Runs the neural network with the given inputs, and returns the
        outputs"""
        x = x + [1]

        ins = x[:]
        hidden_ins = [[0.0] * num_weights for num_weights in self.num_hidden]
        
        for l in range(self.L):
            layer = self.hidden[l]
            for node_idx in range(self.num_hidden[l]):
                weights = layer[node_idx]
                hidden_ins[l][node_idx] = dot(ins, weights)
            ins = hidden_ins[l][:]

        hidden_as = [[0.0] * num_weights for num_weights in self.num_hidden]
        for l in range(self.L):
            for node_idx in range(self.num_hidden[l]):
                hidden_as[l][node_idx] = g(hidden_ins[l][node_idx])

        output_ins = [dot(hidden_as[self.L-1], weights) for weights in self.outputs]
        output_as = [g(s) for s in output_ins]
        
        return output_as

    def error(self, test_set):
        """Returns the test set error."""
        return sum([sum([0.5*(yi - oi)**2 for oi, yi in zip(self.run(x), y)])
                    for x,y in test_set])

    def test(self, test_set):
        for x,y in test_set:
            print x, '->', self.run(x), '\n should be -> ', y

    def train(self, training_set, epochs=1000, alpha=0.2):
        for _ in xrange(epochs):
            self.backpropagate(training_set, alpha)

    def __str__(self):
        return "WEIGHTS\ninputs: {}\nhidden: {}\noutputs: {}".format(len(self.hidden[0]),self.hidden, self.outputs)

def test(epochs=10000):
    # AND, OR, XOR, NOR training set
    training_set = [([1, 1], [1, 1, 0, 0]), ([1, 0], [0, 1, 1, 0]), ([0, 1], [0, 1, 1, 0]), ([0, 0], [0, 0, 0, 1])]
    test_set = training_set

    nn = NeuralNetwork(2, [3,3], 4)

    print
    print "error before training: ", nn.error(test_set)

    nn.train(training_set, epochs, 0.5)

    print "error after training: ", nn.error(test_set)
    print

    nn.test(test_set)

if __name__ == '__main__': test()
