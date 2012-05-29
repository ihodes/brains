"""
brains.py

Author: Isaac Hodes
Date: May 2012
Requires: Python 2.7

"""
import math, random

DEBUG = False
if DEBUG: import pprint


# math utilities 
def dot(a,b):
    """Returns the dot products of a and b"""
    assert(len(a) == len(b)), "Length of a and b must be the same"
    return sum([i*j for (i,j) in zip(a,b)])

def g(x):
    return 1/(1+math.exp(-x))

def g_(x):
    return (1-g(x))*g(x)


class NeuralNetwork(object):
    """
    A three-layer complete neural network with backpropagation training.
    """
    def __init__(self, inputs, hidden, outputs):
        self.num_inputs = inputs # NB: + 1 for bias
        self.num_hidden = hidden # NB: + 1 for bias
        self.num_outputs = outputs
        self.hidden = [[0] * (inputs+1)] * hidden
        self.outputs = [[0] * (hidden+1)] * outputs
        self.initialize_weights()

    def calculate_outputs(self, x):
        """Runs the neural network with the given inputs, and returns the
        outputs"""
        x = [1] + x
        hidden_ins = [dot(x, w) for w in self.hidden]
        hidden_as = [g(i) for i in hidden_ins]
        hidden_as = [1] + hidden_as

        output_ins = [dot(hidden_as, w) for w in self.outputs]
        output_as = [g(i) for i in output_ins]

        return output_as

    def initialize_weights(self):
        """Initialized all weights to random values between 0 and 1."""
        temp_layer = []
        for weights in self.hidden: 
            temp_weights = []
            for w in weights:
                w = random.uniform(-1,1)
                temp_weights.append(w)
            temp_layer.append(temp_weights)
        self.hidden = temp_layer[:]

        temp_layer = []
        for weights in self.outputs: 
            temp_weights = []
            for w in weights:
                w = random.uniform(-1,1)
                temp_weights.append(w)
            temp_layer.append(temp_weights)
        self.outputs = temp_layer[:]

    def train(self,training_set,epochs=1,alpha=.2,tuning=[]):
        """Trains the neural network n times on each data vector,
        or until  the tuning set error begins to increase."""
        for _ in xrange(epochs):
            for x, y in training_set:
                hidden_ins = [dot([1] + x, weights) for weights in self.hidden]
                hidden_as = [g(i) for i in hidden_ins]

                output_ins = [dot([1] + hidden_as, weights) for weights in self.outputs]
                output_as = [g(i) for i in output_ins]

                output_deltas = [g_(output_ins[i]) * (y[i] - output_as[i])
                                 for i in xrange(self.num_outputs)]

                # no delta for the bias (=1) node (not a real node; has no weights)
                hidden_deltas = [g_(hidden_ins[i]) * sum([self.outputs[j][i+1] * output_deltas[j]
                                                          for j in xrange(self.num_outputs)])
                                 for i in xrange(self.num_hidden)]

                hidden_ins = [1] + hidden_ins  # need to update weights from bias node
                # update hidden -> output weights
                for i in xrange(self.num_outputs):
                    for j in xrange(self.num_hidden+1): # (+ 1 bias node) weights from jth hidden to ith node
                        self.outputs[i][j] = self.outputs[i][j] + alpha * hidden_ins[j] * output_deltas[i]

                x = [1] + x # need to update weights from bias node, so we include it here
                # update input -> hidden weights
                for i in xrange(self.num_hidden):
                    for j in xrange(self.num_inputs+1): # (+ 1 bias node) weights from jth x to ith node
                        self.hidden[i][j] = self.hidden[i][j] + alpha * x[j] * hidden_deltas[i]

    def test(self, test_set):
        """Returns the test set error."""
        return sum([sum([0.5*(yi - oi)**2 for oi, yi in zip(self.calculate_outputs(x), y)]) 
                    for x,y in test_set])

    def __str__(self):
        return "WEIGHTS\ninputs: {}\nhidden: {}\noutputs: {}".format(len(self.hidden[0]),self.hidden, self.outputs)

def _test_NN():
    EPOCHS = 10000

    # >= x^2 training set
    #
    # goods = [([x, x**2+random.uniform(0,10)], [1]) for x in [i*.1 for i in  range(10)] for _ in range(100)]
    # bads = [([x, x**2-random.uniform(0,10)], [0]) for x in [i*.1 for i in  range(10)] for _ in range(100)]
    # training_set = goods + bads

    # AND, OR, XOR training set
    # training_set = [([1, 1], [1, 1, 0]), ([1, 0], [0, 1, 1]), ([0, 1], [0, 1, 1]), ([0, 0], [0, 0, 0])]

    # XOR training set
    # training_set = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

    # OR training set
    training_set = [([1, 1], [1]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])] 

    # AND training set
    # training_set = [([1, 1], [1]), ([1, 0], [0]), ([0, 1], [0]), ([0, 0], [0])]

    # NOR training set
    # training_set = [([1, 1], [0]), ([1, 0], [0]), ([0, 1], [0]), ([0, 0], [1])] 

    a = NeuralNetwork(2,2,1)
    a.hidden = [[.5,.5,-.25],[.33,.22,-.11]]
    a.outputs = [[.1,-.25,.5]]

    print
    print "epochs = ", EPOCHS
    print
    print "================================================================================"
    print "================================================================================"
    print "===== Untrained on training_set              total error = ", a.test(training_set), "====="
    print "================================================================================"
    for (x,y) in training_set:
        print "for input:        ", x
        print "expected outputs: ", y
        print "actual outputs:   ", a.calculate_outputs(x)
        print "error:            ", 0.5 * (sum([yi - oi for oi, yi in zip(a.calculate_outputs(x), y)])**2)
        print "--------------------------------------------------------------------------------"

    a.train(training_set, EPOCHS)

    print
    print "================================================================================"
    print "================================================================================"
    print "===== Results on training_set                total error = ", a.test(training_set), "====="
    print "================================================================================"
    for (x,y) in training_set:
        print "for input:        ", x
        print "expected outputs: ", y
        print "actual outputs:   ", a.calculate_outputs(x)
        print "error:            ", 0.5 * (sum([yi - oi for oi, yi in zip(a.calculate_outputs(x), y)])**2)
        print "--------------------------------------------------------------------------------"    

    print a

def _test():
    assert(dot([1,2,3],[2,3,1]) == 11), "Dot product is incorrect"

if __name__ == "__main__":
    _test_NN()

                # if DEBUG:
                #     print "--------------------------------------------------------------------------------"
                #     print "x:", x
                #     print "y:", y
                #     print
                #     print "hidden weights"
                #     pprint.pprint(self.hidden)
                #     print "output weights"
                #     pprint.pprint(self.outputs)
                #     print
                #     print "hidden_ins", hidden_ins
                #     print "hidden_as", hidden_as
                #     print
                #     print "output_ins", output_ins
                #     print "output_as", output_as
                #     print
                #     print "hidden_deltas", hidden_deltas
                #     print "output_deltas", output_deltas
                #     print



## run through a simple one by hand, 1 iteration, set weights to be what they are in the java applet
## make a perceptron

