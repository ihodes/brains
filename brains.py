"""
brains.py

Author: Isaac Hodes
Date: May 2012
Requires: Python 2.7

A simple, 2-layer, neural network.

"""

import math

def dot(a,b):
    """Returns the dot products of a and b"""
    return sum([i*j for (i,j) in zip(a,b)])

def logistic1(x):
    """Returns the value at x of the logistic function."""
    return 1/(1 + math.exp(-x))

def logistic_(x):
    """Returns the value at x of the derivative of logistic."""
    return math.exp(x)/((1 + math.exp(x))**2)

class NeuralNetwork(object):
    def __init__(self, inputs=[], hidden={}, outputs={}):
        """
        inputs of the form ["inputName1", "inputName2 ... ]
        hidden of the form {0:{'inputs': [0,3,4], 'weights': [1.1,2.3,7.1], 't':1.5},
                            1: { ... } ... }
        output of the form {0:{'hidden': [0,1], 'weights': [-0.8, 2.1], 't':-2.3}, 
                            1: { ... } ... }
        """
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

    def add_input(self, name):
        self.inputs.append(name)

    def add_hidden(self, inputs, weights, threshold):
        self.hidden[len(self.hidden)] = {'inputs':inputs, 'weights':weights,
                                         't':threshold}

    def add_output(self, hidden, weights, threshold):
        self.outputs[len(self.outputs)] = {'inputs':hidden, 'weights':weights,
                                           't':threshold}

    def remove_input(self, name):
        idx = self.inputs.index(name)
        self.inputs.remove(name)
        for _,hidden in self.hidden.iteritems():
            if idx in hidden['inputs']:
                hidden['inputs'].remove(idx)

    def remove_hidden(self, idx):
        del self.hidden[idx]
        for _,output in self.outputs.iteritems():
            if idx in output['inputs']:
                output['inputs'].remove(idx)

    def remove_output(self, idx):
        del self.outputs[idx]

    def run(self, input_values):
        # calculate the value of the hidden nodes
        hidden_values = []
        for _,hidden in self.hidden.iteritems():
            req_ins = [1] + [input_values[i] for i in hidden['inputs']]
            req_weights = [hidden['t']] + hidden['weights']
            val = logistic(dot(req_ins, req_weights))
            hidden_values.append(val)

        # calculate the value of the output nodes
        output_values = []
        for _,output in self.outputs.iteritems():
            req_ins = [1] + [hidden_values[i] for i in output['inputs']]
            req_weights = [output['t']] + output['weights']
            val = logistic(dot(req_ins, req_weights))
            output_values.append(val)
        return hidden_values,output_values

    def calculate_outputs(self, input_values):
        """Runs the neural network with the given inputs, and returns the
        outputs"""
        _,output_values = self.run(input_values)
        return output_values

    def train(self,training_set,alpha,n=1,tuning=[]):
        """Trains the neural network n times on each data vector,
        or until the tuning set error begins to increase."""

        deltas = [0]*len(training_set[1][1])
        for _ in range(n):
            for example in training_set:
                inputs, outputs = example
                hidden_values, output_values = self.run(inputs)
                for layer in (self.hidden, self.output):
                    pass
                for idx,node in enumerate(self.output):
                    delta[idx] = logistic_(hidden_values[idx]) * (outputs[idx] - logistic(hidden_values[idx]))
                continue
        ## this is what needs to be finished in order for it all to work

    def test(self,testing):
        """Returns the test set error."""
        return sum([self.error(t[0], t[1]) for t in testing])

    def error(self, inputs, outputs):
        return 0.5 * sum([i - logistic(j),
                          for i,j in zip(outputs, self.calculate_outputs(inputs))])**2

    def clone(self):
        """Returns a new NeuralNetwork object instantiated with the same values that self contains."""
        return NeuralNetwork(self.inputs[:], self.hidden.copy(),
                             self.outputs.copy())


def _test():
    a = NeuralNetwork()
    a.add_input("x")
    a.add_input("lol")
    a.add_input("y")

    a.add_hidden([0,1,2],[0.4,-2.3,-7], 1.1)

    a.add_output([0], [-5.5], -6.1)

    a.remove_input("lol")

    print a.calculate_outputs([1,2,3])

if __name__ == "__main__":
    _test()
