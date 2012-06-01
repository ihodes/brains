"""
main.py

Author: Isaac Hodes
Date: May 2012
Requires: Python 2.7, matplotlib

Example usage:

python main.py --function-spec "math.log(x)" --r-xlim 3 --l-xlim 1 -t -s 'nn.txt' --epochs 100 --nn-spec "2 3 3 1"
python main.py -lnn.txt --function-spec "math.log(x)" --r-xlim 3 --l-xlim 1 --test-png t.png

"""
from nn.big_brains import *
import random as r
import argparse
import math

def gen_pts(xlims, ylims, n):
    (l_xlim, r_xlim) = xlims
    (l_ylim, r_ylim) = ylims
    return [[r.uniform(l_xlim,r_xlim), r.uniform(l_ylim,r_ylim)] for _ in xrange(n)]

def classify_pts(fn, pts):
    """Classifies `pts` as 1 if it falls above or on the given function,
    `fn`, else 0. `fn` is a numeric function which must take a value x
    and return a value y; `pts` is a list of x,y lists."""
    classification = []
    for x,y in pts:
        if y >= fn(x): c = 1
        else: c = 0
        classification.append([[x,y], [c]])
    return classification

def nn_list(s):
    return [int(i) for i in s.split()]

def str_fn(s):
    """Converts a string representing a function in one variable to a function.
    Trusts user input"""
    exec "fn = lambda x: " + s
    return fn

def save_nn(nn, outfile):
    """Saves the neural network `nn` to `outfile`"""
    f = open(outfile, 'w')
    f.write(str(nn.num_inputs-1) + "\n")
    f.write(str([n-1 for n in nn.num_hidden]) + "\n")
    f.write(str(nn.num_outputs) + "\n")
    f.write(str(nn.hidden) + "\n")
    f.write(str(nn.outputs) + "\n")
    f.close()

def load_nn(infile):
    """Loads a neural network from `infile`"""
    f = open(infile, 'r')

    num_inputs = int(f.readline())
    exec "hidden_spec = " + f.readline()
    num_outputs = int(f.readline())
    exec "hidden = " + f.readline()
    exec "outputs = " + f.readline()

    f.close()

    nn = NeuralNetwork(num_inputs,hidden_spec,num_outputs)
    nn.outputs = outputs
    nn.hidden = hidden

    return nn

def build_parser():
    # should be able to save a neural network to a text file, and initialize one from that same format
    parser = argparse.ArgumentParser(description = "Creates and trains a neural network." + 
                                     " Can load network from file, or construct and further" + 
                                     " train a specific network. Output can be saved to file" + 
                                     ", tested, and displayed with matplotlib",
                                     epilog = "A graph_spec or input file must " +
                                     "be specified in order to construct the network.")

    # nn and fn specs
    parser.add_argument("--nn-spec", help="neural network layout: " +
                        "list of nodes in each layer.",
                        metavar="nn_spec", type=nn_list)
    parser.add_argument("--function-spec", help="function in one variable," +
                        " x, which classifies the data. Can use functions from" + 
                        "`math` module.",
                        type=str_fn, metavar="fn")

    # actions (what do we do...?)
    parser.add_argument("-t","--train", help="train the neural network with the given options",
                        default=False, action='store_true')
    parser.add_argument("-r","--run", help="run neural network on given data" +
                        "(a Python list of input lists)")

    # size of training, testing
    parser.add_argument("--trainn", help="number of training points",
                        metavar="training_size", default=1000, type=int)
    parser.add_argument("--testn", help="number of testing points",
                        metavar="testing_size", default=1000, type=int)

    # range for data (default -1,1)
    parser.add_argument("--r-xlim", help="right x limit for data generation",
                        metavar="xlim", default=1, type=int)
    parser.add_argument("--l-xlim", help="left x limit for data generation",
                        metavar="xlim", default=-1, type=int)
    parser.add_argument("--r-ylim", help="right y limit for data generation",
                        metavar="ylim", default=1, type=int)
    parser.add_argument("--l-ylim", help="left y for data generation",
                        metavar="ylim", default=-1, type=int)

    # stopping settings (one of them must be specified)
    parser.add_argument("--targete", help="target error (default behavior is running `epochs` times)",
                        metavar="target_error") # TODO
    parser.add_argument("--tuningn", help="specify a size for the tuning set to be" + 
                        "used to stop training",
                        metavar="tuning_n", type=int) # TODO
    parser.add_argument("--epochs", help="the (max) number of epochs",
                        metavar="number_epochs", default=1000, type=int)

    # png output (improve this?)
    parser.add_argument("--training-png", help="specify training set png",
                        metavar="training_png_file", default=None)
    parser.add_argument("--test-png", help="specify test set png",
                        metavar="test_png_file", default=None)

    # save & load
    parser.add_argument("-l","--load-nn", help="load network from file",
                        metavar="input_file", type=load_nn, default=None)
    parser.add_argument("-s","--save-nn", help="save network to file",
                        metavar="output_file", default=None)

    # verbose & other
    parser.add_argument("-v","--verbose",
                        help="print errors and other information",
                        action="store_false")
    parser.add_argument("-d","--debug",
                        help="print nn and other information",
                        default=False, action='store_true')
    parser.add_argument("--alpha", help="alpha for backpropagation",
                        metavar="alpha", default=0.5, type=float)

    return parser

def main():
    args = build_parser().parse_args()

    # load our neural network
    if args.nn_spec:
        try: args.nn_spec = [int(i) for i in args.nn_spec]
        except: raise ValueError("nn-spec must be a list of integer values")
        nn = NeuralNetwork(args.nn_spec[0],
                           args.nn_spec[1:-1],
                           args.nn_spec[-1])
    elif args.load_nn: nn = args.load_nn
    else: raise Exception("Must specify a neural network.")

    # load the classification function & generate data
    fn = args.function_spec

    if fn: 
        test_set = classify_pts(fn, gen_pts((args.l_xlim,args.r_xlim),
                                            (args.l_ylim,args.r_ylim),
                                            args.testn))
        training_set = classify_pts(fn, gen_pts((args.l_xlim,args.r_xlim),
                                                (args.l_ylim,args.r_ylim),
                                                args.trainn))

    # not implemented
    if args.tuningn:
        training_set = classify_pts(fn, gen_pts((args.l_xlim,args.r_xlim),
                                                (args.l_ylim,args.r_ylim),
                                                args.args.tuningn))
        print "tuning not implemented"

    elif args.targete:
        target_error = args.targete

    if args.debug: 
        print nn        

    if args.train:
        print "before training error: ", nn.error(test_set)
        nn.train(training_set, args.epochs, args.alpha)
        if args.debug: print nn

    if args.run:
        exec "run_set = " + args.run
        nn.run_on(run_set)
    
    if args.train: print "neural network error: ", nn.error(test_set)

    # image settings
    if args.training_png or args.test_png:
        if not fn: raise Exception("Need to specify a function in order to classify data")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as p
            import numpy as np
        except: ImportError("matplotlib & numpy must be installed in order to display graphs")
        x = np.arange(args.l_xlim,args.r_xlim,(-args.l_xlim + args.r_xlim)/2000.0)
        y = [fn(i) for i in x]
        if args.training_png:
            p.clf()
            above_ptsx = []
            above_ptsy = []
            below_ptsy = []
            below_ptsx = []
            for (example,_) in training_set:
                if nn.run(example)[0] >= 0.5: # arbitrary cutoff
                    above_ptsx.append(example[0])
                    above_ptsy.append(example[1])
                else:
                    below_ptsx.append(example[0])
                    below_ptsy.append(example[1])
            p.xlim((args.l_xlim,args.r_xlim))
            p.ylim((args.l_ylim,args.r_ylim))
            p.scatter(above_ptsx, above_ptsy, color='#ee0000', s=3, marker="x")
            p.scatter(below_ptsx, below_ptsy, color='#0044ee', s=3, marker="x")
            p.plot(x, y, "g-", color="#009900", linewidth=1)
            p.grid()
            p.savefig(args.training_png)
        if args.test_png:
            p.clf()
            above_ptsx = []
            above_ptsy = []
            below_ptsy = []
            below_ptsx = []
            for (example,_) in test_set:
                if nn.run(example)[0] >= 0.5: # arbitrary cutoff
                    above_ptsx.append(example[0])
                    above_ptsy.append(example[1])
                else:
                    below_ptsx.append(example[0])
                    below_ptsy.append(example[1])
            p.xlim((args.l_xlim,args.r_xlim))
            p.ylim((args.l_ylim,args.r_ylim))
            p.scatter(above_ptsx, above_ptsy, color='#ee0000', s=3, marker="x")
            p.scatter(below_ptsx, below_ptsy, color='#0044ee', s=3, marker="x")
            p.plot(x, y, "g-", color="#009900", linewidth=1)
            p.grid()
            p.savefig(args.test_png)

    if args.save_nn: save_nn(nn, args.save_nn)

def demo():
    """A small demo exhibiting various features of the neural network."""
    import matplotlib.pyplot as p
    import numpy as np

    fn = lambda x: x**2

    test_pts = gen_pts(1,1, 10000)
    training_pts = gen_pts(1,1, 1000)

    test_set = classify_pts(fn, test_pts)
    training_set = classify_pts(fn, training_pts)

    # modify the below for the most interesting changes
    nn = NeuralNetwork(2,[5,4,2],1)
    epochs = 10

    print "error before training: ", nn.error(test_set)
    nn.train(training_set, epochs, 0.2)
    print "error after training: ", nn.error(test_set)
    print

    above_ptsx = []
    above_ptsy = []
    below_ptsy = []
    below_ptsx = []
    for (example,_) in test_set:
        if nn.run(example)[0] >= 0.5: # arbitrary cutoff
            above_ptsx.append(example[0])
            above_ptsy.append(example[1])
        else:
            below_ptsx.append(example[0])
            below_ptsy.append(example[1])

    p.figure("Classification results.")
    p.xlim((-1,1))
    p.ylim((-1,1))
    p.scatter(above_ptsx, above_ptsy, color='#0000ee', s=2, marker="x")
    p.scatter(below_ptsx, below_ptsy, color='#664433', s=2, marker="x")
    x = np.arange(-1,1,.001)
    p.plot(x, x**2, "g-", color="#009900", linewidth=1)
    p.grid()
    p.show()
    
    print "Classification of a random sample of the test set:"
    r.shuffle(test_set)
    nn.test(test_set[0:10])

if __name__ == '__main__':
    main()
