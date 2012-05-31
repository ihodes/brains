"""
main.py

Author: Isaac Hodes
Date: May 2012
Requires: Python 2.7, matplotlib

"""
from nn.big_brains import *
import random as r
import argparse

def gen_pts(xlim, ylim, n):
    return [[r.uniform(-xlim,xlim), r.uniform(-ylim,ylim)] for _ in xrange(n)]

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

def str_fn(s):
    "Converts a string representing a function in one variable to a function. Trusts user input"
    exec "fn = lambda x: " + s
    return fn

def save_nn(nn, outfile):
    """Saves the neural network `nn` to `outfile`"""
    f = open(outfile, 'w')
    f.write(str(nn.num_inputs) + "\n")
    f.write(str(nn.num_hidden) + "\n")
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
                        nargs='+', metavar="nn_spec")
    parser.add_argument("function_spec", help="function in one variable," +
                        " x, which classifies the data",
                        type=str_fn)

    # size of training, testing, tuning sets
    parser.add_argument("-n","--trainn", help="number of training points",
                        metavar="training_size")
    parser.add_argument("-t","--testn", help="number of testing points",
                        metavar="testing_size")
    parser.add_argument("-u","--tunen", help="number of tuning points",
                        metavar="tuning_size")

    # range for data (default -1,1)
    parser.add_argument("--xlim", help="[-x,x] range for data generation",
                        metavar="xlim", default=1, type=int)
    parser.add_argument("--ylim", help="[-y,y] range for data generation",
                        metavar="ylim", default=1, type=int)

    # stopping settings
    parser.add_argument("-g","--targete", help="target error (default is `epochs`)",
                        metavar="target_error")
    parser.add_argument("-e","--epochs", help="the (max) number of epochs",
                        metavar="number_epochs")

    # img output
    parser.add_argument("-p","--png-output", help="specify png output",
                        metavar="png_output_file", default=None)

    # save & load
    parser.add_argument("-l","--load-nn", help="load network from file",
                        metavar="in_nn", type=load_nn, default=None)
    parser.add_argument("-s","--save-nn", help="save network to file",
                        metavar="output_file", default=None)

    # verbose & other
    parser.add_argument("-v","--verbose",
                        help="print errors and other information",
                        action="store_false")

    return parser

def main():
    args = build_parser().parse_args()
    
    try: args.graph_spec = [int(i) for i in args.nn_spec]
    except: raise ValueError("nn-spec must be a list of integer values")

    if args.png_output: # or other img-related vars
        try: import matplotlib.pyplot as p
        except: ImportError("matplotlib must be installed in order to display graphs")
    

    nn = NeuralNetwork(args.graph_spec[0],
                       args.graph_spec[1:-1],
                       args.graph_spec[-1])

    if args.save_nn: save_nn(nn, args.save_nn)
    
    
    pass

if __name__ == '__main__':
    import cProfile

    main()

    # fn = lambda x: x**2

    # test_pts = gen_pts(4,4, 10000)
    # training_pts = gen_pts(4,4, 100020)

    # test_set = classify_pts(fn, test_pts)
    # training_set = classify_pts(fn, training_pts)

    # nn = NeuralNetwork(2,25,1)

    # print "error before training: ", nn.error(test_set)
    # cProfile.run("nn.train(training_set, 1, 0.2)")
    # print "error after training: ", nn.error(test_set)
    # print

    # above_ptsx = []
    # above_ptsy = []
    # below_ptsy = []
    # below_ptsx = []
    # for (example,_) in test_set:
    #     if nn.run(example)[0] >= 0.5: # arbitrary cutoff
    #         above_ptsx.append(example[0])
    #         above_ptsy.append(example[1])
    #     else: 
    #         below_ptsx.append(example[0])
    #         below_ptsy.append(example[1])
    
    # p.scatter(above_ptsx, above_ptsy, color='purple')
    # p.scatter(below_ptsx, below_ptsy, color='blue')
    # p.scatter([x*.2 for x in range(-10,10)],[(x*.2)**2 for x in range(-10,10)], color='red')
    # p.show()

    #r.shuffle(test_set)
    #nn.test(test_set[0:10])
