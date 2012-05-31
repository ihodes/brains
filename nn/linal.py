import numpy

def dot(a,b):
    sum = 0
    for i in xrange(len(a)):
        sum += a[i]*b[i]
    return sum

# def dot(a,b): # slower
#     return sum([i*j for (i,j) in zip(a,b)]);

def matrix(rows, cols):
    return [[0.0] * rows for _ in range(cols)]
