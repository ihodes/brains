def dot(a,b):
    return sum([i*j for (i,j) in zip(a,b)])

def matrix(rows, cols):
    return [[0.0] * rows for _ in range(cols)]
