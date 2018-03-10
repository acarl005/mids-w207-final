import numpy as np
from random import uniform
from math import floor

def one_hot(labels):
    max_label = np.max(labels)
    one_hot_encoded = np.zeros((len(labels), max_label + 1))
    one_hot_encoded[np.arange(len(labels)), labels] = 1
    return one_hot_encoded

def shuffle_n(*args):
    rand_perm = np.random.permutation(len(args[0]))
    return [np.array(x)[rand_perm] for x in args]

# def shuffle(x, y):
#     rand_perm = np.random.permutation(len(x))
#     x = x[rand_perm]
#     y = y[rand_perm]
#     return x, y

def shuffle(x, y):
    for i in range(len(x) - 1, 0, -1):
        j = floor(uniform(0, i + 1))
        x[[j, i]] = x[[i, j]]
        y[[j, i]] = y[[i, j]]
    return x, y

def get_batch(x, y, i, batch_size):
    index = i * batch_size
    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    return x_batch, y_batch

