import numpy as np
from random import uniform
from math import floor

def one_hot(labels):
    max_label = np.max(labels)
    one_hot_encoded = np.zeros((len(labels), max_label + 1))
    one_hot_encoded[np.arange(len(labels)), labels] = 1
    return one_hot_encoded

def shuffle_n(*args):
    """
    shuffles n number of arrays in the same way, but very memory-inefficient as there is unecessary copying
    """
    rand_perm = np.random.permutation(len(args[0]))
    return [np.array(x)[rand_perm] for x in args]

# this implementation caused out-of-memory exception when called on an array with 64000 images in it
# def shuffle(x, y):
#     rand_perm = np.random.permutation(len(x))
#     x = x[rand_perm]
#     y = y[rand_perm]
#     return x, y

def shuffle(x, y):
    """
    a more memory efficient shuffle implementation. This does it in-place using the Fisher-Yates shuffle.
    no array copying necessary. it works on arrays with 64000 images in it
    """
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

def arg_top_n(ary, n):
    """
    Returns the indices for the n largest values from a numpy array.
    Like np.argmax, but works with the top n instead of just one maximum
    """
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def remove_ticks(plot):
    plot.axes.get_xaxis().set_ticks([])
    plot.axes.get_yaxis().set_ticks([])
