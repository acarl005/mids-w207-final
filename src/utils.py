import numpy as np
from math import floor

def one_hot(labels):
    max_label = np.max(labels)
    one_hot_encoded = np.zeros((len(labels), max_label + 1))
    one_hot_encoded[np.arange(len(labels)), labels] = 1
    return one_hot_encoded

def shuffle_and_split(data, labels, dev_fraction=0.2):
    num_dev_examples = floor(len(data) * dev_fraction)
    shuf_data, shuf_labels = shuffle(data, labels)
    return (shuf_data[:-num_dev_examples],
            shuf_labels[:-num_dev_examples],
            shuf_data[-num_dev_examples:],
            shuf_labels[-num_dev_examples:])

def shuffle(x, y):
    rand_perm = np.random.permutation(len(x))
    x = x[rand_perm]
    y = y[rand_perm]
    return x, y

def get_batch(x, y, i, batch_size):
    index = i * batch_size
    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    return x_batch, y_batch

