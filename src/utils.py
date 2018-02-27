import numpy as np
from math import floor

def one_hot(labels):
    max_label = np.max(labels)
    one_hot_encoded = np.zeros((len(labels), max_label + 1))
    one_hot_encoded[np.arange(len(labels)), labels] = 1
    return one_hot_encoded

def shuffle_and_split(data, labels, dev_fraction=0.2):
    rand_ind = np.random.permutation(len(data))
    shuf_data = data[rand_ind]
    shuf_labels = labels[rand_ind]
    num_dev_examples = floor(len(data) * dev_fraction)
    return (shuf_data[:-num_dev_examples],
            shuf_labels[:-num_dev_examples],
            shuf_data[-num_dev_examples:],
            shuf_labels[-num_dev_examples:])

