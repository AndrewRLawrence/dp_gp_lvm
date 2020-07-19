"""
This module provides some basic expressions.
"""

import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
import tensorflow as tf


def print_and_log(*args, **kwargs):
    """
    TODO
    :param args:
    :param kwargs:
    """

    # Print to terminal.
    print(*args, **kwargs)

    # Create output text file or open existing file to append to it.
    # TODO: Allow for this file and location to be specified.
    with open('output.txt', 'a') as txt_file:
        # Print to file.
        print(*args, **kwargs, file=txt_file)


def nearest_neighbour(x_train, x_test):
    """
    TODO
    :param x_train: [N x D].
    :param x_test: [M x D].
    :return:
    """

    # Reshape inputs.
    x_train_n1d = tf.expand_dims(x_train, axis=1)
    x_test_1md = tf.expand_dims(x_test, axis=0)

    # Calculate L2 distance.
    distances = tf.norm(x_train_n1d - x_test_1md, axis=-1)  # [N x M].

    # Find shortest distances and return index of nearest neighbour for each x_test with respect to x_train.
    return tf.argmin(distances, axis=0)


def principal_component_analysis(x, num_latent_dimensions):
    """
    TODO
    :param x:
    :param num_latent_dimensions:
    :return:
    """

    # Confirm x is 2D numpy array.
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2

    # Validate sizes.
    n, d = x.shape
    assert 0 < num_latent_dimensions < min(n, d), \
        'Number of latent dimensions must be greater than zero and less than the minimum of the number of ' \
        'observations and the number of observed dimensions.'

    # Use regular eigen decomposition if not that many observations.
    if n < (num_latent_dimensions + 2):
        w, v = eig(np.dot(x, x.T))
        v = v[:, :num_latent_dimensions]
    else:
        w, v = eigs(np.dot(x, x.T), k=num_latent_dimensions)

    x_0 = np.real(v)
    x_0 = x_0 / np.mean(x_0.std(axis=0, ddof=1))

    # return PCA(n_components=num_latent_dimensions).fit_transform(x)
    return x_0


def empirical_mean(x):
    """
    This function calculates the empirical mean of x, where the first dimension is the number of samples.
    :param x: TODO
    :return:
    """

    return tf.reduce_mean(x, axis=0)
