"""
This module defines functions for the multinomial distribution.
"""

import tensorflow as tf


def entropy(probs):
    """
    This function calculates the entropy of the multinomial distribution parameterised by the probabilities of each
    class.
    :param probs: The probabilities of each class being drawn. Must sum to 1 and each element must be positive.
    :return: The entropy value of the gamma distribution parameterised by the probabilities of each class.
    """

    return tf.negative(tf.reduce_sum(tf.multiply(probs, tf.log(probs)), axis=-1))
