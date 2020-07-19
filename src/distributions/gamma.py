"""
This module defines functions for the gamma distribution.
"""

import tensorflow as tf


def entropy(alpha, beta):
    """
    This function calculates the entropy of the gamma distribution parameterised by the shape parameter alpha and the
    rate parameter beta.
    :param alpha: The shape parameter alpha, which must be a positive scalar.
    :param beta: The rate parameter beta, which must be a positive scalar.
    :return: The entropy value of the gamma distribution parameterised by alpha and beta.
    """

    return alpha - tf.log(beta) + tf.lgamma(alpha) + (1.0 - alpha) * tf.digamma(alpha)
