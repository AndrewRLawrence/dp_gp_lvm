"""
This module defines functions for the beta distribution.
"""

import tensorflow as tf


def entropy(alpha, beta):
    """
    This function calculates the entropy of the beta distribution parameterised by the shape parameters alpha and beta.
    :param alpha: The shape parameter alpha, which must be a positive scalar.
    :param beta: The shape parameter beta, which must be a positive scalar.
    :return: The entropy value of the beta distribution parameterised by alpha and beta.
    """

    total_concentration = alpha + beta

    return tf.lgamma(alpha) + tf.lgamma(beta) - tf.lgamma(total_concentration) - (alpha - 1.0) * tf.digamma(alpha) - \
           (beta - 1.0) * tf.digamma(beta) + (total_concentration - 2.0) * tf.digamma(total_concentration)
