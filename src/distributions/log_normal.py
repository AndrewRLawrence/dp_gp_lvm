"""
This module defines functions for the univariate and multivariate log-normal distributions.
"""

from src.utils.types import TF_DTYPE

import numpy as np
import tensorflow as tf


# def log_pdf(x, mean=tf.constant(0.0, dtype=TF_DTYPE), var=tf.constant(1.0, dtype=TF_DTYPE)):
#     """
#     This function calculates the log-likelihood of x for a univariate log-normal distribution parameterised by the
#     provided mean and variance.
#     :param x: The location(s) to evaluate the log-likelihood. Each value must be positive.
#     :param mean: The mean of the univariate log-normal distribution. Must be a positive scalar.
#     :param var: The variance of the univariate log-normal distribution. Must be a positive scalar.
#     :return: The log-likelihood values evaluated at x.
#     """
#
#     return -tf.log(x) - 0.5 * (tf.log(2.0 * np.pi * var) + tf.squared_difference(tf.log(x), mean) / var)


def log_pdf(x, mean=None, var=None):
    """
    This function calculates the log-likelihood of x for a univariate log-normal distribution parameterised by the
    provided mean and variance.
    :param x: The location(s) to evaluate the log-likelihood. Each value must be positive.
    :param mean: The mean of the univariate log-normal distribution. Must be a positive scalar.
    :param var: The variance of the univariate log-normal distribution. Must be a positive scalar.
    :return: The log-likelihood values evaluated at x.
    """

    if mean is None:
        mean = tf.zeros_like(x)
    if var is None:
        var = tf.ones_like(x)

    return -tf.log(x) - 0.5 * (tf.log(2.0 * np.pi * var) + tf.squared_difference(tf.log(x), mean) / var)


def pdf(x, mean=tf.constant(0.0, dtype=TF_DTYPE), var=tf.constant(1.0, dtype=TF_DTYPE)):
    """
    This function calculates the likelihood of x for a univariate log-normal distribution parameterised by the provided
    mean and variance.
    :param x: The location(s) to evaluate the likelihood. Each value must be positive.
    :param mean: The mean of the univariate log-normal distribution. Must be a positive scalar.
    :param var: The variance of the univariate log-normal distribution. Must be a positive scalar.
    :return: The likelihood values evaluated at x.
    """

    return tf.exp(log_pdf(x, mean, var))


def entropy(mean=tf.constant(0.0, dtype=TF_DTYPE), var=tf.constant(1.0, dtype=TF_DTYPE)):
    """
    This function calculates the entropy of a univariate log-normal distribution parameterised by the provided mean and
    variance.
    :param mean: The mean of the univariate log-normal distribution. Must be a positive scalar.
    :param var: The variance of the univariate log-normal distribution. Must be a positive scalar.
    :return: The entropy of the univariate log-normal distribution.
    """

    raise NotImplementedError


# TODO: Add multivariate log-normal.
