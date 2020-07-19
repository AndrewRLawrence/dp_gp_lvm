"""
This module defines functions for the univariate and multivariate normal distributions.
"""

from src.utils.types import TF_DTYPE

import numpy as np
import tensorflow as tf

# Add univariate normal.
# Add multivariate normal.


def mvn_log_pdf(x, mean, covariance):
    """
    This function calculates the log-likelihood of x for a multivariate normal distribution parameterised by the
    provided mean and covariance.
    :param x: The location(s) to evaluate the log-likelihood. Must be [B x D], where B is the batch size and D is the
    dimensionality of the multivariate normal. B can be 1.
    :param mean: The mean of the multivariate normal distribution. Must be [1 x D].
    :param covariance: The covariance of the multivariate normal distribution. Must be [D x D].
    :return: The log-likelihood values evaluated at x. This is a B-length vector.
    """

    # Determine number of dimensions of the multivariate normal distribution.
    num_dims = tf.shape(covariance, out_type=TF_DTYPE)[-1]
    # num_dims = covariance.get_shape().as_list()[-1]

    # Calculate log-likelihood.
    diff = tf.transpose(x - mean)  # [D x B].
    chol_covar = tf.cholesky(tf.squeeze(covariance))  # [D x D].

    alpha = tf.transpose(tf.matrix_triangular_solve(chol_covar, diff, lower=True))  # [B x D].
    beta = tf.reduce_sum(tf.log(tf.diag_part(chol_covar)))

    return -0.5 * (tf.reduce_sum(tf.square(alpha), axis=-1) + num_dims * np.log(2.0 * np.pi)) - beta


def mvn_conditional_mean_covar(b, mean_a, mean_b, covar_aa, covar_bb, covar_ab):
    """
    This function calculates the conditional mean and covariance for 'a' given 'b', where 'a' has D_a dimensions with N
    observations and 'b' has D_b dimensions with N observations.
    :param b: The sample values of 'b'. Must be [N x D_b].
    :param mean_a: The mean of 'a'. Must be [N x D_a].
    :param mean_b: The mean of 'b'. Must be [N x D_b].
    :param covar_aa: The covariance of 'a'. Must be [D_a x D_a].
    :param covar_bb: The covariance of 'b'. Must be [D_b x D_b].
    :param covar_ab: The covariance of 'a' and 'b'. Must be [D_a x D_b].
    :return: The conditional mean ([N x D_a]) and covariance ([D_a x D_a]) of 'a' given 'b'.
    """

    # diff_b = tf.transpose(b - mean_b)  # [D_b x N].
    diff_b = b - mean_b
    chol_bb = tf.cholesky(tf.squeeze(covar_bb))  # [D_b x D_b].
    alpha = tf.matrix_triangular_solve(
        tf.transpose(chol_bb),
        tf.matrix_triangular_solve(chol_bb, diff_b, lower=True),
        lower=True
    )  # [D_b x N].

    # v = tf.matrix_triangular_solve(chol_bb, tf.transpose(covar_ab), lower=True)  # [D_b x D_a].
    v = tf.matrix_triangular_solve(chol_bb, tf.squeeze(covar_ab), lower=True)  # [D_b x D_a].

    cond_mean = mean_a + tf.matmul(tf.squeeze(covar_ab), alpha, transpose_a=True)  # [N x D_a].
    cond_covar = covar_aa - tf.matmul(v, v, transpose_a=True)  # [D_a x D_a].

    return cond_mean, tf.squeeze(cond_covar)
