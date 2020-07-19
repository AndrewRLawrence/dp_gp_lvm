"""
This module defines various forms of the linear kernel.
"""

from src.distributions.log_normal import log_pdf as log_normal_log_pdf
from src.kernels.interfaces.kernel import Kernel, KernelHyperparameters
from src.utils.constants import GP_DEFAULT_JITTER
from src.utils.types import TF_DTYPE

import tensorflow as tf


def k_linear(gamma, alpha, beta):
    """
    This initialises a Kernel object for the linear kernel function.
    :param gamma: The inverse length-scale. Must be a real scalar greater than zero.
    :param alpha: The signal variance. Must be a real scalar greater than zero.
    :param beta: The noise precision. Must be a real scalar greater than zero.
    :return: An instance of the Kernel class using the linear kernel function.
    """

    # TODO: Bring over from previous codebase.
    raise NotImplementedError


def k_ard_linear(gamma, alpha, beta):
    """
    This initialises a Kernel object for the linear kernel function with ARD weights as opposed to a single
    inverse length-scale shared across all input dimensions.
    :param gamma: The ARD weights (i.e., inverse length-scales). Must be Q-length vector with postive values, where Q
    is the input/latent dimensionality. Can be [B x Q], where B is the batch size; if batching is unnecessary,
    should be [1 x Q].
    :param alpha: The signal variance. Must be a real scalar greater than zero. Can be [B x 1], where B is the batch
    size; if batching is unnecessary, should be [1 x 1].
    :param beta: The noise precision. Must be a real scalar greater than zero. Can be [B x 1], where B is the batch
    size; if batching is unnecessary, should be [1 x 1].
    :return: An instance of the Kernel class using the linear kernel function with ARD weights.
    """

    # TODO: Validate input.

    # Reshape hyperparameters. B is the batch size and is allowed to be 1.
    alpha_b11 = tf.expand_dims(alpha, axis=-1)  # Signal variance (alpha) is originally [B x 1].
    beta_b11 = tf.expand_dims(beta, axis=-1)  # Noise precision (beta) is originally [B x 1].
    gamma_bqq = tf.matrix_diag(gamma)  # ARD weights (gamma) is originally [B x Q].

    # Create hyperparameters dictionary.
    hyperparameters_dict = {KernelHyperparameters.ARD_WEIGHTS: gamma,
                            KernelHyperparameters.SIGNAL_VARIANCE: alpha,
                            KernelHyperparameters.NOISE_PRECISION: beta}

    # Create hyperpriors dictionary.
    hyperpriors_dict = {KernelHyperparameters.ARD_WEIGHTS: log_normal_log_pdf,
                        KernelHyperparameters.SIGNAL_VARIANCE: log_normal_log_pdf,
                        KernelHyperparameters.NOISE_PRECISION: log_normal_log_pdf}

    # Define covariance matrix function.
    def covariance_matrix_func(input_0, input_1=None, include_noise=False, include_jitter=False):
        """
        TODO
        :param input_0: The first input. Must be [N x Q].
        :param input_1: An optional second input; input_0 is used again if input_1 is not provided. Must be [M x Q].
        :param include_noise: An optional boolean to define whether or not to include noise along the diagonal.
        No noise is default.
        :param include_jitter: An optional boolean to define whether or not to include jitter along the diagonal.
        No jitter is default.
        :return: A valid covariance matrix of size [B x N x M], where B is the batch size and is allowed to be 1.
        """

        x_1nq = tf.expand_dims(input_0, axis=0)
        z_1mq = x_1nq if input_1 is None else tf.expand_dims(input_1, axis=0)

        k_bnm = alpha_b11 * tf.matmul(x_1nq, tf.matmul(gamma_bqq, z_1mq, transpose_b=True))

        if include_noise and input_1 is None:
            # Only add noise if Kxx, i.e. square matrix.
            b = tf.shape(k_bnm)[0]
            n = tf.shape(k_bnm)[1]
            k_bnm += tf.reciprocal(beta_b11) * tf.eye(n, batch_shape=[b], dtype=TF_DTYPE)
        if include_jitter and input_1 is None:
            # Only add noise if Kxx, i.e. square matrix.
            b = tf.shape(k_bnm)[0]
            n = tf.shape(k_bnm)[1]
            k_bnm += GP_DEFAULT_JITTER * tf.eye(n, batch_shape=[b], dtype=TF_DTYPE)

        return k_bnm

    # Define covariance diagonal function.
    def covariance_diagonal_func(input_0, include_noise=False, include_jitter=False):
        """
        TODO
        :param input_0: The first input. Must be [N x Q].
        :param include_noise: An optional boolean to define whether or not to include noise along the diagonal.
        No noise is default.
        :param include_jitter: An optional boolean to define whether or not to include jitter along the diagonal.
        No jitter is default.
        :return: A valid diagonal of size [B x N] from the covariance matrix, where B is the batch size and is allowed
        to be 1.
        """

        # Reshape hyperparameters.
        gamma_b1q = tf.expand_dims(gamma, axis=1)

        # Square input.
        x_squared_1nq = tf.expand_dims(tf.square(input_0), axis=0)

        # Calculate covariance diagonal.
        k_bn = alpha * tf.reduce_sum(x_squared_1nq * gamma_b1q, axis=-1)
        if include_noise:
            k_bn += tf.reciprocal(beta)
        if include_jitter:
            k_bn += GP_DEFAULT_JITTER

        return k_bn

    # Define psi 0 function.
    def calculate_psi_0(latent_input_mean, latent_input_covariance):
        """
        TODO
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: psi 0 with shape [B x 1], where B is the batch size and is allowed to be 1.
        """

        # Reshape hyperparameters.
        gamma_b1qq = tf.expand_dims(gamma_bqq)

        # Reshape input as necessary. Want [B x N x Q x Q].
        x_mean_1n1q = tf.expand_dims(tf.expand_dims(latent_input_mean, axis=1), axis=0)
        x_var_1nqq = tf.expand_dims(latent_input_covariance, axis=0)

        # Calculate psi 0.
        mean_squared_1nqq = tf.matmul(x_mean_1n1q, x_mean_1n1q, transpose_a=True)
        psi_0_bn = tf.trace(tf.matmul(gamma_b1qq, mean_squared_1nqq + x_var_1nqq))

        return tf.reduce_sum(psi_0_bn, axis=-1)

    # Define Psi 1 function.
    def calculate_psi_1(inducing_input, latent_input_mean, latent_input_covariance):
        """
        TODO
        :param inducing_input: The inducing inputs. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: Psi 1 with shape of [B x N x M], where B is the batch size and is allowed to be 1.
        """

        # Reshape input as necessary.
        x_mean_1nq = tf.expand_dims(latent_input_mean, axis=0)
        x_u_1mq = tf.expand_dims(inducing_input, axis=0)

        return tf.matmul(x_mean_1nq, tf.matmul(gamma_bqq, x_u_1mq, transpose_b=True))

    # Define Psi 2 function.
    def calculate_psi_2(inducing_input, latent_input_mean, latent_input_covariance):
        """
        TODO
        :param inducing_input: The inducing inputs. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: Psi 2 with shape of [B x M x M], where B is the batch size and is allowed to be 1.
        """

        # Reshape hyperparameters.
        gamma_b1qq = tf.expand_dims(gamma_bqq, axis=1)

        # Reshape input as necessary. Want [B x N x Q x Q].
        x_mean_1n1q = tf.expand_dims(tf.expand_dims(latent_input_mean, axis=1), axis=0)
        x_var_1nqq = tf.expand_dims(latent_input_covariance, axis=0)
        x_u_1mq = tf.expand_dims(inducing_input, axis=0)

        # Calculate Psi 2.
        mean_squared_1nqq = tf.matmul(x_mean_1n1q, x_mean_1n1q, transpose_a=True)
        gamma_x_u_b1qm = tf.expand_dims(tf.matmul(gamma_bqq, x_u_1mq, transpose_b=True), axis=1)

        psi_2_bnmm = tf.matmul(gamma_x_u_b1qm,
                               tf.matmul(mean_squared_1nqq + x_var_1nqq, gamma_x_u_b1qm),
                               transpose_a=True)

        return tf.reduce_sum(psi_2_bnmm, axis=1)

    return Kernel(covar_matrix_func=covariance_matrix_func, covar_diag_func=covariance_diagonal_func,
                  hyperparameter_dict=hyperparameters_dict, hyperprior_func_dict=hyperpriors_dict,
                  psi_0_func=calculate_psi_0, psi_1_func=calculate_psi_1, psi_2_func=calculate_psi_2)


def k_mahalanobis_linear(weights, gamma, alpha, beta):
    """
    This initialises a Kernel object for the linear kernel function with Mahalanobis distances.
    :param weights: The linear weights to map input into a linear subspace with dimension at most K. Must be [K x Q],
    where K is the lower rank dimensionality and Q is the input/latent dimensionality.
    :param gamma: The ARD weights (i.e., inverse length-scales). Must be Q-length vector with postive values.
    :param alpha: The signal variance. Must be a real scalar greater than zero.
    :param beta: The noise precision. Must be a real scalar greater than zero.
    :return: An instance of the Kernel class using the linear kernel function with Mahalanobis distances.
    """

    # TODO: Bring over from previous codebase.
    raise NotImplementedError
