"""
This module defines various forms of the RBF kernel.
"""

from src.distributions.log_normal import log_pdf as log_normal_log_pdf
from src.kernels.interfaces.kernel import Kernel, KernelHyperparameters
from src.utils.constants import GP_DEFAULT_JITTER
from src.utils.types import TF_DTYPE

import tensorflow as tf


def k_rbf(gamma, alpha, beta):
    """
    This initialises a Kernel object for the radial basis kernel function.
    :param gamma: The inverse length-scale. Must be a real scalar greater than zero.
    :param alpha: The signal variance. Must be a real scalar greater than zero.
    :param beta: The noise precision. Must be a real scalar greater than zero.
    :return: An instance of the Kernel class using the radial basis kernel function.
    """

    # TODO: Bring over from previous codebase.
    raise NotImplementedError


def k_ard_rbf(gamma, alpha, beta):
    """
    This initialises a Kernel object for the radial basis kernel function with ARD weights as opposed to a single
    inverse length-scale shared across all input dimensions.
    :param gamma: The ARD weights (i.e., inverse length-scales). Must be Q-length vector with postive values, where Q
    is the input/latent dimensionality. Can be [B x Q], where B is the batch size; if batching is unnecessary,
    should be [1 x Q].
    :param alpha: The signal variance. Must be a real scalar greater than zero. Can be [B x 1], where B is the batch
    size; if batching is unnecessary, should be [1 x 1].
    :param beta: The noise precision. Must be a real scalar greater than zero. Can be [B x 1], where B is the batch
    size; if batching is unnecessary, should be [1 x 1].
    :return: An instance of the Kernel class using the radial basis kernel function with ARD weights.
    """

    # TODO: Validate input.

    # Reshape hyperparameters. B is the batch size and is allowed to be 1.
    alpha_b11 = tf.expand_dims(alpha, axis=-1)  # Signal variance (alpha) is originally [B x 1].
    beta_b11 = tf.expand_dims(beta, axis=-1)  # Noise precision (beta) is originally [B x 1].
    gamma_b1q = tf.expand_dims(gamma, axis=1)  # ARD weights (gamma) is originally [B x Q].

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

        sqrt_gamma_b1q = tf.sqrt(gamma_b1q)
        x_gamma_bnq = sqrt_gamma_b1q * tf.expand_dims(input_0, axis=0)
        z_gamma_bmq = x_gamma_bnq if input_1 is None else sqrt_gamma_b1q * tf.expand_dims(input_1, axis=0)

        xx_bn1 = -0.5 * tf.reduce_sum(tf.square(x_gamma_bnq), axis=-1, keepdims=True)
        zz_bm1 = xx_bn1 if input_1 is None else -0.5 * tf.reduce_sum(tf.square(z_gamma_bmq), axis=-1, keepdims=True)

        k_bnm = alpha_b11 * tf.exp(xx_bn1 + tf.transpose(zz_bm1, perm=[0, 2, 1]) +
                                   tf.matmul(x_gamma_bnq, z_gamma_bmq, transpose_b=True))

        if include_noise and input_1 is None:
            # Only add noise if Kxx, i.e. square matrix.
            #b, n, _ = tf.shape(k_bnm)
            b = tf.shape(k_bnm)[0]
            n = tf.shape(k_bnm)[1]
            k_bnm += tf.reciprocal(beta_b11) * tf.eye(n, batch_shape=[b], dtype=TF_DTYPE)
        if include_jitter and input_1 is None:
            # Only add noise if Kxx, i.e. square matrix.
            #b, n, _ = tf.shape(k_bnm)
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

        num_samples = tf.shape(input_0)[0]

        k_bn = alpha * tf.ones((1, num_samples), dtype=TF_DTYPE)
        if include_noise:
            k_bn += tf.reciprocal(beta)
        if include_jitter:
            k_bn += GP_DEFAULT_JITTER

        return k_bn

    # Define psi 0 function.
    def calculate_psi_0(inducing_input, latent_input_mean, latent_input_covariance):
        """
        TODO
        :param inducing_input: The inducing inputs. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: psi 0 with shape [B x 1], where B is the batch size and is allowed to be 1.
        """

        num_samples = tf.shape(latent_input_mean, out_type=TF_DTYPE)[0]

        return alpha * num_samples

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

        # Reshape hyperparameters.
        gamma_b11q = tf.expand_dims(gamma_b1q, axis=1)

        # Reshape input as necessary.
        x_mean_n1q = tf.expand_dims(latent_input_mean, axis=1)
        x_var_1nq = tf.expand_dims(tf.matrix_diag_part(latent_input_covariance), axis=0)
        x_u_1mq = tf.expand_dims(inducing_input, axis=0)

        # Calculate Psi 1.
        denominator_bn1q = tf.expand_dims(gamma_b1q * x_var_1nq + 1.0, axis=2)
        numerator_bnmq = gamma_b11q * tf.expand_dims(tf.squared_difference(x_mean_n1q, x_u_1mq), axis=0)

        log_psi_1_bnm = tf.log(alpha_b11) - 0.5 * tf.reduce_sum(numerator_bnmq / denominator_bn1q +
                                                                tf.log(denominator_bn1q), axis=-1)

        return tf.exp(log_psi_1_bnm)

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
        alpha_b111 = tf.expand_dims(alpha_b11, axis=-1)
        gamma_b111q = tf.expand_dims(tf.expand_dims(gamma_b1q, axis=1), axis=1)

        # Reshape input as necessary.
        x_mean_1nq = tf.expand_dims(latent_input_mean, axis=0)
        x_mean_1n11q = tf.expand_dims(tf.expand_dims(x_mean_1nq, axis=-2), axis=-2)
        x_var_1nq = tf.expand_dims(tf.matrix_diag_part(latent_input_covariance), axis=0)
        x_var_1n11q = tf.expand_dims(tf.expand_dims(x_var_1nq, axis=-2), axis=-2)
        x_u_1mq = tf.expand_dims(inducing_input, axis=0)
        x_u_11m1q = tf.expand_dims(tf.expand_dims(x_u_1mq, axis=-2), axis=0)
        x_u_111mq = tf.expand_dims(tf.expand_dims(x_u_1mq, axis=0), axis=0)

        # Calculate Psi 2.
        x_u_bar_11mmq = 0.5 * (x_u_11m1q + x_u_111mq)

        term_1_b1mmq = 0.25 * gamma_b111q * tf.squared_difference(x_u_11m1q, x_u_111mq)

        denominator_bn11q = 2.0 * gamma_b111q * x_var_1n11q + 1.0
        numerator_bnmmq = gamma_b111q * tf.squared_difference(x_mean_1n11q, x_u_bar_11mmq)

        log_psi_2_bnmm = 2.0 * tf.log(alpha_b111) - tf.reduce_sum(0.5 * tf.log(denominator_bn11q) + term_1_b1mmq +
                                                                  numerator_bnmmq / denominator_bn11q, axis=-1)

        return tf.reduce_sum(tf.exp(log_psi_2_bnmm), axis=1)

    return Kernel(covar_matrix_func=covariance_matrix_func, covar_diag_func=covariance_diagonal_func,
                  hyperparameter_dict=hyperparameters_dict, hyperprior_func_dict=hyperpriors_dict,
                  psi_0_func=calculate_psi_0, psi_1_func=calculate_psi_1, psi_2_func=calculate_psi_2)


def k_mahalanobis_rbf(weights, gamma, alpha, beta):
    """
    This initialises a Kernel object for the radial basis kernel function with Mahalanobis distances.
    :param weights: The linear weights to map input into a linear subspace with dimension at most K. Must be [K x Q],
    where K is the lower rank dimensionality and Q is the input/latent dimensionality.
    :param gamma: The ARD weights (i.e., inverse length-scales). Must be Q-length vector with postive values.
    :param alpha: The signal variance. Must be a real scalar greater than zero.
    :param beta: The noise precision. Must be a real scalar greater than zero.
    :return: An instance of the Kernel class using the radial basis kernel function with Mahalanobis distances.
    """

    # TODO: Bring over from previous codebase.
    raise NotImplementedError
