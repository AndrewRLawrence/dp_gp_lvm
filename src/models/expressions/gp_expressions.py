"""
This module defines expressions/functions commonly used within Gaussian process models.
"""

from src.utils.types import TF_DTYPE

import tensorflow as tf


def calculate_kl_divergence_standard_prior(x_mean, x_covar):
    """
    TODO
    :param x_mean: [N x Q].
    :param x_covar: [N x Q x Q].
    :return:
    """
    # TODO: Validate input.
    num_samples = tf.shape(x_mean, out_type=TF_DTYPE)[0]
    num_latent_dims = tf.shape(x_mean, out_type=TF_DTYPE)[1]
    x_covar_diag = tf.matrix_diag_part(x_covar)  # [N x Q].

    kl_q_x_p_x = 0.5 * (tf.reduce_sum(tf.square(x_mean)) + tf.reduce_sum(x_covar_diag - tf.log(x_covar_diag)) -
                        num_samples * num_latent_dims)
    return kl_q_x_p_x


def calculate_elbo_free_energy_term(kernel, inducing_input, latent_input_mean, latent_input_covariance):

    # TODO: Figure out good inputs so not passing a ton of stuff.
    raise NotImplementedError

    # Define psi-statistics.
    psi_0 = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [B x 1].
    psi_1 = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [B x N x M].
    psi_2 = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [B x M x M].

    # Calculate f_hat term from the evidence lower bound (ELBO). Using stable calculation for f_hat.
    beta = kernel.noise_precision
    beta_b11 = tf.expand_dims(beta, axis=-1)  # [B x 1 x 1].

    k_uu = kernel.covariance_matrix(input_0=x_u, input_1=None, include_noise=False, include_jitter=True)  # [B x M x M].
    l_uu = tf.cholesky(k_uu)  # [B x M x M].

    l_uu_inv_psi_2 = tf.matrix_triangular_solve(l_uu, psi_2, lower=True)  # [B x M x M].
    l_uu_inv_psi_2_inv_transpose = tf.transpose(
        tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2, perm=[0, 2, 1]), lower=True),
        perm=[0, 2, 1])  # [B x M x M].

    a = beta_b11 * l_uu_inv_psi_2_inv_transpose + tf.eye(num_inducing_points, batch_shape=[batch_size], dtype=TF_DTYPE)
    l_a = tf.cholesky(a)  # [B x M x M].

    log_det_l_a = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a)))  # Scalar.

    # [B x M x N].
    l_uu_inv_psi_1_transpose = tf.matrix_triangular_solve(l_uu, tf.transpose(psi_1, perm=[0, 2, 1]), lower=True)
    c = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_transpose, lower=True)  # [B x M x N].
    c_transpose_c = tf.squeeze(tf.matmul(c, c, transpose_a=True), axis=0)  # Squeeze since B=1 so cTc is [N x N].
    yy_transpose = tf.matmul(y_train, y_train, transpose_b=True)

    f_hat = 0.5 * num_samples * num_dimensions * (tf.reduce_sum(tf.log(beta)) - np.log(2.0 * np.pi)) - \
            num_dimensions * log_det_l_a + \
            0.5 * num_dimensions * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_inv_transpose) - psi_0)) + \
            0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c, yy_transpose))) - \
            0.5 * tf.reduce_sum(beta * tf.trace(yy_transpose))

    f_hat = 0
    return f_hat