"""
This file defines unit tests for the Bayesian GP-LVM implementation.
"""

from src.distributions.log_normal import log_pdf
from src.kernels.rbf_kernel import k_ard_rbf
from src.models.gaussian_process import bayesian_gp_lvm
from src.utils.types import NP_DTYPE, TF_DTYPE
from test.unittests.kernel_unittests import k_ard_rbf_covariance_matrix_naive, \
    k_ard_rbf_psi_0_naive, k_ard_rbf_psi_1_naive, k_ard_rbf_psi_2_naive

import numpy as np
import tensorflow as tf
import unittest


def free_energy_naive(y, x_mean, x_var, x_u, gamma, alpha, beta):
    """
    TODO
    :param y:
    :param x_mean:
    :param x_var:
    :param x_u:
    :param gamma:
    :param alpha:
    :param beta:
    :return:
    """

    # Determine number of samples, number of inducing points, and number of latent dimensions.
    assert np.shape(x_mean) == np.shape(x_var), 'Shape of mean and variance of q(X) must be the same.'
    [n, q] = np.shape(x_mean)
    [m, qu] = np.shape(x_u)
    [ny, d] = np.shape(y)
    assert q == qu, 'Latent dimensionality of X and inducing input must be the same.'
    assert n == ny, 'Number of observations in X and Y must be equal.'
    assert n > m, 'Number of observations must be greater than number of inducing points.'
    assert q == np.size(gamma), 'ARD weights must be same size as latent dimensionality.'

    # Calculate covariance matrices and psi-statistics.
    k_uu = k_ard_rbf_covariance_matrix_naive(input_0=x_u, gamma=gamma, alpha=alpha, beta=beta,
                                             include_noise=False, include_jitter=True)  # [M x M].
    psi_0 = k_ard_rbf_psi_0_naive(num_samples=n, alpha=alpha)
    psi_1 = k_ard_rbf_psi_1_naive(x_mean=x_mean, x_var=x_var, x_u=x_u, gamma=gamma, alpha=alpha)  # [N x M].
    psi_2 = k_ard_rbf_psi_2_naive(x_mean=x_mean, x_var=x_var, x_u=x_u, gamma=gamma, alpha=alpha)  # [M x M].

    w = beta * np.eye(n) - np.square(beta) * np.matmul(np.matmul(psi_1, np.linalg.inv(beta * psi_2 + k_uu)), psi_1.T)
    yyt = np.matmul(y, y.T)  # [N x N].

    return 0.5 * n * d * np.log(beta) + 0.5 * d * np.log(np.linalg.det(k_uu)) - 0.5 * n * d * np.log(2.0 * np.pi) - \
        0.5 * d * np.log(np.linalg.det(beta * psi_2 + k_uu)) - 0.5 * d * beta * psi_0 + \
        0.5 * d * beta * np.trace(np.matmul(np.linalg.inv(k_uu), psi_2)) - 0.5 * np.trace(np.matmul(w, yyt))


def free_energy_stable(y, x_mean, x_var, x_u, gamma, alpha, beta):
    """
    TODO
    :param y:
    :param x_mean:
    :param x_var:
    :param x_u:
    :param gamma:
    :param alpha:
    :param beta:
    :return:
    """

    # Determine number of samples, number of inducing points, and number of latent dimensions.
    assert np.shape(x_mean) == np.shape(x_var), 'Shape of mean and variance of q(X) must be the same.'
    [n, q] = np.shape(x_mean)
    [m, qu] = np.shape(x_u)
    [ny, d] = np.shape(y)
    assert q == qu, 'Latent dimensionality of X and inducing input must be the same.'
    assert n == ny, 'Number of observations in X and Y must be equal.'
    assert n > m, 'Number of observations must be greater than number of inducing points.'
    assert q == np.size(gamma), 'ARD weights must be same size as latent dimensionality.'

    # Calculate covariance matrices and psi-statistics.
    k_uu = k_ard_rbf_covariance_matrix_naive(input_0=x_u, gamma=gamma, alpha=alpha, beta=beta,
                                             include_noise=False, include_jitter=True)  # [M x M].
    psi_0 = k_ard_rbf_psi_0_naive(num_samples=n, alpha=alpha)  # Scalar.
    psi_1 = k_ard_rbf_psi_1_naive(x_mean=x_mean, x_var=x_var, x_u=x_u, gamma=gamma, alpha=alpha)  # [N x M].
    psi_2 = k_ard_rbf_psi_2_naive(x_mean=x_mean, x_var=x_var, x_u=x_u, gamma=gamma, alpha=alpha)  # [M x M].

    # Calculate f_hat term from the evidence lower bound (ELBO). Using stable calculation for f_hat.
    beta_11 = tf.expand_dims(beta, axis=-1)  # [1 x 1].

    l_uu = tf.cholesky(k_uu)  # [M x M].

    l_uu_inv_psi_2 = tf.matrix_triangular_solve(l_uu, psi_2, lower=True)  # [M x M].
    l_uu_inv_psi_2_inv_transpose = tf.transpose(tf.matrix_triangular_solve(l_uu,
                                                                           tf.transpose(l_uu_inv_psi_2),
                                                                           lower=True
                                                                           )
                                                )  # [M x M].

    a = beta_11 * l_uu_inv_psi_2_inv_transpose + tf.eye(m, dtype=TF_DTYPE)  # [M x M].
    l_a = tf.cholesky(a)  # [M x M].

    log_det_l_a = tf.reduce_sum(tf.log(tf.diag_part(l_a)))  # Scalar.

    # [M x N].
    l_uu_inv_psi_1_transpose = tf.matrix_triangular_solve(l_uu, tf.transpose(psi_1), lower=True)
    c = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_transpose, lower=True)  # [M x N].
    c_transpose_c = tf.matmul(c, c, transpose_a=True)  # [N x N].
    yy_transpose = tf.matmul(y, y, transpose_b=True)  # [N x N].

    f_hat = 0.5 * n * d * (tf.reduce_sum(tf.log(beta)) - np.log(2.0 * np.pi)) - \
        d * log_det_l_a + \
        0.5 * d * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_inv_transpose) - psi_0)) + \
        0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c, yy_transpose))) - \
        0.5 * tf.reduce_sum(beta * tf.trace(yy_transpose))

    # # Cool way to evaluate node within graph.
    # sess = tf.Session()
    # with sess.as_default():
    #     assert tf.get_default_session() is sess
    #     print('Stable: {}'.format(f_hat.eval()))

    return f_hat


def kl_qx_px_naive(x_mean, x_var):
    """
    TODO
    :param x_mean:
    :param x_var:
    :return:
    """

    # Determine number of samples, number of inducing points, and number of latent dimensions.
    assert np.shape(x_mean) == np.shape(x_var), 'Shape of mean and variance of q(X) must be the same.'
    [n, q] = np.shape(x_mean)

    return 0.5 * (np.sum(np.square(x_mean)) + np.sum(x_var) - np.sum(np.log(x_var)) - n * q)


def elbo_naive(y, x_mean, x_var, x_u, gamma, alpha, beta):
    """
    TODO
    :param y:
    :param x_mean:
    :param x_var:
    :param x_u:
    :param gamma:
    :param alpha:
    :param beta:
    :return:
    """

    # Determine number of samples, number of inducing points, and number of latent dimensions.
    assert np.shape(x_mean) == np.shape(x_var), 'Shape of mean and variance of q(X) must be the same.'
    [n, q] = np.shape(x_mean)
    [m, qu] = np.shape(x_u)
    [ny, d] = np.shape(y)
    assert q == qu, 'Latent dimensionality of X and inducing input must be the same.'
    assert n == ny, 'Number of observations in X and Y must be equal.'
    assert n > m, 'Number of observations must be greater than number of inducing points.'
    assert q == np.size(gamma), 'ARD weights must be same size as latent dimensionality.'

    elbo = free_energy_naive(y=y, x_mean=x_mean, x_var=x_var, x_u=x_u, gamma=gamma, alpha=alpha, beta=beta) - \
        kl_qx_px_naive(x_mean=x_mean, x_var=x_var)

    return elbo


def elbo_stable(y, x_mean, x_var, x_u, gamma, alpha, beta):
    """
    TODO
    :param y:
    :param x_mean:
    :param x_var:
    :param x_u:
    :param gamma:
    :param alpha:
    :param beta:
    :return:
    """

    # Determine number of samples, number of inducing points, and number of latent dimensions.
    assert np.shape(x_mean) == np.shape(x_var), 'Shape of mean and variance of q(X) must be the same.'
    [n, q] = np.shape(x_mean)
    [m, qu] = np.shape(x_u)
    [ny, d] = np.shape(y)
    assert q == qu, 'Latent dimensionality of X and inducing input must be the same.'
    assert n == ny, 'Number of observations in X and Y must be equal.'
    assert n > m, 'Number of observations must be greater than number of inducing points.'
    assert q == np.size(gamma), 'ARD weights must be same size as latent dimensionality.'

    elbo = free_energy_stable(y=y, x_mean=x_mean, x_var=x_var, x_u=x_u, gamma=gamma, alpha=alpha, beta=beta) - \
        kl_qx_px_naive(x_mean=x_mean, x_var=x_var)

    return elbo


class TestBGPLVM(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """
        np.random.seed(seed=seed)

        self.n = 50
        self.d = 5
        self.m = 25
        self.q = 3

        self.y = np.random.standard_normal((self.n, self.d)).astype(NP_DTYPE)  # [N x D].

        self.gamma = np.exp(np.random.standard_normal(self.q).astype(NP_DTYPE))  # Q-length vector
        self.alpha = np.square(np.random.standard_normal() + 1.0)  # Scalar
        self.beta = np.square(np.random.standard_normal() + np.sqrt(50.0))  # Scalar

        self.kernel = k_ard_rbf(gamma=self.gamma[np.newaxis, :],
                                alpha=np.reshape(self.alpha, (1, 1)),
                                beta=np.reshape(self.beta, (1, 1)))

        self.bgplvm = bayesian_gp_lvm(y_train=self.y, kernel=self.kernel, num_latent_dims=self.q,
                                      num_inducing_points=self.m)

        # TensorFlow session.
        self.tf_session = tf.Session()

        # Init variables so we can get x_mean, x_covar, and x_u.
        self.tf_session.run(tf.global_variables_initializer())
        self.x_mean, self.x_covar = self.tf_session.run(self.bgplvm.q_x)
        self.x_var = np.stack(tuple([np.diag(self.x_covar[i]) for i in range(self.n)]), axis=0)  # [N x Q].
        self.x_u = self.tf_session.run(self.bgplvm.inducing_input)

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_objective(self):
        # Calculate hyperprior for kernel hyperparameters.
        hyper_prior = self.tf_session.run(tf.reduce_sum(log_pdf(self.gamma)) + log_pdf(self.alpha) + log_pdf(self.beta))

        # Calculate objective naively.
        objective_naive = np.negative(elbo_naive(y=self.y, x_mean=self.x_mean, x_var=self.x_var, x_u=self.x_u,
                                                 gamma=self.gamma, alpha=self.alpha, beta=self.beta) +
                                      hyper_prior)

        # Calculate objective stably.
        objective_stable = np.negative(self.tf_session.run(elbo_stable(y=self.y, x_mean=self.x_mean, x_var=self.x_var,
                                                                       x_u=self.x_u, gamma=self.gamma, alpha=self.alpha,
                                                                       beta=self.beta)) + hyper_prior)

        objective_val = self.tf_session.run(self.bgplvm.objective)

        np.testing.assert_allclose(objective_naive, objective_stable)
        np.testing.assert_allclose(objective_naive, objective_val)
        np.testing.assert_allclose(objective_stable, objective_val)

    def test_prediction(self):
        # TODO: Add test for predicting missing data.
        pass
