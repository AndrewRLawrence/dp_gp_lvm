"""
This file defines unit tests for the various kernel implementations.
"""

from src.kernels.rbf_kernel import k_ard_rbf
from src.utils.constants import GP_DEFAULT_JITTER
from src.utils.types import NP_DTYPE

import numpy as np
import tensorflow as tf
import unittest


def k_ard_rbf_covariance_matrix_naive(input_0, gamma, alpha, beta,
                                      input_1=None, include_noise=False, include_jitter=False):
    """
    TODO
    :param input_0:
    :param input_1:
    :param gamma:
    :param alpha:
    :param beta:
    :param include_noise:
    :param include_jitter:
    :return:
    """

    # Check shapes/sizes.
    [n0, q0] = np.shape(input_0)
    if input_1 is not None:
        include_noise = False  # Override as only provide noise if k(x,x).
        include_jitter = False  # Override as only provide noise if k(x,x).
    else:
        input_1 = input_0
    [n1, q1] = np.shape(input_1)
    assert q0 == q1, 'Input dimensionality must be the same for inputs 0 and 1.'
    assert q0 == np.size(gamma), 'ARD weights must be same size as input dimensionality.'

    cov_matrix = np.zeros((n0, n1))

    for i in range(n0):
        for k in range(n1):
            exp_value = 0.0
            for j in range(q0):
                exp_value += gamma[j] * np.square(input_0[i, j] - input_1[k, j])
            cov_matrix[i, k] = alpha * np.exp(-0.5 * exp_value)

    if include_noise:
        cov_matrix += np.reciprocal(beta) * np.eye(n0)
    if include_jitter:
        cov_matrix += GP_DEFAULT_JITTER * np.eye(n0)

    return cov_matrix


def k_ard_rbf_covariance_diagonal_naive(input_0, gamma, alpha, beta, include_noise=False, include_jitter=False):
    """
    TODO
    :param input_0:
    :param gamma:
    :param alpha:
    :param beta:
    :param include_noise:
    :param include_jitter:
    :return:
    """
    return np.diag(k_ard_rbf_covariance_matrix_naive(input_0, gamma, alpha, beta, input_1=None,
                                                     include_noise=include_noise, include_jitter=include_jitter))


def k_ard_rbf_psi_0_naive(num_samples, alpha):
    """
    TODO
    :param num_samples:
    :param alpha:
    :return:
    """
    return num_samples * alpha


def k_ard_rbf_psi_1_naive(x_mean, x_var, x_u, gamma, alpha):
    """
    TODO
    :param x_mean:
    :param x_var:
    :param x_u:
    :param gamma:
    :param alpha:
    :return:
    """

    # Determine number of samples, number of inducing points, and number of latent dimensions.
    assert np.shape(x_mean) == np.shape(x_var), 'Shape of mean and variance of q(X) must be the same.'
    [n, q] = np.shape(x_mean)
    [m, qu] = np.shape(x_u)
    assert n > m, 'Number of observations must be greater than number of inducing points.'
    assert q == qu, 'Latent dimensionality of X and inducing input must be the same.'
    assert q == np.size(gamma), 'ARD weights must be same size as latent dimensionality.'

    # Initialise log_psi_1.
    log_psi_1 = np.log(alpha) * np.ones((n, m))

    # Loop through each dimension.
    for i in range(n):
        for k in range(m):
            for j in range(q):
                denominator = gamma[j] * x_var[i, j] + 1.0
                log_psi_1[i, k] -= 0.5 * (np.log(denominator) +
                                          gamma[j] * np.square(x_mean[i, j] - x_u[k, j]) / denominator)

    return np.exp(log_psi_1)


def k_ard_rbf_psi_2_naive(x_mean, x_var, x_u, gamma, alpha):
    """
    TODO
    :param x_mean:
    :param x_var:
    :param x_u:
    :param gamma:
    :param alpha:
    :return:
    """

    # Determine number of samples, number of inducing points, and number of latent dimensions.
    assert np.shape(x_mean) == np.shape(x_var), 'Shape of mean and variance of q(X) must be the same.'
    [n, q] = np.shape(x_mean)
    [m, qu] = np.shape(x_u)
    assert n > m, 'Number of observations must be greater than number of inducing points.'
    assert q == qu, 'Latent dimensionality of X and inducing input must be the same.'
    assert q == np.size(gamma), 'ARD weights must be same size as latent dimensionality.'

    # Initialise log_psi_2.
    log_psi_2 = 2.0 * np.log(alpha) * np.ones((n, m, m))

    # Loop through each dimension.
    for i in range(n):
        for k1 in range(m):
            for k2 in range(m):
                for j in range(q):
                    denominator = 2.0 * gamma[j] * x_var[i, j] + 1.0
                    x_u_bar = 0.5 * (x_u[k1, j] + x_u[k2, j])
                    log_psi_2[i, k1, k2] -= 0.5 * np.log(denominator) + \
                        0.25 * gamma[j] * np.square(x_u[k1, j] - x_u[k2, j]) + \
                        gamma[j] * np.square(x_mean[i, j] - x_u_bar) / denominator

    return np.sum(np.exp(log_psi_2), axis=0)  # [M x M].


class TestRbfKernel(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """
        np.random.seed(seed=seed)

        self.n = 100
        self.n0 = 100
        self.n1 = 75
        self.d = 20
        self.m = 25
        self.q = 10

        self.y = np.random.standard_normal((self.n, self.d)).astype(NP_DTYPE)
        self.x = np.random.standard_normal((self.n, self.q)).astype(NP_DTYPE)
        self.x0 = np.random.standard_normal((self.n0, self.q)).astype(NP_DTYPE)
        self.x1 = np.random.standard_normal((self.n1, self.q)).astype(NP_DTYPE)

        self.x_mean = np.random.standard_normal((self.n, self.q)).astype(NP_DTYPE)
        self.x_var = np.square(np.random.standard_normal((self.n, self.q)).astype(NP_DTYPE))  # [N x Q].
        self.x_covar = np.stack(tuple([np.diag(self.x_var[i, :]) for i in range(self.n)]), axis=0)  # [N x Q x Q].
        self.x_u = np.random.standard_normal((self.m, self.q)).astype(NP_DTYPE)

        self.gamma = np.exp(np.random.standard_normal(self.q).astype(NP_DTYPE))
        self.alpha = np.square(np.random.standard_normal(1).astype(NP_DTYPE) + 1.0)
        self.beta = np.square(np.random.standard_normal(1).astype(NP_DTYPE) + np.sqrt(50.0))

        self.kernel = k_ard_rbf(gamma=self.gamma[np.newaxis, :],
                                alpha=np.reshape(self.alpha, (1, 1)),
                                beta=np.reshape(self.beta, (1, 1)))

        # TensorFlow session.
        self.tf_session = tf.Session()

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_covariance_matrix(self):
        # Calculate a bunch of covariance matrices in a naive manner.
        k_xx_naive = k_ard_rbf_covariance_matrix_naive(input_0=self.x0,
                                                       gamma=self.gamma,
                                                       alpha=self.alpha,
                                                       beta=self.beta,
                                                       input_1=None,
                                                       include_noise=False,
                                                       include_jitter=True)
        k_xx_naive_noisy = k_ard_rbf_covariance_matrix_naive(input_0=self.x0,
                                                             gamma=self.gamma,
                                                             alpha=self.alpha,
                                                             beta=self.beta,
                                                             input_1=None,
                                                             include_noise=True,
                                                             include_jitter=True)
        k_01_naive = k_ard_rbf_covariance_matrix_naive(input_0=self.x0,
                                                       gamma=self.gamma,
                                                       alpha=self.alpha,
                                                       beta=self.beta,
                                                       input_1=self.x1,
                                                       include_noise=False,
                                                       include_jitter=True)
        k_10_naive = k_ard_rbf_covariance_matrix_naive(input_0=self.x1,
                                                       gamma=self.gamma,
                                                       alpha=self.alpha,
                                                       beta=self.beta,
                                                       input_1=self.x0,
                                                       include_noise=False,
                                                       include_jitter=False)
        k_uu_naive = k_ard_rbf_covariance_matrix_naive(input_0=self.x_u,
                                                       gamma=self.gamma,
                                                       alpha=self.alpha,
                                                       beta=self.beta,
                                                       input_1=self.x_u,
                                                       include_noise=False,
                                                       include_jitter=False)
        k_uu_naive_noisy = k_ard_rbf_covariance_matrix_naive(input_0=self.x_u,
                                                             gamma=self.gamma,
                                                             alpha=self.alpha,
                                                             beta=self.beta,
                                                             input_1=None,
                                                             include_noise=True,
                                                             include_jitter=True)
        k_xmxm_naive = k_ard_rbf_covariance_matrix_naive(input_0=self.x_mean,
                                                         gamma=self.gamma,
                                                         alpha=self.alpha,
                                                         beta=self.beta,
                                                         input_1=None,
                                                         include_noise=False,
                                                         include_jitter=True)
        k_xmxm_naive_noisy = k_ard_rbf_covariance_matrix_naive(input_0=self.x_mean,
                                                               gamma=self.gamma,
                                                               alpha=self.alpha,
                                                               beta=self.beta,
                                                               input_1=None,
                                                               include_noise=True,
                                                               include_jitter=False)

        k_xx, k_xx_noisy, \
            k_01, k_10, \
            k_uu, k_uu_noisy, \
            k_xmxm, k_xmxm_noisy = self.tf_session.run((tf.squeeze(self.kernel.covariance_matrix(input_0=self.x0,
                                                                                                 input_1=None,
                                                                                                 include_noise=False,
                                                                                                 include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_matrix(input_0=self.x0,
                                                                                                 input_1=None,
                                                                                                 include_noise=True,
                                                                                                 include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_matrix(input_0=self.x0,
                                                                                                 input_1=self.x1,
                                                                                                 include_noise=False,
                                                                                                 include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_matrix(input_0=self.x1,
                                                                                                 input_1=self.x0,
                                                                                                 include_noise=False,
                                                                                                 include_jitter=False)),
                                                        tf.squeeze(self.kernel.covariance_matrix(input_0=self.x_u,
                                                                                                 input_1=self.x_u,
                                                                                                 include_noise=False,
                                                                                                 include_jitter=False)),
                                                        tf.squeeze(self.kernel.covariance_matrix(input_0=self.x_u,
                                                                                                 input_1=None,
                                                                                                 include_noise=True,
                                                                                                 include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_matrix(input_0=self.x_mean,
                                                                                                 input_1=None,
                                                                                                 include_noise=False,
                                                                                                 include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_matrix(input_0=self.x_mean,
                                                                                                 input_1=None,
                                                                                                 include_noise=True,
                                                                                                 include_jitter=False))
                                                        ))

        # Compare all matrices to the naive ones.
        np.testing.assert_equal(k_xx_naive.shape, k_xx.shape)
        np.testing.assert_allclose(k_xx_naive, k_xx)
        np.testing.assert_equal(k_xx_naive_noisy.shape, k_xx_noisy.shape)
        np.testing.assert_allclose(k_xx_naive_noisy, k_xx_noisy)

        np.testing.assert_equal(k_01_naive.shape, k_01.shape)
        np.testing.assert_allclose(k_01_naive, k_01)
        np.testing.assert_equal(k_10_naive.shape, k_10.shape)
        np.testing.assert_allclose(k_10_naive, k_10)

        np.testing.assert_equal(k_uu_naive.shape, k_uu.shape)
        np.testing.assert_allclose(k_uu_naive, k_uu)
        np.testing.assert_equal(k_uu_naive_noisy.shape, k_uu_noisy.shape)
        np.testing.assert_allclose(k_uu_naive_noisy, k_uu_noisy)

        np.testing.assert_equal(k_xmxm_naive.shape, k_xmxm.shape)
        np.testing.assert_allclose(k_xmxm_naive, k_xmxm)
        np.testing.assert_equal(k_xmxm_naive_noisy.shape, k_xmxm_noisy.shape)
        np.testing.assert_allclose(k_xmxm_naive_noisy, k_xmxm_noisy)

    def test_covariance_diag(self):
        # Calculate a bunch of covariance matrix diagonals in a naive manner.
        k_x0_naive = k_ard_rbf_covariance_diagonal_naive(input_0=self.x0,
                                                         gamma=self.gamma,
                                                         alpha=self.alpha,
                                                         beta=self.beta,
                                                         include_noise=False,
                                                         include_jitter=True)
        k_x0_naive_noisy = k_ard_rbf_covariance_diagonal_naive(input_0=self.x0,
                                                               gamma=self.gamma,
                                                               alpha=self.alpha,
                                                               beta=self.beta,
                                                               include_noise=True,
                                                               include_jitter=True)
        k_x1_naive = k_ard_rbf_covariance_diagonal_naive(input_0=self.x1,
                                                         gamma=self.gamma,
                                                         alpha=self.alpha,
                                                         beta=self.beta,
                                                         include_noise=False,
                                                         include_jitter=False)
        k_x1_naive_noisy = k_ard_rbf_covariance_diagonal_naive(input_0=self.x1,
                                                               gamma=self.gamma,
                                                               alpha=self.alpha,
                                                               beta=self.beta,
                                                               include_noise=True,
                                                               include_jitter=False)
        k_uu_naive = k_ard_rbf_covariance_diagonal_naive(input_0=self.x_u,
                                                         gamma=self.gamma,
                                                         alpha=self.alpha,
                                                         beta=self.beta,
                                                         include_noise=False,
                                                         include_jitter=False)
        k_uu_naive_noisy = k_ard_rbf_covariance_diagonal_naive(input_0=self.x_u,
                                                               gamma=self.gamma,
                                                               alpha=self.alpha,
                                                               beta=self.beta,
                                                               include_noise=True,
                                                               include_jitter=True)
        k_xmxm_naive = k_ard_rbf_covariance_diagonal_naive(input_0=self.x_mean,
                                                           gamma=self.gamma,
                                                           alpha=self.alpha,
                                                           beta=self.beta,
                                                           include_noise=False,
                                                           include_jitter=True)
        k_xmxm_naive_noisy = k_ard_rbf_covariance_diagonal_naive(input_0=self.x_mean,
                                                                 gamma=self.gamma,
                                                                 alpha=self.alpha,
                                                                 beta=self.beta,
                                                                 include_noise=True,
                                                                 include_jitter=False)

        k_x0, k_x0_noisy, \
            k_x1, k_x1_noisy, \
            k_uu, k_uu_noisy, \
            k_xmxm, k_xmxm_noisy = self.tf_session.run((tf.squeeze(self.kernel.covariance_diag(input_0=self.x0,
                                                                                               include_noise=False,
                                                                                               include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_diag(input_0=self.x0,
                                                                                               include_noise=True,
                                                                                               include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_diag(input_0=self.x1,
                                                                                               include_noise=False,
                                                                                               include_jitter=False)),
                                                        tf.squeeze(self.kernel.covariance_diag(input_0=self.x1,
                                                                                               include_noise=True,
                                                                                               include_jitter=False)),
                                                        tf.squeeze(self.kernel.covariance_diag(input_0=self.x_u,
                                                                                               include_noise=False,
                                                                                               include_jitter=False)),
                                                        tf.squeeze(self.kernel.covariance_diag(input_0=self.x_u,
                                                                                               include_noise=True,
                                                                                               include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_diag(input_0=self.x_mean,
                                                                                               include_noise=False,
                                                                                               include_jitter=True)),
                                                        tf.squeeze(self.kernel.covariance_diag(input_0=self.x_mean,
                                                                                               include_noise=True,
                                                                                               include_jitter=False))
                                                        ))

        # Compare all matrices to the naive ones.
        np.testing.assert_equal(k_x0_naive.shape, k_x0.shape)
        np.testing.assert_allclose(k_x0_naive, k_x0)
        np.testing.assert_equal(k_x0_naive_noisy.shape, k_x0_noisy.shape)
        np.testing.assert_allclose(k_x0_naive_noisy, k_x0_noisy)

        np.testing.assert_equal(k_x1_naive.shape, k_x1.shape)
        np.testing.assert_allclose(k_x1_naive, k_x1)
        np.testing.assert_equal(k_x1_naive_noisy.shape, k_x1_noisy.shape)
        np.testing.assert_allclose(k_x1_naive_noisy, k_x1_noisy)

        np.testing.assert_equal(k_uu_naive.shape, k_uu.shape)
        np.testing.assert_allclose(k_uu_naive, k_uu)
        np.testing.assert_equal(k_uu_naive_noisy.shape, k_uu_noisy.shape)
        np.testing.assert_allclose(k_uu_naive_noisy, k_uu_noisy)

        np.testing.assert_equal(k_xmxm_naive.shape, k_xmxm.shape)
        np.testing.assert_allclose(k_xmxm_naive, k_xmxm)
        np.testing.assert_equal(k_xmxm_naive_noisy.shape, k_xmxm_noisy.shape)
        np.testing.assert_allclose(k_xmxm_naive_noisy, k_xmxm_noisy)

    def test_psi_0(self):
        # Calculate psi 0s in a naive manner.
        psi_0_xx_naive = k_ard_rbf_psi_0_naive(self.n, np.squeeze(self.alpha))
        psi_0_x0_naive = k_ard_rbf_psi_0_naive(self.n0, np.squeeze(self.alpha))
        psi_0_x1_naive = k_ard_rbf_psi_0_naive(self.n1, np.squeeze(self.alpha))
        psi_0_xu_naive = k_ard_rbf_psi_0_naive(self.m, np.squeeze(self.alpha))

        psi_0_xx, psi_0_x0, psi_0_x1, psi_0_xu = self.tf_session.run((
            tf.squeeze(self.kernel.psi_0(inducing_input=self.x_u,
                                         latent_input_mean=self.x,
                                         latent_input_covariance=self.x_covar)),
            tf.squeeze(self.kernel.psi_0(inducing_input=self.x_u,
                                         latent_input_mean=self.x0,
                                         latent_input_covariance=self.x_covar)),
            tf.squeeze(self.kernel.psi_0(inducing_input=self.x_u,
                                         latent_input_mean=self.x1,
                                         latent_input_covariance=self.x_covar)),
            tf.squeeze(self.kernel.psi_0(inducing_input=self.x_u,
                                         latent_input_mean=self.x_u,
                                         latent_input_covariance=self.x_covar))
        ))

        # Compare all psi 0s to the naive ones.
        np.testing.assert_equal(psi_0_xx_naive.shape, psi_0_xx.shape)
        np.testing.assert_allclose(psi_0_xx_naive, psi_0_xx)
        np.testing.assert_equal(psi_0_x0_naive.shape, psi_0_x0.shape)
        np.testing.assert_allclose(psi_0_x0_naive, psi_0_x0)
        np.testing.assert_equal(psi_0_x1_naive.shape, psi_0_x1.shape)
        np.testing.assert_allclose(psi_0_x1_naive, psi_0_x1)
        np.testing.assert_equal(psi_0_xu_naive.shape, psi_0_xu.shape)
        np.testing.assert_allclose(psi_0_xu_naive, psi_0_xu)

    def test_psi_1(self):
        # Calculate psi 1 in a naive manner.
        psi_1_naive = k_ard_rbf_psi_1_naive(self.x_mean, self.x_var, self.x_u, self.gamma, self.alpha)

        psi_1 = self.tf_session.run(tf.squeeze(self.kernel.psi_1(inducing_input=self.x_u,
                                                                 latent_input_mean=self.x_mean,
                                                                 latent_input_covariance=self.x_covar)))

        # Compare psi 1 to naive one.
        np.testing.assert_equal(psi_1_naive.shape, psi_1.shape)
        np.testing.assert_allclose(psi_1_naive, psi_1)

    def test_psi_2(self):
        # Calculate psi 2 in a naive manner.
        psi_2_naive = k_ard_rbf_psi_2_naive(self.x_mean, self.x_var, self.x_u, self.gamma, self.alpha)

        psi_2 = self.tf_session.run(tf.squeeze(self.kernel.psi_2(inducing_input=self.x_u,
                                                                 latent_input_mean=self.x_mean,
                                                                 latent_input_covariance=self.x_covar)))

        # Compare psi 2 to naive one.
        np.testing.assert_equal(psi_2_naive.shape, psi_2.shape)
        np.testing.assert_allclose(psi_2_naive, psi_2)


class TestRbfBatchKernel(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """
        np.random.seed(seed=seed)

        self.n = 100
        self.n0 = 100
        self.n1 = 75
        self.d = 7
        self.m = 25
        self.q = 10

        self.y = np.random.standard_normal((self.n, self.d)).astype(NP_DTYPE)
        self.x = np.random.standard_normal((self.n, self.q)).astype(NP_DTYPE)
        self.x0 = np.random.standard_normal((self.n0, self.q)).astype(NP_DTYPE)
        self.x1 = np.random.standard_normal((self.n1, self.q)).astype(NP_DTYPE)

        self.x_mean = np.random.standard_normal((self.n, self.q)).astype(NP_DTYPE)
        self.x_var = np.square(np.random.standard_normal((self.n, self.q)).astype(NP_DTYPE))  # [N x Q].
        self.x_covar = np.stack(tuple([np.diag(self.x_var[i, :]) for i in range(self.n)]), axis=0)  # [N x Q x Q].
        self.x_u = np.random.standard_normal((self.m, self.q)).astype(NP_DTYPE)

        self.gamma = np.exp(np.random.standard_normal((self.d, self.q)).astype(NP_DTYPE))
        self.alpha = np.square(np.random.standard_normal((self.d, 1)).astype(NP_DTYPE) + 1.0)
        self.beta = np.square(np.random.standard_normal((self.d, 1)).astype(NP_DTYPE) + np.sqrt(50.0))

        self.kernel = k_ard_rbf(gamma=self.gamma, alpha=self.alpha, beta=self.beta)

        # TensorFlow session.
        self.tf_session = tf.Session()

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_covariance_matrix(self):
        # Calculate a bunch of covariance matrices in a naive manner.
        k_xx_naive = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x0,
                                                                       gamma=self.gamma[i],
                                                                       alpha=self.alpha[i],
                                                                       beta=self.beta[i],
                                                                       input_1=None,
                                                                       include_noise=False,
                                                                       include_jitter=True)
                                     for i in range(self.d)]),
                              axis=0)  # [D x N0 x N0].
        k_xx_naive_noisy = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x0,
                                                                             gamma=self.gamma[i],
                                                                             alpha=self.alpha[i],
                                                                             beta=self.beta[i],
                                                                             input_1=None,
                                                                             include_noise=True,
                                                                             include_jitter=True)
                                           for i in range(self.d)]),
                                    axis=0)  # [D x N0 x N0].
        k_01_naive = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x0,
                                                                       gamma=self.gamma[i],
                                                                       alpha=self.alpha[i],
                                                                       beta=self.beta[i],
                                                                       input_1=self.x1,
                                                                       include_noise=False,
                                                                       include_jitter=True)
                                     for i in range(self.d)]),
                              axis=0)  # [D x N0 x N1].
        k_10_naive = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x1,
                                                                       gamma=self.gamma[i],
                                                                       alpha=self.alpha[i],
                                                                       beta=self.beta[i],
                                                                       input_1=self.x0,
                                                                       include_noise=False,
                                                                       include_jitter=False)
                                     for i in range(self.d)]),
                              axis=0)  # [D x N1 x N0].
        k_uu_naive = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x_u,
                                                                       gamma=self.gamma[i],
                                                                       alpha=self.alpha[i],
                                                                       beta=self.beta[i],
                                                                       input_1=self.x_u,
                                                                       include_noise=False,
                                                                       include_jitter=False)
                                     for i in range(self.d)]),
                              axis=0)  # [D x M x M].
        k_uu_naive_noisy = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x_u,
                                                                             gamma=self.gamma[i],
                                                                             alpha=self.alpha[i],
                                                                             beta=self.beta[i],
                                                                             input_1=None,
                                                                             include_noise=True,
                                                                             include_jitter=True)
                                           for i in range(self.d)]),
                                    axis=0)  # [D x M x M].
        k_xmxm_naive = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x_mean,
                                                                         gamma=self.gamma[i],
                                                                         alpha=self.alpha[i],
                                                                         beta=self.beta[i],
                                                                         input_1=None,
                                                                         include_noise=False,
                                                                         include_jitter=True)
                                       for i in range(self.d)]),
                                axis=0)  # [D x N x N].
        k_xmxm_naive_noisy = np.stack(tuple([k_ard_rbf_covariance_matrix_naive(input_0=self.x_mean,
                                                                               gamma=self.gamma[i],
                                                                               alpha=self.alpha[i],
                                                                               beta=self.beta[i],
                                                                               input_1=None,
                                                                               include_noise=True,
                                                                               include_jitter=False)
                                             for i in range(self.d)]),
                                      axis=0)  # [D x N x N].

        k_xx, k_xx_noisy, \
            k_01, k_10, \
            k_uu, k_uu_noisy, \
            k_xmxm, k_xmxm_noisy = self.tf_session.run((self.kernel.covariance_matrix(input_0=self.x0,
                                                                                      input_1=None,
                                                                                      include_noise=False,
                                                                                      include_jitter=True),
                                                        self.kernel.covariance_matrix(input_0=self.x0,
                                                                                      input_1=None,
                                                                                      include_noise=True,
                                                                                      include_jitter=True),
                                                        self.kernel.covariance_matrix(input_0=self.x0,
                                                                                      input_1=self.x1,
                                                                                      include_noise=False,
                                                                                      include_jitter=True),
                                                        self.kernel.covariance_matrix(input_0=self.x1,
                                                                                      input_1=self.x0,
                                                                                      include_noise=False,
                                                                                      include_jitter=False),
                                                        self.kernel.covariance_matrix(input_0=self.x_u,
                                                                                      input_1=self.x_u,
                                                                                      include_noise=False,
                                                                                      include_jitter=False),
                                                        self.kernel.covariance_matrix(input_0=self.x_u,
                                                                                      input_1=None,
                                                                                      include_noise=True,
                                                                                      include_jitter=True),
                                                        self.kernel.covariance_matrix(input_0=self.x_mean,
                                                                                      input_1=None,
                                                                                      include_noise=False,
                                                                                      include_jitter=True),
                                                        self.kernel.covariance_matrix(input_0=self.x_mean,
                                                                                      input_1=None,
                                                                                      include_noise=True,
                                                                                      include_jitter=False)
                                                        ))

        # Compare all matrices to the naive ones.
        np.testing.assert_equal(k_xx_naive.shape, k_xx.shape)
        np.testing.assert_allclose(k_xx_naive, k_xx)
        np.testing.assert_equal(k_xx_naive_noisy.shape, k_xx_noisy.shape)
        np.testing.assert_allclose(k_xx_naive_noisy, k_xx_noisy)

        np.testing.assert_equal(k_01_naive.shape, k_01.shape)
        np.testing.assert_allclose(k_01_naive, k_01)
        np.testing.assert_equal(k_10_naive.shape, k_10.shape)
        np.testing.assert_allclose(k_10_naive, k_10)

        np.testing.assert_equal(k_uu_naive.shape, k_uu.shape)
        np.testing.assert_allclose(k_uu_naive, k_uu)
        np.testing.assert_equal(k_uu_naive_noisy.shape, k_uu_noisy.shape)
        np.testing.assert_allclose(k_uu_naive_noisy, k_uu_noisy)

        np.testing.assert_equal(k_xmxm_naive.shape, k_xmxm.shape)
        np.testing.assert_allclose(k_xmxm_naive, k_xmxm)
        np.testing.assert_equal(k_xmxm_naive_noisy.shape, k_xmxm_noisy.shape)
        np.testing.assert_allclose(k_xmxm_naive_noisy, k_xmxm_noisy)

    def test_covariance_diag(self):
        # Calculate a bunch of covariance matrix diagonals in a naive manner.
        k_x0_naive = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x0,
                                                                         gamma=self.gamma[i],
                                                                         alpha=self.alpha[i],
                                                                         beta=self.beta[i],
                                                                         include_noise=False,
                                                                         include_jitter=True)
                                     for i in range(self.d)]),
                              axis=0)  # [D x N0].
        k_x0_naive_noisy = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x0,
                                                                               gamma=self.gamma[i],
                                                                               alpha=self.alpha[i],
                                                                               beta=self.beta[i],
                                                                               include_noise=True,
                                                                               include_jitter=True)
                                           for i in range(self.d)]),
                                    axis=0)  # [D x N0].
        k_x1_naive = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x1,
                                                                         gamma=self.gamma[i],
                                                                         alpha=self.alpha[i],
                                                                         beta=self.beta[i],
                                                                         include_noise=False,
                                                                         include_jitter=False)
                                     for i in range(self.d)]),
                              axis=0)  # [D x N1].
        k_x1_naive_noisy = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x1,
                                                                               gamma=self.gamma[i],
                                                                               alpha=self.alpha[i],
                                                                               beta=self.beta[i],
                                                                               include_noise=True,
                                                                               include_jitter=False)
                                           for i in range(self.d)]),
                              axis=0)  # [D x N1].
        k_uu_naive = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x_u,
                                                                         gamma=self.gamma[i],
                                                                         alpha=self.alpha[i],
                                                                         beta=self.beta[i],
                                                                         include_noise=False,
                                                                         include_jitter=False)
                                     for i in range(self.d)]),
                              axis=0)  # [D x M].
        k_uu_naive_noisy = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x_u,
                                                                               gamma=self.gamma[i],
                                                                               alpha=self.alpha[i],
                                                                               beta=self.beta[i],
                                                                               include_noise=True,
                                                                               include_jitter=True)
                                           for i in range(self.d)]),
                                    axis=0)  # [D x M].
        k_xmxm_naive = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x_mean,
                                                                           gamma=self.gamma[i],
                                                                           alpha=self.alpha[i],
                                                                           beta=self.beta[i],
                                                                           include_noise=False,
                                                                           include_jitter=True)
                                       for i in range(self.d)]),
                                axis=0)  # [D x N].
        k_xmxm_naive_noisy = np.stack(tuple([k_ard_rbf_covariance_diagonal_naive(input_0=self.x_mean,
                                                                                 gamma=self.gamma[i],
                                                                                 alpha=self.alpha[i],
                                                                                 beta=self.beta[i],
                                                                                 include_noise=True,
                                                                                 include_jitter=False)
                                             for i in range(self.d)]),
                                      axis=0)  # [D x N].

        k_x0, k_x0_noisy, \
            k_x1, k_x1_noisy, \
            k_uu, k_uu_noisy, \
            k_xmxm, k_xmxm_noisy = self.tf_session.run((self.kernel.covariance_diag(input_0=self.x0,
                                                                                    include_noise=False,
                                                                                    include_jitter=True),
                                                        self.kernel.covariance_diag(input_0=self.x0,
                                                                                    include_noise=True,
                                                                                    include_jitter=True),
                                                        self.kernel.covariance_diag(input_0=self.x1,
                                                                                    include_noise=False,
                                                                                    include_jitter=False),
                                                        self.kernel.covariance_diag(input_0=self.x1,
                                                                                    include_noise=True,
                                                                                    include_jitter=False),
                                                        self.kernel.covariance_diag(input_0=self.x_u,
                                                                                    include_noise=False,
                                                                                    include_jitter=False),
                                                        self.kernel.covariance_diag(input_0=self.x_u,
                                                                                    include_noise=True,
                                                                                    include_jitter=True),
                                                        self.kernel.covariance_diag(input_0=self.x_mean,
                                                                                    include_noise=False,
                                                                                    include_jitter=True),
                                                        self.kernel.covariance_diag(input_0=self.x_mean,
                                                                                    include_noise=True,
                                                                                    include_jitter=False)
                                                        ))

        # Compare all matrices to the naive ones.
        np.testing.assert_equal(k_x0_naive.shape, k_x0.shape)
        np.testing.assert_allclose(k_x0_naive, k_x0)
        np.testing.assert_equal(k_x0_naive_noisy.shape, k_x0_noisy.shape)
        np.testing.assert_allclose(k_x0_naive_noisy, k_x0_noisy)

        np.testing.assert_equal(k_x1_naive.shape, k_x1.shape)
        np.testing.assert_allclose(k_x1_naive, k_x1)
        np.testing.assert_equal(k_x1_naive_noisy.shape, k_x1_noisy.shape)
        np.testing.assert_allclose(k_x1_naive_noisy, k_x1_noisy)

        np.testing.assert_equal(k_uu_naive.shape, k_uu.shape)
        np.testing.assert_allclose(k_uu_naive, k_uu)
        np.testing.assert_equal(k_uu_naive_noisy.shape, k_uu_noisy.shape)
        np.testing.assert_allclose(k_uu_naive_noisy, k_uu_noisy)

        np.testing.assert_equal(k_xmxm_naive.shape, k_xmxm.shape)
        np.testing.assert_allclose(k_xmxm_naive, k_xmxm)
        np.testing.assert_equal(k_xmxm_naive_noisy.shape, k_xmxm_noisy.shape)
        np.testing.assert_allclose(k_xmxm_naive_noisy, k_xmxm_noisy)

    def test_psi_0(self):
        # Calculate psi 0s in a naive manner.
        psi_0_xx_naive = np.stack(tuple([k_ard_rbf_psi_0_naive(self.n, self.alpha[i])
                                         for i in range(self.d)]),
                                  axis=0)  # [D x 1].
        psi_0_x0_naive = np.stack(tuple([k_ard_rbf_psi_0_naive(self.n0, self.alpha[i])
                                         for i in range(self.d)]),
                                  axis=0)  # [D x 1].
        psi_0_x1_naive = np.stack(tuple([k_ard_rbf_psi_0_naive(self.n1, self.alpha[i])
                                         for i in range(self.d)]),
                                  axis=0)  # [D x 1].
        psi_0_xu_naive = np.stack(tuple([k_ard_rbf_psi_0_naive(self.m, self.alpha[i])
                                         for i in range(self.d)]),
                                  axis=0)  # [D x 1].

        psi_0_xx, psi_0_x0, psi_0_x1, psi_0_xu = self.tf_session.run((
            self.kernel.psi_0(inducing_input=self.x_u,
                              latent_input_mean=self.x,
                              latent_input_covariance=self.x_covar),
            self.kernel.psi_0(inducing_input=self.x_u,
                              latent_input_mean=self.x0,
                              latent_input_covariance=self.x_covar),
            self.kernel.psi_0(inducing_input=self.x_u,
                              latent_input_mean=self.x1,
                              latent_input_covariance=self.x_covar),
            self.kernel.psi_0(inducing_input=self.x_u,
                              latent_input_mean=self.x_u,
                              latent_input_covariance=self.x_covar)
        ))

        # Compare all psi 0s to the naive ones.
        np.testing.assert_equal(psi_0_xx_naive.shape, psi_0_xx.shape)
        np.testing.assert_allclose(psi_0_xx_naive, psi_0_xx)
        np.testing.assert_equal(psi_0_x0_naive.shape, psi_0_x0.shape)
        np.testing.assert_allclose(psi_0_x0_naive, psi_0_x0)
        np.testing.assert_equal(psi_0_x1_naive.shape, psi_0_x1.shape)
        np.testing.assert_allclose(psi_0_x1_naive, psi_0_x1)
        np.testing.assert_equal(psi_0_xu_naive.shape, psi_0_xu.shape)
        np.testing.assert_allclose(psi_0_xu_naive, psi_0_xu)

    def test_psi_1(self):
        # Calculate psi 1 in a naive manner.
        psi_1_naive = np.stack(tuple([k_ard_rbf_psi_1_naive(self.x_mean, self.x_var, self.x_u,
                                                            self.gamma[i], self.alpha[i])
                                      for i in range(self.d)]),
                               axis=0)  # [D x N x M].

        psi_1 = self.tf_session.run(self.kernel.psi_1(inducing_input=self.x_u,
                                                      latent_input_mean=self.x_mean,
                                                      latent_input_covariance=self.x_covar))

        # Compare psi 1 to naive one.
        np.testing.assert_equal(psi_1_naive.shape, psi_1.shape)
        np.testing.assert_allclose(psi_1_naive, psi_1)

    def test_psi_2(self):
        # Calculate psi 2 in a naive manner.
        psi_2_naive = np.stack(tuple([k_ard_rbf_psi_2_naive(self.x_mean, self.x_var, self.x_u,
                                                            self.gamma[i], self.alpha[i])
                                      for i in range(self.d)]),
                               axis=0)  # [D x M x M].

        psi_2 = self.tf_session.run(self.kernel.psi_2(inducing_input=self.x_u,
                                                      latent_input_mean=self.x_mean,
                                                      latent_input_covariance=self.x_covar))

        # Compare psi 2 to naive one.
        np.testing.assert_equal(psi_2_naive.shape, psi_2.shape)
        np.testing.assert_allclose(psi_2_naive, psi_2)
