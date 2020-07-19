"""
This file defines unit tests for the DP implementation.
"""

from src.models.dirichlet_process import dirichlet_process

import numpy as np
from scipy.special import digamma, gammaln
import tensorflow as tf
import unittest


def ev_q_log_p_z_given_v_naive(phi, gamma_1, gamma_2):
    """
    TODO
    :param phi:
    :param gamma_1:
    :param gamma_2:
    :return:
    """

    # Validate shapes.
    [n, t] = np.shape(phi)
    assert t == np.size(gamma_1) + 1, 'q(V) must have T-1 variational parameters.'
    assert t == np.size(gamma_2) + 1, 'q(V) must have T-1 variational parameters.'

    # Calculate expected value.
    reverse_phi = phi[:, ::-1]
    phi_cumsum = np.cumsum(reverse_phi, axis=-1)
    reverse_phi_cumsum = phi_cumsum[:, ::-1]
    ev_q_log_p_z_given_v = np.sum(phi[:, 0:-1] * (digamma(gamma_1) - digamma(gamma_1 + gamma_2)) +
                                  reverse_phi_cumsum[:, 1:] * (digamma(gamma_2) - digamma(gamma_1 + gamma_2)))

    return ev_q_log_p_z_given_v


def ev_q_log_p_v_given_alpha_naive(gamma_1, gamma_2, w_1, w_2):
    """
    TODO
    :param gamma_1:
    :param gamma_2:
    :param w_1:
    :param w_2:
    :return:
    """

    # Validate shapes.
    t = np.size(gamma_1) + 1
    assert t == np.size(gamma_2) + 1, 'q(V) must have T-1 variational parameters.'

    # Calculate expected value.
    ev_q_log_p_v_given_alpha = (t - 1.0) * (digamma(w_1) - np.log(w_2)) + \
        ((w_1 / w_2) - 1.0) * np.sum(digamma(gamma_2) - digamma(gamma_1 + gamma_2))

    return ev_q_log_p_v_given_alpha


def ev_q_log_p_alpha_naive(w_1, w_2, s_1, s_2):
    """
    TODO
    :param w_1:
    :param w_2:
    :param s_1:
    :param s_2:
    :return:
    """

    # Calculate expected value.
    ev_q_log_p_alpha = s_1 * np.log(s_2) - gammaln(s_1) + (s_1 - 1.0) * (digamma(w_1) - np.log(w_2)) - \
        s_2 * (w_1 / w_2)

    return ev_q_log_p_alpha


def entropy_q_z_naive(phi):
    """
    TODO
    :param phi:
    :return:
    """

    return np.sum(np.negative(np.sum(phi * np.log(phi), axis=-1)))


def entropy_q_v_naive(gamma_1, gamma_2):
    """
    TODO
    :param gamma_1:
    :param gamma_2:
    :return:
    """
    total_concentration = gamma_1 + gamma_2

    return np.sum(gammaln(gamma_1) + gammaln(gamma_2) - gammaln(total_concentration) -
                  (gamma_1 - 1.0) * digamma(gamma_1) - (gamma_2 - 1.0) * digamma(gamma_2) +
                  (total_concentration - 2.0) * digamma(total_concentration))


def entropy_q_alpha_naive(w_1, w_2):
    """
    TODO
    :param w_1:
    :param w_2:
    :return:
    """

    return w_1 - np.log(w_2) + gammaln(w_1) + (1.0 - w_1) * digamma(w_1)


def elbo_naive(phi, gamma_1, gamma_2, w_1, w_2, s_1, s_2):
    """
    TODO
    :param phi:
    :param gamma_1:
    :param gamma_2:
    :param w_1:
    :param w_2:
    :param s_1:
    :param s_2:
    :return:
    """

    # Calculate evidence lower bound (ELBO).
    elbo = ev_q_log_p_z_given_v_naive(phi=phi, gamma_1=gamma_1, gamma_2=gamma_2) + \
        ev_q_log_p_v_given_alpha_naive(gamma_1=gamma_1, gamma_2=gamma_2, w_1=w_1, w_2=w_2) + \
        ev_q_log_p_alpha_naive(w_1=w_1, w_2=w_2, s_1=s_1, s_2=s_2) + \
        entropy_q_z_naive(phi=phi) + \
        entropy_q_v_naive(gamma_1=gamma_1, gamma_2=gamma_2) + \
        entropy_q_alpha_naive(w_1=w_1, w_2=w_2)

    return elbo


class TestDP(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """
        np.random.seed(seed=seed)

        self.n = 10
        self.t = 20
        self.s_1 = np.square(np.random.normal(loc=1.0, scale=0.05))
        self.s_2 = np.square(np.random.normal(loc=1.0, scale=0.05))
        alpha_prior_params = np.array([self.s_1, self.s_2])

        self.dp = dirichlet_process(num_samples=self.n, truncation_level=self.t, alpha_prior_params=alpha_prior_params)

        # TensorFlow session.
        self.tf_session = tf.Session()

        # Init variables so we can get variational parameters from DP.
        self.tf_session.run(tf.global_variables_initializer())
        self.phi = self.tf_session.run(self.dp.q_z)
        self.gamma_1, self.gamma_2 = self.tf_session.run(self.dp.q_v)
        self.w_1, self.w_2 = self.tf_session.run(self.dp.q_alpha)

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_objective(self):
        # Calculate objective naively.
        objective_naive = np.negative(elbo_naive(phi=self.phi, gamma_1=self.gamma_1, gamma_2=self.gamma_2,
                                                 w_1=self.w_1, w_2=self.w_2, s_1=self.s_1, s_2=self.s_2))

        objective_val = self.tf_session.run(self.dp.objective)

        np.testing.assert_allclose(objective_naive, objective_val)
