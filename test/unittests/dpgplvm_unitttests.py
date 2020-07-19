"""
This file defines unit tests for the DP-GP-LVM implementation.
"""

from src.distributions.log_normal import log_pdf
from src.kernels.rbf_kernel import k_ard_rbf
from src.models.dp_gp_lvm import dp_gp_lvm, dp_gp_lvm_t
from src.utils.constants import DP_DEFAULT_ALPHA_PRIOR_PARAMS
from src.utils.expressions import print_and_log
from src.utils.types import NP_DTYPE, TF_DTYPE
import src.visualisation.plotters as vis
from test.unittests.dp_unittests import elbo_naive as dp_elbo_naive
from test.unittests.bgplvm_unittests import free_energy_naive as gp_f_naive
from test.unittests.bgplvm_unittests import free_energy_stable as gp_f_stable
from test.unittests.bgplvm_unittests import kl_qx_px_naive

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from time import time
import unittest


class TestDPGPLVM(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """
        np.random.seed(seed=seed)

        self.n = 50
        self.d = 10
        self.m = 25
        self.q = 3
        self.t = 8
        self.s_1 = np.square(np.random.normal(loc=1.0, scale=0.05))
        self.s_2 = np.square(np.random.normal(loc=1.0, scale=0.05))
        alpha_prior_params = np.array([self.s_1, self.s_2])

        self.y = np.random.standard_normal((self.n, self.d)).astype(NP_DTYPE)  # [N x D].
        self.y_nd1 = self.y[:, :, np.newaxis]  # [N x D x 1].

        self.dpgplvm = dp_gp_lvm(y_train=self.y,
                                 num_latent_dims=self.q,
                                 num_inducing_points=self.m,
                                 truncation_level=self.t,
                                 alpha_prior_params=alpha_prior_params)

        # TensorFlow session.
        self.tf_session = tf.Session()

        # Init variables so we can get variational parameters.
        self.tf_session.run(tf.global_variables_initializer())
        self.x_mean, self.x_covar = self.tf_session.run(self.dpgplvm.q_x)
        self.x_var = np.stack(tuple([np.diag(self.x_covar[i]) for i in range(self.n)]), axis=0)  # [N x Q].
        self.x_u = self.tf_session.run(self.dpgplvm.inducing_input)
        self.phi = self.tf_session.run(self.dpgplvm.dp.q_z)
        self.gamma_1, self.gamma_2 = self.tf_session.run(self.dpgplvm.dp.q_v)
        self.w_1, self.w_2 = self.tf_session.run(self.dpgplvm.dp.q_alpha)

        # Get kernel hyperparameters.
        self.ard = self.tf_session.run(self.dpgplvm.ard_weights)  # [D x Q].
        self.alpha = self.tf_session.run(self.dpgplvm.signal_variance)  # [D x 1].
        self.beta = self.tf_session.run(self.dpgplvm.noise_precision)  # [D x 1].

        # Get DP atoms. ARD is [T x Q], alpha is [T x 1], and beta is [T x 1].
        self.ard_atoms, self.alpha_atom, self.beta_atom = self.tf_session.run(self.dpgplvm.dp_atoms)

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_objective(self):
        # Calculate DP ELBO.
        dp_elbo = dp_elbo_naive(phi=self.phi, gamma_1=self.gamma_1, gamma_2=self.gamma_2,
                                w_1=self.w_1, w_2=self.w_2,s_1=self.s_1, s_2=self.s_2)

        # Calculate GP-LVM ELBO naively.
        gp_naive_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_naive_elbo += gp_f_naive(y=self.y_nd1[:, i, :],
                                        x_mean=self.x_mean,
                                        x_var=self.x_var,
                                        x_u=self.x_u,
                                        gamma=self.ard[i, :],
                                        alpha=self.alpha[i],
                                        beta=self.beta[i])

        # Calculate GP-LVM ELBO stably.
        gp_stable_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_stable_elbo += self.tf_session.run(gp_f_stable(y=self.y_nd1[:, i, :],
                                                              x_mean=self.x_mean,
                                                              x_var=self.x_var,
                                                              x_u=self.x_u,
                                                              gamma=self.ard[i, :],
                                                              alpha=self.alpha[i],
                                                              beta=self.beta[i]))

        # Calculate hyperprior for kernel hyperparameters.
        hyper_prior = self.tf_session.run(tf.reduce_sum(log_pdf(self.ard_atoms)) +
                                          tf.reduce_sum(log_pdf(self.alpha_atom)) +
                                          tf.reduce_sum(log_pdf(self.beta_atom)))

        # Calculate objective naively.
        objective_naive = np.negative(dp_elbo + gp_naive_elbo + hyper_prior)

        # Calculate objective stably.
        objective_stable = np.negative(dp_elbo + gp_stable_elbo + hyper_prior)

        # Calculate objective from the TF model.
        dp_objective_val = self.tf_session.run(self.dpgplvm.dp.objective)
        objective_val = self.tf_session.run(self.dpgplvm.objective)

        # Double check DP objective since it is easy to query. DP objective is close to zero when T=1.
        np.testing.assert_allclose(np.negative(dp_elbo), dp_objective_val, atol=1.0e-10)

        # Check approximately equal.
        np.testing.assert_allclose(objective_naive, objective_stable)
        np.testing.assert_allclose(objective_naive, objective_val)
        np.testing.assert_allclose(objective_stable, objective_val)

    def test_prediction(self):
        # TODO: Add test for predicting missing data.
        pass


class TestT1(unittest.TestCase):

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
        self.t = 1
        self.s_1 = np.square(np.random.normal(loc=1.0, scale=0.05))
        self.s_2 = np.square(np.random.normal(loc=1.0, scale=0.05))
        alpha_prior_params = np.array([self.s_1, self.s_2])

        self.y = np.random.standard_normal((self.n, self.d)).astype(NP_DTYPE)  # [N x D].
        self.y_nd1 = self.y[:, :, np.newaxis]  # [N x D x 1].

        self.dpgplvm = dp_gp_lvm(y_train=self.y,
                                 num_latent_dims=self.q,
                                 num_inducing_points=self.m,
                                 truncation_level=self.t,
                                 alpha_prior_params=alpha_prior_params)

        # TensorFlow session.
        self.tf_session = tf.Session()

        # Init variables so we can get variational parameters.
        self.tf_session.run(tf.global_variables_initializer())
        self.x_mean, self.x_covar = self.tf_session.run(self.dpgplvm.q_x)
        self.x_var = np.stack(tuple([np.diag(self.x_covar[i]) for i in range(self.n)]), axis=0)  # [N x Q].
        self.x_u = self.tf_session.run(self.dpgplvm.inducing_input)
        self.phi = self.tf_session.run(self.dpgplvm.dp.q_z)
        self.gamma_1, self.gamma_2 = self.tf_session.run(self.dpgplvm.dp.q_v)
        self.w_1, self.w_2 = self.tf_session.run(self.dpgplvm.dp.q_alpha)

        # Get kernel hyperparameters.
        self.ard = self.tf_session.run(self.dpgplvm.ard_weights)  # [D x Q].
        self.alpha = self.tf_session.run(self.dpgplvm.signal_variance)  # [D x 1].
        self.beta = self.tf_session.run(self.dpgplvm.noise_precision)  # [D x 1].

        # Get DP atoms. ARD is [T x Q], alpha is [T x 1], and beta is [T x 1].
        self.ard_atoms, self.alpha_atom, self.beta_atom = self.tf_session.run(self.dpgplvm.dp_atoms)

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_objective(self):
        # Calculate DP ELBO.
        dp_elbo = dp_elbo_naive(phi=self.phi, gamma_1=self.gamma_1, gamma_2=self.gamma_2,
                                w_1=self.w_1, w_2=self.w_2, s_1=self.s_1, s_2=self.s_2)

        # Calculate GP-LVM ELBO naively.
        gp_naive_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_naive_elbo += gp_f_naive(y=self.y_nd1[:, i, :],
                                        x_mean=self.x_mean,
                                        x_var=self.x_var,
                                        x_u=self.x_u,
                                        gamma=self.ard[i, :],
                                        alpha=self.alpha[i],
                                        beta=self.beta[i])

        # Calculate GP-LVM ELBO stably.
        gp_stable_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_stable_elbo += self.tf_session.run(gp_f_stable(y=self.y_nd1[:, i, :],
                                                              x_mean=self.x_mean,
                                                              x_var=self.x_var,
                                                              x_u=self.x_u,
                                                              gamma=self.ard[i, :],
                                                              alpha=self.alpha[i],
                                                              beta=self.beta[i]))

        # Calculate hyperprior for kernel hyperparameters.
        hyper_prior = self.tf_session.run(tf.reduce_sum(log_pdf(self.ard_atoms)) +
                                          tf.reduce_sum(log_pdf(self.alpha_atom)) +
                                          tf.reduce_sum(log_pdf(self.beta_atom)))

        # Calculate objective naively.
        objective_naive = np.negative(dp_elbo + gp_naive_elbo + hyper_prior)

        # Calculate objective stably.
        objective_stable = np.negative(dp_elbo + gp_stable_elbo + hyper_prior)

        # Calculate objective from the TF model.
        dp_objective_val = self.tf_session.run(self.dpgplvm.dp.objective)
        objective_val = self.tf_session.run(self.dpgplvm.objective)

        # Double check DP objective since it is easy to query. DP objective is close to zero when T=1.
        np.testing.assert_allclose(np.negative(dp_elbo), dp_objective_val, atol=1.0e-10)

        # Check approximately equal.
        np.testing.assert_allclose(objective_naive, objective_stable)
        np.testing.assert_allclose(objective_naive, objective_val)
        np.testing.assert_allclose(objective_stable, objective_val)

    def test_prediction(self):
        # TODO: Add test for predicting missing data.
        pass


class TestD2T1(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """
        np.random.seed(seed=seed)

        self.n = 5  # 20
        self.d = 2
        self.m = 3  # 10
        self.q = 1
        self.t = 1  # T must be less than d.
        self.s_1 = np.square(np.random.normal(loc=1.0, scale=0.05))
        self.s_2 = np.square(np.random.normal(loc=1.0, scale=0.05))
        alpha_prior_params = np.array([self.s_1, self.s_2])

        self.y = np.random.standard_normal((self.n, self.d)).astype(NP_DTYPE)  # [N x D].
        self.y_nd1 = self.y[:, :, np.newaxis]  # [N x D x 1].

        self.dpgplvm = dp_gp_lvm(y_train=self.y,
                                 num_latent_dims=self.q,
                                 num_inducing_points=self.m,
                                 truncation_level=self.t,
                                 alpha_prior_params=alpha_prior_params)

        # TensorFlow session.
        self.tf_session = tf.Session()

        # Init variables so we can get variational parameters.
        self.tf_session.run(tf.global_variables_initializer())
        self.x_mean, self.x_covar = self.tf_session.run(self.dpgplvm.q_x)
        self.x_var = np.stack(tuple([np.diag(self.x_covar[i]) for i in range(self.n)]), axis=0)  # [N x Q].
        self.x_u = self.tf_session.run(self.dpgplvm.inducing_input)
        self.phi = self.tf_session.run(self.dpgplvm.dp.q_z)
        self.gamma_1, self.gamma_2 = self.tf_session.run(self.dpgplvm.dp.q_v)
        self.w_1, self.w_2 = self.tf_session.run(self.dpgplvm.dp.q_alpha)

        # Get kernel hyperparameters.
        self.ard = self.tf_session.run(self.dpgplvm.ard_weights)  # [D x Q].
        self.alpha = self.tf_session.run(self.dpgplvm.signal_variance)  # [D x 1].
        self.beta = self.tf_session.run(self.dpgplvm.noise_precision)  # [D x 1].

        # Get DP atoms. ARD is [T x Q], alpha is [T x 1], and beta is [T x 1].
        self.ard_atoms, self.alpha_atom, self.beta_atom = self.tf_session.run(self.dpgplvm.dp_atoms)

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_objective(self):
        # Calculate DP ELBO.
        dp_elbo = dp_elbo_naive(phi=self.phi, gamma_1=self.gamma_1, gamma_2=self.gamma_2,
                                w_1=self.w_1, w_2=self.w_2, s_1=self.s_1, s_2=self.s_2)

        # Calculate GP-LVM ELBO naively.
        gp_naive_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_naive_elbo += gp_f_naive(y=self.y_nd1[:, i, :],
                                        x_mean=self.x_mean,
                                        x_var=self.x_var,
                                        x_u=self.x_u,
                                        gamma=self.ard[i, :],
                                        alpha=self.alpha[i],
                                        beta=self.beta[i])

        # Calculate GP-LVM ELBO stably.
        gp_stable_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_stable_elbo += self.tf_session.run(gp_f_stable(y=self.y_nd1[:, i, :],
                                                              x_mean=self.x_mean,
                                                              x_var=self.x_var,
                                                              x_u=self.x_u,
                                                              gamma=self.ard[i, :],
                                                              alpha=self.alpha[i],
                                                              beta=self.beta[i]))

        # Calculate hyperprior for kernel hyperparameters.
        hyper_prior = self.tf_session.run(tf.reduce_sum(log_pdf(self.ard_atoms)) +
                                          tf.reduce_sum(log_pdf(self.alpha_atom)) +
                                          tf.reduce_sum(log_pdf(self.beta_atom)))

        # Calculate objective naively.
        objective_naive = np.negative(dp_elbo + gp_naive_elbo + hyper_prior)

        # Calculate objective stably.
        objective_stable = np.negative(dp_elbo + gp_stable_elbo + hyper_prior)

        # Calculate objective from the TF model.
        dp_objective_val = self.tf_session.run(self.dpgplvm.dp.objective)
        objective_val = self.tf_session.run(self.dpgplvm.objective)

        # Double check DP objective since it is easy to query. DP objective is close to zero when T=1.
        np.testing.assert_allclose(np.negative(dp_elbo), dp_objective_val, atol=1.0e-10)

        # Check approximately equal.
        np.testing.assert_allclose(objective_naive, objective_stable)
        np.testing.assert_allclose(objective_naive, objective_val)
        np.testing.assert_allclose(objective_stable, objective_val)

    def test_prediction(self):
        # TODO: Add test for predicting missing data.
        pass


class TestD1T1(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """
        np.random.seed(seed=seed)

        self.n = 5  # 20
        self.d = 1
        self.m = 3  # 10
        self.q = 1
        self.t = 1  # T must be less than d.
        self.s_1 = np.square(np.random.normal(loc=1.0, scale=0.05))
        self.s_2 = np.square(np.random.normal(loc=1.0, scale=0.05))
        alpha_prior_params = np.array([self.s_1, self.s_2])

        self.y = np.random.standard_normal((self.n, self.d)).astype(NP_DTYPE)  # [N x D].
        self.y_nd1 = self.y[:, :, np.newaxis]  # [N x D x 1].

        self.dpgplvm = dp_gp_lvm(y_train=self.y,
                                 num_latent_dims=self.q,
                                 num_inducing_points=self.m,
                                 truncation_level=self.t,
                                 alpha_prior_params=alpha_prior_params)

        # TensorFlow session.
        self.tf_session = tf.Session()

        # Init variables so we can get variational parameters.
        self.tf_session.run(tf.global_variables_initializer())
        self.x_mean, self.x_covar = self.tf_session.run(self.dpgplvm.q_x)
        self.x_var = np.stack(tuple([np.diag(self.x_covar[i]) for i in range(self.n)]), axis=0)  # [N x Q].
        self.x_u = self.tf_session.run(self.dpgplvm.inducing_input)
        self.phi = self.tf_session.run(self.dpgplvm.dp.q_z)
        self.gamma_1, self.gamma_2 = self.tf_session.run(self.dpgplvm.dp.q_v)
        self.w_1, self.w_2 = self.tf_session.run(self.dpgplvm.dp.q_alpha)

        # Get kernel hyperparameters.
        self.ard = self.tf_session.run(self.dpgplvm.ard_weights)  # [D x Q].
        self.alpha = self.tf_session.run(self.dpgplvm.signal_variance)  # [D x 1].
        self.beta = self.tf_session.run(self.dpgplvm.noise_precision)  # [D x 1].

        # Get DP atoms. ARD is [T x Q], alpha is [T x 1], and beta is [T x 1].
        self.ard_atoms, self.alpha_atom, self.beta_atom = self.tf_session.run(self.dpgplvm.dp_atoms)

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_objective(self):
        # Calculate DP ELBO.
        dp_elbo = dp_elbo_naive(phi=self.phi, gamma_1=self.gamma_1, gamma_2=self.gamma_2,
                                w_1=self.w_1, w_2=self.w_2, s_1=self.s_1, s_2=self.s_2)

        # Calculate GP-LVM ELBO naively.
        gp_naive_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_naive_elbo += gp_f_naive(y=self.y_nd1[:, i, :],
                                        x_mean=self.x_mean,
                                        x_var=self.x_var,
                                        x_u=self.x_u,
                                        gamma=self.ard[i, :],
                                        alpha=self.alpha[i],
                                        beta=self.beta[i])

        # Calculate GP-LVM ELBO stably.
        gp_stable_elbo = np.negative(kl_qx_px_naive(x_mean=self.x_mean, x_var=self.x_var))
        for i in range(self.d):
            gp_stable_elbo += self.tf_session.run(gp_f_stable(y=self.y_nd1[:, i, :],
                                                              x_mean=self.x_mean,
                                                              x_var=self.x_var,
                                                              x_u=self.x_u,
                                                              gamma=self.ard[i, :],
                                                              alpha=self.alpha[i],
                                                              beta=self.beta[i]))

        # Calculate hyperprior for kernel hyperparameters.
        hyper_prior = self.tf_session.run(tf.reduce_sum(log_pdf(self.ard_atoms)) +
                                          tf.reduce_sum(log_pdf(self.alpha_atom)) +
                                          tf.reduce_sum(log_pdf(self.beta_atom)))

        # Calculate objective naively.
        objective_naive = np.negative(dp_elbo + gp_naive_elbo + hyper_prior)

        # Calculate objective stably.
        objective_stable = np.negative(dp_elbo + gp_stable_elbo + hyper_prior)

        # Calculate objective from the TF model.
        dp_objective_val = self.tf_session.run(self.dpgplvm.dp.objective)
        objective_val = self.tf_session.run(self.dpgplvm.objective)

        # Double check DP objective since it is easy to query. DP objective is close to zero when T=1.
        np.testing.assert_allclose(np.negative(dp_elbo), dp_objective_val, atol=1.0e-10)

        # Check approximately equal.
        np.testing.assert_allclose(objective_naive, objective_stable)
        np.testing.assert_allclose(objective_naive, objective_val)
        np.testing.assert_allclose(objective_stable, objective_val)

    def test_prediction(self):
        # TODO: Add test for predicting missing data.
        pass


class TestFasterDPGPLVM(unittest.TestCase):

    def setUp(self, seed=1):
        """
        TODO
        :param seed:
        :return:
        """

        # Set model parameters.
        self.n = 200
        self.d = 22
        self.m = 75
        self.q = 10
        self.t = 20
        self.learning_rate = 0.01  # 0.05
        self.train_iter = 5000  # 2500

        # Generate simple toy data.
        np.random.seed(seed=seed)
        x = np.random.standard_normal((self.n, 3)).astype(NP_DTYPE)  # Random input. [N x 3].
        ard_1 = tf.constant([[1.0, 1.0, 0.0]], dtype=TF_DTYPE)  # Function of X0 and X1.
        ard_2 = tf.constant([[1.0, 0.0, 1.0]], dtype=TF_DTYPE)  # Function of X0 and X2.
        # Different noise terms per GP component.
        # beta_1 = tf.constant([[10.0]], dtype=TF_DTYPE)  # Noise var is 0.1.
        # beta_2 = tf.constant([[200.0]], dtype=TF_DTYPE)  # Noise var is 0.005.
        beta_1 = tf.constant([[20.0]], dtype=TF_DTYPE)  # Noise var is 0.05.
        # beta_1 = tf.constant([[100.0]], dtype=TF_DTYPE)  # Noise var is 0.01.
        beta_2 = tf.constant([[500.0]], dtype=TF_DTYPE)  # Noise var is 0.002.
        alpha_0 = tf.constant([[1.0]], dtype=TF_DTYPE)
        covar_1 = k_ard_rbf(gamma=ard_1, alpha=alpha_0, beta=beta_1).covariance_matrix(input_0=x,
                                                                                       input_1=None,
                                                                                       include_noise=True,
                                                                                       include_jitter=True)
        covar_2 = k_ard_rbf(gamma=ard_2, alpha=alpha_0, beta=beta_2).covariance_matrix(input_0=x,
                                                                                       input_1=None,
                                                                                       include_noise=True,
                                                                                       include_jitter=True)
        with tf.Session() as sess:
            covar_1_np = np.squeeze(sess.run(covar_1), axis=0)  # [N x N].
            covar_2_np = np.squeeze(sess.run(covar_2), axis=0)  # [N x N].
        d_gp = int(0.5 * self.d)
        self.y = np.concatenate(
            (
                np.transpose(np.random.multivariate_normal(mean=np.zeros(self.n), cov=covar_1_np, size=d_gp)),
                np.transpose(np.random.multivariate_normal(mean=np.zeros(self.n), cov=covar_2_np, size=d_gp)),
            ), axis=1)  # [N x D].
        assert self.y.shape[0] == self.n
        assert self.y.shape[1] == self.d

        # Define DP-GP-LVM over D. This uses more GPU memory.
        np.random.seed(seed=seed)
        self.dpgplvm_d = dp_gp_lvm(y_train=self.y,
                                   num_latent_dims=self.q,
                                   num_inducing_points=self.m,
                                   truncation_level=self.t,
                                   alpha_prior_params=DP_DEFAULT_ALPHA_PRIOR_PARAMS)
        self.dpgplvm_d_objective = self.dpgplvm_d.objective

        # Define DP-GP-LVM over T. Hopefully this is faster and uses less GPU memory.
        self.dpgplvm_t = dp_gp_lvm_t(y_train=self.y,
                                     num_latent_dims=self.q,
                                     num_inducing_points=self.m,
                                     truncation_level=self.t,
                                     alpha_prior_params=DP_DEFAULT_ALPHA_PRIOR_PARAMS,
                                     seed=seed)
        self.dpgplvm_t_objective = self.dpgplvm_t.objective

        # Define optimiser for each model.
        self.dpgplvm_d_opt_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            loss=self.dpgplvm_d_objective)
        self.dpgplvm_t_opt_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            loss=self.dpgplvm_t_objective)

        # Define TensorFlow session and initialise variables.
        self.tf_session = tf.Session()
        self.tf_session.run(tf.global_variables_initializer())

    def tearDown(self):
        """
        Close the TensorFlow session.
        """
        self.tf_session.close()

    def test_objective(self):

        # Assert starting objectives are same.
        np.testing.assert_almost_equal(self.tf_session.run(self.dpgplvm_t_objective),
                                       self.tf_session.run(self.dpgplvm_d_objective))

        # Optimise models.
        start_time = time()
        print_and_log('\nTraining DP-GP-LVM over D:')
        for c in range(self.train_iter):
            self.tf_session.run(self.dpgplvm_d_opt_train)
            if (c % 100) == 0:
                print_and_log('  DP-GP-LVM over D opt iter {:5}: {}'.format(c, self.tf_session.run(
                    self.dpgplvm_d_objective)))
        end_time = time()
        dp_gp_lvm_d_opt_time = end_time - start_time
        print_and_log('Final iter {}, DP-GP-LVM over D: {}'.format(c, self.tf_session.run(self.dpgplvm_d_objective)))
        print_and_log('Time to optimise: {} s'.format(dp_gp_lvm_d_opt_time))

        start_time = time()
        print_and_log('\nTraining DP-GP-LVM over T:')
        for c in range(self.train_iter):
            self.tf_session.run(self.dpgplvm_t_opt_train)
            if (c % 100) == 0:
                print_and_log('  DP-GP-LVM over T opt iter {:5}: {}'.format(c, self.tf_session.run(
                    self.dpgplvm_t_objective)))
        end_time = time()
        dp_gp_lvm_t_opt_time = end_time - start_time
        print_and_log('Final iter {}, DP-GP-LVM over T: {}'.format(c, self.tf_session.run(self.dpgplvm_t_objective)))
        print_and_log('Time to optimise: {} s'.format(dp_gp_lvm_t_opt_time))

        # Check optimisation time for model over T is faster than model over D.
        np.testing.assert_array_less(dp_gp_lvm_t_opt_time, dp_gp_lvm_d_opt_time)

        # TODO: Check that models find same solution.
        # Print converged model parameters.
        phi_d, dp_atoms_d, phi_t, dp_atoms_t = self.tf_session.run((self.dpgplvm_d.assignments,
                                                                    self.dpgplvm_d.dp_atoms,
                                                                    self.dpgplvm_t.assignments,
                                                                    self.dpgplvm_t.dp_atoms))
        ard_atoms_d, alpha_atoms_d, beta_atoms_d = dp_atoms_d
        ard_atoms_t, alpha_atoms_t, beta_atoms_t = dp_atoms_t
        # print('\nPhi for DP-GP-LVM over D:\n{}'.format(phi_d))
        # print('\nPhi for DP-GP-LVM over T:\n{}'.format(phi_t))
        print_and_log('\nBeta atoms for DP-GP-LVM over D:\n{}'.format(beta_atoms_d))
        print_and_log('\nBeta atoms for DP-GP-LVM over T:\n{}'.format(beta_atoms_t))

        # Plot results.
        save_plots = True
        show_plots = False
        # vis.plot_ard(np.sqrt())  # TODO: Plot true ARD weights.
        vis.plot_ard(np.sqrt(ard_atoms_d))
        if save_plots:
            plot.savefig('Sqrt of ARD Atoms for DP-GP-LVM over D.pdf', bbox_inches='tight')
        vis.plot_phi_matrix(phi_d)
        if save_plots:
            plot.savefig('DP Assignments (phi) for DP-GP-LVM over D.pdf', bbox_inches='tight')
        vis.plot_ard(np.sqrt(ard_atoms_t))
        if save_plots:
            plot.savefig('Sqrt of ARD Atoms for DP-GP-LVM over T.pdf', bbox_inches='tight')
        vis.plot_phi_matrix(phi_t)
        if save_plots:
            plot.savefig('DP Assignments (phi) for DP-GP-LVM over T.pdf', bbox_inches='tight')
        if show_plots:
            plot.show()
