"""
This module implements the DP-GP-LVM bgplvm_model.
"""

from src.distributions.log_normal import log_pdf as log_normal_log_pdf
from src.kernels.interfaces.kernel import KernelHyperparameters
from src.kernels.rbf_kernel import k_ard_rbf
from src.models.interfaces.trainable import Trainable
from src.models.expressions.gp_expressions import calculate_kl_divergence_standard_prior
from src.models.dirichlet_process import dirichlet_process
from src.utils.constants import GP_LVM_DEFAULT_LATENT_DIMENSIONS, GP_LVM_DEFAULT_NUM_INDUCING_POINTS, \
    DP_DEFAULT_TRUNCATION_LEVEL, DP_DEFAULT_ALPHA_PRIOR_PARAMS, GP_INIT_GAMMA, GP_INIT_ALPHA, GP_INIT_BETA, \
    MAX_MC_SAMPLES
from src.utils.expressions import nearest_neighbour, empirical_mean, principal_component_analysis as pca
from src.utils.types import TF_DTYPE, create_positive_variable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def dp_gp_lvm(y_train,
              num_latent_dims=GP_LVM_DEFAULT_LATENT_DIMENSIONS,
              num_inducing_points=GP_LVM_DEFAULT_NUM_INDUCING_POINTS,
              truncation_level=DP_DEFAULT_TRUNCATION_LEVEL,
              alpha_prior_params=DP_DEFAULT_ALPHA_PRIOR_PARAMS,
              mask_size=1):
    """
    This function initialises a DP_GP_LVM object.
    :param y_train: The observed data, which is assumed to be normalised with zero mean and unit variance for each
    column. Must be provided as [N x D] numpy array, where N is the number of observations and D is the number of
    observed dimensions.
    :param num_latent_dims: A positive integer specifying the number of latent dimensions, referred to as Q. Q must be
    less than or equal to the minimum of the number of observations and the number of observed dimensions, i.e.,
    Q <= min(N,D), and Q is normally much smaller than D, i.e., Q << D.
    :param num_inducing_points: A positive integer specifying the number of inducing inputs, referred to as M, for the
    GP. M must be less than or equal to the number of observations, i.e., M <= N, and M is normally much smaller than N,
    i.e., M << N.
    :param truncation_level: A positive integer specifying the truncation level, referred to as T, for the truncated
    stick-breaking representation to approximate the DP. T must be less than or equal to the number of observed
    dimensions, i.e., T <= D, and T is normally much smaller than D, i.e., T << D.
    :param alpha_prior_params: The parameters for the Gamma prior on alpha, which is the scaling parameter for the DP.
    :param mask_size: A mask size for grouping adjacent observed dimensions to the same group assignment. Default is 1
    so each dimension is not forced to share a group with its neighbor(s). The mask size must be an integer divisor of
    the total number of observed dimensions, i.e., k * mask_size = D for some positive integer k.
    :return: An instance of the DP_GP_LVM class.
    """

    # Determine tensor dimensions.
    num_samples, num_dimensions = np.shape(y_train)  # y_train provided as numpy array.
    # num_samples, num_dimensions = y_train.get_shape().as_list()
    # TODO: Change to less than; not less than or equal. Only set to equal for unittesting.
    assert 0 < num_latent_dims <= num_dimensions, \
        'Number of latent dimensions must be postive and less than the dimensionality of the observed data.'
    assert 0 < num_inducing_points <= num_samples, \
        'Number of inducing points must be positive and less than the number of observations in the observed data.'
    assert 0 < truncation_level <= min(num_samples, num_dimensions), \
        'The truncation level must be positive and less than the dimensionality of the observed data and ' \
        'less than the number of observations.'

    # Fit latent means using PCA and the inducing inputs as a subset of those with a little noise.
    x_init = pca(y_train, num_latent_dimensions=num_latent_dims)
    x_mean = tf.Variable(x_init, dtype=TF_DTYPE, trainable=True)  # [N x Q].
    # [N x Q x Q].
    # TODO: Maybe add some noise (see noisy init val in types.py.
    # TODO: Maybe try with 0.5 initial value as recommended in BGP-LVM paper.
    x_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                      shape=(num_samples, num_latent_dims),
                                                      is_trainable=True))

    # Initialise inducing inputs as a subset of the PCA values calculated for latent means plus some noise.
    x_u_init = np.random.permutation(x_init)[:num_inducing_points] + \
        np.random.normal(loc=0.0, scale=0.01, size=(num_inducing_points, num_latent_dims))
    x_u = tf.Variable(x_u_init, dtype=TF_DTYPE, trainable=True)  # [M x Q].

    # Define DP that handles the assignments.
    dp_model = dirichlet_process(num_samples=num_dimensions,
                                 alpha_prior_params=alpha_prior_params,
                                 truncation_level=truncation_level,
                                 mask_size=mask_size)

    # Define kernel hyperparameters from DP.
    # [T x Q]. Do not worry about adding noise as the actual gamma is random due to initialisation of phi.
    gamma_atoms = create_positive_variable(initial_value=GP_INIT_GAMMA,
                                           shape=(truncation_level, num_latent_dims),
                                           is_trainable=True)
    # Both are [T x 1]. Do not worry about adding noise as the actual sig_var and beta is random due to initialisation
    # of phi.
    sig_var_atoms = create_positive_variable(initial_value=GP_INIT_ALPHA,
                                             shape=(truncation_level, 1),
                                             is_trainable=True)
    beta_atoms = create_positive_variable(initial_value=GP_INIT_BETA,
                                          shape=(truncation_level, 1),
                                          is_trainable=True)

    kernel_hyperpriors_log_likelihood = tf.reduce_sum(log_normal_log_pdf(gamma_atoms)) + \
        tf.reduce_sum(log_normal_log_pdf(sig_var_atoms)) + \
        tf.reduce_sum(log_normal_log_pdf(beta_atoms))

    gamma = tf.matmul(dp_model.assignments, gamma_atoms)  # [D x Q] as [D x T] x [T x Q].
    sig_var = tf.matmul(dp_model.assignments, sig_var_atoms)  # [D x 1] as [D x T] x [T x 1].
    beta = tf.matmul(dp_model.assignments, beta_atoms)  # [D x 1] as [D x T] x [T x 1].

    # Define kernel.
    kernel = k_ard_rbf(gamma=gamma, alpha=sig_var, beta=beta)  # Therefore, batch size is now D.

    # Define psi-statistics.
    psi_0 = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [D x 1].
    psi_1 = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [D x N x M].
    psi_2 = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [D x M x M].

    # Calculate f_hat term from the evidence lower bound (ELBO). Using stable calculation for f_hat.
    beta_d11 = tf.expand_dims(beta, axis=-1)  # [D x 1 x 1].

    k_uu = kernel.covariance_matrix(input_0=x_u, input_1=None, include_noise=False, include_jitter=True)  # [D x M x M].
    l_uu = tf.cholesky(k_uu)  # [D x M x M].

    l_uu_inv_psi_2 = tf.matrix_triangular_solve(l_uu, psi_2, lower=True)  # [D x M x M].
    l_uu_inv_psi_2_inv_transpose = tf.transpose(
        tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2, perm=[0, 2, 1]), lower=True),
        perm=[0, 2, 1])  # [D x M x M].

    # [D x M x M].
    a = beta_d11 * l_uu_inv_psi_2_inv_transpose + tf.eye(num_inducing_points,
                                                         batch_shape=[num_dimensions],
                                                         dtype=TF_DTYPE)
    l_a = tf.cholesky(a)  # [D x M x M].

    log_det_l_a = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a)))  # Scalar.

    # [D x M x N].
    l_uu_inv_psi_1_transpose = tf.matrix_triangular_solve(l_uu, tf.transpose(psi_1, perm=[0, 2, 1]), lower=True)
    c = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_transpose, lower=True)  # [D x M x N].
    c_transpose_c = tf.matmul(c, c, transpose_a=True)  # [D x N x N].

    y_beta_d1n = tf.expand_dims(tf.transpose(y_train) * beta, axis=1)  # [D x 1 x N].

    f_hat = 0.5 * num_samples * (tf.reduce_sum(tf.log(beta)) - num_dimensions * np.log(2.0 * np.pi)) - \
        log_det_l_a + \
        0.5 * tf.reduce_sum(beta * (tf.reduce_sum(tf.matrix_diag_part(l_uu_inv_psi_2_inv_transpose),
                                                  axis=-1,
                                                  keepdims=True) - psi_0)) - \
        0.5 * tf.reduce_sum(beta * tf.expand_dims(tf.diag_part(tf.matmul(y_train, y_train, transpose_a=True)),
                                                  axis=-1)) + \
        0.5 * tf.reduce_sum(tf.matmul(tf.matmul(y_beta_d1n, c_transpose_c), y_beta_d1n, transpose_b=True))

    # Define KL divergence between q(X) and p(X).
    kl_q_x_p_x = calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar)

    # Define evidence lower bound (ELBO).
    gp_elbo = f_hat - kl_q_x_p_x

    # Define objective function.
    objective = dp_model.objective - gp_elbo - kernel_hyperpriors_log_likelihood

    class DP_GP_LVM(Trainable):
        """
        This class defines a DP_GP_LVM object.
        """

        @property
        def assignments(self):
            """
            TODO
            :return:
            """
            return dp_model.assignments

        @property
        def dp(self):
            """
            TODO
            :return:
            """
            return dp_model

        @property
        def dp_atoms(self):
            """
            TODO
            :return:
            """
            return gamma_atoms, sig_var_atoms, beta_atoms

        @property
        def kernel(self):
            """
            TODO
            :return:
            """
            return kernel

        @property
        def ard_weights(self):
            """
            TODO
            :return:
            """
            return kernel.hyperparameters[KernelHyperparameters.ARD_WEIGHTS]

        @property
        def signal_variance(self):
            """
            TODO
            :return:
            """
            return kernel.hyperparameters[KernelHyperparameters.SIGNAL_VARIANCE]

        @property
        def noise_precision(self):
            """
            TODO
            :return:
            """
            return kernel.noise_precision

        @property
        def inducing_input(self):
            """
            TODO
            :return:
            """
            return x_u

        @property
        def q_x(self):
            """
            TODO
            :return:
            """
            return x_mean, x_covar

        @staticmethod
        def predict_new_latent_variables(y_test, use_pca=False):
            """
            TODO
            :param y_test:
            :param use_pca:
            :return:
            """
            # y_test is [N* x D].
            num_test_points, test_dims = np.shape(y_test)
            assert test_dims == num_dimensions, \
                'Observed dimensionality for prediction must be equal to the dimensionality of the training data.'

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(y_test, num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(y_train, y_test)
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Define extra f_hat terms for y_test.
            psi_0_test = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x M x M].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [D x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [D x M x M].

            a_test = beta_d11 * l_uu_inv_psi_2_test_inv_transpose + \
                tf.eye(num_inducing_points, batch_shape=[num_dimensions], dtype=TF_DTYPE)  # [D x M x M].
            l_a_test = tf.cholesky(a_test)  # [D x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            # [D x M x N*].
            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)
            c_test = tf.matrix_triangular_solve(l_a_test, l_uu_inv_psi_1_test_transpose, lower=True)  # [D x M x N*].
            c_transpose_c_test = tf.matmul(c_test, c_test, transpose_a=True)  # [D x N* x N*].

            y_test_beta_d1n = tf.expand_dims(tf.transpose(y_test), axis=1) * beta_d11  # [D x 1 x N*].

            f_hat_test = 0.5 * num_test_points * (tf.reduce_sum(tf.log(beta)) -
                                                  num_dimensions * np.log(2.0 * np.pi)) - \
                log_det_l_a_test + \
                0.5 * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_test_inv_transpose) - psi_0_test)) - \
                0.5 * tf.reduce_sum(beta * tf.expand_dims(tf.diag_part(tf.matmul(y_test, y_test, transpose_a=True)),
                                                          axis=-1)) + \
                0.5 * tf.reduce_sum(tf.matmul(tf.matmul(y_test_beta_d1n, c_transpose_c_test),
                                              y_test_beta_d1n, transpose_b=True))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define prediction lower bound.
            prediction_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            # Define test log-likelihood (from equation 36 of BGP-LVM journal paper).
            test_log_likelihood = f_hat_test - kl_q_x_test_p_x_test

            return prediction_lower_bound, x_test_mean, x_test_covar, test_log_likelihood

        @staticmethod
        def predict_missing_data(y_test, use_pca=False):
            """
            TODO
            :param y_test:
            :param use_pca:
            :return:
            """
            # TODO: Currently assume y_test is first observed_dims of d. Update to be more generic.
            num_test_points, num_observed_dims = np.shape(y_test)
            assert num_observed_dims < num_dimensions, \
                'Observed dimensionality for missing data scenario must be less than total ' \
                'dimensionality of training data.'

            # Get slice of y_train for remaining unobserved dimensions. Du = D - Do.
            # Slice y_train into observed and unobserved dimensions for the missing data. D = Do + Du
            y_train_observed = tf.slice(y_train, begin=[0, 0], size=[-1, num_observed_dims])  # [N x Do]
            y_train_unobserved = tf.slice(y_train, begin=[0, num_observed_dims], size=[-1, -1])  # [N x Du]

            # Slice beta into observed and unobserved dimensions for the missing data.
            beta_observed = tf.slice(beta, begin=[0, 0], size=[num_observed_dims, -1])  # [Do x 1].
            beta_unobserved = tf.slice(beta, begin=[num_observed_dims, 0], size=[-1, -1])  # [Du x 1].

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(y_test, num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(y_train_observed, y_test)
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))  # [N* x Q].

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Calculate some other intermediate values for prediction.
            l_uu_inv = tf.matrix_triangular_solve(l_uu,
                                                  tf.eye(num_inducing_points, batch_shape=[num_dimensions],
                                                         dtype=TF_DTYPE))  # [D x M x M].
            l_a_inv = tf.matrix_triangular_solve(l_a,
                                                 tf.eye(num_inducing_points, batch_shape=[num_dimensions],
                                                        dtype=TF_DTYPE))  # [D x M x M].

            # Define extra f_hat terms for y_test.
            psi_0_test = kernel.psi_0(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x M x M].

            # Slice into observed and unobserved dimensions.
            psi_0_test_observed = tf.slice(psi_0_test,
                                           begin=[0, 0],
                                           size=[num_observed_dims, -1])  # [Do x 1].
            psi_0_test_unobserved = tf.slice(psi_0_test,
                                             begin=[num_observed_dims, 0],
                                             size=[-1, -1])  # [Du x 1].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [D x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [D x M x M].

            # Slice so only looking at observed dimensions.
            l_uu_inv_psi_2_test_inv_transpose_observed = tf.slice(l_uu_inv_psi_2_test_inv_transpose,
                                                                  begin=[0, 0, 0],
                                                                  size=[num_observed_dims, -1, -1])  # [Do x M x M].

            a_test = tf.expand_dims(beta_observed, axis=-1) * l_uu_inv_psi_2_test_inv_transpose_observed + \
                tf.eye(num_inducing_points, batch_shape=[num_observed_dims], dtype=TF_DTYPE)  # [Do x M x M].
            l_a_test = tf.cholesky(a_test)  # [Do x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)  # [D x M x N*].
            # Slice so only looking at observed dimensions.
            l_uu_inv_psi_1_test_transpose_observed = tf.slice(l_uu_inv_psi_1_test_transpose,
                                                              begin=[0, 0, 0],
                                                              size=[num_observed_dims, -1, -1])  # [Do x M x N*].

            c_test = tf.matrix_triangular_solve(l_a_test,
                                                l_uu_inv_psi_1_test_transpose_observed,
                                                lower=True)  # [Do x M x N*].
            c_transpose_c_test = tf.matmul(c_test, c_test, transpose_a=True)  # [Do x N* x N*].

            y_test_beta_do1n = tf.expand_dims(tf.transpose(y_test) * beta_observed, axis=1)  # [Do x 1 x N*].

            f_hat_test = 0.5 * num_test_points * (tf.reduce_sum(tf.log(beta_observed)) -
                                                  num_observed_dims * np.log(2.0 * np.pi)) - \
                log_det_l_a_test + \
                0.5 * tf.reduce_sum(beta_observed * (tf.trace(l_uu_inv_psi_2_test_inv_transpose_observed) -
                                                     psi_0_test_observed)) - \
                0.5 * tf.reduce_sum(beta_observed * tf.expand_dims(tf.diag_part(tf.matmul(y_test,
                                                                                          y_test,
                                                                                          transpose_a=True)),
                                                                   axis=-1)) + \
                0.5 * tf.reduce_sum(tf.matmul(tf.matmul(y_test_beta_do1n,
                                                        c_transpose_c_test),
                                              y_test_beta_do1n,
                                              transpose_b=True))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define missing data lower bound.
            missing_data_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            # Define extra terms for predicted mean.
            c_predict = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_test_transpose, lower=True)  # [D x M x N*].
            c_predict_c = tf.matmul(c_predict, c, transpose_a=True)  # [D x N* x N].

            # [Du x N* x N].
            c_predict_c_unobserved = tf.slice(c_predict_c, begin=[num_observed_dims, 0, 0], size=[-1, -1, -1])
            y_train_unobserved_du_n1 = tf.expand_dims(tf.transpose(y_train_unobserved), axis=-1)  # [Du x N x 1].

            # [N* x Du].
            predicted_mean = tf.transpose(beta_unobserved *
                                          tf.squeeze(tf.matmul(c_predict_c_unobserved,
                                                               y_train_unobserved_du_n1),
                                                     axis=-1))

            # Define extra terms for predicted covariance.
            g = psi_2_test - tf.matmul(psi_1_test, psi_1_test, transpose_a=True)  # [D x M x M].
            g_unobserved = tf.slice(g, begin=[num_observed_dims, 0, 0], size=[-1, -1, -1])  # [Du x M x M].

            scale_du = tf.matmul(
                tf.matmul(
                    tf.matmul(
                        l_uu_inv,
                        tf.matmul(
                            l_a_inv,
                            l_a_inv,
                            transpose_a=True
                        ),
                        transpose_a=True
                    ),
                    l_uu_inv
                ),
                psi_1,
                transpose_b=True
            )  # [D x M x N].

            # Slice so only looking at unobserved dimensions.
            scale_du_unobserved = tf.slice(scale_du,
                                           begin=[num_observed_dims, 0, 0],
                                           size=[-1, -1, -1])  # [Du x M x N].

            scale_du_yu = tf.matmul(scale_du_unobserved, y_train_unobserved_du_n1)  # [Du x M x 1].

            # [Du x 1].
            yu_variance = tf.square(beta_unobserved) * tf.squeeze(tf.matmul(scale_du_yu,
                                                                            tf.matmul(g_unobserved, scale_du_yu),
                                                                            transpose_a=True),
                                                                  axis=-1)

            # TODO: Make this calculation more efficient by ignoring the observed dimensions.
            trace_term = tf.expand_dims(
                tf.trace(
                    tf.matmul(
                        tf.matmul(
                            tf.matmul(
                                l_uu_inv,
                                tf.eye(num_inducing_points, batch_shape=[num_dimensions], dtype=TF_DTYPE) -
                                tf.matmul(l_a_inv, l_a_inv, transpose_a=True),
                                transpose_a=True
                            ),
                            l_uu_inv
                        ),
                        psi_2_test
                    )
                ),
                axis=-1)  # [D x 1].
            trace_term_unobserved = tf.slice(trace_term, begin=[num_observed_dims, 0], size=[-1, -1])  # [Du x 1].

            # [Du x N* x N*] so there is a specific covariance for each unobserved dimension.
            predicted_covar = tf.expand_dims(yu_variance, axis=-1) + \
                tf.expand_dims(psi_0_test_unobserved + tf.reciprocal(beta_unobserved) + trace_term_unobserved,
                               axis=-1) * \
                tf.eye(num_test_points, batch_shape=[num_dimensions - num_observed_dims], dtype=TF_DTYPE)

            return missing_data_lower_bound, x_test_mean, x_test_covar, predicted_mean, predicted_covar

        @property
        def objective(self):
            """
            TODO
            :return:
            """
            return objective

    return DP_GP_LVM()


def dp_gp_lvm_t(y_train,
                num_latent_dims=GP_LVM_DEFAULT_LATENT_DIMENSIONS,
                num_inducing_points=GP_LVM_DEFAULT_NUM_INDUCING_POINTS,
                truncation_level=DP_DEFAULT_TRUNCATION_LEVEL,
                alpha_prior_params=DP_DEFAULT_ALPHA_PRIOR_PARAMS,
                mask_size=1,
                seed=0):
    """
    This function initialises a DP_GP_LVM object.
    :param y_train: The observed data, which is assumed to be normalised with zero mean and unit variance for each
    column. Must be provided as [N x D] numpy array, where N is the number of observations and D is the number of
    observed dimensions.
    :param num_latent_dims: A positive integer specifying the number of latent dimensions, referred to as Q. Q must be
    less than or equal to the minimum of the number of observations and the number of observed dimensions, i.e.,
    Q <= min(N,D), and Q is normally much smaller than D, i.e., Q << D.
    :param num_inducing_points: A positive integer specifying the number of inducing inputs, referred to as M, for the
    GP. M must be less than or equal to the number of observations, i.e., M <= N, and M is normally much smaller than N,
    i.e., M << N.
    :param truncation_level: A positive integer specifying the truncation level, referred to as T, for the truncated
    stick-breaking representation to approximate the DP. T must be less than or equal to the number of observed
    dimensions, i.e., T <= D, and T is normally much smaller than D, i.e., T << D.
    :param alpha_prior_params: The parameters for the Gamma prior on alpha, which is the scaling parameter for the DP.
    :param mask_size: A mask size for grouping adjacent observed dimensions to the same group assignment. Default is 1
    so each dimension is not forced to share a group with its neighbor(s). The mask size must be an integer divisor of
    the total number of observed dimensions, i.e., k * mask_size = D for some positive integer k.
    :param seed: The seed value for the random number generator in numpy. This is used to ensure repeatable results.
    Default is 0.
    :return: An instance of the DP_GP_LVM class.
    """

    # Determine tensor dimensions and validate input.
    assert isinstance(y_train, np.ndarray), 'Training data must be provided as a numpy array.'
    num_samples, num_dimensions = np.shape(y_train)  # y_train provided as numpy array.
    assert isinstance(num_latent_dims, int), 'Number of latent dimensions must be an integer.'
    assert 0 < num_latent_dims < num_dimensions, \
        'Number of latent dimensions must be postive and less than the dimensionality of the observed data.'
    assert isinstance(num_inducing_points, int), 'Number of inducing points must be an integer.'
    assert 0 < num_inducing_points <= num_samples, \
        'Number of inducing points must be positive and less than or equal to the number of observations in the ' \
        'observed data.'
    assert isinstance(truncation_level, int), 'The truncation level must be an integer.'
    assert 0 < truncation_level <= min(num_samples, num_dimensions), \
        'The truncation level must be positive and less than or equal to the dimensionality of the observed data and ' \
        'less than or equal to the number of observations.'

    # Set random seed value.
    assert isinstance(seed, int) and seed >= 0, 'Seed must be a 32-bit unsigned integer, i.e., 0 <= seed <= 2^32 - 1.'
    np.random.seed(seed=seed)

    # Fit latent means using PCA and the inducing inputs as a subset of those with a little noise.
    x_init = pca(y_train, num_latent_dimensions=num_latent_dims)
    x_mean = tf.Variable(x_init, dtype=TF_DTYPE, trainable=True)  # [N x Q].
    # [N x Q x Q].
    # TODO: Maybe add some noise (see noisy init val in types.py.
    # TODO: Maybe try with 0.5 initial value as recommended in BGP-LVM paper.
    x_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                      shape=(num_samples, num_latent_dims),
                                                      is_trainable=True))

    # Initialise inducing inputs as a subset of the PCA values calculated for latent means plus some noise.
    x_u_init = np.random.permutation(x_init)[:num_inducing_points] + \
        np.random.normal(loc=0.0, scale=0.01, size=(num_inducing_points, num_latent_dims))
    x_u = tf.Variable(x_u_init, dtype=TF_DTYPE, trainable=True)  # [M x Q].

    # Define DP that handles the assignments.
    dp_model = dirichlet_process(num_samples=num_dimensions,
                                 alpha_prior_params=alpha_prior_params,
                                 truncation_level=truncation_level,
                                 mask_size=mask_size)
    phi_td = tf.transpose(dp_model.assignments)  # [T x D].

    # TODO
    '''
    Handle masking here as we don't want to overcount q(Z). Or fix implementation in DP as we only want to count number
    of free parameters in phi, not the full [N x T] phi. Only want [N/mask_size x T] to contribute to lower bound.
    '''

    # Define kernel hyperparameters from DP.
    # [T x Q]. TODO: Maybe add some noise so do not initialise to same value.
    gamma_atoms = create_positive_variable(initial_value=GP_INIT_GAMMA,
                                           shape=(truncation_level, num_latent_dims),
                                           is_trainable=True)
    # Both are [T x 1]. TODO: Maybe add some noise so do not initialise to same value.
    sig_var_atoms = create_positive_variable(initial_value=GP_INIT_ALPHA,
                                             shape=(truncation_level, 1),
                                             is_trainable=True)
    beta_atoms = create_positive_variable(initial_value=GP_INIT_BETA,
                                          shape=(truncation_level, 1),
                                          is_trainable=True)

    kernel_hyperpriors_log_likelihood = tf.reduce_sum(log_normal_log_pdf(gamma_atoms)) + \
        tf.reduce_sum(log_normal_log_pdf(sig_var_atoms)) + \
        tf.reduce_sum(log_normal_log_pdf(beta_atoms))

    # Define kernel.
    kernel = k_ard_rbf(gamma=gamma_atoms, alpha=sig_var_atoms, beta=beta_atoms)  # Therefore, batch size is now T.

    # Define psi-statistics.
    psi_0 = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [T x 1].
    psi_1 = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [T x N x M].
    psi_2 = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [T x M x M].

    # Calculate f_hat term from the evidence lower bound (ELBO). Using stable calculation for f_hat.
    beta_t11 = tf.expand_dims(beta_atoms, axis=-1)  # [T x 1 x 1].

    k_uu = kernel.covariance_matrix(input_0=x_u, input_1=None, include_noise=False, include_jitter=True)  # [T x M x M].
    l_uu = tf.cholesky(k_uu)  # [T x M x M].

    l_uu_inv_psi_2 = tf.matrix_triangular_solve(l_uu, psi_2, lower=True)  # [T x M x M].
    l_uu_inv_psi_2_inv_transpose = tf.transpose(
        tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2, perm=[0, 2, 1]), lower=True),
        perm=[0, 2, 1])  # [T x M x M].
    trace_l_uu_inv_psi_2_inv_transpose = tf.reduce_sum(tf.matrix_diag_part(l_uu_inv_psi_2_inv_transpose),
                                                       axis=-1,
                                                       keepdims=True)  # [T x 1].

    # [T x M x M].
    a = beta_t11 * l_uu_inv_psi_2_inv_transpose + tf.eye(num_inducing_points,
                                                         batch_shape=[truncation_level],
                                                         dtype=TF_DTYPE)
    l_a = tf.cholesky(a)  # [T x M x M].

    log_det_l_a = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a)), axis=1, keepdims=True)  # [T x 1].

    # [T x M x N].
    l_uu_inv_psi_1_transpose = tf.matrix_triangular_solve(l_uu, tf.transpose(psi_1, perm=[0, 2, 1]), lower=True)
    c = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_transpose, lower=True)  # [T x M x N].
    g = beta_t11 * c  # [T x M x N].
    g_transpose_g = tf.matmul(g, g, transpose_a=True)  # [T x N x N].

    # Slice method:
    # # T-length list of [N x N] tensors.
    # gg_slices = [tf.squeeze(tf.slice(g_transpose_g, [t, 0, 0], [1, -1, -1]), axis=[0]) for t in range(truncation_level)]
    # # D-length list of [N x 1] tensors.
    # y_slices = [tf.slice(y_train, [0, d], [-1, 1]) for d in range(num_dimensions)]
    #
    # reconstruction_term = tf.reduce_sum([[tf.slice(phi_td, [t, d], [1, 1]) *
    #                                       tf.matmul(tf.matmul(y_slices[d],
    #                                                           gg_slices[t],
    #                                                           transpose_a=True),
    #                                                 y_slices[d])
    #                                       for d in range(num_dimensions)] for t in range(truncation_level)])

    # Tile method:
    y_tiled = tf.tile(tf.expand_dims(y_train, axis=0), [truncation_level, 1, 1])  # [T x N x D].
    reconstruction_term = tf.reduce_sum(phi_td * tf.reduce_sum(tf.square(tf.matmul(g, y_tiled)), axis=1))

    y_squared_1d = tf.reduce_sum(tf.square(y_train), axis=0, keepdims=True)  # [1 x D].

    f_hat = -0.5 * num_samples * num_dimensions * np.log(2.0 * np.pi) + \
        tf.reduce_sum(phi_td * (0.5 * (num_samples * tf.log(beta_atoms) +
                                       beta_atoms * (trace_l_uu_inv_psi_2_inv_transpose - psi_0)) -
                                log_det_l_a)) - \
        0.5 * tf.reduce_sum(phi_td * beta_atoms * y_squared_1d) + \
        0.5 * reconstruction_term

    # Define KL divergence between q(X) and p(X).
    kl_q_x_p_x = calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar)

    # Define evidence lower bound (ELBO).
    gp_elbo = f_hat - kl_q_x_p_x

    # Define objective function.
    objective = dp_model.objective - gp_elbo - kernel_hyperpriors_log_likelihood

    class DP_GP_LVM(Trainable):
        """
        This class defines a DP_GP_LVM object.
        """

        @property
        def assignments(self):
            """
            TODO
            :return:
            """
            return dp_model.assignments

        @property
        def dp(self):
            """
            TODO
            :return:
            """
            return dp_model

        @property
        def dp_atoms(self):
            """
            TODO
            :return:
            """
            return gamma_atoms, sig_var_atoms, beta_atoms

        @property
        def kernel(self):
            """
            TODO
            :return:
            """
            return kernel

        @property
        def ard_weights(self):
            """
            TODO
            :return:
            """
            return kernel.hyperparameters[KernelHyperparameters.ARD_WEIGHTS]

        @property
        def signal_variance(self):
            """
            TODO
            :return:
            """
            return kernel.hyperparameters[KernelHyperparameters.SIGNAL_VARIANCE]

        @property
        def noise_precision(self):
            """
            TODO
            :return:
            """
            return kernel.noise_precision

        @property
        def inducing_input(self):
            """
            TODO
            :return:
            """
            return x_u

        @property
        def q_x(self):
            """
            TODO
            :return:
            """
            return x_mean, x_covar

        @staticmethod
        def predict_new_latent_variables(y_test, use_pca=False):
            """
            TODO
            :param y_test:
            :param use_pca:
            :return:
            """
            # y_test is [N* x D].
            num_test_points, test_dims = np.shape(y_test)
            assert test_dims == num_dimensions, \
                'Observed dimensionality for prediction must be equal to the dimensionality of the training data.'

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(y_test, num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(y_train, y_test)
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Define extra f_hat terms for y_test.
            psi_0_test = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x M x M].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [D x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [D x M x M].

            a_test = beta_d11 * l_uu_inv_psi_2_test_inv_transpose + \
                tf.eye(num_inducing_points, batch_shape=[num_dimensions], dtype=TF_DTYPE)  # [D x M x M].
            l_a_test = tf.cholesky(a_test)  # [D x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            # [D x M x N*].
            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)
            c_test = tf.matrix_triangular_solve(l_a_test, l_uu_inv_psi_1_test_transpose, lower=True)  # [D x M x N*].
            c_transpose_c_test = tf.matmul(c_test, c_test, transpose_a=True)  # [D x N* x N*].

            y_test_beta_d1n = tf.expand_dims(tf.transpose(y_test), axis=1) * beta_d11  # [D x 1 x N*].

            f_hat_test = 0.5 * num_test_points * (tf.reduce_sum(tf.log(beta)) -
                                                  num_dimensions * np.log(2.0 * np.pi)) - \
                log_det_l_a_test + \
                0.5 * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_test_inv_transpose) - psi_0_test)) - \
                0.5 * tf.reduce_sum(beta * tf.expand_dims(tf.diag_part(tf.matmul(y_test, y_test, transpose_a=True)),
                                                          axis=-1)) + \
                0.5 * tf.reduce_sum(tf.matmul(tf.matmul(y_test_beta_d1n, c_transpose_c_test),
                                              y_test_beta_d1n, transpose_b=True))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define prediction lower bound.
            prediction_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            return prediction_lower_bound, x_test_mean, x_test_covar

        @staticmethod
        def predict_missing_data(y_test, use_pca=False):
            """
            TODO
            :param y_test:
            :param use_pca:
            :return:
            """
            # TODO: Currently assume y_test is first observed_dims of d. Update to be more generic.
            num_test_points, num_observed_dims = np.shape(y_test)
            assert num_observed_dims < num_dimensions, \
                'Observed dimensionality for missing data scenario must be less than total ' \
                'dimensionality of training data.'

            # Get slice of y_train for remaining unobserved dimensions. Du = D - Do.
            # Slice y_train into observed and unobserved dimensions for the missing data. D = Do + Du
            y_train_observed = tf.slice(y_train, begin=[0, 0], size=[-1, num_observed_dims])  # [N x Do]
            y_train_unobserved = tf.slice(y_train, begin=[0, num_observed_dims], size=[-1, -1])  # [N x Du]

            # Slice beta into observed and unobserved dimensions for the missing data.
            beta_observed = tf.slice(beta, begin=[0, 0], size=[num_observed_dims, -1])  # [Do x 1].
            beta_unobserved = tf.slice(beta, begin=[num_observed_dims, 0], size=[-1, -1])  # [Du x 1].

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(y_test, num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(y_train_observed, y_test)
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))  # [N* x Q].

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Calculate some other intermediate values for prediction.
            l_uu_inv = tf.matrix_triangular_solve(l_uu,
                                                  tf.eye(num_inducing_points, batch_shape=[num_dimensions],
                                                         dtype=TF_DTYPE))  # [D x M x M].
            l_a_inv = tf.matrix_triangular_solve(l_a,
                                                 tf.eye(num_inducing_points, batch_shape=[num_dimensions],
                                                        dtype=TF_DTYPE))  # [D x M x M].

            # Define extra f_hat terms for y_test.
            psi_0_test = kernel.psi_0(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x M x M].

            # Slice into observed and unobserved dimensions.
            psi_0_test_observed = tf.slice(psi_0_test,
                                           begin=[0, 0],
                                           size=[num_observed_dims, -1])  # [Do x 1].
            psi_0_test_unobserved = tf.slice(psi_0_test,
                                             begin=[num_observed_dims, 0],
                                             size=[-1, -1])  # [Du x 1].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [D x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [D x M x M].

            # Slice so only looking at observed dimensions.
            l_uu_inv_psi_2_test_inv_transpose_observed = tf.slice(l_uu_inv_psi_2_test_inv_transpose,
                                                                  begin=[0, 0, 0],
                                                                  size=[num_observed_dims, -1, -1])  # [Do x M x M].

            a_test = tf.expand_dims(beta_observed, axis=-1) * l_uu_inv_psi_2_test_inv_transpose_observed + \
                tf.eye(num_inducing_points, batch_shape=[num_observed_dims], dtype=TF_DTYPE)  # [Do x M x M].
            l_a_test = tf.cholesky(a_test)  # [Do x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)  # [D x M x N*].
            # Slice so only looking at observed dimensions.
            l_uu_inv_psi_1_test_transpose_observed = tf.slice(l_uu_inv_psi_1_test_transpose,
                                                              begin=[0, 0, 0],
                                                              size=[num_observed_dims, -1, -1])  # [Do x M x N*].

            c_test = tf.matrix_triangular_solve(l_a_test,
                                                l_uu_inv_psi_1_test_transpose_observed,
                                                lower=True)  # [Do x M x N*].
            c_transpose_c_test = tf.matmul(c_test, c_test, transpose_a=True)  # [Do x N* x N*].

            y_test_beta_do1n = tf.expand_dims(tf.transpose(y_test) * beta_observed, axis=1)  # [Do x 1 x N*].

            f_hat_test = 0.5 * num_test_points * (tf.reduce_sum(tf.log(beta_observed)) -
                                                  num_observed_dims * np.log(2.0 * np.pi)) - \
                log_det_l_a_test + \
                0.5 * tf.reduce_sum(beta_observed * (tf.trace(l_uu_inv_psi_2_test_inv_transpose_observed) -
                                                     psi_0_test_observed)) - \
                0.5 * tf.reduce_sum(beta_observed * tf.expand_dims(tf.diag_part(tf.matmul(y_test,
                                                                                          y_test,
                                                                                          transpose_a=True)),
                                                                   axis=-1)) + \
                0.5 * tf.reduce_sum(tf.matmul(tf.matmul(y_test_beta_do1n,
                                                        c_transpose_c_test),
                                              y_test_beta_do1n,
                                              transpose_b=True))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define missing data lower bound.
            missing_data_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            # Define extra terms for predicted mean.
            c_predict = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_test_transpose, lower=True)  # [D x M x N*].
            c_predict_c = tf.matmul(c_predict, c, transpose_a=True)  # [D x N* x N].

            # [Du x N* x N].
            c_predict_c_unobserved = tf.slice(c_predict_c, begin=[num_observed_dims, 0, 0], size=[-1, -1, -1])
            y_train_unobserved_du_n1 = tf.expand_dims(tf.transpose(y_train_unobserved), axis=-1)  # [Du x N x 1].

            # [N* x Du].
            predicted_mean = tf.transpose(beta_unobserved *
                                          tf.squeeze(tf.matmul(c_predict_c_unobserved,
                                                               y_train_unobserved_du_n1),
                                                     axis=-1))

            # Define extra terms for predicted covariance.
            g = psi_2_test - tf.matmul(psi_1_test, psi_1_test, transpose_a=True)  # [D x M x M].
            g_unobserved = tf.slice(g, begin=[num_observed_dims, 0, 0], size=[-1, -1, -1])  # [Du x M x M].

            scale_du = tf.matmul(
                tf.matmul(
                    tf.matmul(
                        l_uu_inv,
                        tf.matmul(
                            l_a_inv,
                            l_a_inv,
                            transpose_a=True
                        ),
                        transpose_a=True
                    ),
                    l_uu_inv
                ),
                psi_1,
                transpose_b=True
            )  # [D x M x N].

            # Slice so only looking at unobserved dimensions.
            scale_du_unobserved = tf.slice(scale_du,
                                           begin=[num_observed_dims, 0, 0],
                                           size=[-1, -1, -1])  # [Du x M x N].

            scale_du_yu = tf.matmul(scale_du_unobserved, y_train_unobserved_du_n1)  # [Du x M x 1].

            # [Du x 1].
            yu_variance = tf.square(beta_unobserved) * tf.squeeze(tf.matmul(scale_du_yu,
                                                                            tf.matmul(g_unobserved, scale_du_yu),
                                                                            transpose_a=True),
                                                                  axis=-1)

            # TODO: Make this calculation more efficient by ignoring the observed dimensions.
            trace_term = tf.expand_dims(
                tf.trace(
                    tf.matmul(
                        tf.matmul(
                            tf.matmul(
                                l_uu_inv,
                                tf.eye(num_inducing_points, batch_shape=[num_dimensions], dtype=TF_DTYPE) -
                                tf.matmul(l_a_inv, l_a_inv, transpose_a=True),
                                transpose_a=True
                            ),
                            l_uu_inv
                        ),
                        psi_2_test
                    )
                ),
                axis=-1)  # [D x 1].
            trace_term_unobserved = tf.slice(trace_term, begin=[num_observed_dims, 0], size=[-1, -1])  # [Du x 1].

            # [Du x N* x N*] so there is a specific covariance for each unobserved dimension.
            predicted_covar = tf.expand_dims(yu_variance, axis=-1) + \
                tf.expand_dims(psi_0_test_unobserved + tf.reciprocal(beta_unobserved) + trace_term_unobserved,
                               axis=-1) * \
                tf.eye(num_test_points, batch_shape=[num_dimensions - num_observed_dims], dtype=TF_DTYPE)

            return missing_data_lower_bound, x_test_mean, x_test_covar, predicted_mean, predicted_covar

        @property
        def objective(self):
            """
            TODO
            :return:
            """
            return objective

    return DP_GP_LVM()


'''
def dp_gp_lvm_svi(y_train,
                  num_latent_dims=GP_LVM_DEFAULT_LATENT_DIMENSIONS,
                  num_inducing_points=GP_LVM_DEFAULT_NUM_INDUCING_POINTS,
                  truncation_level=DP_DEFAULT_TRUNCATION_LEVEL,
                  alpha_prior_params=DP_DEFAULT_ALPHA_PRIOR_PARAMS,
                  num_latent_samples=1,
                  mask_size=1,
                  seed=0):
    """
    This function initialises a DP_GP_LVM object that estimates the psi_statistics with sampling. The model is optimized
    with stochastic variational inference.
    :param y_train: The observed data, which is assumed to be normalised with zero mean and unit variance for each
    column. Must be provided as [N x D] numpy array, where N is the number of observations and D is the number of
    observed dimensions.
    :param num_latent_dims: A positive integer specifying the number of latent dimensions, referred to as Q. Q must be
    less than or equal to the minimum of the number of observations and the number of observed dimensions, i.e.,
    Q <= min(N,D), and Q is normally much smaller than D, i.e., Q << D.
    :param num_inducing_points: A positive integer specifying the number of inducing inputs, referred to as M, for the
    GP. M must be less than or equal to the number of observations, i.e., M <= N, and M is normally much smaller than N,
    i.e., M << N.
    :param truncation_level: A positive integer specifying the truncation level, referred to as T, for the truncated
    stick-breaking representation to approximate the DP. T must be less than or equal to the number of observed
    dimensions, i.e., T <= D, and T is normally much smaller than D, i.e., T << D.
    :param alpha_prior_params: The parameters for the Gamma prior on alpha, which is the scaling parameter for the DP.
    :param num_latent_samples: A positive integer specifying the number of samples to use when estimating the
    psi_statistics.
    :param mask_size: A mask size for grouping adjacent observed dimensions to the same group assignment. Default is 1
    so each dimension is not forced to share a group with its neighbor(s). The mask size must be an integer divisor of
    the total number of observed dimensions, i.e., k * mask_size = D for some positive integer k.
    :param seed: The seed value for the random number generator in numpy. This is used to ensure repeatable results.
    Default is 0.
    :return: An instance of the DP_GP_LVM class.
    """

    # Determine tensor dimensions and validate input.
    assert isinstance(y_train, np.ndarray), 'Training data must be provided as a numpy array.'
    num_samples, num_dimensions = np.shape(y_train)  # y_train provided as numpy array.
    assert isinstance(num_latent_dims, int), 'Number of latent dimensions must be an integer.'
    assert 0 < num_latent_dims < num_dimensions, \
        'Number of latent dimensions must be postive and less than the dimensionality of the observed data.'
    assert isinstance(num_inducing_points, int), 'Number of inducing points must be an integer.'
    assert 0 < num_inducing_points <= num_samples, \
        'Number of inducing points must be positive and less than or equal to the number of observations in the ' \
        'observed data.'
    assert isinstance(truncation_level, int), 'The truncation level must be an integer.'
    assert 0 < truncation_level <= min(num_samples, num_dimensions), \
        'The truncation level must be positive and less than or equal to the dimensionality of the observed data and ' \
        'less than or equal to the number of observations.'
    assert isinstance(num_latent_samples, int), 'Number of latent space samples must be an integer.'
    assert 0 < num_latent_samples < MAX_MC_SAMPLES, \
        'Number of latent space samples must be positive and less than {}.'.format(MAX_MC_SAMPLES)

    # Set random seed value.
    # TODO: Maybe add parameter for numpy seed and tensorflow seed so they can be different.
    assert isinstance(seed, int) and seed >= 0, 'Seed must be a 32-bit unsigned integer, i.e., 0 <= seed <= 2^32 - 1.'
    np.random.seed(seed=seed)
    tf.set_random_seed(seed=seed)

    # Fit latent means using PCA and the inducing inputs as a subset of those with a little noise.
    x_init = pca(y_train, num_latent_dimensions=num_latent_dims)
    x_mean = tf.Variable(x_init, dtype=TF_DTYPE, trainable=True)  # [N x Q].
    x_var_diag = create_positive_variable(initial_value=0.5,
                                          shape=(num_samples, num_latent_dims),
                                          is_trainable=True)  # [N x Q].
    # N, Q-length multivariate normal distributions with a diagonal covariance.
    q_x = tfp.distributions.MultivariateNormalDiag(loc=x_mean, scale_diag=tf.sqrt(x_var_diag))
    x_samples = q_x.sample([num_latent_samples])  # [num_latent_samples x N x Q].

    # TODO: Should really be Q, N-length mulitvariate normals so it is easy to modify for dynamic prior.
    # # Q, N-length multivariate normal distributions with a diagonal covariance.
    # q_x = tfp.distributions.MultivariateNormalDiag(loc=tf.transpose(x_mean),
    #                                                scale_diag=tf.transpose(tf.sqrt(x_var_diag)))
    # x_samples = q_x.sample([num_latent_samples])  # [num_latent_samples x Q x N].

    # Initialise inducing inputs as a subset of the PCA values calculated for latent means plus some noise.
    x_u_init = np.random.permutation(x_init)[:num_inducing_points] + \
        np.random.normal(loc=0.0, scale=0.01, size=(num_inducing_points, num_latent_dims))
    x_u = tf.Variable(x_u_init, dtype=TF_DTYPE, trainable=True)  # [M x Q].

    # Define DP that handles the assignments.
    dp_model = dirichlet_process(num_samples=num_dimensions,
                                 alpha_prior_params=alpha_prior_params,
                                 truncation_level=truncation_level,
                                 mask_size=mask_size)
    phi_td = tf.transpose(dp_model.assignments)  # [T x D].

    # TODO: Handle masking here as we don't want to overcount q(Z). Or fix implementation in DP as we only want to count
    #  number of free parameters in phi, not the full [N x T] phi. Only want [N/mask_size x T] to contribute to lower 
    #  bound.     

    # Define kernel hyperparameters from DP.
    # [T x Q]. TODO: Maybe add some noise so do not initialise to same value.
    gamma_atoms = create_positive_variable(initial_value=GP_INIT_GAMMA,
                                           shape=(truncation_level, num_latent_dims),
                                           is_trainable=True)
    # Both are [T x 1]. TODO: Maybe add some noise so do not initialise to same value.
    sig_var_atoms = create_positive_variable(initial_value=GP_INIT_ALPHA,
                                             shape=(truncation_level, 1),
                                             is_trainable=True)
    beta_atoms = create_positive_variable(initial_value=GP_INIT_BETA,
                                          shape=(truncation_level, 1),
                                          is_trainable=True)

    kernel_hyperpriors_log_likelihood = tf.reduce_sum(log_normal_log_pdf(gamma_atoms)) + \
        tf.reduce_sum(log_normal_log_pdf(sig_var_atoms)) + \
        tf.reduce_sum(log_normal_log_pdf(beta_atoms))

    # Define kernel.
    kernel = k_ard_rbf(gamma=gamma_atoms, alpha=sig_var_atoms, beta=beta_atoms)  # Therefore, batch size is now T.

    # Define covariance matrices.
    # k_ff_diag = kernel.covariance_diag()
    k_fu = kernel.covariance_matrix(input_0=x_samples, input_1=x_u, include_noise=False, include_jitter=False)  # [num_latent_samples x T x N x M].

    # Define psi-statistics.

    psi_1 = empirical_mean(k_fu)  # [T x N x M].
    psi_2 = empirical_mean(tf.matmul(k_fu, k_fu, transpose_a=True))  # [T x M x M].

    # Define psi-statistics.
    psi_0 = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [T x 1].
    psi_1 = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [T x N x M].
    psi_2 = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)  # [T x M x M].

    # Calculate f_hat term from the evidence lower bound (ELBO). Using stable calculation for f_hat.
    beta_t11 = tf.expand_dims(beta_atoms, axis=-1)  # [T x 1 x 1].

    k_uu = kernel.covariance_matrix(input_0=x_u, input_1=None, include_noise=False, include_jitter=True)  # [T x M x M].
    l_uu = tf.cholesky(k_uu)  # [T x M x M].

    l_uu_inv_psi_2 = tf.matrix_triangular_solve(l_uu, psi_2, lower=True)  # [T x M x M].
    l_uu_inv_psi_2_inv_transpose = tf.transpose(
        tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2, perm=[0, 2, 1]), lower=True),
        perm=[0, 2, 1])  # [T x M x M].
    trace_l_uu_inv_psi_2_inv_transpose = tf.reduce_sum(tf.matrix_diag_part(l_uu_inv_psi_2_inv_transpose),
                                                       axis=-1,
                                                       keepdims=True)  # [T x 1].

    # [T x M x M].
    a = beta_t11 * l_uu_inv_psi_2_inv_transpose + tf.eye(num_inducing_points,
                                                         batch_shape=[truncation_level],
                                                         dtype=TF_DTYPE)
    l_a = tf.cholesky(a)  # [T x M x M].

    log_det_l_a = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a)), axis=1, keepdims=True)  # [T x 1].

    # [T x M x N].
    l_uu_inv_psi_1_transpose = tf.matrix_triangular_solve(l_uu, tf.transpose(psi_1, perm=[0, 2, 1]), lower=True)
    c = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_transpose, lower=True)  # [T x M x N].
    g = beta_t11 * c  # [T x M x N].
    g_transpose_g = tf.matmul(g, g, transpose_a=True)  # [T x N x N].

    # Slice method:
    # # T-length list of [N x N] tensors.
    # gg_slices = [tf.squeeze(tf.slice(g_transpose_g, [t, 0, 0], [1, -1, -1]), axis=[0]) for t in range(truncation_level)]
    # # D-length list of [N x 1] tensors.
    # y_slices = [tf.slice(y_train, [0, d], [-1, 1]) for d in range(num_dimensions)]
    #
    # reconstruction_term = tf.reduce_sum([[tf.slice(phi_td, [t, d], [1, 1]) *
    #                                       tf.matmul(tf.matmul(y_slices[d],
    #                                                           gg_slices[t],
    #                                                           transpose_a=True),
    #                                                 y_slices[d])
    #                                       for d in range(num_dimensions)] for t in range(truncation_level)])

    # Tile method:
    y_tiled = tf.tile(tf.expand_dims(y_train, axis=0), [truncation_level, 1, 1])  # [T x N x D].
    reconstruction_term = tf.reduce_sum(phi_td * tf.reduce_sum(tf.square(tf.matmul(g, y_tiled)), axis=1))

    y_squared_1d = tf.reduce_sum(tf.square(y_train), axis=0, keepdims=True)  # [1 x D].

    f_hat = -0.5 * num_samples * num_dimensions * np.log(2.0 * np.pi) + \
        tf.reduce_sum(phi_td * (0.5 * (num_samples * tf.log(beta_atoms) +
                                       beta_atoms * (trace_l_uu_inv_psi_2_inv_transpose - psi_0)) -
                                log_det_l_a)) - \
        0.5 * tf.reduce_sum(phi_td * beta_atoms * y_squared_1d) + \
        0.5 * reconstruction_term

    # Define KL divergence between q(X) and p(X).
    kl_q_x_p_x = calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar)

    # Define evidence lower bound (ELBO).
    gp_elbo = f_hat - kl_q_x_p_x

    # Define objective function.
    objective = dp_model.objective - gp_elbo - kernel_hyperpriors_log_likelihood

    class DP_GP_LVM(Trainable):
        """
        This class defines a DP_GP_LVM object.
        """

        @property
        def assignments(self):
            """
            TODO
            :return:
            """
            return dp_model.assignments

        @property
        def dp(self):
            """
            TODO
            :return:
            """
            return dp_model

        @property
        def dp_atoms(self):
            """
            TODO
            :return:
            """
            return gamma_atoms, sig_var_atoms, beta_atoms

        @property
        def kernel(self):
            """
            TODO
            :return:
            """
            return kernel

        @property
        def ard_weights(self):
            """
            TODO
            :return:
            """
            return kernel.hyperparameters[KernelHyperparameters.ARD_WEIGHTS]

        @property
        def signal_variance(self):
            """
            TODO
            :return:
            """
            return kernel.hyperparameters[KernelHyperparameters.SIGNAL_VARIANCE]

        @property
        def noise_precision(self):
            """
            TODO
            :return:
            """
            return kernel.noise_precision

        @property
        def inducing_input(self):
            """
            TODO
            :return:
            """
            return x_u

        @property
        def q_x(self):
            """
            TODO
            :return:
            """
            return x_mean, x_covar

        @staticmethod
        def predict_new_latent_variables(y_test, use_pca=False):
            """
            TODO
            :param y_test:
            :param use_pca:
            :return:
            """
            # y_test is [N* x D].
            num_test_points, test_dims = np.shape(y_test)
            assert test_dims == num_dimensions, \
                'Observed dimensionality for prediction must be equal to the dimensionality of the training data.'

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(y_test, num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(y_train, y_test)
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Define extra f_hat terms for y_test.
            psi_0_test = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x M x M].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [D x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [D x M x M].

            a_test = beta_d11 * l_uu_inv_psi_2_test_inv_transpose + \
                tf.eye(num_inducing_points, batch_shape=[num_dimensions], dtype=TF_DTYPE)  # [D x M x M].
            l_a_test = tf.cholesky(a_test)  # [D x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            # [D x M x N*].
            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)
            c_test = tf.matrix_triangular_solve(l_a_test, l_uu_inv_psi_1_test_transpose, lower=True)  # [D x M x N*].
            c_transpose_c_test = tf.matmul(c_test, c_test, transpose_a=True)  # [D x N* x N*].

            y_test_beta_d1n = tf.expand_dims(tf.transpose(y_test), axis=1) * beta_d11  # [D x 1 x N*].

            f_hat_test = 0.5 * num_test_points * (tf.reduce_sum(tf.log(beta)) -
                                                  num_dimensions * np.log(2.0 * np.pi)) - \
                log_det_l_a_test + \
                0.5 * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_test_inv_transpose) - psi_0_test)) - \
                0.5 * tf.reduce_sum(beta * tf.expand_dims(tf.diag_part(tf.matmul(y_test, y_test, transpose_a=True)),
                                                          axis=-1)) + \
                0.5 * tf.reduce_sum(tf.matmul(tf.matmul(y_test_beta_d1n, c_transpose_c_test),
                                              y_test_beta_d1n, transpose_b=True))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define prediction lower bound.
            prediction_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            return prediction_lower_bound, x_test_mean, x_test_covar

        @staticmethod
        def predict_missing_data(y_test, use_pca=False):
            """
            TODO
            :param y_test:
            :param use_pca:
            :return:
            """
            # TODO: Currently assume y_test is first observed_dims of d. Update to be more generic.
            num_test_points, num_observed_dims = np.shape(y_test)
            assert num_observed_dims < num_dimensions, \
                'Observed dimensionality for missing data scenario must be less than total ' \
                'dimensionality of training data.'

            # Get slice of y_train for remaining unobserved dimensions. Du = D - Do.
            # Slice y_train into observed and unobserved dimensions for the missing data. D = Do + Du
            y_train_observed = tf.slice(y_train, begin=[0, 0], size=[-1, num_observed_dims])  # [N x Do]
            y_train_unobserved = tf.slice(y_train, begin=[0, num_observed_dims], size=[-1, -1])  # [N x Du]

            # Slice beta into observed and unobserved dimensions for the missing data.
            beta_observed = tf.slice(beta, begin=[0, 0], size=[num_observed_dims, -1])  # [Do x 1].
            beta_unobserved = tf.slice(beta, begin=[num_observed_dims, 0], size=[-1, -1])  # [Du x 1].

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(y_test, num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(y_train_observed, y_test)
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))  # [N* x Q].

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Calculate some other intermediate values for prediction.
            l_uu_inv = tf.matrix_triangular_solve(l_uu,
                                                  tf.eye(num_inducing_points, batch_shape=[num_dimensions],
                                                         dtype=TF_DTYPE))  # [D x M x M].
            l_a_inv = tf.matrix_triangular_solve(l_a,
                                                 tf.eye(num_inducing_points, batch_shape=[num_dimensions],
                                                        dtype=TF_DTYPE))  # [D x M x M].

            # Define extra f_hat terms for y_test.
            psi_0_test = kernel.psi_0(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u,
                                      latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [D x M x M].

            # Slice into observed and unobserved dimensions.
            psi_0_test_observed = tf.slice(psi_0_test,
                                           begin=[0, 0],
                                           size=[num_observed_dims, -1])  # [Do x 1].
            psi_0_test_unobserved = tf.slice(psi_0_test,
                                             begin=[num_observed_dims, 0],
                                             size=[-1, -1])  # [Du x 1].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [D x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [D x M x M].

            # Slice so only looking at observed dimensions.
            l_uu_inv_psi_2_test_inv_transpose_observed = tf.slice(l_uu_inv_psi_2_test_inv_transpose,
                                                                  begin=[0, 0, 0],
                                                                  size=[num_observed_dims, -1, -1])  # [Do x M x M].

            a_test = tf.expand_dims(beta_observed, axis=-1) * l_uu_inv_psi_2_test_inv_transpose_observed + \
                tf.eye(num_inducing_points, batch_shape=[num_observed_dims], dtype=TF_DTYPE)  # [Do x M x M].
            l_a_test = tf.cholesky(a_test)  # [Do x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)  # [D x M x N*].
            # Slice so only looking at observed dimensions.
            l_uu_inv_psi_1_test_transpose_observed = tf.slice(l_uu_inv_psi_1_test_transpose,
                                                              begin=[0, 0, 0],
                                                              size=[num_observed_dims, -1, -1])  # [Do x M x N*].

            c_test = tf.matrix_triangular_solve(l_a_test,
                                                l_uu_inv_psi_1_test_transpose_observed,
                                                lower=True)  # [Do x M x N*].
            c_transpose_c_test = tf.matmul(c_test, c_test, transpose_a=True)  # [Do x N* x N*].

            y_test_beta_do1n = tf.expand_dims(tf.transpose(y_test) * beta_observed, axis=1)  # [Do x 1 x N*].

            f_hat_test = 0.5 * num_test_points * (tf.reduce_sum(tf.log(beta_observed)) -
                                                  num_observed_dims * np.log(2.0 * np.pi)) - \
                log_det_l_a_test + \
                0.5 * tf.reduce_sum(beta_observed * (tf.trace(l_uu_inv_psi_2_test_inv_transpose_observed) -
                                                     psi_0_test_observed)) - \
                0.5 * tf.reduce_sum(beta_observed * tf.expand_dims(tf.diag_part(tf.matmul(y_test,
                                                                                          y_test,
                                                                                          transpose_a=True)),
                                                                   axis=-1)) + \
                0.5 * tf.reduce_sum(tf.matmul(tf.matmul(y_test_beta_do1n,
                                                        c_transpose_c_test),
                                              y_test_beta_do1n,
                                              transpose_b=True))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define missing data lower bound.
            missing_data_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            # Define extra terms for predicted mean.
            c_predict = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_test_transpose, lower=True)  # [D x M x N*].
            c_predict_c = tf.matmul(c_predict, c, transpose_a=True)  # [D x N* x N].

            # [Du x N* x N].
            c_predict_c_unobserved = tf.slice(c_predict_c, begin=[num_observed_dims, 0, 0], size=[-1, -1, -1])
            y_train_unobserved_du_n1 = tf.expand_dims(tf.transpose(y_train_unobserved), axis=-1)  # [Du x N x 1].

            # [N* x Du].
            predicted_mean = tf.transpose(beta_unobserved *
                                          tf.squeeze(tf.matmul(c_predict_c_unobserved,
                                                               y_train_unobserved_du_n1),
                                                     axis=-1))

            # Define extra terms for predicted covariance.
            g = psi_2_test - tf.matmul(psi_1_test, psi_1_test, transpose_a=True)  # [D x M x M].
            g_unobserved = tf.slice(g, begin=[num_observed_dims, 0, 0], size=[-1, -1, -1])  # [Du x M x M].

            scale_du = tf.matmul(
                tf.matmul(
                    tf.matmul(
                        l_uu_inv,
                        tf.matmul(
                            l_a_inv,
                            l_a_inv,
                            transpose_a=True
                        ),
                        transpose_a=True
                    ),
                    l_uu_inv
                ),
                psi_1,
                transpose_b=True
            )  # [D x M x N].

            # Slice so only looking at unobserved dimensions.
            scale_du_unobserved = tf.slice(scale_du,
                                           begin=[num_observed_dims, 0, 0],
                                           size=[-1, -1, -1])  # [Du x M x N].

            scale_du_yu = tf.matmul(scale_du_unobserved, y_train_unobserved_du_n1)  # [Du x M x 1].

            # [Du x 1].
            yu_variance = tf.square(beta_unobserved) * tf.squeeze(tf.matmul(scale_du_yu,
                                                                            tf.matmul(g_unobserved, scale_du_yu),
                                                                            transpose_a=True),
                                                                  axis=-1)

            # TODO: Make this calculation more efficient by ignoring the observed dimensions.
            trace_term = tf.expand_dims(
                tf.trace(
                    tf.matmul(
                        tf.matmul(
                            tf.matmul(
                                l_uu_inv,
                                tf.eye(num_inducing_points, batch_shape=[num_dimensions], dtype=TF_DTYPE) -
                                tf.matmul(l_a_inv, l_a_inv, transpose_a=True),
                                transpose_a=True
                            ),
                            l_uu_inv
                        ),
                        psi_2_test
                    )
                ),
                axis=-1)  # [D x 1].
            trace_term_unobserved = tf.slice(trace_term, begin=[num_observed_dims, 0], size=[-1, -1])  # [Du x 1].

            # [Du x N* x N*] so there is a specific covariance for each unobserved dimension.
            predicted_covar = tf.expand_dims(yu_variance, axis=-1) + \
                tf.expand_dims(psi_0_test_unobserved + tf.reciprocal(beta_unobserved) + trace_term_unobserved,
                               axis=-1) * \
                tf.eye(num_test_points, batch_shape=[num_dimensions - num_observed_dims], dtype=TF_DTYPE)

            return missing_data_lower_bound, x_test_mean, x_test_covar, predicted_mean, predicted_covar

        @property
        def objective(self):
            """
            TODO
            :return:
            """
            return objective

    return DP_GP_LVM()
'''
