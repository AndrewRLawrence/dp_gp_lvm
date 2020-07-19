"""
This module implements a few different Gaussian process (GP) models, specifically GP regression, GP latent variable
model (GP-LVM), Bayesian GP-LVM (BGP-LVM), and Manifold Relevance Determination (MRD), which is a multi-view extension
to the BGP-LVM.
"""

from src.distributions.normal import mvn_log_pdf, mvn_conditional_mean_covar
from src.kernels.interfaces.kernel import KernelHyperparameters
from src.kernels.rbf_kernel import k_ard_rbf
from src.models.interfaces.trainable import Trainable
from src.models.expressions.gp_expressions import calculate_kl_divergence_standard_prior
from src.utils.constants import GP_INIT_GAMMA, GP_INIT_ALPHA, GP_INIT_BETA, GP_LVM_DEFAULT_LATENT_DIMENSIONS, \
    GP_LVM_DEFAULT_NUM_INDUCING_POINTS, MAX_MC_SAMPLES
from src.utils.expressions import nearest_neighbour, empirical_mean, principal_component_analysis as pca
from src.utils.types import TF_DTYPE, create_positive_variable, validate_kernel

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def gp_regression(x_train, y_train, kernel=None):
    """
    TODO
    :param x_train: The training input. Must be [N x Q].
    :param y_train: The training output. Must be [N x D]. Can be 1D output so can be [N x 1].
    :param kernel:
    :return:
    """

    # TODO: Validate input.

    # Determine dimensions of tensors.
    n, q = x_train.get_shape().as_list()
    d = y_train.get_shape().as_list()[1]

    # Validate kernel or define kernel, if necessary.
    if kernel is not None:
        validate_kernel(kernel)
        # gamma = kernel._hyperparameter_dict[KernelHyperparameters.ARD_WEIGHTS]
    else:
        # TODO: May want to add some noise to gamma init values.
        batch_size = 1
        gamma = create_positive_variable(initial_value=GP_INIT_GAMMA, shape=(batch_size, q), is_trainable=True)
        alpha = create_positive_variable(initial_value=GP_INIT_ALPHA, shape=(batch_size, 1), is_trainable=True)
        beta = create_positive_variable(initial_value=GP_INIT_BETA, shape=(batch_size, 1), is_trainable=True)
        kernel = k_ard_rbf(gamma=gamma, alpha=alpha, beta=beta)

    # Define GP.
    k_xx = kernel.covariance_matrix(input_0=x_train, input_1=None, include_noise=True, include_jitter=True)
    log_likelihood = mvn_log_pdf(x=tf.transpose(y_train),
                                 mean=tf.zeros(shape=(1, n), dtype=TF_DTYPE),
                                 covariance=k_xx) + \
                     kernel.prior_log_likelihood
    objective = tf.negative(tf.reduce_sum(log_likelihood))

    class GaussianProcess(Trainable):
        """
        This class defines a GaussianProcess object.
        """

        @property
        def kernel(self):
            """
            TODO
            :return:
            """
            return kernel

        @property
        def log_likelihood(self):
            """
            TODO
            :return:
            """
            return log_likelihood

        @staticmethod
        def predict_mean_covar(x_test):
            """
            TODO
            :param x_test:
            :return:
            """
            num_test_points = x_test.get_shape().as_list()[0]

            k_ss = kernel.covariance_matrix(input_0=x_test, input_1=None, include_noise=False, include_jitter=True)
            k_xs = kernel.covariance_matrix(input_0=x_train, input_1=x_test, include_noise=False, include_jitter=False)
            predicted_mean, predicated_covar = \
                mvn_conditional_mean_covar(b=y_train,
                                           mean_a=tf.zeros(shape=(num_test_points, 1), dtype=TF_DTYPE),
                                           mean_b=tf.zeros(shape=(n, 1), dtype=TF_DTYPE),
                                           covar_aa=k_ss,
                                           covar_bb=k_xx,
                                           covar_ab=k_xs)

            return predicted_mean, predicated_covar

        @property
        def objective(self):
            """
            TODO
            :return:
            """
            return objective

    return GaussianProcess()


def gp_lvm(y_train, kernel=None, num_latent_dims=GP_LVM_DEFAULT_LATENT_DIMENSIONS):
    """
    TODO
    :param y_train: The training output. Must be [N x D].
    :param kernel:
    :param num_latent_dims:
    :return:
    """

    # TODO: Validate input.

    # Determine dimensions of tensors.
    d = y_train.get_shape().as_list()[1]
    assert 0 < num_latent_dims < d, \
        'Number of latent dimensions must be postive and less than the dimensionality of the observed data.'

    # TODO: PCA will not work if y_train is a tensor.
    x_latent = tf.Variable(pca(y_train, num_latent_dimensions=num_latent_dims), dtype=TF_DTYPE, trainable=True)

    return gp_regression(x_train=x_latent, y_train=y_train, kernel=kernel)


def bayesian_gp_lvm(y_train, kernel=None, num_latent_dims=GP_LVM_DEFAULT_LATENT_DIMENSIONS,
                    num_inducing_points=GP_LVM_DEFAULT_NUM_INDUCING_POINTS, num_latent_samples=0):
    """
    TODO
    This is only standard latent prior so factorised p(X).
    :param y_train: As numpy array of size [N x D].
    :param kernel:
    :param num_latent_dims:
    :param num_inducing_points:
    :param num_latent_samples:
    :return:
    """

    # Determine tensor dimensions.
    num_samples, num_dimensions = np.shape(y_train)
    # num_samples, num_dimensions = y_train.get_shape().as_list()

    # Validate input.
    assert isinstance(num_latent_dims, int), 'Number of latent dimensions must be an integer.'
    assert 0 < num_latent_dims < num_dimensions, \
        'Number of latent dimensions must be postive and less than the dimensionality of the observed data.'
    assert isinstance(num_inducing_points, int), 'Number of inducing points must be an integer.'
    assert 0 < num_inducing_points < num_samples, \
        'Number of inducing points must be positive and less than the number of observations in the observed data.'
    assert isinstance(num_latent_samples, int), 'Number of latent space samples must be an integer.'
    assert 0 <= num_latent_samples < MAX_MC_SAMPLES, \
        'Number of latent space samples must be positive and less than {}.'.format(MAX_MC_SAMPLES)

    # Use stochastic variational inference if num_latent_samples is not zero; otherwise, use normal variational
    #  inference with closed-form versions of psi_statistics.
    if not num_latent_samples:
        use_svi = False
    else:
        use_svi = True

    # Validate kernel or define kernel, if necessary.
    batch_size = 1  # This is B.
    if kernel is not None:
        validate_kernel(kernel)
        # gamma = kernel._hyperparameter_dict[KernelHyperparameters.ARD_WEIGHTS]
    else:
        # TODO: May want to add some noise to gamma init values.
        # batch_size = 1  # This is B.
        gamma = create_positive_variable(initial_value=GP_INIT_GAMMA,
                                         shape=(batch_size, num_latent_dims),
                                         is_trainable=True)
        alpha = create_positive_variable(initial_value=GP_INIT_ALPHA,
                                         shape=(batch_size, 1),
                                         is_trainable=True)
        beta = create_positive_variable(initial_value=GP_INIT_BETA,
                                        shape=(batch_size, 1),
                                        is_trainable=True)
        kernel = k_ard_rbf(gamma=gamma, alpha=alpha, beta=beta)

    # Fit latent means using PCA and the inducing inputs as a subset of those with a little noise.
    x_init = pca(y_train, num_latent_dimensions=num_latent_dims)
    x_mean = tf.Variable(initial_value=x_init, dtype=TF_DTYPE, trainable=True)  # [N x Q].

    # Initialise inducing inputs as a subset of the PCA values calculated for latent means plus some noise.
    x_u_init = np.random.permutation(x_init)[:num_inducing_points] + \
        np.random.normal(loc=0.0, scale=0.01, size=(num_inducing_points, num_latent_dims))
    x_u = tf.Variable(initial_value=x_u_init, dtype=TF_DTYPE, trainable=True)  # [M x Q].

    # Define psi-statistics.
    if use_svi:
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

        kff_trace = tf.reduce_sum(kernel.covariance_diag(input_0=x_samples, include_noise=False, include_jitter=False),
                                  axis=-1)  # [num_latent_samples x B x 1].
        k_fu = kernel.covariance_matrix(input_0=x_samples, input_1=x_u, include_noise=False,
                                        include_jitter=False)  # [num_latent_samples x B x N x M].

        # Define psi-statistics.
        psi_0 = empirical_mean(kff_trace)  # [B x 1].
        psi_1 = empirical_mean(k_fu)  # [B x N x M].
        psi_2 = empirical_mean(tf.matmul(k_fu, k_fu, transpose_a=True))  # [B x M x M].

        # [B x 1].

    else:
        x_covar = tf.matrix_diag(create_positive_variable(initial_value=0.5,
                                                          shape=(num_samples, num_latent_dims),
                                                          is_trainable=True))  # [N x Q x Q].
        # [B x 1].
        psi_0 = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)
        # [B x N x M].
        psi_1 = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)
        # [B x M x M].
        psi_2 = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_mean, latent_input_covariance=x_covar)

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
    yy_transpose = tf.matmul(y_train, y_train, transpose_b=True)  # [N x N].

    f_hat = 0.5 * num_samples * num_dimensions * (tf.reduce_sum(tf.log(beta)) - np.log(2.0 * np.pi)) - \
        num_dimensions * log_det_l_a + \
        0.5 * num_dimensions * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_inv_transpose) - psi_0)) + \
        0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c, yy_transpose))) - \
        0.5 * tf.reduce_sum(beta * tf.trace(yy_transpose))

    # Define KL divergence between q(X) and p(X).
    kl_q_x_p_x = calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar)

    # Define evidence lower bound (ELBO).
    elbo = f_hat - kl_q_x_p_x

    # Define objective function.
    objective = tf.negative(elbo + kernel.prior_log_likelihood)

    # Calculate some other intermediate values for prediction.
    l_uu_inv = tf.matrix_triangular_solve(l_uu, tf.eye(num_inducing_points, batch_shape=[batch_size], dtype=TF_DTYPE))
    l_a_inv = tf.matrix_triangular_solve(l_a, tf.eye(num_inducing_points, batch_shape=[batch_size], dtype=TF_DTYPE))

    class BayesianGPLVM(Trainable):
        """
        This class defines a BayesianGPLVM object.
        """

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
                                      latent_input_covariance=x_test_covar)  # [B x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [B x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [B x M x M].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [B x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [B x M x M].

            a_test = beta_b11 * l_uu_inv_psi_2_test_inv_transpose + \
                tf.eye(num_inducing_points, batch_shape=[batch_size], dtype=TF_DTYPE)  # [B x M x M].
            l_a_test = tf.cholesky(a_test)  # [B x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            # [B x M x N*].
            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)
            c_test = tf.matrix_triangular_solve(l_a_test, l_uu_inv_psi_1_test_transpose, lower=True)  # [B x M x N*].
            c_transpose_c_test = tf.squeeze(tf.matmul(c_test, c_test, transpose_a=True),
                                            axis=0)  # Squeeze since B=1 so cTc is [N* x N*].
            yy_test_transpose = tf.matmul(y_test, y_test, transpose_b=True)  # [N* x N*].

            f_hat_test = 0.5 * num_test_points * num_dimensions * (tf.reduce_sum(tf.log(beta)) -
                                                                   np.log(2.0 * np.pi)) - \
                num_dimensions * log_det_l_a_test + \
                0.5 * num_dimensions * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_test_inv_transpose) -
                                                             psi_0_test)) + \
                0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c_test, yy_test_transpose))) - \
                0.5 * tf.reduce_sum(beta * tf.trace(yy_test_transpose))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define prediction lower bound.
            prediction_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            # Define test log-likelihood (from equation 36 of BGP-LVM journal paper).
            test_log_likelihood = f_hat_test - f_hat

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
                'Observed dimensionality for missing data scenario must be less than the total ' \
                'dimensionality of the training data.'

            # Get slice of y_train for remaining unobserved dimensions. Du = D - Do.
            # Slice y_train into observed and unobserved dimensions for the missing data. D = Do + Du
            y_train_observed = tf.slice(y_train, begin=[0, 0], size=[-1, num_observed_dims])  # [N x Do]
            y_train_unobserved = tf.slice(y_train, begin=[0, num_observed_dims], size=[-1, -1])  # [N x Du]

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(y_test, num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(y_train_observed, y_test)
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Define extra f_hat terms for y_test.
            psi_0_test = kernel.psi_0(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [B x 1].
            psi_1_test = kernel.psi_1(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [B x N* x M].
            psi_2_test = kernel.psi_2(inducing_input=x_u, latent_input_mean=x_test_mean,
                                      latent_input_covariance=x_test_covar)  # [B x M x M].

            l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2_test, lower=True)  # [B x M x M].
            l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                perm=[0, 2, 1])  # [B x M x M].

            a_test = beta_b11 * l_uu_inv_psi_2_test_inv_transpose + \
                     tf.eye(num_inducing_points, batch_shape=[batch_size], dtype=TF_DTYPE)  # [B x M x M].
            l_a_test = tf.cholesky(a_test)  # [B x M x M].

            log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

            # [B x M x N*].
            l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                       tf.transpose(psi_1_test, perm=[0, 2, 1]),
                                                                       lower=True)
            c_test = tf.matrix_triangular_solve(l_a_test, l_uu_inv_psi_1_test_transpose, lower=True)  # [B x M x N*].
            c_transpose_c_test = tf.squeeze(tf.matmul(c_test, c_test, transpose_a=True),
                                            axis=0)  # Squeeze since B=1 so cTc is [N* x N*].
            yy_test_transpose = tf.matmul(y_test, y_test, transpose_b=True)

            f_hat_test = 0.5 * num_test_points * num_observed_dims * (tf.reduce_sum(tf.log(beta)) -
                                                                      np.log(2.0 * np.pi)) - \
                num_observed_dims * log_det_l_a_test + \
                0.5 * num_observed_dims * tf.reduce_sum(beta *
                                                        (tf.trace(l_uu_inv_psi_2_test_inv_transpose) - psi_0_test)) + \
                0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c_test, yy_test_transpose))) - \
                0.5 * tf.reduce_sum(beta * tf.trace(yy_test_transpose))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define missing data lower bound.
            missing_data_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            # Define extra terms for predicted mean.
            c_predict = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_test_transpose, lower=True)  # [B x M x N*].
            # Squeeze since B=1 so c_predict_transpose x c is [N* x N].
            predicted_mean = beta * tf.matmul(tf.squeeze(tf.matmul(c_predict, c, transpose_a=True), axis=0),
                                              y_train_unobserved)  # [N* x Du].

            # Define extra terms for predicted covariance.
            g = psi_2_test - tf.matmul(psi_1_test, psi_1_test, transpose_a=True)  # [B x M x M].

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
                       )  # [B x M x N].
            scale_du_yu = tf.matmul(tf.squeeze(scale_du, axis=0), y_train_unobserved)  # [M x Du].

            # Du-length vector.
            yu_variance = tf.square(tf.squeeze(beta, axis=0)) * tf.diag_part(tf.matmul(
                scale_du_yu,
                tf.matmul(tf.squeeze(g, axis=0), scale_du_yu),
                transpose_a=True))

            # [Du x N* x N*] so there is a specific covariance for each unobserved dimension.
            predicted_covar = tf.expand_dims(tf.expand_dims(yu_variance, axis=-1), axis=-1) + \
                (
                    (psi_0_test + tf.reciprocal(beta) -
                        tf.trace(
                            tf.matmul(
                                tf.matmul(
                                    tf.matmul(
                                        l_uu_inv,
                                        (
                                            tf.eye(num_inducing_points, batch_shape=[batch_size], dtype=TF_DTYPE) -
                                            tf.matmul(l_a_inv, l_a_inv, transpose_a=True)
                                        ),
                                        transpose_a=True
                                    ),
                                    l_uu_inv
                                ),
                                psi_2_test
                            )
                        )
                     ) *
                    tf.eye(num_test_points, batch_shape=[batch_size], dtype=TF_DTYPE)
                )  # [Du x N* x N*].

            return missing_data_lower_bound, x_test_mean, x_test_covar, predicted_mean, predicted_covar

        @property
        def objective(self):
            """
            TODO
            :return:
            """
            return objective

    return BayesianGPLVM()


def manifold_relevance_determination(views_train, num_latent_dims=GP_LVM_DEFAULT_LATENT_DIMENSIONS,
                                     num_inducing_points=GP_LVM_DEFAULT_NUM_INDUCING_POINTS):
    """
    TODO
    :param views_train:
    :param num_latent_dims:
    :param num_inducing_points:
    :return:
    """

    # TODO: Validate input.

    # Determine tensor dimensions.
    num_views = len(views_train)
    shapes = np.array([np.shape(v) for v in views_train])
    num_samples = [shapes[v][0] for v in range(num_views)]
    num_dimensions = [shapes[v][1] for v in range(num_views)]

    assert np.size(np.unique(num_samples)) == 1, 'Each view must have the same number of observations.'
    num_samples = num_samples[0]
    assert 0 < num_latent_dims < np.sum(num_dimensions), \
        'Number of latent dimensions must be postive and less than the dimensionality of the observed data.'
    assert 0 < num_inducing_points < num_samples, \
        'Number of inducing points must be positive and less than the number of observations in the observed data.'

    # Create kernels, one per view.
    batch_size = 1
    gammas = [create_positive_variable(initial_value=GP_INIT_GAMMA,
                                       shape=(batch_size, num_latent_dims),
                                       is_trainable=True)
              for _ in range(num_views)]  # [V x B x Q].
    alphas = [create_positive_variable(initial_value=GP_INIT_ALPHA,
                                       shape=(batch_size, 1),
                                       is_trainable=True)
              for _ in range(num_views)]  # [V x B x 1].
    betas = [create_positive_variable(initial_value=GP_INIT_BETA,
                                      shape=(batch_size, 1),
                                      is_trainable=True)
             for _ in range(num_views)]  # [V x B x 1].
    kernels = [k_ard_rbf(gamma=gammas[v], alpha=alphas[v], beta=betas[v]) for v in range(num_views)]

    # Fit latent means using PCA and the inducing inputs as a subset of those with a little noise.
    x_init = pca(np.hstack(views_train), num_latent_dimensions=num_latent_dims)
    x_mean = tf.Variable(initial_value=x_init, dtype=TF_DTYPE, trainable=True)  # [N x Q].
    x_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                      shape=(num_samples, num_latent_dims),
                                                      is_trainable=True))  # [N x Q x Q].

    # Initialise inducing inputs as a subset of the PCA values calculated for latent means plus some noise.
    # There is a set of inducing inputs per view.
    x_u_inits = [np.random.permutation(x_init)[:num_inducing_points] +
                 np.random.normal(loc=0.0, scale=0.01, size=(num_inducing_points, num_latent_dims))
                 for _ in range(num_views)]
    x_us = [tf.Variable(initial_value=x_u_inits[v], dtype=TF_DTYPE, trainable=True)
            for v in range(num_views)]  # [V x M x Q].

    # Define psi-statistics.
    psi_0s = [kernels[v].psi_0(inducing_input=x_us[v], latent_input_mean=x_mean, latent_input_covariance=x_covar)
              for v in range(num_views)]  # [V x B x 1].
    psi_1s = [kernels[v].psi_1(inducing_input=x_us[v], latent_input_mean=x_mean, latent_input_covariance=x_covar)
              for v in range(num_views)]  # [V x B x N x M].
    psi_2s = [kernels[v].psi_2(inducing_input=x_us[v], latent_input_mean=x_mean, latent_input_covariance=x_covar)
              for v in range(num_views)]  # [V x B x M x M].

    # Define Kuu and its cholesky decomposition.
    k_uus = [kernels[v].covariance_matrix(input_0=x_us[v], input_1=None, include_noise=False, include_jitter=True)
             for v in range(num_views)]  # [V x B x M x M].
    l_uus = [tf.cholesky(k_uus[v]) for v in range(num_views)]  # [V x B x M x M].

    # Calculate f_hat terms from the evidence lower bound (ELBO). Using stable calculation for f_hat.
    f_hat = 0.0
    l_as = []
    for v in range(num_views):
        num_dims = num_dimensions[v]

        beta = kernels[v].noise_precision  # [B x 1].
        beta_b11 = tf.expand_dims(beta, axis=-1)  # [B x 1 x 1].

        l_uu = l_uus[v]  # [B x M x M].

        l_uu_inv_psi_2 = tf.matrix_triangular_solve(l_uu, psi_2s[v], lower=True)  # [B x M x M].
        l_uu_inv_psi_2_inv_transpose = tf.transpose(
            tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2, perm=[0, 2, 1]), lower=True),
            perm=[0, 2, 1])  # [B x M x M].

        a = beta_b11 * l_uu_inv_psi_2_inv_transpose + tf.eye(num_inducing_points,
                                                             batch_shape=[batch_size],
                                                             dtype=TF_DTYPE)
        l_a = tf.cholesky(a)  # [B x M x M].
        l_as.append(l_a)

        log_det_l_a = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a)))  # Scalar.

        # [B x M x N].
        l_uu_inv_psi_1_transpose = tf.matrix_triangular_solve(l_uu,
                                                              tf.transpose(psi_1s[v], perm=[0, 2, 1]),
                                                              lower=True)
        c = tf.matrix_triangular_solve(l_a, l_uu_inv_psi_1_transpose, lower=True)  # [B x M x N].
        c_transpose_c = tf.squeeze(tf.matmul(c, c, transpose_a=True), axis=0)  # Squeeze since B=1 so cTc is [N x N].
        yy_transpose = tf.matmul(views_train[v], views_train[v], transpose_b=True)  # [N x N].

        f_hat += 0.5 * num_samples * num_dims * (tf.reduce_sum(tf.log(beta)) - np.log(2.0 * np.pi)) - \
            num_dims * log_det_l_a + \
            0.5 * num_dims * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_inv_transpose) - psi_0s[v])) + \
            0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c, yy_transpose))) - \
            0.5 * tf.reduce_sum(beta * tf.trace(yy_transpose))

    # Define KL divergence between q(X) and p(X).
    kl_q_x_p_x = calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar)

    # Define evidence lower bound (ELBO).
    elbo = f_hat - kl_q_x_p_x

    # Define objective function.
    objective = tf.negative(elbo + tf.reduce_sum([kernels[v].prior_log_likelihood for v in range(num_views)]))

    class ManifoldRelevanceDetermination(Trainable):
        """
        This class defines a Manifold Relevance Determination object.
        """

        @property
        def number_of_views(self):
            """
            TODO
            :return:
            """
            return num_views

        @property
        def kernels(self):
            """
            TODO
            :return:
            """
            return kernels

        @property
        def ard_weights(self):
            """
            TODO
            :return:
            """
            return [kernel.hyperparameters[KernelHyperparameters.ARD_WEIGHTS] for kernel in kernels]

        @property
        def signal_variance(self):
            """
            TODO
            :return:
            """
            return [kernel.hyperparameters[KernelHyperparameters.SIGNAL_VARIANCE] for kernel in kernels]

        @property
        def noise_precision(self):
            """
            TODO
            :return:
            """
            return [kernel.noise_precision for kernel in kernels]

        @property
        def inducing_input(self):
            """
            TODO
            :return:
            """
            return x_us

        @property
        def q_x(self):
            """
            TODO
            :return:
            """
            return x_mean, x_covar

        @staticmethod
        def predict_new_latent_variables(views_test, use_pca=False):
            """
            TODO
            :param views_test:
            :param use_pca:
            :return:
            """
            # Determine tensor dimensions.
            assert num_views == len(views_test), \
                'The number of test views must be the same as the number of training views.'
            test_shapes = np.array([np.shape(v) for v in views_test])
            num_test_points = [test_shapes[v][0] for v in range(num_views)]
            num_test_dimensions = [test_shapes[v][1] for v in range(num_views)]

            assert np.size(np.unique(num_test_points)) == 1, 'Each view must have the same number of test points.'
            num_test_points = num_test_points[0]
            assert num_dimensions == num_test_dimensions, \
                'Observed dimensionality for prediction must be equal to the dimensionality of the training data ' \
                'for each view.'

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(np.hstack(views_test), num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(np.hstack(views_train), np.hstack(views_test))
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Define psi-statistics for test values.
            psi_0s_test = [kernels[v].psi_0(inducing_input=x_us[v],
                                            latent_input_mean=x_test_mean,
                                            latent_input_covariance=x_test_covar)
                           for v in range(num_views)]  # [V x B x 1].
            psi_1s_test = [kernels[v].psi_1(inducing_input=x_us[v],
                                            latent_input_mean=x_test_mean,
                                            latent_input_covariance=x_test_covar)
                           for v in range(num_views)]  # [V x B x N* x M].
            psi_2s_test = [kernels[v].psi_2(inducing_input=x_us[v],
                                            latent_input_mean=x_test_mean,
                                            latent_input_covariance=x_test_covar)
                           for v in range(num_views)]  # [V x B x M x M].

            # Calculate f_hat terms for prediction.
            f_hat_test = 0.0
            for v in range(num_views):
                num_dims = num_dimensions[v]

                beta = kernels[v].noise_precision  # [B x 1].
                beta_b11 = tf.expand_dims(beta, axis=-1)  # [B x 1 x 1].

                l_uu = l_uus[v]  # [B x M x M].

                l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2s_test[v], lower=True)  # [B x M x M].
                l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                    tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                    perm=[0, 2, 1])  # [B x M x M].

                a_test = beta_b11 * l_uu_inv_psi_2_test_inv_transpose + tf.eye(num_inducing_points,
                                                                               batch_shape=[batch_size],
                                                                               dtype=TF_DTYPE)  # [B x M x M].
                l_a_test = tf.cholesky(a_test)  # [B x M x M].

                log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

                # [B x M x N*].
                l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                           tf.transpose(psi_1s_test[v], perm=[0, 2, 1]),
                                                                           lower=True)
                # [B x M x N*].
                c_test = tf.matrix_triangular_solve(l_a_test, l_uu_inv_psi_1_test_transpose, lower=True)
                c_transpose_c_test = tf.squeeze(tf.matmul(c_test, c_test, transpose_a=True),
                                                axis=0)  # Squeeze since B=1 so cTc is [N* x N*].
                yy_test_transpose = tf.matmul(views_test[v], views_test[v], transpose_b=True)  # [N* x N*].

                f_hat_test += 0.5 * num_test_points * num_dims * (tf.reduce_sum(tf.log(beta)) - np.log(2.0 * np.pi)) - \
                    num_dims * log_det_l_a_test + \
                    0.5 * num_dims * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_test_inv_transpose) -
                                                           psi_0s_test[v])) + \
                    0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c_test, yy_test_transpose))) -\
                    0.5 * tf.reduce_sum(beta * tf.trace(yy_test_transpose))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define prediction lower bound.
            prediction_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            return prediction_lower_bound, x_test_mean, x_test_covar

        @staticmethod
        def predict_missing_data(views_test, use_pca=False):
            """
            TODO
            :param views_test:
            :param use_pca:
            :return:
            """
            # TODO: Currently assume views_test is first views with later ones missing. Update to be more generic.
            # Determine tensor dimensions.
            num_test_views = len(views_test)
            assert 0 < num_test_views < num_views, \
                'The number of test views for the missing data scenario must be less than the number of training views.'
            test_shapes = np.array([np.shape(v) for v in views_test])
            num_test_points = [test_shapes[v][0] for v in range(num_test_views)]
            num_test_dimensions = [test_shapes[v][1] for v in range(num_test_views)]

            assert np.size(np.unique(num_test_points)) == 1, 'Each view must have the same number of test points.'
            num_test_points = num_test_points[0]
            assert num_dimensions[:num_test_views] == num_test_dimensions, \
                'Observed dimensionality for prediction must be equal to the dimensionality of the training data ' \
                'for each observed view.'

            # Get slice of views_train for observed and unobserved views. V = Vo + Vu.
            views_train_observed = views_train[:num_test_views]  # [Vo x N x Dv].
            views_train_unobserved = views_train[num_test_views:]  # [Vu x N x Dv]

            # Define variables for q(X_star).
            if use_pca:
                # Initialise x_test_mean using PCA.
                init_values = pca(np.hstack(views_test), num_latent_dimensions=num_latent_dims)
            else:
                # Initialise x_test_mean with mean from nearest neighbour between training and test sets
                # plus some noise.
                y_test_nn_indices = nearest_neighbour(np.hstack(views_train_observed), np.hstack(views_test))
                init_values = tf.map_fn(lambda index: x_mean[index], y_test_nn_indices, dtype=TF_DTYPE) + \
                    np.random.normal(scale=0.01, size=(num_test_points, num_latent_dims))

            x_test_mean = tf.Variable(initial_value=init_values, dtype=TF_DTYPE, trainable=False)  # [N* x Q].
            x_test_covar = tf.matrix_diag(create_positive_variable(initial_value=1.0,
                                                                   shape=(num_test_points, num_latent_dims),
                                                                   is_trainable=False))  # [N* x Q x Q].

            # Define psi-statistics for test values.
            psi_0s_test = [kernels[v].psi_0(inducing_input=x_us[v],
                                            latent_input_mean=x_test_mean,
                                            latent_input_covariance=x_test_covar)
                           for v in range(num_views)]  # [V x B x 1].
            psi_1s_test = [kernels[v].psi_1(inducing_input=x_us[v],
                                            latent_input_mean=x_test_mean,
                                            latent_input_covariance=x_test_covar)
                           for v in range(num_views)]  # [V x B x N* x M].
            psi_2s_test = [kernels[v].psi_2(inducing_input=x_us[v],
                                            latent_input_mean=x_test_mean,
                                            latent_input_covariance=x_test_covar)
                           for v in range(num_views)]  # [V x B x M x M].

            # Calculate f_hat terms for prediction.
            f_hat_test = 0.0
            for v in range(num_test_views):
                num_dims = num_dimensions[v]

                beta = kernels[v].noise_precision  # [B x 1].
                beta_b11 = tf.expand_dims(beta, axis=-1)  # [B x 1 x 1].

                l_uu = l_uus[v]  # [B x M x M].

                l_uu_inv_psi_2_test = tf.matrix_triangular_solve(l_uu, psi_2s_test[v], lower=True)  # [B x M x M].
                l_uu_inv_psi_2_test_inv_transpose = tf.transpose(
                    tf.matrix_triangular_solve(l_uu, tf.transpose(l_uu_inv_psi_2_test, perm=[0, 2, 1]), lower=True),
                    perm=[0, 2, 1])  # [B x M x M].

                a_test = beta_b11 * l_uu_inv_psi_2_test_inv_transpose + tf.eye(num_inducing_points,
                                                                               batch_shape=[batch_size],
                                                                               dtype=TF_DTYPE)  # [B x M x M].
                l_a_test = tf.cholesky(a_test)  # [B x M x M].

                log_det_l_a_test = tf.reduce_sum(tf.log(tf.matrix_diag_part(l_a_test)))  # Scalar.

                # [B x M x N*].
                l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uu,
                                                                           tf.transpose(psi_1s_test[v], perm=[0, 2, 1]),
                                                                           lower=True)
                # [B x M x N*].
                c_test = tf.matrix_triangular_solve(l_a_test, l_uu_inv_psi_1_test_transpose, lower=True)
                c_transpose_c_test = tf.squeeze(tf.matmul(c_test, c_test, transpose_a=True),
                                                axis=0)  # Squeeze since B=1 so cTc is [N* x N*].
                yy_test_transpose = tf.matmul(views_test[v], views_test[v], transpose_b=True)  # [N* x N*].

                f_hat_test += 0.5 * num_test_points * num_dims * (tf.reduce_sum(tf.log(beta)) - np.log(2.0 * np.pi)) - \
                    num_dims * log_det_l_a_test + \
                    0.5 * num_dims * tf.reduce_sum(beta * (tf.trace(l_uu_inv_psi_2_test_inv_transpose) -
                                                           psi_0s_test[v])) + \
                    0.5 * tf.reduce_sum(tf.square(beta) * tf.trace(tf.matmul(c_transpose_c_test, yy_test_transpose))) -\
                    0.5 * tf.reduce_sum(beta * tf.trace(yy_test_transpose))

            # Define KL divergence between q(X_star) and p(X_star).
            kl_q_x_test_p_x_test = calculate_kl_divergence_standard_prior(x_mean=x_test_mean, x_covar=x_test_covar)

            # Define prediction lower bound.
            prediction_lower_bound = f_hat + f_hat_test - kl_q_x_p_x - kl_q_x_test_p_x_test

            # Define extra terms for predicted mean and covariance.
            predicted_means = []
            predicted_covars = []
            for v in range(num_test_views, num_views):
                l_uu_inv_psi_1_test_transpose = tf.matrix_triangular_solve(l_uus[v],
                                                                           tf.transpose(psi_1s_test[v], perm=[0, 2, 1]),
                                                                           lower=True)
                c_predict = tf.matrix_triangular_solve(l_as[v], l_uu_inv_psi_1_test_transpose)

                predicted_mean = kernels[v].noise_precision * tf.matmul(tf.squeeze(
                    tf.matmul(c_predict, c, transpose_a=True), axis=0), views_train[v])  # [N* x Dv].
                predicted_means.append(predicted_mean)  # [N* x Dv].

                l_uu_inv = tf.matrix_triangular_solve(l_uus[v], tf.eye(num_inducing_points,
                                                                       batch_shape=[batch_size],
                                                                       dtype=TF_DTYPE))
                l_a_inv = tf.matrix_triangular_solve(l_as[v], tf.eye(num_inducing_points,
                                                                     batch_shape=[batch_size],
                                                                     dtype=TF_DTYPE))

                g = psi_2s_test[v] - tf.matmul(psi_1s_test[v], psi_1s_test[v], transpose_a=True)  # [B x M x M].

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
                             psi_1s[v],
                             transpose_b=True
                       )  # [B x M x N].
                scale_du_yu = tf.matmul(tf.squeeze(scale_du, axis=0), views_train[v])  # [M x Dv].

                # Dv-length vector.
                yu_variance = tf.square(tf.squeeze(kernels[v].noise_precision, axis=0)) * tf.diag_part(
                    tf.matmul(scale_du_yu, tf.matmul(tf.squeeze(g, axis=0), scale_du_yu), transpose_a=True))

                # [Dv x N* x N*] so there is a specific covariance for each unobserved dimension.
                predicted_covar = tf.expand_dims(tf.expand_dims(yu_variance, axis=-1), axis=-1) + \
                                  ((psi_0s_test[v] + tf.reciprocal(kernels[v].noise_precision) -
                                    tf.trace(
                                        tf.matmul(
                                            tf.matmul(
                                                tf.matmul(l_uu_inv,
                                                          (tf.eye(num_inducing_points,
                                                                  batch_shape=[batch_size],
                                                                  dtype=TF_DTYPE) -
                                                           tf.matmul(l_a_inv, l_a_inv, transpose_a=True)),
                                                          transpose_a=True),
                                                l_uu_inv),
                                            psi_2s_test[v])
                                    )
                                    ) * tf.eye(num_test_points, batch_shape=[batch_size], dtype=TF_DTYPE))
                predicted_covars.append(predicted_covar)

            return prediction_lower_bound, x_test_mean, x_test_covar, predicted_means, predicted_covars

        @property
        def objective(self):
            """
            TODO
            :return:
            """
            return objective

    return ManifoldRelevanceDetermination()
