"""
This module tests the training of DP-GP-LVM with difficult synthetic (or toy) data generated from known GPs.
"""

from distributions.normal import mvn_log_pdf
from kernels.rbf_kernel import k_ard_rbf
from models.dp_gp_lvm import dp_gp_lvm
from models.gaussian_process import bayesian_gp_lvm
from utils.types import get_training_variables, get_prediction_variables, TF_DTYPE

import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler
from sys import path
import tensorflow as tf
from time import time


def prepare_data(data_file_path, seed_val=1, mask_size=1):
    """
    TODO
    :return:
    """

    original_data = np.load(data_file_path)
    num_samples, num_dimensions = original_data.shape

    # Randomly permute rows and columns.
    # Permute columns by grouping by mask size, i.e., keep 2D coordinates together if mask_size=2.
    np.random.seed(seed=seed_val)
    row_indices = np.random.permutation(num_samples)
    cols = np.arange(num_dimensions).reshape((np.int(num_dimensions / mask_size), mask_size))
    col_indices = cols[np.random.permutation(np.int(num_dimensions / mask_size)), :].flatten()
    permuted_data = original_data[row_indices, :][:, col_indices]

    # Data has already been normalised as it was sampled from zero mean GP.
    normalised_data = permuted_data

    return normalised_data, permuted_data, original_data


def prepare_missing_data(data, percent_samples_observe=0.75, percent_dimensions_observe=0.85):
    """
    TODO
    :return:
    """

    num_samples, num_dimensions = data.shape

    # Separate into training and test data.
    num_training_samples = int(np.ceil(percent_samples_observe * num_samples))
    num_observed_dimensions = int(np.ceil(percent_dimensions_observe * num_dimensions))

    training_data = data[:num_training_samples, :]
    test_data = data[num_training_samples:, :]
    test_data_observed = test_data[:, :num_observed_dimensions]
    test_data_unobserved_ground_truth = test_data[:, num_observed_dimensions:]

    return training_data, test_data_observed, test_data_unobserved_ground_truth


def run_bgplvm(y_train, y_test_observed, y_test_unobserved, num_latent_dimensions, num_inducing_points,
               train_iter, predict_iter, learning_rate, save_file, seed_val=1):
    """
    TODO
    :param y_train:
    :param y_test_observed:
    :param y_test_unobserved:
    :param num_latent_dimensions:
    :param num_inducing_points:
    :param train_iter:
    :param predict_iter:
    :param learning_rate:
    :param save_file:
    :param seed_val:
    :return:
    """

    # Set seed.
    np.random.seed(seed=seed_val)

    # Define instance of Bayesian GP-LVM.
    bgplvm = bayesian_gp_lvm(y_train=y_train,
                             num_latent_dims=num_latent_dimensions,
                             num_inducing_points=num_inducing_points)

    num_unobserved_dimensions = np.shape(y_test_unobserved)[1]

    # Define objectives.
    training_objective = bgplvm.objective
    predict_lower_bound, x_mean_test, x_covar_test, \
        predicted_mean, predicted_covar = bgplvm.predict_missing_data(y_test=y_test_observed)
    predict_objective = tf.negative(predict_lower_bound)

    # Optimisation.
    training_var_list = get_training_variables()
    predict_var_list = get_prediction_variables()

    opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=training_objective,
                                                                             var_list=training_var_list)
    opt_predict = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=predict_objective,
                                                                               var_list=predict_var_list)

    with tf.Session() as s:

        # Initialise variables.
        s.run(tf.variables_initializer(var_list=training_var_list))  # Initialise training variables first.
        s.run(tf.variables_initializer(var_list=predict_var_list))  # Then initialise prediction variables.
        s.run(tf.global_variables_initializer())  # Finally initialise any remaining global variables such as opt ones.

        # Training optimisation loop.
        start_time = time()
        print('\nTraining BGPLVM..')
        for c in range(train_iter):
            s.run(opt_train)
            if (c % 100) == 0:
                print('  BGPLVM opt iter {:5}: {}'.format(c, s.run(training_objective)))
        end_time = time()
        train_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  BGPLVM: {}'.format(s.run(training_objective)))
        print('Time to optimise: {} s'.format(train_opt_time))

        # Get converged values as numpy arrays.
        ard_weights, noise_precision, signal_variance, inducing_input = s.run((bgplvm.ard_weights,
                                                                               bgplvm.noise_precision,
                                                                               bgplvm.signal_variance,
                                                                               bgplvm.inducing_input))
        x_mean, x_covar = s.run(bgplvm.q_x)

        # Initialise prediction variables.
        s.run(tf.variables_initializer(var_list=predict_var_list))

        # Prediction optimisation loop.
        start_time = time()
        print('\nOptimising Predictions..')
        for c in range(predict_iter):
            s.run(opt_predict)
            if (c % 100) == 0:
                print('  BGPLVM opt iter {:5}: {}'.format(c, s.run(predict_objective)))
        end_time = time()
        predict_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  BGPLVM: {}'.format(s.run(predict_objective)))
        print('Time to optimise: {} s'.format(predict_opt_time))

        # Get converged values as numpy arrays.
        x_mean_test_np, x_covar_test_np, predicted_mean_np, predicted_covar_np = s.run((x_mean_test,
                                                                                        x_covar_test,
                                                                                        predicted_mean,
                                                                                        predicted_covar))

        # Calculate log-likelihood of ground truth with predicted posteriors.
        gt_log_likelihoods = [
            mvn_log_pdf(x=tf.transpose(tf.slice(y_test_unobserved, begin=[0, du], size=[-1, 1])),
                        mean=tf.transpose(tf.slice(predicted_mean, begin=[0, du], size=[-1, 1])),
                        covariance=tf.squeeze(tf.slice(predicted_covar, begin=[du, 0, 0], size=[1, -1, -1]),
                                              axis=0))
            for du in range(num_unobserved_dimensions)]
        gt_log_likelihoods_np = np.array(s.run(gt_log_likelihoods))
        gt_log_likelihood = np.sum(gt_log_likelihoods_np)

    # Save results.
    np.savez(save_file, y_train=y_train, y_test_observed=y_test_observed, y_test_unobserved=y_test_unobserved,
             ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
             x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
             x_mean_test=x_mean_test_np, x_covar_test=x_covar_test_np, predicted_mean=predicted_mean_np,
             predicted_covar=predicted_covar_np, predict_opt_time=predict_opt_time,
             gt_log_likelihoods=gt_log_likelihoods_np, gt_log_likelihood=gt_log_likelihood)

    # Print results.
    print('\nBGPLVM:')
    print('  Ground Truth Predicted Posterior Log-Likelihood: {}'.format(gt_log_likelihood))
    print('  Noise Precision: {}'.format(np.squeeze(noise_precision)))


def run_gpdp(y_train, y_test_observed, y_test_unobserved, num_latent_dimensions, num_inducing_points, truncation_level,
             dp_mask_size, train_iter, predict_iter, learning_rate, save_file, seed_val=1):
    """
    TODO
    :param y_train:
    :param y_test_observed:
    :param y_test_unobserved:
    :param num_latent_dimensions:
    :param num_inducing_points:
    :param truncation_level:
    :param dp_mask_size:
    :param train_iter:
    :param predict_iter:
    :param learning_rate:
    :param save_file:
    :param seed_val:
    :return:
    """

    # Set seed.
    np.random.seed(seed=seed_val)

    # Define instance of DP-GP-LVM .
    gpdp = dp_gp_lvm(y_train=y_train,
                     num_latent_dims=num_latent_dimensions,
                     num_inducing_points=num_inducing_points,
                     truncation_level=truncation_level,
                     mask_size=dp_mask_size)

    num_unobserved_dimensions = np.shape(y_test_unobserved)[1]

    # Define objectives.
    training_objective = gpdp.objective
    predict_lower_bound, x_mean_test, x_covar_test, \
        predicted_mean, predicted_covar = gpdp.predict_missing_data(y_test=y_test_observed)
    predict_objective = tf.negative(predict_lower_bound)

    # Optimisation.
    training_var_list = get_training_variables()
    predict_var_list = get_prediction_variables()

    opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=training_objective,
                                                                             var_list=training_var_list)
    opt_predict = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=predict_objective,
                                                                               var_list=predict_var_list)

    with tf.Session() as s:

        # Initialise variables.
        s.run(tf.variables_initializer(var_list=training_var_list))  # Initialise training variables first.
        s.run(tf.variables_initializer(var_list=predict_var_list))  # Then initialise prediction variables.
        s.run(tf.global_variables_initializer())  # Finally initialise any remaining global variables such as opt ones.

        # Training optimisation loop.
        start_time = time()
        print('\nTraining GP-DP..')
        for c in range(train_iter):
            s.run(opt_train)
            if (c % 100) == 0:
                print('  GP-DP opt iter {:5}: {}'.format(c, s.run(training_objective)))
        end_time = time()
        train_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  GP-DP: {}'.format(s.run(training_objective)))
        print('Time to optimise: {} s'.format(train_opt_time))

        # Get converged values as numpy arrays.
        ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
            s.run((gpdp.ard_weights, gpdp.noise_precision, gpdp.signal_variance, gpdp.inducing_input, gpdp.assignments))
        x_mean, x_covar = s.run(gpdp.q_x)
        gamma_atoms, alpha_atoms, beta_atoms = s.run(gpdp.dp_atoms)

        # Initialise prediction variables.
        s.run(tf.variables_initializer(var_list=predict_var_list))

        # Prediction optimisation loop.
        start_time = time()
        print('\nOptimising Predictions..')
        for c in range(predict_iter):
            s.run(opt_predict)
            if (c % 100) == 0:
                print('  GP-DP opt iter {:5}: {}'.format(c, s.run(predict_objective)))
        end_time = time()
        predict_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  GP-DP: {}'.format(s.run(predict_objective)))
        print('Time to optimise: {} s'.format(predict_opt_time))

        # Get converged values as numpy arrays.
        x_mean_test_np, x_covar_test_np, predicted_mean_np, predicted_covar_np = s.run((x_mean_test,
                                                                                        x_covar_test,
                                                                                        predicted_mean,
                                                                                        predicted_covar))

        # Calculate log-likelihood of ground truth with predicted posteriors.
        gt_log_likelihoods = [
            mvn_log_pdf(x=tf.transpose(tf.slice(y_test_unobserved, begin=[0, du], size=[-1, 1])),
                        mean=tf.transpose(tf.slice(predicted_mean, begin=[0, du], size=[-1, 1])),
                        covariance=tf.squeeze(tf.slice(predicted_covar, begin=[du, 0, 0], size=[1, -1, -1]),
                                              axis=0))
            for du in range(num_unobserved_dimensions)]
        gt_log_likelihoods_np = np.array(s.run(gt_log_likelihoods))
        gt_log_likelihood = np.sum(gt_log_likelihoods_np)

    # Save results.
    np.savez(save_file, y_train=y_train, y_test_observed=y_test_observed, y_test_unobserved=y_test_unobserved,
             ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
             x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms,
             beta_atoms=beta_atoms,train_opt_time=train_opt_time, x_mean_test=x_mean_test_np,
             x_covar_test=x_covar_test_np, predicted_mean=predicted_mean_np, predicted_covar=predicted_covar_np,
             predict_opt_time=predict_opt_time, gt_log_likelihoods=gt_log_likelihoods_np,
             gt_log_likelihood=gt_log_likelihood, assignments=assignments)

    # Print results.
    print('\nGP-DP:')
    print('  Ground Truth Predicted Posterior Log-Likelihood: {}'.format(gt_log_likelihood))
    print('  Noise Precisions: {}'.format(np.squeeze(noise_precision)))


if __name__ == '__main__':

    # Create synthetic data if need be.
    # Random seed.
    np.random.seed(1)

    # Generate synthetic data.
    num_samples = 100  # 30  # 80
    num_input_dimensions = 5
    num_output_dimensions = 30  # 50

    # Define paths.
    absolute_path = [ap for ap in path if 'aistats_2019' in ap]
    data_path = absolute_path[-1] + '/test/data/'
    results_path = absolute_path[-1] + '/test/results/'

    # Create synthetic data if it has not been created already.
    file_name = data_path + 'synthetic_data_hard_2afunc_ind_noise.npy'
    if isfile(file_name):

        y_train = np.load(file_name)
        assert y_train.shape[0] == num_samples
        assert y_train.shape[1] == num_output_dimensions

    else:

        # Random input or linspace.
        x = np.random.standard_normal((num_samples, num_input_dimensions))

        # Data generated from 4 different GPs with some shared and some private input dimensions.
        ard_1 = tf.constant([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=TF_DTYPE)  # Function of X0 and X1.
        ard_2 = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=TF_DTYPE)  # Function of X0 and X2.

        # Different noise terms per GP component.
        beta_1 = tf.constant([[100.0]], dtype=TF_DTYPE)  # Noise var is 0.01.
        beta_2 = tf.constant([[500.0]], dtype=TF_DTYPE)  # Noise var is 0.002.

        alpha_generator = tf.constant([[1.0]], dtype=TF_DTYPE)

        covar_1 = k_ard_rbf(gamma=ard_1, alpha=alpha_generator, beta=beta_1).covariance_matrix(input_0=x,
                                                                                               input_1=None,
                                                                                               include_noise=True,
                                                                                               include_jitter=True)
        covar_2 = k_ard_rbf(gamma=ard_2, alpha=alpha_generator, beta=beta_2).covariance_matrix(input_0=x,
                                                                                               input_1=None,
                                                                                               include_noise=True,
                                                                                               include_jitter=True)

        with tf.Session() as sess:
            covar_1_np = np.squeeze(sess.run(covar_1), axis=0)  # [N x N].
            covar_2_np = np.squeeze(sess.run(covar_2), axis=0)  # [N x N].

        # Sample from each GP D/2 times to build Y as [N x D] as each sample from GP is N-length.
        d_gp = int(0.5 * num_output_dimensions)
        y = np.concatenate(
            (
                np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples), cov=covar_1_np, size=d_gp)),
                np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples), cov=covar_2_np, size=d_gp)),
            ), axis=1)
        assert y.shape[0] == num_samples
        assert y.shape[1] == num_output_dimensions

        # Normalise data to zero mean and unit variance.
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y)

        # Save training data in npy file.
        np.save(file_name, y_train)

    # Print info.
    print('\nSynthetic Data:')
    print('  Total number of observations (N): {}'.format(num_samples))
    print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))

    # Optimisation variables.
    learning_rate = 0.05
    num_iter_train_bgplvm = 1500
    num_iter_predict_bgplvm = 750
    num_iter_train_gpdp = 1500
    num_iter_predict_gpdp = 750

    # Different configurations
    seeds = np.arange(10, 15, dtype=int)  # [10 - 14].
    percent_samples_observed = np.array([0.8, 0.9])
    percent_dimensions_observed = np.array([0.7, 0.8, 0.9])

    # Define paths.
    absolute_path = [ap for ap in path if 'aistats_2019' in ap]
    data_path = absolute_path[-1] + '/test/data/'
    results_path = absolute_path[-1] + '/test/results/'

    # Read all synthetic data set made with 2 functions and indpendent noise for each GP component.
    synthetic_ind_noise = data_path + 'synthetic_data_hard_2afunc_ind_noise.npy'

    # Total number of test scenarios.
    counter = 0.0
    total_num_tests = np.size(seeds) * np.size(percent_samples_observed) * np.size(percent_dimensions_observed)

    for seed in seeds:

        # Get randomly permuted, normalised data for independent noise synthetic data.
        ind_noise_np_file = results_path + 'synthetic_data_2afunc_ind_noise_missing_data_seed_{0}.npz'.format(seed)
        if isfile(ind_noise_np_file):
            temp = np.load(ind_noise_np_file)
            normalised_data_ind_noise = temp['normalised_data']
            permuted_data_ind_noise = temp['permuted_data']
            original_data_ind_noise = temp['original_data']
        else:
            normalised_data_ind_noise, permuted_data_ind_noise, original_data_ind_noise = \
                prepare_data(data_file_path=synthetic_ind_noise, seed_val=seed, mask_size=1)

            np.savez(ind_noise_np_file, normalised_data=normalised_data_ind_noise,
                     permuted_data=permuted_data_ind_noise,
                     original_data=original_data_ind_noise)

        for n in percent_samples_observed:

            for d in percent_dimensions_observed:

                # Print current test scenario.
                print('\n--------------------------------------------------------------------------------')
                print('\nCurrent Test Scenario:')
                print('  Random seed:                      {}'.format(seed))
                print('  Percent samples observed (n%):    {}'.format(n))
                print('  Percent dimensions observed (d%): {}'.format(d))

                # Separate into training and test data.
                config_ext = 'n_percent_{0}_d_percent_{1}'.format(n, d)
                ind_noise_np_file = results_path + \
                    'synthetic_data_2afunc_ind_noise_missing_data_seed_{0}_{1}.npz'.format(seed, config_ext)

                if isfile(ind_noise_np_file):
                    temp = np.load(ind_noise_np_file)
                    ind_noise_training_data = temp['training_data']
                    ind_noise_test_data_observed = temp['test_data_observed']
                    ind_noise_test_data_unobserved_ground_truth = temp['test_data_unobserved_ground_truth']
                else:
                    ind_noise_training_data, \
                        ind_noise_test_data_observed, \
                        ind_noise_test_data_unobserved_ground_truth = \
                        prepare_missing_data(data=normalised_data_ind_noise,
                                             percent_samples_observe=n,
                                             percent_dimensions_observe=d)
                    np.savez(ind_noise_np_file, training_data=ind_noise_training_data,
                             test_data_observed=ind_noise_test_data_observed,
                             test_data_unobserved_ground_truth=ind_noise_test_data_unobserved_ground_truth)

                # # From full data test for DP-GP-LVM.
                # num_inducing_points = 50
                # num_latent_dimensions = 10
                # truncation_level = 20

                # Change values slightly from full data test for DP-GP-LVM.
                num_inducing_points = 35
                num_latent_dimensions = 10
                truncation_level = 15

                # Get stats about data.
                num_training_samples, num_output_dimensions = np.shape(ind_noise_training_data)
                num_test_samples, num_observed_dimensions = np.shape(ind_noise_test_data_observed)
                num_unobserved_dimensions = np.shape(ind_noise_test_data_unobserved_ground_truth)[1]
                assert num_output_dimensions == num_observed_dimensions + num_unobserved_dimensions, \
                    'Missing data dimensions do not match.'
                assert num_test_samples == np.shape(ind_noise_test_data_unobserved_ground_truth)[0], \
                    'Number of observations in test data and missing data must be the same.'

                # Print info.
                print('\nSynthetic Data:')
                print('Total number of observations: {}'.format(num_training_samples + num_test_samples))
                print('Total number of output dimensions: {}'.format(num_output_dimensions))
                print('\nTraining Data:')
                print('Number of training samples: {}'.format(num_training_samples))
                print('Number of training dimensions: {}'.format(num_output_dimensions))
                print('\nMissing Data:')
                print('Number of test samples: {}'.format(num_test_samples))
                print('Number of provided/observed dimensions: {}'.format(num_observed_dimensions))
                print('Number of missing/unobserved dimensions: {}'.format(num_unobserved_dimensions))

                # Run each model.
                bgplvm_ind_noise_file = results_path + \
                    'bgplvm_synthetic_data_2afunc_ind_noise_missing_data_seed_{0}_{1}.npz'.format(seed, config_ext)
                if isfile(bgplvm_ind_noise_file):
                    print('\nAlready ran this independent noise configuration for BGPLVM.')
                else:
                    # Reset default graph before building new model graph. This speeds up script.
                    tf.reset_default_graph()
                    # Build BGP-LVM graph and run it for current configuration.
                    run_bgplvm(y_train=ind_noise_training_data,
                               y_test_observed=ind_noise_test_data_observed,
                               y_test_unobserved=ind_noise_test_data_unobserved_ground_truth,
                               num_latent_dimensions=num_latent_dimensions,
                               num_inducing_points=num_inducing_points,
                               train_iter=num_iter_train_bgplvm,
                               predict_iter=num_iter_predict_bgplvm,
                               learning_rate=learning_rate,
                               save_file=bgplvm_ind_noise_file,
                               seed_val=seed)

                # Recalculate gt_log_likelihood using Cov(F*), not Cov(Y*). If model learns lots of noise,
                #   then likelihood is high as it can explain all the data with just noise.
                bgplvm_ind_noise_file_f_star = results_path + \
                    'bgplvm_synthetic_data_2afunc_ind_noise_missing_data_f_star_seed_{0}_{1}.npz'.format(seed,
                                                                                                        config_ext)
                if isfile(bgplvm_ind_noise_file_f_star):
                    print('\nAlready ran this calculation for BGPLVM.')
                else:
                    print('\nRunning Cov(F*) calculation for BGPLVM.')

                    # Get predicted means, predicted covars, noise precision, and ground truth data.
                    results_data = np.load(bgplvm_ind_noise_file)
                    bgplvm_predicted_means = results_data['predicted_mean']
                    bgplvm_predicted_covars = results_data['predicted_covar']
                    bgplvm_beta = results_data['noise_precision']
                    bgplvm_gt_unobserved = results_data['y_test_unobserved']

                    num_test_points, num_unobserved_dimensions = bgplvm_gt_unobserved.shape

                    # Reset default graph before building new model graph. This speeds up script.
                    tf.reset_default_graph()

                    with tf.Session() as sess:
                        # Calculate Cov(F*).
                        f_star_predicted_covars = bgplvm_predicted_covars - \
                                                  tf.reciprocal(bgplvm_beta) * tf.eye(num_test_points,
                                                                                      batch_shape=[
                                                                                          num_unobserved_dimensions],
                                                                                      dtype=TF_DTYPE)

                        f_star_predicted_covars_np = np.array(sess.run(f_star_predicted_covars))

                        # Calculate log-likelihood of ground truth with predicted posteriors.
                        bgplvm_gt_log_likelihoods = [
                            mvn_log_pdf(x=tf.transpose(tf.slice(bgplvm_gt_unobserved,
                                                                begin=[0, du],
                                                                size=[-1, 1])),
                                        mean=tf.transpose(tf.slice(bgplvm_predicted_means,
                                                                   begin=[0, du],
                                                                   size=[-1, 1])),
                                        covariance=tf.squeeze(
                                            tf.slice(f_star_predicted_covars,
                                                     begin=[du, 0, 0],
                                                     size=[1, -1, -1]),
                                            axis=0))
                            for du in range(num_unobserved_dimensions)]

                        bgplvm_gt_log_likelihoods_np = np.array(sess.run(bgplvm_gt_log_likelihoods))

                    bgplvm_gt_log_likelihood_np = np.sum(bgplvm_gt_log_likelihoods_np)

                    # Save results.
                    np.savez(bgplvm_ind_noise_file_f_star, y_test_unobserved=bgplvm_gt_unobserved,
                             noise_precision=bgplvm_beta, predicted_mean=bgplvm_predicted_means,
                             f_star_predicted_covar=f_star_predicted_covars_np,
                             gt_log_likelihoods=bgplvm_gt_log_likelihoods_np,
                             gt_log_likelihood=bgplvm_gt_log_likelihood_np)

                gpdp_ind_noise_file = results_path + \
                    'gpdp_synthetic_data_2afunc_ind_noise_missing_data_seed_{0}_{1}.npz'.format(seed, config_ext)
                if isfile(gpdp_ind_noise_file):
                    print('\nAlready ran this more noise configuration for GP-DP.')
                else:
                    # Reset default graph before building new model graph. This speeds up script.
                    tf.reset_default_graph()
                    # Build GP-DP model graph and run it for current configuration.
                    run_gpdp(y_train=ind_noise_training_data,
                             y_test_observed=ind_noise_test_data_observed,
                             y_test_unobserved=ind_noise_test_data_unobserved_ground_truth,
                             num_latent_dimensions=num_latent_dimensions,
                             num_inducing_points=num_inducing_points,
                             truncation_level=truncation_level,
                             dp_mask_size=1,
                             train_iter=num_iter_train_gpdp,
                             predict_iter=num_iter_predict_gpdp,
                             learning_rate=learning_rate,
                             save_file=gpdp_ind_noise_file,
                             seed_val=seed)

                # Recalculate gt_log_likelihood using Cov(F*), not Cov(Y*). If model learns lots of noise,
                #   then likelihood is high as it can explain all the data with just noise.
                gpdp_ind_noise_file_f_star = results_path + \
                    'gpdp_synthetic_data_2afunc_ind_noise_missing_data_f_star_seed_{0}_{1}.npz'.format(seed, config_ext)
                if isfile(gpdp_ind_noise_file_f_star):
                    print('\nAlready ran this calculation for GP-DP.')
                else:
                    print('\nRunning Cov(F*) calculation for GP-DP.')

                    # Get predicted means, predicted covars, noise precision, and ground truth data.
                    results_data = np.load(gpdp_ind_noise_file)
                    gpdp_predicted_means = results_data['predicted_mean']
                    gpdp_predicted_covars = results_data['predicted_covar']
                    gpdp_beta = results_data['noise_precision']
                    gpdp_gt_unobserved = results_data['y_test_unobserved']

                    num_test_points, num_unobserved_dimensions = gpdp_gt_unobserved.shape
                    num_observed_dimensions = results_data['y_test_observed'].shape[1]

                    # Reset default graph before building new model graph. This speeds up script.
                    tf.reset_default_graph()

                    with tf.Session() as sess:
                        # Calculate Cov(F*).
                        beta_du1 = tf.slice(gpdp_beta,
                                            begin=[num_observed_dimensions, 0],
                                            size=[num_unobserved_dimensions, -1])  # [Du x 1].
                        f_star_predicted_covars = gpdp_predicted_covars - \
                                                  tf.expand_dims(tf.reciprocal(beta_du1), axis=-1) * \
                                                  tf.eye(num_test_points, batch_shape=[num_unobserved_dimensions],
                                                         dtype=TF_DTYPE)

                        f_star_predicted_covars_np = np.array(sess.run(f_star_predicted_covars))

                        # Calculate log-likelihood of ground truth with predicted posteriors.
                        gpdp_gt_log_likelihoods = [
                            mvn_log_pdf(x=tf.transpose(tf.slice(gpdp_gt_unobserved,
                                                                begin=[0, du],
                                                                size=[-1, 1])),
                                        mean=tf.transpose(tf.slice(gpdp_predicted_means,
                                                                   begin=[0, du],
                                                                   size=[-1, 1])),
                                        covariance=tf.squeeze(
                                            tf.slice(f_star_predicted_covars,
                                                     begin=[du, 0, 0],
                                                     size=[1, -1, -1]),
                                            axis=0))
                            for du in range(num_unobserved_dimensions)]
                        # gpdp_gt_log_likelihood = tf.reduce_sum(gpdp_gt_log_likelihoods)
                        # gpdp_gt_log_likelihood_np, gpdp_gt_log_likelihoods_np = \
                        #     sess.run((gpdp_gt_log_likelihood, gpdp_gt_log_likelihoods))
                        gpdp_gt_log_likelihoods_np = np.array(sess.run(gpdp_gt_log_likelihoods))

                    gpdp_gt_log_likelihood_np = np.sum(gpdp_gt_log_likelihoods_np)

                    # Save results.
                    np.savez(gpdp_ind_noise_file_f_star, y_test_unobserved=gpdp_gt_unobserved,
                             noise_precision=gpdp_beta, predicted_mean=gpdp_predicted_means,
                             f_star_predicted_covar=f_star_predicted_covars_np,
                             gt_log_likelihoods=gpdp_gt_log_likelihoods_np,
                             gt_log_likelihood=gpdp_gt_log_likelihood_np)

                # Update number of tests run.
                counter += 1.0
                print('\n--------------------------------------------------------------------------------')
                print('\nPercent tests completed: {}'.format(100.0 * counter / total_num_tests))
