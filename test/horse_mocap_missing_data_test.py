"""
This module tests the predictive posterior for missing data in the Bayesian GP-LVM, MRD, and DP-GP-LVM models. This
module uses equine MOCAP data to perform these tests.
"""

from distributions.normal import mvn_log_pdf
from models.dp_gp_lvm import dp_gp_lvm
from models.gaussian_process import bayesian_gp_lvm, manifold_relevance_determination
from utils.constants import GP_LVM_DEFAULT_LATENT_DIMENSIONS, DP_DEFAULT_TRUNCATION_LEVEL
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

    # Normalise data to zero mean and unit variance for each column.
    scaler = StandardScaler()
    normalised_data = scaler.fit_transform(permuted_data)

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


def run_mrd(y_train, y_test_observed, y_test_unobserved, num_latent_dimensions, num_inducing_points, view_mask,
            train_iter, predict_iter, learning_rate, save_file, seed_val=1):
    """
    TODO
    :param y_train:
    :param y_test_observed:
    :param y_test_unobserved:
    :param num_latent_dimensions:
    :param num_inducing_points:
    :param view_mask:
    :param train_iter:
    :param predict_iter:
    :param learning_rate:
    :param save_file:
    :param seed_val:
    :return:
    """

    # Set seed.
    np.random.seed(seed=seed_val)

    # Segment training data into views of size view_mask.
    num_output_dimensions = np.shape(y_train)[1]
    views_train = [y_train[:, i:i+view_mask] for i in range(0, num_output_dimensions, view_mask)]

    # Define instance of MRD.
    mrd = manifold_relevance_determination(views_train=views_train,
                                           num_latent_dims=num_latent_dimensions,
                                           num_inducing_points=num_inducing_points)

    # Segment observed and unobserved data into views of size view_mask.
    num_observed_dimensions = np.shape(y_test_observed)[1]
    num_unobserved_dimensions = np.shape(y_test_unobserved)[1]
    # Need to make sure observed dimensions is multiple of view_mask, otherwise iterate until it is.
    if num_observed_dimensions % view_mask == 0:
        views_test_observed = [y_test_observed[:, i:i+view_mask] for i in range(0,
                                                                                num_observed_dimensions,
                                                                                view_mask)]
        views_test_unobserved = [y_test_unobserved[:, i:i+view_mask] for i in range(0,
                                                                                    num_unobserved_dimensions,
                                                                                    view_mask)]
    else:
        y_test = np.hstack((y_test_observed, y_test_unobserved))
        # Correct number of observed and unobserved dimensions.
        num_observed_dimensions = num_observed_dimensions + (view_mask - 1)
        num_unobserved_dimensions = num_output_dimensions - num_observed_dimensions
        views_test_observed = [y_test[:, i:i+view_mask] for i in range(0,
                                                                       num_observed_dimensions,
                                                                       view_mask)]
        views_test_unobserved = [y_test[:, i:i+view_mask] for i in range(num_observed_dimensions,
                                                                         num_output_dimensions,
                                                                         view_mask)]

    # Define ground truth depending on how test set was broken up.
    ground_truth = np.hstack(views_test_unobserved)

    # Define objectives.
    training_objective = mrd.objective
    predict_lower_bound, x_mean_test, x_covar_test, \
        predicted_means, predicted_covars = mrd.predict_missing_data(views_test=views_test_observed)
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
        print('\nTraining MRD..')
        for c in range(train_iter):
            s.run(opt_train)
            if (c % 100) == 0:
                print('  MRD opt iter {:5}: {}'.format(c, s.run(training_objective)))
        end_time = time()
        train_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  MRD: {}'.format(s.run(training_objective)))
        print('Time to optimise: {} s'.format(train_opt_time))

        # Get converged values as numpy arrays.
        ard_weights, noise_precisions, signal_variances, inducing_inputs = s.run((mrd.ard_weights,
                                                                                  mrd.noise_precisions,
                                                                                  mrd.signal_variances,
                                                                                  mrd.inducing_inputs))
        x_mean, x_covar = s.run(mrd.q_x)

        # Initialise prediction variables.
        s.run(tf.variables_initializer(var_list=predict_var_list))

        # Prediction optimisation loop.
        start_time = time()
        print('\nOptimising Predictions..')
        for c in range(predict_iter):
            s.run(opt_predict)
            if (c % 100) == 0:
                print('  MRD opt iter {:5}: {}'.format(c, s.run(predict_objective)))
        end_time = time()
        predict_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  MRD: {}'.format(s.run(predict_objective)))
        print('Time to optimise: {} s'.format(predict_opt_time))

        # Get converged values as numpy arrays.
        x_mean_test_np, x_covar_test_np, list_predicted_means, list_predicted_covars = s.run((x_mean_test,
                                                                                              x_covar_test,
                                                                                              predicted_means,
                                                                                              predicted_covars))

        # Convert lists to numpy arrays.
        predicted_means_np = np.hstack(list_predicted_means)
        predicted_covars_np = np.concatenate(list_predicted_covars, axis=0)

        # Calculate log-likelihood of ground truth with predicted posteriors.
        gt_log_likelihoods = [
            mvn_log_pdf(x=tf.transpose(tf.slice(ground_truth, begin=[0, du], size=[-1, 1])),
                        mean=tf.transpose(tf.slice(predicted_means_np, begin=[0, du], size=[-1, 1])),
                        covariance=tf.squeeze(tf.slice(predicted_covars_np, begin=[du, 0, 0], size=[1, -1, -1]),
                                              axis=0))
            for du in range(num_unobserved_dimensions)]
        gt_log_likelihoods_np = np.array(s.run(gt_log_likelihoods))
        gt_log_likelihood = np.sum(gt_log_likelihoods_np)

    # Save results. Converting lists to numpy arrays.
    np.savez(save_file, y_train=y_train, y_test_observed=y_test_observed, y_test_unobserved=y_test_unobserved,
             views_train=views_train, views_test_observed=views_test_observed,
             views_test_unobserved=views_test_unobserved, ard_weights=np.array(ard_weights),
             noise_precision=np.array(noise_precisions), signal_variance=np.array(signal_variances),
             x_u=np.array(inducing_inputs), x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
             x_mean_test=x_mean_test_np, x_covar_test=x_covar_test_np, predicted_mean=predicted_means_np,
             predicted_covar=predicted_covars_np, predict_opt_time=predict_opt_time,
             gt_log_likelihoods=gt_log_likelihoods_np, gt_log_likelihood=gt_log_likelihood)

    # Print results.
    print('\nMRD:')
    print('  Ground Truth Predicted Posterior Log-Likelihood: {}'.format(gt_log_likelihood))
    print('  Noise Precisions: {}'.format(np.squeeze(np.array(noise_precisions))))


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
             gt_log_likelihood=gt_log_likelihood)

    # Print results.
    print('\nGP-DP:')
    print('  Ground Truth Predicted Posterior Log-Likelihood: {}'.format(gt_log_likelihood))
    print('  Noise Precisions: {}'.format(np.squeeze(noise_precision)))


if __name__ == '__main__':

    # Define paths.
    absolute_path = [ap for ap in path if 'aistats_2019' in ap]
    data_path = absolute_path[-1] + '/test/data/horse_walk_data.npy'
    results_path = absolute_path[-1] + '/test/results/'

    # Optimisation variables.
    learning_rate = 0.05
    num_iter_train_bgplvm = 1500
    num_iter_predict_bgplvm = 1000
    num_iter_train_mrd = 1500
    num_iter_predict_mrd = 1000
    num_iter_train_gpdp = 2000
    num_iter_predict_gpdp = 1000

    # Different configurations
    # seeds = np.arange(1, 11, dtype=int)  # [1 - 10].
    seeds = np.arange(1, 6, dtype=int)  # [1 - 5].
    dp_masks = np.array([3], dtype=int)  # Only testing with 3 so group (x, y, z) coordinates together.
    percent_samples_observed = np.array([0.8, 0.9])
    percent_dimensions_observed = np.array([0.7, 0.8])  # np.array([0.7, 0.8, 0.9])

    # GP-LVM parameters
    percent_inducing = 0.55  # Used when training models for horse data.
    # Used default number of latent dimensions. So looks like 10.
    # percent_latent = 0.25

    # Additional DP-GP-LVM parameters.
    truncation_level = DP_DEFAULT_TRUNCATION_LEVEL  # Used default truncation level in original horse experiments.
    # truncation_level = 10

    # Total number of test scenarios.
    counter = 0.0
    total_num_tests = np.size(dp_masks) * np.size(seeds) * \
        np.size(percent_samples_observed) * np.size(percent_dimensions_observed)

    for mask in dp_masks:

        for seed in seeds:

            # Get randomly permuted, normalised data.
            config = 'walk_train_seed_{0}_mask_{1}'.format(seed, mask)
            np_file = results_path + 'horse_missing_data_{0}.npz'.format(config)
            if isfile(np_file):
                temp = np.load(np_file)
                normalised_data = temp['normalised_data']
                permuted_data = temp['permuted_data']
                original_data = temp['original_data']
            else:
                normalised_data, permuted_data, original_data = prepare_data(data_file_path=data_path,
                                                                             seed_val=seed,
                                                                             mask_size=mask)
                np.savez(np_file, normalised_data=normalised_data,
                         permuted_data=permuted_data,
                         original_data=original_data)

            for n in percent_samples_observed:

                for d in percent_dimensions_observed:

                    # Print current test scenario.
                    print('\n--------------------------------------------------------------------------------')
                    print('\nCurrent Test Scenario:')
                    print('  DP Mask:                          {}'.format(mask))
                    print('  Random seed:                      {}'.format(seed))
                    print('  Percent samples observed (n%):    {}'.format(n))
                    print('  Percent dimensions observed (d%): {}'.format(d))

                    # Separate into training and test data.
                    config_ext = 'n_percent_{0}_d_percent_{1}'.format(n, d)
                    np_file = results_path + 'horse_missing_data_{0}_{1}.npz'.format(config, config_ext)
                    if isfile(np_file):
                        temp = np.load(np_file)
                        training_data = temp['training_data']
                        test_data_observed = temp['test_data_observed']
                        test_data_unobserved_ground_truth = temp['test_data_unobserved_ground_truth']
                    else:
                        training_data, test_data_observed, test_data_unobserved_ground_truth = \
                            prepare_missing_data(data=normalised_data,
                                                 percent_samples_observe=n,
                                                 percent_dimensions_observe=d)
                        np.savez(np_file, training_data=training_data,
                                 test_data_observed=test_data_observed,
                                 test_data_unobserved_ground_truth=test_data_unobserved_ground_truth)

                    # Determine parameters for models. This ensures we do not use more latent dims or inducing points
                    # than possible for the model.
                    num_training_samples, num_output_dimensions = np.shape(training_data)
                    num_test_samples, num_observed_dimensions = np.shape(test_data_observed)
                    num_unobserved_dimensions = np.shape(test_data_unobserved_ground_truth)[1]
                    assert num_output_dimensions == num_observed_dimensions + num_unobserved_dimensions, \
                        'Missing data dimensions do not match.'
                    assert num_test_samples == np.shape(test_data_unobserved_ground_truth)[0], \
                        'Number of observations in test data and missing data must be the same.'

                    # This was how number of inducing points was calculated when running the original horse experiments.
                    num_inducing_points = int(np.ceil(percent_inducing * (num_training_samples + num_test_samples)))

                    # Used default number of latent dimensions in original horse experiments.
                    num_latent_dimensions = GP_LVM_DEFAULT_LATENT_DIMENSIONS
                    # num_latent_dimensions = int(np.ceil(percent_latent * num_observed_dimensions))
                    # while num_latent_dimensions > np.min((num_training_samples, num_output_dimensions)):
                    #     num_latent_dimensions -= 1

                    # Print info.
                    print('\nHorse Walking Data:')
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
                    bgplvm_file = results_path + 'bgplvm_horse_missing_data_{0}_{1}.npz'.format(config, config_ext)
                    if isfile(bgplvm_file):
                        print('\nAlready ran this configuration for BGPLVM.')
                    else:
                        # Reset default graph before building new model graph. This speeds up script.
                        tf.reset_default_graph()
                        # Build BGP-LVM graph and run it for current configuration.
                        run_bgplvm(y_train=training_data,
                                   y_test_observed=test_data_observed,
                                   y_test_unobserved=test_data_unobserved_ground_truth,
                                   num_latent_dimensions=num_latent_dimensions,
                                   num_inducing_points=num_inducing_points,
                                   train_iter=num_iter_train_bgplvm,
                                   predict_iter=num_iter_predict_bgplvm,
                                   learning_rate=learning_rate,
                                   save_file=bgplvm_file,
                                   seed_val=seed)

                    mrd_file = results_path + 'mrd_horse_missing_data_{0}_{1}.npz'.format(config, config_ext)
                    if isfile(mrd_file):
                        print('\nAlready ran this configuration for MRD.')
                    else:
                        # Reset default graph before building new model graph. This speeds up script.
                        tf.reset_default_graph()
                        # Build MRD model graph and run it for current configuration.
                        run_mrd(y_train=training_data,
                                y_test_observed=test_data_observed,
                                y_test_unobserved=test_data_unobserved_ground_truth,
                                num_latent_dimensions=num_latent_dimensions,
                                num_inducing_points=num_inducing_points,
                                view_mask=mask,
                                train_iter=num_iter_train_mrd,
                                predict_iter=num_iter_predict_mrd,
                                learning_rate=learning_rate,
                                save_file=mrd_file,
                                seed_val=seed)

                    gpdp_file = results_path + 'gpdp_horse_missing_data_{0}_{1}.npz'.format(config, config_ext)
                    if isfile(gpdp_file):
                        print('\nAlready ran this configuration for GP-DP.')
                    else:
                        # Reset default graph before building new model graph. This speeds up script.
                        tf.reset_default_graph()
                        # Build GP-DP model graph and run it for current configuration.
                        run_gpdp(y_train=training_data,
                                 y_test_observed=test_data_observed,
                                 y_test_unobserved=test_data_unobserved_ground_truth,
                                 num_latent_dimensions=num_latent_dimensions,
                                 num_inducing_points=num_inducing_points,
                                 truncation_level=truncation_level,
                                 dp_mask_size=mask,
                                 train_iter=num_iter_train_gpdp,
                                 predict_iter=num_iter_predict_gpdp,
                                 learning_rate=learning_rate,
                                 save_file=gpdp_file,
                                 seed_val=seed)

                    # Recalculate gt_log_likelihood using Cov(F*), not Cov(Y*). If model learns lots of noise,
                    #   then likelihood is high as it can explain all the data with just noise.
                    bgplvm_file_f_star = results_path + \
                        'bgplvm_horse_missing_data_f_star_{0}_{1}.npz'.format(config, config_ext)
                    if isfile(bgplvm_file_f_star):
                        print('\nAlready ran this calculation for BGPLVM.')
                    else:
                        print('\nRunning Cov(F*) calculation for BGPLVM.')

                        # Get predicted means, predicted covars, noise precision, and ground truth data.
                        results_data = np.load(bgplvm_file)
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
                                                                    batch_shape=[num_unobserved_dimensions],
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
                            # bgplvm_gt_log_likelihood = tf.reduce_sum(bgplvm_gt_log_likelihoods)
                            # bgplvm_gt_log_likelihood_np, bgplvm_gt_log_likelihoods_np = \
                            #     sess.run((bgplvm_gt_log_likelihood, bgplvm_gt_log_likelihoods))
                            bgplvm_gt_log_likelihoods_np = np.array(sess.run(bgplvm_gt_log_likelihoods))

                        bgplvm_gt_log_likelihood_np = np.sum(bgplvm_gt_log_likelihoods_np)

                        # Save results.
                        np.savez(bgplvm_file_f_star, y_test_unobserved=bgplvm_gt_unobserved,
                                 noise_precision=bgplvm_beta, predicted_mean=bgplvm_predicted_means,
                                 f_star_predicted_covar=f_star_predicted_covars_np,
                                 gt_log_likelihoods=bgplvm_gt_log_likelihoods_np,
                                 gt_log_likelihood=bgplvm_gt_log_likelihood_np)

                    # mrd_file_f_star = results_path + \
                    #     'mrd_horse_missing_data_f_star_{0}_{1}.npz'.format(config, config_ext)
                    # if isfile(mrd_file_f_star):
                    #     print('\nAlready ran this calculation for MRD.')
                    # else:
                    #     # Get predicted means, predicted covars, noise precision, and ground truth data.
                    #     results_data = np.load(mrd_file)
                    #     mrd_predicted_means = results_data['predicted_mean']
                    #     mrd_predicted_covars = results_data['predicted_covar']
                    #     mrd_beta = results_data['beta']
                    #     mrd_gt_unobserved = results_data['y_test_unobserved']
                    #
                    #     num_test_points, num_unobserved_dimensions = mrd_gt_unobserved.shape
                    #
                    #     # Reset default graph before building new model graph. This speeds up script.
                    #     tf.reset_default_graph()
                    #
                    #     # Don't bother doing this for MRD.

                    gpdp_file_f_star = results_path + \
                        'gpdp_horse_missing_data_f_star_{0}_{1}.npz'.format(config, config_ext)
                    if isfile(gpdp_file_f_star):
                        print('\nAlready ran this calculation for GP-DP.')
                    else:
                        print('\nRunning Cov(F*) calculation for GP-DP.')

                        # Get predicted means, predicted covars, noise precision, and ground truth data.
                        results_data = np.load(gpdp_file)
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
                                tf.eye(num_test_points, batch_shape=[num_unobserved_dimensions], dtype=TF_DTYPE)

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
                        np.savez(gpdp_file_f_star, y_test_unobserved=gpdp_gt_unobserved,
                                 noise_precision=gpdp_beta, predicted_mean=gpdp_predicted_means,
                                 f_star_predicted_covar=f_star_predicted_covars_np,
                                 gt_log_likelihoods=gpdp_gt_log_likelihoods_np,
                                 gt_log_likelihood=gpdp_gt_log_likelihood_np)

                    # Update number of tests run.
                    counter += 1.0
                    print('\n--------------------------------------------------------------------------------')
                    print('\nPercent tests completed: {}'.format(100.0 * counter / total_num_tests))

