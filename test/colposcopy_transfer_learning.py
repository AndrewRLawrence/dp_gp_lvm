"""
This module tests the predictive posterior for missing data to see how well DP-GP-LVM works with predicting expert
labels for colposcopy images.
"""

from src.data_io.colposcopy_reader import read_hinselmann_modality, read_green_modality, read_schiller_modality
from src.distributions.normal import mvn_log_pdf
from src.models.dp_gp_lvm import dp_gp_lvm
from src.models.gaussian_process import bayesian_gp_lvm
from src.utils.constants import RESULTS_FILE_NAME
from src.utils.types import get_training_variables, get_prediction_variables
import src.visualisation.plotters as vis

from itertools import permutations
import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from time import time


def run_bgplvm(y_train, num_latent_dimensions, num_inducing_points, train_iter, learning_rate, save_file, seed_val=1):
    """
    TODO
    :param y_train:
    :param num_latent_dimensions:
    :param num_inducing_points:
    :param train_iter:
    :param learning_rate:
    :param save_file:
    :param seed_val:
    :return:
    """
    raise NotImplementedError


def run_bgplvm_transfer_learning(y_train, y_test_observed):
    raise NotImplementedError


def run_dp_gp_lvm(y_train, y_test_observed, y_test_unobserved, num_latent_dimensions, num_inducing_points,
                  truncation_level, dp_mask_size, train_iter, predict_iter, learning_rate, save_file, seed_val=1):
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
        print('\nTraining DP-GP-LVM..')
        for c in range(train_iter):
            s.run(opt_train)
            if (c % 100) == 0:
                print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(training_objective)))
        end_time = time()
        train_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  DP-GP-LVM: {}'.format(s.run(training_objective)))
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
                print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(predict_objective)))
        end_time = time()
        predict_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  DP-GP-LVM: {}'.format(s.run(predict_objective)))
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
    print('\nDP-GP-LVM:')
    print('  Ground Truth Predicted Posterior Log-Likelihood: {}'.format(gt_log_likelihood))
    print('  Noise Precisions: {}'.format(np.squeeze(noise_precision)))


if __name__ == '__main__':

    # Optimisation variables.
    learning_rate = 0.05  # 0.01
    num_iter_train = 2500
    num_iter_predict = 2500

    # Model hyperparameters.
    num_inducing_points = 50
    num_latent_dimensions = 15
    truncation_level = 20

    # Read each modality.
    hinselmann_attributes, hinselmann_assessments = read_hinselmann_modality()
    hinselmann_data = np.hstack((hinselmann_attributes, hinselmann_assessments))
    green_attributes, green_assessments = read_green_modality()
    green_data = np.hstack((green_attributes, green_assessments))
    schiller_attributes, schiller_assessments = read_schiller_modality()
    schiller_data = np.hstack((schiller_attributes, schiller_assessments))

    # Normalise data to zero mean and unit variance.
    scaler = StandardScaler()
    normalized_hinselmann_data = scaler.fit_transform(hinselmann_data)
    normalized_green_data = scaler.fit_transform(green_data)
    normalized_schiller_data = scaler.fit_transform(schiller_data)

    # TEMP: Test with Bayesian GP-LVM and DP-GP-LVM.
    # Set seed.
    np.random.seed(seed=1)
    # # Train Bayesian GP-LVM.
    # model = bayesian_gp_lvm(y_train=normalized_hinselmann_data,
    #                         num_latent_dims=num_latent_dimensions,
    #                         num_inducing_points=num_inducing_points)
    # Define instance of DP-GP-LVM .
    model = dp_gp_lvm(y_train=normalized_hinselmann_data,
                      num_latent_dims=num_latent_dimensions,
                      num_inducing_points=num_inducing_points,
                      truncation_level=truncation_level,
                      mask_size=1)

    # Define objectives.
    training_objective = model.objective

    # Optimisation.
    training_var_list = get_training_variables()

    opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=training_objective,
                                                                             var_list=training_var_list)

    with tf.Session() as s:

        # Initialise variables.
        s.run(tf.variables_initializer(var_list=training_var_list))  # Initialise training variables first.
        s.run(tf.global_variables_initializer())  # Finally initialise any remaining global variables such as opt ones.

        # Training optimisation loop.
        start_time = time()
        print('\nTraining..')
        for c in range(num_iter_train):
            s.run(opt_train)
            if (c % 100) == 0:
                print('  opt iter {:5}: {}'.format(c, s.run(training_objective)))
        end_time = time()
        train_opt_time = end_time - start_time
        print('Final iter {:5}: {}'.format(c, s.run(training_objective)))
        print('Time to optimise: {} s'.format(train_opt_time))

        # Get converged values as numpy arrays.
        ard_weights, noise_precision, signal_variance, inducing_input = \
            s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
        x_mean, x_covar = s.run(model.q_x)
        # gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)
        print('SNR: {} (dB)'.format(10.0 * np.log10(signal_variance * noise_precision)))

    vis.plot_ard(np.sqrt(ard_weights))
    plot.colorbar()
    plot.show()

    quit(0)

    # Define all possible {source, target} pairs from list of modalities.
    modalities = ['h', 'g', 's']
    tasks = permutations(modalities, r=2)

    # Loop through each task.
    for (source, target) in tasks:

        # Define file path for results.
        dataset_str = 'colposcopy_{}_{}'.format(source, target)
        dp_gp_lvm_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)

        # Set source training data and normalise to zero mean and unit variance.
        scaler = StandardScaler()
        if source == 'h':
            training_data = scaler.fit_transform(hinselmann_data)
        elif source == 'g':
            training_data = scaler.fit_transform(green_data)
        elif source == 's':
            training_data = scaler.fit_transform(schiller_data)
        else:
            raise NameError

        # Set target test data.
        if target == 'h':
            test_data = scaler.transform(hinselmann_data)
        elif target == 'g':
            test_data = scaler.transform(green_data)
        elif target == 's':
            test_data = scaler.transform(schiller_data)
        else:
            raise NameError

        num_training_samples, num_output_dimensions = training_data.shape
        num_test_samples, num_test_dimensions = test_data.shape
        assert num_output_dimensions == num_test_dimensions

        # Predict all assessments.
        test_data_observed = test_data[:, :62]
        test_data_unobserved_ground_truth = test_data[:, 62:]
        num_observed_dimensions = test_data_observed.shape[1]
        num_unobserved_dimensions = test_data_unobserved_ground_truth.shape[1]
        assert num_observed_dimensions + num_unobserved_dimensions == num_output_dimensions

        # Print info.
        print('\nColposcopy Data:')
        print('Source Data - {}:'.format(source))
        print('  Number of training samples: {}'.format(num_training_samples))
        print('  Number of training dimensions: {}'.format(num_output_dimensions))
        print('Target Data - {}:'.format(target))
        print('  Number of test samples: {}'.format(num_test_samples))
        print('  Number of provided/observed dimensions: {}'.format(num_observed_dimensions))
        print('  Number of missing/unobserved dimensions: {}'.format(num_unobserved_dimensions))

        # Run DP-GP-LVM.
        # # Reset default graph before building new model graph. This speeds up script.
        tf.reset_default_graph()
        # Build DP-GP-LVM graph and run it for current configuration.
        run_dp_gp_lvm(y_train=training_data,
                      y_test_observed=test_data_observed,
                      y_test_unobserved=test_data_unobserved_ground_truth,
                      num_latent_dimensions=num_latent_dimensions,
                      num_inducing_points=num_inducing_points,
                      truncation_level=truncation_level,
                      dp_mask_size=1,
                      train_iter=num_iter_train,
                      predict_iter=num_iter_predict,
                      learning_rate=learning_rate,
                      save_file=dp_gp_lvm_results_file,
                      seed_val=1)



    # # Visualize results if they exist; otherwise, train model.
    # if isfile(dp_gp_lvm_results_file):
    #     results = np.load(dp_gp_lvm_results_file)
    #
    #     ground_truth = np.array(results['y_test_unobserved'] >= 0.0).astype(int)
    #     predicted_mean_assessments = np.array((results['predicted_mean'] -
    #                                            np.mean(results['predicted_mean'])) >= 0.0).astype(int)
    #
    #     confusion_mat = confusion_matrix(ground_truth, predicted_mean_assessments)
    #
    #     vis.plot_phi_matrix(ground_truth, add_labels=False)
    #     plot.title('Ground Truth')
    #     vis.plot_phi_matrix(predicted_mean_assessments, add_labels=False)
    #     plot.title('Predicted Assessments')
    #
    #     plot.figure()
    #     plot.imshow(confusion_mat)
    #
    #     plot.show()
    # else:
    #     # Optimisation variables.
    #     learning_rate = 0.05   # 0.01
    #     num_iter_train = 2500
    #     num_iter_predict = 2500
    #
    #     # Model hyperparameters.
    #     num_inducing_points = 50
    #     num_latent_dimensions = 15
    #     truncation_level = 20
    #
    #     # Read each modality.
    #     hinselmann_attributes, hinselmann_assessments = read_hinselmann_modality()
    #     green_attributes, green_assessments = read_green_modality()
    #     schiller_attributes, schiller_assessments = read_schiller_modality()
    #
    #     # Set source and target modes.
    #     training_data = np.hstack((hinselmann_attributes, hinselmann_assessments))
    #     training_data = np.vstack((training_data, np.hstack((green_attributes[:49, :], green_assessments[:49, :]))))
    #     test_data = np.hstack((green_attributes[49:, :], green_assessments[49:, :]))
    #
    #     # Normalise data to zero mean and unit variance.
    #     scaler = StandardScaler()
    #     normalized_training_data = scaler.fit_transform(training_data)
    #     normalized_test_data = scaler.fit_transform(test_data)
    #     num_training_samples, num_output_dimensions = normalized_training_data.shape
    #     num_test_samples, num_test_dimensions = normalized_test_data.shape
    #     assert num_output_dimensions == num_test_dimensions
    #
    #     # # Predict all assessments for green.
    #     # test_data_observed = green_attributes
    #     # test_data_unobserved_ground_truth = green_assessments
    #
    #     # Predict consensus for green.
    #     normalized_test_data_observed = normalized_test_data[:, :-1]
    #     normalized_test_data_unobserved_ground_truth = normalized_test_data[:, -1, np.newaxis]  # Ensure it is still 2D array.
    #     num_observed_dimensions = normalized_test_data_observed.shape[1]
    #     num_unobserved_dimensions = normalized_test_data_unobserved_ground_truth.shape[1]
    #     assert num_observed_dimensions + num_unobserved_dimensions == num_output_dimensions
    #
    #     # Print info.
    #     print('\nColposcopy Data:')
    #     print('Training Data - Hinselmann Modality:')
    #     print('  Number of training samples: {}'.format(num_training_samples))
    #     print('  Number of training dimensions: {}'.format(num_output_dimensions))
    #     print('Missing Data - Green Modality:')
    #     print('  Number of test samples: {}'.format(num_test_samples))
    #     print('  Number of provided/observed dimensions: {}'.format(num_observed_dimensions))
    #     print('  Number of missing/unobserved dimensions: {}'.format(num_unobserved_dimensions))
    #
    #     # Run DP-GP-LVM.
    #     # # Reset default graph before building new model graph. This speeds up script.
    #     # tf.reset_default_graph()
    #     # Build DP-GP-LVM graph and run it for current configuration.
    #     run_dp_gp_lvm(y_train=normalized_training_data,
    #                   y_test_observed=normalized_test_data_observed,
    #                   y_test_unobserved=normalized_test_data_unobserved_ground_truth,
    #                   num_latent_dimensions=num_latent_dimensions,
    #                   num_inducing_points=num_inducing_points,
    #                   truncation_level=truncation_level,
    #                   dp_mask_size=1,
    #                   train_iter=num_iter_train,
    #                   predict_iter=num_iter_predict,
    #                   learning_rate=learning_rate,
    #                   save_file=dp_gp_lvm_results_file,
    #                   seed_val=1)
