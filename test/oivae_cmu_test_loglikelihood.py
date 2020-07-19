"""This module tests our model against the results from the oi-VAE paper on CMU subject 7."""

from src.models.dp_gp_lvm import dp_gp_lvm
from src.utils.constants import RESULTS_FILE_NAME, DATA_PATH, PLOTS_PATH
from src.utils.types import get_training_variables, get_prediction_variables

import matplotlib.cm as color_map
import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from time import time


if __name__ == '__main__':

    # Optimisation variables.
    learning_rate = 0.025
    num_iter_train = 2500
    num_iter_test = 2000

    # Model hyperparameters.
    frame_subsamples = 3  # Subsample frames
    num_inducing_points = 50
    num_latent_dimensions = 16   # oi-VAE uses 4, 8, and 16.
    truncation_level = 15

    # Define arrays for all log-likelihood results.
    walk_ll = None
    brisk_ll = None

    # Read data.
    seqs = [0, 1, 2, 3, 5, 6, 7, 8, 9]
    # for i in range(10):
    for i in seqs:
        training_sequence = i + 1
        np_file = '07_0{}_joint_angles.npy'.format(training_sequence) if training_sequence < 10 \
            else '07_{}_joint_angles.npy'.format(training_sequence)
        cmu_data = np.load(DATA_PATH + 'cmu_mocap/' + np_file)

        total_num_frames = cmu_data.shape[0]
        num_output_dimensions = cmu_data.shape[1]

        # Subsample sequence
        training_data = cmu_data[::frame_subsamples]
        num_training_samples = training_data.shape[0]

        # Normalise data to zero mean and unit variance.
        scaler = StandardScaler()
        y_train = scaler.fit_transform(training_data)

        # Print info.
        print('\nCMU Subject 7 - Sequence {}:'.format(training_sequence))
        print('  Total number of observations (N): {}'.format(num_training_samples))
        print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
        print('  Total number of inducing points (M): {}'.format(num_inducing_points))
        print('  Total number of latent dimensions (Q): {}'.format(num_latent_dimensions))

        # Define test sequences.
        test_sequence_1 = 11
        test_sequence_2 = 12

        # Load test sequences.
        np_file = '07_0{}_joint_angles.npy'.format(test_sequence_1) if test_sequence_1 < 10 \
            else '07_{}_joint_angles.npy'.format(test_sequence_1)
        cmu_data = np.load(DATA_PATH + 'cmu_mocap/' + np_file)

        # Subsample test sequence.
        test_data_1 = cmu_data[::frame_subsamples]

        # Normalise data to zero mean and unit variance.
        y_test_1 = scaler.transform(test_data_1)

        # Load test sequences.
        np_file = '07_0{}_joint_angles.npy'.format(test_sequence_2) if test_sequence_2 < 10 \
            else '07_{}_joint_angles.npy'.format(test_sequence_2)
        cmu_data = np.load(DATA_PATH + 'cmu_mocap/' + np_file)

        # Subsample test sequence.
        test_data_2 = cmu_data[::frame_subsamples]

        # Normalise data to zero mean and unit variance.
        y_test_2 = scaler.transform(test_data_2)

        # Define file path for results.
        dataset_str = 'cmu_subject7_training_seq_{}_test_seqs_{}_{}'.format(
            training_sequence, test_sequence_1, test_sequence_2)
        dp_gp_lvm_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)

        # Define instance of necessary model.
        if not isfile(dp_gp_lvm_results_file):
            # Reset default graph before building new model graph. This speeds up script.
            tf.reset_default_graph()
            np.random.seed(1)  # Random seed.
            # Define instance of DP-GP-LVM.
            model = dp_gp_lvm(y_train=y_train,
                              num_inducing_points=num_inducing_points,
                              num_latent_dims=num_latent_dimensions,
                              truncation_level=truncation_level,
                              mask_size=1)  # Treat each observed dimension as independent.

            model_training_objective = model.objective
            predict_lower_bound_1, x_mean_test_1, x_covar_test_1, test_log_likelihood_1 = \
                model.predict_new_latent_variables(y_test=y_test_1)
            model_test_objective_1 = tf.negative(predict_lower_bound_1)
            predict_lower_bound_2, x_mean_test_2, x_covar_test_2, test_log_likelihood_2 = \
                model.predict_new_latent_variables(y_test=y_test_2)
            model_test_objective_2 = tf.negative(predict_lower_bound_2)

            # Optimisation.
            training_var_list = get_training_variables()
            test_var_list = get_prediction_variables()

            model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=model_training_objective, var_list=training_var_list)
            model_opt_test_1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=model_test_objective_1, var_list=test_var_list)
            model_opt_test_2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=model_test_objective_2, var_list=test_var_list)

            with tf.Session() as s:
                # Initialise variables.
                s.run(tf.variables_initializer(var_list=training_var_list))  # Initialise training variables first.
                s.run(tf.variables_initializer(var_list=test_var_list))  # Then initialise prediction variables.
                s.run(tf.global_variables_initializer())  # Finally initialise any remaining global variables.

                # Training optimisation loop.
                start_time = time()
                print('\nTraining DP-GP-LVM with sequence {}'.format(training_sequence))
                for c in range(num_iter_train):
                    s.run(model_opt_train)
                    if (c % 100) == 0:
                        print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                end_time = time()
                train_opt_time = end_time - start_time
                print('Final iter {:5}:'.format(c))
                print('  DP-GP-LVM: {}'.format(s.run(model_training_objective)))
                print('Time to optimise: {} s'.format(train_opt_time))

                # Get converged values as numpy arrays.
                ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
                    s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input,
                           model.assignments))
                x_mean, x_covar = s.run(model.q_x)
                w_1, w_2 = s.run(model.dp.q_alpha)
                gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)

                # Initialise test variables.
                s.run(tf.variables_initializer(var_list=test_var_list))

                # Test optimisation loop 1.
                start_time = time()
                print('\nOptimising Tests with sequence {}'.format(test_sequence_1))
                for c in range(num_iter_test):
                    s.run(model_opt_test_1)
                    if (c % 100) == 0:
                        print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_test_objective_1)))
                end_time = time()
                test_opt_time_1 = end_time - start_time
                print('Final iter {:5}:'.format(c))
                print('  DP-GP-LVM: {}'.format(s.run(model_test_objective_1)))
                print('Time to optimise: {} s'.format(test_opt_time_1))

                # Get converged values as numpy arrays.
                x_mean_test_1_np, x_covar_test_1_np, test_log_likelihood_1_np = s.run(
                    (x_mean_test_1, x_covar_test_1, test_log_likelihood_1))

                # Initialise test variables.
                s.run(tf.variables_initializer(var_list=test_var_list))

                # Test optimisation loop 2.
                start_time = time()
                print('\nOptimising Tests with sequence {}'.format(test_sequence_2))
                for c in range(num_iter_test):
                    s.run(model_opt_test_2)
                    if (c % 100) == 0:
                        print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_test_objective_2)))
                end_time = time()
                test_opt_time_2 = end_time - start_time
                print('Final iter {:5}:'.format(c))
                print('  DP-GP-LVM: {}'.format(s.run(model_test_objective_2)))
                print('Time to optimise: {} s'.format(test_opt_time_2))

                # Get converged values as numpy arrays.
                x_mean_test_2_np, x_covar_test_2_np, test_log_likelihood_2_np = s.run(
                    (x_mean_test_2, x_covar_test_2, test_log_likelihood_2))

            # Save results.
            print('\nSaving results to .npz file.')
            np.savez(dp_gp_lvm_results_file, original_training_data=training_data, y_train=y_train,
                     original_test_data_1=test_data_1, y_test_1=y_test_1,
                     original_test_data_2=test_data_2, y_test_2=y_test_2,
                     ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                     x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                     gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                     q_alpha_w1=w_1, q_alpha_w2=w_2,
                     x_mean_test_1=x_mean_test_1_np, x_covar_test_1=x_covar_test_1_np,
                     x_mean_test_2=x_mean_test_2_np, x_covar_test_2=x_covar_test_2_np,
                     train_opt_time=train_opt_time, test_opt_time_1=test_opt_time_1, test_opt_time_2=test_opt_time_2,
                     test_log_likelihood_1=test_log_likelihood_1_np, test_log_likelihood_2=test_log_likelihood_2_np)

        else:
            # Load results.
            results = np.load(dp_gp_lvm_results_file)

            # Print log-likelihoods:
            print('Training sequence: {}'.format(training_sequence))
            print('Test log-likelihood for sequence {} (walk): {}'.format(
                test_sequence_1, results['test_log_likelihood_1']))
            print('Test log-likelihood for sequence {} (brisk walk): {}'.format(
                test_sequence_2, results['test_log_likelihood_2']))

            # Add results to array of all log-likelihoods.
            if walk_ll is None:
                walk_ll = results['test_log_likelihood_1']
            else:
                walk_ll = np.append(walk_ll, results['test_log_likelihood_1'])
            if brisk_ll is None:
                brisk_ll = results['test_log_likelihood_2']
            else:
                brisk_ll = np.append(brisk_ll, results['test_log_likelihood_2'])

            # # Load number of dimensions per joint.
            # joint_dim_dict = np.load(DATA_PATH + 'cmu_mocap/' + '07_joint_dims.npy').item()
            # labels = list(joint_dim_dict.keys())[1:]  # Remove root joint.
            # ticks = np.array(list(joint_dim_dict.values()), dtype=int)
            #
            # # Save plots.
            # dp_gp_lvm_ard = results['ard_weights']
            # plot.imshow(np.sqrt(dp_gp_lvm_ard), interpolation='nearest', aspect='auto',
            #             extent=(0, num_latent_dimensions, num_output_dimensions, 0), origin='upper',
            #             cmap=color_map.Blues)
            # plot.title('Latent factorization for each joint')
            # plot.xlabel('X-Dimension')
            # plot.ylabel('')
            # ax = plot.gca()
            # ax.set_xticks(np.arange(0.5, num_latent_dimensions, 1))
            # ax.set_xticklabels([])
            # ax.set_yticks(np.cumsum(ticks), minor=False)
            # ax.set_yticklabels([], minor=False)
            # ax.set_yticks(np.cumsum(ticks) - 0.5 * ticks, minor=True)
            # ax.set_yticklabels(labels, minor=True)
            # # vis.plot_ard(np.sqrt(dp_gp_lvm_ard))
            # plot_filename = ''.join((PLOTS_PATH, 'cmu_subject7_training_seq', '_{}'.format(training_sequence)))
            # plot.savefig(plot_filename + '.pdf', bbox_inches='tight')

    # Print final results.
    sample_n = walk_ll.size
    sample_std_dev = np.std(walk_ll, ddof=1)
    print('\n--------------------------------------------------------------------------------')
    print('\nWalk 11:')
    print('  Number of test cases: {}'.format(sample_n))
    if sample_n > 1:
        print('  Mean:                 {}'.format(np.mean(walk_ll)))
        print('  Std Dev:              {}'.format(sample_std_dev))
        print('  Std Error:            {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))

    sample_n = brisk_ll.size
    sample_std_dev = np.std(brisk_ll, ddof=1)
    print('\nBrisk Walk 12:')
    print('  Number of test cases:   {}'.format(sample_n))
    if sample_n > 1:
        print('  Mean:                   {}'.format(np.mean(brisk_ll)))
        print('  Std Dev:                {}'.format(sample_std_dev))
        print('  Std Error:              {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))
