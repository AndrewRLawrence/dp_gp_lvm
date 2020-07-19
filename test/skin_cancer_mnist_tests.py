"""
This module tests the various models using the skin cancer MNIST data set.
"""

from src.data_io.skin_cancer_mnist_reader import read_64d_luminance
from src.models.dp_gp_lvm import dp_gp_lvm
from src.models.gaussian_process import bayesian_gp_lvm as bgplvm, manifold_relevance_determination as mrd
from src.utils.constants import ResultKeys, RESULTS_FILE_NAME, DATA_PATH
from src.utils.types import NP_DTYPE
import src.visualisation.plotters as vis

from itertools import combinations
import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from time import time


if __name__ == '__main__':

    # Train model. Model/optimisation parameters. Using values from elros as larger ones use too much GPU memory.
    num_samples = 100
    num_inducing_points = 35
    num_latent_dimensions = 15
    truncation_level = 20
    train_iter = 5000
    learning_rate = 0.01

    # Read original data.
    image_data, labels = read_64d_luminance(DATA_PATH + 'skin_cancer_mnist/')

    # Loop through each unique pair of labels to create training data sets.
    for label_1, label_2 in combinations(np.unique(labels), 2):

        # Define file path for results.
        dataset_str = 'skin_cancer_mnist_{}_{}'.format(label_1, label_2)
        bgplvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm',
                                                       dataset=dataset_str)
        mrd_results_file = RESULTS_FILE_NAME.format(model='mrd',
                                                    dataset=dataset_str)
        # mrd_fully_independent_results_file = RESULTS_FILE_NAME.format(model='mrd_fully_independent',
        #                                                               dataset=dataset_str)
        gpdp_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm',
                                                     dataset=dataset_str)
        # gpdp_mask_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm_mask_64',
        #                                                   dataset=dataset_str)

        # Load randomised data for specific label pair if any of the model results do not exist.
        if any([not isfile(bgplvm_results_file),
                not isfile(mrd_results_file),
                # not isfile(mrd_fully_independent_results_file),
                # not isfile(gpdp_mask_results_file),
                not isfile(gpdp_results_file)]):

            two_conditions_data_file = DATA_PATH + \
                'skin_cancer_mnist/two_conditions_data_{}_{}.npy'.format(label_1, label_2)

            if isfile(two_conditions_data_file):
                # Read numy file of randomised data.
                two_groups_images = np.load(two_conditions_data_file)
            else:
                group_1_images = image_data[np.equal(labels, label_1)]
                group_2_images = image_data[np.equal(labels, label_2)]

                # Update number of samples if specific label does not have enough.
                num_samples = 100
                num_samples = min(num_samples, group_1_images.shape[0], group_2_images.shape[0])

                # Randomly permute observations.
                np.random.seed(1)  # Random seed.
                rand_indices_1 = np.random.permutation(group_1_images.shape[0])
                rand_indices_2 = np.random.permutation(group_2_images.shape[0])

                # Combine into [num_samples x 128] data.
                two_groups_images = np.hstack((group_1_images[rand_indices_1[:num_samples]],
                                               group_2_images[rand_indices_2[:num_samples]]))

                assert two_groups_images.shape[0] == num_samples, 'Number of subsampled observations is not equal.'
                assert two_groups_images.shape[1] == 128, \
                    'Number of pixels does not match expected number of 64 for each group.'

                # Save randomised data.
                np.save(two_conditions_data_file, two_groups_images)

            # Normalise data to zero mean and unit variance.
            scaler = StandardScaler()
            y_train = scaler.fit_transform(two_groups_images)
            num_samples, num_output_dimensions = y_train.shape

            # Print info.
            print('\nSkin Cancer MNIST Data:')
            print('  Total number of observations (N): {}'.format(num_samples))
            print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
            print('  Diagnosis labels: ({}, {})'.format(label_1, label_2))

            # Define instance of necessary model.
            if not isfile(bgplvm_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                model = bgplvm(y_train=y_train,
                               num_inducing_points=num_inducing_points,
                               num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining BGP-LVM:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  BGP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    print('Final iter {:5}:'.format(c))
                    print('  BGP-LVM: {}'.format(s.run(model_training_objective)))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(bgplvm_results_file, original_data=two_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time)

            if not isfile(mrd_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of MRD with known views.
                model = mrd(views_train=[y_train[:, i:i + 64] for i in range(0, 128, 64)],
                            num_inducing_points=num_inducing_points,
                            num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining MRD:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  MRD opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    print('Final iter {:5}:'.format(c))
                    print('  MRD: {}'.format(s.run(model_training_objective)))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(mrd_results_file, original_data=two_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time)

            # if not isfile(mrd_fully_independent_results_file):
            #     # Reset default graph before building new model graph. This speeds up script.
            #     tf.reset_default_graph()
            #     np.random.seed(1)  # Random seed.
            #     # Define instance of fully independent MRD.
            #     model = mrd(views_train=[y_train[:, i:i + 1] for i in range(0, 128, 1)],
            #                 num_inducing_points=num_inducing_points,
            #                 num_latent_dims=num_latent_dimensions)
            #
            #     model_training_objective = model.objective
            #     # Optimisation.
            #     model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            #         loss=model_training_objective)
            #
            #     with tf.Session() as s:
            #         # Initialise variables.
            #         s.run(tf.global_variables_initializer())
            #
            #         # Training optimisation loop.
            #         start_time = time()
            #         print('\nTraining F.I. MRD:')
            #         for c in range(train_iter):
            #             s.run(model_opt_train)
            #             if (c % 100) == 0:
            #                 print('  F.I. MRD opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
            #         end_time = time()
            #         train_opt_time = end_time - start_time
            #         print('Final iter {:5}:'.format(c))
            #         print('  F.I. MRD: {}'.format(s.run(model_training_objective)))
            #         print('Time to optimise: {} s'.format(train_opt_time))
            #
            #         # Get converged values as numpy arrays.
            #         ard_weights, noise_precision, signal_variance, inducing_input = \
            #             s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
            #         x_mean, x_covar = s.run(model.q_x)
            #
            #     # Save results.
            #     print('\nSaving results to .npz file.')
            #     np.savez(mrd_fully_independent_results_file, original_data=two_groups_images, y_train=y_train,
            #              ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
            #              x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time)

            if not isfile(gpdp_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of DP-GP-LVM. DP mask is default to 1.
                model = dp_gp_lvm(y_train=y_train,
                                  num_inducing_points=num_inducing_points,
                                  num_latent_dims=num_latent_dimensions,
                                  truncation_level=truncation_level)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining DP-GP-LVM:')
                    for c in range(train_iter):
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

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(gpdp_results_file, original_data=two_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                         gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                         q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time)

            # if not isfile(gpdp_mask_results_file):
            #     # Reset default graph before building new model graph. This speeds up script.
            #     tf.reset_default_graph()
            #     np.random.seed(1)  # Random seed.
            #     # Define instance of DP-GP-LVM with DP mask of 64.
            #     model = dp_gp_lvm(y_train=y_train,
            #                       num_inducing_points=num_inducing_points,
            #                       num_latent_dims=num_latent_dimensions,
            #                       truncation_level=truncation_level,
            #                       mask_size=64)
            #
            #     model_training_objective = model.objective
            #     # Optimisation.
            #     model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            #         loss=model_training_objective)
            #
            #     with tf.Session() as s:
            #         # Initialise variables.
            #         s.run(tf.global_variables_initializer())
            #
            #         # Training optimisation loop.
            #         start_time = time()
            #         print('\nTraining DP-GP-LVM:')
            #         for c in range(train_iter):
            #             s.run(model_opt_train)
            #             if (c % 100) == 0:
            #                 print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
            #         end_time = time()
            #         train_opt_time = end_time - start_time
            #         print('Final iter {:5}:'.format(c))
            #         print('  DP-GP-LVM: {}'.format(s.run(model_training_objective)))
            #         print('Time to optimise: {} s'.format(train_opt_time))
            #
            #         # Get converged values as numpy arrays.
            #         ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
            #             s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input,
            #                    model.assignments))
            #         x_mean, x_covar = s.run(model.q_x)
            #         w_1, w_2 = s.run(model.dp.q_alpha)
            #         gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)
            #
            #     # Save results.
            #     print('\nSaving results to .npz file.')
            #     np.savez(gpdp_mask_results_file, original_data=two_groups_images, y_train=y_train,
            #              ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
            #              x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
            #              gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
            #              q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time)

    # Loop through each unique triplet of labels to create training data sets.
    for label_1, label_2, label_3 in combinations(np.unique(labels), 3):

        # May need to update variational parameter sizes to avoid GPU memory overflow.
        num_inducing_points = 35
        num_latent_dimensions = 15
        truncation_level = 20

        # Define file path for results.
        dataset_str = 'skin_cancer_mnist_{}_{}_{}'.format(label_1, label_2, label_3)
        bgplvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm',
                                                       dataset=dataset_str)
        mrd_results_file = RESULTS_FILE_NAME.format(model='mrd',
                                                    dataset=dataset_str)
        gpdp_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm',
                                                     dataset=dataset_str)

        # Load randomised data for specific label pair if any of the model results do not exist.
        if any([not isfile(bgplvm_results_file),
                not isfile(mrd_results_file),
                not isfile(gpdp_results_file)]):

            three_conditions_data_file = DATA_PATH + \
                'skin_cancer_mnist/two_conditions_data_{}_{}_{}.npy'.format(label_1, label_2, label_3)

            if isfile(three_conditions_data_file):
                # Read numy file of randomised data.
                three_groups_images = np.load(three_conditions_data_file)
            else:
                group_1_images = image_data[np.equal(labels, label_1)]
                group_2_images = image_data[np.equal(labels, label_2)]
                group_3_images = image_data[np.equal(labels, label_3)]

                # Update number of samples if specific label does not have enough.
                num_samples = 100
                num_samples = min(num_samples, group_1_images.shape[0], group_2_images.shape[0],
                                  group_3_images.shape[0])

                # Randomly permute observations.
                np.random.seed(1)  # Random seed.
                rand_indices_1 = np.random.permutation(group_1_images.shape[0])
                rand_indices_2 = np.random.permutation(group_2_images.shape[0])
                rand_indices_3 = np.random.permutation(group_3_images.shape[0])

                # Combine into [num_samples x 192] data.
                three_groups_images = np.hstack((group_1_images[rand_indices_1[:num_samples]],
                                                 group_2_images[rand_indices_2[:num_samples]],
                                                 group_3_images[rand_indices_3[:num_samples]]))

                assert three_groups_images.shape[0] == num_samples, 'Number of subsampled observations is not equal.'
                assert three_groups_images.shape[1] == 192, \
                    'Number of pixels does not match expected number of 64 for each group.'

                # Save randomised data.
                np.save(three_conditions_data_file, three_groups_images)

            # Normalise data to zero mean and unit variance.
            scaler = StandardScaler()
            y_train = scaler.fit_transform(three_groups_images)
            num_samples, num_output_dimensions = y_train.shape

            # Print info.
            print('\nSkin Cancer MNIST Data:')
            print('  Total number of observations (N): {}'.format(num_samples))
            print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
            print('  Diagnosis labels: ({}, {}, {})'.format(label_1, label_2, label_3))

            # Define instance of necessary model.
            if not isfile(bgplvm_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                model = bgplvm(y_train=y_train,
                               num_inducing_points=num_inducing_points,
                               num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining BGP-LVM:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  BGP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  BGP-LVM: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(bgplvm_results_file, original_data=three_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                         final_cost=final_cost)

            if not isfile(mrd_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of MRD with known views.
                model = mrd(views_train=[y_train[:, i:i + 64] for i in range(0, 192, 64)],
                            num_inducing_points=num_inducing_points,
                            num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining MRD:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  MRD opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  MRD: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(mrd_results_file, original_data=three_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                         final_cost=final_cost)

            if not isfile(gpdp_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of DP-GP-LVM. DP mask is default to 1.
                model = dp_gp_lvm(y_train=y_train,
                                  num_inducing_points=num_inducing_points,
                                  num_latent_dims=num_latent_dimensions,
                                  truncation_level=truncation_level)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining DP-GP-LVM:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  DP-GP-LVM: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input,
                             model.assignments))
                    x_mean, x_covar = s.run(model.q_x)
                    w_1, w_2 = s.run(model.dp.q_alpha)
                    gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(gpdp_results_file, original_data=three_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                         gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                         q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time, final_cost=final_cost)

    # Loop through each unique set of six labels to create training data sets.
    for label_1, label_2, label_3, label_4, label_5, label_6 in combinations(np.unique(labels), 6):

        # May need to update variational parameter sizes to avoid GPU memory overflow.
        num_inducing_points = 21
        num_latent_dimensions = 25
        truncation_level = 20

        # Define file path for results.
        dataset_str = 'skin_cancer_mnist_{}_{}_{}_{}_{}_{}'.format(label_1, label_2, label_3, label_4, label_5, label_6)
        bgplvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm',
                                                       dataset=dataset_str)
        mrd_results_file = RESULTS_FILE_NAME.format(model='mrd',
                                                    dataset=dataset_str)
        gpdp_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm',
                                                     dataset=dataset_str)

        # Load randomised data for specific label pair if any of the model results do not exist.
        if any([not isfile(bgplvm_results_file),
                not isfile(mrd_results_file),
                not isfile(gpdp_results_file)]):

            six_conditions_data_file = DATA_PATH + \
                'skin_cancer_mnist/six_conditions_data_{}_{}_{}_{}_{}_{}.npy'.format(label_1, label_2, label_3, label_4,
                                                                                     label_5, label_6)

            if isfile(six_conditions_data_file):
                # Read numy file of randomised data.
                six_groups_images = np.load(six_conditions_data_file)
            else:
                group_1_images = image_data[np.equal(labels, label_1)]
                group_2_images = image_data[np.equal(labels, label_2)]
                group_3_images = image_data[np.equal(labels, label_3)]
                group_4_images = image_data[np.equal(labels, label_4)]
                group_5_images = image_data[np.equal(labels, label_5)]
                group_6_images = image_data[np.equal(labels, label_6)]

                # Update number of samples if specific label does not have enough.
                num_samples = 70
                num_samples = min(num_samples, group_1_images.shape[0], group_2_images.shape[0],
                                  group_3_images.shape[0], group_4_images.shape[0], group_5_images.shape[0],
                                  group_6_images.shape[0])

                # Randomly permute observations.
                np.random.seed(1)  # Random seed.
                rand_indices_1 = np.random.permutation(group_1_images.shape[0])
                rand_indices_2 = np.random.permutation(group_2_images.shape[0])
                rand_indices_3 = np.random.permutation(group_3_images.shape[0])
                rand_indices_4 = np.random.permutation(group_4_images.shape[0])
                rand_indices_5 = np.random.permutation(group_5_images.shape[0])
                rand_indices_6 = np.random.permutation(group_6_images.shape[0])

                # Combine into [num_samples x 192] data.
                six_groups_images = np.hstack((group_1_images[rand_indices_1[:num_samples]],
                                               group_2_images[rand_indices_2[:num_samples]],
                                               group_3_images[rand_indices_3[:num_samples]],
                                               group_4_images[rand_indices_4[:num_samples]],
                                               group_5_images[rand_indices_5[:num_samples]],
                                               group_6_images[rand_indices_6[:num_samples]]))

                assert six_groups_images.shape[0] == num_samples, 'Number of subsampled observations is not equal.'
                assert six_groups_images.shape[1] == 384, \
                    'Number of pixels does not match expected number of 64 for each group.'

                # Save randomised data.
                np.save(six_conditions_data_file, six_groups_images)

            # Normalise data to zero mean and unit variance.
            scaler = StandardScaler()
            y_train = scaler.fit_transform(six_groups_images)
            num_samples, num_output_dimensions = y_train.shape

            # Print info.
            print('\nSkin Cancer MNIST Data:')
            print('  Total number of observations (N): {}'.format(num_samples))
            print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
            print('  Diagnosis labels: ({}, {}, {}, {}, {}, {})'.format(label_1, label_2, label_3, label_4, label_5,
                                                                        label_6))

            # Define instance of necessary model.
            if not isfile(bgplvm_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                model = bgplvm(y_train=y_train,
                               num_inducing_points=num_inducing_points,
                               num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining BGP-LVM:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  BGP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  BGP-LVM: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(bgplvm_results_file, original_data=six_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                         final_cost=final_cost)

            if not isfile(mrd_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of MRD with known views.
                model = mrd(views_train=[y_train[:, i:i + 64] for i in range(0, 384, 64)],
                            num_inducing_points=num_inducing_points,
                            num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining MRD:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  MRD opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  MRD: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(mrd_results_file, original_data=six_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                         final_cost=final_cost)

            if not isfile(gpdp_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of DP-GP-LVM. DP mask is default to 1.
                model = dp_gp_lvm(y_train=y_train,
                                  num_inducing_points=num_inducing_points,
                                  num_latent_dims=num_latent_dimensions,
                                  truncation_level=truncation_level)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining DP-GP-LVM:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  DP-GP-LVM: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input,
                             model.assignments))
                    x_mean, x_covar = s.run(model.q_x)
                    w_1, w_2 = s.run(model.dp.q_alpha)
                    gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(gpdp_results_file, original_data=six_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                         gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                         q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time, final_cost=final_cost)

    # Loop through each unique set of all labels to create training data sets. This is only one set [0, 6].
    for label_1, label_2, label_3, label_4, label_5, label_6, label_7 in combinations(np.unique(labels), 7):

        # May need to update variational parameter sizes to avoid GPU memory overflow.
        num_inducing_points = 20
        num_latent_dimensions = 22
        truncation_level = 25

        # Define file path for results.
        dataset_str = 'skin_cancer_mnist_{}_{}_{}_{}_{}_{}_{}'.format(label_1, label_2, label_3, label_4, label_5,
                                                                      label_6, label_7)
        bgplvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm',
                                                       dataset=dataset_str)
        mrd_results_file = RESULTS_FILE_NAME.format(model='mrd',
                                                    dataset=dataset_str)
        gpdp_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm',
                                                     dataset=dataset_str)

        # Load randomised data for specific label pair if any of the model results do not exist.
        if any([not isfile(bgplvm_results_file),
                not isfile(mrd_results_file),
                not isfile(gpdp_results_file)]):

            seven_conditions_data_file = DATA_PATH + \
                'skin_cancer_mnist/seven_conditions_data_{}_{}_{}_{}_{}_{}_{}.npy'.format(label_1, label_2, label_3,
                                                                                          label_4, label_5, label_6,
                                                                                          label_7)

            if isfile(seven_conditions_data_file):
                # Read numy file of randomised data.
                seven_groups_images = np.load(seven_conditions_data_file)
            else:
                group_1_images = image_data[np.equal(labels, label_1)]
                group_2_images = image_data[np.equal(labels, label_2)]
                group_3_images = image_data[np.equal(labels, label_3)]
                group_4_images = image_data[np.equal(labels, label_4)]
                group_5_images = image_data[np.equal(labels, label_5)]
                group_6_images = image_data[np.equal(labels, label_6)]
                group_7_images = image_data[np.equal(labels, label_7)]

                # Update number of samples if specific label does not have enough.
                num_samples = 65
                num_samples = min(num_samples, group_1_images.shape[0], group_2_images.shape[0],
                                  group_3_images.shape[0], group_4_images.shape[0], group_5_images.shape[0],
                                  group_6_images.shape[0], group_7_images.shape[0])

                # Randomly permute observations.
                np.random.seed(1)  # Random seed.
                rand_indices_1 = np.random.permutation(group_1_images.shape[0])
                rand_indices_2 = np.random.permutation(group_2_images.shape[0])
                rand_indices_3 = np.random.permutation(group_3_images.shape[0])
                rand_indices_4 = np.random.permutation(group_4_images.shape[0])
                rand_indices_5 = np.random.permutation(group_5_images.shape[0])
                rand_indices_6 = np.random.permutation(group_6_images.shape[0])
                rand_indices_7 = np.random.permutation(group_7_images.shape[0])

                # Combine into [num_samples x 448] data.
                seven_groups_images = np.hstack((group_1_images[rand_indices_1[:num_samples]],
                                                 group_2_images[rand_indices_2[:num_samples]],
                                                 group_3_images[rand_indices_3[:num_samples]],
                                                 group_4_images[rand_indices_4[:num_samples]],
                                                 group_5_images[rand_indices_5[:num_samples]],
                                                 group_6_images[rand_indices_6[:num_samples]],
                                                 group_7_images[rand_indices_7[:num_samples]]))

                assert seven_groups_images.shape[0] == num_samples, 'Number of subsampled observations is not equal.'
                assert seven_groups_images.shape[1] == 448, \
                    'Number of pixels does not match expected number of 64 for each group.'

                # Save randomised data.
                np.save(seven_conditions_data_file, seven_groups_images)

            # Normalise data to zero mean and unit variance.
            scaler = StandardScaler()
            y_train = scaler.fit_transform(seven_groups_images)
            num_samples, num_output_dimensions = y_train.shape

            # Print info.
            print('\nSkin Cancer MNIST Data:')
            print('  Total number of observations (N): {}'.format(num_samples))
            print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
            print('  Diagnosis labels: ({}, {}, {}, {}, {}, {}, {})'.format(label_1, label_2, label_3, label_4, label_5,
                                                                            label_6, label_7))

            # Define instance of necessary model.
            if not isfile(bgplvm_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                model = bgplvm(y_train=y_train,
                               num_inducing_points=num_inducing_points,
                               num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining BGP-LVM:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  BGP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  BGP-LVM: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(bgplvm_results_file, original_data=seven_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                         final_cost=final_cost)

            if not isfile(mrd_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of MRD with known views.
                model = mrd(views_train=[y_train[:, i:i + 64] for i in range(0, 448, 64)],
                            num_inducing_points=num_inducing_points,
                            num_latent_dims=num_latent_dimensions)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining MRD:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  MRD opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  MRD: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
                    x_mean, x_covar = s.run(model.q_x)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(mrd_results_file, original_data=seven_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                         final_cost=final_cost)

            if not isfile(gpdp_results_file):
                # Reset default graph before building new model graph. This speeds up script.
                tf.reset_default_graph()
                np.random.seed(1)  # Random seed.
                # Define instance of DP-GP-LVM. DP mask is default to 1.
                model = dp_gp_lvm(y_train=y_train,
                                  num_inducing_points=num_inducing_points,
                                  num_latent_dims=num_latent_dimensions,
                                  truncation_level=truncation_level)

                model_training_objective = model.objective
                # Optimisation.
                model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss=model_training_objective)

                with tf.Session() as s:
                    # Initialise variables.
                    s.run(tf.global_variables_initializer())

                    # Training optimisation loop.
                    start_time = time()
                    print('\nTraining DP-GP-LVM:')
                    for c in range(train_iter):
                        s.run(model_opt_train)
                        if (c % 100) == 0:
                            print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
                    end_time = time()
                    train_opt_time = end_time - start_time
                    final_cost = s.run(model_training_objective)
                    print('Final iter {:5}:'.format(c))
                    print('  DP-GP-LVM: {}'.format(final_cost))
                    print('Time to optimise: {} s'.format(train_opt_time))

                    # Get converged values as numpy arrays.
                    ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
                        s.run(
                            (model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input,
                             model.assignments))
                    x_mean, x_covar = s.run(model.q_x)
                    w_1, w_2 = s.run(model.dp.q_alpha)
                    gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)

                # Save results.
                print('\nSaving results to .npz file.')
                np.savez(gpdp_results_file, original_data=seven_groups_images, y_train=y_train,
                         ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                         x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                         gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                         q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time, final_cost=final_cost)

