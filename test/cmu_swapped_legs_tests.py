"""
This module tests the various models using the concatenated CMU walking data set with normal and swapped legs.
"""

from src.models.dp_gp_lvm import dp_gp_lvm
from src.models.gaussian_process import bayesian_gp_lvm as bgplvm, manifold_relevance_determination as mrd
from src.utils.constants import ResultKeys, RESULTS_FILE_NAME, DATA_PATH
from src.utils.types import NP_DTYPE
import src.visualisation.plotters as vis

import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from time import time


if __name__ == '__main__':

    # Train model. Model/optimisation parameters. Using values from elros as larger ones use too much GPU memory.
    num_inducing_points = 30
    num_latent_dimensions = 20
    truncation_level = 20
    train_iter = 5000
    learning_rate = 0.01

    # Read data.
    cmu_data = np.load(DATA_PATH + 'cmu_mocap/35_01.npz')
    normal_swapped_motions = cmu_data['concatenated_motion_cols']

    # TODO: Maybe subsample frames as N is rather large now.

    # Normalise data to zero mean and unit variance.
    scaler = StandardScaler()
    y_train = scaler.fit_transform(normal_swapped_motions[::4])  # subsample frames.
    num_samples, num_output_dimensions = y_train.shape

    # Print info.
    print('\nCMU Walking 35 with Normal and Swapped Legs Motion:')
    print('  Total number of observations (N): {}'.format(num_samples))
    print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
    print('  Total number of inducing points (M): {}'.format(num_inducing_points))
    print('  Total number of latent dimensions (Q): {}'.format(num_latent_dimensions))

    # Define file path for results.
    dataset_str = 'cmu_walking_normal_swapped'
    bgplvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm', dataset=dataset_str)
    mrd_results_file = RESULTS_FILE_NAME.format(model='mrd', dataset=dataset_str)  # 93 dims per view.
    # 3 dims per view so keep 3d points together.
    mrd_fully_independent_results_file = RESULTS_FILE_NAME.format(model='mrd_fully_independent', dataset=dataset_str)
    gpdp_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)  # Keep 3d points together.
    gpdp_mask_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm_mask_93', dataset=dataset_str)

    # Define instance of necessary model.
    if not isfile(gpdp_results_file):
        # Reset default graph before building new model graph. This speeds up script.
        tf.reset_default_graph()
        np.random.seed(1)  # Random seed.
        # Define instance of DP-GP-LVM.
        model = dp_gp_lvm(y_train=y_train,
                          num_inducing_points=num_inducing_points,
                          num_latent_dims=num_latent_dimensions,
                          truncation_level=truncation_level,
                          mask_size=3)

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
        np.savez(gpdp_results_file, original_data=normal_swapped_motions, y_train=y_train,
                 ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                 x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                 gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                 q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time, final_cost=final_cost)

    if not isfile(gpdp_mask_results_file):
        # Reset default graph before building new model graph. This speeds up script.
        tf.reset_default_graph()
        np.random.seed(1)  # Random seed.
        # Define instance of DP-GP-LVM with DP mask of 93.
        model = dp_gp_lvm(y_train=y_train,
                          num_inducing_points=num_inducing_points,
                          num_latent_dims=num_latent_dimensions,
                          truncation_level=truncation_level,
                          mask_size=93)

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
        np.savez(gpdp_mask_results_file, original_data=normal_swapped_motions, y_train=y_train,
                 ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                 x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                 gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                 q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time, final_cost=final_cost)

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
            print('  BGP-LVM: {}'.format(s.run(model_training_objective)))
            print('Time to optimise: {} s'.format(train_opt_time))

            # Get converged values as numpy arrays.
            ard_weights, noise_precision, signal_variance, inducing_input = \
                s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
            x_mean, x_covar = s.run(model.q_x)

        # Save results.
        print('\nSaving results to .npz file.')
        np.savez(bgplvm_results_file, original_data=normal_swapped_motions, y_train=y_train,
                 ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                 x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                 final_cost=final_cost)

    if not isfile(mrd_results_file):
        # Reset default graph before building new model graph. This speeds up script.
        tf.reset_default_graph()
        np.random.seed(1)  # Random seed.
        # Define instance of MRD with known views.
        model = mrd(views_train=[y_train[:, i:i + 93] for i in range(0, 186, 93)],
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
            print('  MRD: {}'.format(s.run(model_training_objective)))
            print('Time to optimise: {} s'.format(train_opt_time))

            # Get converged values as numpy arrays.
            ard_weights, noise_precision, signal_variance, inducing_input = \
                s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
            x_mean, x_covar = s.run(model.q_x)

        # Save results.
        print('\nSaving results to .npz file.')
        np.savez(mrd_results_file, original_data=normal_swapped_motions, y_train=y_train,
                 ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                 x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                 final_cost=final_cost)

    if not isfile(mrd_fully_independent_results_file):
        # Reset default graph before building new model graph. This speeds up script.
        tf.reset_default_graph()
        np.random.seed(1)  # Random seed.
        # Define instance of fully independent MRD.
        model = mrd(views_train=[y_train[:, i:i + 3] for i in range(0, 186, 3)],
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
            print('\nTraining F.I. MRD:')
            for c in range(train_iter):
                s.run(model_opt_train)
                if (c % 100) == 0:
                    print('  F.I. MRD opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
            end_time = time()
            train_opt_time = end_time - start_time
            final_cost = s.run(model_training_objective)
            print('Final iter {:5}:'.format(c))
            print('  F.I. MRD: {}'.format(s.run(model_training_objective)))
            print('Time to optimise: {} s'.format(train_opt_time))

            # Get converged values as numpy arrays.
            ard_weights, noise_precision, signal_variance, inducing_input = \
                s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
            x_mean, x_covar = s.run(model.q_x)

        # Save results.
        print('\nSaving results to .npz file.')
        np.savez(mrd_fully_independent_results_file, original_data=normal_swapped_motions, y_train=y_train,
                 ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                 x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
                 final_cost=final_cost)
