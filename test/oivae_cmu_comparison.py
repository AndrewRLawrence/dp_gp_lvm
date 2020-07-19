"""This module tests our model against the results from the oi-VAE paper on CMU subject 7."""

from src.models.dp_gp_lvm import dp_gp_lvm
from src.utils.constants import RESULTS_FILE_NAME, DATA_PATH
from src.utils.types import NP_DTYPE
import src.visualisation.plotters as vis

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
    num_iter_predict = 2000

    # Model hyperparameters.
    num_training_samples = 150
    num_inducing_points = 40
    num_latent_dimensions = 16   # oi-VAE uses 4, 8, and 16.
    truncation_level = 15

    # Read data.
    training_data = None
    for i in range(10):
        sequence = i + 1
        np_file = '07_0{}_joint_angles.npy'.format(sequence) if sequence < 10 \
            else '07_{}_joint_angles.npy'.format(sequence)
        cmu_data = np.load(DATA_PATH + 'cmu_mocap/' + np_file)
        if training_data is None:
            training_data = cmu_data
        else:
            training_data = np.vstack((training_data, cmu_data))
    total_num_frames = training_data.shape[0]
    num_output_dimensions = training_data.shape[1]

    # Randomly sample 200 frames and normalise data to zero mean and unit variance.
    np.random.seed(seed=1)  # Set seed.
    training_indices = np.random.choice(training_data.shape[0], size=num_training_samples, replace=False)
    scaler = StandardScaler()
    y_train = scaler.fit_transform(training_data[training_indices, 6:])  # Remove first 6 dimensions to ignore root.

    # Print info.
    print('\nCMU Subject 7 - Sequences 1-10:')
    print('  Total number of observations (N): {}'.format(num_training_samples))
    print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
    print('  Total number of inducing points (M): {}'.format(num_inducing_points))
    print('  Total number of latent dimensions (Q): {}'.format(num_latent_dimensions))

    # Define file path for results.
    dataset_str = 'cmu_subject7_joint_angles'
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
        # Optimisation.
        model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            loss=model_training_objective)

        with tf.Session() as s:
            # Initialise variables.
            s.run(tf.global_variables_initializer())

            # Training optimisation loop.
            start_time = time()
            print('\nTraining DP-GP-LVM:')
            for c in range(num_iter_train):
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
        np.savez(dp_gp_lvm_results_file, original_data=training_data, y_train=y_train,
                 ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                 x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                 gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                 q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time, final_cost=final_cost)

    else:
        # Load results.
        results = np.load(dp_gp_lvm_results_file)

        # Load number of dimensions per joint.
        joint_dim_dict = np.load(DATA_PATH + 'cmu_mocap/' + '07_joint_dims.npy').item()
        labels = list(joint_dim_dict.keys())
        ticks = np.array(list(joint_dim_dict.values()), dtype=int)
        # labels = list(joint_dim_dict.keys())[1:]  # Remove root joint.
        # ticks = np.array(list(joint_dim_dict.values()), dtype=int)[1:]  # Remove root joint.

        # Plot latent spaces.
        dp_gp_lvm_ard = results['ard_weights']
        # dp_gp_lvm_ard[dp_gp_lvm_ard < 0.1] = 0.0
        dp_gp_lvm_ard = np.sqrt(dp_gp_lvm_ard)

        # plot.figure()
        # # plot.imshow(dp_gp_lvm_ard, interpolation='nearest', aspect='auto',
        # #             extent=(0, num_latent_dimensions, num_output_dimensions, 0), origin='upper')
        # plot.imshow(dp_gp_lvm_ard, interpolation='nearest', aspect='auto',
        #             extent=(0, num_latent_dimensions, num_output_dimensions, 0), origin='upper', cmap=color_map.Blues)
        # # plot.colorbar()
        # plot.title('Latent factorization for each joint')
        # plot.xlabel('X-Dimension')
        # plot.ylabel('')
        # ax = plot.gca()
        # # ax.set_xticks(np.arange(0.5, num_latent_dimensions, 1))
        # ax.set_xticks(np.arange(num_latent_dimensions))
        # ax.set_xticklabels([])
        # ax.set_yticks(np.cumsum(ticks), minor=False)
        # ax.set_yticklabels([], minor=False)
        # ax.set_yticks(np.cumsum(ticks) - 0.5 * ticks, minor=True)
        # ax.set_yticklabels(labels, minor=True)
        # plot.show()

        # Sum sort.
        index = np.argsort(np.sum(dp_gp_lvm_ard, axis=0))
        plot.figure(figsize=(10,5))
        plot.imshow(np.transpose(dp_gp_lvm_ard[:, index[::-1]]), interpolation='nearest', aspect='auto',
                    extent=(0, num_output_dimensions, num_latent_dimensions, 0), origin='upper', cmap=color_map.Blues)
        plot.ylabel('X', rotation='horizontal')
        plot.xlabel('')
        ax = plot.gca()
        ax.set_yticks(np.arange(num_latent_dimensions))
        ax.set_yticklabels([])
        ax.set_xticks(np.cumsum(ticks), minor=False)
        ax.set_xticklabels([], minor=False)
        ax.set_xticks(np.cumsum(ticks) - 0.5 * ticks, minor=True)
        ax.set_xticklabels(labels, minor=True, rotation='vertical', fontweight='bold')
        plot.savefig('cmu_7_sum_sort.pdf', bbox_inches='tight')
        plot.show()

        # # Largest sum.
        # index = np.argsort(np.sum(dp_gp_lvm_ard, axis=1))[::-1]
        # index = np.argsort(dp_gp_lvm_ard[index[0], :])[::-1]
        # plot.figure(figsize=(7,10))
        # plot.imshow(dp_gp_lvm_ard[:, index], interpolation='nearest', aspect='auto',
        #             extent=(0, num_latent_dimensions, num_output_dimensions, 0), origin='upper', cmap=color_map.Blues)
        # plot.title('Latent factorization for each joint')
        # plot.xlabel('X-Dimension')
        # plot.ylabel('')
        # ax = plot.gca()
        # ax.set_xticks(np.arange(num_latent_dimensions))
        # ax.set_xticklabels([])
        # ax.set_yticks(np.cumsum(ticks), minor=False)
        # ax.set_yticklabels([], minor=False)
        # ax.set_yticks(np.cumsum(ticks) - 0.5 * ticks, minor=True)
        # ax.set_yticklabels(labels, minor=True)
        # plot.savefig('cmu_7_largest_sum.pdf', bbox_inches='tight')
        # # plot.show()
        #
        # # Variance
        # index = np.argsort(np.var(dp_gp_lvm_ard, axis=0))[::-1]
        # plot.figure(figsize=(7,10))
        # plot.imshow(dp_gp_lvm_ard[:, index], interpolation='nearest', aspect='auto',
        #             extent=(0, num_latent_dimensions, num_output_dimensions, 0), origin='upper', cmap=color_map.Blues)
        # plot.title('Latent factorization for each joint')
        # plot.xlabel('X-Dimension')
        # plot.ylabel('')
        # ax = plot.gca()
        # ax.set_xticks(np.arange(num_latent_dimensions))
        # ax.set_xticklabels([])
        # ax.set_yticks(np.cumsum(ticks), minor=False)
        # ax.set_yticklabels([], minor=False)
        # ax.set_yticks(np.cumsum(ticks) - 0.5 * ticks, minor=True)
        # ax.set_yticklabels(labels, minor=True)
        # plot.savefig('cmu_7_variance_sort.pdf', bbox_inches='tight')
        # plot.show()

    # Using cartesian coordinates.
    # # Read data.
    # training_data = None
    # for i in range(10):
    #     sequence = i + 1
    #     np_file = '07_0{}.npy'.format(sequence) if sequence < 10 else '07_{}.npy'.format(sequence)
    #     cmu_data = np.load(DATA_PATH + 'cmu_mocap/' + np_file)
    #     if training_data is None:
    #         training_data = cmu_data
    #     else:
    #         training_data = np.vstack((training_data, cmu_data))
    # total_num_frames = training_data.shape[0]
    # num_output_dimensions = training_data.shape[1]
    #
    # # Randomly sample 200 frames and normalise data to zero mean and unit variance.
    # np.random.seed(seed=1)  # Set seed.
    # training_indices = np.random.choice(training_data.shape[0], size=num_training_samples, replace=False)
    # scaler = StandardScaler()
    # y_train = scaler.fit_transform(training_data[training_indices, :])
    #
    # # Print info.
    # print('\nCMU Subject 7 - Sequences 1-10:')
    # print('  Total number of observations (N): {}'.format(num_training_samples))
    # print('  Total number of output dimensions (D): {}'.format(num_output_dimensions))
    # print('  Total number of inducing points (M): {}'.format(num_inducing_points))
    # print('  Total number of latent dimensions (Q): {}'.format(num_latent_dimensions))
    #
    # # Define file path for results.
    # dataset_str = 'cmu_subject7'
    # dp_gp_lvm_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)  # Keep 3d points together
    #
    # # Define instance of necessary model.
    # if not isfile(dp_gp_lvm_results_file):
    #     # Reset default graph before building new model graph. This speeds up script.
    #     tf.reset_default_graph()
    #     np.random.seed(1)  # Random seed.
    #     # Define instance of DP-GP-LVM.
    #     model = dp_gp_lvm(y_train=y_train,
    #                       num_inducing_points=num_inducing_points,
    #                       num_latent_dims=num_latent_dimensions,
    #                       truncation_level=truncation_level,
    #                       mask_size=3)
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
    #         for c in range(num_iter_train):
    #             s.run(model_opt_train)
    #             if (c % 100) == 0:
    #                 print('  DP-GP-LVM opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
    #         end_time = time()
    #         train_opt_time = end_time - start_time
    #         final_cost = s.run(model_training_objective)
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
    #     np.savez(dp_gp_lvm_results_file, original_data=training_data, y_train=y_train,
    #              ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
    #              x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
    #              gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
    #              q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time, final_cost=final_cost)
    #
    # else:
    #     # Load results.
    #     results = np.load(dp_gp_lvm_results_file)
    #
    #     # Plot latent spaces.
    #     dp_gp_lvm_ard = results['ard_weights']
    #
    #     plot.figure()
    #     # plot.imshow(np.sqrt(dp_gp_lvm_ard).T, interpolation='nearest', aspect='auto',
    #     #             extent=(0, num_output_dimensions, num_latent_dimensions, 0), origin='upper')
    #     plot.imshow(np.sqrt(dp_gp_lvm_ard), interpolation='nearest', aspect='auto',
    #                 extent=(0, num_latent_dimensions, num_output_dimensions, 0), origin='upper')
    #     plot.colorbar()
    #     plot.title('ARD Weights')
    #     plot.show()
