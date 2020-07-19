"""
This module runs experiments for training the Bayesian GP-LVM, MRD, and DP-GP-LVM models with the Vicon Physical Action
Data Set (https://archive.ics.uci.edu/ml/datasets/Vicon+Physical+Action+Data+Set). This module also runs prediction
of missing data for the data set.
"""

from models.dp_gp_lvm import dp_gp_lvm
from utils.constants import GP_LVM_DEFAULT_LATENT_DIMENSIONS, DP_DEFAULT_TRUNCATION_LEVEL

import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler
from sys import path
import tensorflow as tf
from time import time


def animate_point_cloud(data, fig=None, is_interactive=True, axes_on=False, title=None):

    # Data is provided as [num_frames x num_markers * 3].
    num_frames = np.shape(data)[0]
    x_coordinates = data[:, 0::3]
    y_coordinates = data[:, 1::3]
    z_coordinates = data[:, 2::3]

    # Create new figure if None is provided or set current figure to one provided.
    if fig is None:
        fig = plot.figure()
    else:
        plot.figure(fig.number)

    # Turn on interactive if necessary.
    if is_interactive and not plot.isinteractive():
        plot.ion()
    assert (plot.isinteractive() == is_interactive), 'Matplotlib is not in the correct mode.'

    # Create new 3D axes in the provided figure.
    ax = Axes3D(fig)

    ax.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
    ax.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
    ax.set_zlim(np.min(z_coordinates), np.max(z_coordinates))

    if axes_on:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax.set_axis_off()

    current_scatter = None

    for frame in range(num_frames):
        if title is None:
            fig.suptitle('Frame: {0}'.format(frame))
        else:
            fig.suptitle(title + ' - Frame: {0}'.format(frame))
        if ax.has_data():
            current_scatter.remove()
        current_scatter = ax.scatter(x_coordinates[frame, :], y_coordinates[frame, :], z_coordinates[frame, :],
                                     c='k', marker='o')
        plot.draw()
        plot.pause(0.001)

    return fig


def prepare_data(original_data, seed_val=1, mask_size=3):
    """
    TODO
    :return:
    """

    num_samples, num_dimensions = original_data.shape

    # Randomly permute rows and columns.
    # Permute columns by grouping by mask size, i.e., keep 3D coordinates together if mask_size=3.
    np.random.seed(seed=seed_val)
    row_indices = np.random.permutation(num_samples)
    cols = np.arange(num_dimensions).reshape((np.int(num_dimensions / mask_size), mask_size))
    col_indices = cols[np.random.permutation(np.int(num_dimensions / mask_size)), :].flatten()
    permuted_data = original_data[row_indices, :][:, col_indices]

    # Normalise data to zero mean and unit variance for each column.
    scaler = StandardScaler()
    normalised_data = scaler.fit_transform(permuted_data)

    return normalised_data, permuted_data


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


def run_gpdp(y_train, num_latent_dimensions, num_inducing_points, truncation_level, dp_mask_size, train_iter,
             learning_rate, save_file, seed_val=1):
    """
    TODO
    :param y_train:
    :param num_latent_dimensions:
    :param num_inducing_points:
    :param truncation_level:
    :param dp_mask_size:
    :param train_iter:
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

    # Define objectives.
    training_objective = gpdp.objective

    # Optimisation.
    opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=training_objective)

    with tf.Session() as s:

        # Initialise variables.
        s.run(tf.global_variables_initializer())  # Initialise all global variables.

        # Training optimisation loop.
        start_time = time()
        print('\nTraining GP-DP..')
        for c in range(train_iter):
            s.run(opt_train)
            if (c % 50) == 0:
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

    # Save results.
    np.savez(save_file, y_train=y_train, ard_weights=ard_weights, noise_precision=noise_precision,
             signal_variance=signal_variance, x_u=inducing_input, assignments=assignments, x_mean=x_mean,
             x_covar=x_covar, gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
             train_opt_time=train_opt_time)

    # Print results.
    print('\nGP-DP:')
    print('  Noise Precisions: {}'.format(np.squeeze(noise_precision)))

    # Plot results.
    plot.figure()
    plot.imshow(ard_weights.T)
    plot.title('GP-DP ARD Weights')

    plot.figure()
    plot.imshow(assignments.T)
    plot.title('GP-DP Assignments')

    plot.show()


if __name__ == '__main__':

    # Define subsample constant.
    SUBSAMPLE_INTERVAL = 3

    # Optimisation variables.
    learning_rate = 0.02
    num_iter_train = 2000
    num_iter_predict = 1000

    # Define paths.
    absolute_path = [ap for ap in path if 'aistats_2019' in ap]
    data_path = absolute_path[-1] + '/test/data/cmu_mocap/'
    results_path = absolute_path[-1] + '/test/results/'

    # Data file.
    np_file = '35_01.npz'
    np_data = np.load(data_path + np_file)
    normal_walk = np_data['normal_motion']
    swapped_walk = np_data['legs_swapped_motion']
    concatenated_walk = np_data['concatenated_motion']

    # Subsample concatenated walk.
    subsample_walk = concatenated_walk[::SUBSAMPLE_INTERVAL, :]  # Remove some frames to make N smaller.
    num_frames = np.shape(subsample_walk)[0]
    print('\nNumber of Frames: {}'.format(num_frames))

    # Animate data to see what it looks like.
    # animate_point_cloud(subsample_walk, title='Subject {0} {1}'.format(35, 'Walking'))

    # Normalise data to zero mean and unit variance per column.
    scaler = StandardScaler()
    normalised_data = scaler.fit_transform(subsample_walk)

    # Train DP-GP-LVM model.
    num_latent_dimensions = GP_LVM_DEFAULT_LATENT_DIMENSIONS
    # num_inducing_points = np.int(0.35 * num_frames)
    num_inducing_points = 40
    truncation_level = 10  # DP_DEFAULT_TRUNCATION_LEVEL

    gpdp_file = results_path + 'gpdp_walk_legs_swapped_mocap_gpu_run_3.npz'
    if isfile(gpdp_file):
        print('\nAlready ran this configuration for GP-DP.')
    else:
        # Reset default graph before building new model graph. This speeds up script.
        tf.reset_default_graph()
        # Build GP-DP model graph and run it for current configuration.
        run_gpdp(y_train=normalised_data,
                 num_latent_dimensions=num_latent_dimensions,
                 num_inducing_points=num_inducing_points,
                 truncation_level=truncation_level,
                 dp_mask_size=3,
                 train_iter=num_iter_train,
                 learning_rate=learning_rate,
                 save_file=gpdp_file,
                 seed_val=1)
