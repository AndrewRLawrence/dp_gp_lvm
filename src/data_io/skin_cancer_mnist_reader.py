"""
This module reads and parses the skin cancer MNIST data set captured in CSV files. It does not read the image files.
"""

from src.models.dp_gp_lvm import dp_gp_lvm
from src.models.gaussian_process import bayesian_gp_lvm as bgplvm, manifold_relevance_determination as mrd
from src.utils.constants import ResultKeys, RESULTS_FILE_NAME, DATA_PATH
from src.utils.types import NP_DTYPE
import src.visualisation.plotters as vis

from csv import reader as csv_reader
import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from time import time


def read_hmnist_csv(file_path):
    """
    TODO
    :param file_path:
    :return:
    """

    assert isfile(file_path), 'Specified file does not exist.'

    with open(file_path, newline='') as csv_file:
        reader = csv_reader(csv_file)
        counter = 0

        observed_data = None
        group_label = None

        for row in reader:
            # First row is just labels so can ignore it.
            if counter > 0:

                if observed_data is None:
                    observed_data = im2double(row[:-1], bit_depth=8)
                else:
                    observed_data = np.vstack((observed_data, im2double(row[:-1], bit_depth=8)))

                # Last column is diagnosis type.
                if group_label is None:
                    group_label = np.array(np.int(row[-1]), dtype=int)
                else:
                    group_label = np.append(group_label, np.int(row[-1]))
            else:
                counter += 1  # Only need to increase once as we don't care after the first row.

        assert observed_data.shape[0] == group_label.shape[0], 'Number of images and diagnosis is not the same.'

    return observed_data, group_label


def im2double(image, bit_depth=8):
    """
    TODO
    :param image:
    :param bit_depth:
    :return:
    """

    # min_val = 0.0
    max_val = np.power(2, bit_depth).astype('float64') - 1.0

    # return (image.astype('float64') - min_val) / (max_val - min_val)
    return np.array(image, dtype=NP_DTYPE) / max_val


def read_64d_luminance(skin_cancer_mnist_path):
    """
    TODO
    :return:
    """

    file_path = skin_cancer_mnist_path + 'hmnist_8_8_L.csv'

    observed_data, diagnosis_labels = read_hmnist_csv(file_path)
    assert observed_data.shape[1] == 64, 'Number of pixels does not match expected number of 64.'

    return observed_data, diagnosis_labels


if __name__ == '__main__':

    # Define file path for results.
    bgplvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm', dataset='skin_cancer_mnist')
    mrd_results_file = RESULTS_FILE_NAME.format(model='mrd', dataset='skin_cancer_mnist')
    mrd_fully_independent_results_file = RESULTS_FILE_NAME.format(model='mrd_fully_independent',
                                                                  dataset='skin_cancer_mnist')
    gpdp_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset='skin_cancer_mnist')
    gpdp_mask_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm_mask_64', dataset='skin_cancer_mnist')

    # Choose what model we are looking at.
    results_file = bgplvm_results_file

    if isfile(results_file):
        # Read results.
        results = np.load(results_file)

        two_groups_images = results['original_data']

        y_train = results['y_train']
        ard_weights = results['ard_weights']
        alphas = results['signal_variance']
        betas = results['noise_precision']
        x_u = results['x_u']
        x_mean = results['x_mean']
        x_covar = results['x_covar']
        # phi = results['assignments']
        # gamma_atoms = results['gamma_atoms']
        # alpha_atoms = results['alpha_atoms']
        # beta_atoms = results['beta_atoms']

        # Get info about training data and model parameters.
        num_samples, num_output_dimensions = y_train.shape
        num_inducing_points, num_latent_dimensions = x_u.shape
        # truncation_level = beta_atoms.shape[0]

        # # Scale ARD weights.
        # minmax_scaler = MinMaxScaler()
        # scaled_ard = minmax_scaler.fit_transform(ard_weights)
        # vis.plot_ard(scaled_ard)
        # plot.gca().set_xticklabels([])  # Remove number of y-dimensions from x-axis.

        vis.plot_data(y_train)
        vis.plot_ard(ard_weights)
        # vis.plot_phi(phi)
        # vis.plot_phi_matrix(phi)
        plot.show()

    else:
        # Train model.
        num_samples = 100

        # Read data.
        two_conditions_data_file = DATA_PATH + 'skin_cancer_mnist/two_conditions_data.npy'
        if isfile(two_conditions_data_file):
            # Read numpy file of randomised data.
            two_groups_images = np.load(two_conditions_data_file)
        else:
            # Read original data.
            image_data, labels = read_64d_luminance(DATA_PATH + 'skin_cancer_mnist/')

            # Look at two groups.
            group_0_images = image_data[np.equal(labels, 0)]
            group_1_images = image_data[np.equal(labels, 1)]

            # Update number of samples if specific label does not have enough.
            num_samples = min(num_samples, group_0_images.shape[0], group_1_images.shape[0])

            # Randomly permute observations.
            np.random.seed(1)  # Random seed.
            rand_indices_0 = np.random.permutation(group_0_images.shape[0])
            rand_indices_1 = np.random.permutation(group_1_images.shape[0])

            # Combine into [num_samples x 128] data.
            two_groups_images = np.hstack((group_0_images[rand_indices_0[:num_samples]],
                                           group_1_images[rand_indices_1[:num_samples]]))

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

        # Optimisation variables.
        train_iter = 5000  # 1500
        learning_rate = 0.01  # 0.05

        # Construct models. Using values from elros as larger ones use too much GPU memory.
        num_inducing_points = 35  # 50  # 75  # GP_LVM_DEFAULT_NUM_INDUCING_POINTS
        num_latent_dimensions = 15  # GP_LVM_DEFAULT_LATENT_DIMENSIONS
        truncation_level = 20  # DP_DEFAULT_TRUNCATION_LEVEL
        print('\nConstructing Model:')
        print('  Number of inducing inputs (M): {}'.format(num_inducing_points))
        print('  Number of latent dimensions (Q): {}'.format(num_latent_dimensions))
        print('  DP truncation level (T): {}'.format(truncation_level))

        # Define instance of model.
        np.random.seed(1)  # Random seed.
        # model = bgplvm(y_train=y_train,
        #                num_inducing_points=num_inducing_points,
        #                num_latent_dims=num_latent_dimensions)
        # Define instance of MRD with known views.
        # model = mrd(views_train=[y_train[:, i:i + 64] for i in range(0, 128, 64)],
        #             num_inducing_points=num_inducing_points,
        #             num_latent_dims=num_latent_dimensions)
        # Define instance of fully independent MRD.
        # model = mrd(views_train=[y_train[:, i:i + 1] for i in range(0, 128, 1)],
        #             num_inducing_points=num_inducing_points,
        #             num_latent_dims=num_latent_dimensions)
        # Define instance of DP-GP-LVM. DP mask is default to 1.
        model = dp_gp_lvm(y_train=y_train,
                          num_inducing_points=num_inducing_points,
                          num_latent_dims=num_latent_dimensions,
                          truncation_level=truncation_level)
        # model = dp_gp_lvm(y_train=y_train,
        #                   num_inducing_points=num_inducing_points,
        #                   num_latent_dims=num_latent_dimensions,
        #                   truncation_level=truncation_level,
        #                   mask_size=64)
        model_training_objective = model.objective

        # Optimisation.
        model_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=model_training_objective)

        with tf.Session() as s:
            # Initialise variables.
            s.run(tf.global_variables_initializer())

            # Training optimisation loop.
            start_time = time()
            print('\nTraining Model:')
            for c in range(train_iter):
                s.run(model_opt_train)
                if (c % 100) == 0:
                    print('  Model opt iter {:5}: {}'.format(c, s.run(model_training_objective)))
            end_time = time()
            train_opt_time = end_time - start_time
            print('Final iter {:5}:'.format(c))
            print('  Model: {}'.format(s.run(model_training_objective)))
            print('Time to optimise: {} s'.format(train_opt_time))

            # Get converged values as numpy arrays.
            # ard_weights, noise_precision, signal_variance, inducing_input = \
            #     s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input))
            ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
                s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input,
                       model.assignments))
            x_mean, x_covar = s.run(model.q_x)
            w_1, w_2 = s.run(model.dp.q_alpha)
            gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)

        # Save results.
        print('\nSaving results to .npz file.')
        # np.savez(results_file, original_data=two_groups_images, y_train=y_train,
        #          ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
        #          x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time)
        np.savez(results_file, original_data=two_groups_images, y_train=y_train,
                 ard_weights=ard_weights, noise_precision=noise_precision, signal_variance=signal_variance,
                 x_u=inducing_input, assignments=assignments, x_mean=x_mean, x_covar=x_covar,
                 gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms,
                 q_alpha_w1=w_1, q_alpha_w2=w_2, train_opt_time=train_opt_time)

        # Print results.
        print('\nModel:')
        print('  Noise Precision:\n  {}'.format(np.squeeze(noise_precision)))
