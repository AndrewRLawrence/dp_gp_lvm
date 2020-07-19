"""
This module tests the training of DP-GP-LVM with difficult synthetic (or toy) data generated from known GPs.
"""

from kernels.rbf_kernel import k_ard_rbf
from models.dp_gp_lvm import dp_gp_lvm
from utils.constants import GP_LVM_DEFAULT_NUM_INDUCING_POINTS, \
    GP_LVM_DEFAULT_LATENT_DIMENSIONS, DP_DEFAULT_TRUNCATION_LEVEL
from utils.types import TF_DTYPE

import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler
from sys import path
import tensorflow as tf
from time import time


# Random seed.
np.random.seed(10)

# Optimisation variables.
train_iter = 2500  #1500
learning_rate = 0.01  # 0.05

# Generate synthetic data.
num_samples = 100  # 30  # 80
num_input_dimensions = 5
num_output_dimensions = 60  # 50
# noise_var = 1.0e-2
# noise_std = np.sqrt(noise_var)

# Define paths.
absolute_path = [ap for ap in path if 'aistats_2019' in ap]
data_path = absolute_path[-1] + '/test/data/'
results_path = absolute_path[-1] + '/test/results/'

# Create synthetic data if it has not been created already.
# file_name = data_path + 'synthetic_data_hard_4func.npy'
file_name = data_path + 'synthetic_data_hard_4func_ind_noise.npy'
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
    ard_3 = tf.constant([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype=TF_DTYPE)  # Function of X1 and X3.
    ard_4 = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0]], dtype=TF_DTYPE)  # Function of X0 and X3.

    # Different noise terms per GP component.
    betas = np.square(0.1 + 0.2 * np.random.standard_normal(4))
    beta_1 = tf.constant([[1.0 / betas[0]]], dtype=TF_DTYPE)
    beta_2 = tf.constant([[1.0 / betas[1]]], dtype=TF_DTYPE)
    beta_3 = tf.constant([[1.0 / betas[2]]], dtype=TF_DTYPE)
    beta_4 = tf.constant([[1.0 / betas[3]]], dtype=TF_DTYPE)

    alpha_generator = tf.constant([[1.0]], dtype=TF_DTYPE)

    covar_1 = k_ard_rbf(gamma=ard_1, alpha=alpha_generator, beta=beta_1).covariance_matrix(input_0=x,
                                                                                           input_1=None,
                                                                                           include_noise=True,
                                                                                           include_jitter=True)
    covar_2 = k_ard_rbf(gamma=ard_2, alpha=alpha_generator, beta=beta_2).covariance_matrix(input_0=x,
                                                                                           input_1=None,
                                                                                           include_noise=True,
                                                                                           include_jitter=True)
    covar_3 = k_ard_rbf(gamma=ard_3, alpha=alpha_generator, beta=beta_3).covariance_matrix(input_0=x,
                                                                                           input_1=None,
                                                                                           include_noise=True,
                                                                                           include_jitter=True)
    covar_4 = k_ard_rbf(gamma=ard_4, alpha=alpha_generator, beta=beta_4).covariance_matrix(input_0=x,
                                                                                           input_1=None,
                                                                                           include_noise=True,
                                                                                           include_jitter=True)

    with tf.Session() as sess:
        covar_1_np = np.squeeze(sess.run(covar_1), axis=0)  # [N x N].
        covar_2_np = np.squeeze(sess.run(covar_2), axis=0)  # [N x N].
        covar_3_np = np.squeeze(sess.run(covar_3), axis=0)  # [N x N].
        covar_4_np = np.squeeze(sess.run(covar_4), axis=0)  # [N x N].

    # Sample from each GP D/4 times to build Y as [N x D] as each sample from GP is N-length.
    d_gp = int(0.25 * num_output_dimensions)
    y = np.concatenate(
        (
            np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples), cov=covar_1_np, size=d_gp)),
            np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples), cov=covar_2_np, size=d_gp)),
            np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples), cov=covar_3_np, size=d_gp)),
            np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples), cov=covar_4_np, size=d_gp))
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

# Construct models.
num_inducing_points = 50  # GP_LVM_DEFAULT_NUM_INDUCING_POINTS
num_latent_dimensions = 10  # GP_LVM_DEFAULT_LATENT_DIMENSIONS
truncation_level = 20  # DP_DEFAULT_TRUNCATION_LEVEL
print('\nConstructing Models:')
print('  Number of inducing inputs (M): {}'.format(num_inducing_points))
print('  Number of latent dimensions (Q): {}'.format(num_latent_dimensions))
print('  DP truncation level (T): {}'.format(truncation_level))

# Reset default graph before building new model graph.
# This speeds up script as it removes graph nodes for generating synthetic data.
tf.reset_default_graph()

# # Define instance of Bayesian GP-LVM with default parameters.
# bgplvm = bayesian_gp_lvm(y_train=y_train,
#                          num_inducing_points=num_inducing_points,
#                          num_latent_dims=num_latent_dimensions)
# bgplvm_training_objective = bgplvm.objective

# Define instance of DP-GP-LVM with default parameters. DP mask is default to 1.
gpdp = dp_gp_lvm(y_train=y_train,
                 num_inducing_points=num_inducing_points,
                 num_latent_dims=num_latent_dimensions,
                 truncation_level=truncation_level)
gpdp_training_objective = gpdp.objective

# Optimisation.
# opt_bgplvm_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=bgplvm_training_objective)
gpdp_opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=gpdp_training_objective)

with tf.Session() as s:
    # Initialise variables.
    s.run(tf.global_variables_initializer())

    # Training optimisation loop.
    start_time = time()
    print('\nTraining GP-DP..')
    for c in range(train_iter):
        s.run(gpdp_opt_train)
        if (c % 100) == 0:
            print('  GP-DP opt iter {:5}: {}'.format(c, s.run(gpdp_training_objective)))
    end_time = time()
    train_opt_time = end_time - start_time
    print('Final iter {:5}:'.format(c))
    print('  GP-DP: {}'.format(s.run(gpdp_training_objective)))
    print('Time to optimise: {} s'.format(train_opt_time))

    # Get converged values as numpy arrays.
    ard_weights, noise_precision, signal_variance, inducing_input, assignments = \
        s.run((gpdp.ard_weights, gpdp.noise_precision, gpdp.signal_variance, gpdp.inducing_input, gpdp.assignments))
    x_mean, x_covar = s.run(gpdp.q_x)
    gamma_atoms, alpha_atoms, beta_atoms = s.run(gpdp.dp_atoms)

# Save results.
gpdp_file = results_path + 'gpdp_hard_synthetic_data_test.npz'
np.savez(gpdp_file, y_train=y_train, ard_weights=ard_weights, noise_precision=noise_precision,
         signal_variance=signal_variance, x_u=inducing_input, x_mean=x_mean, x_covar=x_covar, assignments=assignments,
         gamma_atoms=gamma_atoms, alpha_atoms=alpha_atoms, beta_atoms=beta_atoms, train_opt_time=train_opt_time)

# Print results.
print('\nGP-DP:')
print('  Noise Precision:\n  {}'.format(np.squeeze(noise_precision)))

# Plot results.
plot.figure()
plot.imshow(assignments.T)
plot.title('GP-DP Assignments')

plot.figure()
plot.imshow(ard_weights.T)
plot.title('GP-DP ARD Weights')

plot.show()
