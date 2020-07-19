"""
This module tests the training in the Bayesian GP-LVM, MRD, and DP-GP-LVM models with synthetic (or toy) data sampled
from a known GP.
"""

from kernels.rbf_kernel import k_ard_rbf
from models.dp_gp_lvm import dp_gp_lvm
from models.gaussian_process import bayesian_gp_lvm
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
np.random.seed(1)

# Optimisation variables.
num_iter = 1500
learning_rate = 0.05
refresh_iter = int(np.ceil(num_iter / 15.0))

# Generate synthetic data.
num_samples = 100  # 30  # 80
num_input_dimensions = 5
num_output_dimensions = 20  # 50  # 100
noise_var = 1.0e-2

# Create synthetic data if it has not been created already.
absolute_path = [ap for ap in path if 'aistats_2019' in ap]
file_name = absolute_path[-1] + '/test/data/synthetic_data_gp_single_func.npy'
if isfile(file_name):

    y_train = np.load(file_name)

else:

    # Random input or linspace.
    x = np.random.standard_normal((num_samples, num_input_dimensions))

    # All data generated from same GP.
    gamma_generator = tf.constant([[1.0, 2.0, 0.0, 0.0, 0.0]], dtype=TF_DTYPE)
    alpha_generator = tf.constant([[1.0]], dtype=TF_DTYPE)
    beta_generator = tf.constant([[1.0 / noise_var]], dtype=TF_DTYPE)
    k_generator = k_ard_rbf(gamma=gamma_generator, alpha=alpha_generator, beta=beta_generator)
    covar_synthetic = k_generator.covariance_matrix(input_0=x, input_1=None, include_noise=True, include_jitter=True)

    with tf.Session() as sess:
        covar_np = np.squeeze(sess.run(covar_synthetic), axis=0)  # [N x N].

    # Sample from GP D times to build Y as [N x D] as each sample from GP is N-length.
    y = np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples),
                                                   cov=covar_np,
                                                   size=num_output_dimensions))

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
num_inducing_points = GP_LVM_DEFAULT_NUM_INDUCING_POINTS
num_latent_dimensions = GP_LVM_DEFAULT_LATENT_DIMENSIONS
truncation_level = DP_DEFAULT_TRUNCATION_LEVEL
print('\nConstructing Models:')
print('  Number of inducing inputs (M): {}'.format(num_inducing_points))
print('  Number of latent dimensions (Q): {}'.format(num_latent_dimensions))
print('  DP truncation level (T): {}'.format(truncation_level))

# Define instance of Bayesian GP-LVM with default parameters.
bgplvm = bayesian_gp_lvm(y_train=y_train,
                         num_inducing_points=num_inducing_points,
                         num_latent_dims=num_latent_dimensions)
bgplvm_training_objective = bgplvm.objective

# Define instance of DP-GP-LVM with T=1 and other parameters as default.
gpdp_1 = dp_gp_lvm(y_train=y_train,
                   num_inducing_points=num_inducing_points,
                   num_latent_dims=num_latent_dimensions,
                   truncation_level=1)
gpdp_1_training_objective = gpdp_1.objective

# Define instance of DP-GP-LVM with default parameters.
gpdp_t = dp_gp_lvm(y_train=y_train,
                   num_inducing_points=num_inducing_points,
                   num_latent_dims=num_latent_dimensions,
                   truncation_level=truncation_level)
gpdp_t_training_objective = gpdp_t.objective

# Optimisation.
opt_bgplvm_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=bgplvm_training_objective)
opt_gpdp_1_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=gpdp_1_training_objective)
opt_gpdp_t_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=gpdp_t_training_objective)

with tf.Session() as s:
    # Initialise variables.
    s.run(tf.global_variables_initializer())  # Finally initialize any remaining global variables.

    # Training optimisation loop.
    start_time = time()
    print('\nTraining Models:')
    for i in range(num_iter):
        bgplvm_opt, bpglvm_cost, \
            gpdp_1_opt, gpdp_1_cost, \
            gpdp_t_opt, gpdp_t_cost = s.run((opt_bgplvm_train, bgplvm_training_objective,
                                             opt_gpdp_1_train, gpdp_1_training_objective,
                                             opt_gpdp_t_train, gpdp_t_training_objective))
        if (i % refresh_iter) == 0:
            print('  BGPLVM   opt iter {:4}: {}'.format(i, bpglvm_cost))
            print('  GP-DP_1  opt iter {:4}: {}'.format(i, gpdp_1_cost))
            print('  GP-DP_T  opt iter {:4}: {}'.format(i, gpdp_t_cost))

    print('\nFinal iter {:5}:'.format(i))
    print('  BGPLVM:   {}'.format(bpglvm_cost))
    print('  GP-DP_1:  {}'.format(gpdp_1_cost))
    print('  GP-DP_T:  {}'.format(gpdp_t_cost))

    end_time = time()
    print('\nTime to optimise: {} s'.format(end_time - start_time))

    # Get final values.
    bgplvm_ard_weights, bgplvm_noise_precision, \
        gpdp_1_assignments, gpdp_1_ard_weights, gpdp_1_noise_precision, \
        gpdp_t_assignments, gpdp_t_ard_weights, gpdp_t_noise_precision = s.run((bgplvm.ard_weights,
                                                                                bgplvm.noise_precision,
                                                                                gpdp_1.assignments, gpdp_1.ard_weights,
                                                                                gpdp_1.noise_precision,
                                                                                gpdp_t.assignments,
                                                                                gpdp_t.ard_weights,
                                                                                gpdp_t.noise_precision))

# Print results.
print('\nBGPLVM:')
print('  ARD Weights:\n  {}'.format(np.squeeze(bgplvm_ard_weights)))
print('  Noise Precision:\n  {}'.format(np.squeeze(bgplvm_noise_precision)))

print('\nGP-DP_1:')
# print('  Assignments:\n  {}'.format(np.squeeze(gpdp_1_assignments)))
# print('  ARD Weights:\n  {}'.format(np.squeeze(gpdp_1_ard_weights)))
print('  Noise Precision:\n  {}'.format(np.squeeze(gpdp_1_noise_precision)))

print('\nGP-DP_T:')
# print('  Assignments:\n  {}'.format(np.squeeze(gpdp_t_assignments)))
# print('  ARD Weights:\n  {}'.format(np.squeeze(gpdp_t_ard_weights)))
print('  Noise Precision:\n  {}'.format(np.squeeze(gpdp_t_noise_precision)))

# Plot results.
plot.figure()
plot.imshow(bgplvm_ard_weights.T)
plot.title('BGPLVM ARD Weights')

plot.figure()
plot.imshow(gpdp_1_assignments.T)
plot.title('GP-DP_1 Assignments')

plot.figure()
plot.imshow(gpdp_1_ard_weights.T)
plot.title('GP-DP_1 ARD Weights')

plot.figure()
plot.imshow(gpdp_t_assignments.T)
plot.title('GP-DP_T Assignments')

plot.figure()
plot.imshow(gpdp_t_ard_weights.T)
plot.title('GP-DP_T ARD Weights')

plot.show()
