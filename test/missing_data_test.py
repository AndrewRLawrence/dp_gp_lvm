"""
This module tests the predictive posterior for missing data in the Bayesian GP-LVM, MRD, and DP-GP-LVM models. This
module uses synthetic (or toy) data to perform these tests.
"""

from distributions.normal import mvn_log_pdf
from kernels.rbf_kernel import k_ard_rbf
from models.dp_gp_lvm import dp_gp_lvm
from models.gaussian_process import bayesian_gp_lvm
from utils.types import TF_DTYPE, get_training_variables, get_prediction_variables

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from time import time


# Random seed.
np.random.seed(1)

# Optimisation variables.
num_iter = 1000
learning_rate = 0.05
refresh_iter = int(np.ceil(num_iter / 15.0))

# Generate synthetic data.
num_samples = 80
num_training_samples = int(np.ceil(0.75 * num_samples))
num_input_dimensions = 5
num_output_dimensions = 100
num_missing_dimensions = int(np.floor(0.15 * num_output_dimensions))
num_provided_dimensions = num_output_dimensions - num_missing_dimensions
noise_var = 1.0e-2

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
y = np.transpose(np.random.multivariate_normal(mean=np.zeros(num_samples), cov=covar_np, size=num_output_dimensions))
y_train = y[:num_training_samples, :]
y_test = y[num_training_samples:, :]
y_test_provided = y_test[:, :num_provided_dimensions]
y_test_missing_ground_truth = y_test[:, num_provided_dimensions:]

# Print info.
print('\nToy Data:')
print('Total number of observations: {}'.format(num_samples))
print('Total number of output dimensions: {}'.format(num_output_dimensions))
print('\nTraining Data:')
print('Number of training samples: {}'.format(num_training_samples))
print('Number of training observations: {}'.format(num_output_dimensions))
print('\nMissing Data:')
print('Number of test samples: {}'.format(num_samples-num_training_samples))
print('Number of provided dimensions: {}'.format(num_provided_dimensions))
print('Number of missing dimensions: {}'.format(num_missing_dimensions))

# Define instance of Bayesian GP-LVM.
bgplvm_model = bayesian_gp_lvm(y_train=y_train)
training_objective = bgplvm_model.objective
missing_data_lower_bound, predicted_mean, predicted_covar = bgplvm_model.predict_missing_data(y_test=y_test_provided)
prediction_objective = tf.negative(missing_data_lower_bound)

# Define instance of DP-GP-LVM with T=1.
gpdp_1 = dp_gp_lvm(y_train=y_train, truncation_level=1)

# Define instance of DP-GP-LVM with default T.
gpdp_t = dp_gp_lvm(y_train=y_train)

# Optimisation.
opt_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=training_objective,
                                                                         var_list=get_training_variables())
opt_predict = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=prediction_objective,
                                                                           var_list=get_prediction_variables())

with tf.Session() as s:
    # Initialise variables.
    s.run(tf.variables_initializer(var_list=get_training_variables()))  # Initialize training variables first.
    s.run(tf.variables_initializer(var_list=get_prediction_variables()))  # Then initialize prediction variables.
    s.run(tf.global_variables_initializer())  # Finally initialize any remaining global variables.

    start_time = time()
    print('\nTraining Model..')
    for i in range(num_iter + 250):
        opt, cost = s.run((opt_train, training_objective))
        if (i % refresh_iter) == 0:
            print('  opt iter {:5}: {}'.format(i, cost))

    print('Final iter {:5}: {}'.format(i, cost))
    end_time = time()
    print('Time to optimise: {} s'.format(end_time - start_time))

    [ard_weights, noise_precision] = s.run([bgplvm_model.ard_weights, bgplvm_model.noise_precision])

    start_time = time()
    print('\nOptimising Prediction..')
    for i in range(num_iter):
        opt, cost = s.run((opt_predict, prediction_objective))
        if (i % refresh_iter) == 0:
            print('  opt iter {:5}: {}'.format(i, cost))

    print('Final iter {:5}: {}'.format(i, cost))
    end_time = time()
    print('Time to optimise: {} s'.format(end_time - start_time))

    # Calculate log-likelihood of ground truth with predicted posterior.
    gt_log_likelihoods = [mvn_log_pdf(x=tf.transpose(tf.slice(y_test_missing_ground_truth, begin=[0, d], size=[-1, 1])),
                                      mean=tf.transpose(tf.slice(predicted_mean, begin=[0, d], size=[-1, 1])),
                                      covariance=tf.squeeze(tf.slice(predicted_covar,
                                                                     begin=[d, 0, 0],
                                                                     size=[1, -1, -1]),
                                                            axis=0))
                          for d in range(num_missing_dimensions)]
    final_gt_log_likelihoods = s.run(gt_log_likelihoods)
    [final_predicted_mean, final_predicted_covar] = s.run([predicted_mean, predicted_covar])

# Print results.
print('\nARD Weights: {}'.format(np.squeeze(ard_weights)))
print('Noise Precision: {}'.format(np.squeeze(noise_precision)))

final_gt_log_likelihoods = np.array(final_gt_log_likelihoods)
print('\nL2 Norm of Differences: {}'.format(np.linalg.norm(y_test_missing_ground_truth - final_predicted_mean, axis=0)))
print('Ground Truth Predicted Posterior Log Likelihoods: {}'.format(final_gt_log_likelihoods))
print('\nGround Truth Predicted Posterior Log Likelihood: {}'.format(np.sum(final_gt_log_likelihoods)))

plot.figure()
plot.imshow(ard_weights)
plot.title('ARD Weights')

for md in range(num_missing_dimensions):
    plot.figure()
    plot.imshow(final_predicted_covar[md, :, :])
    plot.title('Predicted Covar for Missing Dimension {}'.format(md+1))

plot.show()
