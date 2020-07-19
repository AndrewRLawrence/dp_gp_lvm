"""
This module tests the predictive posterior for missing bottom half of face to see how well DP-GP-LVM works with
predicting missing pixels from the Frey faces images. The module compares the results with those from MRD.
"""

from src.data_io.frey_faces_reader import read_frey_mat
from src.distributions.normal import mvn_log_pdf
from src.models.dp_gp_lvm import dp_gp_lvm
from src.models.gaussian_process import manifold_relevance_determination
from src.utils.constants import RESULTS_FILE_NAME, PLOTS_PATH
from src.utils.types import get_training_variables, get_prediction_variables

import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from time import time


def run_mrd(y_train, y_test_observed, y_test_unobserved, num_latent_dimensions, num_inducing_points, view_mask,
            train_iter, predict_iter, learning_rate, save_file, seed_val=1):
    """
    TODO
    :param y_train:
    :param y_test_observed:
    :param y_test_unobserved:
    :param num_latent_dimensions:
    :param num_inducing_points:
    :param view_mask:
    :param train_iter:
    :param predict_iter:
    :param learning_rate:
    :param save_file:
    :param seed_val:
    :return:
    """

    # Set seed.
    np.random.seed(seed=seed_val)

    # Segment training data into views of size view_mask.
    num_output_dimensions = np.shape(y_train)[1]
    views_train = [y_train[:, i:i+view_mask] for i in range(0, num_output_dimensions, view_mask)]

    # Define instance of MRD.
    mrd = manifold_relevance_determination(views_train=views_train,
                                           num_latent_dims=num_latent_dimensions,
                                           num_inducing_points=num_inducing_points)

    # Segment observed and unobserved data into views.
    num_unobserved_dimensions = np.shape(y_test_unobserved)[1]
    views_test_observed = [y_test_observed]
    views_test_unobserved = [y_test_unobserved]

    # Define ground truth.
    ground_truth = y_test_unobserved

    # Define objectives.
    training_objective = mrd.objective
    predict_lower_bound, x_mean_test, x_covar_test, \
        predicted_means, predicted_covars = mrd.predict_missing_data(views_test=views_test_observed)
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
        print('\nTraining MRD..')
        for c in range(train_iter):
            s.run(opt_train)
            if (c % 100) == 0:
                print('  MRD opt iter {:5}: {}'.format(c, s.run(training_objective)))
        end_time = time()
        train_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  MRD: {}'.format(s.run(training_objective)))
        print('Time to optimise: {} s'.format(train_opt_time))

        # Get converged values as numpy arrays.
        ard_weights, noise_precisions, signal_variances, inducing_inputs = s.run((mrd.ard_weights,
                                                                                  mrd.noise_precision,
                                                                                  mrd.signal_variance,
                                                                                  mrd.inducing_input))
        x_mean, x_covar = s.run(mrd.q_x)

        # Initialise prediction variables.
        s.run(tf.variables_initializer(var_list=predict_var_list))

        # Prediction optimisation loop.
        start_time = time()
        print('\nOptimising Predictions..')
        for c in range(predict_iter):
            s.run(opt_predict)
            if (c % 100) == 0:
                print('  MRD opt iter {:5}: {}'.format(c, s.run(predict_objective)))
        end_time = time()
        predict_opt_time = end_time - start_time
        print('Final iter {:5}:'.format(c))
        print('  MRD: {}'.format(s.run(predict_objective)))
        print('Time to optimise: {} s'.format(predict_opt_time))

        # Get converged values as numpy arrays.
        x_mean_test_np, x_covar_test_np, list_predicted_means, list_predicted_covars = s.run((x_mean_test,
                                                                                              x_covar_test,
                                                                                              predicted_means,
                                                                                              predicted_covars))

        # Convert lists to numpy arrays.
        predicted_means_np = np.hstack(list_predicted_means)
        predicted_covars_np = np.concatenate(list_predicted_covars, axis=0)

        # Calculate log-likelihood of ground truth with predicted posteriors.
        gt_log_likelihoods = [
            mvn_log_pdf(x=tf.transpose(tf.slice(ground_truth, begin=[0, du], size=[-1, 1])),
                        mean=tf.transpose(tf.slice(predicted_means_np, begin=[0, du], size=[-1, 1])),
                        covariance=tf.squeeze(tf.slice(predicted_covars_np, begin=[du, 0, 0], size=[1, -1, -1]),
                                              axis=0))
            for du in range(num_unobserved_dimensions)]
        gt_log_likelihoods_np = np.array(s.run(gt_log_likelihoods))
        gt_log_likelihood = np.sum(gt_log_likelihoods_np)

    # Save results. Converting lists to numpy arrays.
    np.savez(save_file, y_train=y_train, y_test_observed=y_test_observed, y_test_unobserved=y_test_unobserved,
             views_train=views_train, views_test_observed=views_test_observed,
             views_test_unobserved=views_test_unobserved, ard_weights=np.array(ard_weights),
             noise_precision=np.array(noise_precisions), signal_variance=np.array(signal_variances),
             x_u=np.array(inducing_inputs), x_mean=x_mean, x_covar=x_covar, train_opt_time=train_opt_time,
             x_mean_test=x_mean_test_np, x_covar_test=x_covar_test_np, predicted_mean=predicted_means_np,
             predicted_covar=predicted_covars_np, predict_opt_time=predict_opt_time,
             gt_log_likelihoods=gt_log_likelihoods_np, gt_log_likelihood=gt_log_likelihood)

    # # Print results.
    # print('\nMRD:')
    # print('  Ground Truth Predicted Posterior Log-Likelihood: {}'.format(gt_log_likelihood))
    # print('  Noise Precisions: {}'.format(np.squeeze(np.array(noise_precisions))))


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
    model = dp_gp_lvm(y_train=y_train,
                      num_latent_dims=num_latent_dimensions,
                      num_inducing_points=num_inducing_points,
                      truncation_level=truncation_level,
                      mask_size=dp_mask_size)

    num_unobserved_dimensions = np.shape(y_test_unobserved)[1]

    # Define objectives.
    training_objective = model.objective
    predict_lower_bound, x_mean_test, x_covar_test, \
        predicted_mean, predicted_covar = model.predict_missing_data(y_test=y_test_observed)
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
            s.run((model.ard_weights, model.noise_precision, model.signal_variance, model.inducing_input,
                   model.assignments))
        x_mean, x_covar = s.run(model.q_x)
        gamma_atoms, alpha_atoms, beta_atoms = s.run(model.dp_atoms)

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

    # # Print results.
    # print('\nDP-GP-LVM:')
    # print('  Ground Truth Predicted Posterior Log-Likelihood: {}'.format(gt_log_likelihood))
    # print('  Noise Precisions: {}'.format(np.squeeze(noise_precision)))


if __name__ == '__main__':

    # Optimisation variables.
    learning_rate = 0.025
    num_iter_train = 2500
    num_iter_predict = 2000

    # Model hyperparameters.
    num_inducing_points = 20
    num_latent_dimensions = 15
    truncation_level = 18

    # Set number of samples for training and prediction.
    num_training_samples = 100
    num_test_samples = 10

    # Read all faces.
    TOTAL_NUM_SAMPLES = 1965
    TOTAL_NUM_PIXELS = 560
    faces = read_frey_mat()
    assert faces.shape[0] == TOTAL_NUM_SAMPLES, \
        'Number of samples does not match expected value of {}.'.format(TOTAL_NUM_SAMPLES)
    assert faces.shape[1] == TOTAL_NUM_PIXELS,\
        'Number of pixels (dimensions) does not match expected value of {}.'.format(TOTAL_NUM_PIXELS)

    # Loop through a few seeds for randomly sampling the data.
    data_seeds = np.arange(5)
    for s in data_seeds:
        # Set seed.
        np.random.seed(seed=s)

        # Define training and test samples.
        indices = np.random.choice(faces.shape[0], size=(num_training_samples + num_test_samples), replace=False)
        training_indices = indices[:num_training_samples]
        test_indices = indices[num_training_samples:]

        # Normalise data to zero mean and unit variance.
        scaler = StandardScaler()
        training_data = scaler.fit_transform(faces[training_indices, :])
        test_data = scaler.transform(faces[test_indices, :])
        assert training_data.shape[0] == num_training_samples, \
            'Number of training samples does not match expected value of {}'.format(num_training_samples)
        assert training_data.shape[1] == TOTAL_NUM_PIXELS, \
            'Number of pixels (dimensions) does not match expected value of {}.'.format(TOTAL_NUM_PIXELS)
        assert test_data.shape[0] == num_test_samples, \
            'Number of test samples does not match expected value of {}'.format(num_test_samples)
        assert test_data.shape[1] == TOTAL_NUM_PIXELS, \
            'Number of pixels (dimensions) does not match expected value of {}.'.format(TOTAL_NUM_PIXELS)

        # Remove some pixels for prediction. This is slice of face.
        percent_missing_pixels = 0.5
        num_observed_dimensions = int(np.ceil(TOTAL_NUM_PIXELS * (1.0 - percent_missing_pixels)))
        num_unobserved_dimensions = TOTAL_NUM_PIXELS - num_observed_dimensions

        # Print info.
        print('\nFrey Faces:')
        print('  Seed: {}'.format(s))
        print('  Number of training samples: {}'.format(num_training_samples))
        print('  Number of training dimensions: {}'.format(TOTAL_NUM_PIXELS))
        print('  Number of test samples: {}'.format(num_test_samples))
        print('  Number of provided/observed dimensions: {}'.format(num_observed_dimensions))
        print('  Number of missing/unobserved dimensions: {}'.format(num_unobserved_dimensions))

        # # Test face is split in half from top and bottom.
        # half_faces = np.hstack((faces[training_indices, :num_observed_dimensions],
        #                         np.zeros(shape=(num_training_samples, num_unobserved_dimensions))))
        # plot.imshow(half_faces[0, :].reshape(28, 20), cmap='gray', vmin=0.0, vmax=1.0)
        # # plot.imshow(scaler.inverse_transform(np.hstack(
        # #     (training_data[0, :num_observed_dimensions],
        # #      np.zeros(shape=num_unobserved_dimensions)))).reshape(28, 20),
        # #             cmap='gray', vmin=0.0, vmax=1.0)
        # plot.show()
        # quit(0)

        # Define file path for results.
        seed_val = 10
        dataset_str = 'frey_faces_bottom_missing_data_seed{}_n{}_m{}_q{}_t{}_init_seed{}'.format(s,
                                                                                                 num_training_samples,
                                                                                                 num_inducing_points,
                                                                                                 num_latent_dimensions,
                                                                                                 truncation_level,
                                                                                                 seed_val)
        mrd_results_file = RESULTS_FILE_NAME.format(model='mrd', dataset=dataset_str)
        dp_gp_lvm_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)

        # Run MRD.
        if not isfile(mrd_results_file):
            # Reset default graph before building new model graph. This speeds up script.
            tf.reset_default_graph()
            # Build Bayesian GP-LVM graph and run it for current configuration.
            run_mrd(y_train=training_data,
                    y_test_observed=test_data[:, :num_observed_dimensions],
                    y_test_unobserved=test_data[:, num_observed_dimensions:],
                    num_latent_dimensions=num_latent_dimensions,
                    num_inducing_points=num_inducing_points,
                    view_mask=num_observed_dimensions,
                    train_iter=num_iter_train,
                    predict_iter=num_iter_predict,
                    learning_rate=learning_rate,
                    save_file=mrd_results_file,
                    seed_val=seed_val)

        # Run DP-GP-LVM.
        if not isfile(dp_gp_lvm_results_file):
            # Reset default graph before building new model graph. This speeds up script.
            tf.reset_default_graph()
            # Build DP-GP-LVM graph and run it for current configuration.
            run_dp_gp_lvm(y_train=training_data,
                          y_test_observed=test_data[:, :num_observed_dimensions],
                          y_test_unobserved=test_data[:, num_observed_dimensions:],
                          num_latent_dimensions=num_latent_dimensions,
                          num_inducing_points=num_inducing_points,
                          truncation_level=truncation_level,
                          dp_mask_size=1,
                          train_iter=num_iter_train,
                          predict_iter=num_iter_predict,
                          learning_rate=learning_rate,
                          save_file=dp_gp_lvm_results_file,
                          seed_val=seed_val)

        # Plot latent spaces.
        mrd_ard = np.load(mrd_results_file)['ard_weights']
        dp_gp_lvm_ard = np.load(dp_gp_lvm_results_file)['ard_weights']
        # plot.figure()
        # plot.imshow(np.sqrt(mrd_ard[0]).T, interpolation='nearest', aspect='auto',
        #             extent=(0, num_observed_dimensions, num_latent_dimensions, 0), origin='upper')
        # plot.colorbar()
        # plot.title('MRD - Top Half View')
        # plot.figure()
        # plot.imshow(np.sqrt(mrd_ard[1]).T, interpolation='nearest', aspect='auto',
        #             extent=(0, num_unobserved_dimensions, num_latent_dimensions, 0), origin='upper')
        # plot.colorbar()
        # plot.title('MRD - Bottom Half View')
        # plot.figure()
        # plot.imshow(np.sqrt(dp_gp_lvm_ard).T, interpolation='nearest', aspect='auto',
        #             extent=(0, TOTAL_NUM_PIXELS, num_latent_dimensions, 0), origin='upper')
        # plot.colorbar()
        # plot.title('DP-GP-LVM')
        # plot.show()
        # quit(0)

        # Plot ARD weights in image space.
        for q in range(num_latent_dimensions):
            current_ard = np.sqrt(dp_gp_lvm_ard[:, q]).reshape(28, 20)
            # plot.figure(figsize=(4, 4))
            plot.figure()
            plot.imshow(current_ard, interpolation='nearest', aspect='auto', origin='upper')
            plot.colorbar()
            plot.title('Latent Dim {}'.format(q))
        plot.show()
        quit(0)



        # Inverse the normalization and view predicted image.
        show_plots = False
        save_plots = False
        if show_plots or save_plots:
            ground_truth = scaler.inverse_transform(test_data)
            mrd_predicted_mean = np.load(mrd_results_file)['predicted_mean']
            dp_gp_lvm_predicted_mean = np.load(dp_gp_lvm_results_file)['predicted_mean']
            mrd_predicted_images = scaler.inverse_transform(np.hstack((test_data[:, :num_observed_dimensions],
                                                                       mrd_predicted_mean)))
            dp_gp_lvm_predicted_images = scaler.inverse_transform(np.hstack((test_data[:, :num_observed_dimensions],
                                                                             dp_gp_lvm_predicted_mean)))
            # assert ground_truth.shape[0] == predicted_images.shape[0]
            for i in range(ground_truth.shape[0]):
                fig_size = (3, 2)  # (10, 5)
                fig, (ax1, ax2, ax3) = plot.subplots(nrows=1, ncols=3, sharey='row', figsize=fig_size)
                plot.suptitle('Face {}'.format(test_indices[i]))
                ax1.imshow(ground_truth[i, :].reshape(28, 20), cmap='gray', vmin=0.0, vmax=1.0)
                ax1.set_axis_off()
                ax1.set_title('GT', fontdict={'fontsize': 8})
                ax2.imshow(mrd_predicted_images[i, :].reshape(28, 20), cmap='gray', vmin=0.0, vmax=1.0)
                ax2.set_axis_off()
                ax2.set_title('MRD', fontdict={'fontsize': 8})
                ax3.imshow(dp_gp_lvm_predicted_images[i, :].reshape(28, 20), cmap='gray', vmin=0.0, vmax=1.0)
                ax3.set_axis_off()
                ax3.set_title('DP-GP-LVM', fontdict={'fontsize': 8})

                # Save plots.
                if save_plots:
                    plot_filename = ''.join((PLOTS_PATH, 'frey_faces_bottom_missing', '_{}_{}'.format(s, i)))
                    plot.savefig(plot_filename + '.pdf', bbox_inches='tight')

                # Close all figures if no need to display them.
                if not show_plots:
                    plot.close('all')

        if show_plots:
            # Show plots.
            plot.show()
