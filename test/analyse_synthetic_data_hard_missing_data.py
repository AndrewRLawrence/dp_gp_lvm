"""
This module analyses the results for the missing data tests with the predictive posterior from the Bayesian GP-LVM, MRD,
and DP-GP-LVM models. This module analyses results from tests with PoseTrack data.
"""

import numpy as np
from os import listdir
from sys import path

if __name__ == '__main__':

    # Define path of result files.
    absolute_path = [ap for ap in path if 'aistats_2019' in ap]
    results_path = absolute_path[-1] + '/test/results/'

    # Define configuration values we want to analyse.
    seeds = np.arange(10, 15, dtype=int)  # [10 - 14].
    percent_samples_observed = np.array([0.8, 0.9])
    # percent_dimensions_observed = np.array([0.7, 0.8, 0.9])
    percent_dimensions_observed = np.array([0.8, 0.9])

    # Find configurations with best results for DP-GP-LVM vs. Bayesian GP-LVM.
    for n in percent_samples_observed:
        for d in percent_dimensions_observed:
            for cov_calc in ['synthetic_data_2afunc_ind_noise_missing_data_seed_{0}_',
                             'synthetic_data_2afunc_ind_noise_missing_data_f_star_seed_{0}_']:
            # for cov_calc in ['synthetic_data_ind_noise_missing_data_seed_{0}_',
            #                  'synthetic_data_ind_noise_missing_data_f_star_seed_{0}_']:
            # for noise_amount in ['more', 'less']:

                # desired_configs = ['synthic_data_{0}_noise_missing_data_f_star_seed_{1}_'.format(noise_amount, i) +
                #                    'n_percent_{0}_d_percent_{1}.npz'.format(n, d) for i in seeds]

                desired_configs = [cov_calc.format(i) + 'n_percent_{0}_d_percent_{1}.npz'.format(n, d) for i in seeds]
                # desired_configs = ['synthetic_data_ind_noise_missing_data_seed_{0}_'.format(i) +
                #                    'n_percent_{0}_d_percent_{1}.npz'.format(n, d) for i in seeds]
                # desired_configs = ['synthetic_data_ind_noise_missing_data_f_star_seed_{0}_'.format(i) +
                #                    'n_percent_{0}_d_percent_{1}.npz'.format(n, d) for i in seeds]
                desired_bgplvm_configs = ['bgplvm_' + i for i in desired_configs]  # There is typo when saving files.
                desired_gpdp_configs = ['gpdp_' + i for i in desired_configs]

                bgplvm_result_files = [np_file for np_file in listdir(results_path)
                                       if np_file in desired_bgplvm_configs]
                gpdp_result_files = [np_file for np_file in listdir(results_path)
                                     if np_file in desired_gpdp_configs]

                # Print current configuration.
                print('\n--------------------------------------------------------------------------------')
                print('Current Configuration:')
                # print('  Noise Amount: {}'.format(noise_amount))
                print('  Covariance used: {}'.format('F*' if 'f_star' in cov_calc else 'Y*'))
                print('  Percent samples observed (n%):    {}'.format(n))
                print('  Percent dimensions observed (d%): {}'.format(d))

                if len(bgplvm_result_files) == 0 and len(gpdp_result_files) == 0:
                    print('\nNo results found for this configuration.')
                else:
                    # Get all ground truth log-likelihoods.
                    bgplvm_gt_log_likelihoods = np.array([np.load(results_path + i)['gt_log_likelihood']
                                                          for i in bgplvm_result_files])
                    gpdp_gt_log_likelihoods = np.array([np.load(results_path + i)['gt_log_likelihood']
                                                        for i in gpdp_result_files])

                    # Print results.
                    sample_n = len(bgplvm_result_files)
                    print('\nBayesian GP-LVM:')
                    print('  Number of test cases: {}'.format(sample_n))
                    if sample_n > 1:
                        print('  Mean:                 {}'.format(np.mean(bgplvm_gt_log_likelihoods)))
                        sample_std_dev = np.std(bgplvm_gt_log_likelihoods, ddof=1)
                        print('  Std Dev:              {}'.format(sample_std_dev))
                        print('  Std Error:            {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))
                    elif sample_n == 1:
                        print('  Mean:                 {}'.format(np.mean(bgplvm_gt_log_likelihoods)))

                    sample_n = len(gpdp_result_files)
                    print('\nDP-GP-LVM:')
                    print('  Number of test cases:   {}'.format(sample_n))
                    if sample_n > 1:
                        print('  Mean:                   {}'.format(np.mean(gpdp_gt_log_likelihoods)))
                        sample_std_dev = np.std(gpdp_gt_log_likelihoods, ddof=1)
                        print('  Std Dev:                {}'.format(sample_std_dev))
                        print('  Std Error:              {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))
                    elif sample_n == 1:
                        print('  Mean:                   {}'.format(np.mean(gpdp_gt_log_likelihoods)))

