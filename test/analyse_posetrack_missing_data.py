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
    seeds = np.arange(1, 11, dtype=int)  # [1 - 10].
    dp_masks = np.arange(1, 3, dtype=int)  # [1, 2].
    percent_samples_observed = np.linspace(0.5, 1.0, 5, endpoint=False)  # [0.5, 0.6, 0.7, 0.8, 0.9]
    percent_dimensions_observed = np.linspace(0.5, 1.0, 5, endpoint=False)  # [0.5, 0.6, 0.7, 0.8, 0.9]

    # Find configurations with best results for DP-GP-LVM vs. Bayesian GP-LVM.
    for mask in dp_masks:
        for n in percent_samples_observed:
            for d in percent_dimensions_observed:
                # mask = 1
                # n = 0.8  # 0.7, 0.9
                # d = 0.7  # 0.8, 0.9
                # desired_configs = ['8820_2_people_seed_{0}_mask_{1}_'.format(i, mask) +
                #                    'n_percent_{0}_d_percent_{1}.npz'.format(n, d) for i in seeds]
                desired_configs = ['8820_4_people_seed_{0}_mask_{1}_'.format(i, mask) +
                                   'n_percent_{0}_d_percent_{1}.npz'.format(n, d) for i in seeds]
                desired_bgplvm_configs = ['bgplvm_posetrack_missing_data_' + i for i in desired_configs]
                desired_mrd_configs = ['mrd_posetrack_missing_data_' + i for i in desired_configs]
                desired_gpdp_configs = ['gpdp_posetrack_missing_data_' + i for i in desired_configs]

                bgplvm_result_files = [np_file for np_file in listdir(results_path)
                                       if np_file in desired_bgplvm_configs]
                mrd_result_files = [np_file for np_file in listdir(results_path)
                                    if np_file in desired_mrd_configs]
                gpdp_result_files = [np_file for np_file in listdir(results_path)
                                     if np_file in desired_gpdp_configs]

                # assert len(bgplvm_result_files) > 0, 'No results found for that configuration.'
                # assert len(bgplvm_result_files) == len(gpdp_result_files), \
                #     'Number of result files should be same for each model.'

                # Print current configuration.
                print('\n--------------------------------------------------------------------------------')
                print('Current Configuration:')
                print('  DP Mask:                          {}'.format(mask))
                print('  Percent samples observed (n%):    {}'.format(n))
                print('  Percent dimensions observed (d%): {}'.format(d))

                if len(bgplvm_result_files) == 0 and len(mrd_result_files) == 0 and len(gpdp_result_files) == 0:
                    print('\nNo results found for this configuration.')
                else:
                    # Get all ground truth log-likelihoods.
                    bgplvm_gt_log_likelihoods = np.array([np.load(results_path + i)['gt_log_likelihood']
                                                          for i in bgplvm_result_files])
                    mrd_gt_log_likelihoods = np.array([np.load(results_path + i)['gt_log_likelihood']
                                                       for i in mrd_result_files])
                    gpdp_gt_log_likelihoods = np.array([np.load(results_path + i)['gt_log_likelihood']
                                                        for i in gpdp_result_files])

                    # Print results.
                    sample_n = len(bgplvm_result_files)
                    sample_std_dev = np.std(bgplvm_gt_log_likelihoods, ddof=1)
                    print('\nBayesian GP-LVM:')
                    print('  Number of test cases: {}'.format(sample_n))
                    if sample_n > 1:
                        print('  Mean:                 {}'.format(np.mean(bgplvm_gt_log_likelihoods)))
                        print('  Std Dev:              {}'.format(sample_std_dev))
                        print('  Std Error:            {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))

                    sample_n = len(mrd_result_files)
                    sample_std_dev = np.std(mrd_gt_log_likelihoods, ddof=1)
                    print('\nMRD:')
                    print('  Number of test cases:   {}'.format(sample_n))
                    if sample_n > 1:
                        print('  Mean:                   {}'.format(np.mean(mrd_gt_log_likelihoods)))
                        print('  Std Dev:                {}'.format(sample_std_dev))
                        print('  Std Error:              {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))

                    sample_n = len(gpdp_result_files)
                    sample_std_dev = np.std(gpdp_gt_log_likelihoods, ddof=1)
                    print('\nDP-GP-LVM:')
                    print('  Number of test cases:   {}'.format(sample_n))
                    if sample_n > 1:
                        print('  Mean:                   {}'.format(np.mean(gpdp_gt_log_likelihoods)))
                        print('  Std Dev:                {}'.format(sample_std_dev))
                        print('  Std Error:              {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))
