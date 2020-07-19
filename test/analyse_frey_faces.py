"""
This module analyses the results for the missing data tests with the predictive posterior from the Bayesian GP-LVM, MRD,
and DP-GP-LVM models. This module analyses results from tests with the Frey Faces data.
"""

from src.utils.constants import RESULTS_PATH

import numpy as np
from os import listdir

if __name__ == '__main__':

    # Print results for 50% random pixels missing. Compares DP-GP-LVM and Bayesian GP-LVM.
    bgplvm_gt_log_likelihoods = None
    dp_gp_lvm_gt_log_likelihoods = None

    # Find configurations with best results for DP-GP-LVM vs. Bayesian GP-LVM.
    for s in np.arange(10):
        # Define data set string.
        dataset_str = 'frey_faces_50_missing_data_seed{}'.format(s)
        bayesian_gp_lvm_results_file = 'bgplvm_' + dataset_str
        # bayesian_gp_lvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm', dataset=dataset_str)
        dp_gp_lvm_results_file = 'dp_gp_lvm_' + dataset_str
        # dp_gp_lvm_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)

        # Find results for each model.
        bgplvm_result_files = [np_file for np_file in listdir(RESULTS_PATH) if bayesian_gp_lvm_results_file in np_file]
        dp_gp_lvm_result_files = [np_file for np_file in listdir(RESULTS_PATH) if dp_gp_lvm_results_file in np_file]

        # # Print current configuration.
        # print('\n--------------------------------------------------------------------------------')
        # print('Current Configuration:')
        # print('  Seed: {}'.format(s))

        if len(bgplvm_result_files) == 0 and len(dp_gp_lvm_result_files) == 0:
            # Print current configuration.
            print('\n--------------------------------------------------------------------------------')
            print('Current Configuration:')
            print('  Seed: {}'.format(s))
            print('\nNo results found for this configuration.')
        else:
            # Get all ground truth log-likelihoods.
            if bgplvm_gt_log_likelihoods is None:
                bgplvm_gt_log_likelihoods = np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                      for i in bgplvm_result_files]).flatten()
            else:
                bgplvm_gt_log_likelihoods = np.hstack((bgplvm_gt_log_likelihoods,
                                                       np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                                 for i in bgplvm_result_files]).flatten()))

            if dp_gp_lvm_gt_log_likelihoods is None:
                dp_gp_lvm_gt_log_likelihoods = np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                         for i in dp_gp_lvm_result_files]).flatten()
            else:
                dp_gp_lvm_gt_log_likelihoods = np.hstack((dp_gp_lvm_gt_log_likelihoods,
                                                          np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                                    for i in dp_gp_lvm_result_files]).flatten()))

            # # Get all ground truth log-likelihoods.
            # if bgplvm_gt_log_likelihoods is None:
            #     bgplvm_gt_log_likelihoods = np.array([np.load(RESULTS_PATH + i)['gt_log_likelihood']
            #                                           for i in bgplvm_result_files])
            # else:
            #     bgplvm_gt_log_likelihoods = np.hstack((bgplvm_gt_log_likelihoods,
            #                                            np.array([np.load(RESULTS_PATH + i)['gt_log_likelihood']
            #                                                      for i in bgplvm_result_files])))
            #
            # if dp_gp_lvm_gt_log_likelihoods is None:
            #     dp_gp_lvm_gt_log_likelihoods = np.array([np.load(RESULTS_PATH + i)['gt_log_likelihood']
            #                                              for i in dp_gp_lvm_result_files])
            # else:
            #     dp_gp_lvm_gt_log_likelihoods = np.hstack((dp_gp_lvm_gt_log_likelihoods,
            #                                               np.array([np.load(RESULTS_PATH + i)['gt_log_likelihood']
            #                                                         for i in dp_gp_lvm_result_files])))

    # Print results.
    sample_n = bgplvm_gt_log_likelihoods.size
    sample_std_dev = np.std(bgplvm_gt_log_likelihoods, ddof=1)
    print('\n--------------------------------------------------------------------------------')
    print('\nBayesian GP-LVM:')
    print('  Number of test cases: {}'.format(sample_n))
    if sample_n > 1:
        print('  Mean:                 {}'.format(np.mean(bgplvm_gt_log_likelihoods)))
        print('  Std Dev:              {}'.format(sample_std_dev))
        print('  Std Error:            {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))

    sample_n = dp_gp_lvm_gt_log_likelihoods.size
    sample_std_dev = np.std(dp_gp_lvm_gt_log_likelihoods, ddof=1)
    print('\nDP-GP-LVM:')
    print('  Number of test cases:   {}'.format(sample_n))
    if sample_n > 1:
        print('  Mean:                   {}'.format(np.mean(dp_gp_lvm_gt_log_likelihoods)))
        print('  Std Dev:                {}'.format(sample_std_dev))
        print('  Std Error:              {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))

    # Print results for bottom half of face missing. Compares DP-GP-LVM and MRD.
    mrd_gt_log_likelihoods = None
    dp_gp_lvm_gt_log_likelihoods = None

    # Find configurations with best results for DP-GP-LVM vs. MRD.
    for s in np.arange(5):
        # Define data set string.
        dataset_str = 'frey_faces_bottom_missing_data_seed{}'.format(s)
        mrd_results_file = 'mrd_' + dataset_str
        dp_gp_lvm_results_file = 'dp_gp_lvm_' + dataset_str

        # Find results for each model.
        mrd_result_files = [np_file for np_file in listdir(RESULTS_PATH) if mrd_results_file in np_file]
        dp_gp_lvm_result_files = [np_file for np_file in listdir(RESULTS_PATH) if dp_gp_lvm_results_file in np_file]

        if len(mrd_result_files) == 0 and len(dp_gp_lvm_result_files) == 0:
            # Print current configuration.
            print('\n--------------------------------------------------------------------------------')
            print('Current Configuration:')
            print('  Seed: {}'.format(s))
            print('\nNo results found for this configuration.')
        else:
            # Get all ground truth log-likelihoods.
            if mrd_gt_log_likelihoods is None:
                mrd_gt_log_likelihoods = np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                   for i in mrd_result_files]).flatten()
            else:
                mrd_gt_log_likelihoods = np.hstack((mrd_gt_log_likelihoods,
                                                    np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                              for i in mrd_result_files]).flatten()))

            if dp_gp_lvm_gt_log_likelihoods is None:
                dp_gp_lvm_gt_log_likelihoods = np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                         for i in dp_gp_lvm_result_files]).flatten()
            else:
                dp_gp_lvm_gt_log_likelihoods = np.hstack((dp_gp_lvm_gt_log_likelihoods,
                                                          np.array([np.load(RESULTS_PATH + i)['gt_log_likelihoods']
                                                                    for i in dp_gp_lvm_result_files]).flatten()))

    # Print results.
    sample_n = mrd_gt_log_likelihoods.size
    sample_std_dev = np.std(mrd_gt_log_likelihoods, ddof=1)
    print('\n--------------------------------------------------------------------------------')
    print('\nMRD:')
    print('  Number of test cases: {}'.format(sample_n))
    if sample_n > 1:
        print('  Mean:                 {}'.format(np.mean(mrd_gt_log_likelihoods)))
        print('  Std Dev:              {}'.format(sample_std_dev))
        print('  Std Error:            {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))

    sample_n = dp_gp_lvm_gt_log_likelihoods.size
    sample_std_dev = np.std(dp_gp_lvm_gt_log_likelihoods, ddof=1)
    print('\nDP-GP-LVM:')
    print('  Number of test cases:   {}'.format(sample_n))
    if sample_n > 1:
        print('  Mean:                   {}'.format(np.mean(dp_gp_lvm_gt_log_likelihoods)))
        print('  Std Dev:                {}'.format(sample_std_dev))
        print('  Std Error:              {}'.format(sample_std_dev / np.sqrt(1.0 * sample_n)))
