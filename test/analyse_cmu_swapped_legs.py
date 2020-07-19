"""
This module analyses the results of the concatenated CMU walking data set with normal and swapped legs test on the
various models.
"""

from src.utils.constants import ResultKeys, RESULTS_FILE_NAME, PLOTS_PATH
import src.visualisation.plotters as vis

import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile


if __name__ == '__main__':

    # Define booleans. TODO: Allow them to be set from command line.
    save_plots = False
    show_plots = True

    dataset_str = 'cmu_walking_normal_swapped'

    bgplvm_results_file = RESULTS_FILE_NAME.format(model='bgplvm', dataset=dataset_str)

    # 93 dims per view.
    mrd_results_file = RESULTS_FILE_NAME.format(model='mrd', dataset=dataset_str)
    # 3 dims per view so keep 3d points together.
    mrd_fully_independent_results_file = RESULTS_FILE_NAME.format(model='mrd_fully_independent', dataset=dataset_str)

    # Keep 3d points together so mask of 3.
    gpdp_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)
    # Keep skeletons together so mask of 93.
    gpdp_mask_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm_mask_93', dataset=dataset_str)

    # Load result files.

