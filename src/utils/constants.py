"""
This module defines constant values for the DP-GP-LVM package.
"""

from src.utils.types import NP_DTYPE

from enum import Enum
import numpy as np
from os import getlogin
from socket import gethostname
from sys import path


# Define enums for array names in data set files.
class DataSetKeys(Enum):
    """
    This enumeration class captures the key string values for the various possible numpy arrays from data sets.
    """

    # Data set.
    FULL_DATA_SET = 'full_data_set'
    TRAINING_DATA = 'training_data'
    TEST_DATA = 'test_data'
    OBSERVED_TEST_DATA = 'observed_test_data'
    UNOBSERVED_TEST_DATA = 'unobserved_test_data'
    # TODO: Maybe add randomised and normalised data keys.

    # Set attributes.
    NUM_OBSERVATIONS = 'num_observations'
    NUM_DIMENSIONS = 'num_dimensions'
    NUM_TRAINING_SAMPLES = 'num_training_samples'
    NUM_TEST_SAMPLES = 'num_test_samples'
    NUM_OBSERVED_DIMENSIONS = 'num_observed_dimensions'
    NUM_UNOBSERVED_DIMENSIONS = 'num_unobserved_dimensions'


# Define enums for array names in result files.
class ResultKeys(Enum):
    """
    This enumeration class captures the key string values for the various possible numpy arrays from converged models.
    """

    # Data set.
    ORIGINAL_DATA = 'original_data'
    RANDOMISED_DATA = 'randomised_data'
    NORMALISED_DATA = 'normalised_data'

    # Training data and latent variables for BGP-LVM.
    TRAINING_DATA = 'y_train'
    TRAINING_INPUT_MEAN = 'x_mean'
    TRAINING_INPUT_COVAR = 'x_covar'
    INDUCING_INPUT = 'x_u'

    # Test data and latent variables for BGP-LVM.
    TEST_DATA = 'y_test'
    TEST_INPUT_MEAN = 'x_test_mean'
    TEST_INPUT_COVAR = 'x_test_covar'

    # Kernel hyperparameters.
    ARD_WEIGHTS = 'ard_weights'
    SIGNAL_VARIANCE = 'signal_variance'
    NOISE_PRECISION = 'noise_precision'

    # DP variational parameters.
    DP_ASSIGNMENTS = 'assignments'
    Q_ALPHA_W1 = 'q_alpha_w1'
    Q_ALPHA_W2 = 'q_alpha_w2'
    Q_V_A = 'q_v_a'
    Q_V_B = 'q_v_b'
    ARD_WEIGHTS_ATOMS = 'gamma_atoms'
    SIGNAL_VARIANCE_ATOMS = 'alpha_atoms'
    NOISE_PRECISION_ATOMS = 'beta_atoms'


# Define string constants for file paths and file names.
ABSOLUTE_PATH = [ap for ap in path if 'dp_gp_lvm' in ap][-1]
WEB_IMAGES_RELATIVE_PATH = 'web/assets'
DATA_PATH = ABSOLUTE_PATH + '/src/data_io/data_sets/'
RESULTS_PATH = ABSOLUTE_PATH + '/results/'
RESULTS_FILE_NAME = RESULTS_PATH + '{model}_{dataset}_test.npz'
# Set plot directory based off hostname or user.
PLOTS_PATH = None


# Define constants for optimisation.
OPT_DEFAULT_LEARNING_RATE = 0.05
OPT_DEFAULT_ITERS = 901
OPT_MAX_ITERS = 2501


# Define constants for Monte Carlo integration.
MAX_MC_SAMPLES = 5000  # TODO: Define an appropriate constant to set upper limit of distribution samples.


# Define constants for Gaussian process (GP).
GP_DEFAULT_JITTER = 1.0e-8
GP_INIT_GAMMA = 1.0
GP_INIT_ALPHA = 1.0
GP_INIT_BETA = 1.0
GP_DEFAULT_ALPHA_PRIOR_PARAMS = np.array([1.0, 1.0], dtype=NP_DTYPE)
GP_DEFAULT_BETA_PRIOR_PARAMS = np.array([1001.0, 1.0], dtype=NP_DTYPE)
GP_DEFAULT_ARD_PRIOR_PARAMS = np.array([1.0, 1.0], dtype=NP_DTYPE)


# Define constants for Gaussian process Latent Variable Model (GP-LVM).
GP_LVM_DEFAULT_LATENT_DIMENSIONS = 10
GP_LVM_DEFAULT_NUM_INDUCING_POINTS = 25
GP_LVM_MAX_LATENT_DIMENSIONS = 25


# Define constants for Manifold Relevance Determination (MRD).
MRD_DEFAULT_EPSILON = 1.0e-3


# Define constants for Dirichlet process (DP).
DP_MAX_NUM_SAMPLES = 5000
DP_MAX_TRUNCATION_LEVEL = 50
DP_DEFAULT_ALPHA = 1.0
DP_DEFAULT_ALPHA_PRIOR_PARAMS = np.array([1.0, 1.0], dtype=NP_DTYPE)
# DP_DEFAULT_ALPHA_PRIOR_PARAMS = np.array([10.0, 5.0], dtype=NP_DTYPE)
DP_DEFAULT_TRUNCATION_LEVEL = 8


# Matplotlib backends.
# See: http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
#      http://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
MATPLOTLIB_DEFAULT_BACKEND = 'macosx'
MATPLOTLIB_DEFAULT_INTERACTIVE_BACKEND = 'macosx'
MATPLOTLIB_DEFAULT_NON_INTERACTIVE_BACKEND = 'agg'
MATPLOTLIB_DEFAULT_REMI_WEB_BACKEND = 'agg'
MATPLOTLIB_PNG_BACKEND = 'agg'
MATPLOTLIB_PDF_BACKEND = 'pdf'
MATPLOTLIB_GUI_BACKEND = 'macosx'


# Random seeds.
NP_SEED = 1
TF_SEED = 87654321  # This is default seed in TensorFlow source code as of r1.10.1.
# https://github.com/tensorflow/tensorflow/blob/v1.10.1/tensorflow/python/framework/random_seed.py


# Style colours.
UOB_BLUE = '#012169'
UOB_GREY = '#333F48'
