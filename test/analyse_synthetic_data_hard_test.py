"""
This module analyses the results for the hard synthetic data tests.
"""

from utils.constants import NP_DTYPE
from visualisation.plotters import plot_ard, plot_phi

from matplotlib2tikz import save as tikz_save
import matplotlib.pyplot as plot
import numpy as np
from sys import path

# Known structure.
ard_1 = np.tile(np.concatenate((np.ones((1, 2), dtype=NP_DTYPE),
                                np.zeros((1, 8), dtype=NP_DTYPE)),
                               axis=1),
                (15, 1))  # Function of X0 and X1
ard_2 = np.tile(np.concatenate((np.ones((1, 1), dtype=NP_DTYPE),
                                np.zeros((1, 1), dtype=NP_DTYPE),
                                np.ones((1, 1), dtype=NP_DTYPE),
                                np.zeros((1, 7), dtype=NP_DTYPE)),
                               axis=1),
                (15, 1))  # Function of X0 and X2.
ard_3 = np.tile(np.concatenate((np.zeros((1, 1), dtype=NP_DTYPE),
                                np.ones((1, 1), dtype=NP_DTYPE),
                                np.zeros((1, 1), dtype=NP_DTYPE),
                                np.ones((1, 1), dtype=NP_DTYPE),
                                np.zeros((1, 6), dtype=NP_DTYPE)),
                               axis=1),
                (15, 1))  # Function of X1 and X3.
ard_4 = np.tile(np.concatenate((np.ones((1, 1), dtype=NP_DTYPE),
                                np.zeros((1, 2), dtype=NP_DTYPE),
                                np.ones((1, 1), dtype=NP_DTYPE),
                                np.zeros((1, 6), dtype=NP_DTYPE)),
                               axis=1),
                (15, 1))  # Function of X0 and X3.

known_ard = np.concatenate((ard_1.T, ard_2.T, ard_3.T, ard_4.T), axis=1)

# Define paths.
absolute_path = [ap for ap in path if 'aistats_2019' in ap]
results_path = absolute_path[-1] + '/test/results/'

# Analyse results.
more_noise_results = np.load(results_path + 'gpdp_hard_synthetic_data_test_noise_precision_100.npz')
less_noise_results = np.load(results_path + 'gpdp_hard_synthetic_data_test_noise_precision_10.npz')

print(more_noise_results.keys())

plot_ard(known_ard.T)
plot.title('Known Latent Structure of Synthetic Data')
# plot.title('Ground Truth ARD Weights of Synthetic Data')
ax = plot.gca()
ax.set_xticklabels([])
plot.savefig('ground_truth_hard_synthetic_data.pdf', bbox_inches='tight')
tikz_save('ground_truth_hard_synthetic_data.tex')

# more_noise_scaler = RobustScaler()
# test = more_noise_scaler.fit_transform(more_noise_results['ard_weights'])
# plot_ard(test)
more_noise_ard = more_noise_results['ard_weights'] / np.max(more_noise_results['ard_weights'])
plot_ard(more_noise_ard)
# plot_ard(more_noise_results['ard_weights'])
plot.title('Learned ARD Weights from Noisy, Synthetic Data')
ax = plot.gca()
ax.set_xticklabels([])
plot.savefig('gpdp_ard_weights_hard_synthetic_data_noise_precision_10.pdf', bbox_inches='tight')
tikz_save('gpdp_ard_weights_hard_synthetic_data_noise_precision_10.tex')

# There is one massive outlier so use second biggest max.
sorted_less_noise_ard = np.sort(less_noise_results['ard_weights'].flatten())
less_noise_ard = less_noise_results['ard_weights'] / sorted_less_noise_ard[-2]
plot_ard(less_noise_ard)
# plot_ard(less_noise_results['ard_weights'])
plot.title('Learned ARD Weights from Less Noisy, Synthetic Data')
ax = plot.gca()
ax.set_xticklabels([])
plot.savefig('gpdp_ard_weights_hard_synthetic_data_noise_precision_100.pdf', bbox_inches='tight')
tikz_save('gpdp_ard_weights_hard_synthetic_data_noise_precision_100.tex')

plot.show()


