"""
This module tests the predictive posterior for missing data to see how well DP-GP-LVM works with predicting expert
labels for colposcopy images.
"""

from src.utils.constants import RESULTS_FILE_NAME
import src.visualisation.plotters as vis

from itertools import permutations
import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    # # Read each modality.
    # hinselmann_attributes, hinselmann_assessments = read_hinselmann_modality()
    # hinselmann_data = np.hstack((hinselmann_attributes, hinselmann_assessments))
    # green_attributes, green_assessments = read_green_modality()
    # green_data = np.hstack((green_attributes, green_assessments))
    # schiller_attributes, schiller_assessments = read_schiller_modality()
    # schiller_data = np.hstack((schiller_attributes, schiller_assessments))
    #
    # # Normalise data to zero mean and unit variance.
    # scaler = StandardScaler()
    # normalized_hinselmann_data = scaler.fit_transform(hinselmann_data)
    # normalized_green_data = scaler.fit_transform(green_data)
    # normalized_schiller_data = scaler.fit_transform(schiller_data)

    # Define all possible {source, target} pairs from list of modalities.
    modalities = ['h', 'g', 's']
    tasks = permutations(modalities, r=2)

    # Loop through each task.
    for (source, target) in tasks:

        # Define file path for results.
        dataset_str = 'colposcopy_{}_{}'.format(source, target)
        dp_gp_lvm_results_file = RESULTS_FILE_NAME.format(model='dp_gp_lvm', dataset=dataset_str)

        # Visualise results if they exist.
        if isfile(dp_gp_lvm_results_file):
            results = np.load(dp_gp_lvm_results_file)

            ground_truth = results['y_test_unobserved'][:, :-1]
            predicted_mean_assessments = results['predicted_mean'][:, :-1]

            plot.imshow(ground_truth, interpolation='nearest', aspect='auto',
                        extent=(0, ground_truth.shape[1], ground_truth.shape[0], 0), origin='upper')
            plot.colorbar()
            plot.xlabel('Expert')
            plot.ylabel('Observation')
            plot.title('Ground Truth - {} to {}'.format(source, target))
            plot.figure()
            plot.imshow(predicted_mean_assessments, interpolation='nearest', aspect='auto',
                        extent=(0, predicted_mean_assessments.shape[1], predicted_mean_assessments.shape[0], 0),
                        origin='upper')
            plot.colorbar()
            plot.xlabel('Expert')
            plot.ylabel('Observation')
            plot.title('Predicted Assessments - {} to {}'.format(source, target))

            vis.plot_ard(np.sqrt(results['ard_weights']))
            plot.title('ARD Weights - {} to {}'.format(source, target))

            # ground_truth = np.array(results['y_test_unobserved'] >= 0.0).astype(int)
            # predicted_mean_assessments = np.array(results['predicted_mean'] >= 0.0).astype(int)
            # # predicted_mean_assessments = np.array((results['predicted_mean'] -
            # #                                        np.mean(results['predicted_mean'])) >= 0.0).astype(int)
            #
            # # Ignore last column as it is consensus rating but just want to look at 6 experts.
            # if ground_truth.shape[1] > 1 and predicted_mean_assessments.shape[1] > 1:
            #     ground_truth = ground_truth[:, :-1]
            #     predicted_mean_assessments = predicted_mean_assessments[:, :-1]
            #
            # vis.plot_phi_matrix(ground_truth, add_labels=False)
            # plot.title('Ground Truth - {} to {}'.format(source, target))
            # vis.plot_phi_matrix(predicted_mean_assessments, add_labels=False)
            # plot.title('Predicted Assessments - {} to {}'.format(source, target))
            # vis.plot_ard(np.sqrt(results['ard_weights']))
            # plot.title('ARD Weights - {} to {}'.format(source, target))
            #
            # for expert in range(6):
            #     confusion_mat = confusion_matrix(ground_truth[:, expert], predicted_mean_assessments[:, expert])
            #     plot.figure()
            #     plot.imshow(confusion_mat)
            #     plot.colorbar()
            #     plot.title('Confusion Matrix - {} to {} - Expert {}'.format(source, target, expert))

    # Show plots at the end.
    plot.show()
