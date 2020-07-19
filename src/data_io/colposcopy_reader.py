"""
This module reads and parses the skin cancer MNIST data set captured in CSV files. It does not read the image files.
"""

from src.utils.constants import DATA_PATH
from src.utils.types import NP_DTYPE
import src.visualisation.plotters as vis

from csv import reader as csv_reader
import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile


def read_colposcopy_csv(file_path):
    """
    TODO
    :param file_path:
    :return:
    """

    assert isfile(file_path), 'Specified file does not exist.'

    with open(file_path, newline='') as csv_file:
        reader = csv_reader(csv_file)
        counter = 0

        predictive_attributes = None
        assessments = None

        for row in reader:
            # First row is just labels so can ignore it.
            if counter > 0:

                if predictive_attributes is None:
                    predictive_attributes = np.array(row[:-7], dtype=NP_DTYPE)
                else:
                    predictive_attributes = np.vstack((predictive_attributes, np.array(row[:-7], dtype=NP_DTYPE)))

                # Last 7 columns are expert assessments, with final column being the consensus.
                if assessments is None:
                    assessments = np.array(row[-7:], dtype=NP_DTYPE)
                else:
                    assessments = np.vstack((assessments, np.array(row[-7:], dtype=NP_DTYPE)))
            else:
                counter += 1  # Only need to increase once as we don't care after the first row.

        assert predictive_attributes.shape[0] == assessments.shape[0], \
            'Number of attributes and assessments is not the same.'
        assert predictive_attributes.shape[1] == 62, 'Number of attributes does not match expected value of 62.'
        assert assessments.shape[1] == 7, 'Number of assessment does not match expected value of 7.'

    return predictive_attributes, assessments


def get_colposcopy_path():
    return DATA_PATH + 'colposcopy/'


def read_hinselmann_modality():
    """
    TODO
    :return:
    """

    return read_colposcopy_csv(get_colposcopy_path() + 'hinselmann.csv')


def read_green_modality():
    """
    TODO
    :return:
    """

    return read_colposcopy_csv(get_colposcopy_path() + 'green.csv')


def read_schiller_modality():
    """
    TODO
    :return:
    """

    return read_colposcopy_csv(get_colposcopy_path() + 'schiller.csv')


if __name__ == '__main__':

    # Define file path for data.
    colposcopy_data_path = DATA_PATH + 'colposcopy/'

    # Read each modality.
    hinselmann_attributes, hinselmann_assessments = read_hinselmann_modality()
    green_attributes, green_assessments = read_green_modality()
    schiller_attributes, schiller_assessments = read_schiller_modality()

    # Print shapes
    print('Hinselmann Modality:')
    print('  Attributes: {}'.format(hinselmann_attributes.shape))
    print('  Assessments: {}'.format(hinselmann_assessments.shape))
    print('Green Modality:')
    print('  Attributes: {}'.format(green_attributes.shape))
    print('  Assessments: {}'.format(green_assessments.shape))
    print('Schiller Modality:')
    print('  Attributes: {}'.format(schiller_attributes.shape))
    print('  Assessments: {}'.format(schiller_assessments.shape))

    # Plot data.
    show_plots = False
    if show_plots:
        vis.plot_data(hinselmann_attributes)
        vis.plot_data(green_attributes)
        vis.plot_data(schiller_attributes)
        vis.plot_phi_matrix(hinselmann_assessments, add_labels=False)
        vis.plot_phi_matrix(green_assessments, add_labels=False)
        vis.plot_phi_matrix(schiller_assessments, add_labels=False)

        plot.show()

    from sklearn.preprocessing import StandardScaler
    from src.utils.expressions import principal_component_analysis as pca
    scaler = StandardScaler()
    h_data = scaler.fit_transform(np.hstack((hinselmann_attributes, hinselmann_assessments)))
    g_data = scaler.fit_transform(np.hstack((green_attributes, green_assessments)))
    s_data = scaler.fit_transform(np.hstack((schiller_attributes, schiller_assessments)))

    num_latent_dimensions = 15
    h_pca = pca(h_data, num_latent_dimensions=num_latent_dimensions)
    g_pca = pca(g_data, num_latent_dimensions=num_latent_dimensions)
    s_pca = pca(s_data, num_latent_dimensions=num_latent_dimensions)

    for i in range(10, 15):
        plot.figure()
        plot.subplot(211)
        plot.hist(hinselmann_attributes[:, i])
        plot.subplot(212)
        plot.hist(h_data[:, i])
        plot.suptitle('Hinselmann Data: Dim {}'.format(i))

    # plot.figure()
    # plot.scatter(h_pca[:, 0], h_pca[:, 1])
    # plot.figure()
    # plot.scatter(g_pca[:, 0], g_pca[:, 1])
    # plot.figure()
    # plot.scatter(s_pca[:, 0], s_pca[:, 1])

    # plot.figure()
    # plot.imshow(np.dot(h_data.T, h_data))
    # plot.figure()
    # plot.imshow(np.dot(g_data.T, g_data))
    # plot.figure()
    # plot.imshow(np.dot(s_data.T, s_data))
    #
    # from scipy.linalg import eig
    # eig_h = eig(np.dot(h_data.T, h_data))
    # eig_g = eig(np.dot(g_data.T, g_data))
    # eig_s = eig(np.dot(s_data.T, s_data))

    # plot.figure()
    # plot.imshow(np.dot(h_data, h_data.T))
    # plot.figure()
    # plot.imshow(np.dot(g_data, g_data.T))
    # plot.figure()
    # plot.imshow(np.dot(s_data, s_data.T))

    plot.show()


