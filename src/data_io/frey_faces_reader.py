"""
This module reads and parses the Frey faces data set captured in a MAT file.
"""

from src.utils.constants import DATA_PATH
from src.utils.types import NP_DTYPE

from scipy.io import loadmat
import matplotlib.pyplot as plot
import numpy as np
from os.path import isfile


def read_mat_file(file_path):
    """
    TODO
    :param file_path:
    :return:
    """

    assert isfile(file_path), 'Specified file does not exist.'

    return loadmat(file_path)


def im2double(image, bit_depth=8):
    """
    TODO
    :param image:
    :param bit_depth:
    :return:
    """

    # min_val = 0.0
    max_val = np.power(2, bit_depth).astype('float64') - 1.0

    return np.array(image, dtype=NP_DTYPE) / max_val


def get_frey_path():
    """The function returns the full path for the Frey faces data set."""
    return DATA_PATH + 'frey_faces/'


def read_frey_mat():
    """
    TODO
    :return:
    """

    # Read MAT file.
    mat_contents_dict = read_mat_file(get_frey_path() + 'frey_rawface.mat')

    # Data is under key 'ff'.
    faces = mat_contents_dict['ff']  # [560 x 1965] uint8 numpy array.

    # Number of faces is 1965 and each image is [20 x 28] pixels, which is 560 pixels. Return as [N x D] float64 array.
    return np.transpose(im2double(faces))


if __name__ == '__main__':

    # Read mat file.
    data = read_frey_mat()

    seed = 1
    num_samples = 5
    np.random.seed(seed=seed)  # Random seed.

    # Visualise some random face images.
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    for i in indices:
        plot.figure()
        plot.imshow(data[i, :].reshape(28, 20), cmap='gray', vmin=0.0, vmax=1.0)
        plot.title('Face {}'.format(i))

    # Plot histograms to see if random pixels are normally distributed across the different faces.
    pixels = np.random.choice(data.shape[1], size=num_samples, replace=False)
    for i in pixels:
        plot.figure()
        plot.hist(data[:, i])
        plot.title('Pixel {}'.format(i))

    # Show plots.
    plot.show()
