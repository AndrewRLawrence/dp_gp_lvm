"""
This module defines constants and functions for managing data types in numpy and TensorFlow.
"""

from src.kernels.interfaces.kernel import Kernel
from src.models.interfaces.trainable import Trainable

import numpy as np
import tensorflow as tf


# Type constants.
TF_DTYPE = tf.float64
NP_DTYPE = np.float64

# # Confirm that the TensorFlow and numpy types agree.
# assert TF_DTYPE.is_numpy_compatible, 'TensorFlow dtype must be compatible with numpy.'
# assert TF_DTYPE.as_numpy_dtype is NP_DTYPE, 'TensorFlow dtype must be equivalent to the numpy dtype.'


def get_training_variables():
    """
    This function returns a list of the 'trainable' TensorFlow variables.
    :return: A list of all the nodes of 'trainable' TensorFlow variables.
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


def get_prediction_variables():
    """
    This function returns a list of the non-'trainable' TensorFlow variables. These are the ones optimized during
    testing, i.e., prediction/inference.
    :return: A list of all the nodes of non-'trainable' TensorFlow variables.
    """
    # Remove all trainable variables from list of all global variables.
    return list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) -
                set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))


def create_positive_variable(initial_value, shape=None, is_trainable=True):
    """
    This function creates a positive TensorFlow node encapsulating a TensorFlow Variable with the provided initial
    value.
    :param initial_value: The initial value.
    :param shape: The desired shape of the node.
    :param is_trainable: If True, the default, also adds the variable to the graph collection
    GraphKeys.TRAINABLE_VARIABLES.
    :return: A TensorFlow node that is constrained to positive values and encapsulates a TF Variable.
    """

    assert initial_value > 0, 'Initial value must be positive.'
    variable_init_value = np.log(np.exp(initial_value) - 1.0) * np.ones(shape=shape, dtype=NP_DTYPE)

    # TODO: Maybe add some noise
    # variable_init_value = np.random.normal(loc=np.log(np.exp(initial_value) - 1.0), scale=0.01, size=shape)

    return tf.nn.softplus(tf.Variable(initial_value=variable_init_value, dtype=TF_DTYPE, trainable=is_trainable))


def create_random_positive_variable(shape, is_trainable=True):
    """
    This function creates a positive TensorFlow node encapsulating a TensorFlow Variable of the provided shape and
    initialised randomly.
    :param shape: The desired shape of the node.
    :param is_trainable: If True, the default, also adds the variable to the graph collection
    GraphKeys.TRAINABLE_VARIABLES.
    :return: A TensorFlow node of the provided shape that is constrained to be positive and encapsulates a TF Variable.
    """

    return tf.nn.softplus(tf.Variable(initial_value=np.random.standard_normal(size=shape),
                                      dtype=TF_DTYPE,
                                      trainable=is_trainable))


def validate_positive(x, dtype=int):
    """
    This function validates that the provided input (x) is positive and of the specified type.
    :param x: The value to be validated.
    :param dtype: The desired type of the input.
    """

    assert isinstance(dtype, type), 'Specified dtype is not a valid type.'
    assert isinstance(x, dtype), 'The input must be of type {}.'.format(dtype)
    assert (x > 0), 'The input must be positive.'


def validate_np_array(x, num_dims=None, shape=None, dtype=None, flatten=False, is_positive=False):
    """
   This function validates that the provided input (x) is a valid numpy array and matches the number of dimensions,
   shape, and/or type if those are provided.
   :param x: The numpy array to be validated.
   :param num_dims: The desired number of dimensions of the numpy array.
   :param shape: The desired shape of the numpy array.
   :param dtype: The desired type of the elements in the numpy array.
   :param flatten: A boolean to set whether or not to flatten the array when validating it.
   :param is_positive: A boolean to set whether or not to check if all elements are positive.
   """

    assert isinstance(x, np.ndarray), 'Input is not a numpy array.'

    # Flatten, if necessary.
    if flatten:
        x = x.flatten()

    # Validate number of dimensions of x, if provided.
    if num_dims is not None:
        assert isinstance(num_dims, int), 'Number of dimensions must be provided as an integer.'
        assert num_dims > 0, 'Number of dimensions must be positive.'
        assert x.ndim == num_dims, 'Number of dimensions of input array does not match specified number of dimensions.'

    # Validate shape of x, if provided.
    if shape is not None:
        assert isinstance(shape, tuple) or isinstance(shape, list), 'Shape must be provided as a list or tuple.'
        assert all([isinstance(i, int) for i in shape]), 'Values in shape must be provided as integers.'
        assert all([i > 0 for i in shape]), 'Values in shape must be positive.'
        assert len(shape) == x.ndim, 'Shape does not match the number of dimensions of input array.'

        if isinstance(shape, list):
            shape = tuple(shape)
        assert x.shape == shape, 'Shape of input array does not match specified shape.'

    # Validate type of elements in x, if provided.
    if dtype is not None:
        assert isinstance(dtype, type), 'Specified dtype is not a valid type.'
        assert all([isinstance(i, dtype) for i in x.flatten()]), \
            'All elements of input array must be of the specified dtype.'

    # Validate positive, if necessary.
    if is_positive:
        assert np.all(x > 0), 'All elements of input array must be greater than zero.'


def validate_model(model):
    """
    This function validates that the provided input is an instance of a class that inherets from the Trainable abstract
    base class.
    :param model: The object to be validated.
    """

    assert isinstance(model, Trainable), \
        'Provided model is not an instance of a class that inherets from the Trainable abstract base class.'


def validate_kernel(kernel):
    """
    This function validates that the provided input is an instance of the Kernel class, which inherets from the
    AbstractKernel abstract base class.
    :param kernel: The object to be validated.
    """

    assert isinstance(kernel, Kernel), 'Provided kernel is not an instance of the Kernel class.'
