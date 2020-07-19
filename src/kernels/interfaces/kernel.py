"""
This module defines an enumeration for the types of kernel hyperparameters, the abstract base class for a Kernel object,
and a generic Kernel class.
"""

from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf


class KernelHyperparameters(Enum):
    """
    This enumeration class captures the key string values for the various possible kernel hyperparameters.
    """
    ARD_WEIGHTS = 'gamma'
    SIGNAL_VARIANCE = 'alpha'
    NOISE_PRECISION = 'beta'
    FREQUENCY = 'freq'
    PERIOD = 'period'
    LENGTH_SCALES = 'l'
    LINEAR_WEIGHTS = 'W'


class AbstractKernel(ABC):
    """
    This abstract base class represents any kernel function that can produce a valid covariance matrix. Therefore, the
    kernel features some hyperparameters and a function to calculate the full covariance matrix and the diagonal of
    the covariance matrix.
    """

    @property
    @abstractmethod
    def prior_log_likelihood(self):
        """
        This abstract property represents the log-likelihood for the prior distributions on the hyperparameters.
        :return: The log-likelihood for the prior distributions on the hyperparameters.
        """
        pass

    @property
    @abstractmethod
    def noise_precision(self):
        """
        This abstract property represents the noise precision for the spherical noise model.
        :return: The noise precision for the spherical noise model.
        """
        pass

    @property
    @abstractmethod
    def hyperparameters(self):
        """
        This abstract property represents the hyperparameters for the kernel function.
        :return: The hyperparameters of the kernel function. Potentially as a dictionary.
        """
        pass

    @abstractmethod
    def covariance_matrix(self, input_0, input_1=None, include_noise=False, include_jitter=False):
        """
        This abstract method represents the function for calculating the covariance matrix created by the kernel
        function with inputs 0 and 1.
        :param input_0: The first input. Must be [N0 x Q].
        :param input_1: An optional second input; input_0 is used again if input_1 is not provided. Must be [N1 x Q].
        :param include_noise: An optional boolean to define whether or not to include noise along the diagonal.
        No noise is default.
        :param include_jitter: An optional boolean to define whether or not to include jitter along the diagonal.
        No jitter is default.
        :return: A valid covariance matrix of size [N0 x N1].
        """
        pass

    @abstractmethod
    def covariance_diag(self, num_samples, include_noise=False, include_jitter=False):
        """
        This abstract method represents the function for calculating the diagonal of the covariance matrix created by
        the kernel function. We are currently only looking at stationary kernels so the input is not necessary.
        :param num_samples: The number of samples, N, which must be postive.
        :param include_noise: An optional boolean to define whether or not to include noise along the diagonal.
        No noise is default.
        :param include_jitter: An optional boolean to define whether or not to include jitter along the diagonal.
        No jitter is default.
        :return: A valid diagonal from the covariance matrix of size [N, ].
        """
        pass

    @abstractmethod
    def psi_0(self, num_samples):
        """
        This abstract method represents the function for calculating the psi_0 statistic, which is the trace of an
        expectation of a covariance matrix (Kff) over q(X) and involves convolutions of the kernel function with
        Gaussian densities.
        :param num_samples: The number of samples, N, which must be postive.
        :return: The psi_0 statistic.
        """
        pass

    @abstractmethod
    def psi_1(self, inducing_input, latent_input_mean, latent_input_covariance):
        """
        This abstract method represents the function for calculating the psi_1 statistic, which is an expectation of a
        covariance matrix (Kfu) over q(X) and involves convolutions of the kernel function with Gaussian densities.
        :param inducing_input: The inducing input. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: The psi_1 statistic.
        """
        pass

    @abstractmethod
    def psi_2(self, inducing_input, latent_input_mean, latent_input_covariance):
        """
        This abstract method represents the function for calculating the psi_2 statistic, which is an expectation of a
        covariance matrix (Kuf x Kfu) over q(X) and involves convolutions of the kernel function with Gaussian
        densities.
        :param inducing_input: The inducing input. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: The psi_2 statistic.
        """
        pass


class Kernel(AbstractKernel):
    """
    This class defines a generic structure for instantiating a Kernel object.
    """

    def __init__(self, covar_matrix_func, covar_diag_func, hyperparameter_dict, hyperprior_func_dict,
                 psi_0_func=None, psi_1_func=None, psi_2_func=None):
        """
        The constructor defines a valid Kernel object when provided with appropriate covariance matrix and covariance
        diagonal functions and corresponding dictionaries for the necessary hyperparameters and hyperpriors.
        :param covar_matrix_func: A function for defining a full covariance matrix.
        :param covar_diag_func: A function for defining the diagonal of a covariance matrix.
        :param hyperparameter_dict: A dictionary providing the kernel hyperparameters.
        :param hyperprior_func_dict: A dictionary providing the log-likelihood for the priors on the kernel
        hyperparameters.
        :param psi_0_func: A function for defining the psi 0 statistic.
        :param psi_1_func: A function for defining the Psi 1 statistic.
        :param psi_2_func: A function for defining the Psi 2 statistic.
        """

        # Assert inputs are of correct type.
        assert callable(covar_matrix_func), 'Covariance matrix function must be callable.'
        assert callable(covar_diag_func), 'Covariance diagonal function must be callable.'
        assert isinstance(hyperparameter_dict, dict)
        assert isinstance(hyperprior_func_dict, dict)

        # Assert that all keys are type of KernelHyperparameters enumeration.
        assert all([isinstance(hp, KernelHyperparameters) for hp in hyperparameter_dict.keys()]), \
            'All dictionary keys must be of type KernelHyperparameters enumeration.'

        # Assert that both dictionaries have the same keys.
        assert set(hyperparameter_dict.keys()) == set(hyperprior_func_dict.keys()), \
            'Both dictionaries must have the same keys.'

        # Assert that all functions in hyperprior_func_dict are callable.
        assert all([callable(prior_func) for prior_func in hyperprior_func_dict.values()]), \
            'All hyperprior functions must be callable.'

        # TODO: Should validate inputs further.

        self._covar_matrix_func = covar_matrix_func
        self._covar_diag_func = covar_diag_func
        self._hyperparameter_dict = hyperparameter_dict
        self._hyperprior_dict = hyperprior_func_dict
        self._psi_0_func = psi_0_func
        self._psi_1_func = psi_1_func
        self._psi_2_func = psi_2_func

        self.hyperprior_log_likelihood = tf.reduce_sum([tf.reduce_sum(hyperprior_func_dict[hp](hyperparameter_dict[hp]))
                                                        for hp in hyperparameter_dict.keys()])

    @property
    def prior_log_likelihood(self):
        """
        This property provides the sum of the log-likelihoods of each prior distribution for each hyperparameter.
        :return: A scalar value of the sum of the log-likelihoods of the hyperpriors.
        """
        return self.hyperprior_log_likelihood

    @property
    def noise_precision(self):
        """
        This property provides the noise precision for the spherical noise model.
        :return: A scalar value of the noise precision for the spherical noise model.
        """
        return self._hyperparameter_dict[KernelHyperparameters.NOISE_PRECISION]

    @property
    def hyperparameters(self):
        """
        This property provides a dictionary of the kernel hyperparameters.
        :return: The hyperparameters of the kernel function as a dictionary.
        """
        return self._hyperparameter_dict

    def covariance_matrix(self, input_0, input_1=None, include_noise=False, include_jitter=False):
        """
        This method calculates the covariance matrix defined by the kernel function evaluated at inputs 0 and 1.
        :param input_0: The first input. Must be [N0 x Q].
        :param input_1: An optional second input; input_0 is used again if input_1 is not provided. Must be [N1 x Q].
        :param include_noise: An optional boolean to define whether or not to include noise along the diagonal.
        No noise is default.
        :param include_jitter: An optional boolean to define whether or not to include jitter along the diagonal.
        No jitter is default.
        :return: A valid covariance matrix of size [N0 x N1].
        """
        return self._covar_matrix_func(input_0=input_0, input_1=input_1, include_noise=include_noise,
                                       include_jitter=include_jitter)

    def covariance_diag(self, input_0, include_noise=False, include_jitter=False):
        """
        This method calculates the diagonal of the covariance matrix defined by the kernel function evaluated at the
        same inputs. We are currently only looking at stationary kernels so the input is not necessary.
        :param input_0: The first input. Must be [N x Q].
        :param include_noise: An optional boolean to define whether or not to include noise along the diagonal.
        No noise is default.
        :param include_jitter: An optional boolean to define whether or not to include jitter along the diagonal.
        No jitter is default.
        :return: A valid diagonal from the covariance matrix of size [N, ].
        """
        return self._covar_diag_func(input_0=input_0, include_noise=include_noise, include_jitter=include_jitter)

    def psi_0(self, inducing_input, latent_input_mean, latent_input_covariance):
        """
        This method calculates the psi_0 statistic, which is the trace of an expectation of a covariance matrix (Kff)
        over q(X) and involves convolutions of the kernel function with Gaussian densities.
        :param inducing_input: The inducing inputs. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: The psi_0 statistic.
        """
        if self._psi_0_func is not None:
            return self._psi_0_func(inducing_input=inducing_input, latent_input_mean=latent_input_mean,
                                    latent_input_covariance=latent_input_covariance)
        else:
            raise NotImplementedError

    def psi_1(self, inducing_input, latent_input_mean, latent_input_covariance):
        """
        This method calculates the psi_1 statistic, which is an expectation of a covariance matrix (Kfu) over q(X) and
        involves convolutions of the kernel function with Gaussian densities.
        :param inducing_input: The inducing input. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: The psi_1 statistic.
        """
        if self._psi_1_func is not None:
            return self._psi_1_func(inducing_input=inducing_input, latent_input_mean=latent_input_mean,
                                    latent_input_covariance=latent_input_covariance)
        else:
            raise NotImplementedError

    def psi_2(self, inducing_input, latent_input_mean, latent_input_covariance):
        """
        This method calculates the psi_2 statistic, which is an expectation of a covariance matrix (Kuf x Kfu) over q(X)
        and involves convolutions of the kernel function with Gaussian densities.
        :param inducing_input: The inducing input. Must be [M x Q].
        :param latent_input_mean: The mean of q(X), which is the variational distribution on the latent input X.
        Must be [N x Q].
        :param latent_input_covariance: The covariance of q(X), which is the variational distribution on the latent
        input X. Must be [N x Q x Q].
        :return: The psi_2 statistic.
        """
        if self._psi_2_func is not None:
            return self._psi_2_func(inducing_input=inducing_input, latent_input_mean=latent_input_mean,
                                    latent_input_covariance=latent_input_covariance)
        else:
            raise NotImplementedError
