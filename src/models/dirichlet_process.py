"""
This module implements a Dirichlet process (DP) model. No information regarding the base distribution of a DP mixture is
captured here. DP mixtures should be implemented in a separate module.
"""

from src.distributions.beta import entropy as beta_dist_entropy
from src.distributions.gamma import entropy as gamma_dist_entropy
from src.distributions.multinomial import entropy as multinomial_dist_entropy
from src.models.interfaces.trainable import Trainable
from src.utils.constants import DP_DEFAULT_ALPHA_PRIOR_PARAMS, DP_DEFAULT_TRUNCATION_LEVEL
from src.utils.types import TF_DTYPE, create_positive_variable, create_random_positive_variable

import numpy as np
import tensorflow as tf


def dirichlet_process(num_samples, alpha_prior_params=DP_DEFAULT_ALPHA_PRIOR_PARAMS,
                      truncation_level=DP_DEFAULT_TRUNCATION_LEVEL, mask_size=1):
    """
    This initialises a DirichletProcess object.
    :param num_samples: The number of samples (N).
    :param alpha_prior_params: The parameters for the Gamma prior on alpha.
    :param truncation_level: The truncation level (T) for the truncated stick-breaking representation to approximate
    the DP. Must be a positive scalar and should be much less than the number of samples.
    :param mask_size: A mask size for grouping adjacent samples to the same group assignment. Default is 1 so each
    sample is not forced to share a group with its neighbor(s).
    :return: An instance of the DirichletProcess class.
    """

    # TODO: Validate input.

    # Set p(alpha) parameters as TensorFlow constants.
    s_1 = tf.constant(alpha_prior_params[0], dtype=TF_DTYPE)
    s_2 = tf.constant(alpha_prior_params[1], dtype=TF_DTYPE)

    # Define variational distribution parameters.

    # q(Z) is parameterised by phi, which has a shape of [N x T]:
    if mask_size == 1:
        phi = tf.nn.softmax(tf.Variable(initial_value=np.random.standard_normal((num_samples, truncation_level)),
                                        dtype=TF_DTYPE,
                                        trainable=True))
    else:
        mask_depth = np.int(np.divide(num_samples, mask_size))
        mask_indices = np.repeat(np.arange(mask_depth), mask_size)
        phi_mask = tf.one_hot(mask_indices, mask_depth, dtype=TF_DTYPE)

        logits = tf.Variable(initial_value=np.random.standard_normal((mask_depth, truncation_level)),
                             dtype=TF_DTYPE,
                             trainable=True)
        phi = tf.matmul(phi_mask, tf.nn.softmax(logits))  # phi is [num_samples x truncation_level].

    # q(V) is parameterised by gamma_1 and gamma_2, which are each (T-1)-length vectors.
    gamma_1 = create_random_positive_variable(shape=truncation_level - 1, is_trainable=True)
    gamma_2 = create_random_positive_variable(shape=truncation_level - 1, is_trainable=True)

    # q(alpha) is parameterised by w_1 and w_2, which are both positive scalars. Initialise to parameters of p(alpha).
    w_1 = create_positive_variable(initial_value=alpha_prior_params[0], is_trainable=True)
    w_2 = create_positive_variable(initial_value=alpha_prior_params[1], is_trainable=True)

    # Define expectations needed to calculate the objective function.

    # Only sum to T-1 as gamma is only 1 to T-1.
    ev_q_log_p_z_given_v = tf.reduce_sum(tf.multiply(phi[:, 0:-1], tf.digamma(gamma_1) - tf.digamma(gamma_1 + gamma_2))
                                         + tf.multiply(tf.cumsum(phi, axis=-1, exclusive=True, reverse=True)[:, 0:-1],
                                                       tf.digamma(gamma_2) - tf.digamma(gamma_1 + gamma_2)))

    ev_q_log_p_v_given_alpha = (truncation_level - 1.0) * (tf.digamma(w_1) - tf.log(w_2)) + \
                               ((w_1 / w_2) - 1.0) * tf.reduce_sum(tf.digamma(gamma_2) - tf.digamma(gamma_1 + gamma_2))

    ev_q_log_p_alpha = s_1 * tf.log(s_2) - tf.lgamma(s_1) + (s_1 - 1.0) * (tf.digamma(w_1) - tf.log(w_2)) - \
        s_2 * (w_1 / w_2)

    # Define entropies needed to calculate the objective function.
    entropy_q_z = tf.reduce_sum(multinomial_dist_entropy(phi))
    entropy_q_v = tf.reduce_sum(beta_dist_entropy(gamma_1, gamma_2))
    entropy_q_alpha = gamma_dist_entropy(w_1, w_2)

    # Define evidence lower bound (ELBO).
    elbo = ev_q_log_p_z_given_v + \
        ev_q_log_p_v_given_alpha + \
        ev_q_log_p_alpha + \
        entropy_q_z + \
        entropy_q_v + \
        entropy_q_alpha

    # Define objective function.
    objective = tf.negative(elbo)

    class DirichletProcess(Trainable):
        """
        This class defines a DirichletProcess object. A Dirichlet process mixture should be implemented elsewhere as
        this class does not care about the base distribution of the mixture.
        """

        @property
        def assignments(self):
            """
            TODO
            :return:
            """
            return phi

        @property
        def q_z(self):
            """
            TODO
            :return:
            """
            return phi

        @property
        def q_v(self):
            """
            TODO
            :return:
            """
            return gamma_1, gamma_2

        @property
        def q_alpha(self):
            """
            TODO
            :return:
            """
            return w_1, w_2

        @property
        def objective(self):
            """
            TODO
            :return:
            """
            return objective

    return DirichletProcess()
