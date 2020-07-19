"""
This module defines the abstract base class for a trainable model.
"""

from abc import ABC, abstractmethod


class Trainable(ABC):
    """
    This abstract base class represents any model that can be trained. Therefore, the model features some objective
    function (i.e., cost) that should be optimised during training. All trainable models (such as a Gaussian process)
    should inherit from this base class.
    """

    @property
    @abstractmethod
    def objective(self):
        """
        This abstract property represents the objective function, also known as the cost, of the model.
        :return: the objective function for the model
        """
        pass
