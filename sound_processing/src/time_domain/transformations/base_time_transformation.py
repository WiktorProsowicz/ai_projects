# -*- coding: utf-8 -*-
"""Module containing definition of linkable signal transformation class."""


from abc import abstractmethod

from time_domain.transformations import time_transformation_interface as trans_int


class BaseTimeTransformation(trans_int.ITimeTransformation):
    """Base abstract class implementing ITimeTransform interface.

    It adds new linking mechanics providing a way to stack several
    transformations in chain-of-responsibility manner.

    Attributes:
        next_transformation: Succeeding transformation to send the request to.
    """

    def __init__(self):
        """Initializes class' fields."""

        self.next_transformation: BaseTimeTransformation = None

    @abstractmethod
    def get_identification(self) -> str:
        """Generates a serialized string identification of the transformation.

        Returns:
            A string containing information about the transformation with its
            internal parameters. Generated string should recursively attach info about
            the next transformation.
        """
