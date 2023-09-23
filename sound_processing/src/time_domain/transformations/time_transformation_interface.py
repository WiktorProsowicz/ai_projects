# -*- coding: utf-8 -*-
"""Module defines interface for time-domain signal transformations."""


from abc import ABC, abstractmethod

from time_domain import audio_signal


class ITimeTransformation(ABC):
    """Interface for classes performing transformations over audio signals."""

    @abstractmethod
    def transform(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Transforms input signal into another, changing its properties.

        Input signal should not be directly modified, instead there should be
        created a copy with new desired properties. The type of applied modification
        depends on the concrete implementation of the interface.

        Args:
            signal: Input time-domain signal.

        Returns:
            Audio signal with new properties.
        """
