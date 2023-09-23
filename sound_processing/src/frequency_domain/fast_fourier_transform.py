# -*- coding: utf-8 -*-
"""Contains code performing Fast Fourier Transform over time domain signal."""

import dataclasses
from typing import List

from frequency_domain import spectrogram


@dataclasses.dataclass
class FFTParams:
    """Parameters specifying the FFT internal configuration."""

    # Tells whether to add zero-padding automatically or throw an error
    # FFT expects the input to have length being a power of two
    add_padding: str


class FFT:
    """Represents Fast Fourier Transform.

    Instances encapsulate single requests of the transform which can yield collected results.
    """

    def __init__(self, params: FFTParams) -> None:
        """Sets parameters of the transform.

        Args:
            params: Parameters influencing the behavior of the transform/
        """

        self._params = params
        self._coefficients: List[complex] = None
        self._spectrogram: spectrogram.Spectrogram = None
