# -*- coding: utf-8 -*-
"""Contains definition of time-frequency domain spectrogram."""


import dataclasses

import numpy as np


@dataclasses.dataclass
class SpectrogramParams:
    """Contains parameters used to initialize internal spectrogram's state."""

    min_frequency: float = 0
    max_frequency: float


class Spectrogram:
    """Represents time-frequency domain spectrogram."""

    def __init__(self, data: np.ndarray, params: SpectrogramParams) -> None:
        """"""
