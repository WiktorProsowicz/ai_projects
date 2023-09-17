# -*- coding: utf-8 -*-
"""Module containing definitions related to audio signal data."""

import dataclasses

import numpy as np


@dataclasses.dataclass(repr=True, init=True)
class AudioSignalMeta:
    """Contains format-agnostic meta data for decoded audio signal."""

    sampling_rate: int  # Number of samples per second
    channels: int  # Number of channels
    bits_per_sample: int  # Describes number of bits to encode the sample value


class AudioError(Exception):
    """Indicates failure while setting the internal state of the audio signal."""


class AudioSignal:
    """
    Represents a decoded audio signal file in a format-agnostic way.

    Attributes:
        data: An array containing audio samples. The shape of the array is
        (channels, sampling_rate * duration).
        meta_data: Struct containing data needed to properly interpret the audio signal.
    """

    def __init__(self, data: np.ndarray, meta_data: AudioSignalMeta):
        """Initializes both signal's data and meta data.

        There are performed checks connected with compatibility of the given data and specification.

        Args:
            data: An array of signal's data.
            meta_data: Struct holding additional info about the signal.

        Raises:
            AudioError: If the given data or meta-data is invalid.
        """

        if not self._is_data_valid(data, meta_data):
            raise AudioError("Given audio samples are invalid.")

        self._data = data
        self._meta_data = meta_data

    def resample(self, new_sampling_rate: int):
        """Changes the sampling rate of the audio signal and adjusts its data.

        TODO: Implement the resampling algorithm.
        """

    def drop_channel(self, channel_number: int):
        """Erases the sound channel specified by `channel_number`.

        Args:
            channel_number: Index of the channel to drop, counted from 0.

        Raises:
            AudioError: If the channel number `channel_number` is not present.
        """

        if 0 > channel_number or channel_number >= self._meta_data.channels:
            raise AudioError(
                f"Channel number {channel_number} is not present in the signal."
            )

        self._data = np.delete(self._data, channel_number, 0)
        self._meta_data.channels -= 1

    def duplicate_channel(self, channel_number: int, position: int):
        """Duplicates given channel data and places it at a given position.

        Args:
            channel_number: Index of the channel to duplicate.
            position: Index before which the duplicated channel will be placed.

        Raise:
            AudioError: If the `channel_number`'th channel is not present in the signal.
            AudioError: If the `position`' is not available..
        """

        if 0 > channel_number or channel_number >= self._meta_data.channels:
            raise AudioError(
                f"Channel number {channel_number} is not present in the signal."
            )

        if 0 > position or position > self._meta_data.channels:
            raise AudioError(
                f"Position {position} is not available to insert the duplicated channel."
            )

        if 0 < position < self.meta_data.channels:
            self._data = np.concatenate(
                (
                    self._data[:position, :],
                    self._data[channel_number : channel_number + 1, :],
                    self._data[position:, :],
                )
            )

        elif position == 0:
            self._data = np.concatenate(
                (self._data[channel_number : channel_number + 1, :], self._data[:])
            )

        else:
            self._data = np.concatenate(
                (self._data[:], self._data[channel_number : channel_number + 1, :])
            )

        self._meta_data.channels += 1

    @property
    def data(self) -> np.ndarray:
        """Returns read-only signal's data."""
        return self._data

    @property
    def meta_data(self) -> AudioSignalMeta:
        """Returns read-only signal's meta data."""
        return self._meta_data

    @property
    def duration(self) -> float:
        """Returns the audio signal's length in seconds."""
        return self.data.shape[-1] / self.meta_data.sampling_rate

    def _is_data_valid(self, data: np.ndarray, meta_data: AudioSignalMeta) -> bool:
        """Tells whether the given audio signal's data complies to the given specification."""

        return len(data.shape) == 2 and data.shape[0] == meta_data.channels
