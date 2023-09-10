# -*- coding: utf-8 -*-
"""Module containing interfaces for classes used for serializing audio signal."""

from abc import ABC, abstractmethod

from sound_processing.src.time_domain import audio_signal


class AudioDecoderError(Exception):
    """Indicates failure while decoding audio file."""


class IAudioDecoder(ABC):
    """An interface for sound-decoding classes.

    Defines abstract methods for classes handling the conversion between the sound data formats
    and the format-agnostic representation. Format-specific configuration should be passed
    in the constructor of the subclass or in any other desired way.
    """

    @abstractmethod
    def decode(self, file_path: str) -> audio_signal.AudioSignal:
        """Reads the given file and converts it to AudioSignal class.

        Allowed format is specified by the concrete class implementing this interface.
        There should be performed checks to ensure that the given file is valid.

        Args:
            file_path: Path to file to be decoded.

        Returns:
            Decoded audio file in format-agnostic format.
        """

    @abstractmethod
    def is_file_valid(self, file_path: str) -> bool:
        """Tells if the given file is properly coded.

        Checks if the file complies with the specification of sound data format specified
        by the concrete implementation of the interface.

        Args:
            file_path: Path to file to validate.

        Returns:
            true: If the given file is valid.
            false: If not.
        """


class AudioEncoderError(Exception):
    """Indicates failure while encoding audio file."""


class IAudioEncoder(ABC):
    """An interface for sound-serializing classes.

    Defines abstract methods for classes handling the conversion between the format-agnostic sound
    representation and specific data format. Format-specific configuration should be passed
    in the constructor of the subclass or in any other desired way.
    """

    @abstractmethod
    def encode(self, signal: audio_signal.AudioSignal, file_path: str):
        """Writes the given AudioSignal to file with specific format.

        Used format is specified by concrete implementation of the interface.

        Args:
            signal: Abstract representation of the audio signal to be encoded.
            file_path: Path to the file to which the signal should be written.
        """
