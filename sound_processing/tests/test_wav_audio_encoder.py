# -*- coding: utf-8 -*-
"""Module containing tests for WavAudioEncoder class."""

import numpy as np
import os
import pathlib
import pytest

from time_domain import audio_signal
from time_domain.serialization import wav_audio_encoder

# ------------------------------------------
# Text fixtures
# ------------------------------------------


@pytest.fixture
def res_path() -> pathlib.Path:
    """Returns path to test resources folder."""
    return pathlib.Path(__file__).absolute().parent.joinpath("res")


@pytest.fixture
def sample_correct_path(res_path) -> pathlib.Path:
    """Returns path to example correctly specified WAVE file."""
    return res_path.joinpath("correct_wave.wav")


@pytest.fixture
def sample_correct_signal() -> audio_signal.AudioSignal:
    """Returns audio signal corresponding with sample_correct_path."""

    signal_data = np.array(
        [
            [(i % 1000) - 500 for i in range(44100 * 2) if i % 2 == 0],
            [(i % 1000) - 500 for i in range(44100 * 2) if i % 2 != 0],
        ],
        dtype=np.int16,
    )
    signal_meta = audio_signal.AudioSignalMeta(44100, 2, 16)

    return audio_signal.AudioSignal(signal_data, signal_meta)


# ------------------------------------------
# Unit tests
# ------------------------------------------


def test_encoding_file_with_comparison(
    res_path, sample_correct_path, sample_correct_signal
):
    """Encodes given correct audio signal and compares it with the model file."""

    dest_file_path: str = res_path.joinpath("__encoded_correct_wave.wav").as_posix()
    exp_file_path: str = sample_correct_path.as_posix()

    encoder = wav_audio_encoder.WAVEncoder()

    encoder.encode(sample_correct_signal, dest_file_path)

    assert os.path.exists(dest_file_path) and os.path.isfile(dest_file_path)

    with open(dest_file_path, "rb") as encoded_file:
        with open(exp_file_path, "rb") as expected_file:
            assert list(encoded_file) == list(expected_file)

    os.remove(dest_file_path)
