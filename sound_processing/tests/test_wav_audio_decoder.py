# -*- coding: utf-8 -*-
"""Module containing tests for WavAudioDecoder class."""

import pathlib
import pytest
import numpy as np

from time_domain import audio_signal
from time_domain.serialization import wav_audio_decoder

# pylint: disable=W0621

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
def not_wave_path(res_path) -> pathlib.Path:
    """Returns path to example file not being a WAV file."""
    return res_path.joinpath("not_wave.txt")


@pytest.fixture
def no_header_path(res_path) -> pathlib.Path:
    """Returns path to example file without header chunk."""
    return res_path.joinpath("missing_header_chunk.wav")


@pytest.fixture
def no_fmt_path(res_path) -> pathlib.Path:
    """Returns path to example file without fmt chunk."""
    return res_path.joinpath("missing_fmt_chunk.wav")


@pytest.fixture
def no_data_path(res_path) -> pathlib.Path:
    """Returns path to example file without data chunk."""
    return res_path.joinpath("missing_data_chunk.wav")


@pytest.fixture
def sample_correct_signal() -> audio_signal.AudioSignal:
    """Returns AudioSignal containing data present under sample_correct_path."""

    signal_data = np.array(
        [
            [(i % 1000) - 500 for i in range(44100 * 2) if i % 2 == 0],
            [(i % 1000) - 500 for i in range(44100 * 2) if i % 2 != 0],
        ]
    )
    signal_meta = audio_signal.AudioSignalMeta(44100, 2, 16)

    return audio_signal.AudioSignal(signal_data, signal_meta)


# ------------------------------------------
# Unit tests
# ------------------------------------------


@pytest.mark.xfail(raises=wav_audio_decoder.WAVDecoderError)
def test_opening_wrong_path():
    """Tests if WAVAudioDecoder correctly identifies non-existent path."""

    assert wav_audio_decoder.WAVDecoder().is_file_valid("/non/existent/path") is False

    wav_audio_decoder.WAVDecoder().decode("/non/existent/path")


@pytest.mark.xfail(raises=wav_audio_decoder.WAVDecoderError)
def test_opening_directory_path(res_path):
    """Tests if WAVAudioDecoder correctly identifies path being a directory."""

    path_str = res_path.as_posix()

    assert wav_audio_decoder.WAVDecoder().is_file_valid(path_str) is False

    wav_audio_decoder.WAVDecoder().decode(path_str)


@pytest.mark.xfail(raises=wav_audio_decoder.WAVDecoderError)
def test_opening_not_wav_path(not_wave_path):
    """Tests if WAVAudioDecoder correctly identifies path with wrong extension."""

    path_str = not_wave_path.as_posix()

    assert wav_audio_decoder.WAVDecoder().is_file_valid(path_str) is False

    wav_audio_decoder.WAVDecoder().decode(path_str)


@pytest.mark.xfail(raises=wav_audio_decoder.WAVDecoderError)
def test_opening_no_header_path(no_header_path):
    """Tests if WAVAudioDecoder correctly identifies file without header chunk."""

    path_str = no_header_path.as_posix()

    assert wav_audio_decoder.WAVDecoder().is_file_valid(path_str) is False

    wav_audio_decoder.WAVDecoder().decode(path_str)


@pytest.mark.xfail(raises=wav_audio_decoder.WAVDecoderError)
def test_opening_no_fmt_path(no_fmt_path):
    """Tests if WAVAudioDecoder correctly identifies file without fmt chunk."""

    path_str = no_fmt_path.as_posix()

    assert wav_audio_decoder.WAVDecoder().is_file_valid(path_str) is False

    wav_audio_decoder.WAVDecoder().decode(path_str)


@pytest.mark.xfail(raises=wav_audio_decoder.WAVDecoderError)
def test_opening_no_data_path(no_data_path):
    """Tests if WAVAudioDecoder correctly identifies file without data chunk."""

    path_str = no_data_path.as_posix()

    assert wav_audio_decoder.WAVDecoder().is_file_valid(path_str) is False

    wav_audio_decoder.WAVDecoder().decode(path_str)


def test_opening_correctly_specified_file(
    sample_correct_path: pathlib.Path, sample_correct_signal: audio_signal.AudioSignal
):
    """Tests if WavAudioDecoder can correctly decode given input."""

    decoder = wav_audio_decoder.WAVDecoder()

    # assert decoder.is_file_valid(sample_correct_path.as_posix()) is True

    decoded_signal = decoder.decode(sample_correct_path.as_posix())

    assert (
        decoded_signal.meta_data.sampling_rate
        == sample_correct_signal.meta_data.sampling_rate
    )

    assert decoded_signal.meta_data.channels == sample_correct_signal.meta_data.channels

    assert (
        decoded_signal.meta_data.bits_per_sample
        == sample_correct_signal.meta_data.bits_per_sample
    )

    assert np.array_equal(decoded_signal.data, sample_correct_signal.data)
