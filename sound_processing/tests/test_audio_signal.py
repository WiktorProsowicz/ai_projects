# -*- coding: utf-8 -*-
"""Module containing tests for AudioSignal class."""

import numpy as np
import pytest

from src.time_domain import audio_signal

# pylint: disable=W0621

# ------------------------------------------
# Text fixtures
# ------------------------------------------


@pytest.fixture
def sample_correct_audio():
    """Returns example audio signal that is correctly specified."""

    data = np.array(
        [
            [22, 46, 43, 32, 123, 45, 90, 49, 2, 4],
            [54, 35, 21, 2, 24, 11, 100, 150, 4, 11],
            [34, 32, 22, 1, 66, 122, 254, 11, 0, 3],
        ]
    )

    meta_data = audio_signal.AudioSignalMeta(5, 3, 8)

    return audio_signal.AudioSignal(data, meta_data)


# ------------------------------------------
# Unit tests
# ------------------------------------------


@pytest.mark.xfail(raises=audio_signal.AudioError)
def test_audio_signal_creation_with_illformed_data():
    """Tests AudioSignal constructor with wrong shape of data."""

    data = np.random.random_sample((2, 3, 10))
    meta_data = audio_signal.AudioSignalMeta(5, 3, 32)

    audio_signal.AudioSignal(data, meta_data)


@pytest.mark.xfail(raises=audio_signal.AudioError)
def test_audio_signal_creation_with_illformed_metadata():
    """Tests AudioSignal constructor with meta-data incompatible with the data."""

    data = np.ones((3, 2, 10))
    meta_data = audio_signal.AudioSignalMeta(5, 5, 32)

    audio_signal.AudioSignal(data, meta_data)


@pytest.mark.xfail(raises=audio_signal.AudioError)
def test_dropping_wrong_channel(sample_correct_audio):
    """Tries to drop channel that is not present in the signal."""

    sample_correct_audio.drop_channel(3)


def test_dropping_right_channel(sample_correct_audio):
    """Tries to drop correctly pointed channel and checks the result."""

    sample_correct_audio.drop_channel(1)

    assert np.array_equal(
        sample_correct_audio.data,
        np.array(
            [
                [22, 46, 43, 32, 123, 45, 90, 49, 2, 4],
                [34, 32, 22, 1, 66, 122, 254, 11, 0, 3],
            ]
        ),
    )

    assert sample_correct_audio.meta_data.channels == 2


@pytest.mark.parametrize("channel,position", [(-1, 0), (3, 1), (5, 2)])
@pytest.mark.xfail(raises=audio_signal.AudioError)
def test_duplicating_wrong_channel(sample_correct_audio, channel, position):
    """Tries to duplicate channel that is not present in the signal."""

    sample_correct_audio.duplicate_channel(channel, position)


@pytest.mark.parametrize("channel,position", [(0, -1), (1, 4), (2, 15)])
@pytest.mark.xfail(raises=audio_signal.AudioError)
def test_duplicating_channel_to_wrong_position(sample_correct_audio, channel, position):
    """Tries to duplicate channel with wrongly specified destination position."""

    sample_correct_audio.duplicate_channel(channel, position)


def test_duplicating_right_channel(sample_correct_audio):
    """Duplicates chosen channel and checks the result."""

    sample_correct_audio.duplicate_channel(1, 0)

    assert np.array_equal(
        sample_correct_audio.data,
        np.array(
            [
                [54, 35, 21, 2, 24, 11, 100, 150, 4, 11],
                [22, 46, 43, 32, 123, 45, 90, 49, 2, 4],
                [54, 35, 21, 2, 24, 11, 100, 150, 4, 11],
                [34, 32, 22, 1, 66, 122, 254, 11, 0, 3],
            ]
        ),
    )

    assert sample_correct_audio.meta_data.channels == 4


def test_signal_duration(sample_correct_audio):
    """Checks if the signal's duration property is correct."""

    assert sample_correct_audio.duration == 2
