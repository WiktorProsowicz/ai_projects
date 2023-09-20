# -*- coding: utf-8 -*-
"""Module containing encoder of audio files using WAV format.

WAV files specification source got from http://soundfile.sapp.org/doc/WaveFormat/.
"""

from typing import IO

import numpy as np

from time_domain import audio_signal
from time_domain.serialization import serializers_interfaces


class WAVEncoder(serializers_interfaces.IAudioEncoder):
    """Class used to encode AudioSignal following WAV encoding format."""

    def encode(self, signal: audio_signal.AudioSignal, file_path: str) -> None:
        """Overrides IAudioEncoder.encode."""

        with open(file_path, "wb") as output_file:
            bytes_per_sample = signal.meta_data.bits_per_sample // 8

            data_length = (
                bytes_per_sample * signal.data.shape[1] * signal.meta_data.channels
            )

            # 16 bytes for PCM format and additional 8 for chunk_size and chunk_id
            fmt_chunk_length = 16 + 8

            # Data and chunk_size + chunk_id
            data_chunk_length = data_length + 8

            self._append_header(output_file, fmt_chunk_length + data_chunk_length + 4)

            self._append_fmt(output_file, signal.meta_data)

            self._append_data(output_file, signal.data)

    def _append_header(self, buffer: IO, chunk_size: int) -> None:
        """Adds RIFF header to the file.

        Args:
            buffer: Stream to append the header's bytes to.
            chunk_size: Declared size of the whole wave chunk.
        """

        buffer.write(b"RIFF")
        buffer.write(chunk_size.to_bytes(4, "little"))
        buffer.write(b"WAVE")

    def _append_fmt(self, buffer: IO, meta_data: audio_signal.AudioSignalMeta) -> None:
        """Adds fmt chunk to the file.

        The fmt chunk is encoded according to PCM (linear quantization) audio format.

        Args:
            buffer: Stream to append the fmt's bytes to.
            meta_data: Struct containing the description of currently serialized signal.
        """

        buffer.write(b"fmt ")
        buffer.write(int(16).to_bytes(4, "little"))

        # Linear quantization (PCM)
        buffer.write(int(1).to_bytes(2, "little"))

        buffer.write(meta_data.channels.to_bytes(2, "little"))
        buffer.write(meta_data.sampling_rate.to_bytes(4, "little"))

        block_align = meta_data.channels * meta_data.bits_per_sample // 8
        byte_rate = block_align * meta_data.sampling_rate

        buffer.write(byte_rate.to_bytes(4, "little"))
        buffer.write(block_align.to_bytes(2, "little"))

        buffer.write(meta_data.bits_per_sample.to_bytes(2, "little"))

    def _append_data(self, buffer: IO, data: np.ndarray) -> None:
        """Adds data chunk to the file.

        Args:
            buffer: File to append the data chunk's bytes to.
            data: Audio signal's data.
        """

        n_bytes = data.size * data.dtype.itemsize

        buffer.write(b"data")
        buffer.write(n_bytes.to_bytes(4, "little"))

        axis_swapped_data = np.swapaxes(data, 0, 1).flatten()

        buffer.write(axis_swapped_data.tobytes())
