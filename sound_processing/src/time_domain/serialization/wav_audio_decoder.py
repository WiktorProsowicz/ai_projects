# -*- coding: utf-8 -*-
"""Module containing decoder of audio files using WAV format.

WAV files specification source got from http://soundfile.sapp.org/doc/WaveFormat/.
"""

import dataclasses
import os
from typing import Tuple, IO

import numpy as np

from time_domain import audio_signal
from time_domain.serialization import serializers_interfaces as ser_int


class WAVDecoderError(Exception):
    """Indicates a failure at WAV file decoding."""


class WAVDecoder(ser_int.IAudioDecoder):
    """Decodes WAV files and converts them into audio_signal.AudioSignal format."""

    _WAVE_FORMAT_PCM = b"\01\00"
    _WAVE_FORMAT_IEEE_FLOAT = b"\03\00"
    _WAVE_FORMAT_ALAW = b"\06\00"
    _WAVE_FORMAT_MULAW = b"\07\00"
    _WAVE_FORMAT_EXTENSIBLE = b"\170\255"

    @dataclasses.dataclass
    class _FmtChunkInfo:
        """Represents the data obtained from PCM-type fmt chunk."""

        num_channels: int
        sample_rate: int
        byte_rate: int
        block_align: int
        bits_per_sample: int

    def __init__(self):
        pass

    def decode(self, file_path: str) -> audio_signal.AudioSignal:
        """Overrides IAudioDecoder.decode

        Raises:
            WAVDecoderError: If the path to the file is invalid.
            WAVDecoderError; If the file is ill-formed.
        """

        WAVDecoder._validate_file(file_path)

        with open(file_path, "rb") as input_file:
            input_file.seek(12, 1)

            _, fmt_chunk_size = WAVDecoder._get_chunk_info(input_file)
            fmt_info = WAVDecoder._get_fmt_chunk_info(input_file)

            input_file.seek(8 + fmt_chunk_size, 1)

            _, data_chunk_size = WAVDecoder._get_chunk_info(input_file)

            input_file.seek(8, 1)

            total_num_samples = (
                data_chunk_size // fmt_info.block_align * fmt_info.num_channels
            )

            samples = np.frombuffer(
                input_file.read(data_chunk_size),
                np.dtype(f"<i{fmt_info.bits_per_sample // 8}"),
                total_num_samples,
            )

            samples = np.reshape(
                samples,
                (-1, fmt_info.num_channels),
            )

            samples = np.swapaxes(samples, 0, 1)

            audio_meta = audio_signal.AudioSignalMeta(
                fmt_info.sample_rate, fmt_info.num_channels, fmt_info.bits_per_sample
            )

            return audio_signal.AudioSignal(samples, audio_meta)

    def is_file_valid(self, file_path: str) -> bool:
        try:
            WAVDecoder._validate_file(file_path)

        except WAVDecoderError:
            return False

        else:
            return True

    @staticmethod
    def _validate_file(file_path: str):
        """Checks if the given file is a proper WAV file.

        Args:
            file_path: Path to the file to validate.

        Raises:
            WAVDecoderError: If the file is invalid at any point.
        """

        WAVDecoder._validate_path(file_path)

        left_stream_size = os.path.getsize(file_path)

        with open(file_path, "rb") as input_file:
            WAVDecoder._validate_header_chunk(input_file, left_stream_size)

            input_file.seek(12, 1)
            left_stream_size -= 12

            WAVDecoder._validate_fmt_chunk(input_file, left_stream_size)

            _, fmt_chunk_size = WAVDecoder._get_chunk_info(input_file)
            fmt_info = WAVDecoder._get_fmt_chunk_info(input_file)

            input_file.seek(8 + fmt_chunk_size, 1)
            left_stream_size -= 8 + fmt_chunk_size

            WAVDecoder._validate_data_chunk(input_file, left_stream_size, fmt_info)

    @staticmethod
    def _validate_path(file_path: str):
        """Checks if the given path is a valid path to WAV file.

        Args:
            file_path: Path to validate.

        Raises:
            WAVDecoderError: If the path is not a file or has wrong extension.
        """

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise WAVDecoderError(f"Given path is not a valid file: '{file_path}'")

        _, file_extension = os.path.splitext(file_path)

        if file_extension not in [".wav", ".wave"]:
            raise WAVDecoderError(
                f"Incorrect WAV file extension: '{file_extension}'. Should be either .wav or .wave."
            )

        return True

    @staticmethod
    def _get_chunk_info(buffer: IO[bytes]) -> Tuple[bytes, int]:
        """Harvests type and length of the file chunk from given stream.

        The stream is supposed to have reading position set to the beginning of the chunk.
        The number of chars available to read should be at least 8. The buffer is reset
        after collecting the values.

        Args:
            buffer: Stream with file's data.

        Returns:
            Pair containing the four character chunk's signature and uint32 chunk's length.
        """

        chunk_type = buffer.read(4)
        chunk_length = int.from_bytes(buffer.read(4), byteorder="little")

        buffer.seek(-8, 1)

        return (chunk_type, chunk_length)

    @staticmethod
    def _validate_header_chunk(buffer: IO[bytes], buffer_left_size: int):
        """Checks whether the WAV header is correct.

        `Buffer` is supposed to be set at the beginning of the checked chunk. After validation
        the buffer's position is reset.

        Args:
            buffer: Binary file handle.
            buffer_left_size: Number of bytes to the end of the file from the point specified
                by the reading position.

        Raises:
            WAVDecoderError: If the header chunk is ill-formed.
        """

        if buffer_left_size < 8:
            raise WAVDecoderError("Header chunk should be at least 8 bytes long.")

        chunk_type, chunk_length = WAVDecoder._get_chunk_info(buffer)

        if chunk_type != b"RIFF":
            raise WAVDecoderError(
                f"Unexpected header chunk's type. Expected b'RIFF', got {chunk_type}."
            )

        if chunk_length != buffer_left_size - 8:
            raise WAVDecoderError(
                "Declared header chunk's length is smaller than the file buffer allows."
            )

        buffer.seek(8, 1)

        if chunk_length < 4 or buffer.read(4) != b"WAVE":
            raise WAVDecoderError("Missing 'WAVE' format specifier.")

        buffer.seek(-12, 1)

    @staticmethod
    def _validate_fmt_chunk(buffer: IO[bytes], buffer_left_size: int):
        """Checks if the WAV 'fmt' chunk is correct.

        `Buffer` is supposed to be set at the beginning of the checked chunk. Buffer's position
        is reset after validation.

        Args:
            buffer: Binary file handle.
            buffer_left_size: Number of bytes to the end of the file from the point specified
                by the reading position.

        Raises:
            WAVDecoderError: If the 'fmt' chunk is ill-formed.
        """

        if buffer_left_size < 8:
            raise WAVDecoderError("Fmt chunk should be at least 8 bytes long.")

        chunk_type, chunk_length = WAVDecoder._get_chunk_info(buffer)

        if chunk_type != b"fmt ":
            raise WAVDecoderError(
                f"Unexpected header chunk's type. Expected b'fmt ', got {chunk_type}."
            )

        if chunk_length > buffer_left_size - 8:
            raise WAVDecoderError(
                "Declared 'fmt' chunk's length is smaller than the file buffer allows."
            )

        buffer.seek(8, 1)

        audio_format = buffer.read(2)

        if audio_format != WAVDecoder._WAVE_FORMAT_PCM:
            raise WAVDecoderError(
                f"Unhandled audio format: {audio_format}. Currently handled is only \
                PCM (Linear quantization) = {WAVDecoder._WAVE_FORMAT_PCM}."
            )

        if chunk_length != 16:
            raise WAVDecoderError(
                "Wrong fmt chunk length for PCM audio format (should be 16)."
            )

        buffer.seek(-10, 1)

        fmt_data = WAVDecoder._get_fmt_chunk_info(buffer)

        if fmt_data.bits_per_sample % 8 != 0:
            raise WAVDecoderError(
                f"Incorrect number of bytes per sample: {fmt_data.bits_per_sample}."
                "Expected multiple of 8."
            )

        expected_block_align = fmt_data.num_channels * fmt_data.bits_per_sample / 8

        if expected_block_align != fmt_data.block_align:
            raise WAVDecoderError(
                f"Incorrect block align: {fmt_data.block_align}. Should be {expected_block_align}."
            )

        expected_byte_rate = expected_block_align * fmt_data.sample_rate

        if expected_byte_rate != fmt_data.byte_rate:
            raise WAVDecoderError(
                f"Incorrect byte rate: {fmt_data.byte_rate}. Should be {expected_byte_rate}."
            )

    @staticmethod
    def _validate_data_chunk(
        buffer: IO[bytes], buffer_left_size: int, fmt_info: _FmtChunkInfo
    ):
        """Checks if the data chunk is correct.

        The `buffer` is supposed to be reset after validation.

        Args:
            buffer: Stream with file's data.
            buffer_left_size: Number of bytes left to the end of the stream.
            fmt_info: Data harvested from the 'fmt' chunk.

        Raises:
            WAVDecoderError: If the 'data' chunk is invalid at any point.
        """

        if buffer_left_size < 8:
            raise WAVDecoderError("Data chunk should be at least 8 bytes long.")

        chunk_type, chunk_length = WAVDecoder._get_chunk_info(buffer)

        if chunk_type != b"data":
            raise WAVDecoderError(
                f"Unexpected header chunk's type. Expected b'data', got {chunk_type}."
            )

        if chunk_length != buffer_left_size - 8:
            raise WAVDecoderError(
                "Declared 'data' chunk's length is smaller than the file buffer allows."
            )

        if chunk_length % fmt_info.block_align > 1:
            raise WAVDecoderError(
                "Data chunk length is not a multiple of block-align, \
                even taking the pad byte into account."
            )

    @staticmethod
    def _get_fmt_chunk_info(buffer: IO[bytes]) -> _FmtChunkInfo:
        """Harvests data from fmt chunk.

        The `buffer` is reset after collecting.

        Args:
            buffer: Stream with fmt chunk data.

        Returns:
            FmtChunkInfo: Info collected from the chunk.
        """

        buffer.seek(10, 1)

        num_channels = int.from_bytes(buffer.read(2), byteorder="little")
        sample_rate = int.from_bytes(buffer.read(4), byteorder="little")
        byte_rate = int.from_bytes(buffer.read(4), byteorder="little")
        block_align = int.from_bytes(buffer.read(2), byteorder="little")
        bits_per_sample = int.from_bytes(buffer.read(2), byteorder="little")

        buffer.seek(-24, 1)

        return WAVDecoder._FmtChunkInfo(
            num_channels, sample_rate, byte_rate, block_align, bits_per_sample
        )
