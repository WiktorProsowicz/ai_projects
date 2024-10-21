# -*- coding: utf-8 -*-
"""Contains utilities for encoding/decoding tensors."""

import dataclasses
import logging
import os
from typing import List, Dict

import numpy as np


def _logger() -> logging.Logger:
    return logging.getLogger('serialized_tensors')


@dataclasses.dataclass
class TensorsSet:
    """Contains a list of tensors.

    Attributes:
        tensors: Stored arrays to serialize/deserialize.
    """

    tensors: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.tensors)


def _extract_tensor(raw_data: bytes, start_idx: int) -> np.ndarray:

    shape_size = np.frombuffer(raw_data[start_idx:start_idx + 8], dtype=np.uint64)[0]
    shape = np.frombuffer(raw_data[start_idx + 8:start_idx + 8 + shape_size*8], dtype=np.uint64)

    tensor_size = np.prod(shape)
    data_start_idx = start_idx + 8 + shape_size*8
    tensor_raw = np.frombuffer(
        raw_data[data_start_idx:data_start_idx + tensor_size*8], dtype=np.float64)

    return tensor_raw.reshape(shape)


def load_tensors_set(tensors_path: str) -> TensorsSet:
    """Loads serialized tensors from a file.

    Args:
        path: Path to the file containing serialized tensors.
    """

    if not os.path.exists(tensors_path) or not os.path.isfile(tensors_path):
        _logger().critical("Path '%s' is not a file!", tensors_path)

    with open(tensors_path, 'rb') as tensors_f:
        raw_data = tensors_f.read()

    tensors = []

    n_tensors = np.frombuffer(raw_data[:8], dtype=np.uint32)[0]

    data_idx = 8
    for _ in range(n_tensors):

        tensor = _extract_tensor(raw_data, data_idx)

        tensor_raw_size = 8 + len(tensor.shape) * 8 + int(np.prod(tensor.shape)) * 8
        data_idx += tensor_raw_size

        tensors.append(tensor)

    return TensorsSet(tensors)


def load_tensors_from_dir(tensors_dir: str) -> Dict[str, TensorsSet]:
    """Loads serialized tensors from a directory.

    Args:
        tensors_dir: Path to the directory containing serialized tensors.

    Returns:
        Dictionary with filenames as keys and TensorsSet objects as values.
    """

    if not os.path.isdir(tensors_dir):
        _logger().critical("Path '%s' is not a directory!", tensors_dir)

    tensors_files = [f for f in os.listdir(
        tensors_dir) if os.path.isfile(os.path.join(tensors_dir, f))]

    return {f: load_tensors_set(os.path.join(tensors_dir, f)) for f in tensors_files}


def save_tensors_set(tensors: TensorsSet, tensors_path: str):
    """Saves tensors to a file.

    Args:
        tensors: Tensors to serialize.
        tensors_path: Path to save the serialized tensors.
    """

    with open(tensors_path, 'wb') as tensors_f:

        tensors_f.write(np.array(len(tensors.tensors), dtype=np.uint64).tobytes())

        for tensor in tensors.tensors:

            raw_tensor = np.array(tensor.ndim, dtype=np.uint64).tobytes()
            raw_tensor += np.array(tensor.shape, dtype=np.uint64).tobytes()
            raw_tensor += tensor.tobytes()

            tensors_f.write(raw_tensor)
