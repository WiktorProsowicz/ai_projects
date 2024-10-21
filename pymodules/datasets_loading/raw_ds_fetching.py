# -*- coding: utf-8 -*-
"""Contains functions for fetching raw datasets from web."""

import logging
from os import path
import subprocess


TITANIC_DS_URL = (
    'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/' +
    'heads/master/titanic.csv')


def _logger() -> logging.Logger:
    return logging.getLogger('raw_ds_fetching')


def fetch_titanic(destination_path: str):
    """Downloads and saves the Titanic classification dataset.

    Official dataset page: https://www.kaggle.com/datasets/yasserh/titanic-dataset
    Source repository: https://github.com/datasciencedojo/datasets

    Args:
        destination_path: Path to save the dataset.
    """

    if not path.exists(destination_path):
        _logger().critical("Destination path '%s' does not exist!", destination_path)

    csv_path = path.join(destination_path, 'titanic.csv')

    try:

        _logger().info("Downloading Titanic dataset to '%s'...", destination_path)

        subprocess.run(['wget', TITANIC_DS_URL, '-O', csv_path], check=True)

    except subprocess.SubprocessError as e:
        _logger().error("Failed to download Titanic dataset: %s", e)
