# -*- coding: utf-8 -*-
"""Contains utilities for transforming raw datasets into unified format."""

import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np

from datasets_loading import tensors_serializing


def _logger() -> logging.Logger:
    return logging.getLogger('raw_ds_processing')


def _titanic_fill_missing(inputs: pd.DataFrame):
    """Samples 'Age' values in where nan inplace."""

    mean, std = inputs['Age'].mean(), inputs['Age'].std()
    min_age = inputs['Age'].min()

    def sample_and_clamp():
        value = np.random.normal(mean, std)
        return min_age if value < min_age else value

    inputs['Age'] = inputs['Age'].apply(lambda x: sample_and_clamp() if pd.isnull(x) else x)


def _titanic_process_inputs(inputs: pd.DataFrame) -> pd.DataFrame:
    """Handles categorical features, converts to float."""

    pclass_onehot = pd.get_dummies(inputs['Pclass'], prefix='Pclass').astype(float)
    inputs = inputs.join(pclass_onehot)

    sex_onehot = pd.get_dummies(inputs['Sex'], prefix='Sex').astype(float)
    inputs = inputs.join(sex_onehot)

    inputs['Family'] = (inputs['SibSp'] + inputs['Parch']).astype(float)

    inputs.drop(['Pclass', 'Sex', 'SibSp', 'Parch'], axis=1, inplace=True)

    return inputs


def decode_titanic_ds(raw_ds_path: str
                      ) -> Tuple[tensors_serializing.TensorsSet, tensors_serializing.TensorsSet]:
    """Decodes the raw Titanic dataset into unified format.

    Args:
        raw_ds_path: Path to the raw Titanic dataset. The dataset should be
                        compatible with the output of the raw_ds_fetching.fetch_titanic function.

    Returns:
        Tuple of inputs and labels in the unified format. Every row in the `inputs` tensor has the
            following features:
            ['Age', 'Fare', 'Is class 1', 'Is class 2', 'Is class 3',
            'Is female', 'Is male', 'Number of family members']
    """

    if not os.path.exists(raw_ds_path) or not os.path.isdir(raw_ds_path):
        _logger().critical("Cannot deserialize the Titanic dataset. Path '%s' is not a directory!",
                           raw_ds_path)

    dataframe = pd.read_csv(os.path.join(raw_ds_path, 'titanic.csv'))
    dataframe.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    labels_df = pd.DataFrame(dataframe[['Survived', 'PassengerId']])
    inputs_df = dataframe.drop('Survived', axis=1)

    labels_df.set_index('PassengerId', inplace=True)
    inputs_df.set_index('PassengerId', inplace=True)

    inputs_df = _titanic_process_inputs(inputs_df)

    _titanic_fill_missing(inputs_df)

    # Label data processing
    labels_df['Survived'] = labels_df['Survived'].astype(float)

    inputs_df = inputs_df.to_numpy(np.float64)
    labels_df = labels_df.to_numpy(np.float64)

    return tensors_serializing.TensorsSet(inputs_df), tensors_serializing.TensorsSet(labels_df)
