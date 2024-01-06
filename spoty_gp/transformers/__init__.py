import numpy as np
import pandas as pd

from .data_transformers import DataTransformer


CATEGORICAL_FEATURES = ["track_explicit", "key", "mode"]


def get_preprocessor(cat_features=CATEGORICAL_FEATURES):
    preproc = DataTransformer(columns=CATEGORICAL_FEATURES)
    return preproc


__all__ = {"get_preprocessor"}
