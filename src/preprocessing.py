"""Preprocessing helpers for gene expression classification.

This module builds reusable scikit-learn preprocessing objects that can be
shared by training scripts, notebooks, and the Streamlit app. It expects a
model-ready feature matrix such as the ``X`` returned by
``src.data.align_expression_and_labels``.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessing_pipeline(
    missing_strategy: str = "median",
    variance_threshold: float = 0.0,
    with_mean: bool = True,
    with_std: bool = True,
) -> Pipeline:
    """Create a reusable preprocessing pipeline for gene expression features.

    The pipeline applies these steps in order:

    1. Missing value imputation with ``SimpleImputer``.
    2. Variance-based feature filtering with ``VarianceThreshold``.
    3. Standard scaling with ``StandardScaler``.

    Parameters
    ----------
    missing_strategy:
        Imputation strategy passed to ``SimpleImputer``. Common choices are
        ``"median"``, ``"mean"``, ``"most_frequent"``, and ``"constant"``.
    variance_threshold:
        Features with variance less than or equal to this threshold are removed.
        Use ``0.0`` to remove only constant features.
    with_mean:
        Whether ``StandardScaler`` should center features.
    with_std:
        Whether ``StandardScaler`` should scale features to unit variance.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted-ready scikit-learn pipeline.

    Raises
    ------
    ValueError
        If the pipeline parameters are invalid.
    """

    valid_strategies = {"mean", "median", "most_frequent", "constant"}
    if missing_strategy not in valid_strategies:
        raise ValueError(
            f"missing_strategy must be one of {sorted(valid_strategies)}. "
            f"Got: {missing_strategy!r}"
        )

    if not isinstance(variance_threshold, Real):
        raise ValueError("variance_threshold must be a numeric value.")
    if variance_threshold < 0:
        raise ValueError("variance_threshold must be greater than or equal to 0.")

    if not isinstance(with_mean, bool):
        raise ValueError("with_mean must be a boolean.")
    if not isinstance(with_std, bool):
        raise ValueError("with_std must be a boolean.")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=missing_strategy)),
            ("variance_filter", VarianceThreshold(threshold=variance_threshold)),
            ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
        ]
    )


def stratified_train_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[Any, Any, Any, Any]:
    """Split expression features and labels into stratified train/test sets.

    Parameters
    ----------
    X:
        Numeric gene expression feature matrix. For project data, pass the
        ``X`` dataframe returned by ``align_expression_and_labels``.
    y:
        Subtype labels aligned to ``X``.
    test_size:
        Fraction of samples assigned to the test set.
    random_state:
        Random seed for reproducible splits.

    Returns
    -------
    tuple
        ``X_train, X_test, y_train, y_test`` from scikit-learn's
        ``train_test_split``.

    Raises
    ------
    ValueError
        If features, labels, or stratification settings are invalid.
    """

    _validate_feature_matrix(X)
    y_series = _validate_labels(y, expected_length=len(X))
    _validate_split_settings(y_series, test_size, random_state)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_series,
    )


def _validate_feature_matrix(X: pd.DataFrame | np.ndarray) -> None:
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError("X must be a pandas DataFrame or numpy ndarray.")

    if len(X) == 0:
        raise ValueError("X must contain at least one sample.")

    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if shape[1] == 0:
        raise ValueError("X must contain at least one feature column.")

    if isinstance(X, pd.DataFrame):
        non_numeric_cols = [
            col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
        ]
        if non_numeric_cols:
            raise ValueError(
                "X must contain only numeric gene expression features. "
                f"Non-numeric columns: {non_numeric_cols}"
            )
        return

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must contain only numeric gene expression features.")


def _validate_labels(
    y: pd.Series | np.ndarray | list[str],
    expected_length: int,
) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        raise ValueError("y must be one-dimensional, not a DataFrame.")

    y_series = pd.Series(y)
    if len(y_series) != expected_length:
        raise ValueError(
            f"X and y must contain the same number of samples. "
            f"Got {expected_length} rows in X and {len(y_series)} labels."
        )
    if y_series.isna().any():
        raise ValueError("y contains missing labels.")

    return y_series


def _validate_split_settings(
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> None:
    if not isinstance(test_size, Real):
        raise ValueError("test_size must be a numeric fraction between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    if not isinstance(random_state, Integral):
        raise ValueError("random_state must be an integer.")

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Stratified split requires at least two label classes.")

    too_small = class_counts[class_counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Each class must have at least two samples for a stratified split. "
            f"Too-small classes: {too_small.to_dict()}"
        )

    n_samples = len(y)
    n_classes = len(class_counts)
    n_test = int(np.ceil(n_samples * test_size))
    n_train = n_samples - n_test

    if n_test < n_classes:
        raise ValueError(
            "test_size is too small for stratification: the test set needs at "
            f"least one sample per class ({n_classes} classes)."
        )
    if n_train < n_classes:
        raise ValueError(
            "test_size is too large for stratification: the train set needs at "
            f"least one sample per class ({n_classes} classes)."
        )
