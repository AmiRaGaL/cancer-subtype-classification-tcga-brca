"""Inference helpers for trained cancer subtype classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


MODEL_PATH = Path("models") / "best_model.joblib"
SAMPLE_ID_COL = "sample_id"


def load_model(model_path: str | Path = MODEL_PATH) -> Any:
    """Load the trained model pipeline from disk.

    Parameters
    ----------
    model_path:
        Path to the saved ``joblib`` model. Defaults to
        ``models/best_model.joblib``.

    Returns
    -------
    object
        A trained scikit-learn pipeline containing preprocessing and classifier.

    Raises
    ------
    FileNotFoundError
        If the model artifact does not exist.
    ValueError
        If the artifact cannot be loaded.
    """

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run `python src/train.py --demo` to create it."
        )

    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise ValueError(f"Could not load model file '{model_path}': {exc}") from exc


def load_input_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load an inference CSV into a pandas DataFrame.

    Raises a clear error when the CSV path is missing, empty, or unreadable.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a .csv file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input CSV is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Input CSV could not be parsed: {csv_path}") from exc

    if df.empty:
        raise ValueError(f"Input CSV contains no rows: {csv_path}")

    return df


def validate_input_data(input_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Validate and order input features for a trained model pipeline.

    Parameters
    ----------
    input_df:
        DataFrame containing one or more samples and the required gene columns.
        A ``sample_id`` column is allowed but not required.
    model:
        Trained scikit-learn pipeline loaded by ``load_model``.

    Returns
    -------
    pandas.DataFrame
        Numeric feature matrix ordered exactly as the model expects.
    """

    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("input_df must be a pandas DataFrame.")
    if input_df.empty:
        raise ValueError("input_df must contain at least one sample.")

    required_columns = _get_required_columns(model)
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    if missing_columns:
        raise ValueError(
            "Input data is missing required gene columns: "
            f"{missing_columns[:10]}"
            f"{'...' if len(missing_columns) > 10 else ''}"
        )

    features = input_df.loc[:, required_columns].copy()
    non_numeric_columns = [
        col for col in features.columns if not pd.api.types.is_numeric_dtype(features[col])
    ]
    if non_numeric_columns:
        raise ValueError(
            "Gene expression columns must be numeric. "
            f"Non-numeric columns: {non_numeric_columns}"
        )

    return features


def predict_subtypes(
    input_df: pd.DataFrame,
    model: Any | None = None,
    model_path: str | Path = MODEL_PATH,
) -> tuple[pd.Series, pd.DataFrame]:
    """Predict subtype labels and class probabilities for input samples.

    Parameters
    ----------
    input_df:
        DataFrame containing one or more samples with required gene columns.
    model:
        Optional already-loaded model pipeline. Supplying this is useful in
        ``app.py`` to avoid reloading the model on every interaction.
    model_path:
        Path used when ``model`` is not provided.

    Returns
    -------
    tuple
        ``predicted_labels, probability_table`` where labels are a Series and
        probabilities are a DataFrame with one column per subtype class.
    """

    if model is None:
        model = load_model(model_path)

    features = validate_input_data(input_df, model)
    predictions = pd.Series(model.predict(features), name="predicted_subtype")

    probabilities = _predict_probabilities(model, features)

    if SAMPLE_ID_COL in input_df.columns:
        sample_ids = input_df[SAMPLE_ID_COL].astype(str).to_list()
        predictions.index = sample_ids
        predictions.index.name = SAMPLE_ID_COL
        probabilities.index = sample_ids
        probabilities.index.name = SAMPLE_ID_COL

    return predictions, probabilities


def predict_from_csv(
    csv_path: str | Path,
    model: Any | None = None,
    model_path: str | Path = MODEL_PATH,
) -> tuple[pd.Series, pd.DataFrame]:
    """Load an input CSV and predict subtype labels and probabilities."""

    input_df = load_input_csv(csv_path)
    return predict_subtypes(input_df=input_df, model=model, model_path=model_path)


def _get_required_columns(model: Any) -> list[str]:
    required_columns = getattr(model, "feature_names_in_", None)
    if required_columns is None:
        raise ValueError(
            "Model does not include required feature column metadata. "
            "Retrain using `python src/train.py --demo`."
        )

    return [str(col) for col in required_columns]


def _predict_probabilities(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support class probabilities.")

    probability_values = model.predict_proba(features)
    class_labels = _get_class_labels(model)

    return pd.DataFrame(probability_values, columns=class_labels)


def _get_class_labels(model: Any) -> list[str]:
    classifier = getattr(model, "named_steps", {}).get("classifier")
    class_labels = getattr(classifier, "classes_", None)

    if class_labels is None:
        class_labels = getattr(model, "classes_", None)
    if class_labels is None:
        raise ValueError("Loaded model does not expose class labels.")

    return [str(label) for label in class_labels]
