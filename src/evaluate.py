"""Evaluation utilities for cancer subtype classification models.

The functions here are intentionally small wrappers around scikit-learn metrics
so they can be reused from ``src/train.py`` or notebooks without duplicating
plotting and JSON export code.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


OUTPUTS_DIR = Path("outputs")
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"


def calculate_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Calculate classification accuracy.

    Parameters
    ----------
    y_true:
        Ground-truth subtype labels.
    y_pred:
        Predicted subtype labels.

    Returns
    -------
    float
        Accuracy score between 0 and 1.
    """

    y_true_series, y_pred_series = _validate_targets(y_true, y_pred)
    return float(accuracy_score(y_true_series, y_pred_series))


def calculate_macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Calculate macro-averaged F1 score.

    Macro F1 gives each subtype equal weight, which is useful when classes are
    imbalanced.
    """

    y_true_series, y_pred_series = _validate_targets(y_true, y_pred)
    return float(f1_score(y_true_series, y_pred_series, average="macro", zero_division=0))


def create_classification_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Create a scikit-learn classification report as a dictionary."""

    y_true_series, y_pred_series = _validate_targets(y_true, y_pred)
    clean_labels = _validate_labels(labels)

    report = classification_report(
        y_true_series,
        y_pred_series,
        labels=clean_labels,
        output_dict=True,
        zero_division=0,
    )
    return _to_builtin_types(report)


def plot_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    output_path: str | Path = CONFUSION_MATRIX_PATH,
) -> Path:
    """Save a confusion matrix plot.

    Parameters
    ----------
    y_true:
        Ground-truth subtype labels.
    y_pred:
        Predicted subtype labels.
    labels:
        Optional label order for matrix rows and columns.
    output_path:
        Destination path. Defaults to ``outputs/confusion_matrix.png``.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG file.
    """

    y_true_series, y_pred_series = _validate_targets(y_true, y_pred)
    clean_labels = _validate_labels(labels)
    output_path = _prepare_output_path(output_path, expected_suffix=".png")

    if clean_labels is None:
        clean_labels = sorted(set(y_true_series) | set(y_pred_series))

    matrix = confusion_matrix(y_true_series, y_pred_series, labels=clean_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=clean_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def export_metrics_json(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    output_path: str | Path = METRICS_PATH,
) -> dict[str, Any]:
    """Save accuracy, macro F1, and classification report to JSON.

    Parameters
    ----------
    y_true:
        Ground-truth subtype labels.
    y_pred:
        Predicted subtype labels.
    labels:
        Optional label order used in the classification report.
    output_path:
        Destination path. Defaults to ``outputs/metrics.json``.

    Returns
    -------
    dict
        The metrics dictionary that was written to disk.
    """

    y_true_series, y_pred_series = _validate_targets(y_true, y_pred)
    output_path = _prepare_output_path(output_path, expected_suffix=".json")

    metrics = {
        "accuracy": calculate_accuracy(y_true_series, y_pred_series),
        "macro_f1": calculate_macro_f1(y_true_series, y_pred_series),
        "classification_report": create_classification_report(
            y_true_series,
            y_pred_series,
            labels=labels,
        ),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    output_dir: str | Path = OUTPUTS_DIR,
) -> dict[str, Any]:
    """Evaluate predictions and save standard project outputs.

    This convenience function writes ``metrics.json`` and
    ``confusion_matrix.png`` under ``output_dir``. By default, outputs are saved
    to the relative ``outputs/`` folder.
    """

    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    confusion_matrix_path = output_dir / "confusion_matrix.png"

    metrics = export_metrics_json(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        output_path=metrics_path,
    )
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        output_path=confusion_matrix_path,
    )

    return metrics


def _validate_targets(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[pd.Series, pd.Series]:
    if isinstance(y_true, pd.DataFrame) or isinstance(y_pred, pd.DataFrame):
        raise ValueError("y_true and y_pred must be one-dimensional label sequences.")

    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)

    if y_true_series.empty:
        raise ValueError("y_true and y_pred must contain at least one sample.")
    if len(y_true_series) != len(y_pred_series):
        raise ValueError(
            "y_true and y_pred must have the same length. "
            f"Got {len(y_true_series)} true labels and {len(y_pred_series)} predictions."
        )
    if y_true_series.isna().any():
        raise ValueError("y_true contains missing labels.")
    if y_pred_series.isna().any():
        raise ValueError("y_pred contains missing labels.")

    return y_true_series, y_pred_series


def _validate_labels(labels: Sequence[Any] | None) -> list[Any] | None:
    if labels is None:
        return None

    clean_labels = list(labels)
    if not clean_labels:
        raise ValueError("labels must contain at least one label when provided.")
    if pd.Series(clean_labels).isna().any():
        raise ValueError("labels contains missing values.")
    if len(clean_labels) != len(set(clean_labels)):
        raise ValueError("labels must not contain duplicates.")

    return clean_labels


def _prepare_output_path(output_path: str | Path, expected_suffix: str) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != expected_suffix:
        raise ValueError(f"output_path must end with '{expected_suffix}'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _to_builtin_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin_types(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
