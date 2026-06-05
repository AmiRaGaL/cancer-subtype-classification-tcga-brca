"""Data loading and synthetic demo data utilities.

The helpers in this module intentionally avoid any real patient data. They are
small building blocks for ``src/train.py`` and ``app.py``: load expression
matrices, load labels, align both tables by ``sample_id``, and create a
reproducible synthetic dataset for portfolio demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SAMPLE_ID_COL = "sample_id"
LABEL_COL = "subtype"
GENE_PREFIX = "gene_"
SUBTYPES = ("Luminal A", "Luminal B", "HER2-enriched", "Basal-like")


def load_expression_data(
    csv_path: str | Path,
    sample_id_col: str = SAMPLE_ID_COL,
    gene_prefix: str = GENE_PREFIX,
) -> pd.DataFrame:
    """Load a gene expression CSV and validate sample and gene columns.

    Parameters
    ----------
    csv_path:
        Relative or absolute path to a CSV file containing one row per sample.
    sample_id_col:
        Name of the sample identifier column.
    gene_prefix:
        Prefix used to identify expression feature columns. The project demo
        uses columns such as ``gene_001`` and ``gene_002``.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing ``sample_id`` and numeric gene columns.

    Raises
    ------
    ValueError
        If required columns are missing, sample IDs are invalid, or gene values
        are non-numeric.
    """

    df = _read_csv(csv_path)
    _validate_sample_ids(df, sample_id_col)

    gene_cols = _get_gene_columns(df, gene_prefix)
    _validate_numeric_gene_columns(df, gene_cols)

    return df[[sample_id_col, *gene_cols]].copy()


def load_labels(
    csv_path: str | Path,
    sample_id_col: str = SAMPLE_ID_COL,
    label_col: str = LABEL_COL,
    allowed_labels: Iterable[str] = SUBTYPES,
) -> pd.DataFrame:
    """Load subtype labels from a CSV and validate label values.

    The labels CSV may be a standalone two-column file or a combined demo file
    that also contains gene expression columns.
    """

    df = _read_csv(csv_path)
    _validate_sample_ids(df, sample_id_col)

    if label_col not in df.columns:
        raise ValueError(
            f"Labels CSV must include a '{label_col}' column. "
            f"Found columns: {list(df.columns)}"
        )

    labels = df[[sample_id_col, label_col]].copy()
    if labels[label_col].isna().any():
        raise ValueError(f"Column '{label_col}' contains missing labels.")

    allowed = set(allowed_labels)
    observed = set(labels[label_col].astype(str))
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise ValueError(
            f"Column '{label_col}' contains unsupported labels: {unexpected}. "
            f"Expected one of: {sorted(allowed)}"
        )

    return labels


def align_expression_and_labels(
    expression_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    sample_id_col: str = SAMPLE_ID_COL,
    label_col: str = LABEL_COL,
    gene_prefix: str = GENE_PREFIX,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align expression features and subtype labels by ``sample_id``.

    Returns model-ready ``X`` and ``y`` with matching sample IDs as their index.
    The row order follows ``expression_df`` after filtering to samples that have
    labels.
    """

    _validate_sample_ids(expression_df, sample_id_col, table_name="expression_df")
    _validate_sample_ids(labels_df, sample_id_col, table_name="labels_df")

    if label_col not in labels_df.columns:
        raise ValueError(f"labels_df must include a '{label_col}' column.")

    gene_cols = _get_gene_columns(expression_df, gene_prefix)
    _validate_numeric_gene_columns(expression_df, gene_cols)

    label_lookup = labels_df.set_index(sample_id_col)[label_col]
    expression_sample_ids = expression_df[sample_id_col]
    common_mask = expression_sample_ids.isin(label_lookup.index)

    if not common_mask.any():
        raise ValueError(
            "No overlapping sample IDs were found between expression data and labels."
        )

    missing_label_ids = expression_sample_ids.loc[~common_mask].tolist()
    if missing_label_ids:
        raise ValueError(
            "Some expression samples are missing labels: "
            f"{missing_label_ids[:5]}"
            f"{'...' if len(missing_label_ids) > 5 else ''}"
        )

    aligned_expression = expression_df.loc[common_mask, [sample_id_col, *gene_cols]]
    sample_ids = aligned_expression[sample_id_col]

    X = aligned_expression[gene_cols].copy()
    X.index = sample_ids
    X.index.name = sample_id_col

    y = label_lookup.loc[sample_ids].copy()
    y.index = sample_ids
    y.index.name = sample_id_col
    y.name = label_col

    return X, y


def generate_synthetic_demo_dataset(
    n_samples: int = 24,
    n_genes: int = 12,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible synthetic breast cancer subtype demo dataset.

    The generated values are simulated expression-like numbers with a small
    subtype-specific signal. They are not derived from real patients or public
    biomedical datasets.
    """

    if n_samples < len(SUBTYPES):
        raise ValueError(f"n_samples must be at least {len(SUBTYPES)}.")
    if n_genes < 4:
        raise ValueError("n_genes must be at least 4 so each subtype has signal.")

    rng = np.random.default_rng(random_state)
    sample_ids = [f"DEMO_{idx:03d}" for idx in range(1, n_samples + 1)]
    labels = np.resize(np.array(SUBTYPES), n_samples)
    rng.shuffle(labels)

    gene_cols = [f"{GENE_PREFIX}{idx:03d}" for idx in range(1, n_genes + 1)]
    expression = rng.normal(loc=0.0, scale=0.6, size=(n_samples, n_genes))

    # Add simple, interpretable subtype signals to separate the synthetic classes.
    subtype_to_gene_index = {
        "Luminal A": 0,
        "Luminal B": 1,
        "HER2-enriched": 2,
        "Basal-like": 3,
    }
    for row_idx, subtype in enumerate(labels):
        signal_idx = subtype_to_gene_index[subtype]
        expression[row_idx, signal_idx] += 2.0

    expression_df = pd.DataFrame(np.round(expression, 4), columns=gene_cols)
    expression_df.insert(0, SAMPLE_ID_COL, sample_ids)

    labels_df = pd.DataFrame({SAMPLE_ID_COL: sample_ids, LABEL_COL: labels})
    return expression_df, labels_df


def save_synthetic_demo_dataset(
    output_path: str | Path = Path("data") / "sample_input.csv",
    n_samples: int = 24,
    n_genes: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate and save a combined synthetic demo CSV.

    The output contains ``sample_id``, gene columns, and ``subtype`` so it can be
    used with both ``load_expression_data`` and ``load_labels``.
    """

    expression_df, labels_df = generate_synthetic_demo_dataset(
        n_samples=n_samples,
        n_genes=n_genes,
        random_state=random_state,
    )
    combined_df = expression_df.merge(
        labels_df,
        on=SAMPLE_ID_COL,
        how="inner",
        validate="one_to_one",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    return combined_df


def _read_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"CSV file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file with .csv extension: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}") from exc


def _validate_sample_ids(
    df: pd.DataFrame,
    sample_id_col: str,
    table_name: str = "CSV",
) -> None:
    if sample_id_col not in df.columns:
        raise ValueError(
            f"{table_name} must include a '{sample_id_col}' column. "
            f"Found columns: {list(df.columns)}"
        )
    if df[sample_id_col].isna().any():
        raise ValueError(f"{table_name} column '{sample_id_col}' contains missing IDs.")
    if df[sample_id_col].duplicated().any():
        duplicates = df.loc[df[sample_id_col].duplicated(), sample_id_col].tolist()
        raise ValueError(
            f"{table_name} column '{sample_id_col}' contains duplicate IDs: "
            f"{duplicates[:5]}"
            f"{'...' if len(duplicates) > 5 else ''}"
        )


def _get_gene_columns(df: pd.DataFrame, gene_prefix: str) -> list[str]:
    gene_cols = [col for col in df.columns if col.startswith(gene_prefix)]
    if not gene_cols:
        raise ValueError(
            f"No gene expression columns found. Expected columns named like "
            f"'{gene_prefix}001', '{gene_prefix}002', etc."
        )
    return gene_cols


def _validate_numeric_gene_columns(df: pd.DataFrame, gene_cols: list[str]) -> None:
    non_numeric_cols = [
        col for col in gene_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            f"Gene columns must be numeric. Non-numeric columns: {non_numeric_cols}"
        )
    if df[gene_cols].isna().any().any():
        missing_cols = df[gene_cols].columns[df[gene_cols].isna().any()].tolist()
        raise ValueError(f"Gene columns contain missing values: {missing_cols}")
