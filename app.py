"""Streamlit app for the cancer subtype classification portfolio demo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import MODEL_PATH, load_model, predict_subtypes


DEMO_DATA_PATH = Path("data") / "sample_input.csv"
SAMPLE_ID_COL = "sample_id"
LABEL_COL = "subtype"


st.set_page_config(
    page_title="Cancer Subtype Classification Demo",
    layout="wide",
)


@st.cache_resource
def get_model():
    """Load and cache the trained model pipeline."""

    return load_model(MODEL_PATH)


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    """Load the bundled synthetic demo sample CSV."""

    if not DEMO_DATA_PATH.exists():
        raise FileNotFoundError(f"Demo data file not found: {DEMO_DATA_PATH}")
    return pd.read_csv(DEMO_DATA_PATH)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV with clear Streamlit-facing errors."""

    if uploaded_file is None:
        raise ValueError("Please upload a CSV file.")

    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Uploaded CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("Uploaded CSV could not be parsed.") from exc

    if df.empty:
        raise ValueError("Uploaded CSV contains no rows.")

    return df


def show_prediction_results(input_df: pd.DataFrame, model) -> None:
    """Run inference and render labels, probabilities, and chart."""

    predicted_labels, probabilities = predict_subtypes(input_df, model=model)

    results = pd.DataFrame({"predicted_subtype": predicted_labels})
    st.subheader("Predicted Subtype")
    st.dataframe(results, width="stretch")

    st.subheader("Class Probability Table")
    st.dataframe(probabilities, width="stretch")

    st.subheader("Probability Bar Chart")
    if len(probabilities) == 1:
        st.bar_chart(probabilities.iloc[0])
    else:
        st.bar_chart(probabilities)


def main() -> None:
    """Render the Streamlit app."""

    st.title("Cancer Subtype Classification Demo")
    st.info(
        "This is a portfolio demo using synthetic demo data. "
        "It is not a clinical diagnostic tool and should not be used for "
        "medical decisions."
    )

    try:
        model = get_model()
    except FileNotFoundError:
        st.warning("No trained model was found.")
        st.code("python src/train.py --demo", language="bash")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
        st.stop()

    source = st.radio(
        "Input source",
        options=["Use demo sample", "Upload CSV"],
        horizontal=True,
    )

    try:
        if source == "Use demo sample":
            demo_df = load_demo_data()
            if SAMPLE_ID_COL in demo_df.columns:
                sample_options = demo_df[SAMPLE_ID_COL].astype(str).tolist()
                selected_sample = st.selectbox("Demo sample", sample_options)
                input_df = demo_df.loc[
                    demo_df[SAMPLE_ID_COL].astype(str) == selected_sample
                ].head(1)
            else:
                input_df = demo_df.head(1)

            display_cols = [
                col
                for col in [SAMPLE_ID_COL, LABEL_COL]
                if col in input_df.columns
            ]
            if display_cols:
                st.caption("Synthetic demo sample")
                st.dataframe(input_df[display_cols], width="stretch")

        else:
            uploaded_file = st.file_uploader("Upload gene expression CSV", type=["csv"])
            if uploaded_file is None:
                st.stop()
            input_df = read_uploaded_csv(uploaded_file)
            st.caption("Uploaded input preview")
            st.dataframe(input_df.head(), width="stretch")

        show_prediction_results(input_df, model)

    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
