# Cancer Subtype Classification using Gene Expression Data

This project demonstrates a machine learning workflow for classifying breast cancer molecular subtypes from gene expression-style data.

The project is structured as a reproducible portfolio demo with:

- data loading utilities
- preprocessing pipeline
- baseline machine learning models
- model evaluation
- saved demo model artifact
- Streamlit prediction app

## Disclaimer

This project is for portfolio and educational demonstration only. The included demo data is synthetic and is not intended for clinical use, diagnosis, or treatment decisions.

## Planned Methods

- Gene expression matrix loading
- Sample and label alignment
- Variance-based feature filtering
- Standard scaling
- Logistic Regression baseline
- Random Forest classifier
- Accuracy, macro F1, classification report
- Confusion matrix visualization
- Streamlit demo app

## How to Run

```bash
pip install -r requirements.txt
python src/train.py --demo
streamlit run app.py