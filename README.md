# AuraFinance Credit Scoring Model

## Overview
A Flask-based API for predicting credit risk, inspired by Nigeria's informal economy and NGX data. Built as part of my transition from architecture (ESUT B.Sc. 2020) to AI, leveraging Google PM certs (2024).

## Tech Stack
- Python, Pandas, Scikit-learn
- Flask/Waitress for API
- Logistic Regression with SMOTE for balanced prediction

## Setup
1. Install Python 3.13 and dependencies: `pip install -r requirements.txt` (create this file with `flask`, `pandas`, `scikit-learn`, `waitress`, `joblib`).
2. Run `python save_model.py` to generate model files.
3. Start the server: `python -m waitress --host=0.0.0.0 --port=5000 app:app`
4. Test with `python appfix.py`

## Results
Example prediction: `{'default_prediction': 0, 'default_probability': 0.3888976687157056}`

## Personal Note
My architecture background influenced feature design, while PM skills ensured project delivery.
