# Fintech Credit Scoring Model

## Overview
This project develops a credit scoring model using Logistic Regression to predict loan defaults, leveraging the Kaggle "Credit Risk Dataset." With a 0.76 accuracy on 5-fold cross-validation and test sets (improved by SMOTE from 0.65), it demonstrates robust generalization for fintech applications, such as in Nigeria's informal economy. Inspired by NGX data, it reflects my transition from architecture (ESUT B.Sc. 2020, 2.57 CGPA) to AI, enhanced by Google PM certs (2024).

## Tech Stack
- Python, Pandas, Scikit-learn
- Flask/Waitress for API
- Logistic Regression with SMOTE for balanced prediction

## Setup
1. Install Python 3.13 and dependencies: `pip install -r requirements.txt`.
2. Run `python save_model.py` to generate model files.
3. Start the server: `python -m waitress --host=0.0.0.0 --port=5000 app:app`.
4. Test with `python appfix.py`.

## Usage
- API Endpoint: `POST /predict`
- Input JSON (example): `{"person_income": 50000, "person_age": 30, "person_emp_length": 5, "loan_amnt": 10000, "loan_int_rate": 10, "loan_percent_income": 0.2, "cb_person_cred_hist_length": 5, "person_home_ownership": "RENT", "loan_intent": "EDUCATION", "loan_grade": "B", "cb_person_default_on_file": "N"}`
- Output: `{'default_prediction': 0, 'default_probability': 0.3888976687157056}`

## Results
- Example prediction: `{'default_prediction': 0, 'default_probability': 0.3888976687157056}`

## Personal Note
My architecture background influenced feature design (e.g., spatial analysis for risk patterns), while PM skills ensured project delivery.

## Future Work
- Add gamification (e.g., savings badges).
- Deploy to a cloud platform (e.g., Heroku).
- Explore XGBoost for higher accuracy.

## Contributing
Open to suggestions! Please submit issues or pull requests on GitHub.

## Footer
© 2025 GitHub, Inc.
