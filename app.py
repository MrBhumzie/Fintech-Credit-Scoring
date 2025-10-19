from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import os
import joblib

app = Flask(__name__)

# Load and preprocess data (simplified for deployment)
file_path = 'credit_risk_dataset.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    features_to_use = ['person_income', 'person_age', 'person_emp_length', 'loan_amnt',
                       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    df = df[features_to_use + ['loan_status']]
    df = df.rename(columns={'loan_status': 'default'})
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    features_to_scale = [col for col in df.columns if col != 'default' and df[col].dtype in ['int64', 'float64']]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    X = df.drop('default', axis=1)
    y = df['default']
    model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
    model.fit(X, y)  # Train on full dataset for simplicity in deployment
    # Save trained objects for consistency (optional but recommended)
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(model, 'credit_model.pkl')
else:
    raise FileNotFoundError("Dataset not found. Please add credit_risk_dataset.csv.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])
    
    # Define numeric and categorical columns based on input
    numeric_cols = [col for col in input_df.columns if input_df[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in input_df.columns if input_df[col].dtype == 'object']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    if numeric_cols:
        input_df[numeric_cols] = imputer.fit_transform(input_df[numeric_cols])
    
    # One-hot encode categorical variables
    for col in categorical_cols:
        input_df = pd.get_dummies(input_df, columns=[col], drop_first=True)
    
    # Align with training columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Scale features
    scaler = StandardScaler()
    input_df[features_to_scale] = scaler.fit_transform(input_df[features_to_scale])
    
    # Predict
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1][0]
    return jsonify({
        'default_prediction': int(prediction[0]),
        'default_probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)