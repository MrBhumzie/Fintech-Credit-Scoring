import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib

# Load and preprocess data
df = pd.read_csv('credit_risk_dataset.csv')
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Include all relevant columns before encoding
features_to_use = ['person_income', 'person_age', 'person_emp_length', 'loan_amnt',
                   'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                   'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file',
                   'loan_status']
df = df[features_to_use]
df = df.rename(columns={'loan_status': 'default'})

# One-hot encode categorical variables
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Define features to scale and scale them
scaler = StandardScaler()
features_to_scale = [col for col in df.columns if col != 'default' and df[col].dtype in ['int64', 'float64']]
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Prepare X and y
X = df.drop('default', axis=1)
y = df['default']

# Train model
model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
model.fit(X, y)

# Save the trained objects
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'credit_model.pkl')
print("Model, imputer, and scaler saved!")