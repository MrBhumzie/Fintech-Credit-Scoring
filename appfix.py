import requests
url = 'http://127.0.0.1:5000/predict'
data = {
    "person_income": 50000,
    "person_age": 30,
    "person_emp_length": 5,
    "loan_amnt": 10000,
    "loan_int_rate": 10,
    "loan_percent_income": 0.2,
    "cb_person_cred_hist_length": 5,
    "person_home_ownership": "RENT",
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "cb_person_default_on_file": "N"
}
response = requests.post(url, json=data)
print(response.json())