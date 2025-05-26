import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "person_age": 30,
    "person_income": 50000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "loan_amnt": 15000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.3,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 3
}

response = requests.post(url, json=data)
print(response.json())
