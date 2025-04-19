import joblib
import pandas as pd

# Load trained model
model = joblib.load("xgboost_credit_risk_model.pkl")
print("Model loaded successfully!")

# Load encoders
label_encoders = joblib.load("label_encoders.pkl")

# Define new test case
new_data = pd.DataFrame({
    "person_age": [30],
    "person_income": [50000],
    "person_home_ownership": ["rent"],
    "person_emp_length": [5.0],
    "loan_intent": ["medical"],
    "loan_grade": ["b"],
    "loan_amnt": [10000],
    "loan_int_rate": [12.5],
    "loan_percent_income": [0.2],
    "cb_person_default_on_file": ["n"],
    "cb_person_cred_hist_length": [3]
})

# Encode categorical columns
categorical_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
for col in categorical_cols:
    new_data[col] = label_encoders[col].transform(new_data[col])

# Predict loan status
prediction = model.predict(new_data)
print("Predicted Loan Status:", "Approved" if prediction[0] == 1 else "Not Approved")
