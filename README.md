# 📊 Loan Default Prediction using XGBoost

This project predicts whether a loan applicant is likely to default using machine learning. By analyzing applicant and loan-related features, it provides a binary classification — **default** (1) or **no default** (0). The model was trained using the powerful and efficient **XGBoost** algorithm.

---

## 🧠 Project Overview

Loan default prediction is a critical task for banks and lending institutions. It helps assess borrower risk, reduce financial losses, and make data-informed lending decisions. This project builds a predictive pipeline that includes:

- Data cleaning and preprocessing  
- Feature encoding  
- Model training and evaluation using XGBoost  
- Model serialization for reuse  

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `dataset.csv` | Original raw dataset |
| `cleaning.py` | Script to clean and preprocess the data |
| `cleaned_credit_risk_dataset.csv` | Cleaned version used for modeling |
| `training.py` | Trains the XGBoost model and saves model + encoders |
| `testing.py` | Loads saved model and encoders to make predictions |
| `xgboost_credit_risk_model.pkl` | Trained model (serialized) |
| `label_encoders.pkl` | Encoded mappings for categorical variables |

---

## 📂 Dataset Details

The dataset includes various features related to the personal, financial, and credit history of loan applicants. It is designed to reflect real-world lending criteria and risk assessment factors.

| Feature | Description |
|---------|-------------|
| `person_age` | Age of the loan applicant |
| `person_income` | Annual income in USD |
| `person_home_ownership` | Type of home ownership (e.g., Rent, Own) |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan (e.g., education, medical) |
| `loan_grade` | Creditworthiness grade assigned by lender |
| `loan_amnt` | Amount of the loan applied for |
| `loan_int_rate` | Interest rate on the loan |
| `loan_percent_income` | Ratio of loan amount to annual income |
| `cb_person_default_on_file` | Whether the applicant has previously defaulted |
| `cb_person_cred_hist_length` | Length of the applicant’s credit history |
| `loan_status` | Target variable (1 = Default, 0 = No Default) |

---

## 🔍 Features Used in the Model

The model uses all features listed above **except `loan_status`**, which is the label we aim to predict.

---

## 📈 Model Performance

After training the model and evaluating it on a held-out test set, it achieved:

| Metric | Value |
|--------|-------|
| **Accuracy** | 93.38% |
| **Precision (Default Class)** | 0.95 |
| **Recall (Default Class)** | 0.74 |
| **F1-Score (Default Class)** | 0.83 |

The model performs exceptionally well at distinguishing defaulters from non-defaulters, with high overall accuracy and strong precision for risk detection.

---

## 🧰 Tools & Technologies

- **Python**  
- **Pandas**  
- **Scikit-learn**  
- **XGBoost**  
- **Joblib** (for model persistence)
