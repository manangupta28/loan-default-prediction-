import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('cleaned_credit_risk_dataset.csv')

# Print dataset info
#print(df.info())

# Define categorical columns
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Create a dictionary to store encoders
label_encoders = {}

# Encode categorical variables and store encoders
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for later use

# Define features and target
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss') 
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save the trained model
model_filename = "xgboost_credit_risk_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Save the label encoders
encoder_filename = "label_encoders.pkl"
joblib.dump(label_encoders, encoder_filename)
print(f"Label encoders saved as {encoder_filename}")





