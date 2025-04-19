import pandas as pd
import numpy as np 

df = pd.read_csv('dataset.csv') 

df = df.dropna()
df = df.drop_duplicates()

df = df[(df['person_age'] >= 18) & (df['person_age'] <= 100)]
df = df[(df['person_emp_length'] >= 0) & (df['person_emp_length'] <= 50)]

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].astype('category')

df[cat_cols] = df[cat_cols].apply(lambda x: x.str.strip().str.lower())

df.to_csv("cleaned_credit_risk_dataset.csv", index=False)

print("Data Cleaning Complete. Cleaned file saved as 'cleaned_credit_risk_dataset.csv'.")
