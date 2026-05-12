import pandas as pd
import joblib
import os
from train_model import _engineer_features

# Load data to understand feature ranges
df = pd.read_csv('Student Depression Dataset.csv')
df.columns = df.columns.str.strip()
df = _engineer_features(df)

print("Feature Ranges in Training Data:")
print(f"  Age: {df['Age'].min():.0f} - {df['Age'].max():.0f}")
print(f"  Academic Pressure: {df['Academic Pressure'].min():.0f} - {df['Academic Pressure'].max():.0f}")
print(f"  Work Pressure: {df['Work Pressure'].min():.0f} - {df['Work Pressure'].max():.0f}")
print(f"  Financial Stress: {df['Financial Stress'].min():.0f} - {df['Financial Stress'].max():.0f}")
print(f"  Study Satisfaction: {df['Study Satisfaction'].min():.0f} - {df['Study Satisfaction'].max():.0f}")
print(f"  Job Satisfaction: {df['Job Satisfaction'].min():.0f} - {df['Job Satisfaction'].max():.0f}")
print(f"  Work/Study Hours: {df['Work/Study Hours'].min():.0f} - {df['Work/Study Hours'].max():.0f}")
print(f"  CGPA: {df['CGPA'].min():.1f} - {df['CGPA'].max():.1f}")

# Load model
pipeline = joblib.load(os.path.join("model", "depression_model.pkl"))
metadata = joblib.load(os.path.join("model", "depression_metadata.pkl"))
optimal_threshold = metadata.get("optimal_threshold", 0.5)

print(f"\n{'='*70}")
print(f"Testing with various stress levels (using realistic ranges)")
print(f"{'='*70}\n")

test_cases = [
    {
        "name": "Low stress (should be No Depression)",
        "data": {
            "Age": 22,
            "Gender": "Male",
            "Academic Pressure": 1.0,
            "Work Pressure": 1.0,
            "CGPA": 8.0,
            "Study Satisfaction": 4.0,
            "Job Satisfaction": 4.0,
            "Work/Study Hours": 3.0,
            "Financial Stress": 1.0,
            "Sleep Duration": "7-8 hours",
            "Dietary Habits": "Healthy",
            "Have you ever had suicidal thoughts ?": "No",
            "Family History of Mental Illness": "No",
        }
    },
    {
        "name": "Medium stress",
        "data": {
            "Age": 23,
            "Gender": "Female",
            "Academic Pressure": 3.0,
            "Work Pressure": 2.0,
            "CGPA": 6.5,
            "Study Satisfaction": 2.0,
            "Job Satisfaction": 2.0,
            "Work/Study Hours": 6.0,
            "Financial Stress": 2.0,
            "Sleep Duration": "5-6 hours",
            "Dietary Habits": "Moderate",
            "Have you ever had suicidal thoughts ?": "No",
            "Family History of Mental Illness": "No",
        }
    },
    {
        "name": "High stress (should be Depression)",
        "data": {
            "Age": 24,
            "Gender": "Male",
            "Academic Pressure": 5.0,
            "Work Pressure": 3.0,
            "CGPA": 4.5,
            "Study Satisfaction": 1.0,
            "Job Satisfaction": 1.0,
            "Work/Study Hours": 10.0,
            "Financial Stress": 4.0,
            "Sleep Duration": "Less than 5 hours",
            "Dietary Habits": "Unhealthy",
            "Have you ever had suicidal thoughts ?": "No",
            "Family History of Mental Illness": "No",
        }
    },
    {
        "name": "Very high risk (Suicidal + High stress)",
        "data": {
            "Age": 25,
            "Gender": "Female",
            "Academic Pressure": 5.0,
            "Work Pressure": 2.0,
            "CGPA": 3.5,
            "Study Satisfaction": 1.0,
            "Job Satisfaction": 1.0,
            "Work/Study Hours": 8.0,
            "Financial Stress": 5.0,
            "Sleep Duration": "5-6 hours",
            "Dietary Habits": "Unhealthy",
            "Have you ever had suicidal thoughts ?": "Yes",
            "Family History of Mental Illness": "Yes",
        }
    },
]

for test in test_cases:
    test_df = pd.DataFrame([test["data"]])
    test_df = _engineer_features(test_df)
    
    proba = pipeline.predict_proba(test_df)[0]
    pred_class = 1 if proba[1] >= optimal_threshold else 0
    prediction = ["No Depression", "Depression"][pred_class]
    
    print(f"{test['name']}:")
    print(f"  Prediction: {prediction}")
    print(f"  Depression Probability: {proba[1]:.1%}")
    print(f"  No Depression Probability: {proba[0]:.1%}")
    print()
