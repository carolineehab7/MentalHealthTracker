import pandas as pd
import joblib
import os
from train_model import _engineer_features, NUMERIC_COLS, CATEGORICAL_COLS

# Load a real depression case from the dataset
df = pd.read_csv('Student Depression Dataset.csv')
df.columns = df.columns.str.strip()
df = _engineer_features(df)

# Find a depression case with high risk
high_risk_depression = df[(df['Depression']==1) & (df['Pressure_Satisfaction_Ratio'] >= 3.0)]
if len(high_risk_depression) > 0:
    sample = high_risk_depression.iloc[0]
    
    # Convert to input dict format
    test_input = {
        "Age": float(sample['Age']),
        "Gender": sample['Gender'],
        "Academic Pressure": float(sample['Academic Pressure']),
        "Work Pressure": float(sample['Work Pressure']),
        "CGPA": float(sample['CGPA']),
        "Study Satisfaction": float(sample['Study Satisfaction']),
        "Job Satisfaction": float(sample['Job Satisfaction']),
        "Work/Study Hours": float(sample['Work/Study Hours']),
        "Financial Stress": float(sample['Financial Stress']),
        "Sleep Duration": sample['Sleep Duration'],
        "Dietary Habits": sample['Dietary Habits'],
        "Have you ever had suicidal thoughts ?": sample['Have you ever had suicidal thoughts ?'],
        "Family History of Mental Illness": sample['Family History of Mental Illness'],
    }
    
    print("Testing with actual high-risk depression case from dataset:")
    print(f"  Original Depression Label: {sample['Depression']}")
    print(f"  Pressure-Satisfaction Ratio: {sample['Pressure_Satisfaction_Ratio']:.2f}")
    print(f"  Suicidal: {sample['Suicidal']}")
    print(f"  Stress Score: {sample['Stress_Score']:.2f}")
    print(f"  Academic Pressure: {sample['Academic Pressure']:.2f}")
    print(f"  Sleep: {sample['Sleep Duration']}")
    print()
    
    # Load model
    pipeline = joblib.load(os.path.join("model", "depression_model.pkl"))
    metadata = joblib.load(os.path.join("model", "depression_metadata.pkl"))
    
    test_df = pd.DataFrame([test_input])
    test_df = _engineer_features(test_df)
    
    proba = pipeline.predict_proba(test_df)[0]
    optimal_threshold = metadata.get("optimal_threshold", 0.5)
    pred_class = 1 if proba[1] >= optimal_threshold else 0
    
    print(f"Model Prediction: {['No Depression', 'Depression'][pred_class]}")
    print(f"Depression Probability: {proba[1]:.1%}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Class 0 (No Depression) prob: {proba[0]:.1%}")
    print(f"Class 1 (Depression) prob: {proba[1]:.1%}")
