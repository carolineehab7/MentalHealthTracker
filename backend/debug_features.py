import pandas as pd
from train_model import _engineer_features, predict_depression

# Test a high-risk case and trace through feature engineering
high_risk = {
    "Age": 25,
    "Gender": "Female",
    "Academic Pressure": 5,
    "Work Pressure": 5,
    "CGPA": 3.5,
    "Study Satisfaction": 1,
    "Job Satisfaction": 1,
    "Work/Study Hours": 10,
    "Financial Stress": 5,
    "Sleep Duration": "Less than 5 hours",
    "Dietary Habits": "Unhealthy",
    "Have you ever had suicidal thoughts ?": "Yes",
    "Family History of Mental Illness": "Yes",
}

# Check feature engineering
df = pd.DataFrame([high_risk])
print("Original features:")
print(df)

df_eng = _engineer_features(df)
print("\nEngineered features:")
print(df_eng[['Stress_Score', 'Satisfaction_Score', 'Sleep_Quality', 'Diet_Quality', 
              'Suicidal', 'Family_Risk', 'Risk_Factors', 'Wellbeing_Index', 'Pressure_Satisfaction_Ratio']])

print("\nPrediction result:")
result = predict_depression(high_risk)
print(f"Prediction: {result['stressLevel']}")
print(f"Confidence: {result['confidence']}%")
print(f"Probabilities: {result['probabilities']}")
print(f"Depression probability: {result['probabilities']['Depression']}%")
