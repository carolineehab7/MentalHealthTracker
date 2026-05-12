from train_model import predict_depression, NUMERIC_COLS, CATEGORICAL_COLS
import joblib
import os

# Load the trained model
model_path = os.path.join("model", "depression_model.pkl")
pipeline = joblib.load(model_path)

# Test a row with high depression indicators
test_from_dataset = {
    "Age": 33.0,
    "Gender": "Male",
    "Academic Pressure": 5.0,
    "Work Pressure": 4.0,
    "CGPA": 6.5,
    "Study Satisfaction": 1.0,
    "Job Satisfaction": 2.0,
    "Work/Study Hours": 8.0,
    "Financial Stress": 4.0,
    "Sleep Duration": "5-6 hours",
    "Dietary Habits": "Unhealthy",
    "Have you ever had suicidal thoughts ?": "Yes",
    "Family History of Mental Illness": "Yes",
}

print("Test with high depression indicators:")
result = predict_depression(test_from_dataset)
print(f"  Prediction: {result['stressLevel']}")
print(f"  Confidence: {result['confidence']}%")
print(f"  Probabilities: {result['probabilities']}\n")

# Show top features
print("Top 10 important features:")
for item in result['featureImportance'][:10]:
    print(f"  {item['label']:<35} {item['pct']:>6.1f}%")
