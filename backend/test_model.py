from train_model import predict_depression

# Test 1: High stress (should indicate depression)
test1 = {
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

# Test 2: Low stress (should indicate no depression)
test2 = {
    "Age": 25,
    "Gender": "Female",
    "Academic Pressure": 1,
    "Work Pressure": 1,
    "CGPA": 8.0,
    "Study Satisfaction": 5,
    "Job Satisfaction": 5,
    "Work/Study Hours": 3,
    "Financial Stress": 1,
    "Sleep Duration": "7-8 hours",
    "Dietary Habits": "Healthy",
    "Have you ever had suicidal thoughts ?": "No",
    "Family History of Mental Illness": "No",
}

# Test 3: Medium stress
test3 = {
    "Age": 22,
    "Gender": "Male",
    "Academic Pressure": 3,
    "Work Pressure": 2,
    "CGPA": 6.5,
    "Study Satisfaction": 3,
    "Job Satisfaction": 3,
    "Work/Study Hours": 5,
    "Financial Stress": 2,
    "Sleep Duration": "7-8 hours",
    "Dietary Habits": "Moderate",
    "Have you ever had suicidal thoughts ?": "No",
    "Family History of Mental Illness": "No",
}

print("Test 1 (High stress):")
result1 = predict_depression(test1)
print(f"  Prediction: {result1['stressLevel']}")
print(f"  Confidence: {result1['confidence']}%")
print(f"  Probabilities: {result1['probabilities']}\n")

print("Test 2 (Low stress):")
result2 = predict_depression(test2)
print(f"  Prediction: {result2['stressLevel']}")
print(f"  Confidence: {result2['confidence']}%")
print(f"  Probabilities: {result2['probabilities']}\n")

print("Test 3 (Medium stress):")
result3 = predict_depression(test3)
print(f"  Prediction: {result3['stressLevel']}")
print(f"  Confidence: {result3['confidence']}%")
print(f"  Probabilities: {result3['probabilities']}")
