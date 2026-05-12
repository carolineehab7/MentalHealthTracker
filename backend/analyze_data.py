import pandas as pd
import numpy as np
from train_model import _engineer_features

# Load the dataset
df = pd.read_csv('Student Depression Dataset.csv')
df.columns = df.columns.str.strip()

# Apply feature engineering
df = _engineer_features(df)

# Analyze the relationship between features and depression
print("Depression Distribution:")
print(df['Depression'].value_counts())
print(f"\nDepression rate: {df['Depression'].mean():.1%}")

# Check feature means by depression status
print("\n" + "="*70)
print("Feature Means by Depression Status")
print("="*70)

features_to_check = [
    'Academic Pressure', 'Work Pressure', 'Financial Stress', 
    'Study Satisfaction', 'Job Satisfaction',
    'Stress_Score', 'Satisfaction_Score', 'Pressure_Satisfaction_Ratio',
    'Risk_Factors', 'Suicidal', 'Family_Risk'
]

for feature in features_to_check:
    if feature in df.columns:
        mean_no_depression = df[df['Depression'] == 0][feature].mean()
        mean_depression = df[df['Depression'] == 1][feature].mean()
        print(f"\n{feature:30} | No Dep: {mean_no_depression:7.2f} | Dep: {mean_depression:7.2f} | Diff: {mean_depression - mean_no_depression:+7.2f}")

# Check specific combinations
print("\n" + "="*70)
print("Depression rates by specific conditions:")
print("="*70)

print(f"Has suicidal thoughts & Depression: {df[(df['Suicidal']==1)]['Depression'].mean():.1%}")
print(f"No suicidal thoughts & Depression: {df[(df['Suicidal']==0)]['Depression'].mean():.1%}")

high_stress = df['Stress_Score'] >= 4
print(f"High stress (>=4) & Depression: {df[high_stress]['Depression'].mean():.1%}")
print(f"Low stress (<4) & Depression: {df[~high_stress]['Depression'].mean():.1%}")

high_pressure_ratio = df['Pressure_Satisfaction_Ratio'] >= 2
print(f"High pressure-to-satisfaction ratio (>=2) & Depression: {df[high_pressure_ratio]['Depression'].mean():.1%}")
print(f"Low pressure-to-satisfaction ratio (<2) & Depression: {df[~high_pressure_ratio]['Depression'].mean():.1%}")
