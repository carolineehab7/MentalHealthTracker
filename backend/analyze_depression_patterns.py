import pandas as pd
import numpy as np
from train_model import _engineer_features

# Load and analyze the dataset
df = pd.read_csv('Student Depression Dataset.csv')
df.columns = df.columns.str.strip()
df = _engineer_features(df)

# Find cases with depression and high pressure-satisfaction ratio
depression_cases = df[df['Depression'] == 1]
high_risk_ratio = depression_cases[depression_cases['Pressure_Satisfaction_Ratio'] >= 3.0]

print(f"Total depression cases: {len(depression_cases)}")
print(f"Depression cases with Pressure-Satisfaction Ratio >= 3.0: {len(high_risk_ratio)}")

if len(high_risk_ratio) > 0:
    print("\nSample high-risk depression cases:")
    cols = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 
            'Dietary Habits', 'Have you ever had suicidal thoughts ?', 
            'Stress_Score', 'Satisfaction_Score', 'Pressure_Satisfaction_Ratio']
    print(high_risk_ratio[cols].head())

# Check suicidal thoughts distribution
print(f"\nSuicidal thoughts in depression cases:")
print(depression_cases['Have you ever had suicidal thoughts ?'].value_counts())
print(f"No depression cases with suicidal thoughts: {len(df[(df['Depression']==0) & (df['Suicidal']==1)])}")

# Check if there are any depression cases with no suicidal thoughts AND low stress
no_suicidal_low_stress = df[(df['Suicidal']==0) & (df['Stress_Score'] < 2) & (df['Depression']==1)]
print(f"\nDepression cases WITHOUT suicidal thoughts and WITH low stress: {len(no_suicidal_low_stress)}")

# Show average features for depression vs non-depression
print("\nAverage feature values by depression status:")
print(f"{'Feature':<35} | {'No Depression':>12} | {'Depression':>12}")
print("-" * 65)
for col in ['Stress_Score', 'Satisfaction_Score', 'Risk_Factors', 'Pressure_Satisfaction_Ratio', 'Suicidal']:
    nd_mean = df[df['Depression']==0][col].mean()
    d_mean = df[df['Depression']==1][col].mean()
    print(f"{col:<35} | {nd_mean:>12.3f} | {d_mean:>12.3f}")
