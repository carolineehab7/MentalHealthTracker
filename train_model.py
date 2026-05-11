"""
MindCheck – Stress Level Classifier
Member 1: ML Engineer
=======================
Dataset : https://www.kaggle.com/datasets/lainguyn123/student-stress-factors-a-comprehensive-analysis
Install : pip install pandas scikit-learn joblib matplotlib seaborn
Run     : python train_model.py
Output  : model/stress_model.pkl  +  model/features.pkl
          model/labels.pkl         +  model/importance.pkl
          plots/confusion_matrix.png
          plots/feature_importance.png
"""

import os, sys
import numpy  as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.metrics          import (classification_report,
                                       confusion_matrix,
                                       accuracy_score)

# ─────────────────────────────────────────────
# CONSTANTS  (imported by backend/app.py)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]

TARGET_COL = "stress_level"

LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}

FEATURE_LABELS = {
    "anxiety_level":                "Anxiety Level",
    "self_esteem":                  "Self Esteem",
    "mental_health_history":        "Mental Health History",
    "depression":                   "Depression Score",
    "headache":                     "Headache Frequency",
    "blood_pressure":               "Blood Pressure",
    "sleep_quality":                "Sleep Quality",
    "breathing_problem":            "Breathing Problems",
    "noise_level":                  "Noise / Environment",
    "living_conditions":            "Living Conditions",
    "safety":                       "Sense of Safety",
    "basic_needs":                  "Basic Needs Met",
    "academic_performance":         "Academic Performance",
    "study_load":                   "Study / Work Load",
    "teacher_student_relationship": "Support from Authority",
    "future_career_concerns":       "Career Concerns",
    "social_support":               "Social Support",
    "peer_pressure":                "Peer Pressure",
    "extracurricular_activities":   "Extracurricular Activities",
    "bullying":                     "Bullying Experience",
}

MOOD_MAP = {
    "Very sad": 1,
    "Stressed":  2,
    "Neutral":   5,
    "Good":      8,
    "Great":     10,
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "StressLevelDataset.csv")


# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────
def load_data(path):
    if not os.path.exists(path):
        print("⚠  Dataset CSV not found — generating synthetic data for demo purposes.")
        return _synthetic(1200)
    df = pd.read_csv(path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def _synthetic(n=1200):
    np.random.seed(42)
    data = {col: np.random.randint(0, 11, n) for col in FEATURE_COLS}
    df   = pd.DataFrame(data)
    for col in ["mental_health_history", "blood_pressure", "breathing_problem"]:
        df[col] = (df[col] > 5).astype(int)
    score = (
        df["anxiety_level"]          * 0.20 +
        (10 - df["sleep_quality"])   * 0.15 +
        df["depression"]             * 0.15 +
        df["study_load"]             * 0.10 +
        df["peer_pressure"]          * 0.10 +
        (10 - df["self_esteem"])     * 0.10 +
        df["future_career_concerns"] * 0.08 +
        df["headache"]               * 0.06 +
        (10 - df["social_support"])  * 0.06
    )
    df[TARGET_COL] = pd.cut(score, bins=3, labels=[0, 1, 2]).astype(int)
    print(f"✓ Synthetic dataset ready: {n} rows")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    return df


# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────
def train(df):
    df = preprocess(df)
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    dt_cv = cross_val_score(dt, X, y, cv=5, scoring="accuracy")
    print(f"\nDecision Tree  CV: {dt_cv.mean():.3f} ± {dt_cv.std():.3f}")
    print(classification_report(y_test, dt.predict(X_test),
                                 target_names=list(LABEL_MAP.values())))

    # Final model
    rf = RandomForestClassifier(n_estimators=150, max_depth=12,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_cv  = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
    y_pred = rf.predict(X_test)
    print(f"Random Forest  CV: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")
    print(classification_report(y_test, y_pred,
                                 target_names=list(LABEL_MAP.values())))
    print(f"✓ Final test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    return rf, X_test, y_test, y_pred


# ─────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────
def plot_results(model, X_test, y_test, y_pred):
    os.makedirs(os.path.join(os.path.dirname(__file__), "plots"), exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_MAP.values(),
                yticklabels=LABEL_MAP.values())
    plt.title("Confusion Matrix — Random Forest")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "confusion_matrix.png"), dpi=150)
    plt.close()

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=True).tail(10)
    imp.index = [FEATURE_LABELS.get(i, i) for i in imp.index]
    plt.figure(figsize=(8, 5))
    imp.plot(kind="barh", color="#c8440a")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "feature_importance.png"), dpi=150)
    plt.close()
    print("✓ Plots saved → ml/plots/")


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def compute_importance(model):
    raw   = model.feature_importances_
    total = raw.sum()
    imp   = [
        {
            "feature": feat,
            "label":   FEATURE_LABELS.get(feat, feat),
            "pct":     round(float(v / total) * 100, 1),
        }
        for feat, v in zip(FEATURE_COLS, raw)
    ]
    imp.sort(key=lambda x: x["pct"], reverse=True)
    return imp


# ─────────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model):
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model,                    os.path.join(model_dir, "stress_model.pkl"))
    joblib.dump(FEATURE_COLS,             os.path.join(model_dir, "features.pkl"))
    joblib.dump(LABEL_MAP,                os.path.join(model_dir, "labels.pkl"))
    joblib.dump(compute_importance(model),os.path.join(model_dir, "importance.pkl"))
    print("✓ Model saved → ml/model/")


# ─────────────────────────────────────────────
# 7. INFERENCE  (called by backend/app.py)
# ─────────────────────────────────────────────
def predict_stress(input_dict):
    """
    Accepts a dict whose keys match FEATURE_COLS.
    Returns { stressLevel, confidence, probabilities, featureImportance }.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    model      = joblib.load(os.path.join(model_dir, "stress_model.pkl"))
    label_map  = joblib.load(os.path.join(model_dir, "labels.pkl"))
    features   = joblib.load(os.path.join(model_dir, "features.pkl"))
    importance = joblib.load(os.path.join(model_dir, "importance.pkl"))

    row        = pd.DataFrame([[input_dict.get(f, 0) for f in features]], columns=features)
    pred_class = int(model.predict(row)[0])
    proba      = model.predict_proba(row)[0]
    confidence = round(float(proba[pred_class]) * 100)

    return {
        "stressLevel":       label_map[pred_class],
        "confidence":        confidence,
        "probabilities":     {label_map[i]: round(float(p) * 100) for i, p in enumerate(proba)},
        "featureImportance": importance,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    model, X_test, y_test, y_pred = train(df)
    plot_results(model, X_test, y_test, y_pred)
    save_model(model)
    print("\n✅ Done! Next step → cd ../backend && python app.py")
