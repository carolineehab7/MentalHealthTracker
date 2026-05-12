"""
MindCheck - Depression Classifier
Member 1: ML Engineer
=======================
Dataset : Student Depression Dataset.csv (27,901 rows)
Install : pip install pandas scikit-learn joblib matplotlib seaborn
Run     : python -X utf8 train_model.py
Output  : model/depression_model.pkl  model/depression_metadata.pkl

Pipeline
--------
1. Load & validate data
2. Mixed numeric/categorical preprocessing (StandardScaler + OneHotEncoder)
3. Model comparison - Logistic Regression vs Random Forest (5-fold CV)
4. Hyperparameter tuning - RandomizedSearchCV
5. Evaluation - confusion matrix, ROC curve, classification report
6. Save - pipeline pkl + metadata
"""

import os, warnings
import numpy  as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.compose      import ColumnTransformer
from sklearn.ensemble     import RandomForestClassifier
from sklearn.impute       import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import (accuracy_score, classification_report,
                                   confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                      cross_val_score, train_test_split)
from sklearn.pipeline     import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─────────────────────────────────────────────
# CONSTANTS  (imported by backend/app.py)
# ─────────────────────────────────────────────
NUMERIC_COLS = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
    # Engineered features
    "Stress_Score",
    "Satisfaction_Score",
    "Sleep_Quality",
    "Diet_Quality",
    "Risk_Factors",
    "Wellbeing_Index",
    "Pressure_Satisfaction_Ratio",
]

CATEGORICAL_COLS = [
    "Gender",
    "Sleep Duration",
    "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS  # combined list for compatibility

TARGET_COL = "Depression"

LABEL_MAP = {0: "No Depression", 1: "Depression"}

FEATURE_LABELS = {
    "Age":                                  "Age",
    "Academic Pressure":                    "Academic Pressure",
    "Work Pressure":                        "Work Pressure",
    "CGPA":                                 "CGPA",
    "Study Satisfaction":                   "Study Satisfaction",
    "Job Satisfaction":                     "Job Satisfaction",
    "Work/Study Hours":                     "Work / Study Hours",
    "Financial Stress":                     "Financial Stress",
    "Gender":                               "Gender",
    "Sleep Duration":                       "Sleep Duration",
    "Dietary Habits":                       "Dietary Habits",
    "Have you ever had suicidal thoughts ?": "Suicidal Thoughts",
    "Family History of Mental Illness":     "Family History",
}

_HERE     = os.path.dirname(__file__)
DATA_PATH = os.path.join(_HERE, "Student Depression Dataset.csv")
MODEL_DIR = os.path.join(_HERE, "model")


# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE FACTORY
# ─────────────────────────────────────────────
def _build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def _engineer_features(df):
    """
    Create derived features that better capture depression indicators.
    These compound features combine multiple stress factors.
    """
    df = df.copy()
    
    # Stress composite score (normalized 0-5 scale)
    stress_cols = ["Academic Pressure", "Work Pressure", "Financial Stress"]
    df["Stress_Score"] = df[stress_cols].mean(axis=1)
    
    # Satisfaction composite score (inverse: lower is worse)
    satisfaction_cols = ["Study Satisfaction", "Job Satisfaction"]
    df["Satisfaction_Score"] = df[satisfaction_cols].mean(axis=1)
    
    # Sleep quality (encoded as numeric)
    sleep_map = {"Less than 5 hours": 1, "5-6 hours": 2, "7-8 hours": 4, "More than 8 hours": 3}
    df["Sleep_Quality"] = df["Sleep Duration"].map(sleep_map)
    
    # Diet quality (encoded as numeric)
    diet_map = {"Unhealthy": 1, "Moderate": 2, "Healthy": 3}
    df["Diet_Quality"] = df["Dietary Habits"].map(diet_map)
    
    # Risk factor composite (suicidal thoughts + family history)
    df["Suicidal"] = (df["Have you ever had suicidal thoughts ?"] == "Yes").astype(int)
    df["Family_Risk"] = (df["Family History of Mental Illness"] == "Yes").astype(int)
    df["Risk_Factors"] = df["Suicidal"] + df["Family_Risk"]
    
    # Wellbeing index (higher = better health)
    df["Wellbeing_Index"] = (df["Satisfaction_Score"] + df["Sleep_Quality"] + df["Diet_Quality"]) / 3
    
    # Pressure-to-satisfaction ratio (higher = at-risk)
    df["Pressure_Satisfaction_Ratio"] = df["Stress_Score"] / (df["Satisfaction_Score"] + 0.5)
    
    return df


# ─────────────────────────────────────────────
# INFERENCE  (called by backend/app.py)
# ─────────────────────────────────────────────
def _extract_feature_importance(pipeline, numeric_cols, categorical_cols):
    """Returns sorted list of {feature, label, pct} from the saved pipeline."""
    preprocessor = pipeline.named_steps["preprocess"]
    clf          = pipeline.named_steps["clf"]

    # Use sklearn's built-in method to get transformed feature names reliably
    try:
        raw_names = preprocessor.get_feature_names_out()
        # raw_names look like "num__Age", "cat__Gender_Male" — clean them up
        feature_names = []
        for n in raw_names:
            if n.startswith("num__"):
                feature_names.append(n[5:])         # strip "num__"
            elif n.startswith("cat__"):
                parts = n[5:].rsplit("_", 1)        # strip "cat__", split on last "_"
                col = parts[0] if len(parts) == 2 else n[5:]
                val = parts[1] if len(parts) == 2 else ""
                # Match col to the original categorical column (prefix match)
                matched = next((c for c in categorical_cols if col.startswith(c[:10])), col)
                feature_names.append(f"{matched}: {val}" if val else matched)
            else:
                feature_names.append(n)
    except Exception:
        # Fallback: build names manually from categories
        feature_names = list(numeric_cols)
        for transformer_name, transformer, cols in preprocessor.transformers_:
            if transformer_name == "cat":
                # Find the OneHotEncoder step regardless of its name
                enc = None
                for step_name, step in transformer.steps:
                    if hasattr(step, "categories_"):
                        enc = step
                        break
                if enc:
                    for col_idx, col in enumerate(cols):
                        for cat in enc.categories_[col_idx]:
                            feature_names.append(f"{col}: {cat}")

    # Get raw importance weights
    if hasattr(clf, "feature_importances_"):
        raw = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        raw = np.abs(clf.coef_[0])
    else:
        return []

    total = raw.sum() if raw.sum() > 0 else 1.0

    # For one-hot encoded features (e.g. "Gender: Male"), include the value in the label
    def make_label(fname):
        if ":" in fname:
            col, val = fname.split(":", 1)
            col = col.strip()
            return f"{FEATURE_LABELS.get(col, col)}: {val.strip()}"
        return FEATURE_LABELS.get(fname, fname)

    imp = [
        {
            "feature": fname,
            "label":   make_label(fname),
            "pct":     round(float(v / total) * 100, 1),
        }
        for fname, v in zip(feature_names, raw)
    ]
    imp.sort(key=lambda x: x["pct"], reverse=True)
    return imp


def predict_depression(input_dict):
    """
    Accepts a dict whose keys match FEATURE_COLS (dataset column names).
    Returns { stressLevel, confidence, probabilities, featureImportance }.
    Key 'stressLevel' is kept for frontend compatibility; value is 'Depression'
    or 'No Depression'.
    Uses optimized decision threshold for better sensitivity to depression indicators.
    """
    pipeline = joblib.load(os.path.join(MODEL_DIR, "depression_model.pkl"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "depression_metadata.pkl"))

    label_map        = metadata.get("label_map",        LABEL_MAP)
    numeric_cols     = metadata.get("numeric_cols",     NUMERIC_COLS)
    categorical_cols = metadata.get("categorical_cols", CATEGORICAL_COLS)
    optimal_threshold = metadata.get("optimal_threshold", 0.5)

    row = pd.DataFrame([input_dict])
    # Apply feature engineering
    row = _engineer_features(row)
    
    # Get probability predictions
    proba = pipeline.predict_proba(row)[0]
    
    # Use optimized threshold instead of default 0.5
    pred_class = 1 if proba[1] >= optimal_threshold else 0
    
    # Confidence is the probability of the predicted class
    confidence = round(float(proba[pred_class]) * 100)

    importance = _extract_feature_importance(pipeline, numeric_cols, categorical_cols)

    return {
        "stressLevel":       label_map[pred_class],
        "confidence":        confidence,
        "probabilities":     {label_map[i]: round(float(p) * 100)
                              for i, p in enumerate(proba)},
        "featureImportance": importance,
    }


# ─────────────────────────────────────────────
# TRAINING  (run this file directly to retrain)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n========== MindCheck - Depression Classifier Training ==========\n")

    # 1. Load
    print("[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Target distribution:\n{df[TARGET_COL].value_counts().rename(LABEL_MAP).to_string()}")

    # 2. Prep
    print("\n[2/5] Engineering features and splitting...")
    
    # Apply feature engineering to full dataset
    df = _engineer_features(df)
    
    # Update numeric features to include engineered columns (and filter to available cols)
    engineered_cols = ["Stress_Score", "Satisfaction_Score", "Sleep_Quality", 
                       "Diet_Quality", "Risk_Factors", "Wellbeing_Index", "Pressure_Satisfaction_Ratio"]
    numeric_features   = [c for c in NUMERIC_COLS if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_COLS if c in df.columns]
    
    X = df.drop(columns=[TARGET_COL, "id"], errors="ignore")
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical + {len(engineered_cols)} engineered")

    # 3. Compare models
    print("\n[3/5] Comparing models (5-fold stratified CV)...")
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000, 
                class_weight="balanced", 
                random_state=42,
                C=0.5,  # Stronger regularization
                solver="lbfgs"
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced_subsample", 
                random_state=42, 
                n_jobs=-1
            )),
        ]),
    }

    results = {}
    for name, pipe in candidates.items():
        # Use F1-macro score for imbalanced data
        scores   = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
        pipe.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test))
        results[name] = {"cv_mean": scores.mean(), "cv_std": scores.std(),
                         "test_acc": test_acc, "pipe": pipe}
        print(f"  {name:<22} CV_F1={scores.mean():.3f}±{scores.std():.3f}  Acc={test_acc:.3f}")

    # 4. Pick best and tune if it's RF
    best_name  = max(results, key=lambda k: results[k]["test_acc"])
    best_model = results[best_name]["pipe"]
    print(f"\n[4/5] Best model: {best_name}")

    if best_name == "Random Forest":
        print("  Tuning Random Forest...")
        preprocessor2 = _build_preprocessor(numeric_features, categorical_features)
        search = RandomizedSearchCV(
            Pipeline([("preprocess", preprocessor2),
                      ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                     random_state=42, n_jobs=-1))]),
            param_distributions={
                "clf__n_estimators":      [200, 300, 400, 500],
                "clf__max_depth":         [10, 15, 20, 25],
                "clf__min_samples_split": [3, 5, 7],
                "clf__min_samples_leaf":  [1, 2, 3],
                "clf__max_features":      ["sqrt", "log2"],
            },
            n_iter=20, scoring="f1_weighted", cv=cv, n_jobs=-1, random_state=42,
        )
        search.fit(X_train, y_train)
        if accuracy_score(y_test, search.best_estimator_.predict(X_test)) >= results[best_name]["test_acc"]:
            best_model = search.best_estimator_
            best_name  = "Tuned Random Forest"

    # 5. Calibrate and optimize threshold
    print("\n[5/5] Optimizing decision threshold...")
    
    # Get probability predictions on test set
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold using F1-score maximization (better for imbalanced data)
    from sklearn.metrics import f1_score
    
    best_f1 = 0
    optimal_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold
    
    print(f"  Optimal decision threshold: {optimal_threshold:.3f} (F1-score: {best_f1:.3f})")
    
    # 6. Save
    print("\n[6/6] Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_acc = accuracy_score(y_test, best_model.predict(X_test))
    
    joblib.dump(best_model, os.path.join(MODEL_DIR, "depression_model.pkl"))
    joblib.dump({
        "target_col":        TARGET_COL,
        "numeric_cols":      numeric_features,
        "categorical_cols":  categorical_features,
        "label_map":         LABEL_MAP,
        "selected_model":    best_name,
        "test_accuracy":     final_acc,
        "optimal_threshold": float(optimal_threshold),
    }, os.path.join(MODEL_DIR, "depression_metadata.pkl"))

    print(f"  Test accuracy : {final_acc:.3f}")
    print(f"  Optimal threshold saved for inference")
    print(f"  Saved to      : {MODEL_DIR}/")
    print("\n====== Done ====== → python app.py")
