"""
MindCheck - Stress Level Classifier
Member 1: ML Engineer
=======================
Dataset : https://www.kaggle.com/datasets/lainguyn123/student-stress-factors-a-comprehensive-analysis
Install : pip install pandas scikit-learn joblib matplotlib seaborn
Run     : python -X utf8 train_model.py
Output  : model/stress_model.pkl  model/features.pkl
          model/labels.pkl         model/importance.pkl
          plots/ (6 diagnostic plots)

Pipeline
--------
1. Load & validate data
2. EDA  - class distribution, correlation heatmap, feature-target correlation
3. Preprocess  - type coercion, missing-value fill, clip
4. sklearn Pipeline  - StandardScaler + classifier
5. Model comparison  - Logistic Regression vs Decision Tree vs Random Forest (5-fold CV)
6. Hyperparameter tuning  - RandomizedSearchCV on Random Forest
7. Evaluation  - confusion matrix, ROC curves (one-vs-rest), full report
8. Save  - pipeline pkl + metadata
"""

import os, warnings
import numpy  as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve, auc,
)

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

_HERE      = os.path.dirname(__file__)
DATA_PATH  = os.path.join(_HERE, "data", "StressLevelDataset.csv")
PLOTS_DIR  = os.path.join(_HERE, "plots")
MODEL_DIR  = os.path.join(_HERE, "model")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data(path):
    if not os.path.exists(path):
        print("  [!] CSV not found - using synthetic data for demonstration.")
        return _synthetic(1200)
    df = pd.read_csv(path)
    print(f"  Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Class counts:\n{df[TARGET_COL].value_counts().sort_index().rename(LABEL_MAP).to_string()}")
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
    print(f"  Synthetic dataset: {n} rows")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def run_eda(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 2a. Class distribution ──────────────────
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]
    colors = ["#059669", "#d97706", "#dc2626"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution - Stress Levels", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Stress Level", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # ── 2b. Correlation heatmap ─────────────────
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    tick_labels = [FEATURE_LABELS.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.4, ax=ax, annot=False,
        xticklabels=tick_labels, yticklabels=tick_labels,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # ── 2c. Feature correlation with target ─────
    target_corr = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    feat_labels = [FEATURE_LABELS.get(i, i) for i in target_corr.index]
    bar_colors  = [
        "#2563eb" if v > 0.5 else "#60a5fa" if v > 0.3 else "#bfdbfe"
        for v in target_corr.values
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(feat_labels[::-1], target_corr.values[::-1], color=bar_colors[::-1])
    ax.axvline(0.3, color="#d97706", linestyle="--", linewidth=1.2,
               alpha=0.8, label="Moderate threshold (0.30)")
    ax.axvline(0.5, color="#dc2626", linestyle="--", linewidth=1.2,
               alpha=0.8, label="Strong threshold (0.50)")
    ax.set_xlabel("|Pearson r| with Stress Level", fontsize=11)
    ax.set_title("Feature Correlation with Target Variable", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "target_correlation.png"), dpi=150)
    plt.close()

    print("  EDA plots saved -> plots/")


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    """
    - Drop rows missing any feature or label
    - Coerce to numeric, fill remaining NaNs with 0
    - Clip negatives to 0 (ordinal features have no valid negative range)
    """
    df = df.copy()
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    return df


# ─────────────────────────────────────────────
# 4. MODEL COMPARISON  (5-fold stratified CV)
# ─────────────────────────────────────────────
def compare_models(X_train, X_test, y_train, y_test):
    """
    Wraps each classifier in a StandardScaler Pipeline so evaluation is fair.
    Uses class_weight='balanced' to handle any mild class imbalance.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42,
                                          class_weight="balanced")),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(max_depth=8, random_state=42,
                                              class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42,
                                              class_weight="balanced", n_jobs=-1)),
        ]),
    }

    results = {}
    print(f"\n  {'Model':<24} {'CV Acc':>8}  {'+/-':>6}  {'Test Acc':>9}")
    print("  " + "-" * 52)
    for name, pipe in candidates.items():
        scores   = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test))
        results[name] = {
            "cv_mean": scores.mean(), "cv_std": scores.std(),
            "test_acc": test_acc, "pipeline": pipe,
        }
        print(f"  {name:<24} {scores.mean():.3f}    {scores.std():.3f}    {test_acc:.3f}")

    # Bar chart
    names  = list(results.keys())
    means  = [results[n]["cv_mean"] for n in names]
    stds   = [results[n]["cv_std"]  for n in names]
    colors = ["#bfdbfe", "#60a5fa", "#2563eb"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.006,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("5-Fold CV Accuracy", fontsize=11)
    ax.set_title("Model Comparison - 5-Fold Stratified CV", fontsize=13,
                 fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    return results


# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def tune_random_forest(X_train, y_train):
    """
    RandomizedSearchCV over the Random Forest Pipeline.
    Uses StratifiedKFold so each fold preserves class proportions.
    """
    param_dist = {
        "clf__n_estimators":      [100, 150, 200, 300],
        "clf__max_depth":         [8, 10, 12, 15, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2"],
    }

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(random_state=42, class_weight="balanced",
                                          n_jobs=-1)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=25,
        cv=cv, scoring="accuracy", n_jobs=-1,
        random_state=42, verbose=0,
    )
    search.fit(X_train, y_train)

    p = search.best_params_
    print(f"  Best params : n_estimators={p['clf__n_estimators']}, "
          f"max_depth={p['clf__max_depth']}, "
          f"max_features={p['clf__max_features']}")
    print(f"  Best CV acc : {search.best_score_:.3f}")

    return search.best_estimator_


# ─────────────────────────────────────────────
# 6. EVALUATION PLOTS
# ─────────────────────────────────────────────
def plot_evaluation(pipeline, X_test, y_test, y_pred):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Confusion matrix ────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(LABEL_MAP.values()),
        yticklabels=list(LABEL_MAP.values()),
        ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Confusion Matrix - Tuned Random Forest", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── ROC curves (one-vs-rest) ────────────────
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    y_prob = pipeline.predict_proba(X_test)
    classes = list(LABEL_MAP.values())
    colors  = ["#059669", "#d97706", "#dc2626"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{cls} stress  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.45, label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves - One-vs-Rest (Tuned Random Forest)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    macro_auc = roc_auc_score(y_bin, y_prob, average="macro")
    print(f"  Macro AUC : {macro_auc:.3f}")
    print(f"  Test Accuracy : {accuracy_score(y_test, y_pred):.3f}")


# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(pipeline):
    rf  = pipeline.named_steps["clf"]
    raw = rf.feature_importances_

    imp = pd.Series(raw, index=FEATURE_COLS).sort_values(ascending=True).tail(10)
    imp.index = [FEATURE_LABELS.get(i, i) for i in imp.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(imp.index, imp.values, color="#2563eb", edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, imp.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Gini Feature Importance", fontsize=11)
    ax.set_title("Top 10 Feature Importances - Random Forest", fontsize=13,
                 fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def compute_importance(pipeline):
    rf    = pipeline.named_steps["clf"]
    raw   = rf.feature_importances_
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
# 8. SAVE
# ─────────────────────────────────────────────
def save_model(pipeline):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline,                    os.path.join(MODEL_DIR, "stress_model.pkl"))
    joblib.dump(FEATURE_COLS,               os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(LABEL_MAP,                  os.path.join(MODEL_DIR, "labels.pkl"))
    joblib.dump(compute_importance(pipeline), os.path.join(MODEL_DIR, "importance.pkl"))
    print("  Model pipeline saved -> model/")


# ─────────────────────────────────────────────
# 9. INFERENCE  (called by backend/app.py)
# ─────────────────────────────────────────────
def predict_stress(input_dict):
    """
    Accepts a dict whose keys match FEATURE_COLS.
    Returns { stressLevel, confidence, probabilities, featureImportance }.
    The saved artifact is a full sklearn Pipeline (scaler + RF), so
    scaling is applied automatically inside pipeline.predict().
    """
    pipeline   = joblib.load(os.path.join(MODEL_DIR, "stress_model.pkl"))
    label_map  = joblib.load(os.path.join(MODEL_DIR, "labels.pkl"))
    features   = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    importance = joblib.load(os.path.join(MODEL_DIR, "importance.pkl"))

    row        = pd.DataFrame([[input_dict.get(f, 0) for f in features]], columns=features)
    pred_class = int(pipeline.predict(row)[0])
    proba      = pipeline.predict_proba(row)[0]
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
    print("\n========== MindCheck - ML Training Pipeline ==========\n")

    # 1. Load
    print("[1/6] Loading data...")
    df = load_data(DATA_PATH)

    # 2. EDA
    print("\n[2/6] Running EDA...")
    run_eda(df)

    # 3. Preprocess + split
    print("\n[3/6] Preprocessing...")
    df = preprocess(df)
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # 4. Model comparison
    print("\n[4/6] Comparing models (5-fold stratified CV)...")
    compare_models(X_train, X_test, y_train, y_test)

    # 5. Hyperparameter tuning
    print("\n[5/6] Tuning Random Forest (RandomizedSearchCV, 25 iterations)...")
    best_pipeline = tune_random_forest(X_train, y_train)

    # 6. Final evaluation
    print("\n[6/6] Evaluating tuned model on held-out test set...")
    y_pred = best_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred,
                                 target_names=list(LABEL_MAP.values())))
    plot_evaluation(best_pipeline, X_test, y_test, y_pred)
    plot_feature_importance(best_pipeline)
    save_model(best_pipeline)

    print("\n====== Done ======")
    print("  Plots  -> backend/plots/")
    print("  Model  -> backend/model/")
    print("  Next   -> python app.py")
