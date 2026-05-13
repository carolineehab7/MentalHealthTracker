# MindCheck — Depression Screener

A full-stack web application for student depression risk assessment, combining a React frontend, Flask REST API, and a trained machine learning classifier on a 27,901-record student mental health dataset.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Backend API](#backend-api)
- [Frontend](#frontend)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Assessment Input Features](#assessment-input-features)
- [API Reference](#api-reference)
- [PDF Export](#pdf-export)
- [History & Persistence](#history--persistence)
- [Disease Risk Assessment](#disease-risk-assessment)
- [Coping Recommendations](#coping-recommendations)
- [Disclaimer](#disclaimer)
- [UI](#ui)
- [PDF Sample](#pdf-sample)

---

## Overview

MindCheck is a depression screening tool designed for students. It uses a Random Forest classifier trained on academic, lifestyle, and mental health data to predict depression risk and deliver personalized recommendations. The application runs a Flask API locally with a Vite-powered React UI.

**Key Highlights:**

- 13-feature depression classifier achieving **84.3% accuracy** and **0.919 ROC AUC**
- Real-time interactive assessment form with sliders and option chips
- Radar chart visualization across 5 wellness dimensions
- Rule-based disease risk engine covering 8 comorbid conditions
- Personalized coping suggestions with icon-based cards
- Exportable PDF reports with charts, recommendations, and help resources
- localStorage-based history tracking (last 30 assessments)

---

## Features

### Assessment Form

- **13 input fields** across three card sections
- Numeric inputs rendered as **smooth sliders** with live value display
- Categorical inputs rendered as **option chip selectors**
- Toggle buttons for binary mental health history fields
- Sensible defaults pre-filled for all fields
- One-click submit with animated loading state
- Smooth-scroll to result panel on completion

### Result Panel

- **Prediction badge** — Depression or No Depression
- **Confidence percentage** — model probability for the predicted class
- **Probability distribution** — horizontal bar showing both class probabilities
- **Radar chart** — spider/polar visualization across 5 wellness axes (Academic, Financial, Sleep, Workload, Satisfaction)
- **Top 5 contributing factors** — pie chart derived from feature importance scores
- **4 coping strategy cards** — personalized to prediction class, each with a custom icon
- **8 disease risk cards** — severity-labeled risk indicators for comorbid conditions
- **Export to PDF** — generates a downloadable professional report
- **Clear history** — one-click localStorage reset

### History Page

- **Summary cards** — total assessments, latest result, depression count
- **Trend line chart** — last 10 assessments plotted over time (Recharts)
- **Full history list** — timestamped entries with prediction and confidence
- **Clear all history** — deletes all stored assessments
- Persisted in browser localStorage (key: `mc_history`, max 30 entries)

### Sidebar Navigation

- MindCheck brand logo and name
- Navigation links: Assessment, History
- Model info badge: 13 features
- Disclaimer text

---

## Tech Stack

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| React | 18.3.1 | UI framework |
| React Router DOM | 6.26.0 | Client-side routing |
| Recharts | 2.12.7 | Charts (line, radar, pie, bar) |
| Lucide React | 0.400.0 | Icon library |
| Vite | 5.3.4 | Build tool and dev server |
| @vitejs/plugin-react | 4.3.1 | Vite React integration |
| CSS Modules | — | Component-scoped styling |
| Google Fonts (Inter, Sora) | — | Typography |

### Backend

| Technology | Purpose |
|---|---|
| Flask | REST API server |
| Flask-CORS | Cross-origin request handling |
| joblib | Model serialization / deserialization |
| pandas | Data manipulation and preprocessing |
| scikit-learn | ML pipeline, model training, prediction |
| reportlab | PDF generation with embedded charts |
| matplotlib | Chart rendering for PDF export |
| seaborn | Statistical visualizations in notebook |
| numpy | Numerical operations |

### Machine Learning

| Component | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| Comparison model | Logistic Regression |
| Preprocessing | ColumnTransformer (StandardScaler + OneHotEncoder) |
| Hyperparameter search | RandomizedSearchCV (12 iterations, 5-fold CV) |
| Model serialization | joblib (.pkl files) |
| Notebook environment | Jupyter (train_model.ipynb) |

---

## Project Structure

```
MentalHealthTracker/
│
├── backend/
│   ├── app.py                           # Flask REST API (4 endpoints)
│   ├── train_model.py                   # Standalone training script
│   ├── train_model.ipynb                # Jupyter notebook: EDA + training
│   ├── Student Depression Dataset.csv   # 27,901-record training dataset
│   └── model/
│       ├── depression_model.pkl         # Trained pipeline (preprocessor + classifier)
│       └── depression_metadata.pkl      # Feature names, classes, dataset info
│
├── src/
│   ├── main.jsx                         # React entry point
│   ├── App.jsx                          # Root component with layout and routing
│   ├── App.module.css
│   │
│   ├── pages/
│   │   ├── AssessPage.jsx               # Main 13-field assessment form
│   │   ├── AssessPage.module.css
│   │   ├── HistoryPage.jsx              # History list + trend chart
│   │   └── HistoryPage.module.css
│   │
│   ├── components/
│   │   ├── UI.jsx                       # Reusable primitives (Slider, OptionChip, Toggle)
│   │   ├── UI.module.css
│   │   ├── ResultPanel.jsx              # Prediction result display
│   │   ├── ResultPanel.module.css
│   │   └── RadarChartPanel.jsx          # Recharts radar/spider chart
│   │
│   ├── hooks/
│   │   └── usePredict.js                # Custom hook: API call + localStorage save
│   │
│   ├── assets/                          # 13 PNG recommendation icons
│   │
│   └── styles/
│       └── index.css                    # Global reset and base styles
│
├── index.html                           # HTML template (Google Fonts loaded here)
├── vite.config.js                       # Vite config: dev server port 3000, proxy to :5000
├── package.json                         # Frontend dependencies
└── package-lock.json
```

---

## Dataset

**File:** `backend/Student Depression Dataset.csv`

| Property | Value |
|---|---|
| Total records | 27,901 |
| Train split | 22,325 (80%) |
| Test split | 5,576 (20%) |
| Split strategy | Stratified by target |
| Target distribution | No Depression: 59.4% / Depression: 40.6% |
| Source | Kaggle — Student Mental Health Dataset |

**Columns used for training (13 features):**

| Feature | Type | Range / Options |
|---|---|---|
| Age | Numeric | 18–59 |
| Gender | Categorical | Male, Female |
| Academic Pressure | Numeric | 0–5 |
| Work Pressure | Numeric | 0–5 |
| CGPA | Numeric | 0–10 |
| Study Satisfaction | Numeric | 0–5 |
| Job Satisfaction | Numeric | 0–4 |
| Work/Study Hours | Numeric | 0–12 |
| Financial Stress | Numeric | 1–5 |
| Sleep Duration | Categorical | Less than 5 hours, 5-6 hours, 7-8 hours, More than 8 hours, Others |
| Dietary Habits | Categorical | Healthy, Moderate, Unhealthy |
| Have you ever had suicidal thoughts? | Categorical | Yes, No |
| Family History of Mental Illness | Categorical | Yes, No |

**Target column:** `Depression` (0 = No Depression, 1 = Depression)

---

## Machine Learning Pipeline

### Preprocessing

```
ColumnTransformer
├── Numeric features (8)
│   ├── SimpleImputer (strategy=median)
│   └── StandardScaler
└── Categorical features (5)
    ├── SimpleImputer (strategy=most_frequent)
    └── OneHotEncoder (handle_unknown=ignore)
```

### Model Comparison

Two models were trained with 5-fold cross-validation:

| Model | Test Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | baseline | baseline |
| **Random Forest** | **84.3%** | **0.919** |

Random Forest was selected as the final model because it produces **better-calibrated probability scores**, which is critical for the confidence display and radar chart calculations.

### Hyperparameter Tuning

RandomizedSearchCV was applied to the Random Forest with 12 iterations and 5-fold CV over the following search space:

| Hyperparameter | Search Range |
|---|---|
| n_estimators | 100–500 |
| max_depth | None, 10, 20, 30 |
| min_samples_split | 2, 5, 10 |
| min_samples_leaf | 1, 2, 4 |
| max_features | sqrt, log2 |

### Final Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 84.3% |
| ROC AUC | 0.919 |
| Precision (Depression class) | 0.88 |
| Recall (Depression class) | 0.85 |

### Top Predictive Features (by importance)

1. Academic Pressure
2. Financial Stress
3. Work Pressure
4. CGPA
5. Study Satisfaction

### Model Artifacts

| File | Contents |
|---|---|
| `backend/model/depression_model.pkl` | Full sklearn Pipeline (preprocessor + classifier) |
| `backend/model/depression_metadata.pkl` | Feature names, class labels, dataset statistics |

### Notebook (train_model.ipynb)

The Jupyter notebook documents the full workflow across 18 analysis sections:

1. Library imports and configuration constants
2. Data loading and shape inspection
3. Target distribution analysis
4. Missing value audit
5. Feature correlation with target (horizontal bar chart)
6. Correlation heatmap across all numeric features
7. Box plots — feature distributions by depression status (2×4 grid)
8. Scatter plots — top 4 features vs target
9. Histograms — top 2 features split by class
10. Categorical feature grouped bar charts (2×3 grid)
11. Train/test stratified split
12. Preprocessing pipeline construction
13. Logistic Regression baseline training and evaluation
14. Random Forest training with 5-fold CV comparison
15. Probability calibration analysis (basis for RF selection)
16. Confusion matrix, ROC curve, precision-recall curve (4-subplot dashboard)
17. RandomizedSearchCV hyperparameter tuning
18. Final model and metadata save to disk

---

## Backend API

**Server:** Flask on `http://localhost:5000`

CORS is enabled for all origins. The model pipeline is loaded once at startup into memory.

### Endpoints

#### `GET /`
Health check. Returns service status and model readiness.

**Response:**
```json
{
  "service": "MindCheck Depression Screener",
  "version": "2.0",
  "model_loaded": true,
  "features": 13,
  "status": "healthy"
}
```

---

#### `POST /predict`
Run depression classification on a 13-feature input.

**Request body:**
```json
{
  "age": 22,
  "gender": "Male",
  "academic_pressure": 3.5,
  "work_pressure": 2.0,
  "cgpa": 7.5,
  "study_satisfaction": 3.0,
  "job_satisfaction": 2.5,
  "work_study_hours": 8,
  "financial_stress": 3.0,
  "sleep_duration": "7-8 hours",
  "dietary_habits": "Moderate",
  "suicidal_thoughts": "No",
  "family_history": "No"
}
```

**Response:**
```json
{
  "stressLevel": "No Depression",
  "confidence": 78,
  "probabilities": {
    "No Depression": 78,
    "Depression": 22
  },
  "radarScores": {
    "academic": 70,
    "financial": 50,
    "sleep": 85,
    "workload": 67,
    "satisfaction": 75
  },
  "featureImportance": [
    { "label": "Academic Pressure", "pct": 15.2 },
    { "label": "Financial Stress", "pct": 12.8 },
    { "label": "CGPA", "pct": 11.1 },
    { "label": "Study Satisfaction", "pct": 9.7 },
    { "label": "Work Pressure", "pct": 9.3 }
  ],
  "suggestions": [
    {
      "category": "Maintain Your Balance",
      "text": "...",
      "icon": "balance"
    }
  ],
  "diseaseRisk": [
    { "condition": "Academic Burnout", "risk": "Low", "indicator": "..." }
  ],
  "timestamp": "2026-05-14T12:00:00"
}
```

**Radar score calculation:**

| Axis | Derived from |
|---|---|
| academic | Academic Pressure + CGPA + Study Satisfaction |
| financial | Inverse mapping of Financial Stress |
| sleep | Sleep duration quality mapping (Less than 5 h → lowest score) |
| workload | Work/Study Hours + Work Pressure |
| satisfaction | Study Satisfaction + Job Satisfaction average |

---

#### `POST /export-pdf`
Generate a downloadable PDF report from a prediction result.

**Request body:** Full response object from `/predict`

**Response:** Binary PDF file (`Content-Type: application/pdf`)

---

#### `GET /model-info`
Returns model metadata without running a prediction.

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "features": ["Age", "Academic Pressure", "..."],
  "classes": ["No Depression", "Depression"],
  "training_samples": 22325,
  "test_accuracy": 0.843
}
```

---

## Frontend

### Pages

#### Assessment Page (`/`)

Three card sections collect the 13 input features:

**Card 1 — Academic & Professional Metrics**

| Field | Control | Range |
|---|---|---|
| Age | Slider | 18–59 |
| Academic Pressure | Slider | 0–5 |
| Work Pressure | Slider | 0–5 |
| CGPA | Slider | 0.0–10.0 |
| Study Satisfaction | Slider | 0–5 |
| Job Satisfaction | Slider | 0–4 |
| Work/Study Hours per Day | Slider | 0–12 |
| Financial Stress | Slider | 1–5 |

**Card 2 — Lifestyle**

| Field | Control | Options |
|---|---|---|
| Sleep Duration | Option Chips | Less than 5 hours / 5-6 hours / 7-8 hours / More than 8 hours / Others |
| Dietary Habits | Option Chips | Healthy / Moderate / Unhealthy |
| Gender | Option Chips | Male / Female |

**Card 3 — Mental Health Background**

| Field | Control | Options |
|---|---|---|
| Family History of Mental Illness | Toggle | Yes / No |
| Suicidal Thoughts | Toggle | Yes / No |

#### History Page (`/history`)

- **Summary row:** 3 stat cards (Total Assessments, Latest Result, Depression Count)
- **Trend chart:** Recharts LineChart of confidence across last 10 assessments
- **History list:** Scrollable entries showing date, prediction badge, and confidence
- **Clear All** button

### Custom Components

| Component | File | Purpose |
|---|---|---|
| `Slider` | [src/components/UI.jsx](src/components/UI.jsx) | Styled range input with live value label |
| `OptionChip` | [src/components/UI.jsx](src/components/UI.jsx) | Selectable pill buttons for categorical options |
| `Toggle` | [src/components/UI.jsx](src/components/UI.jsx) | Yes/No toggle for binary fields |
| `ResultPanel` | [src/components/ResultPanel.jsx](src/components/ResultPanel.jsx) | Full result display (badge, charts, cards, export) |
| `RadarChartPanel` | [src/components/RadarChartPanel.jsx](src/components/RadarChartPanel.jsx) | Recharts radar chart for 5 wellness axes |

### Custom Hook

#### `usePredict` — [src/hooks/usePredict.js](src/hooks/usePredict.js)

- Manages prediction request state (`loading`, `result`, `error`)
- Sends `POST /api/predict` (proxied to Flask by Vite)
- On success: saves result to localStorage and returns structured data
- Exposes `predict(formData)` and `clearHistory()` functions

### Client-Side Routing

| Path | Component | Description |
|---|---|---|
| `/` | `AssessPage` | Assessment form and result panel |
| `/history` | `HistoryPage` | Assessment history and trend chart |

---

## Getting Started

### Prerequisites

- **Node.js** >= 18
- **Python** >= 3.9
- pip

### 1. Clone the repository

```bash
git clone https://github.com/carolineehab7/MentalHealthTracker.git
cd MentalHealthTracker
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Install backend dependencies

```bash
pip install flask flask-cors joblib pandas scikit-learn reportlab matplotlib seaborn numpy
```

### 4. Train the model (first time only)

```bash
python backend/train_model.py
```

This reads `backend/Student Depression Dataset.csv`, trains the full pipeline, and writes:
- `backend/model/depression_model.pkl`
- `backend/model/depression_metadata.pkl`

Alternatively, open `backend/train_model.ipynb` in Jupyter to run the full EDA and training workflow step by step.

### 5. Start the Flask backend

```bash
python backend/app.py
```

The API will be available at `http://localhost:5000`.

### 6. Start the React frontend

```bash
npm run dev
```

The app will be available at `http://localhost:3000`. Vite proxies all `/api/*` requests to `http://localhost:5000`.

### 7. Build for production

```bash
npm run build
```

Output goes to the `dist/` folder and can be served by any static file server.

---

## Configuration

### Vite ([vite.config.js](vite.config.js))

```js
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:5000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

All `/api/predict`, `/api/export-pdf`, and `/api/model-info` calls from the frontend are transparently forwarded to Flask during development.

### Flask ([backend/app.py](backend/app.py))

| Setting | Value |
|---|---|
| Host | 0.0.0.0 |
| Port | 5000 |
| Debug | True (development) |
| CORS | All origins |
| Model directory | `backend/model/` |

---

## Assessment Input Features

| Field | UI Control | Type | Range / Options |
|---|---|---|---|
| Age | Slider | Numeric | 18–59 |
| Gender | Option Chips | Categorical | Male, Female |
| Academic Pressure | Slider | Numeric | 0–5 |
| Work Pressure | Slider | Numeric | 0–5 |
| CGPA | Slider | Numeric | 0.0–10.0 |
| Study Satisfaction | Slider | Numeric | 0–5 |
| Job Satisfaction | Slider | Numeric | 0–4 |
| Work/Study Hours | Slider | Numeric | 0–12 |
| Financial Stress | Slider | Numeric | 1–5 |
| Sleep Duration | Option Chips | Categorical | Less than 5 hours, 5-6 hours, 7-8 hours, More than 8 hours, Others |
| Dietary Habits | Option Chips | Categorical | Healthy, Moderate, Unhealthy |
| Suicidal Thoughts | Toggle | Categorical | Yes, No |
| Family History | Toggle | Categorical | Yes, No |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check and model status |
| POST | `/predict` | Run depression classification |
| POST | `/export-pdf` | Generate PDF report |
| GET | `/model-info` | Retrieve model metadata |

All endpoints return JSON except `/export-pdf`, which returns `application/pdf`.

---

## PDF Export

The export uses **reportlab** to generate a multi-section PDF server-side:

- Triggered by `POST /export-pdf` with the full prediction result as the request body
- Matplotlib charts (probability histogram, feature importance pie chart) are embedded as images
- Icons from `src/assets/` are embedded alongside recommendation text
- Includes contact information for mental health support organizations:
  - **Egypt:** ENMA (Egyptian National Mental Health Association), Behman Hospital
  - **Global:** WHO Mental Health, Crisis Text Line
- Closes with a medical disclaimer

---

## History & Persistence

- Results are stored in browser **localStorage** under the key `mc_history`
- Maximum of **30 assessments** are retained; oldest entries are pruned automatically
- Each stored entry contains: timestamp, stressLevel, confidence, probabilities, radarScores
- The History page reads from localStorage directly — no backend call is made
- `clearHistory()` removes all entries from localStorage and resets the UI state

---

## Disease Risk Assessment

The backend applies a **rule-based engine** (separate from the ML model) to flag comorbid conditions based on raw input values:

| Condition | Triggered by |
|---|---|
| Academic Burnout | High academic pressure + low study satisfaction |
| Anxiety Disorder | High work/academic pressure combined |
| Sleep Disorder / Insomnia | Sleep duration less than 6 hours |
| Nutritional Imbalance | Unhealthy dietary habits |
| Hereditary Depression Risk | Family history = Yes |
| Mental Health Crisis Risk | Suicidal thoughts = Yes |
| Academic Stress Syndrome | High work/study hours + high academic pressure |
| Financial Stress Syndrome | Financial stress >= 4 |

Each condition is rated **Low / Moderate / High** with a short indicator description displayed on the risk card.

---

## Coping Recommendations

Four personalized coping strategy cards are returned with every prediction. Each card has:

- **Category** — short label (e.g., "Maintain Your Balance")
- **Text** — actionable advice tailored to the predicted class
- **Icon** — one of 13 custom PNG icons from `src/assets/`

Strategies differ between **No Depression** (maintenance-focused advice) and **Depression** (recovery-focused advice) outcomes.

---

## Disclaimer

MindCheck is a **screening tool only** and does not constitute a medical diagnosis. Results are based on statistical patterns learned from training data and should not replace consultation with a licensed mental health professional. If you are experiencing a mental health crisis, please contact a qualified healthcare provider or crisis helpline immediately.

---
## UI
<img width="1248" height="1395" alt="image" src="https://github.com/user-attachments/assets/05d4fd49-114e-4929-91dd-39e8f020b4be" />
<img width="994" height="1168" alt="image" src="https://github.com/user-attachments/assets/57e44c03-71fe-441b-a45f-2184364ca451" />
<img width="960" height="955" alt="image" src="https://github.com/user-attachments/assets/be1625e9-7d18-4f33-a6e9-04edcddd5710" />
<img width="986" height="825" alt="image" src="https://github.com/user-attachments/assets/444d9355-a23a-4ee9-9ca3-5dcb62149105" />

 ---
 ## PDF Sample
 <img width="969" height="1314" alt="Screenshot 2026-05-14 010442" src="https://github.com/user-attachments/assets/2dde5ee4-6b07-49bd-a9f8-f6b5c4a228ce" />
