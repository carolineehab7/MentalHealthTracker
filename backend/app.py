"""
MindCheck – Flask REST API  (Depression Classifier)
Member 2: Backend Developer
============================
Install : pip install flask flask-cors joblib pandas scikit-learn reportlab
Run     : python app.py
Base URL: http://localhost:5000

Endpoints
---------
GET  /              → health check
POST /predict       → run depression ML model
POST /export-pdf    → generate downloadable PDF report
GET  /model-info    → model metadata + feature importances
"""

import io, os, sys, json
import joblib
from datetime   import datetime
from flask      import Flask, request, jsonify, send_file
from flask_cors import CORS

ML_DIR = os.path.dirname(__file__)
sys.path.insert(0, ML_DIR)
from train_model import (
    predict_depression,
    FEATURE_COLS,
    LABEL_MAP,
    FEATURE_LABELS,
)

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(ML_DIR, "model")

# ─────────────────────────────────────────────
# COPING SUGGESTIONS  (keyed by prediction)
# ─────────────────────────────────────────────
COPING = {
    "No Depression": [
        {"category": "Maintain Your Balance",
         "text": "You're showing resilience. Keep your current routines of adequate sleep, balanced meals, and social connection."},
        {"category": "Stay Physically Active",
         "text": "Regular exercise (30 min/day) is a powerful preventive measure — it boosts mood and reduces future risk."},
        {"category": "Academic Balance",
         "text": "Take regular study breaks. The Pomodoro technique (25 min on, 5 min off) prevents cognitive burnout."},
        {"category": "Stay Connected",
         "text": "Nurture your social network. Strong relationships act as a long-term buffer against depression."},
    ],
    "Depression": [
        {"category": "Seek Professional Help", 
         "text": "Consider speaking with a counselor, therapist, or psychiatrist. Reaching out is a sign of strength, not weakness."},
        {"category": "Reduce Your Load", 
         "text": "Talk to your academic advisor or employer about workload adjustments. Protecting your mental health comes first."},
        {"category": "Prioritise Sleep", 
         "text": "Aim for 7–9 hours of sleep each night. A consistent sleep schedule significantly improves mood regulation."},
        {"category": "Build Your Support Network", 
         "text": "Reach out to a trusted friend, family member, or helpline. Isolation worsens depression — connection heals it."},
    ],
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def map_ui_to_features(data: dict) -> dict:
    """
    Converts the simplified UI payload into the dataset column format the
    ML pipeline expects.  Keys must match the training DataFrame columns exactly.
    """
    return {
        "Age":                                   float(data.get("age",                22)),
        "Gender":                                data.get("gender",                   "Male"),
        "Academic Pressure":                     float(data.get("academic_pressure",   2)),
        "Work Pressure":                         float(data.get("work_pressure",       2)),
        "CGPA":                                  float(data.get("cgpa",               7.0)),
        "Study Satisfaction":                    float(data.get("study_satisfaction",  2)),
        "Job Satisfaction":                      float(data.get("job_satisfaction",    2)),
        "Work/Study Hours":                      float(data.get("work_study_hours",    6)),
        "Financial Stress":                      float(data.get("financial_stress",    2)),
        "Sleep Duration":                        data.get("sleep_duration",            "7-8 hours"),
        "Dietary Habits":                        data.get("dietary_habits",            "Moderate"),
        "Have you ever had suicidal thoughts ?": data.get("suicidal_thoughts",         "No"),
        "Family History of Mental Illness":      data.get("family_history",            "No"),
    }


# Sleep Duration → quality score 0-100 (higher = better sleep)
_SLEEP_SCORES = {
    "Less than 5 hours": 15,
    "5-6 hours":         45,
    "7-8 hours":         85,
    "More than 8 hours": 65,
    "Others":            50,
}

# Dietary Habits → satisfaction score 0-100
_DIET_SCORES = {
    "Healthy":   90,
    "Moderate":  55,
    "Unhealthy": 20,
    "Others":    50,
}


def build_radar(feat: dict) -> dict:
    sleep_score = _SLEEP_SCORES.get(feat.get("Sleep Duration", "7-8 hours"), 50)
    diet_score  = _DIET_SCORES.get(feat.get("Dietary Habits",  "Moderate"),  55)

    academic   = round(feat.get("Academic Pressure", 2) / 5 * 100)
    financial  = round((feat.get("Financial Stress", 2) - 1) / 4 * 100)
    workload   = round(feat.get("Work/Study Hours",  6) / 12 * 100)

    study_sat  = feat.get("Study Satisfaction", 2) / 5
    job_sat    = feat.get("Job Satisfaction",   2) / 4
    satisfaction = round((study_sat + job_sat) / 2 * 100)

    return {
        "academic":     academic,
        "financial":    financial,
        "sleep":        sleep_score,
        "workload":     workload,
        "satisfaction": satisfaction,
    }


# ─────────────────────────────────────────────
# DISEASE RISK  (rule-based, uses raw UI values)
# ─────────────────────────────────────────────
def compute_disease_risk(ui: dict) -> list:
    academic  = float(ui.get("academic_pressure",  2))   # 0-5
    work      = float(ui.get("work_pressure",      2))   # 0-5
    financial = float(ui.get("financial_stress",   2))   # 1-5
    hours     = float(ui.get("work_study_hours",   6))   # 0-12
    cgpa      = float(ui.get("cgpa",              7.0))  # 0-10
    study_sat = float(ui.get("study_satisfaction", 2))   # 0-5
    job_sat   = float(ui.get("job_satisfaction",   2))   # 0-4
    sleep     = ui.get("sleep_duration",  "7-8 hours")
    diet      = ui.get("dietary_habits",  "Moderate")
    suicidal  = ui.get("suicidal_thoughts", "No")
    family    = ui.get("family_history",    "No")

    risks = []

    # ── Academic Burnout ──────────────────────
    burnout = (academic / 5) * 0.40 + (hours / 12) * 0.35 + ((5 - study_sat) / 5) * 0.25
    if burnout > 0.45:
        risks.append({
            "condition": "Academic Burnout",
            "risk":      "High" if burnout > 0.68 else "Moderate",
            "indicator": "High academic pressure combined with long study hours and low satisfaction signals burnout.",
        })

    # ── Anxiety Disorder ──────────────────────
    anxiety = (academic / 5) * 0.35 + (work / 5) * 0.30 + ((financial - 1) / 4) * 0.35
    if anxiety > 0.45:
        risks.append({
            "condition": "Anxiety Disorder",
            "risk":      "High" if anxiety > 0.68 else "Moderate",
            "indicator": "Persistent academic, work, and financial pressure are major risk factors for anxiety disorder.",
        })

    # ── Sleep Disorder ────────────────────────
    if sleep in ("Less than 5 hours", "5-6 hours"):
        risks.append({
            "condition": "Sleep Disorder / Insomnia",
            "risk":      "High" if sleep == "Less than 5 hours" else "Moderate",
            "indicator": f"Sleeping {sleep} is below the recommended 7–9 hours and worsens depression and cognitive function.",
        })

    # ── Nutritional Imbalance ─────────────────
    if diet == "Unhealthy":
        risks.append({
            "condition": "Nutritional Imbalance",
            "risk":      "Moderate",
            "indicator": "Poor dietary habits are linked to increased depression risk and reduced cognitive performance.",
        })

    # ── Hereditary / Family Risk ──────────────
    if family == "Yes":
        fam_score = 0.55 + ((financial - 1) / 4) * 0.25 + (academic / 5) * 0.20
        risks.append({
            "condition": "Hereditary Depression Risk",
            "risk":      "High" if fam_score > 0.72 else "Moderate",
            "indicator": "Family history of mental illness significantly raises personal risk. Proactive monitoring is advised.",
        })

    # ── Suicidal Ideation History ─────────────
    if suicidal == "Yes":
        risks.append({
            "condition": "Mental Health Crisis Risk",
            "risk":      "High",
            "indicator": "A history of suicidal thoughts requires professional mental health support. Please consult a specialist.",
        })

    # ── Academic Underperformance ─────────────
    if cgpa < 5.0 and academic > 3:
        risks.append({
            "condition": "Academic Stress Syndrome",
            "risk":      "Moderate",
            "indicator": "Low CGPA combined with high academic pressure creates a cycle of stress and poor performance.",
        })

    # ── Financial Depression ──────────────────
    if financial > 3.5 and job_sat < 2:
        risks.append({
            "condition": "Financial Stress Syndrome",
            "risk":      "High" if financial > 4.5 else "Moderate",
            "indicator": "Severe financial stress combined with low job satisfaction is a significant depression trigger.",
        })

    if not risks:
        risks.append({
            "condition": "No Significant Risk Detected",
            "risk":      "Low",
            "indicator": "Your current metrics suggest a balanced lifestyle. Continue maintaining your healthy habits.",
        })

    return risks


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    model_ok = os.path.exists(os.path.join(MODEL_DIR, "depression_model.pkl"))
    return jsonify({
        "status":      "ok",
        "service":     "MindCheck Depression Classifier API",
        "version":     "3.0.0",
        "model_ready": model_ok,
        "features":    len(FEATURE_COLS),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON) – fields sent by the React frontend:
    {
        age, gender, academic_pressure, work_pressure, cgpa,
        study_satisfaction, job_satisfaction, work_study_hours,
        financial_stress, sleep_duration, dietary_habits,
        suicidal_thoughts, family_history
    }
    """
    try:
        data         = request.get_json(force=True)
        features     = map_ui_to_features(data)
        result       = predict_depression(features)
        radar        = build_radar(features)
        disease_risk = compute_disease_risk(data)

        prediction = result["stressLevel"]   # "Depression" or "No Depression"

        return jsonify({
            "stressLevel":       prediction,
            "confidence":        result["confidence"],
            "probabilities":     result["probabilities"],
            "radarScores":       radar,
            "featureImportance": result["featureImportance"][:5],
            "suggestions":       COPING.get(prediction, COPING["No Depression"]),
            "diseaseRisk":       disease_risk,
            "timestamp":         datetime.now().isoformat(),
        })

    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run: cd backend && python train_model.py"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    """
    POST /export-pdf
    Body: the full predict response JSON the frontend already has.
    Returns a downloadable PDF report.
    """
    try:
        from reportlab.lib.pagesizes   import A4
        from reportlab.lib             import colors
        from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units       import cm
        from reportlab.platypus        import (SimpleDocTemplate, Paragraph,
                                               Spacer, Table, TableStyle,
                                               HRFlowable)

        data      = request.get_json(force=True)
        level     = data.get("stressLevel", "N/A")
        conf      = data.get("confidence",  0)
        sugs      = data.get("suggestions", [])
        imp       = data.get("featureImportance", [])
        probs     = data.get("probabilities", {})
        ts        = data.get("timestamp", datetime.now().isoformat())[:19].replace("T", " ")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm,  bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        def P(text, size=11, bold=False, color="#1c1916", space=8):
            return Paragraph(text, ParagraphStyle(
                "x", fontSize=size, spaceAfter=space,
                fontName="Helvetica-Bold" if bold else "Helvetica",
                textColor=colors.HexColor(color),
            ))

        story += [
            P("MindCheck — Depression Assessment Report", 20, bold=True, space=4),
            P(f"Generated: {ts}", 10, color="#9c968f", space=14),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e8e4de")),
            Spacer(1, 0.4*cm),
        ]

        level_color = {"No Depression": "#15803d", "Depression": "#b91c1c"}.get(level, "#000")
        story.append(P(f'Prediction: <font color="{level_color}"><b>{level}</b></font>'
                       f'&nbsp;&nbsp;|&nbsp;&nbsp;Confidence: <b>{conf}%</b>', 14, space=14))

        story.append(P("Probability Distribution", 12, bold=True, space=6))
        prob_data = [["Outcome", "Probability"]] + [[k, f"{v}%"] for k, v in probs.items()]
        t = Table(prob_data, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1c1916")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0ede8"), colors.white]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#e8e4de")),
            ("FONTSIZE",       (0, 0), (-1, -1), 10),
            ("PADDING",        (0, 0), (-1, -1), 7),
        ]))
        story += [t, Spacer(1, 0.5*cm)]

        if imp:
            story.append(P("Top Contributing Factors", 12, bold=True, space=6))
            imp_data = [["Factor", "Importance"]] + [[i["label"], f"{i['pct']}%"] for i in imp]
            t2 = Table(imp_data, colWidths=[12*cm, 4*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#c8440a")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0ede8"), colors.white]),
                ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#e8e4de")),
                ("FONTSIZE",       (0, 0), (-1, -1), 10),
                ("PADDING",        (0, 0), (-1, -1), 7),
            ]))
            story += [t2, Spacer(1, 0.5*cm)]

        story.append(P("Personalised Recommendations", 12, bold=True, space=6))
        for s in sugs:
            story.append(P(s["text"], 10, space=10))

        story += [
            Spacer(1, 0.4*cm),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e8e4de")),
            Spacer(1, 0.2*cm),
            P("This report is for informational purposes only and does not constitute "
              "medical advice. If you are experiencing distress, please consult a "
              "qualified mental health professional.", 8, color="#9c968f"),
        ]

        doc.build(story)
        buf.seek(0)
        fname = f"depression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=True, download_name=fname)

    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model":       "Logistic Regression / Random Forest (auto-selected)",
        "features":    len(FEATURE_COLS),
        "feature_list": FEATURE_COLS,
        "classes":     list(LABEL_MAP.values()),
        "dataset":     "Student Depression Dataset (27,901 records)",
        "featureLabels": FEATURE_LABELS,
    })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(os.path.join(MODEL_DIR, "depression_model.pkl")):
        print("⚠  Model not found. Run:  python train_model.py")
    else:
        print("✓  Depression model loaded — 13 features active")
    print("✓  API starting on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
