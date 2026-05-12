"""
MindCheck – Flask REST API
Member 2: Backend Developer
============================
Install : pip install flask flask-cors joblib pandas scikit-learn reportlab
Run     : python app.py
Base URL: http://localhost:5000

Endpoints
---------
GET  /              → health check
POST /predict       → run ML model, return stress classification
POST /export-pdf    → generate downloadable PDF report
GET  /model-info    → model metadata + feature importances
"""

import io, os, sys, json
import joblib
from datetime   import datetime
from flask      import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── import shared constants + inference function from ml/ ──
ML_DIR = os.path.dirname(__file__)          # backend/
sys.path.insert(0, ML_DIR)
from train_model import (
    predict_stress,
    MOOD_MAP,
    FEATURE_COLS,
    LABEL_MAP,
    FEATURE_LABELS,
)

app = Flask(__name__)
CORS(app)   # allow all origins (needed by Vite dev server on :3000)

MODEL_DIR = os.path.join(ML_DIR, "model")  # backend/model/

# ─────────────────────────────────────────────
# COPING SUGGESTIONS  (keyed by stress level)
# ─────────────────────────────────────────────
COPING = {
    "Low": [
        {"category": "Maintenance",  "icon": "✅",
         "text": "Your stress levels look healthy. Keep maintaining your current sleep schedule and social habits."},
        {"category": "Exercise",     "icon": "🏃",
         "text": "Light daily exercise (20–30 min walk) reinforces your low-stress baseline and boosts mood."},
        {"category": "Mindfulness",  "icon": "🧘",
         "text": "A 5-minute daily breathing exercise helps keep stress from creeping up during busy periods."},
        {"category": "Social",       "icon": "🤝",
         "text": "Continue nurturing your social connections — they are a proven buffer against future stress."},
    ],
    "Moderate": [
        {"category": "Sleep",        "icon": "😴",
         "text": "Prioritise 7–9 hours of sleep. Set a consistent bedtime and avoid screens 30 minutes before bed."},
        {"category": "Workload",     "icon": "📋",
         "text": "Try the Pomodoro technique: 25 min focused work + 5 min break. This prevents long-term burnout."},
        {"category": "Social",       "icon": "💬",
         "text": "Schedule at least one social activity this week — even a short call with a friend reduces cortisol."},
        {"category": "Mindfulness",  "icon": "🧘",
         "text": "Try a 10-minute guided meditation (Headspace, Calm) before bed to lower residual anxiety."},
    ],
    "High": [
        {"category": "Urgent: Sleep","icon": "🚨",
         "text": "Your sleep is critically affecting your health. See a doctor if you consistently sleep fewer than 6 hours."},
        {"category": "Professional", "icon": "🏥",
         "text": "Consider speaking with a mental health professional — high stress sustained over days is a clinical concern."},
        {"category": "Immediate",    "icon": "🌬️",
         "text": "Try box breathing right now: inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 5 times."},
        {"category": "Reduce Load",  "icon": "⚠️",
         "text": "Identify your top 3 stressors and discuss workload reduction with your manager or support network today."},
    ],
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def map_ui_to_features(data: dict) -> dict:
    """
    Converts the simplified 12-field UI payload into the full
    20-feature vector the ML model expects.
    """
    mood_val = MOOD_MAP.get(data.get("mood", "Neutral"), 5)
    sleep_h  = float(data.get("sleep_hours",     7))
    anxiety  = int(  data.get("anxiety_score",   5))
    social   = int(  data.get("social_activity", 5))
    work_h   = float(data.get("work_hours",       8))

    sleep_q  = min(10, round(sleep_h / 12 * 10))
    study_ld = min(10, round(work_h  / 16 * 10))
    soc_sup  = min(3,  max(1, round(social / 3.5)))

    return {
        "anxiety_level":                anxiety,
        "self_esteem":                  mood_val,
        "mental_health_history":        int(data.get("mental_health_history", 0)),
        "depression":                   int(data.get("depression", max(0, anxiety - 3))),
        "headache":                     int(data.get("headache",   0)),
        "blood_pressure":               int(data.get("blood_pressure", 0)),
        "sleep_quality":                sleep_q,
        "breathing_problem":            int(data.get("breathing_problem", 0)),
        "noise_level":                  int(data.get("noise_level",   3)),
        "living_conditions":            int(data.get("living_conditions", 5)),
        "safety":                       int(data.get("safety",       7)),
        "basic_needs":                  int(data.get("basic_needs",  7)),
        "academic_performance":         int(data.get("academic_performance", 5)),
        "study_load":                   study_ld,
        "teacher_student_relationship": int(data.get("teacher_student_relationship", 5)),
        "future_career_concerns":       int(data.get("future_career_concerns",       5)),
        "social_support":               soc_sup,
        "peer_pressure":                int(data.get("peer_pressure",  3)),
        "extracurricular_activities":   int(data.get("extracurricular_activities",   3)),
        "bullying":                     int(data.get("bullying",       0)),
    }


def build_radar(feat: dict) -> dict:
    return {
        "sleep":    round(feat.get("sleep_quality",  5) / 10 * 100),
        "anxiety":  round(feat.get("anxiety_level",  5) * 10),
        "social":   round(feat.get("social_support", 2) / 3  * 100),
        "workload": round(feat.get("study_load",      5) * 10),
        "mood":     round(feat.get("self_esteem",     5) * 10),
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    model_ok = os.path.exists(os.path.join(MODEL_DIR, "stress_model.pkl"))
    return jsonify({
        "status":      "ok",
        "service":     "MindCheck Stress Classifier API",
        "version":     "2.0.0",
        "model_ready": model_ok,
        "features":    len(FEATURE_COLS),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON) – fields sent by the React frontend:
    {
        sleep_hours, anxiety_score, social_activity, work_hours, mood,
        depression, future_career_concerns, peer_pressure, headache,
        mental_health_history, blood_pressure, breathing_problem, bullying
    }
    """
    try:
        data     = request.get_json(force=True)
        features = map_ui_to_features(data)
        result   = predict_stress(features)
        radar    = build_radar(features)

        return jsonify({
            "stressLevel":       result["stressLevel"],
            "confidence":        result["confidence"],
            "probabilities":     result["probabilities"],
            "radarScores":       radar,
            "featureImportance": result["featureImportance"][:5],
            "suggestions":       COPING[result["stressLevel"]],
            "timestamp":         datetime.now().isoformat(),
        })

    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run: cd ml && python train_model.py"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    """
    POST /export-pdf
    Body: the full predict response JSON the frontend already has.
    Returns a downloadable PDF file.
    """
    try:
        from reportlab.lib.pagesizes   import A4
        from reportlab.lib             import colors
        from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units       import cm
        from reportlab.platypus        import (SimpleDocTemplate, Paragraph,
                                               Spacer, Table, TableStyle,
                                               HRFlowable)

        data  = request.get_json(force=True)
        level = data.get("stressLevel", "N/A")
        conf  = data.get("confidence",  0)
        sugs  = data.get("suggestions", [])
        imp   = data.get("featureImportance", [])
        probs = data.get("probabilities", {})
        ts    = data.get("timestamp", datetime.now().isoformat())[:19].replace("T", " ")

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

        # Title block
        story += [
            P("MindCheck — Stress Assessment Report", 20, bold=True, space=4),
            P(f"Generated: {ts}", 10, color="#9c968f", space=14),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e8e4de")),
            Spacer(1, 0.4*cm),
        ]

        # Result
        level_color = {"Low": "#15803d", "Moderate": "#b45309", "High": "#b91c1c"}.get(level, "#000")
        story.append(P(f'Stress Level: <font color="{level_color}"><b>{level}</b></font>'
                       f'&nbsp;&nbsp;|&nbsp;&nbsp;Confidence: <b>{conf}%</b>', 14, space=14))

        # Probability table
        story.append(P("Probability Distribution", 12, bold=True, space=6))
        prob_data = [["Class", "Probability"]] + [[k, f"{v}%"] for k, v in probs.items()]
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

        # Feature importance table
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

        # Suggestions
        story.append(P("Personalised Coping Suggestions", 12, bold=True, space=6))
        for s in sugs:
            story.append(P(f'<b>{s["icon"]} {s["category"]}</b>', 10,
                           color="#c8440a", space=2))
            story.append(P(s["text"], 10, space=10))

        # Disclaimer
        story += [
            Spacer(1, 0.4*cm),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e8e4de")),
            Spacer(1, 0.2*cm),
            P("This report is for informational purposes only and does not constitute "
              "medical advice. If you are experiencing severe distress, please consult "
              "a qualified healthcare professional.", 8, color="#9c968f"),
        ]

        doc.build(story)
        buf.seek(0)
        fname = f"stress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=True, download_name=fname)

    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    try:
        imp = joblib.load(os.path.join(MODEL_DIR, "importance.pkl")) if \
              os.path.exists(os.path.join(MODEL_DIR, "importance.pkl")) else []
    except Exception:
        imp = []
    return jsonify({
        "model":             "Random Forest (150 trees, max_depth=12)",
        "features":          len(FEATURE_COLS),
        "feature_list":      FEATURE_COLS,
        "classes":           list(LABEL_MAP.values()),
        "dataset":           "Student Stress Factors — Kaggle (CC0)",
        "featureImportance": imp,
    })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import joblib
    if not os.path.exists(os.path.join(MODEL_DIR, "stress_model.pkl")):
        print("⚠  Model not found. Run:  cd ml && python train_model.py")
    else:
        print("✓  Model loaded — 20 features active")
    print("✓  API starting on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
