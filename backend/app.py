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
    Converts the simplified UI payload into the full 20-feature vector the
    ML model expects.  Scales are matched to the training dataset:
      anxiety_level  0-21  |  self_esteem   0-30  |  depression   0-27
      headache       0-5   |  sleep_quality 0-5   |  study_load   0-5
      future_career  0-5   |  peer_pressure 0-5   |  bullying     0-5
      blood_pressure 1-3   |  breathing     0-5   |  social_support 0-3
    """
    mood_val  = MOOD_MAP.get(data.get("mood", "Neutral"), 15)   # 0-30
    sleep_h   = float(data.get("sleep_hours",            7))
    anxiety   = int(  data.get("anxiety_score",          5))    # UI 1-10
    social    = int(  data.get("social_activity",        5))    # UI 1-10
    work_h    = float(data.get("work_hours",             8))
    dep_ui    = int(  data.get("depression",             3))    # UI 0-10
    headache_ui = int(data.get("headache",               2))    # UI 0-10
    career_ui = int(  data.get("future_career_concerns", 5))    # UI 0-10
    peer_ui   = int(  data.get("peer_pressure",          3))    # UI 0-10

    # Scale UI 1-10 → training 0-21
    anxiety_scaled = min(21, round(anxiety * 2.1))
    # Scale UI 0-10 → training 0-27
    depression_scaled = min(27, round(dep_ui * 2.7))
    # Sleep hours → quality score 0-5 (higher hours = better quality)
    sleep_q = min(5, round((sleep_h - 2) / 10 * 5))
    # Scale UI 0-10 → training 0-5
    headache_scaled = min(5, round(headache_ui / 2))
    study_ld        = min(5, round(work_h / 16 * 5))
    career_scaled   = min(5, round(career_ui / 2))
    peer_scaled     = min(5, round(peer_ui / 2))
    # Social support UI 1-10 → 0-3
    soc_sup = min(3, max(0, round(social / 3.5)))
    # Binary toggles mapped to training scale
    bp        = 3 if data.get("blood_pressure",  0) else 1   # 1=normal, 3=high
    breathing = 4 if data.get("breathing_problem", 0) else 0  # 0-5 scale
    bullying  = 5 if data.get("bullying",          0) else 0  # 0-5 scale

    # Infer environmental factors (0-5) from visible stress indicators
    stress_proxy = (anxiety / 10 + dep_ui / 10) / 2   # 0-1
    living_cond = max(0, min(5, round(5 - stress_proxy * 4)))
    safety_val  = max(0, min(5, round(5 - stress_proxy * 3)))
    basic_needs = max(0, min(5, round(5 - stress_proxy * 3)))
    acad_perf   = max(0, min(5, round(5 - stress_proxy * 4)))

    return {
        "anxiety_level":                anxiety_scaled,
        "self_esteem":                  mood_val,
        "mental_health_history":        int(data.get("mental_health_history", 0)),
        "depression":                   depression_scaled,
        "headache":                     headache_scaled,
        "blood_pressure":               bp,
        "sleep_quality":                sleep_q,
        "breathing_problem":            breathing,
        "noise_level":                  3,
        "living_conditions":            living_cond,
        "safety":                       safety_val,
        "basic_needs":                  basic_needs,
        "academic_performance":         acad_perf,
        "study_load":                   study_ld,
        "teacher_student_relationship": 3,
        "future_career_concerns":       career_scaled,
        "social_support":               soc_sup,
        "peer_pressure":                peer_scaled,
        "extracurricular_activities":   3,
        "bullying":                     bullying,
    }


def build_radar(feat: dict) -> dict:
    return {
        "sleep":    round(feat.get("sleep_quality",  3) / 5  * 100),   # 0-5
        "anxiety":  round(feat.get("anxiety_level", 10) / 21 * 100),   # 0-21
        "social":   round(feat.get("social_support", 2) / 3  * 100),   # 0-3
        "workload": round(feat.get("study_load",      3) / 5  * 100),   # 0-5
        "mood":     round(feat.get("self_esteem",    15) / 30 * 100),   # 0-30
    }


# ─────────────────────────────────────────────
# DISEASE RISK  (rule-based, uses raw UI values)
# ─────────────────────────────────────────────
def compute_disease_risk(ui: dict) -> list:
    """
    Returns a list of disease risk objects derived from the raw UI inputs.
    Each entry: { condition, risk, icon, indicator }
    """
    anxiety  = int(  ui.get("anxiety_score",          5))   # 1-10
    sleep_h  = float(ui.get("sleep_hours",             7))
    dep      = int(  ui.get("depression",              3))   # 0-10
    work_h   = float(ui.get("work_hours",              8))
    headache = int(  ui.get("headache",                2))   # 0-10
    career   = int(  ui.get("future_career_concerns",  5))   # 0-10
    peer     = int(  ui.get("peer_pressure",           3))   # 0-10
    social   = int(  ui.get("social_activity",         5))   # 1-10
    bp       = int(  ui.get("blood_pressure",          0))   # 0/1
    breathing= int(  ui.get("breathing_problem",       0))   # 0/1
    bullying = int(  ui.get("bullying",                0))   # 0/1
    mh_hist  = int(  ui.get("mental_health_history",   0))   # 0/1
    mood     = ui.get("mood", "Neutral")

    risks = []

    # ── Burnout Syndrome ─────────────────────────
    burnout = (work_h / 16) * 0.40 + (career / 10) * 0.35 + (anxiety / 10) * 0.25
    if burnout > 0.45:
        risks.append({
            "condition": "Burnout Syndrome",
            "risk": "High" if burnout > 0.68 else "Moderate",
            "icon": "🔥",
            "indicator": "Sustained high workload combined with career anxiety can cause emotional exhaustion.",
        })

    # ── Generalized Anxiety Disorder ─────────────
    gad = (anxiety / 10) * 0.45 + (breathing * 0.15) + (peer / 10) * 0.20 + \
          (0.20 if mood in ("Very sad", "Stressed") else 0)
    if gad > 0.45:
        risks.append({
            "condition": "Generalized Anxiety Disorder",
            "risk": "High" if gad > 0.68 else "Moderate",
            "icon": "😰",
            "indicator": "Persistent high anxiety with physical symptoms such as breathing issues is a clinical marker.",
        })

    # ── Clinical Depression ───────────────────────
    dep_score = (dep / 10) * 0.50 + ((10 - social) / 10) * 0.30 + \
                (0.20 if mood == "Very sad" else 0.10 if mood == "Stressed" else 0)
    if dep_score > 0.38:
        risks.append({
            "condition": "Clinical Depression",
            "risk": "High" if dep_score > 0.62 else "Moderate",
            "icon": "🌧️",
            "indicator": "Elevated depression scores combined with social withdrawal are key diagnostic markers.",
        })

    # ── Insomnia / Sleep Disorder ─────────────────
    insomnia = (max(0, 8 - sleep_h) / 6) * 0.60 + (anxiety / 10) * 0.40
    if insomnia > 0.38:
        risks.append({
            "condition": "Insomnia / Sleep Disorder",
            "risk": "High" if sleep_h < 5 and anxiety > 6 else "Moderate",
            "icon": "😴",
            "indicator": f"Sleeping {sleep_h:.0f} h/night with elevated anxiety chronically disrupts restorative sleep.",
        })

    # ── Tension Headache / Migraine ───────────────
    head_score = (headache / 10) * 0.50 + (anxiety / 10) * 0.30 + (work_h / 16) * 0.20
    if head_score > 0.38:
        risks.append({
            "condition": "Tension Headache / Migraine",
            "risk": "High" if head_score > 0.60 else "Moderate",
            "icon": "🤕",
            "indicator": "Frequent headaches under sustained stress suggest a chronic tension pattern.",
        })

    # ── Hypertension Risk ─────────────────────────
    if bp:
        hyp = 0.50 + (anxiety / 10) * 0.30 + (max(0, 8 - sleep_h) / 6) * 0.20
        risks.append({
            "condition": "Hypertension Risk",
            "risk": "High" if hyp > 0.70 else "Moderate",
            "icon": "❤️",
            "indicator": "Existing blood pressure issues are significantly amplified by chronic stress and poor sleep.",
        })

    # ── PTSD / Trauma-related ─────────────────────
    if bullying and (anxiety > 6 or dep > 5):
        risks.append({
            "condition": "Trauma / PTSD Risk",
            "risk": "High" if anxiety > 7 and dep > 6 else "Moderate",
            "icon": "🛡️",
            "indicator": "Exposure to bullying alongside high anxiety and depression can trigger trauma responses.",
        })

    # ── All clear ─────────────────────────────────
    if not risks:
        risks.append({
            "condition": "No Significant Disease Risk Detected",
            "risk": "Low",
            "icon": "✅",
            "indicator": "Your current metrics do not indicate immediate disease risk. Keep maintaining healthy habits.",
        })

    return risks


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
        data        = request.get_json(force=True)
        features    = map_ui_to_features(data)
        result      = predict_stress(features)
        radar       = build_radar(features)
        disease_risk = compute_disease_risk(data)

        return jsonify({
            "stressLevel":       result["stressLevel"],
            "confidence":        result["confidence"],
            "probabilities":     result["probabilities"],
            "radarScores":       radar,
            "featureImportance": result["featureImportance"][:5],
            "suggestions":       COPING[result["stressLevel"]],
            "diseaseRisk":       disease_risk,
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
