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
        {"category": "Maintain Your Balance", "icon": "./../src/assets/life-skills.png",
         "text": "You're showing resilience. Keep your current routines of adequate sleep, balanced meals, and social connection."},
        {"category": "Stay Physically Active", "icon": "./../src/assets/excercise.png",
         "text": "Regular exercise (30 min/day) is a powerful preventive measure — it boosts mood and reduces future risk."},
        {"category": "Academic Balance", "icon": "./../src/assets/academic.png",
         "text": "Take regular study breaks. The Pomodoro technique (25 min on, 5 min off) prevents cognitive burnout."},
        {"category": "Stay Connected", "icon": "./../src/assets/community.png",
         "text": "Nurture your social network. Strong relationships act as a long-term buffer against depression."},
    ],
    "Depression": [
        {"category": "Seek Professional Help", "icon": "./../src/assets/helping-hand.png",
         "text": "Consider speaking with a counselor, therapist, or psychiatrist. Reaching out is a sign of strength, not weakness."},
        {"category": "Reduce Your Load", "icon": "./../src/assets/work-load.png",
         "text": "Talk to your academic advisor or employer about workload adjustments. Protecting your mental health comes first."},
        {"category": "Prioritise Sleep", "icon": "./../src/assets/sleeping.png",
         "text": "Aim for 7–9 hours of sleep each night. A consistent sleep schedule significantly improves mood regulation."},
        {"category": "Build Your Support Network", "icon": "./../src/assets/huddle.png",
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
        from reportlab.lib.enums       import TA_CENTER
        from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units       import cm
        from reportlab.platypus        import (SimpleDocTemplate, Paragraph,
                               Spacer, Table, TableStyle,
                               HRFlowable, Image, KeepInFrame)
        from reportlab.graphics.shapes import Drawing, Rect
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.piecharts import Pie

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

        BODY_FONT = "Times-Roman"
        BOLD_FONT = "Times-Bold"
        ITALIC_FONT = "Times-Italic"
        BASE_SIZE = 13
        BASE_COLOR = "#1f2937"
        ASSET_DIR = os.path.abspath(os.path.join(ML_DIR, "..", "src", "assets"))

        def P(text, size=BASE_SIZE, bold=False, color=BASE_COLOR, space=8):
            return Paragraph(text, ParagraphStyle(
                "x", fontSize=size, leading=size + 2, spaceAfter=space,
                fontName=BOLD_FONT if bold else BODY_FONT,
                textColor=colors.HexColor(color), alignment=TA_CENTER,
            ))

        small = ParagraphStyle(
            "small", parent=styles["Normal"], fontSize=BASE_SIZE - 1,
            leading=BASE_SIZE + 1, textColor=colors.HexColor(BASE_COLOR),
            fontName=BODY_FONT, alignment=TA_CENTER,
        )
        rec_style = ParagraphStyle(
            "rec", parent=styles["Normal"], fontSize=BASE_SIZE,
            leading=BASE_SIZE + 4,
            textColor=colors.HexColor(BASE_COLOR), alignment=TA_CENTER,
            fontName=BODY_FONT, italicFontName=ITALIC_FONT,
        )
        help_title = ParagraphStyle(
            "help_title", parent=styles["Normal"], fontSize=BASE_SIZE + 2,
            leading=BASE_SIZE + 4, textColor=colors.HexColor(BASE_COLOR),
            fontName=BOLD_FONT, alignment=TA_CENTER,
        )
        section_title = ParagraphStyle(
            "section_title", parent=styles["Normal"], fontSize=BASE_SIZE + 2,
            leading=BASE_SIZE + 4, textColor=colors.HexColor(BASE_COLOR),
            fontName=BOLD_FONT, alignment=TA_CENTER,
        )
        def short_label(text: str, max_len: int = 22) -> str:
            if not text:
                return ""
            return text if len(text) <= max_len else f"{text[:max_len - 3]}..."


        prob_colors = {
            "No Depression": colors.HexColor("#22c55e"),
            "Depression":    colors.HexColor("#ef4444"),
            "Low":           colors.HexColor("#22c55e"),
            "Moderate":      colors.HexColor("#f59e0b"),
            "High":          colors.HexColor("#ef4444"),
        }
        pie_colors = [
            colors.HexColor("#6b6bf1"),
            colors.HexColor("#f97316"),
            colors.HexColor("#22c55e"),
            colors.HexColor("#60a5fa"),
            colors.HexColor("#fbbf24"),
            colors.HexColor("#a78bfa"),
        ]

        HELP_ORGS = [
            {
                "name": "Befrienders Worldwide",
                "meta": "Global",
                "desc": "Find support lines in your country.",
                "url": "https://www.befrienders.org/",
            },
            {
                "name": "International Association for Suicide Prevention",
                "meta": "Global",
                "desc": "Directory of crisis centers worldwide.",
                "url": "https://www.iasp.info/resources/Crisis_Centres/",
            },
            {
                "name": "WHO Mental Health Resources",
                "meta": "Global",
                "desc": "Guidance and resources for mental health.",
                "url": "https://www.who.int/health-topics/mental-health#tab=tab_1",
            },
            {
                "name": "988 Lifeline",
                "meta": "United States",
                "desc": "Call or text 988 for 24/7 support.",
                "url": "https://988lifeline.org/",
            },
        ]

        def short_action(text: str) -> str:
            if not text:
                return ""
            first = text.split(".")[0].strip()
            if len(first) <= 64:
                return first
            return f"{first[:61].rstrip()}..."

        def build_histogram(probabilities: dict, width: float):
            raw_labels = list(probabilities.keys())
            values = [probabilities.get(k, 0) for k in raw_labels]
            if not raw_labels:
                return None

            labels = [short_label(k, 16) for k in raw_labels]
            drawing = Drawing(width, 150)
            chart = VerticalBarChart()
            chart.width = max(200, width - 110)
            chart.x = max((width - chart.width) / 2, 0)
            chart.y = 16
            chart.height = 100
            chart.data = [values]
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = 100
            chart.valueAxis.valueStep = 20
            chart.valueAxis.labels.fontSize = 8
            chart.categoryAxis.categoryNames = labels
            chart.categoryAxis.labels.fontSize = 9
            chart.categoryAxis.labels.angle = 25
            chart.categoryAxis.labels.dy = -8
            chart.barWidth = 16
            chart.groupSpacing = 8
            chart.barSpacing = 3

            for i, label in enumerate(raw_labels):
                chart.bars[(0, i)].fillColor = prob_colors.get(
                    label, colors.HexColor("#6b7280")
                )

            drawing.add(chart)
            wrapper = Table([[drawing]], colWidths=[width])
            wrapper.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            return wrapper

        def build_pie(importance: list, width: float):
            if not importance:
                return None

            data = [i.get("pct", 0) for i in importance]
            labels = [short_label(i.get("label", ""), 26) for i in importance]
            chart_size = 120
            drawing = Drawing(chart_size, chart_size)
            pie = Pie()
            pie.x = 0
            pie.y = 0
            pie.width = chart_size
            pie.height = chart_size
            pie.data = data
            pie.labels = None
            pie.simpleLabels = 0

            for i in range(len(data)):
                pie.slices[i].fillColor = pie_colors[i % len(pie_colors)]
                pie.slices[i].strokeColor = colors.white
                pie.slices[i].strokeWidth = 0.5

            drawing.add(pie)

            legend_rows = []
            for i, label in enumerate(labels):
                swatch = Drawing(10, 10)
                swatch.add(Rect(0, 0, 8, 8, fillColor=pie_colors[i % len(pie_colors)],
                                strokeColor=colors.HexColor("#e5e7eb")))
                legend_rows.append([
                    swatch,
                    Paragraph(f"{label} ({data[i]}%)", small),
                ])

            legend_table = Table(legend_rows, colWidths=[0.35 * cm, width - 4.5 * cm])
            legend_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]))

            container = Table([[drawing, legend_table]], colWidths=[4.5 * cm, width - 4.5 * cm])
            container.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            return container

        def icon_for_category(category: str) -> str:
            key = (category or "").lower()
            icon_dir = ASSET_DIR
            mapping = {
                "support": "huddle.png",
                "network": "huddle.png",
                "help": "helping-hand.png",
                "professional": "helping-hand.png",
                "sleep": "napping.png",
                "nap": "napping.png",
                "load": "work-load.png",
                "break": "work-load.png",
            }
            for token, fname in mapping.items():
                if token in key:
                    path = os.path.join(icon_dir, fname)
                    return path if os.path.exists(path) else ""
            return ""

        def build_recommendations(suggestions: list):
            if not suggestions:
                return None

            cells = []
            for s in suggestions[:4]:
                icon = s.get("icon", "")
                category = s.get("category", "").lower()
                action = short_action(s.get("text", ""))
                icon_path = icon_for_category(category)
                if icon_path:
                    icon_flowable = Image(icon_path, width=1.0 * cm, height=1.0 * cm)
                else:
                    icon_flowable = Paragraph(
                        f"<para align='center'><font size='14'>{icon}</font></para>",
                        rec_style,
                    )
                cell_text = Paragraph(
                    f"<para align='center'>{category} = <i>{action}</i></para>",
                    rec_style,
                )
                cells.append([icon_flowable, Spacer(1, 0.1 * cm), cell_text])

            rows = []
            for i in range(0, len(cells), 2):
                rows.append([
                    cells[i],
                    cells[i + 1] if i + 1 < len(cells) else "",
                ])

            row_heights = [2.2 * cm] * len(rows)
            table = Table(rows, colWidths=[doc.width / 2, doc.width / 2], rowHeights=row_heights)
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            return table

        def build_help_header():
            handshake_path = os.path.join(ASSET_DIR, "handshake.png")
            if os.path.exists(handshake_path):
                icon = Image(handshake_path, width=1.1 * cm, height=1.1 * cm)
            else:
                icon = Spacer(1, 1.1 * cm)

            header = Table(
                [[icon, Paragraph("Contact organizations for help", help_title)]],
                colWidths=[1.1 * cm, doc.width - 1.1 * cm],
            )
            header.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            return header

        def build_help_cards(help_items: list):
            if not help_items:
                return None

            cells = []
            help_style = ParagraphStyle(
                "help_card", parent=styles["Normal"], fontSize=BASE_SIZE - 1,
                leading=BASE_SIZE + 1, textColor=colors.HexColor(BASE_COLOR),
                fontName=BODY_FONT, alignment=TA_CENTER,
            )
            for org in help_items:
                text = (
                    f"<para align='center'><b>{org['name']}</b><br/>"
                    f"<i>{org['meta']}</i><br/>"
                    f"{org['desc']}<br/><font color='#2563eb'>{org['url']}</font></para>"
                )
                cell_text = Paragraph(text, help_style)
                cells.append([cell_text])

            rows = []
            for i in range(0, len(cells), 2):
                rows.append([
                    cells[i][0],
                    cells[i + 1][0] if i + 1 < len(cells) else "",
                ])

            row_heights = [2.2 * cm] * len(rows)
            table = Table(rows, colWidths=[doc.width / 2, doc.width / 2], rowHeights=row_heights)
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            return table

        story += [
            P("Depression indications", 16, bold=True, space=2),
            P(f"Generated: {ts}", BASE_SIZE - 1, color=BASE_COLOR, space=8),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e8e4de")),
            Spacer(1, 0.2 * cm),
        ]

        level_color = {"No Depression": "#15803d", "Depression": "#b91c1c"}.get(level, BASE_COLOR)
        story.append(P(f'Prediction: <font color="{level_color}"><b>{level}</b></font>'
                   f'&nbsp;&nbsp;|&nbsp;&nbsp;Confidence: <b>{conf}%</b>', 11, space=8))

        def section_block(title: str, body_flowable, height_cm: float, title_flowable=None, center=True):
            if title_flowable is not None:
                items = [title_flowable, Spacer(1, 0.15 * cm)]
            else:
                items = [Paragraph(title, section_title), Spacer(1, 0.15 * cm)] if title else []
            if title in ("Personalised Recommendations", "Contact organizations for help"):
                items.append(Spacer(1, 0.2 * cm))
            if body_flowable:
                items.append(body_flowable)
            else:
                items.append(P("No data available.", 9, color=BASE_COLOR, space=2))
            items.append(Spacer(1, 0.05 * cm))
            h_align = "CENTER" if center else "LEFT"
            return KeepInFrame(doc.width, height_cm * cm, items, hAlign=h_align, vAlign="TOP", mode="shrink")

        hist = build_histogram(probs, doc.width)
        pie = build_pie(imp[:5], doc.width) if imp else None
        rec_table = build_recommendations(sugs)
        help_table = build_help_cards(HELP_ORGS)
        help_header = build_help_header()

        section_height = 5.0
        sections = Table(
            [
                [section_block("Probability Distribution", hist, section_height)],
                [section_block("Top Contributing Factors", pie, section_height)],
                [section_block("Personalised Recommendations", rec_table, section_height)],
                [section_block("", help_table, section_height, title_flowable=help_header, center=False)],
            ],
            colWidths=[doc.width],
            rowHeights=[section_height * cm] * 4,
        )
        sections.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ]))
        story += [sections, Spacer(1, 0.2 * cm)]

        story += [
            Spacer(1, 0.2 * cm),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e8e4de")),
            Spacer(1, 0.1 * cm),
            P("This report is for informational purposes only and does not constitute "
              "medical advice. If you are experiencing distress, please consult a "
              "qualified mental health professional.", 8, color=BASE_COLOR),
        ]

        doc.build(story)
        buf.seek(0)
        fname = f"depression_indications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
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
