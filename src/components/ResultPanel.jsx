import { FileDown, Sparkles, ShieldCheck, AlertTriangle } from "lucide-react";
import {
  Card,
  SLabel,
  StressBadge,
  StatBox,
  ProbBar,
  ImpBar,
  SugItem,
  BtnDanger,
  BtnOutline,
  Divider,
} from "./UI";
import RadarChartPanel from "./RadarChartPanel";
import styles from "./ResultPanel.module.css";

const PROB_COLORS = {
  "No Depression": "#22c55e",
  "Depression": "#ef4444",
  "Low": "#22c55e",
  "Moderate": "#f59e0b",
  "High": "#ef4444",
};

const LEVEL_COLORS = {
  "No Depression": "var(--ok)",
  "Depression": "var(--danger)",
  "Low": "var(--ok)",
  "Moderate": "var(--warn)",
  "High": "var(--danger)",
};

const RISK_COLORS = {
  Low: { bg: "#f0fdf4", border: "#86efac", text: "#15803d" },
  Moderate: { bg: "#fffbeb", border: "#fcd34d", text: "#b45309" },
  High: { bg: "#fef2f2", border: "#fca5a5", text: "#b91c1c" },
};

function DepressionAlert() {
  return (
    <div className={styles.alertDep}>
      <div className={styles.alertIcon}>
        <AlertTriangle size={22} />
      </div>
      <div className={styles.alertText}>
        <strong>Depression indicators detected.</strong> If you have been feeling
        this way for two weeks or more, please speak with a mental health
        professional.{" "}
        <a
          href="https://www.who.int/news-room/fact-sheets/detail/depression"
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--danger)" }}
        >
          WHO Depression Resources →
        </a>
      </div>
    </div>
  );
}

function DiseaseRiskCard({ condition, risk, icon, indicator }) {
  const c = RISK_COLORS[risk] || RISK_COLORS.Low;
  return (
    <div style={{
      display: "flex", gap: "0.75rem", alignItems: "flex-start",
      padding: "0.75rem 1rem", borderRadius: "0.5rem", marginBottom: "0.5rem",
      background: c.bg, border: `1px solid ${c.border}`,
    }}>
      <span style={{ fontSize: "1.25rem", flexShrink: 0 }}>{icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>{condition}</span>
          <span style={{
            fontSize: "0.7rem", fontWeight: 700, padding: "0.1rem 0.5rem",
            borderRadius: "999px", background: c.border, color: c.text,
            textTransform: "uppercase", letterSpacing: "0.05em",
          }}>{risk} Risk</span>
        </div>
        <p style={{ margin: "0.25rem 0 0", fontSize: "0.8rem", color: "#6b7280", lineHeight: 1.5 }}>
          {indicator}
        </p>
      </div>
    </div>
  );
}

export default function ResultPanel({ data, onExportPDF, onClear }) {
  if (!data) return null;

  const {
    stressLevel,
    confidence,
    probabilities,
    radarScores,
    featureImportance,
    suggestions,
    diseaseRisk,
  } = data;

  const isDepression = stressLevel === "Depression";
  const probs = probabilities || {};
  const imp = featureImportance || [];
  const sugs = suggestions || [];
  const maxImp = imp.length ? imp[0].pct : 1;
  const probEntries = Object.entries(probs);

  return (
    <div className={styles.wrap}>
      {isDepression && <DepressionAlert />}

      <Card>
        {/* ── Header ── */}
        <div className={styles.header}>
          <div>
            <div className={styles.title}>Assessment Result</div>
            <div className={styles.sub}>
              Based on your academic and lifestyle profile across 13 clinical features.
            </div>
          </div>
          <StressBadge level={stressLevel} />
        </div>

        {/* ── Stats ── */}
        <div className={styles.statsRow}>
          <StatBox
            value={stressLevel}
            label="Prediction"
            color={LEVEL_COLORS[stressLevel]}
          />
          <StatBox value={`${confidence}%`} label="Model confidence" />
          <StatBox
            value={`${probs["No Depression"] ?? probs["Low"] ?? 0}%`}
            label="Chance of no depression"
          />
        </div>

        <Divider />

        {/* ── Probability bars ── */}
        <SLabel>Probability distribution</SLabel>
        {probEntries.map(([label, pct]) => (
          <ProbBar
            key={label}
            label={label}
            pct={pct}
            color={PROB_COLORS[label] ?? "#6b7280"}
          />
        ))}

        <Divider />

        {/* ── Radar chart ── */}
        <SLabel>Health profile radar</SLabel>
        <RadarChartPanel scores={radarScores} />

        <Divider />

        {/* ── Feature importance ── */}
        <SLabel>Top contributing factors</SLabel>
        <p className={styles.impDesc}>
          These are the features the model weighted most heavily. A higher
          percentage means that factor had a greater influence on the
          classification.
        </p>
        {imp.slice(0, 5).map((f, i) => (
          <ImpBar key={i} icon={f.icon} label={f.label} pct={f.pct} maxPct={maxImp} />
        ))}

        <Divider />

        {/* ── Suggestions ── */}
        <SLabel>Personalised recommendations</SLabel>
        {sugs.map((s, i) => (
          <SugItem key={i} icon={s.icon} category={s.category} text={s.text} />
        ))}

        {/* ── Disease Risk ── */}
        {diseaseRisk && diseaseRisk.length > 0 && (
          <>
            <Divider />
            <SLabel>Potential risk assessment</SLabel>
            <p className={styles.impDesc}>
              Based on your reported metrics, the following conditions may
              warrant attention. This is not a diagnosis — consult a doctor
              for any concerns.
            </p>
            {diseaseRisk.map((d, i) => (
              <DiseaseRiskCard
                key={i}
                condition={d.condition}
                risk={d.risk}
                icon={d.icon}
                indicator={d.indicator}
              />
            ))}
          </>
        )}

        {/* ── Actions ── */}
        <div className={styles.btnRow}>
          <BtnDanger onClick={onExportPDF}>
            <FileDown size={16} /> Download PDF Report
          </BtnDanger>
          <BtnOutline onClick={onClear}>Clear result</BtnOutline>
        </div>

        <div className={styles.disclaimer}>
          <ShieldCheck
            size={15}
            style={{ flexShrink: 0, marginTop: 1, color: "var(--accent)" }}
          />
          This tool is for informational purposes only and does not constitute
          medical advice. If you are experiencing distress, please consult a
          qualified mental health professional.
        </div>
      </Card>
    </div>
  );
}
