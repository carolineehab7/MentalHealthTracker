import {
  Card, SLabel, StressBadge, StatBox,
  ProbBar, ImpBar, SugItem,
  BtnDanger, BtnOutline,
  AlertHigh, Divider,
} from './UI'
import RadarChartPanel from './RadarChartPanel'
import styles from './ResultPanel.module.css'

const PROB_COLORS = {
  Low:      '#22c55e',
  Moderate: '#f59e0b',
  High:     '#ef4444',
}

const LEVEL_COLORS = {
  Low:      'var(--ok)',
  Moderate: 'var(--warn)',
  High:     'var(--danger)',
}

export default function ResultPanel({ data, onExportPDF, onClear }) {
  if (!data) return null

  const {
    stressLevel, confidence, probabilities,
    radarScores, featureImportance, suggestions,
  } = data

  const probs  = probabilities     || {}
  const imp    = featureImportance || []
  const sugs   = suggestions       || []
  const maxImp = imp.length ? imp[0].pct : 1

  return (
    <div className={styles.wrap}>
      {stressLevel === 'High' && <AlertHigh />}

      <Card>
        {/* ── Header ── */}
        <div className={styles.header}>
          <div>
            <div className={styles.title}>Assessment Result</div>
            <div className={styles.sub}>
              Based on your reported health metrics across 20 clinical factors.
            </div>
          </div>
          <StressBadge level={stressLevel} />
        </div>

        {/* ── Stats ── */}
        <div className={styles.statsRow}>
          <StatBox
            value={stressLevel}
            label="Stress level"
            color={LEVEL_COLORS[stressLevel]}
          />
          <StatBox value={`${confidence}%`} label="Model confidence" />
          <StatBox value={`${probs.Low ?? 0}%`} label="Chance of Low" />
        </div>

        <Divider />

        {/* ── Probability bars ── */}
        <SLabel>Probability distribution</SLabel>
        {['Low', 'Moderate', 'High'].map(l => (
          <ProbBar key={l} label={l} pct={probs[l] ?? 0} color={PROB_COLORS[l]} />
        ))}

        <Divider />

        {/* ── Radar chart ── */}
        <SLabel>Health profile radar</SLabel>
        <RadarChartPanel scores={radarScores} />

        <Divider />

        {/* ── Feature importance ── */}
        <SLabel>🔍 Why this result? — Top contributing factors</SLabel>
        <p className={styles.impDesc}>
          These are the features the model weighted most heavily.
          A higher percentage means that factor had a greater influence on the classification.
        </p>
        {imp.slice(0, 5).map(f => (
          <ImpBar key={f.feature} label={f.label} pct={f.pct} maxPct={maxImp} />
        ))}

        <Divider />

        {/* ── Suggestions ── */}
        <SLabel>Personalised coping suggestions</SLabel>
        {sugs.map((s, i) => (
          <SugItem key={i} icon={s.icon} category={s.category} text={s.text} />
        ))}

        {/* ── Actions ── */}
        <div className={styles.btnRow}>
          <BtnDanger onClick={onExportPDF}>📄 Download PDF Report</BtnDanger>
          <BtnOutline onClick={onClear}>Clear result</BtnOutline>
        </div>

        <div className={styles.disclaimer}>
          ⚕️ This tool is for informational purposes only and does not constitute medical advice.
          If you are experiencing severe distress, please consult a qualified healthcare professional.
        </div>
      </Card>
    </div>
  )
}
