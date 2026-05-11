import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'
import { Trash2 } from 'lucide-react'
import { Card, SLabel, Pill, StressBadge, BtnOutline } from '../components/UI'
import { getHistory, clearHistory } from '../hooks/usePredict'
import styles from './HistoryPage.module.css'

/* ── Helpers ── */
const LEVEL_NUM  = { Low: 1, Moderate: 2, High: 3 }
const NUM_LEVEL  = { 1: 'Low', 2: 'Moderate', 3: 'High' }
const DOT_COLOR  = { Low: '#22c55e', Moderate: '#f59e0b', High: '#ef4444' }
const LVL_COLOR  = { Low: 'var(--ok)', Moderate: 'var(--warn)', High: 'var(--danger)' }

/* ── Custom tooltip for chart ── */
function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className={styles.tooltip}>
      <div className={styles.ttDate}>{d.dateLabel}</div>
      <div className={styles.ttLevel} style={{ color: DOT_COLOR[d.level] }}>
        {d.level} Stress
      </div>
      <div className={styles.ttConf}>{d.confidence}% confidence</div>
    </div>
  )
}

/* ── Coloured dots on chart ── */
function ColorDot({ cx, cy, payload }) {
  return (
    <circle
      cx={cx} cy={cy} r={6}
      fill={DOT_COLOR[payload.level] || '#888'}
      stroke="#fff" strokeWidth={2}
    />
  )
}

/* ── Main page ── */
export default function HistoryPage() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    setHistory(getHistory())
  }, [])

  function handleClear() {
    if (window.confirm('Clear all saved history?')) {
      clearHistory()
      setHistory([])
    }
  }

  /* ── Empty state ── */
  if (!history.length) {
    return (
      <div className={styles.page}>
        <div className={styles.hero}>
          <Pill blue>Tracking</Pill>
          <h1 className={styles.h1}>Your stress history</h1>
        </div>
        <Card>
          <div className={styles.empty}>
            <div className={styles.emptyIcon}>📊</div>
            <p className={styles.emptyTitle}>No assessments yet</p>
            <p className={styles.emptySub}>
              Complete your first assessment on the Assessment tab and your
              results will appear here automatically.
            </p>
          </div>
        </Card>
      </div>
    )
  }

  /* ── Build chart data — last 10 entries in chronological order ── */
  const chartData = [...history]
    .reverse()
    .slice(-10)
    .map(h => {
      const d = new Date(h.date)
      return {
        dateLabel: d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
                 + ' ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
        shortDate: d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }),
        value:     LEVEL_NUM[h.stressLevel],
        level:     h.stressLevel,
        confidence: h.confidence,
      }
    })

  /* ── Summary counts ── */
  const counts = { Low: 0, Moderate: 0, High: 0 }
  history.forEach(h => { counts[h.stressLevel] = (counts[h.stressLevel] || 0) + 1 })
  const latest = history[0]

  return (
    <div className={styles.page}>

      {/* ── Hero ── */}
      <div className={styles.hero}>
        <Pill blue>Tracking</Pill>
        <h1 className={styles.h1}>Your stress history</h1>
        <p className={styles.sub}>
          Every completed assessment is saved automatically so you can
          monitor your wellbeing over time.
        </p>
      </div>

      {/* ── Summary cards ── */}
      <div className={styles.summaryRow}>
        <Card className={styles.summaryCard}>
          <div className={styles.summaryVal}>{history.length}</div>
          <div className={styles.summaryLbl}>Total assessments</div>
        </Card>
        <Card className={styles.summaryCard}>
          <div
            className={styles.summaryVal}
            style={{ color: LVL_COLOR[latest.stressLevel] }}
          >
            {latest.stressLevel}
          </div>
          <div className={styles.summaryLbl}>Latest result</div>
        </Card>
        <Card className={styles.summaryCard}>
          <div className={styles.summaryVal} style={{ color: 'var(--danger)' }}>
            {counts.High}
          </div>
          <div className={styles.summaryLbl}>High stress sessions</div>
        </Card>
      </div>

      {/* ── Trend chart ── */}
      <Card>
        <SLabel>Stress trend — last {chartData.length} assessments</SLabel>
        <div className={styles.chartWrap}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 10, bottom: 0, left: -20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
              <XAxis
                dataKey="shortDate"
                tick={{ fontSize: 11, fill: '#9c968f', fontFamily: 'DM Sans' }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                domain={[0.5, 3.5]}
                ticks={[1, 2, 3]}
                tickFormatter={v => NUM_LEVEL[v] || ''}
                tick={{ fontSize: 11, fill: '#9c968f', fontFamily: 'DM Sans' }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<ChartTooltip />} />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#1d4ed8"
                strokeWidth={2.5}
                dot={<ColorDot />}
                activeDot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* ── History list ── */}
      <Card>
        <SLabel>All assessments ({history.length})</SLabel>
        {history.map((h, i) => {
          const d       = new Date(h.date)
          const dateStr = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
          const timeStr = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
          return (
            <div key={i} className={styles.histItem}>
              <StressBadge level={h.stressLevel} />
              <div className={styles.histMeta}>
                <div className={styles.histDate}>{dateStr} at {timeStr}</div>
                <div className={styles.histConf}>{h.confidence}% confidence</div>
              </div>
            </div>
          )
        })}
      </Card>

      {/* ── Clear button ── */}
      <BtnOutline onClick={handleClear} icon={<Trash2 size={15} />}>
        Clear all history
      </BtnOutline>

    </div>
  )
}
