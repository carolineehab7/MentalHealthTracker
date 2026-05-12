import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Trash2, BarChart3 } from "lucide-react";
import { Card, SLabel, Pill, StressBadge, BtnOutline } from "../components/UI";
import { getHistory, clearHistory } from "../hooks/usePredict";
import styles from "./HistoryPage.module.css";

/* ── Helpers ── */
const LEVEL_NUM = { "No Depression": 1, "Depression": 2 };
const NUM_LEVEL = { 1: "No Depression", 2: "Depression" };
const DOT_COLOR = { "No Depression": "#059669", "Depression": "#dc2626" };
const LVL_COLOR = {
  "No Depression": "var(--ok)",
  "Depression": "var(--danger)",
};

/* ── Custom tooltip for chart ── */
function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className={styles.tooltip}>
      <div className={styles.ttDate}>{d.dateLabel}</div>
      <div className={styles.ttLevel} style={{ color: DOT_COLOR[d.level] }}>
        {d.level}
      </div>
      <div className={styles.ttConf}>{d.confidence}% confidence</div>
    </div>
  );
}

/* ── Coloured dots on chart ── */
function ColorDot({ cx, cy, payload }) {
  return (
    <circle
      cx={cx}
      cy={cy}
      r={6}
      fill={DOT_COLOR[payload.level] || "#888"}
      stroke="#fff"
      strokeWidth={2}
    />
  );
}

/* ── Main page ── */
export default function HistoryPage() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    setHistory(getHistory());
  }, []);

  function handleClear() {
    if (window.confirm("Clear all saved history?")) {
      clearHistory();
      setHistory([]);
    }
  }

  /* ── Empty state ── */
  if (!history.length) {
    return (
      <div className={styles.page}>
        <div className={styles.hero}>
          <Pill blue>Tracking</Pill>
          <h1 className={styles.h1}>Your depression history</h1>
        </div>
        <Card>
          <div className={styles.empty}>
            <div className={styles.emptyIcon}>
              <BarChart3 size={28} />
            </div>
            <p className={styles.emptyTitle}>No assessments yet</p>
            <p className={styles.emptySub}>
              Complete your first assessment on the Assessment tab and your
              results will appear here automatically.
            </p>
          </div>
        </Card>
      </div>
    );
  }

  /* ── Build chart data — last 10 entries in chronological order ── */
  const chartData = [...history]
    .reverse()
    .slice(-10)
    .map((h) => {
      const d = new Date(h.date);
      return {
        dateLabel:
          d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }) +
          " " +
          d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }),
        shortDate: d.toLocaleDateString("en-GB", {
          day: "numeric",
          month: "short",
        }),
        value: LEVEL_NUM[h.stressLevel] || 1,
        level: h.stressLevel,
        confidence: h.confidence,
      };
    });

  /* ── Summary counts ── */
  const counts = { "No Depression": 0, "Depression": 0 };
  history.forEach((h) => {
    counts[h.stressLevel] = (counts[h.stressLevel] || 0) + 1;
  });
  const latest = history[0];

  return (
    <div className={styles.page}>
      {/* ── Hero ── */}
      <div className={styles.hero}>
        <Pill blue>Tracking</Pill>
        <h1 className={styles.h1}>Your depression history</h1>
        <p className={styles.sub}>
          Every completed assessment is saved automatically so you can monitor
          your wellbeing over time.
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
          <div className={styles.summaryVal} style={{ color: "var(--danger)" }}>
            {counts["Depression"]}
          </div>
          <div className={styles.summaryLbl}>Depressed results</div>
        </Card>
      </div>

      {/* ── Trend chart ── */}
      <Card>
        <SLabel>Depression trend — last {chartData.length} assessments</SLabel>
        <div className={styles.chartWrap}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 10, bottom: 0, left: -20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
              <XAxis
                dataKey="shortDate"
                tick={{ fontSize: 11, fill: "#94a3b8", fontFamily: "Inter" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                domain={[0.8, 2.2]}
                ticks={[1, 2]}
                tickFormatter={(v) => NUM_LEVEL[v] || ""}
                tick={{ fontSize: 11, fill: "#94a3b8", fontFamily: "Inter" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<ChartTooltip />} />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#2563eb"
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
          const d = new Date(h.date);
          const dateStr = d.toLocaleDateString("en-GB", {
            day: "numeric",
            month: "short",
            year: "numeric",
          });
          const timeStr = d.toLocaleTimeString("en-GB", {
            hour: "2-digit",
            minute: "2-digit",
          });
          return (
            <div key={i} className={styles.histItem}>
              <StressBadge level={h.stressLevel} />
              <div className={styles.histMeta}>
                <div className={styles.histDate}>
                  {dateStr} at {timeStr}
                </div>
                <div className={styles.histConf}>
                  {h.confidence}% confidence
                </div>
              </div>
            </div>
          );
        })}
      </Card>

      {/* ── Clear button ── */}
      <BtnOutline onClick={handleClear} icon={<Trash2 size={15} />}>
        Clear all history
      </BtnOutline>
    </div>
  );
}
