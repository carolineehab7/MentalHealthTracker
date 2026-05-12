import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const AXES = [
  { key: "academic",     label: "Academic" },
  { key: "financial",    label: "Financial" },
  { key: "sleep",        label: "Sleep" },
  { key: "workload",     label: "Workload" },
  { key: "satisfaction", label: "Satisfaction" },
];

export default function RadarChartPanel({ scores }) {
  if (!scores) return null;

  const data = AXES.map(({ key, label }) => ({
    subject: label,
    value: Math.round(scores[key] ?? 0),
    fullMark: 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart cx="50%" cy="50%" outerRadius="72%" data={data}>
        <PolarGrid stroke="rgba(0,0,0,0.08)" />
        <PolarAngleAxis
          dataKey="subject"
          tick={{
            fontSize: 12,
            fill: "#475569",
            fontFamily: "Inter, sans-serif",
            fontWeight: 500,
          }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tickCount={4}
          tick={{ fontSize: 10, fill: "#94a3b8" }}
        />
        <Radar
          dataKey="value"
          stroke="#2563eb"
          fill="#2563eb"
          fillOpacity={0.12}
          strokeWidth={2.5}
          dot={{ r: 4, fill: "#2563eb", strokeWidth: 0 }}
        />
        <Tooltip
          contentStyle={{
            background: "#ffffff",
            border: "1px solid rgba(99,123,192,0.18)",
            borderRadius: 10,
            fontSize: 13,
            fontFamily: "Inter, sans-serif",
            boxShadow: "0 4px 20px rgba(15,23,42,0.1)",
          }}
          formatter={(v) => [`${v} / 100`, "Score"]}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
