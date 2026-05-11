import {
  RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts'

const AXES = [
  { key: 'sleep',    label: 'Sleep'    },
  { key: 'anxiety',  label: 'Anxiety'  },
  { key: 'social',   label: 'Social'   },
  { key: 'workload', label: 'Workload' },
  { key: 'mood',     label: 'Mood'     },
]

export default function RadarChartPanel({ scores }) {
  if (!scores) return null

  const data = AXES.map(({ key, label }) => ({
    subject:  label,
    value:    Math.round(scores[key] ?? 0),
    fullMark: 100,
  }))

  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart cx="50%" cy="50%" outerRadius="72%" data={data}>
        <PolarGrid stroke="rgba(0,0,0,0.08)" />
        <PolarAngleAxis
          dataKey="subject"
          tick={{ fontSize: 12, fill: '#6b6560', fontFamily: 'DM Sans, sans-serif' }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tickCount={4}
          tick={{ fontSize: 10, fill: '#9c968f' }}
        />
        <Radar
          dataKey="value"
          stroke="#1d4ed8"
          fill="#1d4ed8"
          fillOpacity={0.13}
          strokeWidth={2}
          dot={{ r: 4, fill: '#1d4ed8', strokeWidth: 0 }}
        />
        <Tooltip
          contentStyle={{
            background: '#faf9f7',
            border: '1px solid rgba(28,25,22,0.1)',
            borderRadius: 8,
            fontSize: 13,
            fontFamily: 'DM Sans, sans-serif',
          }}
          formatter={(v) => [`${v} / 100`, 'Score']}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}
