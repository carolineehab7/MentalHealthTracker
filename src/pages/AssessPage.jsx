import { useState } from 'react'
import {
  Frown, Zap, Meh, Smile, SmilePlus,
  Activity, Heart, Shield,
  Brain, HeartPulse, Wind, Users,
} from 'lucide-react'
import {
  Card, SLabel, Pill,
  SliderField, ToggleField, MoodButton,
  BtnPrimary, SpinnerCard, ErrorBox,
} from '../components/UI'
import ResultPanel   from '../components/ResultPanel'
import { usePredict } from '../hooks/usePredict'
import styles from './AssessPage.module.css'

/* ── Mood options ── */
const MOODS = [
  { icon: Frown,     label: 'Very sad'  },
  { icon: Zap,       label: 'Stressed'  },
  { icon: Meh,       label: 'Neutral'   },
  { icon: Smile,     label: 'Good'      },
  { icon: SmilePlus, label: 'Great'     },
]

/* ── Initial slider state ── */
const DEFAULTS = {
  sleep_hours:            7,
  anxiety_score:          4,
  social_activity:        6,
  work_hours:             8,
  depression:             3,
  future_career_concerns: 5,
  peer_pressure:          3,
  headache:               2,
  mental_health_history:  false,
  blood_pressure:         false,
  breathing_problem:      false,
  bullying:               false,
}

export default function AssessPage() {
  const [inputs, setInputs] = useState(DEFAULTS)
  const [mood,   setMood]   = useState('Neutral')
  const { predict, exportPDF, result, loading, error } = usePredict()

  function set(key, val) {
    setInputs(prev => ({ ...prev, [key]: val }))
  }

  async function handleAnalyze() {
    /* Build the payload the Flask /predict endpoint expects */
    const payload = {
      ...inputs,
      mood,
      mental_health_history: inputs.mental_health_history ? 1 : 0,
      blood_pressure:        inputs.blood_pressure        ? 1 : 0,
      breathing_problem:     inputs.breathing_problem     ? 1 : 0,
      bullying:              inputs.bullying               ? 1 : 0,
    }
    await predict(payload)
    /* Scroll smoothly to the result */
    setTimeout(() => {
      document.getElementById('result-anchor')
        ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 150)
  }

  return (
    <div className={styles.page}>

      {/* ── Hero ── */}
      <div className={styles.hero}>
        <Pill>Self-Assessment Tool</Pill>
        <h1 className={styles.h1}>How stressed are you today?</h1>
        <p className={styles.sub}>
          Fill in your health metrics below. Our Random Forest model — trained on 1,200
          real survey records across <strong>20 clinical features</strong> — will classify
          your stress level and explain what drove the prediction.
        </p>
      </div>

      {/* ── Core metrics ── */}
      <Card>
        <SLabel icon={Activity}>Core daily metrics</SLabel>
        <div className={styles.grid2}>
          <SliderField
            label="Sleep hours / night" id="sleep"
            min={2} max={12} step={0.5} value={inputs.sleep_hours}
            onChange={v => set('sleep_hours', v)}
            leftLabel="2 h" rightLabel="12 h" suffix=" h"
          />
          <SliderField
            label="Anxiety score" id="anxiety"
            min={1} max={10} step={1} value={inputs.anxiety_score}
            onChange={v => set('anxiety_score', v)}
            leftLabel="1 (none)" rightLabel="10 (severe)" suffix=" / 10"
          />
          <SliderField
            label="Social activity level" id="social"
            min={1} max={10} step={1} value={inputs.social_activity}
            onChange={v => set('social_activity', v)}
            leftLabel="1 (isolated)" rightLabel="10 (very active)" suffix=" / 10"
          />
          <SliderField
            label="Work / study hours / day" id="work"
            min={0} max={16} step={0.5} value={inputs.work_hours}
            onChange={v => set('work_hours', v)}
            leftLabel="0 h" rightLabel="16 h" suffix=" h"
          />
          <SliderField
            label="Depression score" id="depression"
            min={0} max={10} step={1} value={inputs.depression}
            onChange={v => set('depression', v)}
            leftLabel="0 (none)" rightLabel="10 (severe)" suffix=" / 10"
          />
          <SliderField
            label="Career concern level" id="career"
            min={0} max={10} step={1} value={inputs.future_career_concerns}
            onChange={v => set('future_career_concerns', v)}
            leftLabel="0 (none)" rightLabel="10 (very worried)" suffix=" / 10"
          />
          <SliderField
            label="Peer pressure level" id="peer"
            min={0} max={10} step={1} value={inputs.peer_pressure}
            onChange={v => set('peer_pressure', v)}
            leftLabel="0 (none)" rightLabel="10 (extreme)" suffix=" / 10"
          />
          <SliderField
            label="Headache frequency" id="headache"
            min={0} max={10} step={1} value={inputs.headache}
            onChange={v => set('headache', v)}
            leftLabel="0 (never)" rightLabel="10 (daily)" suffix=" / 10"
          />
        </div>
      </Card>

      {/* ── Mood ── */}
      <Card>
        <SLabel icon={Heart}>Current mood</SLabel>
        <div className={styles.moodGrid}>
          {MOODS.map(m => (
            <MoodButton
              key={m.label}
              icon={m.icon}
              label={m.label}
              active={mood === m.label}
              onClick={() => setMood(m.label)}
            />
          ))}
        </div>
      </Card>

      {/* ── Health conditions ── */}
      <Card>
        <SLabel icon={Shield}>Additional health conditions</SLabel>
        <ToggleField
          icon={Brain}
          label="Mental health history"
          sub="Have you had prior mental health issues?"
          checked={inputs.mental_health_history}
          onChange={v => set('mental_health_history', v)}
        />
        <ToggleField
          icon={HeartPulse}
          label="Blood pressure issues"
          sub="High or low blood pressure diagnosed"
          checked={inputs.blood_pressure}
          onChange={v => set('blood_pressure', v)}
        />
        <ToggleField
          icon={Wind}
          label="Breathing problems"
          sub="Asthma, shortness of breath, etc."
          checked={inputs.breathing_problem}
          onChange={v => set('breathing_problem', v)}
        />
        <ToggleField
          icon={Users}
          label="Experience of bullying"
          sub="Current or recent bullying situation"
          checked={inputs.bullying}
          onChange={v => set('bullying', v)}
        />
      </Card>

      {/* ── Submit ── */}
      <BtnPrimary onClick={handleAnalyze} loading={loading}>
        {loading ? 'Analyzing…' : 'Analyze My Stress Level →'}
      </BtnPrimary>

      {/* ── Result area ── */}
      <div id="result-anchor" style={{ scrollMarginTop: '1.5rem' }} />
      {loading && <SpinnerCard />}
      {error   && <ErrorBox message={error} />}
      {result  && !loading && (
        <ResultPanel
          data={result}
          onExportPDF={() => exportPDF(result)}
          onClear={() => window.location.reload()}
        />
      )}
    </div>
  )
}
