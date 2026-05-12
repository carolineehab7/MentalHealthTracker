import { useState } from 'react'
import {
  User, BookOpen, Briefcase, DollarSign,
  Moon, Utensils, Brain, Users,
  Activity, GraduationCap, Heart,
} from 'lucide-react'
import {
  Card, SLabel, Pill,
  SliderField, ToggleField, OptionField,
  BtnPrimary, SpinnerCard, ErrorBox,
} from '../components/UI'
import ResultPanel   from '../components/ResultPanel'
import { usePredict } from '../hooks/usePredict'
import styles from './AssessPage.module.css'

/* ── Static option lists ── */
const SLEEP_OPTIONS   = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
const DIET_OPTIONS    = ['Healthy', 'Moderate', 'Unhealthy']
const GENDER_OPTIONS  = ['Male', 'Female']

/* ── Default state ── */
const DEFAULTS = {
  age:                   22,
  academic_pressure:     2,
  work_pressure:         1,
  cgpa:                  7.0,
  study_satisfaction:    2,
  job_satisfaction:      2,
  work_study_hours:      6,
  financial_stress:      2,
  gender:                'Male',
  sleep_duration:        '7-8 hours',
  dietary_habits:        'Moderate',
  family_history:        false,
  suicidal_thoughts:     false,
}

export default function AssessPage() {
  const [inputs, setInputs] = useState(DEFAULTS)
  const { predict, exportPDF, result, loading, error } = usePredict()

  function set(key, val) {
    setInputs(prev => ({ ...prev, [key]: val }))
  }

  async function handleAnalyze() {
    const payload = {
      ...inputs,
      suicidal_thoughts: inputs.suicidal_thoughts ? 'Yes' : 'No',
      family_history:    inputs.family_history    ? 'Yes' : 'No',
    }
    await predict(payload)
    setTimeout(() => {
      document.getElementById('result-anchor')
        ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 150)
  }

  return (
    <div className={styles.page}>

      {/* ── Hero ── */}
      <div className={styles.hero}>
        <Pill>Depression Screening Tool</Pill>
        <h1 className={styles.h1}>How are you doing, really?</h1>
        <p className={styles.sub}>
          Fill in your academic and lifestyle details. Our classifier — trained on{' '}
          <strong>27,901 student records</strong> across{' '}
          <strong>13 clinical features</strong> — will assess your depression risk
          and offer personalised recommendations.
        </p>
      </div>

      {/* ── Academic & Professional Metrics ── */}
      <Card>
        <SLabel icon={GraduationCap}>Academic &amp; Professional Metrics</SLabel>
        <div className={styles.grid2}>
          <SliderField
            label="Age" id="age"
            min={18} max={59} step={1} value={inputs.age}
            onChange={v => set('age', v)}
            leftLabel="18" rightLabel="59" suffix=" yrs"
          />
          <SliderField
            label="Academic Pressure" id="acad"
            min={0} max={5} step={0.5} value={inputs.academic_pressure}
            onChange={v => set('academic_pressure', v)}
            leftLabel="0 (none)" rightLabel="5 (extreme)" suffix=" / 5"
          />
          <SliderField
            label="Work Pressure" id="work"
            min={0} max={5} step={0.5} value={inputs.work_pressure}
            onChange={v => set('work_pressure', v)}
            leftLabel="0 (none)" rightLabel="5 (extreme)" suffix=" / 5"
          />
          <SliderField
            label="CGPA" id="cgpa"
            min={0} max={10} step={0.1} value={inputs.cgpa}
            onChange={v => set('cgpa', v)}
            leftLabel="0.0" rightLabel="10.0" suffix=""
          />
          <SliderField
            label="Study Satisfaction" id="studysat"
            min={0} max={5} step={0.5} value={inputs.study_satisfaction}
            onChange={v => set('study_satisfaction', v)}
            leftLabel="0 (very low)" rightLabel="5 (very high)" suffix=" / 5"
          />
          <SliderField
            label="Job Satisfaction" id="jobsat"
            min={0} max={4} step={0.5} value={inputs.job_satisfaction}
            onChange={v => set('job_satisfaction', v)}
            leftLabel="0 (very low)" rightLabel="4 (very high)" suffix=" / 4"
          />
          <SliderField
            label="Work / Study Hours per day" id="hours"
            min={0} max={12} step={0.5} value={inputs.work_study_hours}
            onChange={v => set('work_study_hours', v)}
            leftLabel="0 h" rightLabel="12 h" suffix=" h"
          />
          <SliderField
            label="Financial Stress" id="fin"
            min={1} max={5} step={0.5} value={inputs.financial_stress}
            onChange={v => set('financial_stress', v)}
            leftLabel="1 (none)" rightLabel="5 (severe)" suffix=" / 5"
          />
        </div>
      </Card>

      {/* ── Lifestyle ── */}
      <Card>
        <SLabel icon={Moon}>Lifestyle</SLabel>
        <div className={styles.optionGroup}>
          <OptionField
            label="Sleep Duration"
            icon={Moon}
            options={SLEEP_OPTIONS}
            value={inputs.sleep_duration}
            onChange={v => set('sleep_duration', v)}
          />
          <OptionField
            label="Dietary Habits"
            icon={Utensils}
            options={DIET_OPTIONS}
            value={inputs.dietary_habits}
            onChange={v => set('dietary_habits', v)}
          />
          <OptionField
            label="Gender"
            icon={User}
            options={GENDER_OPTIONS}
            value={inputs.gender}
            onChange={v => set('gender', v)}
          />
        </div>
      </Card>

      {/* ── Mental Health Background ── */}
      <Card>
        <SLabel icon={Brain}>Mental Health Background</SLabel>
        <ToggleField
          icon={Users}
          label="Family History of Mental Illness"
          sub="Has a close family member been diagnosed with a mental illness?"
          checked={inputs.family_history}
          onChange={v => set('family_history', v)}
        />
        <ToggleField
          icon={Heart}
          label="History of Suicidal Thoughts"
          sub="Have you ever experienced thoughts of suicide or self-harm?"
          checked={inputs.suicidal_thoughts}
          onChange={v => set('suicidal_thoughts', v)}
        />
      </Card>

      {/* ── Submit ── */}
      <BtnPrimary onClick={handleAnalyze} loading={loading}>
        {loading ? 'Analyzing…' : 'Assess My Depression Risk →'}
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
