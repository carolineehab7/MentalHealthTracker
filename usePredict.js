import { useState } from 'react'

// Vite proxies /api/* → http://localhost:5000/*
const BASE = '/api'

/* ─────────────────────────────────────────────
   usePredict  — main hook used by AssessPage
───────────────────────────────────────────── */
export function usePredict() {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function predict(inputs) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res  = await fetch(`${BASE}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(inputs),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Server error')
      setResult(data)
      saveHistory(data)     // auto-save to localStorage
      return data
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function exportPDF(data) {
    try {
      const res = await fetch(`${BASE}/export-pdf`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(data),
      })
      if (!res.ok) {
        const e = await res.json()
        throw new Error(e.error)
      }
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `stress_report_${Date.now()}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert('PDF export failed: ' + e.message)
    }
  }

  return { predict, exportPDF, result, loading, error }
}

/* ─────────────────────────────────────────────
   History helpers  — used by HistoryPage
───────────────────────────────────────────── */
export function saveHistory(data) {
  const history = getHistory()
  history.unshift({
    date:        new Date().toISOString(),
    stressLevel: data.stressLevel,
    confidence:  data.confidence,
  })
  localStorage.setItem('mc_history', JSON.stringify(history.slice(0, 30)))
}

export function getHistory() {
  try { return JSON.parse(localStorage.getItem('mc_history') || '[]') }
  catch { return [] }
}

export function clearHistory() {
  localStorage.removeItem('mc_history')
}
