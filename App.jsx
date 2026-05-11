import { Routes, Route, NavLink } from 'react-router-dom'
import { ClipboardList, Clock, Activity } from 'lucide-react'
import AssessPage  from './pages/AssessPage'
import HistoryPage from './pages/HistoryPage'
import styles from './App.module.css'

export default function App() {
  return (
    <div className={styles.shell}>

      {/* ── SIDEBAR ── */}
      <aside className={styles.sidebar}>
        <div className={styles.brand}>
          <span className={styles.brandIcon}>🧠</span>
          <div>
            <div className={styles.brandName}>MindCheck</div>
            <div className={styles.brandSub}>Stress Classifier</div>
          </div>
        </div>

        <nav className={styles.nav}>
          <NavLink to="/"        end className={navCls}>
            <ClipboardList size={17} /> Assessment
          </NavLink>
          <NavLink to="/history"    className={navCls}>
            <Clock size={17} /> History
          </NavLink>
        </nav>

        <div className={styles.footer}>
          <div className={styles.modelBadge}>
            <Activity size={12} />
            Random Forest · 20 features
          </div>
          <p className={styles.disclaimer}>
            For informational use only. Not a substitute for professional medical advice.
          </p>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main className={styles.main}>
        <Routes>
          <Route path="/"        element={<AssessPage  />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </main>

    </div>
  )
}

function navCls({ isActive }) {
  return [styles.navLink, isActive ? styles.navLinkActive : ''].join(' ')
}
