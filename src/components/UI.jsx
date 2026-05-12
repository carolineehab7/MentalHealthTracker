import { AlertTriangle } from "lucide-react";
import styles from "./UI.module.css";

/* ── Card ── */
export function Card({ children, className = "" }) {
  return <div className={`${styles.card} ${className}`}>{children}</div>;
}

/* ── Section label ── */
export function SLabel({ children, icon: Icon }) {
  return (
    <p className={styles.slabel}>
      {Icon && <Icon size={13} strokeWidth={2.5} />}
      {children}
    </p>
  );
}

/* ── Pill badge ── */
export function Pill({ children, blue = false }) {
  return (
    <span className={`${styles.pill} ${blue ? styles.pillBlue : ""}`}>
      {children}
    </span>
  );
}

/* ── Prediction badge (handles both stress levels and depression labels) ── */
export function StressBadge({ level }) {
  const isOk     = level === "Low"  || level === "No Depression";
  const isWarn   = level === "Moderate";
  const isDanger = level === "High" || level === "Depression";
  const cls = isOk ? styles.sbOk : isWarn ? styles.sbWarn : styles.sbDanger;
  return <span className={`${styles.stressBadge} ${cls}`}>{level}</span>;
}

/* ── Option chip group (radio-like selector) ── */
export function OptionField({ label, options, value, onChange, icon: Icon }) {
  return (
    <div className={styles.field}>
      <div className={styles.fieldTop}>
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {Icon && <Icon size={13} strokeWidth={2.5} style={{ color: "var(--accent)", flexShrink: 0 }} />}
          {label}
        </label>
      </div>
      <div className={styles.optionChips}>
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            className={`${styles.optionChip} ${value === opt ? styles.optionChipActive : ""}`}
            onClick={() => onChange(opt)}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ── Slider field ── */
export function SliderField({
  label,
  id,
  min,
  max,
  step,
  value,
  onChange,
  leftLabel,
  rightLabel,
  suffix,
}) {
  const display = Number.isInteger(value)
    ? value
    : parseFloat(value).toFixed(1);
  return (
    <div className={styles.field}>
      <div className={styles.fieldTop}>
        <label htmlFor={id}>{label}</label>
        <span className={styles.val}>
          {display}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) =>
          onChange(
            step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value),
          )
        }
      />
      <div className={styles.rangeEnds}>
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  );
}

/* ── Toggle switch ── */
export function ToggleField({ icon: Icon, label, sub, checked, onChange }) {
  return (
    <div className={styles.toggleRow}>
      <div className={styles.toggleInfo}>
        {Icon && (
          <div className={styles.toggleIconWrap}>
            <Icon size={16} strokeWidth={2} />
          </div>
        )}
        <div>
          <div className={styles.toggleLabel}>{label}</div>
          {sub && <div className={styles.toggleSub}>{sub}</div>}
        </div>
      </div>
      <label className={styles.toggle}>
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className={styles.slider} />
      </label>
    </div>
  );
}

/* ── Mood button ── */
export function MoodButton({ icon: Icon, label, active, onClick }) {
  return (
    <button
      type="button"
      className={`${styles.moodBtn} ${active ? styles.moodActive : ""}`}
      onClick={onClick}
    >
      <Icon size={24} strokeWidth={1.75} className={styles.moodIcon} />
      <span className={styles.moodLabel}>{label}</span>
    </button>
  );
}

/* ── Primary button ── */
export function BtnPrimary({ children, onClick, disabled, loading }) {
  return (
    <button
      type="button"
      className={styles.btnPrimary}
      onClick={onClick}
      disabled={disabled || loading}
    >
      {loading && <span className={styles.btnSpinner} />}
      {children}
    </button>
  );
}

/* ── Outline button ── */
export function BtnOutline({ children, onClick, icon }) {
  return (
    <button type="button" className={styles.btnOutline} onClick={onClick}>
      {icon && <span style={{ display: "flex" }}>{icon}</span>}
      {children}
    </button>
  );
}

/* ── Danger button ── */
export function BtnDanger({ children, onClick }) {
  return (
    <button type="button" className={styles.btnDanger} onClick={onClick}>
      {children}
    </button>
  );
}

/* ── Stat box ── */
export function StatBox({ value, label, color }) {
  return (
    <div className={styles.statBox}>
      <div className={styles.statVal} style={color ? { color } : {}}>
        {value}
      </div>
      <div className={styles.statLbl}>{label}</div>
    </div>
  );
}

/* ── Probability bar ── */
export function ProbBar({ label, pct, color }) {
  return (
    <div className={styles.probRow}>
      <div className={styles.probLabels}>
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className={styles.probTrack}>
        <div
          className={styles.probFill}
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

/* ── Importance bar ── */
export function ImpBar({ label, pct, maxPct }) {
  const w = maxPct > 0 ? Math.round((pct / maxPct) * 100) : 0;
  return (
    <div className={styles.impRow}>
      <div className={styles.impLabel}>{label}</div>
      <div className={styles.impTrack}>
        <div className={styles.impFill} style={{ width: `${w}%` }} />
      </div>
      <div className={styles.impPct}>{pct}%</div>
    </div>
  );
}

/* ── Suggestion item ── */
export function SugItem({ icon, category, text }) {
  return (
    <div className={styles.sugItem}>
      <div className={styles.sugIcon}>{icon}</div>
      <div>
        <div className={styles.sugCat}>{category}</div>
        <div className={styles.sugText}>{text}</div>
      </div>
    </div>
  );
}

/* ── Loading spinner card ── */
export function SpinnerCard() {
  return (
    <Card>
      <div className={styles.spinnerWrap}>
        <div className={styles.spinner} />
        <span className={styles.spinnerText}>Running model prediction…</span>
      </div>
    </Card>
  );
}

/* ── Error box ── */
export function ErrorBox({ message }) {
  return (
    <div className={styles.errorBox}>
      <div className={styles.errorHeader}>
        <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
        <span>{message}</span>
      </div>
      <small>
        Make sure the Flask server is running:{" "}
        <code>cd backend &amp;&amp; python app.py</code>
      </small>
    </div>
  );
}

/* ── High stress alert ── */
export function AlertHigh() {
  return (
    <div className={styles.alertHigh}>
      <div className={styles.alertIcon}>
        <AlertTriangle size={22} />
      </div>
      <div className={styles.alertText}>
        <strong>High stress detected.</strong> If you have been feeling this way
        for several days, please consider speaking with a mental health
        professional or a trusted person.{" "}
        <a
          href="https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response"
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--danger)" }}
        >
          WHO Mental Health Resources →
        </a>
      </div>
    </div>
  );
}

/* ── Horizontal divider ── */
export function Divider() {
  return <hr className={styles.divider} />;
}
