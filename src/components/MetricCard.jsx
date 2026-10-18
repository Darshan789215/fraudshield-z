export default function MetricCard({ label, value, detail, tone = 'neutral' }) {
  return <div className={`metric-card metric-${tone}`}><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>;
}
