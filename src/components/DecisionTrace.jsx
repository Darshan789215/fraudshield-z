import { Check, CircleAlert, ShieldAlert } from 'lucide-react';

export default function DecisionTrace({ items = [] }) {
  return (
    <div className="trace-list">
      {items.map((item, index) => {
        const Icon = item.status === 'alert' || item.status === 'blocked' ? (item.status === 'blocked' ? ShieldAlert : CircleAlert) : Check;
        return <div className={`trace-item trace-${item.status}`} key={item.key}>
          <div className="trace-icon"><Icon size={13} /></div>
          <div className="trace-copy"><span>{String(index + 1).padStart(2, '0')}</span><strong>{item.label}</strong>{item.value && <em>{item.value}</em>}</div>
          {index < items.length - 1 && <div className="trace-line" />}
        </div>;
      })}
    </div>
  );
}
