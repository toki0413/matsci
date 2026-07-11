/**
 * ContextBar — shows context window utilization percentage.
 *
 * Only visible when contextPct > 0 (after first compaction event).
 * Color shifts: green < 60, amber < 85, red >= 85.
 */
import { useTranslation } from 'react-i18next';

interface ContextBarProps {
  pct: number;
}

export function ContextBar({ pct }: ContextBarProps) {
  const { t } = useTranslation();
  if (pct <= 0) return null;

  const color = pct >= 85 ? '#ef4444' : pct >= 60 ? '#f59e0b' : '#22c55e';
  const label = pct >= 85 ? t('context.full') : pct >= 60 ? t('context.filling') : t('context.ok');

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '4px 12px',
        background: 'var(--bg-tertiary)',
        borderBottom: '1px solid var(--border)',
        fontSize: 11,
        color: 'var(--fg-muted)',
      }}
    >
      <span style={{ fontWeight: 600 }}>{label}</span>
      <div
        style={{
          flex: 1,
          maxWidth: 200,
          height: 4,
          background: 'var(--bg-primary)',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${Math.min(pct, 100)}%`,
            height: '100%',
            background: color,
            borderRadius: 2,
            transition: 'width 0.3s ease',
          }}
        />
      </div>
      <span style={{ color, fontWeight: 600 }}>{Math.round(pct)}%</span>
    </div>
  );
}
