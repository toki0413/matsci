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
  const estTokens = Math.round(pct * 320);

  return (
    <div
      title={`~${estTokens.toLocaleString()} / 32,000 tokens used`}
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
          maxWidth: 240,
          height: 4,
          background: 'var(--bg-primary)',
          borderRadius: 2,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <div
          style={{
            width: `${Math.min(pct, 100)}%`,
            height: '100%',
            background: `linear-gradient(90deg, ${color}, ${color}dd)`,
            borderRadius: 2,
            transition: 'width 0.3s ease',
            boxShadow: pct >= 85 ? `0 0 6px ${color}80` : 'none',
          }}
        />
        {/* threshold markers at 60% and 85% */}
        <div style={{ position: 'absolute', left: '60%', top: -1, bottom: -1, width: 1, background: 'rgba(255,255,255,0.15)' }} />
        <div style={{ position: 'absolute', left: '85%', top: -1, bottom: -1, width: 1, background: 'rgba(255,255,255,0.2)' }} />
      </div>
      <span style={{ color, fontWeight: 600 }}>{Math.round(pct)}%</span>
    </div>
  );
}
