/**
 * DepthModeSelector — Quick / Deep / Research radio group.
 *
 * Renders a compact pill-style selector for conversation depth modes.
 * Used in the chat toolbar area.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

interface DepthModeSelectorProps {
  mode: string;
  onChange: (mode: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

export const DepthModeSelector: React.FC<DepthModeSelectorProps> = ({
  mode,
  onChange,
  className,
  style,
}) => {
  const { t } = useTranslation();
  const modes = [
    { key: 'quick', label: t('depth.quick'), dots: 1 },
    { key: 'deep', label: t('depth.deep'), dots: 2 },
    { key: 'research', label: t('depth.research'), dots: 3 },
  ];

  return (
    <div
      data-component="depth-mode-selector"
      role="radiogroup"
      aria-label={t('depth.label')}
      style={style}
      className={className}
    >
      {modes.map((m) => (
        <button
          key={m.key}
          className={`depth-option ${mode === m.key ? 'depth-option--active' : ''}`}
          onClick={() => onChange(m.key)}
          role="radio"
          aria-checked={mode === m.key}
        >
          <span className="depth-dot" />
          {m.label}
        </button>
      ))}
    </div>
  );
};
