/**
 * ChatModeSelector — Chat / Plan / Build radio group.
 *
 * Compact pill-style selector matching the production agent modes.
 * Replaces the old DepthModeSelector (Quick/Deep/Research).
 */
import React from 'react';
import { useTranslation } from 'react-i18next';
import { MessageSquare, ClipboardList, Code2 } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

type ChatMode = 'chat' | 'plan' | 'build';

interface ChatModeSelectorProps {
  mode: ChatMode;
  onChange: (mode: ChatMode) => void;
  className?: string;
  style?: React.CSSProperties;
}

const modeConfig: { key: ChatMode; icon: LucideIcon }[] = [
  { key: 'chat', icon: MessageSquare },
  { key: 'plan', icon: ClipboardList },
  { key: 'build', icon: Code2 },
];

export const ChatModeSelector: React.FC<ChatModeSelectorProps> = ({
  mode,
  onChange,
  className,
  style,
}) => {
  const { t } = useTranslation();

  return (
    <div
      data-component="chat-mode-selector"
      role="radiogroup"
      aria-label="Chat mode"
      className={className}
      style={style}
    >
      {modeConfig.map((m) => {
        const Icon = m.icon;
        const active = mode === m.key;
        return (
          <button
            key={m.key}
            className={`chat-mode-option ${active ? 'chat-mode-option--active' : ''}`}
            onClick={() => onChange(m.key)}
            role="radio"
            aria-checked={active}
            title={t(`chat.mode.${m.key}.desc`)}
          >
            <Icon size={12} />
            {t(`chat.mode.${m.key}`)}
          </button>
        );
      })}
    </div>
  );
};
