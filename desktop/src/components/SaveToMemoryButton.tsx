/**
 * SaveToMemoryButton — Quick-save a chat message to Huginn's memory system.
 *
 * Renders a small bookmark icon that expands into a compact popup with
 * category and tier selectors. Calls POST /memory on save.
 */
import React, { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Bookmark, CheckCircle2 } from 'lucide-react';
import { api } from '../lib/api';

interface SaveToMemoryButtonProps {
  content: string;
  className?: string;
  style?: React.CSSProperties;
}

export const SaveToMemoryButton: React.FC<SaveToMemoryButtonProps> = ({
  content,
  className,
  style,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [category, setCategory] = useState('insight');
  const [tier, setTier] = useState('mid');
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setError('');
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(
        '/memory',
        {
          content: content.slice(0, 2000),
          category,
          tags: ['chat-save'],
          importance: 0.6,
          tier,
        }
      );
      if (data.success) {
        setSaved(true);
        setOpen(false);
        setTimeout(() => setSaved(false), 2500);
      } else {
        setError(data.error || 'Save failed');
      }
    } catch (e: any) {
      setError(e.message || 'Save error');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      data-component="save-to-memory"
      ref={ref}
      style={style}
      className={className}
    >
      <button
        className={`save-btn ${saved ? 'save-btn--saved' : ''}`}
        onClick={() => setOpen(!open)}
        title={t('save.toMemory')}
        aria-label={t('save.toMemory')}
      >
        {saved ? <CheckCircle2 size={12} /> : <Bookmark size={12} />}
        <span>{saved ? t('save.saved') : t('save.toMemory')}</span>
      </button>
      {open && (
        <div className="save-popup">
          <div className="save-popup-header">{t('save.toMemory')}</div>
          <div style={{ display: 'flex', gap: '6px', marginBottom: '6px' }}>
            <select
              className="input-field"
              style={{ fontSize: '10px', padding: '2px 4px', flex: 1 }}
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="fact">fact</option>
              <option value="insight">insight</option>
              <option value="conversation">conversation</option>
              <option value="calculation">calculation</option>
              <option value="episode">episode</option>
            </select>
            <select
              className="input-field"
              style={{ fontSize: '10px', padding: '2px 4px', flex: 1 }}
              value={tier}
              onChange={(e) => setTier(e.target.value)}
            >
              <option value="short">short</option>
              <option value="mid">mid</option>
              <option value="long">long</option>
            </select>
          </div>
          {error && (
            <div style={{ fontSize: '10px', color: 'var(--error)', marginBottom: '4px' }}>
              {error}
            </div>
          )}
          <button
            className="save-popup-item save-popup-item--new"
            onClick={handleSave}
            disabled={saving}
            style={{ textAlign: 'center', border: 'none', width: '100%', opacity: saving ? 0.6 : 1 }}
          >
            {saving ? '…' : t('save.confirm')}
          </button>
        </div>
      )}
    </div>
  );
};
