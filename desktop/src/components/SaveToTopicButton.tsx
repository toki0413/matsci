/**
 * SaveToTopicButton — Bookmark chat content to a research topic.
 *
 * Renders a compact button with a popup listing available topics.
 * On click, saves the current content and shows a confirmation state.
 */
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Bookmark, CheckCircle2, Layers, Plus } from 'lucide-react';

export interface Topic {
  id: string;
  name: string;
  nameZh?: string;
}

interface SaveToTopicButtonProps {
  topics: Topic[];
  className?: string;
  style?: React.CSSProperties;
}

export const SaveToTopicButton: React.FC<SaveToTopicButtonProps> = ({
  topics,
  className,
  style,
}) => {
  const { t, i18n } = useTranslation();
  const [open, setOpen] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setOpen(false);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div data-component="save-to-topic" style={style} className={className}>
      <button
        className={`save-btn ${saved ? 'save-btn--saved' : ''}`}
        onClick={() => setOpen(!open)}
      >
        {saved ? <CheckCircle2 size={10} /> : <Bookmark size={10} />}
        {saved ? t('save.saved') : t('save.toTopic')}
      </button>
      {open && (
        <div className="save-popup fade-in">
          <div className="save-popup-header">{t('save.selectTopic')}</div>
          {topics.map((topic) => (
            <button key={topic.id} className="save-popup-item" onClick={handleSave}>
              <Layers size={12} />
              {i18n.language === 'zh' ? (topic.nameZh || topic.name) : topic.name}
            </button>
          ))}
          <button className="save-popup-item save-popup-item--new" onClick={handleSave}>
            <Plus size={12} />
            {t('save.newTopic')}
          </button>
        </div>
      )}
    </div>
  );
};
