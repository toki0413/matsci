/**
 * TopicCard & TopicDetail — Knowledge Hub topic components.
 *
 * TopicCard renders a summary card with icon, name, description, and stats.
 * TopicDetail renders the drill-down view with documents grouped by source.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  Cpu,
  BarChart3,
  BookOpen,
  Layers,
  ChevronRight,
  Upload,
  MessageSquare,
  StickyNote,
  FileText,
} from 'lucide-react';

// ── Types ──────────────────────────────────────────────────────

export interface TopicDoc {
  id: string;
  name: string;
  source: 'upload' | 'chat' | 'note';
  size: string;
  date: string;
}

export interface TopicData {
  id: string;
  name: string;
  nameZh?: string;
  description: string;
  descriptionZh?: string;
  icon: 'cpu' | 'chart' | 'book';
  lastUpdated: string;
  docCount: number;
  noteCount: number;
  chatCount: number;
  docs: TopicDoc[];
}

// ── TopicCard ──────────────────────────────────────────────────

interface TopicCardProps {
  topic: TopicData;
  onClick: (topic: TopicData) => void;
  lang: string;
  className?: string;
  style?: React.CSSProperties;
}

const iconMap = { cpu: Cpu, chart: BarChart3, book: BookOpen };

export const TopicCard: React.FC<TopicCardProps> = ({
  topic,
  onClick,
  lang,
  className,
  style,
}) => {
  const { t } = useTranslation();
  const Icon = iconMap[topic.icon] || Layers;
  const name = lang === 'zh' ? (topic.nameZh || topic.name) : topic.name;
  const desc =
    lang === 'zh'
      ? (topic.descriptionZh || topic.description)
      : topic.description;

  return (
    <div
      data-component="topic-card"
      onClick={() => onClick(topic)}
      className={className || ''}
      style={style}
    >
      <div className="topic-card-top">
        <div className="topic-card-icon">
          <Icon size={16} />
        </div>
        <div className="topic-card-info">
          <div className="topic-card-name">{name}</div>
          <div className="topic-card-date">
            {t('topic.lastUpdated')} {topic.lastUpdated}
          </div>
        </div>
      </div>
      <div className="topic-card-desc">{desc}</div>
      <div className="topic-card-stats">
        <span className="topic-stat">
          <span className="topic-stat-num">{topic.docCount}</span>{' '}
          {t('topic.docs')}
        </span>
        {topic.noteCount > 0 && (
          <span className="topic-stat">
            <span className="topic-stat-num">{topic.noteCount}</span>{' '}
            {t('topic.notes')}
          </span>
        )}
        {topic.chatCount > 0 && (
          <span className="topic-stat">
            <span className="topic-stat-num">{topic.chatCount}</span>{' '}
            {t('topic.chatSaves')}
          </span>
        )}
      </div>
    </div>
  );
};

// ── TopicDetail ────────────────────────────────────────────────

interface TopicDetailProps {
  topic: TopicData;
  lang: string;
  onBack: () => void;
  className?: string;
  style?: React.CSSProperties;
}

const sourceIcons: Record<string, typeof Upload> = {
  upload: Upload,
  chat: MessageSquare,
  note: StickyNote,
};

export const TopicDetail: React.FC<TopicDetailProps> = ({
  topic,
  lang,
  onBack,
  className,
  style,
}) => {
  const { t } = useTranslation();
  const name = lang === 'zh' ? (topic.nameZh || topic.name) : topic.name;
  const sourceLabels: Record<string, string> = {
    upload: t('topic.source.upload'),
    chat: t('topic.source.chat'),
    note: t('topic.source.note'),
  };

  const grouped = topic.docs.reduce(
    (acc: Record<string, TopicDoc[]>, doc: TopicDoc) => {
      if (!acc[doc.source]) acc[doc.source] = [];
      acc[doc.source].push(doc);
      return acc;
    },
    {},
  );

  return (
    <div
      data-component="topic-detail"
      className={`fade-in ${className || ''}`}
      style={style}
    >
      <div className="td-header">
        <button className="td-back" onClick={onBack}>
          <ChevronRight size={12} style={{ transform: 'rotate(180deg)' }} />
          {t('topic.back')}
        </button>
        <span className="td-title">{name}</span>
      </div>
      <div className="td-doc-list">
        {Object.entries(grouped).map(([source, docs]) => {
          const SourceIcon = sourceIcons[source] || FileText;
          return (
            <div key={source}>
              <div className="td-section-label">
                {sourceLabels[source] || source}
              </div>
              {docs.map((doc: TopicDoc) => (
                <div className="td-doc-item" key={doc.id}>
                  <SourceIcon size={14} className="td-doc-icon" />
                  <div className="td-doc-body">
                    <div className="td-doc-name">{doc.name}</div>
                    <div className="td-doc-meta">
                      {doc.size !== '—' ? doc.size + ' · ' : ''}
                      {doc.date}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
};
