/**
 * StructuredOutput — Outline / Mindmap / Table view switcher.
 *
 * Renders DFT workflow results in three alternative views:
 * a hierarchical outline, an SVG mindmap, and a parameter table.
 * All content is i18n-aware.
 */
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronRight } from 'lucide-react';

interface StructuredOutputProps {
  view?: string;
  className?: string;
  style?: React.CSSProperties;
}

export const StructuredOutput: React.FC<StructuredOutputProps> = ({
  view,
  className,
  style,
}) => {
  const { t } = useTranslation();
  const [activeView, setActiveView] = useState(view || 'outline');

  const tabs = [
    { key: 'outline', label: t('output.outline') },
    { key: 'mindmap', label: t('output.mindmap') },
    { key: 'table', label: t('output.table') },
  ];

  return (
    <div
      data-component="structured-output"
      className={`fade-in ${className || ''}`}
      style={style}
    >
      <div className="so-toolbar">
        <div className="so-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              className={`so-tab ${activeView === tab.key ? 'so-tab--active' : ''}`}
              onClick={() => setActiveView(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      <div className="so-body">
        {activeView === 'outline' && (
          <div className="so-outline">
            <div className="so-outline-item so-outline-item--l1">
              <ChevronRight size={10} className="so-outline-bullet" />
              {t('output.outline.l1.params')}
            </div>
            <div className="so-outline-item so-outline-item--l2">
              <span className="so-outline-bullet">–</span>
              {t('output.outline.l2.poscar')}
            </div>
            <div className="so-outline-item so-outline-item--l2">
              <span className="so-outline-bullet">–</span>
              {t('output.outline.l2.incar')}
            </div>
            <div className="so-outline-item so-outline-item--l1">
              <ChevronRight size={10} className="so-outline-bullet" />
              {t('output.outline.l1.convergence')}
            </div>
            <div className="so-outline-item so-outline-item--l2">
              <span className="so-outline-bullet">–</span>
              {t('output.outline.l2.encut')}
            </div>
            <div className="so-outline-item so-outline-item--l2">
              <span className="so-outline-bullet">–</span>
              {t('output.outline.l2.kpoints')}
            </div>
            <div className="so-outline-item so-outline-item--l1">
              <ChevronRight size={10} className="so-outline-bullet" />
              {t('output.outline.l1.results')}
            </div>
            <div className="so-outline-item so-outline-item--l2">
              <span className="so-outline-bullet">–</span>
              {t('output.outline.l2.energy')}
            </div>
            <div className="so-outline-item so-outline-item--l2">
              <span className="so-outline-bullet">–</span>
              {t('output.outline.l2.magmom')}
            </div>
          </div>
        )}
        {activeView === 'mindmap' && (
          <div className="so-mindmap">
            <svg viewBox="0 0 360 160" fill="none">
              <rect x="130" y="65" width="100" height="30" rx="6" fill="var(--bg-surface)" stroke="var(--border)" strokeWidth="1" />
              <text x="180" y="84" textAnchor="middle" fontSize="10" fontWeight="550" fill="var(--fg-primary)">{t('output.mindmap.center')}</text>
              <line x1="130" y1="80" x2="70" y2="40" stroke="var(--border-strong)" strokeWidth="1" />
              <line x1="130" y1="80" x2="70" y2="80" stroke="var(--border-strong)" strokeWidth="1" />
              <line x1="130" y1="80" x2="70" y2="120" stroke="var(--border-strong)" strokeWidth="1" />
              <line x1="230" y1="80" x2="290" y2="40" stroke="var(--border-strong)" strokeWidth="1" />
              <line x1="230" y1="80" x2="290" y2="80" stroke="var(--border-strong)" strokeWidth="1" />
              <line x1="230" y1="80" x2="290" y2="120" stroke="var(--border-strong)" strokeWidth="1" />
              <rect x="15" y="28" width="55" height="24" rx="4" fill="var(--bg-surface)" stroke="var(--border)" strokeWidth="1" />
              <text x="42" y="44" textAnchor="middle" fontSize="9" fill="var(--fg-secondary)">POSCAR</text>
              <rect x="15" y="68" width="55" height="24" rx="4" fill="var(--bg-surface)" stroke="var(--border)" strokeWidth="1" />
              <text x="42" y="84" textAnchor="middle" fontSize="9" fill="var(--fg-secondary)">INCAR</text>
              <rect x="15" y="108" width="55" height="24" rx="4" fill="var(--bg-surface)" stroke="var(--border)" strokeWidth="1" />
              <text x="42" y="124" textAnchor="middle" fontSize="9" fill="var(--fg-secondary)">KPOINTS</text>
              <rect x="265" y="28" width="80" height="24" rx="4" fill="var(--accent-subtle)" stroke="var(--border)" strokeWidth="1" />
              <text x="305" y="44" textAnchor="middle" fontSize="9" fill="var(--accent)">-8.312 eV</text>
              <rect x="265" y="68" width="80" height="24" rx="4" fill="var(--success-subtle)" stroke="var(--border)" strokeWidth="1" />
              <text x="305" y="84" textAnchor="middle" fontSize="9" fill="var(--success)">{t('output.mindmap.converged')}</text>
              <rect x="265" y="108" width="80" height="24" rx="4" fill="var(--bg-surface)" stroke="var(--border)" strokeWidth="1" />
              <text x="305" y="124" textAnchor="middle" fontSize="9" fill="var(--fg-secondary)">2.22 μB</text>
            </svg>
          </div>
        )}
        {activeView === 'table' && (
          <table className="so-table">
            <thead>
              <tr>
                <th>{t('output.table.header.parameter')}</th>
                <th>{t('output.table.header.value')}</th>
                <th>{t('output.table.header.status')}</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>{t('output.table.functional')}</td><td>PBE</td><td>{t('output.table.standard')}</td></tr>
              <tr><td>ENCUT</td><td>520 eV</td><td>{t('output.table.converged')}</td></tr>
              <tr><td>K-points</td><td>11×11×11</td><td>{t('output.table.converged')}</td></tr>
              <tr><td>EDIFF</td><td>1E-6 eV</td><td>{t('output.table.tight')}</td></tr>
              <tr><td>ISIF</td><td>3</td><td>{t('output.table.fullRelax')}</td></tr>
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
