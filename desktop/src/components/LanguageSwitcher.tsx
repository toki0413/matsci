/**
 * LanguageSwitcher — EN/ZH toggle button for the toolbar.
 *
 * Uses react-i18next to toggle between English and Chinese.
 * Renders a compact pill button showing the current language.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

export const LanguageSwitcher: React.FC = () => {
  const { i18n } = useTranslation();
  const isZh = i18n.language === 'zh';

  const toggle = () => {
    i18n.changeLanguage(isZh ? 'en' : 'zh');
  };

  return (
    <button
      onClick={toggle}
      className="inline-flex items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-xs font-medium text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors"
      title={isZh ? 'Switch to English' : '切换为中文'}
    >
      <svg
        width="13"
        height="13"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M2 12h20" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
      </svg>
      {isZh ? '中' : 'EN'}
    </button>
  );
};
