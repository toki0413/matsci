import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import {
  Settings, Cpu, Users, Lock, Smile, ShieldCheck, KeyRound,
  Server, Download, Bot, SlidersHorizontal,
  type LucideIcon,
} from "lucide-react";

export type SettingsTab =
  | "general"
  | "models"
  | "agents"
  | "privacy"
  | "pet"
  | "security"
  | "credentials"
  | "jobs"
  | "export"
  | "bot"
  | "advanced";

// Per-tab icon + localized label key so the nav stays consistent in both langs.
const TAB_META: Record<SettingsTab, { labelKey: string; Icon: LucideIcon }> = {
  general:     { labelKey: "settings.general", Icon: Settings },
  models:      { labelKey: "settings.models", Icon: Cpu },
  agents:      { labelKey: "settings.agents", Icon: Users },
  privacy:     { labelKey: "settings.privacy", Icon: Lock },
  pet:         { labelKey: "settings.pet", Icon: Smile },
  security:    { labelKey: "settings.security", Icon: ShieldCheck },
  credentials: { labelKey: "settings.credentials", Icon: KeyRound },
  jobs:        { labelKey: "settings.jobs", Icon: Server },
  export:      { labelKey: "settings.export", Icon: Download },
  bot:         { labelKey: "settings.bot", Icon: Bot },
  advanced:    { labelKey: "settings.advanced", Icon: SlidersHorizontal },
};

// 11 tabs grouped into four sections instead of one flat strip.
const TAB_GROUPS: Array<{ titleKey: string; tabs: SettingsTab[] }> = [
  { titleKey: "settingGroup.preferences", tabs: ["general", "models", "agents"] },
  { titleKey: "settingGroup.security", tabs: ["security", "credentials"] },
  { titleKey: "settingGroup.experience", tabs: ["privacy", "pet", "export"] },
  { titleKey: "settingGroup.runtime", tabs: ["jobs", "bot", "advanced"] },
];

export function SettingsTabNav({
  activeTab,
  onTabChange,
}: {
  activeTab: SettingsTab;
  onTabChange: (tab: SettingsTab) => void;
}) {
  const { t } = useTranslation();
  return (
    <aside className="flex w-56 shrink-0 flex-col overflow-y-auto border-r border-border bg-bg-secondary p-3">
      <div className="px-2 pb-3 text-sm font-semibold">{t('settings.title')}</div>
      {TAB_GROUPS.map((g) => (
        <div key={g.titleKey} className="mb-3 last:mb-0">
          <div className="px-2 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
            {t(g.titleKey)}
          </div>
          {g.tabs.map((tb) => {
            const { labelKey, Icon } = TAB_META[tb];
            return (
              <button
                key={tb}
                onClick={() => onTabChange(tb)}
                aria-current={activeTab === tb ? "page" : undefined}
                className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-colors ${
                  activeTab === tb
                    ? "bg-accent text-white"
                    : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
                }`}
              >
                <Icon size={14} aria-hidden="true" />
                {t(labelKey)}
              </button>
            );
          })}
        </div>
      ))}
    </aside>
  );
}

/**
 * Standard label + wrapper for a single config input.
 * Use `full` for fields that should span both columns on md+.
 */
export function ConfigField({
  label,
  full,
  children,
}: {
  label: string;
  full?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={full ? "md:col-span-2" : undefined}>
      <label className="mb-1.5 block text-xs font-medium text-text-secondary">
        {label}
      </label>
      {children}
    </div>
  );
}

/** Reusable panel header bar — title on left, optional actions on right. */
export function PanelHeader({
  title,
  children,
  className,
}: {
  title: string;
  children?: ReactNode;
  className?: string;
}) {
  // ponytail: className replaces the default px-4 so callers keep their own padding
  // (px-6 for full-width panels) and hook classes (kb-header / mem-header). If a title
  // ever needs per-element styling (e.g. truncate on a dynamic path), keep a raw div
  // or add a titleClassName prop — don't shove it on the container.
  return (
    <div className={`flex h-12 items-center justify-between border-b border-border bg-bg-secondary ${className ?? "px-4"}`}>
      <h2 className="text-sm font-semibold text-text-primary">{title}</h2>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  );
}
