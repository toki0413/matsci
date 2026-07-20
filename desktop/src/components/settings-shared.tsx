import type { ReactNode } from "react";

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

const TAB_LABELS: Record<SettingsTab, string> = {
  general: "general",
  models: "models",
  agents: "agents",
  privacy: "privacy",
  pet: "pet",
  security: "security",
  credentials: "credentials",
  jobs: "SSH Jobs",
  export: "Export",
  bot: "Bot",
  advanced: "Advanced",
};

const TAB_ORDER: SettingsTab[] = [
  "general", "models", "agents", "privacy", "pet",
  "security", "credentials", "jobs", "export", "bot", "advanced",
];

export function SettingsTabNav({
  activeTab,
  onTabChange,
}: {
  activeTab: SettingsTab;
  onTabChange: (tab: SettingsTab) => void;
}) {
  return (
    <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
      <span className="text-sm font-semibold">Settings</span>
      <div className="flex items-center gap-2">
        {TAB_ORDER.map((t) => (
          <button
            key={t}
            onClick={() => onTabChange(t)}
            className={`rounded px-3 py-1 text-xs capitalize ${
              activeTab === t
                ? "bg-accent text-white"
                : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
            }`}
          >
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>
    </div>
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
