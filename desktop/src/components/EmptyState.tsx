/** Shared empty-state placeholder — icon + title + subtitle, centered. */
import type { LucideIcon } from "lucide-react";

export default function EmptyState({
  icon: Icon,
  title,
  subtitle,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="kh-empty flex flex-col items-center justify-center py-16 text-center">
      {Icon && <Icon size={36} className="text-text-muted opacity-40" />}
      <p className="mt-3 text-sm font-medium text-text-secondary">{title}</p>
      {subtitle && (
        <p className="mt-1 max-w-xs text-xs text-text-muted">{subtitle}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
