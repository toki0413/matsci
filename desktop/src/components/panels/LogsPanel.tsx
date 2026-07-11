import { useTranslation } from 'react-i18next';
import { PanelHeader } from '../settings-shared';
import type { BackendLogEvent } from '../../types/domain';

interface LogsPanelProps {
  backendLogs: BackendLogEvent[];
  logFilter: "all" | "stdout" | "stderr";
  backendLogEndRef: React.RefObject<HTMLDivElement>;
  setLogFilter: (v: "all" | "stdout" | "stderr") => void;
  setBackendLogs: (v: BackendLogEvent[]) => void;
}

export function LogsPanel({
  backendLogs, logFilter, backendLogEndRef, setLogFilter, setBackendLogs,
}: LogsPanelProps) {
  const { t } = useTranslation();
  return (
    <div className="flex h-full flex-col bg-bg-tertiary text-text-primary">
      <PanelHeader title={t('logs.title')}>
        <div className="flex rounded-lg border border-border bg-bg-tertiary p-0.5 text-xs">
          {(["all", "stdout", "stderr"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setLogFilter(f)}
              className={`rounded px-2.5 py-1 capitalize ${
                logFilter === f
                  ? "bg-accent text-white"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        <button
          onClick={() =>
            navigator.clipboard.writeText(
              backendLogs.map((l) => `[${l.time}][${l.source}] ${l.text}`).join("")
            )
          }
          className="btn-secondary px-3 py-1.5 text-xs"
        >
          {t('logs.copy')}
        </button>
        <button
          onClick={() => setBackendLogs([])}
          className="btn-secondary px-3 py-1.5 text-xs"
        >
          {t('logs.clear')}
        </button>
      </PanelHeader>
      <div className="flex-1 overflow-y-auto p-3 font-mono text-sm">
        {backendLogs
          .filter((l) => logFilter === "all" || l.source === logFilter)
          .map((l, i) => (
            <div
              key={i}
              className={`whitespace-pre-wrap break-all ${
                l.source === "stderr" ? "text-error" : "text-text-primary"
              }`}
            >
              <span className="text-text-muted">[{l.time}]</span>{" "}
              {l.text}
            </div>
          ))}
        <div ref={backendLogEndRef} />
      </div>
    </div>
  );
}
