import { useTranslation } from 'react-i18next';
import { PanelHeader } from '../settings-shared';

interface Thread {
  id: string;
  label: string;
}

interface ThreadsPanelProps {
  threads: Thread[];
  activeThread: string;
  setThreads: (t: Thread[]) => void;
  switchThread: (id: string) => void;
  createThread: () => void;
  renameThread: (id: string, label: string) => void;
  deleteThread: (id: string) => void;
}

export function ThreadsPanel({
  threads, activeThread, setThreads, switchThread, createThread, renameThread, deleteThread,
}: ThreadsPanelProps) {
  const { t } = useTranslation();
  return (
    <div className="flex h-full flex-col">
      <PanelHeader title="Threads" className="px-6">
        <button onClick={createThread} className="btn-primary px-3 py-1.5 text-xs">
          + {t('threads.new')}
        </button>
      </PanelHeader>
      <div className="flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {threads.map((th) => (
            <div
              key={th.id}
              className={`rounded-xl border p-4 transition-colors ${
                activeThread === th.id
                  ? "border-accent bg-accent/10"
                  : "border-border bg-bg-secondary hover:bg-bg-tertiary"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <input
                  value={th.label}
                  onChange={(e) => {
                    const next = threads.map((t) =>
                      t.id === th.id ? { ...t, label: e.target.value } : t
                    );
                    setThreads(next);
                  }}
                  onBlur={(e) => renameThread(th.id, e.target.value)}
                  className="w-full bg-transparent text-sm font-semibold text-text-primary focus:outline-none"
                />
                <button
                  onClick={() => deleteThread(th.id)}
                  className="text-xs text-error hover:underline"
                >
                  {t('common.delete')}
                </button>
              </div>
              <div className="mt-2 text-[10px] text-text-muted">ID: {th.id}</div>
              <button
                onClick={() => switchThread(th.id)}
                disabled={activeThread === th.id}
                className="mt-3 w-full rounded-lg border border-border bg-bg-tertiary py-1.5 text-xs text-text-secondary hover:text-text-primary disabled:opacity-50"
              >
                {activeThread === th.id ? t('threads.active') : t('threads.switch')}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
