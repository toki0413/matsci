import type { Message } from '../../hooks/useChatAndConnection';
import { formatTime } from '../../lib/constants';

interface Thread {
  id: string;
  label: string;
}

interface ThreadsPanelProps {
  threads: Thread[];
  activeThread: string;
  setThreads: (t: Thread[]) => void;
  setActiveThread: (id: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  createThread: () => void;
  renameThread: (id: string, label: string) => void;
  deleteThread: (id: string) => void;
}

export function ThreadsPanel({
  threads, activeThread, setThreads, setActiveThread, setMessages,
  createThread, renameThread, deleteThread,
}: ThreadsPanelProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
        <span className="text-sm font-semibold">Threads</span>
        <button onClick={createThread} className="btn-primary px-3 py-1.5 text-xs">
          + New thread
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {threads.map((t) => (
            <div
              key={t.id}
              className={`rounded-xl border p-4 transition-colors ${
                activeThread === t.id
                  ? "border-accent bg-accent/10"
                  : "border-border bg-bg-secondary hover:bg-bg-tertiary"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <input
                  value={t.label}
                  onChange={(e) => {
                    const next = threads.map((th) =>
                      th.id === t.id ? { ...th, label: e.target.value } : th
                    );
                    setThreads(next);
                  }}
                  onBlur={(e) => renameThread(t.id, e.target.value)}
                  className="w-full bg-transparent text-sm font-semibold text-text-primary focus:outline-none"
                />
                <button
                  onClick={() => deleteThread(t.id)}
                  className="text-xs text-error hover:underline"
                >
                  Delete
                </button>
              </div>
              <div className="mt-2 text-[10px] text-text-muted">ID: {t.id}</div>
              <button
                onClick={() => {
                  setActiveThread(t.id);
                  setMessages([
                    {
                      role: "assistant",
                      content: `Switched to thread **${t.label}**.`,
                      timestamp: formatTime(),
                    },
                  ]);
                }}
                disabled={activeThread === t.id}
                className="mt-3 w-full rounded-lg border border-border bg-bg-tertiary py-1.5 text-xs text-text-secondary hover:text-text-primary disabled:opacity-50"
              >
                {activeThread === t.id ? "Active" : "Switch"}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
