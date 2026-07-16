import { useState, useEffect, useRef } from 'react';
import { Search, MoreVertical, GitFork, Archive, ArchiveRestore } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { PanelHeader } from '../settings-shared';

interface Thread {
  id: string;
  label: string;
  archived?: boolean;
}

interface ThreadsPanelProps {
  threads: Thread[];
  activeThread: string;
  setThreads: (t: Thread[]) => void;
  switchThread: (id: string) => void;
  createThread: () => void;
  renameThread: (id: string, label: string) => void;
  deleteThread: (id: string) => void;
  forkThread: (id: string) => void;
  archiveThread: (id: string) => void;
  unarchiveThread: (id: string) => void;
  loadThreads: (includeArchived?: boolean) => void;
}

export function ThreadsPanel({
  threads, activeThread, setThreads, switchThread, createThread, renameThread, deleteThread,
  forkThread, archiveThread, unarchiveThread, loadThreads,
}: ThreadsPanelProps) {
  const { t } = useTranslation();
  const [threadSearch, setThreadSearch] = useState('');
  const [showArchived, setShowArchived] = useState(false);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // 切换归档视图时重新拉. 之前 include_archived 硬编码 false, 看不到归档线程.
  useEffect(() => { loadThreads(showArchived); }, [showArchived]);

  // 点外面关菜单
  useEffect(() => {
    if (!menuOpenId) return;
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpenId(null);
    };
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, [menuOpenId]);

  const filteredThreads = threadSearch
    ? threads.filter(th => th.label.toLowerCase().includes(threadSearch.toLowerCase()))
    : threads;

  return (
    <div className="flex h-full flex-col">
      <PanelHeader title="Threads" className="px-6">
        <button onClick={createThread} className="btn-primary px-3 py-1.5 text-xs">
          + {t('threads.new')}
        </button>
      </PanelHeader>
      <div className="flex items-center gap-2 border-b border-border px-6 py-2">
        <Search size={14} className="shrink-0 text-text-muted" />
        <input
          type="text"
          value={threadSearch}
          onChange={(e) => setThreadSearch(e.target.value)}
          placeholder="Search threads..."
          className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
        />
        {threadSearch && <span className="text-[11px] text-text-muted">{filteredThreads.length}/{threads.length}</span>}
        <button
          onClick={() => setShowArchived(v => !v)}
          className={`text-[11px] px-2 py-1 rounded border ${
            showArchived
              ? "border-accent bg-accent/10 text-accent"
              : "border-border text-text-muted hover:text-text-primary"
          }`}
          title={showArchived ? "Showing archived threads" : "Show archived"}
        >
          <Archive size={11} style={{ display: 'inline', marginRight: 4 }} />
          {showArchived ? "Archived" : "Active"}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {threads.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <div className="text-5xl opacity-30">💬</div>
            <div>
              <div className="text-base font-medium text-text-secondary">{t('threads.empty') || 'No threads yet'}</div>
              <div className="mt-1 text-sm text-text-muted">{t('threads.emptyHint') || 'Create a thread to start a new conversation'}</div>
            </div>
            <button onClick={createThread} className="btn-primary px-4 py-2 text-sm">
              + {t('threads.new')}
            </button>
          </div>
        ) : filteredThreads.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-center text-text-muted">
            <Search size={28} className="opacity-30" />
            <span className="text-sm">No threads match "{threadSearch}"</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {filteredThreads.map((th) => (
              <div
                key={th.id}
                className={`rounded-xl border p-4 transition-colors relative ${
                  activeThread === th.id
                    ? "border-accent bg-accent/10"
                    : "border-border bg-bg-secondary hover:bg-bg-tertiary"
                } ${th.archived ? "opacity-60" : ""}`}
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
                    onClick={() => setMenuOpenId(menuOpenId === th.id ? null : th.id)}
                    className="rounded p-1 text-text-muted hover:text-text-primary hover:bg-bg-tertiary"
                    aria-label="More actions"
                  >
                    <MoreVertical size={14} />
                  </button>
                </div>
                <div className="mt-2 text-[10px] text-text-muted">
                  ID: {th.id}
                  {th.archived && <span className="ml-2 text-warning">· archived</span>}
                  {th.id.includes('fork') && <span className="ml-2 text-purple-400">· forked</span>}
                </div>
                <button
                  onClick={() => switchThread(th.id)}
                  disabled={activeThread === th.id}
                  className="mt-3 w-full rounded-lg border border-border bg-bg-tertiary py-1.5 text-xs text-text-secondary hover:text-text-primary disabled:opacity-50"
                >
                  {activeThread === th.id ? t('threads.active') : t('threads.switch')}
                </button>
                {menuOpenId === th.id && (
                  <div
                    ref={menuRef}
                    className="absolute right-2 top-10 z-20 w-36 rounded-lg border border-border bg-bg-secondary shadow-lg overflow-hidden"
                  >
                    <button
                      onClick={() => { forkThread(th.id); setMenuOpenId(null); }}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-text-primary hover:bg-bg-tertiary"
                    >
                      <GitFork size={11} /> Fork
                    </button>
                    {th.archived ? (
                      <button
                        onClick={() => { unarchiveThread(th.id); setMenuOpenId(null); }}
                        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-text-primary hover:bg-bg-tertiary"
                      >
                        <ArchiveRestore size={11} /> Unarchive
                      </button>
                    ) : (
                      <button
                        onClick={() => { archiveThread(th.id); setMenuOpenId(null); }}
                        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-text-primary hover:bg-bg-tertiary"
                      >
                        <Archive size={11} /> Archive
                      </button>
                    )}
                    <button
                      onClick={() => { deleteThread(th.id); setMenuOpenId(null); }}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-error hover:bg-error/10"
                    >
                      {t('common.delete')}
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
