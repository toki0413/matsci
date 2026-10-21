import { useCallback, useEffect, useState, type ReactNode } from 'react';
import {
  GitFork, Plus, RefreshCw, Send, Trash2, ArrowUpRight, FileText,
  ChevronDown, ChevronRight, Loader2, MessageSquare, Package,
} from 'lucide-react';
import { PanelHeader } from '../settings-shared';
import EmptyState from '../EmptyState';
import { api } from '../../lib/api';
import { toast } from '../Toast';

interface SideArtifact {
  path: string;
  tool?: string;
}

interface SideTask {
  id: string;
  thread_id: string;
  title: string;
  parent_thread_id: string;
  fork_message_index: number | null;
  seed_count: number;
  seed_consumed: boolean;
  message_count: number;
  artifacts: SideArtifact[];
  status: 'side' | 'promoted';
  is_promoted: boolean;
  created_at: string;
  updated_at: string;
  task_card?: Record<string, any> | null;
  seed_messages?: { role: string; content: string }[];
  messages?: { role: string; content: string; at?: string }[];
}

interface SidePanelProps {
  activeThread: string;
  focusTaskId?: string | null;
  onOpenThread: (threadId: string) => void;
  onThreadsChanged?: () => void;
}

function Badge({ children, tone = 'muted' }: { children: ReactNode; tone?: 'muted' | 'accent' | 'success' }) {
  const cls =
    tone === 'accent'
      ? 'border-accent/40 bg-accent/10 text-accent'
      : tone === 'success'
        ? 'border-success/40 bg-success/10 text-success'
        : 'border-border bg-bg-tertiary text-text-muted';
  return <span className={`rounded border px-1.5 py-0.5 text-[10px] ${cls}`}>{children}</span>;
}

export function SidePanel({ activeThread, focusTaskId, onOpenThread, onThreadsChanged }: SidePanelProps) {
  const [tasks, setTasks] = useState<SideTask[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SideTask | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [showSeed, setShowSeed] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get<any>('/side/tasks');
      setTasks(Array.isArray(res?.tasks) ? res.tasks : []);
      setError('');
    } catch (e: any) {
      setError(e.message || 'Failed to load side tasks');
    } finally {
      setLoading(false);
    }
  }, []);

  const openTask = useCallback(async (id: string) => {
    setSelectedId(id);
    setShowSeed(false);
    try {
      const res = await api.get<any>(`/side/tasks/${id}`);
      setDetail(res?.success ? res.task : null);
    } catch (e: any) {
      setError(e.message || 'Failed to load side task');
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (focusTaskId) openTask(focusTaskId);
  }, [focusTaskId, openTask]);

  const createSideTask = async () => {
    if (!activeThread) {
      toast.error('Open a conversation first');
      return;
    }
    try {
      const res = await api.post<any>('/side/tasks', { parent_thread_id: activeThread });
      if (!res?.success) {
        toast.error(res?.error || 'Fork failed');
        return;
      }
      await load();
      await openTask(res.task.id);
      toast.success(`Side task created (${res.inherited_messages} inherited messages)`);
    } catch (e: any) {
      toast.error(e.message || 'Fork failed');
    }
  };

  const send = async () => {
    const content = input.trim();
    if (!content || !detail || sending) return;
    setSending(true);
    try {
      const res = await api.post<any>(`/side/tasks/${detail.id}/messages`, { content });
      if (!res?.success) {
        toast.error(res?.error || 'Side task failed');
        return;
      }
      setInput('');
      setDetail(res.task);
      await load();
    } catch (e: any) {
      toast.error(e.message || 'Side task failed');
    } finally {
      setSending(false);
    }
  };

  const promote = async () => {
    if (!detail) return;
    try {
      const res = await api.post<any>(`/side/tasks/${detail.id}/promote`, {});
      if (!res?.success) {
        toast.error(res?.error || 'Promote failed');
        return;
      }
      toast.success('Promoted to a formal task');
      onThreadsChanged?.();
      onOpenThread(res.thread_id);
      await load();
      await openTask(detail.id);
    } catch (e: any) {
      toast.error(e.message || 'Promote failed');
    }
  };

  const discard = async () => {
    if (!detail) return;
    try {
      const res = await api.del<any>(`/side/tasks/${detail.id}`);
      if (!res?.success) {
        toast.error(res?.error || 'Discard failed');
        return;
      }
      setDetail(null);
      setSelectedId(null);
      await load();
    } catch (e: any) {
      toast.error(e.message || 'Discard failed');
    }
  };

  const sideTasks = tasks;

  return (
    <div className="flex h-full flex-col">
      <PanelHeader title="Side tasks" className="px-6">
        <button onClick={createSideTask} className="btn-primary flex items-center gap-1 px-3 py-1.5 text-xs">
          <Plus size={12} /> New side task
        </button>
        <button onClick={load} className="btn-secondary flex items-center gap-1 px-3 py-1.5 text-xs" aria-label="Refresh">
          <RefreshCw size={12} className={loading ? 'animate-spin motion-reduce:animate-none' : ''} />
        </button>
      </PanelHeader>

      <div className="flex min-h-0 flex-1">
        {/* task list */}
        <div className="w-72 shrink-0 overflow-y-auto border-r border-border p-3">
          {error && <div className="mb-2 text-xs text-error">{error}</div>}
          {sideTasks.length === 0 ? (
            <EmptyState
              icon={GitFork}
              title="No side tasks"
              subtitle="Fork a message from the main chat, or create one from the current context."
            />
          ) : (
            <div className="space-y-2">
              {sideTasks.map((task) => (
                <button
                  key={task.id}
                  onClick={() => openTask(task.id)}
                  className={`w-full rounded-lg border p-3 text-left transition-colors ${
                    selectedId === task.id
                      ? 'border-accent bg-accent/10'
                      : 'border-border bg-bg-secondary hover:bg-bg-tertiary'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <GitFork size={13} className="mt-0.5 shrink-0 text-accent" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-text-primary">{task.title}</div>
                      <div className="mt-1 flex flex-wrap items-center gap-1">
                        <Badge>{task.seed_count} inherited</Badge>
                        <Badge>{task.message_count} turns</Badge>
                        {task.artifacts.length > 0 && <Badge tone="success">{task.artifacts.length} artifacts</Badge>}
                        {task.is_promoted && <Badge tone="success">formal</Badge>}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* detail */}
        <div className="flex min-w-0 flex-1 flex-col">
          {!detail ? (
            <div className="flex h-full items-center justify-center p-8 text-sm text-text-muted">
              Select a side task to continue the conversation.
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2 border-b border-border px-5 py-3">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold text-text-primary">{detail.title}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-1">
                    <Badge tone={detail.is_promoted ? 'success' : 'accent'}>
                      {detail.is_promoted ? 'formal task' : 'side task'}
                    </Badge>
                    <Badge>{detail.seed_count} inherited</Badge>
                    <Badge>{detail.message_count} turns</Badge>
                  </div>
                </div>
                {!detail.is_promoted && (
                  <button onClick={promote} className="btn-primary flex items-center gap-1 px-3 py-1.5 text-xs">
                    <ArrowUpRight size={12} /> Promote
                  </button>
                )}
                {!detail.is_promoted && (
                  <button onClick={discard} className="btn-secondary flex items-center gap-1 px-3 py-1.5 text-xs text-error">
                    <Trash2 size={12} /> Discard
                  </button>
                )}
              </div>

              <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4">
                {detail.seed_count > 0 && (
                  <div className="rounded-lg border border-border bg-bg-secondary">
                    <button
                      onClick={() => setShowSeed((v) => !v)}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-text-secondary"
                    >
                      {showSeed ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                      Inherited context ({detail.seed_count} messages from {detail.parent_thread_id})
                    </button>
                    {showSeed && (
                      <div className="space-y-2 border-t border-border px-3 py-2">
                        {(detail.seed_messages || []).map((m, i) => (
                          <div key={i} className="text-xs">
                            <span className="mr-2 uppercase text-text-muted">{m.role}</span>
                            <span className="whitespace-pre-wrap text-text-secondary">{m.content}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {(detail.messages || []).length === 0 ? (
                  <div className="flex items-center gap-2 text-xs text-text-muted">
                    <MessageSquare size={13} /> No side messages yet.
                  </div>
                ) : (
                  (detail.messages || []).map((m, i) => (
                    <div
                      key={i}
                      className={`rounded-lg border px-3 py-2 text-sm ${
                        m.role === 'user'
                          ? 'ml-8 border-accent/30 bg-accent/5'
                          : 'mr-8 border-border bg-bg-secondary'
                      }`}
                    >
                      <div className="mb-1 text-[10px] uppercase text-text-muted">{m.role}</div>
                      <div className="whitespace-pre-wrap text-text-primary">{m.content}</div>
                    </div>
                  ))
                )}

                {detail.artifacts.length > 0 && (
                  <div className="rounded-lg border border-border bg-bg-secondary p-3">
                    <div className="mb-2 flex items-center gap-1 text-xs font-medium text-text-secondary">
                      <Package size={12} /> Artifacts
                    </div>
                    <div className="space-y-1">
                      {detail.artifacts.map((a, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs text-text-secondary">
                          <FileText size={12} className="shrink-0" />
                          <span className="truncate">{a.path}</span>
                          {a.tool && <Badge>{a.tool}</Badge>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="border-t border-border p-3">
                <div className="flex items-end gap-2">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        send();
                      }
                    }}
                    rows={2}
                    placeholder="Ask a side question, then discard or promote…"
                    aria-label="Side task message"
                    className="input min-h-[48px] flex-1 resize-none text-sm"
                  />
                  <button
                    onClick={send}
                    disabled={!input.trim() || sending}
                    className="btn-primary flex h-10 items-center gap-1 px-3 text-xs disabled:opacity-50"
                  >
                    {sending ? <Loader2 size={13} className="animate-spin motion-reduce:animate-none" /> : <Send size={13} />}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
