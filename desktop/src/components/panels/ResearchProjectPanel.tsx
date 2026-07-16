import { useState, useEffect, useCallback } from 'react';
import { PanelHeader } from '../settings-shared';
import { SkeletonCard } from '../Skeleton';
import { api } from '../../lib/api';

// types kept inline — one panel, no need for a shared domain file
interface ResearchProject {
  id: string;
  title: string;
  description: string;
  instructions: string;
  thread_ids: string[];
  knowledge_doc_ids: string[];
  created_at: string;
  updated_at: string;
  search_scope: 'local' | 'web' | 'both';
}

interface Thread {
  id: string;
  label: string;
}

interface KbDoc {
  doc_id: string;
  filename: string;
}

interface ResearchProjectPanelProps {
  onOpenInChat?: (project: ResearchProject) => void;
}

const SCOPES = ['local', 'web', 'both'] as const;

export function ResearchProjectPanel({ onOpenInChat }: ResearchProjectPanelProps) {
  const [projects, setProjects] = useState<ResearchProject[]>([]);
  const [selected, setSelected] = useState<ResearchProject | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState('');

  // create form
  const [showCreate, setShowCreate] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newInstr, setNewInstr] = useState('');

  // inline editing of instructions
  const [editInstr, setEditInstr] = useState('');
  const [saving, setSaving] = useState(false);

  // dropdowns
  const [threads, setThreads] = useState<Thread[]>([]);
  const [kbDocs, setKbDocs] = useState<KbDoc[]>([]);
  const [attachThread, setAttachThread] = useState('');
  const [attachDoc, setAttachDoc] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get<{ projects: ResearchProject[] }>('/projects');
      setProjects(res.projects ?? []);
    } catch (e: any) {
      setMsg(e.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  // grab threads + KB docs when a project is opened (for the dropdowns)
  useEffect(() => {
    if (!selected) return;
    setEditInstr(selected.instructions);
    api.get<{ threads: Thread[] }>('/threads').then(r => setThreads(r.threads ?? [])).catch(() => {});
    api.get<{ documents: KbDoc[] }>('/knowledge').then(r => setKbDocs(r.documents ?? [])).catch(() => {});
  }, [selected]);

  const create = async () => {
    if (!newTitle.trim()) return;
    try {
      const res = await api.post<{ project: ResearchProject }>('/projects', {
        title: newTitle, description: newDesc, instructions: newInstr,
      });
      setProjects(prev => [...prev, res.project]);
      setSelected(res.project);
      setShowCreate(false);
      setNewTitle(''); setNewDesc(''); setNewInstr('');
    } catch (e: any) { setMsg(e.message); }
  };

  const remove = async (pid: string) => {
    if (!confirm('Delete this research project?')) return;
    try {
      await api.del(`/projects/${pid}`);
      setProjects(prev => prev.filter(p => p.id !== pid));
      if (selected?.id === pid) setSelected(null);
    } catch (e: any) { setMsg(e.message); }
  };

  const saveInstr = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      const res = await api.patch<{ project: ResearchProject }>(
        `/projects/${selected.id}`, { instructions: editInstr },
      );
      setSelected(res.project);
      setProjects(prev => prev.map(p => p.id === res.project.id ? res.project : p));
      setMsg('Instructions saved');
      setTimeout(() => setMsg(''), 2000);
    } catch (e: any) { setMsg(e.message); }
    finally { setSaving(false); }
  };

  const setScope = async (scope: string) => {
    if (!selected) return;
    try {
      const res = await api.patch<{ project: ResearchProject }>(
        `/projects/${selected.id}`, { search_scope: scope },
      );
      setSelected(res.project);
      setProjects(prev => prev.map(p => p.id === res.project.id ? res.project : p));
    } catch (e: any) { setMsg(e.message); }
  };

  const addThread = async () => {
    if (!selected || !attachThread) return;
    try {
      const res = await api.post<{ project: ResearchProject }>(
        `/projects/${selected.id}/threads`, { thread_id: attachThread },
      );
      setSelected(res.project);
      setProjects(prev => prev.map(p => p.id === res.project.id ? res.project : p));
      setAttachThread('');
    } catch (e: any) { setMsg(e.message); }
  };

  const rmThread = async (tid: string) => {
    if (!selected) return;
    try {
      const res = await api.del<{ project: ResearchProject }>(
        `/projects/${selected.id}/threads/${tid}`,
      );
      setSelected(res.project);
      setProjects(prev => prev.map(p => p.id === res.project.id ? res.project : p));
    } catch (e: any) { setMsg(e.message); }
  };

  const addDoc = async () => {
    if (!selected || !attachDoc) return;
    try {
      const res = await api.post<{ project: ResearchProject }>(
        `/projects/${selected.id}/knowledge`, { doc_id: attachDoc },
      );
      setSelected(res.project);
      setProjects(prev => prev.map(p => p.id === res.project.id ? res.project : p));
      setAttachDoc('');
    } catch (e: any) { setMsg(e.message); }
  };

  const rmDoc = async (docId: string) => {
    if (!selected) return;
    try {
      const res = await api.del<{ project: ResearchProject }>(
        `/projects/${selected.id}/knowledge/${docId}`,
      );
      setSelected(res.project);
      setProjects(prev => prev.map(p => p.id === res.project.id ? res.project : p));
    } catch (e: any) { setMsg(e.message); }
  };

  // helpers for displaying attached items by id
  const threadLabel = (id: string) => threads.find(t => t.id === id)?.label ?? id;
  const docLabel = (id: string) => kbDocs.find(d => d.doc_id === id)?.filename ?? id;

  // ── Detail view ──
  if (selected) {
    return (
      <div className="flex h-full flex-col">
        <PanelHeader title={selected.title} className="px-6">
          <button onClick={() => setSelected(null)} className="btn-secondary px-3 py-1.5 text-xs">
            Back
          </button>
          {onOpenInChat && (
            <button
              onClick={() => onOpenInChat(selected)}
              className="btn-primary px-3 py-1.5 text-xs ml-2"
            >
              Open in Chat
            </button>
          )}
        </PanelHeader>

        <div className="flex flex-1 overflow-hidden">
          {/* left: instructions + scope */}
          <aside className="flex w-1/2 flex-col border-r border-border bg-bg-secondary p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold">Project Instructions</h3>
              <button onClick={saveInstr} disabled={saving} className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50">
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>

            {/* scope toggle — Perplexity-style */}
            <div className="mb-3">
              <div className="mb-1.5 text-xs font-medium text-text-secondary">Search Scope</div>
              <div className="flex rounded-lg border border-border bg-bg-tertiary p-0.5">
                {SCOPES.map(s => (
                  <button
                    key={s}
                    onClick={() => setScope(s)}
                    className={
                      'flex-1 rounded-md px-2.5 py-1 text-xs font-medium capitalize transition-colors ' +
                      (selected.search_scope === s
                        ? 'bg-accent text-white'
                        : 'text-text-secondary hover:text-text-primary')
                    }
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            <textarea
              value={editInstr}
              onChange={e => setEditInstr(e.target.value)}
              placeholder="Project-level system prompt — coding conventions, domain context, preferred methods…"
              className="input flex-1 resize-none font-mono text-sm"
              spellCheck={false}
            />

            {selected.description && (
              <div className="mt-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
                {selected.description}
              </div>
            )}
          </aside>

          {/* right: threads + knowledge */}
          <div className="flex flex-1 flex-col overflow-y-auto bg-bg-primary p-4 space-y-4">
            {msg && (
              <div className="rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
                {msg}
              </div>
            )}

            {/* threads */}
            <section>
              <h3 className="mb-2 text-sm font-semibold">
                Threads ({selected.thread_ids.length})
              </h3>
              <div className="mb-2 flex gap-2">
                <select
                  value={attachThread}
                  onChange={e => setAttachThread(e.target.value)}
                  className="input flex-1 text-xs"
                >
                  <option value="">Select a thread…</option>
                  {threads
                    .filter(t => !selected.thread_ids.includes(t.id))
                    .map(t => (
                      <option key={t.id} value={t.id}>{t.label}</option>
                    ))}
                </select>
                <button onClick={addThread} disabled={!attachThread} className="btn-primary text-xs disabled:opacity-50">
                  Attach
                </button>
              </div>
              {selected.thread_ids.length === 0 ? (
                <p className="text-xs text-text-muted">No threads attached.</p>
              ) : (
                <ul className="space-y-1">
                  {selected.thread_ids.map(tid => (
                    <li key={tid} className="flex items-center justify-between rounded-lg border border-border bg-bg-secondary px-3 py-1.5">
                      <span className="truncate text-xs text-text-primary">{threadLabel(tid)}</span>
                      <button onClick={() => rmThread(tid)} className="text-xs text-error hover:underline">Detach</button>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {/* knowledge docs */}
            <section>
              <h3 className="mb-2 text-sm font-semibold">
                Knowledge Docs ({selected.knowledge_doc_ids.length})
              </h3>
              <div className="mb-2 flex gap-2">
                <select
                  value={attachDoc}
                  onChange={e => setAttachDoc(e.target.value)}
                  className="input flex-1 text-xs"
                >
                  <option value="">Select a document…</option>
                  {kbDocs
                    .filter(d => !selected.knowledge_doc_ids.includes(d.doc_id))
                    .map(d => (
                      <option key={d.doc_id} value={d.doc_id}>{d.filename}</option>
                    ))}
                </select>
                <button onClick={addDoc} disabled={!attachDoc} className="btn-primary text-xs disabled:opacity-50">
                  Attach
                </button>
              </div>
              {selected.knowledge_doc_ids.length === 0 ? (
                <p className="text-xs text-text-muted">No documents attached.</p>
              ) : (
                <ul className="space-y-1">
                  {selected.knowledge_doc_ids.map(did => (
                    <li key={did} className="flex items-center justify-between rounded-lg border border-border bg-bg-secondary px-3 py-1.5">
                      <span className="truncate text-xs text-text-primary">{docLabel(did)}</span>
                      <button onClick={() => rmDoc(did)} className="text-xs text-error hover:underline">Detach</button>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <button onClick={() => remove(selected.id)} className="mt-auto self-start text-xs text-error hover:underline">
              Delete project
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── List view ──
  return (
    <div className="flex h-full flex-col">
      <PanelHeader title="Research Projects" className="px-6">
        <button
          onClick={() => setShowCreate(s => !s)}
          className="btn-primary px-3 py-1.5 text-xs"
        >
          {showCreate ? 'Cancel' : '+ New'}
        </button>
      </PanelHeader>

      <div className="flex-1 overflow-y-auto p-4">
        {msg && (
          <div className="mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
            {msg}
          </div>
        )}

        {/* create form */}
        {showCreate && (
          <div className="mb-4 rounded-lg border border-accent/20 bg-accent/5 p-4 space-y-2">
            <input
              value={newTitle}
              onChange={e => setNewTitle(e.target.value)}
              placeholder="Project title"
              className="input text-sm"
              autoFocus
            />
            <input
              value={newDesc}
              onChange={e => setNewDesc(e.target.value)}
              placeholder="Short description (optional)"
              className="input text-sm"
            />
            <textarea
              value={newInstr}
              onChange={e => setNewInstr(e.target.value)}
              placeholder="Instructions — system prompt for this research topic"
              className="input h-24 resize-none text-sm"
              spellCheck={false}
            />
            <button onClick={create} disabled={!newTitle.trim()} className="btn-primary text-xs disabled:opacity-50">
              Create
            </button>
          </div>
        )}

        {loading ? (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {[0, 1, 2].map(i => <SkeletonCard key={i} />)}
          </div>
        ) : projects.length === 0 ? (
          <div className="text-sm text-text-muted">
            No research projects yet. Click "+ New" to create one.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {projects.map(p => (
              <button
                key={p.id}
                onClick={() => setSelected(p)}
                className="flex flex-col rounded-xl border border-border bg-bg-secondary p-4 text-left hover:bg-bg-tertiary transition-colors"
              >
                <div className="text-sm font-semibold text-text-primary">{p.title}</div>
                {p.description && (
                  <div className="mt-1 line-clamp-2 text-xs text-text-secondary">{p.description}</div>
                )}
                <div className="mt-2 flex gap-3 text-[10px] text-text-muted">
                  <span>📄 {p.knowledge_doc_ids.length} docs</span>
                  <span>💬 {p.thread_ids.length} threads</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
