import { useState } from 'react';
import { ChevronDown, Brain, Pencil, Check, X } from 'lucide-react';
import { PanelHeader } from '../settings-shared';
import EmptyState from '../EmptyState';
import type { MemoryEntry, MemoryStats } from '../../types/domain';

export interface MemoryPanelProps {
  memories: MemoryEntry[];
  memoriesLoading?: boolean;
  memoryStats: MemoryStats | null;
  memorySearch: string;
  memoryFilter: { category: string; tier: string };
  memoryForm: { content: string; category: string; tags: string; importance: number; tier: string };
  memoryMsg: string;
  memoryView: 'browse' | 'add';
  setMemorySearch: (v: string) => void;
  setMemoryFilter: (v: { category: string; tier: string }) => void;
  setMemoryForm: (v: { content: string; category: string; tags: string; importance: number; tier: string }) => void;
  setMemoryView: (v: 'browse' | 'add') => void;
  loadMemory: () => void;
  loadMemoryStats: () => void;
  searchMemory: () => void;
  createMemory: () => void;
  deleteMemory: (id: string) => void;
  updateMemory: (id: string, patch: { content?: string; importance?: number; tags?: string[] }) => void;
  promoteMemory: (id: string) => void;
  pruneMemory: () => void;
  syncMemoryMd: () => void;
}

export function MemoryPanel({
  memories,
  memoriesLoading,
  memoryStats,
  memorySearch,
  memoryFilter,
  memoryForm,
  memoryMsg,
  memoryView,
  setMemorySearch,
  setMemoryFilter,
  setMemoryForm,
  setMemoryView,
  loadMemory,
  searchMemory,
  createMemory,
  deleteMemory,
  updateMemory,
  promoteMemory,
  pruneMemory,
  syncMemoryMd,
}: MemoryPanelProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");

  const startEdit = (m: MemoryEntry) => {
    setEditingId(m.id);
    setEditContent(m.content);
  };

  const saveEdit = (id: string) => {
    updateMemory(id, { content: editContent });
    setEditingId(null);
  };

  // Sort by last_accessed descending — most recent on top
  const sortedMemories = [...memories].sort((a, b) => {
    const da = new Date(a.last_accessed || a.created_at).getTime();
    const db = new Date(b.last_accessed || b.created_at).getTime();
    return db - da;
  });

  const isRecent = (m: MemoryEntry) => {
    const last = new Date(m.last_accessed || m.created_at).getTime();
    return Date.now() - last < 3600_000; // within 1 hour
  };
  return (
    <div data-component="memory-panel" className="mem-panel flex h-full flex-col">
      <PanelHeader title="Memory" className="mem-header px-6">
        <div className="mem-header-actions flex items-center gap-2">
          <button onClick={syncMemoryMd} className="btn-secondary px-3 py-1.5 text-xs">Sync MEMORY.md</button>
          <button onClick={pruneMemory} className="btn-secondary px-3 py-1.5 text-xs">Prune</button>
          <button onClick={loadMemory} className="btn-secondary px-3 py-1.5 text-xs">Refresh</button>
        </div>
      </PanelHeader>
      <div className="flex flex-1 overflow-hidden">
        <div className="w-80 overflow-y-auto border-r border-border bg-bg-secondary p-4">
          <div className="card mem-stats mb-4">
            <h3 className="text-sm font-semibold">Stats</h3>
            <div className="mt-2">
              <div className="mem-stats-row"><span className="text-text-muted">Total</span><span>{memoryStats?.longterm_entries ?? "\u2014"}</span></div>
              <div className="mem-stats-row"><span className="text-text-muted">Short</span><span>{memoryStats?.tier_counts?.short ?? 0}</span></div>
              <div className="mem-stats-row"><span className="text-text-muted">Mid</span><span>{memoryStats?.tier_counts?.mid ?? 0}</span></div>
              <div className="mem-stats-row"><span className="text-text-muted">Long</span><span>{memoryStats?.tier_counts?.long ?? 0}</span></div>
            </div>
          </div>
          <div className="card mb-4">
            <button
              onClick={() => setMemoryView(memoryView === "add" ? "browse" : "add")}
              className="flex w-full items-center justify-between text-left"
            >
              <h3 className="text-sm font-semibold">
                {memoryView === "add" ? "Add Memory" : "+ Add memory"}
              </h3>
              <ChevronDown size={14} className={`text-text-muted transition-transform duration-150 ${memoryView === "add" ? "rotate-0" : "-rotate-90"}`} />
            </button>
            {memoryView === "add" && (
              <div className="mt-3">
                <textarea
                  className="input-field mb-2 min-h-[80px] text-xs"
                  placeholder="Content..."
                  value={memoryForm.content}
                  onChange={(e) => setMemoryForm({ ...memoryForm, content: e.target.value })}
                />
                <div className="mb-2 grid grid-cols-2 gap-2">
                  <select className="input-field text-xs" value={memoryForm.category} onChange={(e) => setMemoryForm({ ...memoryForm, category: e.target.value })}>
                    <option value="fact">fact</option>
                    <option value="insight">insight</option>
                    <option value="conversation">conversation</option>
                    <option value="calculation">calculation</option>
                    <option value="error">error</option>
                    <option value="episode">episode</option>
                  </select>
                  <select className="input-field text-xs" value={memoryForm.tier} onChange={(e) => setMemoryForm({ ...memoryForm, tier: e.target.value })}>
                    <option value="short">short (6h)</option>
                    <option value="mid">mid (7d)</option>
                    <option value="long">long (perm)</option>
                  </select>
                </div>
                <input
                  className="input-field mb-2 text-xs"
                  placeholder="tags, comma separated"
                  value={memoryForm.tags}
                  onChange={(e) => setMemoryForm({ ...memoryForm, tags: e.target.value })}
                />
                <div className="mb-2 flex items-center gap-2 text-xs">
                  <span className="text-text-muted">Importance</span>
                  <input type="range" min={0} max={1} step={0.05} value={memoryForm.importance} onChange={(e) => setMemoryForm({ ...memoryForm, importance: parseFloat(e.target.value) })} />
                  <span>{memoryForm.importance.toFixed(2)}</span>
                </div>
                <button onClick={createMemory} className="btn-primary w-full py-1.5 text-xs" disabled={!memoryForm.content.trim()}>
                  Remember
                </button>
              </div>
            )}
          </div>
          {memoryMsg && <p className="text-xs text-text-secondary">{memoryMsg}</p>}
        </div>
        <div className="flex flex-1 flex-col overflow-hidden bg-bg-primary p-4">
          <div className="mem-search-bar items-center gap-2">
            <input
              className="input-field flex-1 text-xs"
              placeholder="Search memory..."
              value={memorySearch}
              onChange={(e) => setMemorySearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && searchMemory()}
            />
            <button onClick={searchMemory} className="btn-primary px-3 py-1.5 text-xs">Search</button>
            <select className="input-field text-xs" value={memoryFilter.category} onChange={(e) => setMemoryFilter({ ...memoryFilter, category: e.target.value })}>
              <option value="">all categories</option>
              <option value="fact">fact</option>
              <option value="insight">insight</option>
              <option value="conversation">conversation</option>
              <option value="calculation">calculation</option>
              <option value="error">error</option>
              <option value="episode">episode</option>
            </select>
            <select className="input-field text-xs" value={memoryFilter.tier} onChange={(e) => setMemoryFilter({ ...memoryFilter, tier: e.target.value })}>
              <option value="">all tiers</option>
              <option value="short">short</option>
              <option value="mid">mid</option>
              <option value="long">long</option>
            </select>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2">
            {memoriesLoading ? (
              <div className="flex items-center justify-center py-16">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-border border-t-accent" />
              </div>
            ) : memories.length === 0 && (
              <EmptyState
                icon={Brain}
                title={memorySearch.trim() ? "No matching memories" : "No memories yet"}
                subtitle={memorySearch.trim()
                  ? "No memories match your search. Try different keywords or clear the filter."
                  : "Memories help Huginn remember context across conversations. Add your first memory using the panel on the left."
                }
              />
            )}
            {sortedMemories.map((m) => (
              <div key={m.id} className={`card mem-entry ${isRecent(m) ? 'ring-1 ring-accent/30' : ''}`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 text-xs">
                      <span className={`mem-badge mem-badge--tier-${m.tier}`}>{m.tier}</span>
                      <span className="mem-badge mem-badge--category">{m.category}</span>
                      <span className="text-text-muted">importance {m.importance}</span>
                      {isRecent(m) && (
                        <span className="rounded-full bg-accent/20 px-1.5 py-0.5 text-[10px] font-medium text-accent">
                          recent
                        </span>
                      )}
                    </div>
                    {editingId === m.id ? (
                      <div className="mt-1">
                        <textarea
                          className="input-field min-h-[60px] w-full text-sm"
                          value={editContent}
                          onChange={(e) => setEditContent(e.target.value)}
                          autoFocus
                        />
                        <div className="mt-1 flex gap-1">
                          <button onClick={() => saveEdit(m.id)} className="btn-primary px-2 py-1 text-xs">
                            <Check size={12} className="inline mr-1" />Save
                          </button>
                          <button onClick={() => setEditingId(null)} className="btn-secondary px-2 py-1 text-xs">
                            <X size={12} className="inline mr-1" />Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p className="mt-1 whitespace-pre-wrap text-sm">{m.content}</p>
                    )}
                    <p className="mt-1 text-xs text-text-muted">tags: {m.tags || "\u2014"} · source: {m.source || "\u2014"}</p>
                    <p className="text-xs text-text-muted">expires: {m.expires_at ? new Date(m.expires_at).toLocaleString() : "never"} · accessed {m.access_count ?? 0}</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    {editingId !== m.id && (
                      <button onClick={() => startEdit(m)} className="btn-secondary px-2 py-1 text-xs" title="Edit" aria-label="Edit memory">
                        <Pencil size={12} />
                      </button>
                    )}
                    {m.tier !== "long" && (
                      <button onClick={() => promoteMemory(m.id)} className="btn-secondary px-2 py-1 text-xs" title="Promote to long" aria-label="Promote memory to long-term">
                        ⬆
                      </button>
                    )}
                    <button onClick={() => deleteMemory(m.id)} className="btn-secondary px-2 py-1 text-xs" title="Delete" aria-label="Delete memory">
                      🗑
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
