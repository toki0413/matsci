import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useScrollMemory } from '../../hooks/useScrollMemory';
import { ChevronDown, Brain, Pencil, Check, X, Layers } from 'lucide-react';
import { PanelHeader } from '../settings-shared';
import EmptyState from '../EmptyState';
import { SkeletonText } from '../Skeleton';
import type { MemoryEntry, MemoryLayers, MemoryStats } from '../../types/domain';

export interface MemoryPanelProps {
  memories: MemoryEntry[];
  memoriesLoading?: boolean;
  memoryHasMore?: boolean;
  loadMoreMemory?: () => void;
  memoryStats: MemoryStats | null;
  memorySearch: string;
  memoryFilter: { category: string; tier: string };
  memoryForm: { content: string; category: string; tags: string; importance: number; tier: string };
  memoryMsg: string;
  memoryView: 'browse' | 'add' | 'layers';
  memoryLayers: MemoryLayers | null;
  memoryLayersLoading?: boolean;
  setMemorySearch: (v: string) => void;
  setMemoryFilter: (v: { category: string; tier: string }) => void;
  setMemoryForm: (v: { content: string; category: string; tags: string; importance: number; tier: string }) => void;
  setMemoryView: (v: 'browse' | 'add' | 'layers') => void;
  loadMemory: () => void;
  loadMemoryStats: () => void;
  searchMemory: () => void;
  createMemory: () => void;
  deleteMemory: (id: string) => void;
  updateMemory: (id: string, patch: { content?: string; importance?: number; tags?: string[] }) => void;
  promoteMemory: (id: string) => void;
  pruneMemory: () => void;
  syncMemoryMd: () => void;
  loadMemoryLayers: () => void;
}

export function MemoryPanel({
  memories,
  memoriesLoading,
  memoryHasMore,
  loadMoreMemory,
  memoryStats,
  memorySearch,
  memoryFilter,
  memoryForm,
  memoryMsg,
  memoryView,
  memoryLayers,
  memoryLayersLoading,
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
  loadMemoryLayers,
}: MemoryPanelProps) {
  const { t } = useTranslation();
  const [editingId, setEditingId] = useState<string | null>(null);
  const scrollRef = useScrollMemory('memory-list');
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
      <PanelHeader title={t('memory.title')} className="mem-header px-6">
        <div className="mem-header-actions flex items-center gap-2">
          <button
            onClick={() => {
              if (memoryView === 'layers') {
                setMemoryView('browse');
              } else {
                setMemoryView('layers');
                loadMemoryLayers();
              }
            }}
            className={`btn-secondary px-3 py-1.5 text-xs ${memoryView === 'layers' ? 'ring-1 ring-accent' : ''}`}
            title="Toggle 4-layer memory view"
          >
            <Layers size={12} className="inline mr-1" />Layers
          </button>
          <button onClick={syncMemoryMd} className="btn-secondary px-3 py-1.5 text-xs">{t('memory.sync')}</button>
          <button onClick={pruneMemory} className="btn-secondary px-3 py-1.5 text-xs">{t('memory.prune')}</button>
          <button onClick={loadMemory} className="btn-secondary px-3 py-1.5 text-xs">{t('memory.refresh')}</button>
        </div>
      </PanelHeader>
      {memoryView === 'layers' ? (
        <MemoryLayersView
          layers={memoryLayers}
          loading={!!memoryLayersLoading}
          onRefresh={loadMemoryLayers}
        />
      ) : (
      <div className="flex flex-1 overflow-hidden">
        <div className="w-80 overflow-y-auto border-r border-border bg-bg-secondary p-4">
          <div className="card mem-stats mb-4">
            <h3 className="text-sm font-semibold">{t('memory.stats')}</h3>
            <div className="mt-2">
              <div className="mem-stats-row"><span className="text-text-muted">{t('memory.total')}</span><span>{memoryStats?.longterm_entries ?? "\u2014"}</span></div>
              <div className="mem-stats-row"><span className="text-text-muted">{t('memory.short')}</span><span>{memoryStats?.tier_counts?.short ?? 0}</span></div>
              <div className="mem-stats-row"><span className="text-text-muted">{t('memory.mid')}</span><span>{memoryStats?.tier_counts?.mid ?? 0}</span></div>
              <div className="mem-stats-row"><span className="text-text-muted">{t('memory.long')}</span><span>{memoryStats?.tier_counts?.long ?? 0}</span></div>
            </div>
          </div>
          <div className="card mb-4">
            <button
              onClick={() => setMemoryView(memoryView === "add" ? "browse" : "add")}
              className="flex w-full items-center justify-between text-left"
            >
              <h3 className="text-sm font-semibold">
                {memoryView === "add" ? t('memory.add') : t('memory.addBtn')}
              </h3>
              <ChevronDown size={14} className={`text-text-muted transition-transform duration-150 ${memoryView === "add" ? "rotate-0" : "-rotate-90"}`} />
            </button>
            {memoryView === "add" && (
              <div className="mt-3">
                <textarea
                  className="input-field mb-2 min-h-[80px] text-xs"
                  placeholder={t('memory.contentPh')}
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
                  <span className="text-text-muted">{t('memory.importance')}</span>
                  <input type="range" min={0} max={1} step={0.05} value={memoryForm.importance} onChange={(e) => setMemoryForm({ ...memoryForm, importance: parseFloat(e.target.value) })} />
                  <span>{memoryForm.importance.toFixed(2)}</span>
                </div>
                <button onClick={createMemory} className="btn-primary w-full py-1.5 text-xs" disabled={!memoryForm.content.trim()}>
                  {t('memory.remember')}
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
              placeholder={t('memory.searchPh')}
              value={memorySearch}
              onChange={(e) => setMemorySearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && searchMemory()}
            />
            <button onClick={searchMemory} className="btn-primary px-3 py-1.5 text-xs">{t('memory.search')}</button>
            <select className="input-field text-xs" value={memoryFilter.category} onChange={(e) => setMemoryFilter({ ...memoryFilter, category: e.target.value })}>
              <option value="">{t('memory.allCats')}</option>
              <option value="fact">fact</option>
              <option value="insight">insight</option>
              <option value="conversation">conversation</option>
              <option value="calculation">calculation</option>
              <option value="error">error</option>
              <option value="episode">episode</option>
            </select>
            <select className="input-field text-xs" value={memoryFilter.tier} onChange={(e) => setMemoryFilter({ ...memoryFilter, tier: e.target.value })}>
              <option value="">{t('memory.allTiers')}</option>
              <option value="short">short</option>
              <option value="mid">mid</option>
              <option value="long">long</option>
            </select>
          </div>
          <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-2">
            {memoriesLoading ? (
              [0, 1, 2].map(i => (
                <div key={i} className="card">
                  <SkeletonText lines={3} />
                </div>
              ))
            ) : memories.length === 0 && (
              <EmptyState
                icon={Brain}
                title={memorySearch.trim() ? t('memory.noMatch') : t('memory.noMemories')}
                subtitle={memorySearch.trim()
                  ? t('memory.noMatchHint')
                  : t('memory.emptyHint')
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
                          {t('memory.recent')}
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
                            <Check size={12} className="inline mr-1" />{t('memory.save')}
                          </button>
                          <button onClick={() => setEditingId(null)} className="btn-secondary px-2 py-1 text-xs">
                            <X size={12} className="inline mr-1" />{t('memory.cancel')}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p className="mt-1 whitespace-pre-wrap text-sm">{m.content}</p>
                    )}
                    <p className="mt-1 text-xs text-text-muted">{t('memory.tags')} {m.tags || "\u2014"} · {t('memory.source')} {m.source || "\u2014"}</p>
                    <p className="text-xs text-text-muted">{t('memory.expires')} {m.expires_at ? new Date(m.expires_at).toLocaleString() : t('memory.never')} · {t('memory.accessed')} {m.access_count ?? 0}</p>
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
            {memoryHasMore && (
              <button
                onClick={loadMoreMemory}
                className="w-full rounded-lg border border-border py-2 text-xs text-text-secondary hover:bg-bg-tertiary transition-colors"
              >
                {t('memory.loadMore')}
              </button>
            )}
          </div>
        </div>
      </div>
      )}
    </div>
  );
}

// ── 4 层 Memory 视图 (WM/EM/SM/PM) ────────────────────────────────────
// 单文件内组件, 不拆 MemoryLayersPanel.tsx — 复用 panel shell, ponytail.
// ponytail: 不引新的 lucide 图标, 不引新的 i18n key, 字面量 label 够用.

interface MemoryLayersViewProps {
  layers: MemoryLayers | null;
  loading: boolean;
  onRefresh: () => void;
}

function MemoryLayersView({ layers, loading, onRefresh }: MemoryLayersViewProps) {
  if (loading && !layers) {
    return (
      <div className="flex-1 overflow-y-auto p-6 space-y-3">
        {[0, 1, 2, 3].map(i => (
          <div key={i} className="card"><SkeletonText lines={4} /></div>
        ))}
      </div>
    );
  }

  if (!layers) {
    return (
      <div className="flex-1 overflow-y-auto p-6">
        <EmptyState
          icon={Layers}
          title="No layer data"
          subtitle="Click refresh to load the 4-layer memory snapshot."
        />
        <div className="mt-4 text-center">
          <button onClick={onRefresh} className="btn-primary px-4 py-2 text-xs">Refresh</button>
        </div>
      </div>
    );
  }

  const wm = layers.wm;
  const em = layers.em;
  const sm = layers.sm;
  const pm = layers.pm;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-text-muted">
          4-layer memory snapshot · each layer independent · failure isolated
        </p>
        <button onClick={onRefresh} className="btn-secondary px-3 py-1.5 text-xs" disabled={loading}>
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* WM — Working Memory: 进程内 SessionContext */}
        <LayerCard
          title="WM · Working Memory"
          subtitle="in-process SessionContext"
          available={wm?.available !== false}
          error={wm?.error}
        >
          {wm && (
            <>
              <LayerRow label="Token used" value={wm.token_used != null ? `${wm.token_used} / ${wm.token_budget ?? '—'}` : '—'} />
              <LayerRow label="Messages" value={String(wm.messages_count ?? 0)} />
              <LayerRow label="Tool calls" value={String(wm.tool_calls_count ?? 0)} />
              <LayerRow label="Summaries" value={String(wm.summaries_count ?? 0)} />
              <LayerRow label="Last summarize" value={wm.last_summarize_at ? new Date(wm.last_summarize_at).toLocaleString() : 'never'} />
              <LayerRow label="Extreme dispatch" value={wm.extreme_dispatch ? 'ON' : 'off'} highlight={wm.extreme_dispatch} />
            </>
          )}
        </LayerCard>

        {/* EM — Episodic Memory: SQLite+FTS5+embedding */}
        <LayerCard
          title="EM · Episodic Memory"
          subtitle="SQLite + FTS5 + embedding"
          available={em?.available !== false}
          error={em?.error}
        >
          {em && (
            <>
              <LayerRow label="Total entries" value={String(em.total_entries ?? 0)} />
              <LayerRow label="Tier short" value={String(em.tier_counts?.short ?? 0)} />
              <LayerRow label="Tier mid" value={String(em.tier_counts?.mid ?? 0)} />
              <LayerRow label="Tier long" value={String(em.tier_counts?.long ?? 0)} />
              <div className="mt-3">
                <p className="text-xs font-semibold text-text-muted mb-1">Recent episodes</p>
                {(em.recent_episodes || []).length === 0 ? (
                  <p className="text-xs text-text-muted italic">— none —</p>
                ) : (
                  <ul className="space-y-1">
                    {em.recent_episodes!.slice(0, 5).map((ep) => (
                      <li key={ep.id} className="text-xs bg-bg-tertiary rounded p-2">
                        <div className="flex items-center gap-2 mb-1">
                          {ep.source && <span className="mem-badge mem-badge--category">{ep.source}</span>}
                          {ep.importance != null && (
                            <span className="text-text-muted">imp {ep.importance.toFixed(2)}</span>
                          )}
                        </div>
                        <p className="text-text-primary line-clamp-2">{ep.content}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </LayerCard>

        {/* SM — Semantic Memory: KB chunks + KG nodes */}
        <LayerCard
          title="SM · Semantic Memory"
          subtitle="KB chunks + KG entities"
          available={sm?.available !== false}
          error={sm?.error}
        >
          {sm && (
            <>
              <LayerRow label="KB chunks" value={String(sm.kb_chunks ?? 0)} />
              <LayerRow label="KG nodes" value={String(sm.kg_nodes ?? 0)} />
              <LayerRow label="KG edges" value={String(sm.kg_edges ?? 0)} />
              {sm.kg_node_types && Object.keys(sm.kg_node_types).length > 0 && (
                <div className="mt-1">
                  <p className="text-xs text-text-muted mb-1">Node types:</p>
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(sm.kg_node_types).map(([t, n]) => (
                      <span key={t} className="mem-badge mem-badge--category">{t} · {n}</span>
                    ))}
                  </div>
                </div>
              )}
              <div className="mt-3">
                <p className="text-xs font-semibold text-text-muted mb-1">Recent trajectory patterns</p>
                {(sm.recent_patterns || []).length === 0 ? (
                  <p className="text-xs text-text-muted italic">— none —</p>
                ) : (
                  <ul className="space-y-1">
                    {sm.recent_patterns!.map((p) => (
                      <li key={p.doc_id} className="text-xs bg-bg-tertiary rounded p-2">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="mem-badge mem-badge--category">{p.task_pattern || 'pattern'}</span>
                          <span className="text-text-muted">conf {p.confidence.toFixed(2)}</span>
                          {p.run_id && <span className="text-text-muted text-[10px]">{p.run_id}</span>}
                        </div>
                        {p.objective && <p className="text-text-primary line-clamp-1">{p.objective}</p>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </LayerCard>

        {/* PM — Procedural Memory: stable_principles + top patterns by confidence */}
        <LayerCard
          title="PM · Procedural Memory"
          subtitle="stable_principles JSONL + pattern confidence"
          available={pm?.available !== false}
          error={pm?.error}
        >
          {pm && (
            <>
              <LayerRow label="Stable principles" value={String(pm.stable_principles_count ?? 0)} />
              {pm.stable_principles_preview && pm.stable_principles_preview.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs font-semibold text-text-muted mb-1">Top principles</p>
                  <ul className="space-y-1">
                    {pm.stable_principles_preview.map((p, i) => (
                      <li key={i} className="text-xs bg-bg-tertiary rounded p-2 line-clamp-2">{p}</li>
                    ))}
                  </ul>
                </div>
              )}
              <div className="mt-3">
                <p className="text-xs font-semibold text-text-muted mb-1">Top patterns by confidence</p>
                {(pm.top_patterns_by_confidence || []).length === 0 ? (
                  <p className="text-xs text-text-muted italic">— none —</p>
                ) : (
                  <ul className="space-y-1">
                    {pm.top_patterns_by_confidence!.map((p) => (
                      <li key={p.doc_id} className="text-xs bg-bg-tertiary rounded p-2">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="mem-badge mem-badge--category">{p.task_pattern || 'pattern'}</span>
                          <span className="text-text-muted">conf {p.confidence.toFixed(2)}</span>
                        </div>
                        {p.objective && <p className="text-text-primary line-clamp-1">{p.objective}</p>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </LayerCard>
      </div>
    </div>
  );
}

interface LayerCardProps {
  title: string;
  subtitle: string;
  available: boolean;
  error?: string;
  children?: React.ReactNode;
}

function LayerCard({ title, subtitle, available, error, children }: LayerCardProps) {
  return (
    <div className={`card ${available ? '' : 'opacity-50'}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          <p className="text-[11px] text-text-muted">{subtitle}</p>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full ${available ? 'bg-accent/20 text-accent' : 'bg-red-500/20 text-red-400'}`}>
          {available ? 'available' : 'offline'}
        </span>
      </div>
      {error && (
        <p className="text-xs text-red-400 mb-2 break-words">{error}</p>
      )}
      {available && children}
    </div>
  );
}

function LayerRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="mem-stats-row">
      <span className="text-text-muted text-xs">{label}</span>
      <span className={`text-xs ${highlight ? 'text-accent font-semibold' : ''}`}>{value}</span>
    </div>
  );
}
