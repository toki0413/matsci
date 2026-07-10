import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PanelHeader } from '../settings-shared';
import type { AppConfig } from '../../types/domain';

interface KbDoc {
  doc_id: string;
  filename: string;
}

interface KbChunk {
  text: string;
  distance?: number;
  metadata?: { filename?: string };
}

interface KnowledgePanelProps {
  config: AppConfig;
  setConfig: (c: AppConfig) => void;
  saveConfig: (c: AppConfig) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  parseFileInputRef: React.RefObject<HTMLInputElement>;
  parseLoading: boolean;
  uploadPct?: number;
  kbLoading?: boolean;
  kbMsg: string;
  kbDocs: KbDoc[];
  kbAvailable: boolean;
  kbQuery: string;
  kbChunks: KbChunk[];
  setKbQuery: (v: string) => void;
  uploadKnowledge: (file: File) => void;
  parseDocument: (file: File) => void;
  loadDocumentGraph: (docId: string) => void;
  deleteKnowledge: (docId: string) => void;
  queryKnowledge: () => void;
  ingestUrl: (url: string) => void;
  loadProvenanceDag: () => Promise<any>;
}

type ViewMode = 'concise' | 'detailed' | 'research';

// rough token estimate — words vary, but chars/4 tracks BPE closely enough for a counter
const estimateTokens = (text: string) => Math.ceil((text?.length ?? 0) / 4);

// wrap query terms in <mark> so users see why a chunk was retrieved.
// naive regex split: fine for short natural-language queries, not a tokenizer.
function highlightTerms(text: string, query: string) {
  const terms = query.trim().split(/\s+/).filter((t) => t.length > 2);
  if (!terms.length) return text;
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const splitRe = new RegExp(`(${escaped.join('|')})`, 'gi');
  const testRe = new RegExp(`^(?:${escaped.join('|')})$`, 'i');
  return text.split(splitRe).map((part, i) =>
    part && testRe.test(part) ? (
      <mark key={i} className="rounded bg-accent/30 px-0.5 text-text-primary">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

export function KnowledgePanel({
  config, setConfig, saveConfig,
  fileInputRef, parseFileInputRef, parseLoading, uploadPct, kbLoading,
  kbMsg, kbDocs, kbAvailable, kbQuery, kbChunks, setKbQuery,
  uploadKnowledge, parseDocument, loadDocumentGraph, deleteKnowledge, queryKnowledge,
  ingestUrl, loadProvenanceDag,
}: KnowledgePanelProps) {
  const { t } = useTranslation();
  const [viewMode, setViewMode] = useState<ViewMode>('detailed');
  const [searching, setSearching] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [showDag, setShowDag] = useState(false);
  const [dagData, setDagData] = useState<{ nodes: any[]; edges: any[] } | null>(null);
  // per-chunk toggles — tracks which cards have their citation / thinking / body expanded
  const [openCitations, setOpenCitations] = useState<Set<number>>(new Set());
  const [openThinking, setOpenThinking] = useState<Set<number>>(new Set());
  const [openText, setOpenText] = useState<Set<number>>(new Set());

  // wrap the parent handler so we can drive the "searching" animation locally
  const runQuery = async () => {
    setSearching(true);
    try {
      await queryKnowledge();
    } finally {
      setSearching(false);
    }
  };

  const handleIngestUrl = async () => {
    if (!urlInput.trim()) return;
    await ingestUrl(urlInput);
    setUrlInput('');
  };

  const toggleDag = async () => {
    if (!showDag && !dagData) {
      const resp = await loadProvenanceDag();
      if (resp?.success) setDagData(resp.data);
    }
    setShowDag(!showDag);
  };

  const toggle = (set: Set<number>, i: number, setter: (s: Set<number>) => void) => {
    const next = new Set(set);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    setter(next);
  };

  const switchView = (mode: ViewMode) => {
    setViewMode(mode);
    // research mode unfolds the retrieval trace for every chunk by default
    setOpenThinking(mode === 'research' ? new Set(kbChunks.map((_, i) => i)) : new Set());
  };

  const estTokens = kbChunks.reduce((n, c) => n + estimateTokens(c.text), 0);
  // concise view trims to the top 3 hits
  const visibleChunks = viewMode === 'concise' ? kbChunks.slice(0, 3) : kbChunks;
  const TEXT_LIMIT = 240;

  return (
    <div data-component="knowledge-panel" className="kb-panel flex h-full flex-col">
      <PanelHeader title="Knowledge Base" className="kb-header px-6">
        <label className="kb-rag-toggle flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={config.rag_enabled}
            onChange={(e) => {
              const next = { ...config, rag_enabled: e.target.checked };
              setConfig(next);
              saveConfig(next);
            }}
            className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
          />
          <span className="text-xs text-text-secondary">Use RAG in chat</span>
        </label>
      </PanelHeader>
      <div className="flex flex-1 overflow-hidden">
        {/* Upload / docs */}
        <aside className="flex w-80 flex-col border-r border-border bg-bg-secondary p-4">
          <div className="kb-upload mb-4 rounded-lg border border-dashed border-border bg-bg-tertiary p-4 text-center">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".txt,.md,.pdf,.py,.json,.yaml,.yml"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) uploadKnowledge(file);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-primary w-full text-xs"
            >
              📤 Upload document
            </button>
            <p className="mt-2 text-xs text-text-muted">
              Supports TXT, MD, PDF, code files
            </p>
          </div>

          {/* Deep PDF parsing — 6-stage document analysis pipeline */}
          <div className="kb-deep-parse mb-4 rounded-lg border border-accent/20 bg-accent/5 p-3">
            <input
              ref={parseFileInputRef}
              type="file"
              className="hidden"
              accept=".pdf"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) parseDocument(file);
                if (parseFileInputRef.current) parseFileInputRef.current.value = "";
              }}
            />
            <button
              onClick={() => parseFileInputRef.current?.click()}
              disabled={parseLoading}
              className="w-full rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-xs font-medium text-accent hover:bg-accent/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {parseLoading ? "⏳ Parsing…" : "📊 Deep PDF Parse"}
            </button>
            <p className="mt-1.5 text-xs text-text-muted">
              6-stage: extract → figures → graph → relations → validate → assemble
            </p>
          </div>

          {/* URL ingestion — Metaso-style multi-source input */}
          <div className="kb-url-input mb-4 rounded-lg border border-border bg-bg-tertiary p-3">
            <div className="mb-1.5 text-xs font-medium text-text-secondary">Add from URL</div>
            <div className="flex gap-2">
              <input
                type="url"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://..."
                className="input flex-1 text-xs"
                onKeyDown={(e) => e.key === "Enter" && handleIngestUrl()}
              />
              <button
                onClick={handleIngestUrl}
                disabled={!urlInput.trim()}
                className="btn-primary text-xs disabled:opacity-50"
              >
                Fetch
              </button>
            </div>
          </div>

          {/* Provenance DAG toggle */}
          <button
            onClick={toggleDag}
            className="mb-3 flex w-full items-center justify-between rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-secondary hover:text-text-primary transition-colors"
          >
            <span>Provenance DAG</span>
            <span className="text-text-muted">{showDag ? '▾' : '▸'}</span>
          </button>
          {showDag && dagData && (
            <ProvenanceDagView nodes={dagData.nodes} edges={dagData.edges} />
          )}
          {showDag && !dagData && (
            <div className="mb-3 text-xs text-text-muted">No provenance data yet</div>
          )}

          {kbMsg && (
            <div className="kb-status mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
              {kbMsg}
              {uploadPct !== undefined && uploadPct > 0 && (
                <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-bg-secondary">
                  <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${uploadPct}%` }} />
                </div>
              )}
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {kbLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-border border-t-accent" />
              </div>
            ) : (
            <>
            <div className="mb-2 text-xs font-medium text-text-secondary">
              Documents ({kbDocs.length})
            </div>
            {!kbAvailable && (
              <div className="text-xs text-text-muted">
                Knowledge base backend is not available. Install chromadb and sentence-transformers.
              </div>
            )}
            {kbDocs.map((doc) => (
              <div
                key={doc.doc_id}
                className="kb-doc-item mb-2 flex items-center justify-between rounded-lg border border-border bg-bg-tertiary p-2"
              >
                <span className="truncate text-xs text-text-primary">{doc.filename}</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => loadDocumentGraph(doc.doc_id)}
                    className="text-xs text-accent hover:underline"
                    title="View document structure graph"
                    aria-label="View document structure graph"
                  >
                    📊
                  </button>
                  <button
                    onClick={() => deleteKnowledge(doc.doc_id)}
                    className="text-xs text-error hover:underline"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
            </>
            )}
          </div>
        </aside>

        {/* Query tester — Metaso-style "transparent brain" retrieval view */}
        <div className="kb-query-area flex flex-1 flex-col bg-bg-primary p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold">{t('kb.testRetrieval')}</h3>
            {/* three search-depth modes, mirrors Metaso's concise / detailed / research */}
            <div className="flex rounded-lg border border-border bg-bg-secondary p-0.5">
              {(['concise', 'detailed', 'research'] as ViewMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => switchView(m)}
                  className={
                    'rounded-md px-2.5 py-1 text-xs font-medium transition-colors ' +
                    (viewMode === m
                      ? 'bg-accent text-white'
                      : 'text-text-secondary hover:text-text-primary')
                  }
                >
                  {t(`kb.view.${m}`)}
                </button>
              ))}
            </div>
          </div>

          <div className="mb-3 flex gap-2">
            <input
              type="text"
              value={kbQuery}
              onChange={(e) => setKbQuery(e.target.value)}
              placeholder={t('kb.queryPlaceholder')}
              className="input flex-1"
              onKeyDown={(e) => e.key === "Enter" && runQuery()}
            />
            <button onClick={runQuery} disabled={searching} className="btn-primary disabled:opacity-50">
              {t('kb.search')}
            </button>
          </div>

          {/* live status counter — Metaso left-corner style */}
          {kbChunks.length > 0 && (
            <div className="mb-3 flex items-center gap-3 rounded-lg border border-border bg-bg-secondary px-3 py-1.5 text-xs text-text-secondary">
              <span>📊 {t('kb.sources', { n: kbChunks.length })}</span>
              <span className="text-border">|</span>
              <span>🔤 ~{t('kb.tokens', { n: estTokens })}</span>
            </div>
          )}

          <div className="flex-1 overflow-y-auto space-y-3">
            {searching ? (
              <div className="flex items-center gap-2 text-sm text-text-secondary">
                <span className="animate-pulse">🔍</span>
                <span className="animate-pulse">{t('kb.searching')}</span>
              </div>
            ) : kbChunks.length === 0 ? (
              <div className="text-xs text-text-muted">{t('kb.query.empty')}</div>
            ) : (
              visibleChunks.map((chunk, i) => {
                const filename = chunk.metadata?.filename ?? 'unknown';
                const distance = chunk.distance;
                const tokens = estimateTokens(chunk.text);
                const citationOpen = openCitations.has(i);
                const thinkingOpen = openThinking.has(i);
                // concise mode collapses long bodies until the user expands them
                const showFull = openText.has(i) || viewMode !== 'concise';
                const tooLong = chunk.text.length > TEXT_LIMIT;
                const shownText = showFull
                  ? chunk.text
                  : chunk.text.slice(0, TEXT_LIMIT) + '…';

                return (
                  <div key={i} className="kb-chunk rounded-lg border border-border bg-bg-secondary p-3">
                    {/* header: clickable citation marker + source + distance badge */}
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => toggle(openCitations, i, setOpenCitations)}
                          className="inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded bg-accent/20 px-1 text-xs font-bold text-accent hover:bg-accent/30"
                          title={t('kb.source')}
                        >
                          [{i + 1}]
                        </button>
                        <span className="truncate text-xs text-text-primary">{filename}</span>
                      </div>
                      {distance != null && (
                        <span className="shrink-0 rounded-full bg-bg-tertiary px-2 py-0.5 text-[10px] font-medium text-text-secondary">
                          {t('kb.distance')}: {distance.toFixed(3)}
                        </span>
                      )}
                    </div>

                    {/* expanded citation chain — source / rank / distance / token cost */}
                    {citationOpen && (
                      <div className="mt-2 space-y-0.5 rounded-md bg-bg-tertiary p-2 text-[11px] text-text-secondary">
                        <div>📚 {t('kb.source')}: {filename}</div>
                        <div>📊 {t('kb.rank')}: #{i + 1}</div>
                        {distance != null && <div>📐 {t('kb.distance')}: {distance.toFixed(4)}</div>}
                        <div>🔤 ~{tokens} tokens</div>
                      </div>
                    )}

                    {/* collapsible retrieval trace — the "how did we get here" brain */}
                    <button
                      onClick={() => toggle(openThinking, i, setOpenThinking)}
                      className="mt-2 flex items-center gap-1 text-[11px] text-text-muted hover:text-text-secondary"
                    >
                      <span className={'inline-block transition-transform ' + (thinkingOpen ? 'rotate-90' : '')}>▸</span>
                      💭 {t('kb.thinkingProcess')}
                    </button>
                    {thinkingOpen && (
                      <div className="mt-1 rounded-md border border-border bg-bg-tertiary p-2 font-mono text-[10px] text-text-muted">
                        vector search → query embedded → top_k=5 → rank #{i + 1}
                        {distance != null ? ` (distance ${distance.toFixed(4)})` : ''}
                      </div>
                    )}

                    {/* chunk body — query terms highlighted inline */}
                    <p className="mt-2 whitespace-pre-wrap text-xs text-text-primary">
                      {highlightTerms(shownText, kbQuery)}
                    </p>

                    {viewMode === 'concise' && tooLong && (
                      <button
                        onClick={() => toggle(openText, i, setOpenText)}
                        className="mt-1 text-[11px] text-accent hover:underline"
                      >
                        {showFull ? t('kb.showLess') : t('kb.showMore')}
                      </button>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Provenance DAG view ── lightweight SVG, no extra deps ── */
function ProvenanceDagView({ nodes, edges }: { nodes: any[]; edges: any[] }) {
  if (!nodes?.length) {
    return <div className="mb-3 text-xs text-text-muted">No provenance entries</div>;
  }

  // simple circular layout — fine for <50 nodes
  const R = Math.max(80, nodes.length * 8);
  const cx = R + 40;
  const cy = R + 20;
  const positions = new Map<string, { x: number; y: number }>();
  nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
    positions.set(n.id, { x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) });
  });

  return (
    <div className="mb-3 overflow-auto rounded-lg border border-border bg-bg-secondary p-2">
      <svg width="100%" height={Math.max(200, cy * 2)} viewBox={`0 0 ${cx * 2 + 40} ${cy * 2 + 20}`}>
        {/* edges */}
        {edges.map((e, i) => {
          const s = positions.get(e.source);
          const t = positions.get(e.target);
          if (!s || !t) return null;
          return (
            <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
              stroke="var(--border, #e5e5e5)" strokeWidth="1" opacity="0.5" />
          );
        })}
        {/* nodes */}
        {nodes.map((n) => {
          const pos = positions.get(n.id);
          if (!pos) return null;
          const label = (n.label || n.id || '').toString().slice(0, 20);
          return (
            <g key={n.id}>
              <circle cx={pos.x} cy={pos.y} r="4"
                fill={n.tool ? 'var(--accent, #3b82f6)' : 'var(--text-muted, #999)'} />
              <text x={pos.x + 7} y={pos.y + 3}
                fontSize="9" fontFamily="Arial, sans-serif"
                fill="var(--seed-text-secondary, #666)">
                {label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-1 text-[10px] text-text-muted">
        {nodes.length} nodes · {edges.length} edges
      </div>
    </div>
  );
}
