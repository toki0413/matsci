import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PanelHeader } from '../settings-shared';
import { Skeleton, SkeletonText } from '../Skeleton';
import { getApiBase } from '../../lib/api-client';
import type { AppConfig, DocumentGraph } from '../../types/domain';

interface KbDoc {
  doc_id: string;
  filename: string;
  source?: 'seed' | 'distill' | 'auto' | 'upload';
}

interface KbChunk {
  text: string;
  distance?: number;
  metadata?: { filename?: string; source_type?: string; domain?: string };
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
  uploadKnowledgeMany: (files: FileList | File[]) => void;
  parseDocument: (file: File) => void;
  loadDocumentGraph: (docId: string) => void;
  deleteKnowledge: (docId: string) => void;
  queryKnowledge: () => void;
  ingestUrl: (url: string) => void;
  loadProvenanceDag: () => Promise<any>;
  // 全文/分块预览
  viewingDoc: { doc_id: string; filename: string } | null;
  docChunks: any[] | null;
  docLoading: boolean;
  loadDocumentContent: (doc: { doc_id: string; filename: string }) => void;
  clearDocView: () => void;
  // 文档结构图谱(SVG 渲染)
  docGraph: DocumentGraph | null;
  viewDocGraph: (docId: string) => void;
  clearDocGraph: () => void;
  // 无依赖友好提示: 传入 embedding 下载状态, 面板文案跟着变
  embeddingDownload?: { status: 'idle' | 'downloading' | 'done' | 'error'; percent?: number; error?: string };
  // 文档详情预览 Tab: 分块/原文/图片/报告
  docImages: any[] | null;
  imagesLoading: boolean;
  reportLoading: boolean;
  reportContent: string | null;
  reportError: string | null;
  loadDocImages: (docId: string) => void;
  generateReport: (docId: string, title: string) => void;
  fetchRawText: (docId: string) => Promise<string>;
  downloadRaw: (docId: string, filename: string) => Promise<boolean>;
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

// 来源徽标: key 与后端 store._classify_doc_source 的取值一致
const SOURCE_META: Record<string, { label: string; cls: string }> = {
  seed: { label: '预置', cls: 'bg-accent/15 text-accent' },
  distill: { label: '蒸馏', cls: 'bg-purple-500/15 text-purple-300' },
  auto: { label: '任务沉淀', cls: 'bg-amber-500/15 text-amber-300' },
  upload: { label: '上传', cls: 'bg-emerald-500/15 text-emerald-300' },
};
const SOURCE_ORDER = ['upload', 'distill', 'auto', 'seed'] as const;

function DocSourceBadge({ src }: { src?: string }) {
  const m = (src && SOURCE_META[src]) || SOURCE_META.upload;
  return (
    <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${m.cls}`}>
      {m.label}
    </span>
  );
}

// 通用 SVG 节点-连线图: 文档结构图谱 / Provenance DAG 共用同一布局,
// 环形摆法够用(文档图一般 <100 节点), 不引额外依赖.
function SimpleGraphView({ nodes, edges }: { nodes: any[]; edges: any[] }) {
  if (!nodes?.length) {
    return <div className="text-xs text-text-muted">暂无图谱数据</div>;
  }
  const R = Math.max(80, nodes.length * 9);
  const cx = R + 60;
  const cy = R + 40;
  const positions = new Map<string, { x: number; y: number }>();
  nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
    positions.set(n.id, { x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) });
  });
  const edgeKeys = new Set<string>(edges.map((e) => `${e.source}->${e.target}`));
  return (
    <svg width="100%" height={Math.max(220, cy * 2)} viewBox={`0 0 ${cx * 2 + 60} ${cy * 2 + 40}`}>
      {edges.map((e, i) => {
        const s = positions.get(e.source);
        const t = positions.get(e.target);
        if (!s || !t) return null;
        return (
          <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
            stroke="var(--border, #e5e5e5)" strokeWidth="1" opacity="0.5" />
        );
      })}
      {nodes.map((n) => {
        const pos = positions.get(n.id);
        if (!pos) return null;
        const isHub = edgeKeys.size > 0 && nodes.reduce((acc, m) => acc + (edgeKeys.has(`${m.id}->${n.id}`) ? 1 : 0), 0) > 1;
        const label = (n.label || n.id || '').toString().slice(0, 18);
        return (
          <g key={n.id}>
            <circle cx={pos.x} cy={pos.y} r={isHub ? 6 : 4}
              fill={n.tool ? 'var(--accent, #3b82f6)' : 'var(--text-muted, #999)'} />
            <text x={pos.x + 8} y={pos.y + 3}
              fontSize="10" fontFamily="Arial, sans-serif"
              fill="var(--seed-text-secondary, #666)">
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// 空状态新手引导: 告诉新用户"这三条路都能建库"。发上传/贴链接/任务沉淀。
function EmptyStateGuide({ onPickFile }: { onPickFile: () => void }) {
  return (
    <div className="mb-3 rounded-lg border border-accent/20 bg-accent/5 p-3">
      <div className="mb-2 text-xs font-semibold text-text-primary">🚀 知识库还空着，三条路都能开始</div>
      <div className="space-y-1.5 text-xs text-text-secondary">
        <div>1️⃣ 点上面「选择文件上传」，或直接拖文件进来（支持 PDF/Word/图片 OCR）</div>
        <div>2️⃣ 在下方「Add from URL」粘贴网页链接，把文章直接抓进来</div>
        <div>3️⃣ 直接跟 Agent 对话跑任务，结果会自动沉淀进知识库（来源标注「任务沉淀」）</div>
      </div>
      <button
        onClick={onPickFile}
        className="mt-2 w-full rounded-lg border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs font-medium text-accent hover:bg-accent/20 transition-colors"
      >
        上传第一个文件
      </button>
    </div>
  );
}

// 无依赖友好提示: 不抛"Install chromadb"死墙, 而是看 embedding 下载进度给可行动引导.
function EmptyStateHint({ embedding }: { embedding?: KnowledgePanelProps['embeddingDownload'] }) {
  if (embedding?.status === 'downloading' || embedding?.status === 'idle') {
    return (
      <div className="mb-2 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
        {embedding.status === 'downloading'
          ? `⏳ 知识库模型下载中（${Math.round(embedding.percent || 0)}%），完成后即可上传检索`
          : '⌛ 知识库后端正在初始化（首次需下载模型），请稍候'}
      </div>
    );
  }
  if (embedding?.status === 'done') {
    return (
      <div className="mb-2 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
        ✅ 模型已就绪。若仍不可用，可重启应用后再试。
      </div>
    );
  }
  return (
    <div className="mb-2 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
      知识库后端暂不可用，可能是依赖缺失或首次初始化未完成。请重启应用重试；离线环境请先下载 embedding 模型。
    </div>
  );
}

export function KnowledgePanel({
  config, setConfig, saveConfig,
  fileInputRef, parseFileInputRef, parseLoading, uploadPct, kbLoading,
  kbMsg, kbDocs, kbAvailable, kbQuery, kbChunks, setKbQuery,
  uploadKnowledgeMany, parseDocument, deleteKnowledge, queryKnowledge,
  ingestUrl, loadProvenanceDag,
  viewingDoc, docChunks, docLoading, loadDocumentContent, clearDocView,
  docGraph, viewDocGraph, clearDocGraph, embeddingDownload,
  docImages, imagesLoading, reportLoading, reportContent, reportError,
  loadDocImages, generateReport, fetchRawText, downloadRaw,
}: KnowledgePanelProps) {
  const { t } = useTranslation();
  const [viewMode, setViewMode] = useState<ViewMode>('detailed');
  const [searching, setSearching] = useState(false);
  const [kbDocLimit, setKbDocLimit] = useState(20);
  const [urlInput, setUrlInput] = useState('');
  const [showDag, setShowDag] = useState(false);
  const [dagData, setDagData] = useState<{ nodes: any[]; edges: any[] } | null>(null);
  // per-chunk toggles — tracks which cards have their citation / thinking / body expanded
  const [openCitations, setOpenCitations] = useState<Set<number>>(new Set());
  const [openThinking, setOpenThinking] = useState<Set<number>>(new Set());
  const [openText, setOpenText] = useState<Set<number>>(new Set());
  // 拖拽上传高亮 + 文档来源过滤
  const [dragOver, setDragOver] = useState(false);
  const [srcFilter, setSrcFilter] = useState<string | null>(null);

  // 文档详情 Tab + 原文文本缓存; 切换文档时重置
  const [docTab, setDocTab] = useState<'chunks' | 'raw' | 'images' | 'report'>('chunks');
  const [rawText, setRawText] = useState<string | null>(null);

  // 拖拽上传: 拖进高亮, 松开放下即批量摄入
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer?.files?.length) uploadKnowledgeMany(e.dataTransfer.files);
  };
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  // switch to a new doc → reset tab + raw text
  useEffect(() => {
    setDocTab('chunks');
    setRawText(null);
  }, [viewingDoc?.doc_id]);

  // text-like extensions get inline preview; others only a download button
  const isTextLike = (name: string) => /\.(txt|md|markdown|csv|json|yaml|yml|py|tex|log)$/i.test(name);

  const openRawTab = async () => {
    setDocTab('raw');
    if (!viewingDoc || rawText !== null) return;
    if (isTextLike(viewingDoc.filename)) {
      setRawText(''); // loading marker
      const txt = await fetchRawText(viewingDoc.doc_id);
      setRawText(txt || null);
    }
  };
  const openImagesTab = () => {
    setDocTab('images');
    if (docImages === null && viewingDoc) loadDocImages(viewingDoc.doc_id);
  };
  const runReport = () => {
    if (!viewingDoc) return;
    generateReport(viewingDoc.doc_id, `${viewingDoc.filename} 分析报告`);
  };

  const filteredDocs = srcFilter ? kbDocs.filter((d) => d.source === srcFilter) : kbDocs;

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
          <div
            className={`kb-upload mb-4 rounded-lg border border-dashed p-4 text-center transition-colors ${
              dragOver ? 'border-accent bg-accent/10' : 'border-border bg-bg-tertiary'
            }`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={() => setDragOver(false)}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              accept=".txt,.md,.pdf,.docx,.xlsx,.csv,.json,.yaml,.yml,.tex,image/*"
              onChange={(e) => {
                if (e.target.files?.length) uploadKnowledgeMany(e.target.files);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-primary w-full text-xs"
            >
              📤 选择文件上传
            </button>
            <p className="mt-2 text-xs text-text-muted">
              支持 TXT / MD / PDF / Word / Excel / CSV / 代码 / 图片(OCR)，可拖拽到此处
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
              <div className="space-y-2">
                {[0, 1, 2, 3].map(i => (
                  <Skeleton key={i} className="h-8 w-full" />
                ))}
              </div>
            ) : (
            <>
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-text-secondary">
                Documents ({kbDocs.length}{kbDocs.length > 20 ? ` · showing ${kbDocLimit}` : ''})
              </span>
            </div>

            {!kbAvailable && (
              <EmptyStateHint embedding={embeddingDownload} />
            )}

            {/* 来源过滤: 让用户看懂库里都有什么 */}
            {kbDocs.length > 0 && (
              <div className="mb-2 flex flex-wrap items-center gap-1">
                <button
                  onClick={() => setSrcFilter(null)}
                  className={`rounded-full px-2 py-0.5 text-[10px] ${srcFilter === null ? 'bg-accent text-white' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'}`}
                >
                  全部
                </button>
                {SOURCE_ORDER.map((s) => {
                  const n = kbDocs.filter((d) => d.source === s).length;
                  if (!n) return null;
                  return (
                    <button
                      key={s}
                      onClick={() => setSrcFilter(srcFilter === s ? null : s)}
                      className={`rounded-full px-2 py-0.5 text-[10px] ${
                        srcFilter === s ? 'bg-accent text-white' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      {SOURCE_META[s].label} {n}
                    </button>
                  );
                })}
              </div>
            )}

            {/* 空状态新手引导 */}
            {!kbLoading && kbDocs.length === 0 && kbAvailable && (
              <EmptyStateGuide onPickFile={() => fileInputRef.current?.click()} />
            )}

            {filteredDocs.slice(0, kbDocLimit).map((doc) => (
              <div
                key={doc.doc_id}
                className="kb-doc-item mb-2 flex items-center justify-between gap-2 rounded-lg border border-border bg-bg-tertiary p-2"
              >
                <div className="min-w-0 flex-1">
                  <button
                    onClick={() => loadDocumentContent(doc)}
                    className="block w-full truncate text-left text-xs text-text-primary hover:text-accent"
                    title="查看文档内容"
                    aria-label={`View document ${doc.filename}`}
                  >
                    {doc.filename}
                  </button>
                  <div className="mt-1">
                    <DocSourceBadge src={doc.source} />
                  </div>
                </div>
                <div className="flex shrink-0 gap-2">
                  <button
                    onClick={() => viewDocGraph(doc.doc_id)}
                    className="text-xs text-accent hover:underline"
                    title="查看文档结构图谱"
                    aria-label="查看文档结构图谱"
                  >
                    📊
                  </button>
                  <button
                    onClick={() => deleteKnowledge(doc.doc_id)}
                    className="text-xs text-error hover:underline"
                  >
                    删
                  </button>
                </div>
              </div>
            ))}
            {filteredDocs.length > kbDocLimit && (
              <button
                onClick={() => setKbDocLimit(prev => prev + 20)}
                className="w-full rounded-lg border border-border py-2 text-xs text-text-secondary hover:bg-bg-tertiary transition-colors"
              >
                更多 ({filteredDocs.length - kbDocLimit})…
              </button>
            )}
            </>
            )}
          </div>
        </aside>

        {/* Query tester — Metaso-style "transparent brain" retrieval view */}
        <div className="kb-query-area flex flex-1 flex-col bg-bg-primary p-4">
          <div className="mb-3 flex items-center justify-between">
            {docGraph ? (
              <h3 className="min-w-0 text-sm font-semibold">
                <button
                  onClick={clearDocGraph}
                  className="mr-2 text-accent hover:underline"
                  title="Back to search"
                  aria-label="Back to search"
                >
                  ← 检索
                </button>
                <span className="truncate">📊 文档结构图谱</span>
              </h3>
            ) : viewingDoc ? (
              <h3 className="min-w-0 text-sm font-semibold">
                <button
                  onClick={clearDocView}
                  className="mr-2 text-accent hover:underline"
                  title="Back to search"
                  aria-label="Back to search"
                >
                  ← {t('kb.search')}
                </button>
                <span className="truncate">{viewingDoc.filename}</span>
              </h3>
            ) : (
              <>
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
              </>
            )}
          </div>

          {/* 文档详情预览: Tab 栏 → 分块/原文/图片/报告 */}
          {docGraph ? (
            <div className="flex-1 overflow-auto">
              <div className="rounded-lg border border-border bg-bg-secondary p-3">
                <SimpleGraphView nodes={docGraph.nodes || []} edges={docGraph.edges || []} />
                <div className="mt-1 text-[10px] text-text-muted">
                  {(docGraph.nodes?.length || 0)} 节点 · {(docGraph.edges?.length || 0)} 条边
                </div>
              </div>
            </div>
          ) : viewingDoc ? (
            <div className="flex flex-1 flex-col overflow-hidden">
              {/* Tab bar */}
              <div className="mb-2 flex gap-1 rounded-lg border border-border bg-bg-secondary p-0.5">
                {(['chunks', 'raw', 'images', 'report'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => {
                      setDocTab(tab);
                      if (tab === 'raw') openRawTab();
                      if (tab === 'images') openImagesTab();
                    }}
                    className={'rounded-md px-2.5 py-1 text-xs font-medium transition-colors ' +
                      (docTab === tab ? 'bg-accent text-white' : 'text-text-secondary hover:text-text-primary')}
                  >
                    {{ chunks: '分块', raw: '原文', images: '图片', report: '报告' }[tab]}
                  </button>
                ))}
                <button
                  onClick={() => downloadRaw(viewingDoc.doc_id, viewingDoc.filename)}
                  className="rounded-md px-2 py-1 text-xs text-accent hover:bg-accent/10 ml-auto"
                  title="下载原始文件"
                >
                  ⬇
                </button>
              </div>

              {/* Tab content */}
              {docTab === 'chunks' && (
                <div className="flex-1 overflow-y-auto space-y-3">
                  {docLoading ? (
                    [0, 1, 2].map(i => (
                      <div key={i} className="kb-chunk rounded-lg border border-border bg-bg-secondary p-3">
                        <SkeletonText lines={3} />
                      </div>
                    ))
                  ) : !docChunks || docChunks.length === 0 ? (
                    <div className="text-xs text-text-muted">此文件暂无可预览的分块内容</div>
                  ) : (
                    docChunks.map((c, i) => (
                      <div key={i} className="kb-chunk rounded-lg border border-border bg-bg-secondary p-3">
                        <div className="mb-1 text-[10px] text-text-muted">#{c.chunk != null ? c.chunk + 1 : i + 1}</div>
                        <p className="whitespace-pre-wrap text-xs text-text-primary">{c.text}</p>
                      </div>
                    ))
                  )}
                </div>
              )}

              {docTab === 'raw' && (
                <div className="flex-1 overflow-y-auto">
                  {rawText === '' ? (
                    <SkeletonText lines={5} />
                  ) : rawText ? (
                    <pre className="whitespace-pre-wrap break-all rounded-lg border border-border bg-bg-secondary p-3 text-xs text-text-primary font-mono">{rawText}</pre>
                  ) : (
                    <div className="rounded-lg border border-border bg-bg-secondary p-3 text-xs text-text-muted">
                      {isTextLike(viewingDoc.filename)
                        ? '无法读取原文（可能文件损坏或无权访问）'
                        : '非文本文件，可点击 ⬇ 下载原始文件后本地查看'}
                    </div>
                  )}
                </div>
              )}

              {docTab === 'images' && (
                <div className="flex-1 overflow-y-auto space-y-3">
                  {imagesLoading ? (
                    [0, 1].map(i => (
                      <div key={i} className="rounded-lg border border-border bg-bg-secondary p-3">
                        <Skeleton className="h-48 w-full" />
                      </div>
                    ))
                  ) : !docImages?.length ? (
                    <div className="rounded-lg border border-border bg-bg-secondary p-3 text-xs text-text-muted">
                      该文档没有提取到的图片（压缩页/图表），或文档未经过深解析
                    </div>
                  ) : (
                    docImages.map((img, i) => (
                      <div key={i} className="rounded-lg border border-border bg-bg-secondary p-2">
                        <img
                          src={`${getApiBase()}${img.url}`}
                          alt={img.caption || `页面 ${img.page ?? i + 1}`}
                          className="max-w-full rounded object-contain"
                          loading="lazy"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                        {img.caption && (
                          <p className="mt-1 text-[10px] text-text-muted">{img.caption.slice(0, 200)}</p>
                        )}
                        {img.page != null && (
                          <p className="text-[10px] text-text-muted">第 {img.page + 1} 页</p>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}

              {docTab === 'report' && (
                <div className="flex-1 overflow-y-auto">
                  {reportLoading ? (
                    <div className="space-y-3">
                      <SkeletonText lines={5} />
                      <SkeletonText lines={3} />
                    </div>
                  ) : reportContent ? (
                    <div className="rounded-lg border border-border bg-bg-secondary p-3">
                      <div className="prose prose-sm max-w-none whitespace-pre-wrap text-xs text-text-primary">
                        {reportContent}
                      </div>
                      <button
                        onClick={() => {
                          const blob = new Blob([reportContent], { type: 'text/markdown' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `${viewingDoc.filename}.report.md`;
                          a.click();
                          URL.revokeObjectURL(url);
                        }}
                        className="mt-2 text-xs text-accent hover:underline"
                      >
                        下载报告
                      </button>
                    </div>
                  ) : reportError ? (
                    <div className="rounded-lg border border-error/30 bg-error/5 p-3 text-xs text-error">
                      {reportError}
                    </div>
                  ) : (
                    <div className="rounded-lg border border-border bg-bg-secondary p-3 text-xs text-text-muted">
                      <p className="mb-2">将当前文档内容自动汇总生成一份结构化的 Markdown 报告。</p>
                      <button
                        onClick={runReport}
                        disabled={reportLoading}
                        className="btn-primary w-full text-xs disabled:opacity-50"
                      >
                        {reportLoading ? '⏳ 生成中（需 LLM，约 30s-2min）…' : '📄 生成综合报告'}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <>
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
              [0, 1, 2].map(i => (
                <div key={i} className="kb-chunk rounded-lg border border-border bg-bg-secondary p-3">
                  <SkeletonText lines={3} />
                </div>
              ))
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
            </>
          )}
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
  return (
    <div className="mb-3 overflow-auto rounded-lg border border-border bg-bg-secondary p-2">
      <SimpleGraphView nodes={nodes} edges={edges} />
      <div className="mt-1 text-[10px] text-text-muted">
        {nodes.length} nodes · {edges.length} edges
      </div>
    </div>
  );
}
