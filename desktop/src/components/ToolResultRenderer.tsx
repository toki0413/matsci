/**
 * ToolResultRenderer — Smart renderer for tool call results.
 *
 * Routes by toolName to specialized renderers (band structure, convergence,
 * 3D structure). Falls back to JSON table/KV/raw for everything else.
 *
 * ponytail: no plugin registry. A simple detectSpecial() switch is enough.
 * recharts already in deps — used it instead of adding plotly.
 */
import React, { useState, useMemo, useEffect, lazy, Suspense } from 'react';
import { useTranslation } from 'react-i18next';
import { Table2, List, Box, Activity, BarChart3, Image as ImageIcon, AlertTriangle } from 'lucide-react';
import { detectStructure } from './InlineStructure3D';
import { API_BASE } from '../lib/config-store';
import { authHeaders } from '../lib/api';

// Lazy-load heavy components
const InlineStructure3D = lazy(() => import('./InlineStructure3D'));
const BandStructureChart = lazy(() => import('./charts/BandStructureChart'));
const ConvergenceChart = lazy(() => import('./charts/ConvergenceChart'));

// <img> 标签不能带 Authorization 头. 服务器开 auth 后 GET /knowledge/image 会 401.
// fetch + createObjectURL 拿 blob 再塞 src. dev mode (HUGINN_DEV_MODE=1) 下 auth
// 自动 bypass, 走普通 <img> 也行 — 但这里统一走 AuthImage 省得分两种路径.
// ponytail: 每张图一次 fetch, 没做缓存. KB 查询 top_k 默认 5, 量小够用.
// 升级路径: 加 LRU cache (URL.createObjectURL 的 key 按 url hash).
const AuthImage: React.FC<{ src: string; alt: string; className?: string; loading?: 'lazy' | 'eager' }> =
  ({ src, alt, className, loading }) => {
    const [blobUrl, setBlobUrl] = useState<string | null>(null);
    const [err, setErr] = useState(false);
    useEffect(() => {
      let cancelled = false;
      let objUrl: string | null = null;
      setErr(false);
      fetch(src, { headers: authHeaders() })
        .then(r => {
          if (!r.ok) throw new Error(String(r.status));
          return r.blob();
        })
        .then(b => {
          if (cancelled) return;
          objUrl = URL.createObjectURL(b);
          setBlobUrl(objUrl);
        })
        .catch(() => { if (!cancelled) setErr(true); });
      return () => {
        cancelled = true;
        if (objUrl) URL.revokeObjectURL(objUrl);
      };
    }, [src]);
    if (err) return <span className="text-xs text-error">image load failed</span>;
    if (!blobUrl) return <span className="text-xs text-text-muted">loading…</span>;
    return <img src={blobUrl} alt={alt} className={className} loading={loading} />;
  };

interface ToolResultRendererProps {
  content: string;
  toolName?: string;
  maxRows?: number;
  className?: string;
  style?: React.CSSProperties;
  onExpand?: () => void;  // ← NEW: "view details" callback for side panel
}

type ParsedData =
  | { kind: 'table'; headers: string[]; rows: Record<string, unknown>[] }
  | { kind: 'kv'; entries: [string, unknown][] }
  | { kind: 'text' };

type SpecialType = 'band' | 'convergence' | 'structure' | 'visual' | 'knowledge' | null;

/** Detect if tool result has special visualization needs. */
function detectSpecial(toolName: string | undefined, content: string): SpecialType {
  if (!toolName) return null;
  const tn = toolName.toLowerCase();
  // Visual primitives: backend enrich_with_visual injects _visual_base64 / _visual_hint
  // 之前这些字段被当 JSON 字符串塞进表格, base64 成乱码. 检测到就切 visual 视图.
  try {
    const data = JSON.parse(content);
    if (data?._visual_base64 || data?._visual_hint || data?._visual_self_check) {
      return 'visual';
    }
    // rag_tool 的 PDF 视觉压缩页 chunk 带 image_url (后端 GET /knowledge/image 服务).
    // 之前 image_ref 路径只塞 agent prompt 文本, 前端永远看不到图.
    if ((tn.includes('rag') || tn.includes('knowledge')) && Array.isArray(data?.results)) {
      if (data.results.some((r: any) => r?.image_url)) return 'knowledge';
    }
  } catch { /* not JSON */ }
  // Band structure: vasp_band, band_structure, dos
  if (tn.includes('band') || tn.includes('dos') || tn.includes('electronic')) {
    try {
      const data = JSON.parse(content);
      if (data?.bands || data?.dos || data?.kpath || data?.band_structure) return 'band';
    } catch { /* not JSON */ }
  }
  // Convergence: convergence_check, scf, optimization
  if (tn.includes('converg') || tn.includes('scf') || tn.includes('optim')) {
    try {
      const data = JSON.parse(content);
      if (data?.energies || data?.convergence || data?.scf_energies) return 'convergence';
    } catch { /* not JSON */ }
  }
  // Structure: detect from content (existing logic)
  if (detectStructure(content) !== null) return 'structure';
  return null;
}

/** Extract visual fields from content JSON, stripping them so they don't pollute table view. */
function extractVisual(content: string): {
  base64?: string;
  hint?: string;
  selfCheck?: { confidence: number; caveats: string[] };
  cleaned: string;  // content with visual fields stripped, for table/raw view
  rawData?: Record<string, unknown>;
} {
  try {
    const data = JSON.parse(content) as Record<string, unknown>;
    const base64 = data._visual_base64 as string | undefined;
    const hint = data._visual_hint as string | undefined;
    const selfCheck = data._visual_self_check as { confidence: number; caveats: string[] } | undefined;
    // strip visual fields so they don't show up as table rows
    const cleaned = { ...data };
    delete cleaned._visual_base64;
    delete cleaned._visual_hint;
    delete cleaned._visual_self_check;
    delete cleaned._visual_primitives;
    return { base64, hint, selfCheck, cleaned: JSON.stringify(cleaned), rawData: cleaned };
  } catch {
    return { cleaned: content };
  }
}

function tryParse(content: string): ParsedData {
  let data: unknown;
  try {
    data = JSON.parse(content);
  } catch {
    return { kind: 'text' };
  }

  if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object' && data[0] !== null) {
    const headers = [...new Set(data.flatMap((row) => Object.keys(row as Record<string, unknown>)))];
    return { kind: 'table', headers, rows: data as Record<string, unknown>[] };
  }

  if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    const entries = Object.entries(obj);
    if (entries.length > 0) {
      return { kind: 'kv', entries };
    }
  }

  if (Array.isArray(data) && data.length > 0) {
    return { kind: 'table', headers: ['value'], rows: data.map((v) => ({ value: v })) };
  }

  return { kind: 'text' };
}

function formatCell(val: unknown): string {
  if (val === null || val === undefined) return '—';
  if (typeof val === 'object') return JSON.stringify(val);
  return String(val);
}

export const ToolResultRenderer: React.FC<ToolResultRendererProps> = ({
  content,
  toolName,
  maxRows = 50,
  className,
  style,
  onExpand,
}) => {
  const { t } = useTranslation();
  const special = useMemo(() => detectSpecial(toolName, content), [toolName, content]);
  // visual 分支: 把 _visual_base64 / _visual_hint / _visual_self_check 提取出来,
  // 剩余 content 给 tryParse 做表格视图, 避免大 base64 污染表格.
  const visualInfo = useMemo(() => special === 'visual' ? extractVisual(content) : null, [special, content]);
  const effectiveContent = visualInfo ? visualInfo.cleaned : content;
  const parsed = useMemo(() => tryParse(effectiveContent), [effectiveContent]);

  const [view, setView] = useState<'chart' | '3d' | 'visual' | 'knowledge' | 'table' | 'raw'>(
    special === 'band' ? 'chart' :
    special === 'convergence' ? 'chart' :
    special === 'structure' ? '3d' :
    special === 'visual' ? 'visual' :
    special === 'knowledge' ? 'knowledge' :
    (parsed.kind !== 'text' ? 'table' : 'raw')
  );

  // Plain text — just render as preformatted
  if (parsed.kind === 'text' && !special) {
    return (
      <pre className={`mt-1 max-h-60 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs ${className || ''}`} style={style}>
        {content}
      </pre>
    );
  }

  const displayRows = parsed.kind === 'table' ? parsed.rows.slice(0, maxRows) : [];
  const truncated = parsed.kind === 'table' && parsed.rows.length > maxRows;

  const showChartTab = special === 'band' || special === 'convergence';
  const show3DTab = special === 'structure';
  const showVisualTab = special === 'visual';
  const showKnowledgeTab = special === 'knowledge';

  return (
    <div data-component="structured-output" className={`fade-in ${className || ''}`} style={style}>
      <div className="so-toolbar">
        <div className="so-tabs">
          {showChartTab && (
            <button className={`so-tab ${view === 'chart' ? 'so-tab--active' : ''}`} onClick={() => setView('chart')}>
              {special === 'band' ? <BarChart3 size={10} style={{ display: 'inline', marginRight: 4 }} /> : <Activity size={10} style={{ display: 'inline', marginRight: 4 }} />}
              {special === 'band' ? 'Bands' : 'Convergence'}
            </button>
          )}
          {show3DTab && (
            <button className={`so-tab ${view === '3d' ? 'so-tab--active' : ''}`} onClick={() => setView('3d')}>
              <Box size={10} style={{ display: 'inline', marginRight: 4 }} />
              3D
            </button>
          )}
          {showVisualTab && (
            <button className={`so-tab ${view === 'visual' ? 'so-tab--active' : ''}`} onClick={() => setView('visual')}>
              <ImageIcon size={10} style={{ display: 'inline', marginRight: 4 }} />
              Visual
            </button>
          )}
          {showKnowledgeTab && (
            <button className={`so-tab ${view === 'knowledge' ? 'so-tab--active' : ''}`} onClick={() => setView('knowledge')}>
              <ImageIcon size={10} style={{ display: 'inline', marginRight: 4 }} />
              Pages
            </button>
          )}
          <button className={`so-tab ${view === 'table' ? 'so-tab--active' : ''}`} onClick={() => setView('table')}>
            <Table2 size={10} style={{ display: 'inline', marginRight: 4 }} />
            {parsed.kind === 'table' ? t('output.table') : 'KV'}
          </button>
          <button className={`so-tab ${view === 'raw' ? 'so-tab--active' : ''}`} onClick={() => setView('raw')}>
            <List size={10} style={{ display: 'inline', marginRight: 4 }} />
            {t('output.raw')}
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginLeft: 'auto', gap: 8 }}>
          {toolName && (
            <span style={{ fontSize: 10, color: 'var(--fg-muted)' }}>{toolName}</span>
          )}
          {onExpand && (
            <button
              onClick={onExpand}
              style={{ fontSize: 10, color: 'var(--fg-muted)', cursor: 'pointer', background: 'none', border: 'none' }}
              title="Open in side panel"
            >
              ⤢
            </button>
          )}
        </div>
      </div>
      <div className="so-body">
        {view === 'chart' && special === 'band' && (
          <Suspense fallback={<div className="p-4 text-xs text-text-muted">Loading chart…</div>}>
            <BandStructureChart content={content} />
          </Suspense>
        )}
        {view === 'chart' && special === 'convergence' && (
          <Suspense fallback={<div className="p-4 text-xs text-text-muted">Loading chart…</div>}>
            <ConvergenceChart content={content} />
          </Suspense>
        )}
        {view === '3d' && show3DTab && (
          <Suspense fallback={<div className="p-4 text-xs text-text-muted">Loading 3D viewer…</div>}>
            <InlineStructure3D content={content} />
          </Suspense>
        )}
        {view === 'visual' && showVisualTab && visualInfo && (
          <div className="space-y-2 p-2">
            {visualInfo.base64 && (
              <img
                src={`data:image/png;base64,${visualInfo.base64}`}
                alt="tool output visualization"
                className="max-w-full rounded-lg border border-border"
                style={{ maxHeight: 400, objectFit: 'contain' }}
              />
            )}
            {visualInfo.selfCheck && (
              <div className="flex items-center gap-2 text-xs">
                <AlertTriangle
                  size={12}
                  style={{
                    color: visualInfo.selfCheck.confidence < 0.5
                      ? 'var(--warning, #f59e0b)'
                      : 'var(--success, #10b981)'
                  }}
                />
                <span style={{ color: 'var(--fg-muted)' }}>
                  confidence: {(visualInfo.selfCheck.confidence * 100).toFixed(0)}%
                </span>
                {visualInfo.selfCheck.caveats.length > 0 && (
                  <span style={{ color: 'var(--warning, #f59e0b)' }}>
                    ({visualInfo.selfCheck.caveats.length} caveats)
                  </span>
                )}
              </div>
            )}
            {visualInfo.hint && (
              <details className="text-xs">
                <summary style={{ cursor: 'pointer', color: 'var(--fg-muted)' }}>
                  Visual primitives (Mirage)
                </summary>
                <pre className="mt-1 max-h-40 overflow-auto rounded bg-bg-tertiary p-2 text-xs">
                  {visualInfo.hint}
                </pre>
              </details>
            )}
          </div>
        )}
        {view === 'knowledge' && showKnowledgeTab && (() => {
          // rag_tool 视觉压缩页: 每个 result 项渲染 image_url (PDF 页快照) + 文本.
          // 之前 image_ref 路径只塞 agent prompt 文本, 前端永远看不到图.
          try {
            const data = JSON.parse(content) as { results?: any[] };
            const items = (data?.results || []).filter((r) => r?.image_url);
            if (items.length === 0) return null;
            return (
              <div className="space-y-3 p-2">
                {items.map((r, i) => (
                  <div key={r.id || i} className="rounded-lg border border-border overflow-hidden">
                    <AuthImage
                      src={`${API_BASE}${r.image_url}`}
                      alt={`chunk ${r.id ?? i}`}
                      className="w-full max-h-[400px] object-contain bg-bg-tertiary"
                      loading="lazy"
                    />
                    {r.document && (
                      <details className="p-2 text-xs">
                        <summary style={{ cursor: 'pointer', color: 'var(--fg-muted)' }}>
                          chunk text ({r.document.length} chars) · distance {String(r.distance ?? '—')}
                        </summary>
                        <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-bg-tertiary p-2">
                          {r.document}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            );
          } catch {
            return null;
          }
        })()}
        {view === 'table' && parsed.kind === 'table' && (
          <>
            <table className="so-table">
              <thead>
                <tr>{parsed.headers.map((h) => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {displayRows.map((row, i) => (
                  <tr key={i}>
                    {parsed.headers.map((h) => <td key={h}>{formatCell(row[h])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
            {truncated && (
              <div style={{ fontSize: 10, color: 'var(--fg-muted)', marginTop: 8, textAlign: 'center' }}>
                Showing {maxRows} of {parsed.rows.length} rows
              </div>
            )}
          </>
        )}
        {view === 'table' && parsed.kind === 'kv' && (
          <table className="so-table">
            <thead>
              <tr>
                <th>{t('output.table.header.parameter')}</th>
                <th>{t('output.table.header.value')}</th>
              </tr>
            </thead>
            <tbody>
              {parsed.entries.map(([key, val]) => (
                <tr key={key}>
                  <td style={{ fontWeight: 500 }}>{key}</td>
                  <td>{formatCell(val)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {view === 'raw' && (
          <pre className="max-h-60 overflow-auto text-xs" style={{ margin: 0, background: 'transparent', border: 'none', padding: 0 }}>
            {content}
          </pre>
        )}
      </div>
    </div>
  );
};
