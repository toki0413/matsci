/**
 * ToolResultRenderer — Smart renderer for tool call results.
 *
 * Routes by toolName to specialized renderers (band structure, convergence,
 * 3D structure). Falls back to JSON table/KV/raw for everything else.
 *
 * ponytail: no plugin registry. A simple detectSpecial() switch is enough.
 * recharts already in deps — used it instead of adding plotly.
 */
import React, { useState, useMemo, lazy, Suspense } from 'react';
import { useTranslation } from 'react-i18next';
import { Table2, List, Box, Activity, BarChart3 } from 'lucide-react';
import { detectStructure } from './InlineStructure3D';

// Lazy-load heavy components
const InlineStructure3D = lazy(() => import('./InlineStructure3D'));
const BandStructureChart = lazy(() => import('./charts/BandStructureChart'));
const ConvergenceChart = lazy(() => import('./charts/ConvergenceChart'));

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

type SpecialType = 'band' | 'convergence' | 'structure' | null;

/** Detect if tool result has special visualization needs. */
function detectSpecial(toolName: string | undefined, content: string): SpecialType {
  if (!toolName) return null;
  const tn = toolName.toLowerCase();
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
  const parsed = useMemo(() => tryParse(content), [content]);
  const special = useMemo(() => detectSpecial(toolName, content), [toolName, content]);

  const [view, setView] = useState<'chart' | '3d' | 'table' | 'raw'>(
    special === 'band' ? 'chart' :
    special === 'convergence' ? 'chart' :
    special === 'structure' ? '3d' :
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
