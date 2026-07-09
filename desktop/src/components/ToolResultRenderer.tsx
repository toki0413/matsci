/**
 * ToolResultRenderer — Smart renderer for tool call results.
 *
 * Detects structured JSON data in tool results and renders them
 * as tables or key-value pairs. Falls back to raw text for plain output.
 */
import React, { useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Table2, List } from 'lucide-react';

interface ToolResultRendererProps {
  content: string;
  toolName?: string;
  maxRows?: number;
  className?: string;
  style?: React.CSSProperties;
}

type ParsedData =
  | { kind: 'table'; headers: string[]; rows: Record<string, unknown>[] }
  | { kind: 'kv'; entries: [string, unknown][] }
  | { kind: 'text' };

function tryParse(content: string): ParsedData {
  // Try JSON parse
  let data: unknown;
  try {
    data = JSON.parse(content);
  } catch {
    return { kind: 'text' };
  }

  // Array of objects → table
  if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object' && data[0] !== null) {
    const headers = [...new Set(data.flatMap((row) => Object.keys(row as Record<string, unknown>)))];
    return { kind: 'table', headers, rows: data as Record<string, unknown>[] };
  }

  // Single object → key-value
  if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    // If values are all simple, show as KV. If nested, try table for arrays within.
    const entries = Object.entries(obj);
    if (entries.length > 0) {
      return { kind: 'kv', entries };
    }
  }

  // Array of primitives → simple list
  if (Array.isArray(data) && data.length > 0) {
    return {
      kind: 'table',
      headers: ['value'],
      rows: data.map((v) => ({ value: v })),
    };
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
}) => {
  const { t } = useTranslation();
  const parsed = useMemo(() => tryParse(content), [content]);
  const [view, setView] = useState<'table' | 'raw'>(
    parsed.kind !== 'text' ? 'table' : 'raw'
  );

  // Plain text — just render as preformatted
  if (parsed.kind === 'text') {
    return (
      <pre className={`mt-1 max-h-60 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs ${className || ''}`} style={style}>
        {content}
      </pre>
    );
  }

  const displayRows = parsed.kind === 'table' ? parsed.rows.slice(0, maxRows) : [];
  const truncated = parsed.kind === 'table' && parsed.rows.length > maxRows;

  return (
    <div
      data-component="structured-output"
      className={`fade-in ${className || ''}`}
      style={style}
    >
      <div className="so-toolbar">
        <div className="so-tabs">
          <button
            className={`so-tab ${view === 'table' ? 'so-tab--active' : ''}`}
            onClick={() => setView('table')}
          >
            <Table2 size={10} style={{ display: 'inline', marginRight: 4 }} />
            {parsed.kind === 'table' ? t('output.table') : 'KV'}
          </button>
          <button
            className={`so-tab ${view === 'raw' ? 'so-tab--active' : ''}`}
            onClick={() => setView('raw')}
          >
            <List size={10} style={{ display: 'inline', marginRight: 4 }} />
            {t('output.raw')}
          </button>
        </div>
        {toolName && (
          <span style={{ fontSize: 10, color: 'var(--fg-muted)', marginLeft: 'auto' }}>
            {toolName}
          </span>
        )}
      </div>
      <div className="so-body">
        {view === 'table' && parsed.kind === 'table' && (
          <>
            <table className="so-table">
              <thead>
                <tr>
                  {parsed.headers.map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayRows.map((row, i) => (
                  <tr key={i}>
                    {parsed.headers.map((h) => (
                      <td key={h}>{formatCell(row[h])}</td>
                    ))}
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
