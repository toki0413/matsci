/**
 * CodeSearchPanel — semantic code search across workspace.
 *
 * Calls POST /codebase/search which hits the backend CodebaseIndex.
 */
import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Search, FileCode, Loader2 } from 'lucide-react';

interface SearchResult {
  file: string;
  line?: number;
  snippet: string;
  score?: number;
}

interface SearchResponse {
  results: SearchResult[];
  error?: string;
}

export function CodeSearchPanel({ apiBase }: { apiBase: string }) {
  const { t } = useTranslation();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [available, setAvailable] = useState<boolean | null>(null);

  React.useEffect(() => {
    fetch(`${apiBase}/codebase`)
      .then(r => r.json())
      .then(d => setAvailable(d.available !== false))
      .catch(() => setAvailable(false));
  }, [apiBase]);

  const doSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${apiBase}/codebase/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 10 }),
      });
      const data: SearchResponse = await res.json();
      if (data.error) {
        setError(data.error);
        setResults([]);
      } else {
        setResults(data.results || []);
      }
    } catch (e) {
      setError(String(e));
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, apiBase]);

  if (available === false) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <FileCode size={32} className="text-text-muted" />
        <p className="text-sm text-text-secondary">
          {t('codesearch.notAvailable')}
          <br />
          {t('codesearch.hint')}
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex gap-2 border-b border-border p-3">
        <div className="relative flex-1">
          <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && doSearch()}
            placeholder={t('codesearch.placeholder')}
            className="w-full rounded-lg border border-border bg-bg-tertiary py-1.5 pl-8 pr-3 text-sm text-text-primary placeholder-text-muted focus:border-accent focus:outline-none"
          />
        </div>
        <button
          onClick={doSearch}
          disabled={loading || !query.trim()}
          className="rounded-lg bg-accent px-3 py-1.5 text-sm text-white disabled:opacity-40"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : t('codesearch.search')}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {error && (
          <div className="mb-2 rounded-lg bg-red-500/10 p-2 text-xs text-red-500">{error}</div>
        )}
        {results.length === 0 && !loading && !error && query && (
          <p className="text-center text-xs text-text-muted mt-8">{t('codesearch.noResults')}</p>
        )}
        {results.map((r, i) => (
          <div key={i} className="mb-2 rounded-lg border border-border bg-bg-secondary p-3 hover:border-accent/40">
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <FileCode size={12} />
              <span className="font-mono">{r.file}{r.line ? `:${r.line}` : ''}</span>
              {r.score !== undefined && (
                <span className="ml-auto">{(r.score * 100).toFixed(0)}%</span>
              )}
            </div>
            <pre className="mt-1 overflow-x-auto text-xs text-text-secondary">
              {r.snippet}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}
