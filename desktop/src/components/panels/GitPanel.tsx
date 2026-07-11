/**
 * GitPanel — basic Git workflow UI (status, diff, commit, push).
 *
 * Calls POST /tools/git_tool with different actions.
 */
import { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { GitBranch, GitCommit, RefreshCw, FileText } from 'lucide-react';

interface GitFile {
  status: string;
  file: string;
}

export function GitPanel({ apiBase }: { apiBase: string }) {
  const { t } = useTranslation();
  const [files, setFiles] = useState<GitFile[]>([]);
  const [log, setLog] = useState('');
  const [commitMsg, setCommitMsg] = useState('');
  const [loading, setLoading] = useState('');
  const [error, setError] = useState('');
  const [branch, setBranch] = useState('');

  const callGit = useCallback(async (action: string, extra?: Record<string, string>) => {
    setLoading(action);
    setError('');
    try {
      const res = await fetch(`${apiBase}/tools/git_tool`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...extra }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return null;
      }
      return data;
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      setLoading('');
    }
  }, [apiBase]);

  const refresh = useCallback(async () => {
    const data = await callGit('status');
    if (data?.result?.stdout) {
      const lines = data.result.stdout.trim().split('\n').filter(Boolean);
      setFiles(lines.map((l: string) => {
        const [status, ...rest] = l.split(/\s+/);
        return { status, file: rest.join(' ') };
      }));
    }
    const logData = await callGit('log');
    if (logData?.result?.stdout) {
      setLog(logData.result.stdout);
      const m = logData.result.stdout.match(/\((.*?)\)/);
      if (m) setBranch(m[1]);
    }
  }, [callGit]);

  useEffect(() => { refresh(); }, [refresh]);

  const doCommit = async () => {
    if (!commitMsg.trim()) return;
    await callGit('add', { files: '.' });
    await callGit('commit', { message: commitMsg });
    setCommitMsg('');
    refresh();
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-border p-3">
        <GitBranch size={14} className="text-text-muted" />
        <span className="text-sm font-medium">{branch || t('git.detached')}</span>
        <button
          onClick={refresh}
          disabled={loading === 'status'}
          className="ml-auto rounded p-1 text-text-muted hover:text-text-primary"
          title={t('git.refresh')}
        >
          <RefreshCw size={14} className={loading === 'status' ? 'animate-spin' : ''} />
        </button>
      </div>

      {error && (
        <div className="m-2 rounded-lg bg-red-500/10 p-2 text-xs text-red-500">{error}</div>
      )}

      <div className="flex-1 overflow-y-auto">
        {files.length === 0 ? (
          <p className="p-4 text-center text-xs text-text-muted">{t('git.clean')}</p>
        ) : (
          <div className="p-2">
            {files.map((f, i) => (
              <div key={i} className="flex items-center gap-2 rounded px-2 py-1 text-xs hover:bg-bg-tertiary">
                <span className="w-5 text-center font-mono" style={{
                  color: f.status === 'M' ? '#f59e0b' : f.status === 'A' ? '#22c55e' : f.status === 'D' ? '#ef4444' : '#888'
                }}>
                  {f.status}
                </span>
                <span className="font-mono text-text-secondary">{f.file}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="border-t border-border p-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={commitMsg}
            onChange={e => setCommitMsg(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && doCommit()}
            placeholder={t('git.commitPlaceholder')}
            className="flex-1 rounded-lg border border-border bg-bg-tertiary px-3 py-1.5 text-sm text-text-primary placeholder-text-muted focus:border-accent focus:outline-none"
          />
          <button
            onClick={doCommit}
            disabled={!commitMsg.trim() || loading === 'commit'}
            className="flex items-center gap-1 rounded-lg bg-accent px-3 py-1.5 text-sm text-white disabled:opacity-40"
            title={t('git.commitTitle')}
          >
            <GitCommit size={14} />
            {t('git.commit')}
          </button>
        </div>

        {log && (
          <details className="mt-2">
            <summary className="cursor-pointer text-xs text-text-muted">
              <FileText size={10} style={{ display: 'inline', marginRight: 4 }} />
              {t('git.recentCommits')}
            </summary>
            <pre className="mt-1 max-h-32 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs text-text-secondary">{log}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
