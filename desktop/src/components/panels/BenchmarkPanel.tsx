interface BenchmarkPanelProps {
  isConnected: boolean;
  benchEvolve: boolean;
  setBenchEvolve: (v: boolean) => void;
  benchCategories: string;
  setBenchCategories: (v: string) => void;
  benchRunning: boolean;
  benchError: string;
  benchResult: any;
  benchRun: () => void;
}

export function BenchmarkPanel({
  isConnected, benchEvolve, setBenchEvolve, benchCategories, setBenchCategories,
  benchRunning, benchError, benchResult, benchRun,
}: BenchmarkPanelProps) {
  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mx-auto max-w-3xl space-y-5">
        <div className="card">
          <h2 className="mb-2 text-base font-semibold">Benchmark</h2>
          <p className="text-sm text-text-secondary">Run standardized tasks and measure pass rate.</p>
        </div>
        <div className="card space-y-3">
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={benchEvolve} onChange={(e) => setBenchEvolve(e.target.checked)} className="h-4 w-4 rounded border-border" />
            Run evolution cycle afterward
          </label>
          <input
            type="text"
            value={benchCategories}
            onChange={(e) => setBenchCategories(e.target.value)}
            placeholder="Categories, comma separated (empty = all)"
            className="input text-sm"
          />
          <button onClick={benchRun} disabled={benchRunning || !isConnected} className="btn-primary text-xs">
            {benchRunning ? "Running…" : "▶ Run benchmark"}
          </button>
          {benchError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{benchError}</div>}
        </div>
        {benchResult && (
          <div className="card space-y-3">
            <h3 className="text-sm font-semibold">Report</h3>
            <div className="text-xs text-text-secondary">
              Pass rate: {(benchResult.metrics?.pass_rate * 100).toFixed(0)}% · Total: {benchResult.total} · Passed: {benchResult.passed} · Failed: {benchResult.failed} · Skipped: {benchResult.skipped}
            </div>
            <div className="space-y-2">
              {(benchResult.results || []).map((r: any) => (
                <div key={r.task_id} className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs">
                  <span className={`font-semibold ${r.passed ? "text-success" : "text-error"}`}>{r.passed ? "✓" : "✗"}</span>{" "}
                  <span className="font-mono">{r.task_id}</span> — {r.reason}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
