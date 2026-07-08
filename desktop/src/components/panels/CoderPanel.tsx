interface CoderPanelProps {
  isConnected: boolean;
  coderTask: string;
  setCoderTask: (v: string) => void;
  coderAutoApprove: boolean;
  setCoderAutoApprove: (v: boolean) => void;
  coderMaxIters: number | "";
  setCoderMaxIters: (v: number | "") => void;
  coderRunning: boolean;
  coderError: string;
  coderResult: string | null;
  handleCoderRun: () => void;
}

export function CoderPanel({
  isConnected, coderTask, setCoderTask, coderAutoApprove, setCoderAutoApprove,
  coderMaxIters, setCoderMaxIters, coderRunning, coderError, coderResult,
  handleCoderRun,
}: CoderPanelProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-3xl space-y-5">
          <div className="card">
            <h2 className="mb-2 text-base font-semibold">💻 Coder Mode</h2>
            <p className="text-sm text-text-secondary">
              Give Muninn a coding task and let it read, write, edit, run shell commands, and commit changes autonomously.
            </p>
          </div>

          <div className="card space-y-3">
            <label className="block text-xs font-medium text-text-secondary">Task</label>
            <textarea
              value={coderTask}
              onChange={(e) => setCoderTask(e.target.value)}
              placeholder="e.g. Refactor the VASP parser to use the Rust extension, then add a test."
              rows={5}
              disabled={coderRunning}
              className="input resize-none"
            />
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex cursor-pointer items-center gap-2 text-sm text-text-primary">
                <input
                  type="checkbox"
                  checked={coderAutoApprove}
                  onChange={(e) => setCoderAutoApprove(e.target.checked)}
                  disabled={coderRunning}
                  className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                />
                Auto-approve destructive actions
              </label>
              <div className="flex items-center gap-2">
                <label className="text-xs text-text-secondary">Max iterations</label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={coderMaxIters}
                  onChange={(e) => setCoderMaxIters(e.target.value === "" ? "" : parseInt(e.target.value, 10))}
                  disabled={coderRunning}
                  placeholder="default"
                  className="input w-24 px-2 py-1 text-xs"
                />
              </div>
            </div>
            <button
              onClick={handleCoderRun}
              disabled={!isConnected || coderRunning || !coderTask.trim()}
              className="btn-primary px-4 py-1.5 text-xs"
            >
              {coderRunning ? "Coding…" : "▶ Run coder"}
            </button>
            {coderRunning && (
              <div className="flex items-center gap-2 text-xs text-text-secondary">
                <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                Muninn is coding…
              </div>
            )}
            {coderError && (
              <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">
                {coderError}
              </div>
            )}
          </div>

          {coderResult && (
            <div className="card space-y-3">
              <h3 className="text-sm font-semibold">Result</h3>
              <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-sm whitespace-pre-wrap">
                {coderResult}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
