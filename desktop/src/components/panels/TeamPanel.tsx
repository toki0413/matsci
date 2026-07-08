import type { AppConfig } from '../../types/domain';

interface TeamPanelProps {
  config: AppConfig;
  setConfig: (c: AppConfig) => void;
  setConfigDirty: (v: boolean) => void;
  saveConfig: (c: AppConfig) => void;
  isConnected: boolean;
  teamObjective: string;
  setTeamObjective: (v: string) => void;
  teamRunning: boolean;
  teamError: string;
  teamPlan: any[];
  teamResult: any;
  handleTeamPlan: () => void;
  handleTeamRun: () => void;
}

export function TeamPanel({
  config, setConfig, setConfigDirty, saveConfig, isConnected,
  teamObjective, setTeamObjective, teamRunning, teamError, teamPlan, teamResult,
  handleTeamPlan, handleTeamRun,
}: TeamPanelProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-3xl space-y-5">
          <div className="card">
            <h2 className="mb-2 text-base font-semibold">👥 Multi-Agent Team</h2>
            <p className="text-sm text-text-secondary">
              The lead agent breaks your objective into subtasks and delegates them to the configured agent profiles.
            </p>
            <div className="mt-4 flex items-center gap-2">
              <label className="flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
                <input
                  type="checkbox"
                  checked={config.team_mode_enabled}
                  onChange={(e) => {
                    const next = { ...config, team_mode_enabled: e.target.checked };
                    setConfig(next);
                    setConfigDirty(true);
                    saveConfig(next);
                  }}
                />
                Enable team mode
              </label>
            </div>
          </div>

          <div className="card space-y-3">
            <label className="block text-xs font-medium text-text-secondary">Objective</label>
            <textarea
              value={teamObjective}
              onChange={(e) => setTeamObjective(e.target.value)}
              placeholder="e.g. Compare VASP and Quantum ESPRESSO for silicon band structure, then suggest which is cheaper for a 50-atom cell."
              rows={4}
              disabled={teamRunning}
              className="input resize-none"
            />
            <div className="flex items-center gap-2">
              <button
                onClick={handleTeamPlan}
                disabled={!isConnected || teamRunning || !teamObjective.trim()}
                className="btn-secondary px-3 py-1.5 text-xs"
              >
                📋 Plan
              </button>
              <button
                onClick={handleTeamRun}
                disabled={!isConnected || teamRunning || !teamObjective.trim()}
                className="btn-primary px-3 py-1.5 text-xs"
              >
                {teamRunning ? "Running…" : "▶ Run team"}
              </button>
            </div>
            {teamRunning && (
              <div className="flex items-center gap-2 text-xs text-text-secondary">
                <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                Working with the team…
              </div>
            )}
            {teamError && (
              <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">
                {teamError}
              </div>
            )}
          </div>

          {teamPlan && teamPlan.length > 0 && (
            <div className="card space-y-3">
              <h3 className="text-sm font-semibold">Planned tasks</h3>
              <div className="space-y-2">
                {teamPlan.map((t) => (
                  <div key={t.task_id} className="rounded-lg border border-border bg-bg-tertiary p-3">
                    <div className="flex items-center gap-2 text-xs font-semibold">
                      <span className="text-accent">{t.task_id}</span>
                      <span className="text-text-muted">→</span>
                      <span>{t.agent_id}</span>
                    </div>
                    <p className="mt-1 text-xs text-text-secondary">{t.prompt}</p>
                    {t.depends_on?.length > 0 && (
                      <p className="mt-1 text-[10px] text-text-muted">Depends on: {t.depends_on.join(", ")}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {teamResult && (
            <div className="card space-y-3">
              <h3 className="text-sm font-semibold">Result</h3>
              <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-sm whitespace-pre-wrap">
                {teamResult.summary}
              </div>
              {Object.keys(teamResult.outputs || {}).length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-text-secondary">Sub-agent outputs</h4>
                  {Object.entries(teamResult.outputs).map(([taskId, output]: [string, any]) => (
                    <details key={taskId} className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs">
                      <summary className="cursor-pointer font-medium">{taskId}</summary>
                      <div className="mt-2 whitespace-pre-wrap text-text-secondary">{String(output)}</div>
                    </details>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
