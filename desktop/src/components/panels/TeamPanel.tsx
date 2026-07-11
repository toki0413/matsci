import { useTranslation } from 'react-i18next';
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
  teamFusionResult: any;
  handleTeamPlan: () => void;
  handleTeamRun: () => void;
  handleTeamFusion: (rounds: number) => void;
}

export function TeamPanel({
  config, setConfig, setConfigDirty, saveConfig, isConnected,
  teamObjective, setTeamObjective, teamRunning, teamError, teamPlan, teamResult,
  teamFusionResult,
  handleTeamPlan, handleTeamRun, handleTeamFusion,
}: TeamPanelProps) {
  const { t } = useTranslation();

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-3xl space-y-5">
          {/* Config */}
          <div className="card">
            <h2 className="mb-2 text-base font-semibold">{t('team.title')}</h2>
            <p className="text-sm text-text-secondary">{t('team.desc')}</p>
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
                {t('team.enable')}
              </label>
              <span className="text-xs text-text-muted">{t('team.hintKeyword')}</span>
            </div>
          </div>

          {/* Objective input */}
          <div className="card space-y-3">
            <label className="block text-xs font-medium text-text-secondary">{t('team.objective')}</label>
            <textarea
              value={teamObjective}
              onChange={(e) => setTeamObjective(e.target.value)}
              placeholder={t('team.placeholder')}
              rows={4}
              disabled={teamRunning}
              className="input resize-none"
            />

            {/* Action buttons */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={handleTeamPlan}
                disabled={!isConnected || teamRunning || !teamObjective.trim()}
                className="btn-secondary px-3 py-1.5 text-xs"
              >
                {t('team.plan')}
              </button>
              <button
                onClick={handleTeamRun}
                disabled={!isConnected || teamRunning || !teamObjective.trim()}
                className="btn-primary px-3 py-1.5 text-xs"
              >
                {teamRunning ? t('team.running') : t('team.run')}
              </button>
              <div className="mx-1 h-5 w-px bg-border" />
              {/* Fusion with rounds control */}
              <div className="flex items-center gap-1">
                <button
                  onClick={() => handleTeamFusion(1)}
                  disabled={!isConnected || teamRunning || !teamObjective.trim()}
                  className="btn-primary px-3 py-1.5 text-xs"
                  style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
                  title={t('fusion.title1')}
                >
                  {t('fusion.button')}
                </button>
                <button
                  onClick={() => handleTeamFusion(2)}
                  disabled={!isConnected || teamRunning || !teamObjective.trim()}
                  className="px-2 py-1.5 text-xs rounded-lg border border-border text-text-secondary hover:text-text-primary"
                  style={{ background: 'linear-gradient(135deg, #8b5cf6, #ec4899)' }}
                  title={t('fusion.title2')}
                >
                  {t('fusion.button2')}
                </button>
                <button
                  onClick={() => handleTeamFusion(3)}
                  disabled={!isConnected || teamRunning || !teamObjective.trim()}
                  className="px-2 py-1.5 text-xs rounded-lg border border-border text-text-secondary hover:text-text-primary"
                  style={{ background: 'linear-gradient(135deg, #ec4899, #f59e0b)' }}
                  title={t('fusion.title3')}
                >
                  {t('fusion.button3')}
                </button>
              </div>
            </div>

            {/* Mode hint */}
            <div className="flex flex-wrap gap-3 text-[10px] text-text-muted">
              <span><b className="text-text-secondary">{t('fusion.hintPlan')}</b></span>
              <span><b className="text-text-secondary">{t('fusion.hintFusion')}</b></span>
              <span><b className="text-text-secondary">{t('fusion.hintRounds')}</b></span>
            </div>

            {teamRunning && (
              <div className="flex items-center gap-2 text-xs text-text-secondary">
                <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                {t('team.working')}
              </div>
            )}
            {teamError && (
              <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">
                {teamError}
              </div>
            )}
          </div>

          {/* Fusion result */}
          {teamFusionResult && (
            <div className="card space-y-4">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <span>{t('fusion.result')}</span>
                <span className="text-xs text-text-muted">
                  {teamFusionResult.panel_responses?.length || 0} {t('fusion.models')} ·
                  {teamFusionResult.rounds > 1 ? ` ${teamFusionResult.rounds} ${t('fusion.rounds')}` : ` 1 ${t('fusion.round')}`} ·
                  {t('fusion.synthesizer')}: {teamFusionResult.synthesizer?.model || '?'}
                </span>
              </h3>

              {/* Multi-round: show all rounds */}
              {teamFusionResult.all_rounds?.length > 1 && (
                <div className="space-y-3">
                  {teamFusionResult.all_rounds.map((round: any[], ri: number) => (
                    <div key={ri} className="space-y-2">
                      <h4 className="text-xs font-semibold text-text-secondary border-b border-border pb-1">
                        📝 {t('fusion.round')} {ri + 1}/{teamFusionResult.all_rounds.length}
                        {ri === 0 && <span className="ml-2 text-text-muted">{t('fusion.independent')}</span>}
                        {ri > 0 && <span className="ml-2 text-text-muted">{t('fusion.reviewPeers')}</span>}
                      </h4>
                      {round.map((r: any, i: number) => (
                        <details key={i} className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs">
                          <summary className="cursor-pointer font-medium flex items-center gap-2">
                            <span className="text-accent">{r.role}</span>
                            <span className="text-text-muted">{r.model}</span>
                            <span className="ml-auto text-text-muted">{r.duration_ms}ms</span>
                          </summary>
                          <div className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap text-text-secondary">
                            {r.answer}
                          </div>
                        </details>
                      ))}
                    </div>
                  ))}
                </div>
              )}

              {/* Single round: show panel responses */}
              {(!teamFusionResult.all_rounds || teamFusionResult.all_rounds.length <= 1) &&
                teamFusionResult.panel_responses?.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-text-secondary">{t('fusion.panelResponses')}</h4>
                  {teamFusionResult.panel_responses.map((r: any, i: number) => (
                    <details key={i} className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs">
                      <summary className="cursor-pointer font-medium flex items-center gap-2">
                        <span className="text-accent">{r.role}</span>
                        <span className="text-text-muted">{r.model}</span>
                        <span className="ml-auto text-text-muted">{r.duration_ms}ms</span>
                      </summary>
                      <div className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap text-text-secondary">
                        {r.answer}
                      </div>
                    </details>
                  ))}
                </div>
              )}

              {/* Consensus */}
              {teamFusionResult.consensus && (
                <div className="rounded-lg border border-green-500/20 bg-green-500/5 p-3">
                  <h4 className="mb-1 text-xs font-semibold text-green-500">{t('fusion.consensus')}</h4>
                  <p className="text-xs text-text-secondary whitespace-pre-wrap">{teamFusionResult.consensus}</p>
                </div>
              )}

              {/* Dissent */}
              {teamFusionResult.dissent && (
                <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3">
                  <h4 className="mb-1 text-xs font-semibold text-amber-500">{t('fusion.divergence')}</h4>
                  <p className="text-xs text-text-secondary whitespace-pre-wrap">{teamFusionResult.dissent}</p>
                </div>
              )}

              {/* Final answer */}
              {teamFusionResult.final_answer && (
                <div className="rounded-lg border border-border bg-bg-tertiary p-3">
                  <h4 className="mb-1 text-xs font-semibold text-text-secondary">{t('fusion.synthesized')}</h4>
                  <p className="text-sm text-text-primary whitespace-pre-wrap">{teamFusionResult.final_answer}</p>
                </div>
              )}
            </div>
          )}

          {/* Planned tasks */}
          {teamPlan && teamPlan.length > 0 && (
            <div className="card space-y-3">
              <h3 className="text-sm font-semibold">{t('team.plannedTasks')}</h3>
              <div className="space-y-2">
                {teamPlan.map((task) => (
                  <div key={task.task_id} className="rounded-lg border border-border bg-bg-tertiary p-3">
                    <div className="flex items-center gap-2 text-xs font-semibold">
                      <span className="text-accent">{task.task_id}</span>
                      <span className="text-text-muted">→</span>
                      <span>{task.agent_id}</span>
                    </div>
                    <p className="mt-1 text-xs text-text-secondary">{task.prompt}</p>
                    {task.depends_on?.length > 0 && (
                      <p className="mt-1 text-[10px] text-text-muted">{t('team.dependsOn')} {task.depends_on.join(", ")}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Team run result */}
          {teamResult && (
            <div className="card space-y-3">
              <h3 className="text-sm font-semibold">{t('team.result')}</h3>
              <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-sm whitespace-pre-wrap">
                {teamResult.summary}
              </div>
              {Object.keys(teamResult.outputs || {}).length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-text-secondary">{t('team.subOutputs')}</h4>
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
