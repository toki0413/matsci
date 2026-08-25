import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import type { AppConfig } from '../../types/domain';
import type { TeamRunStatus } from '../../hooks/useChatAndConnection';

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
  teamRuns: Record<string, TeamRunStatus>;
}

const ROLE_EMOJI: Record<string, string> = {
  planner: '🧭', scientist: '🔬', coder: '💻', executor: '⚙️',
  critic: '🕵️', vision: '👁️', synthesizer: '🧩',
};

function fmtMs(ms?: number): string {
  if (ms === undefined) return '–';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function fmtTokens(tokens?: Record<string, number>): string {
  if (!tokens) return '';
  const total = Object.values(tokens).reduce((s, v) => s + v, 0);
  return total > 0 ? `${total.toLocaleString()} tok` : '';
}

export function TeamPanel({
  config, setConfig, setConfigDirty, saveConfig, isConnected,
  teamObjective, setTeamObjective, teamRunning, teamError, teamPlan, teamResult,
  teamFusionResult,
  handleTeamPlan, handleTeamRun, handleTeamFusion,
  teamRuns,
}: TeamPanelProps) {
  const { t } = useTranslation();
  // 可展开的成员节点: `${runId}:${role}` → 展开看工具调用序列
  const [expanded, setExpanded] = useState<string | null>(null);

  const runs = Object.values(teamRuns).sort((a, b) => b.startedAt - a.startedAt);

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

          {/* ── 子任务面板: 实时并行状态 / 耗时 / token / 工具调用序列 ── */}
          {runs.length > 0 && (
            <div className="card space-y-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <span>🛰️ Sub-tasks</span>
                <span className="text-xs text-text-muted">{runs.length} run(s) · live SSE</span>
              </h3>
              {runs.map((run) => {
                const members = Object.values(run.members);
                const running = members.filter((m) => m.status === "running").length;
                const done = members.filter((m) => m.status === "done").length;
                return (
                  <div key={run.run_id} className="rounded-lg border border-border bg-bg-tertiary p-3 space-y-2">
                    {/* Run header */}
                    <div className="flex items-center gap-2 text-xs">
                      <span className={`inline-flex h-2 w-2 rounded-full ${run.status === "running" ? "bg-amber-400 animate-pulse" : "bg-emerald-400"}`} />
                      <span className="font-mono text-text-muted">{run.run_id}</span>
                      <span className="text-text-secondary truncate flex-1">{run.task || "—"}</span>
                      <span className="text-text-muted whitespace-nowrap">
                        {running > 0 && <span className="text-amber-400">{running} running</span>}
                        {running > 0 && done > 0 && <span className="mx-1">·</span>}
                        {done > 0 && <span className="text-emerald-400">{done} done</span>}
                      </span>
                    </div>
                    {/* Member nodes */}
                    {members.length === 0 && (
                      <div className="text-[11px] text-text-muted italic">Waiting for members to start…</div>
                    )}
                    {members.map((m) => {
                      const key = `${run.run_id}:${m.role}`;
                      const isOpen = expanded === key;
                      const statusColor = m.status === "running" ? "text-amber-400"
                        : m.status === "failed" ? "text-red-400" : "text-emerald-400";
                      return (
                        <div key={key} className="rounded border border-border/70 bg-bg-secondary/60 px-2 py-1.5">
                          <button
                            type="button"
                            onClick={() => setExpanded(isOpen ? null : key)}
                            className="flex w-full items-center gap-2 text-left"
                          >
                            <span className="text-xs">{isOpen ? "▾" : "▸"}</span>
                            <span className="text-sm">{ROLE_EMOJI[m.role] || '🤖'}</span>
                            <span className="text-xs font-semibold text-accent">{m.role}</span>
                            {m.model && <span className="text-[10px] text-text-muted">{m.model}</span>}
                            <span className={`ml-auto text-[10px] font-medium ${statusColor}`}>
                              {m.status === "running" ? "● running" : m.status === "failed" ? "✕ failed" : "✓ done"}
                            </span>
                            <span className="text-[10px] text-text-muted tabular-nums w-16 text-right">{fmtMs(m.duration_ms)}</span>
                            {fmtTokens(m.tokens) && (
                              <span className="text-[10px] text-text-muted tabular-nums">{fmtTokens(m.tokens)}</span>
                            )}
                          </button>
                          {/* Expanded: tool call sequence + current step */}
                          {isOpen && (
                            <div className="mt-2 ml-6 space-y-1">
                              {m.task && (
                                <div className="text-[11px] text-text-secondary">
                                  <span className="text-text-muted">step:</span> {m.task}
                                </div>
                              )}
                              <div className="text-[10px] text-text-muted">tool calls: {m.toolCalls.length}</div>
                              {m.toolCalls.length === 0 ? (
                                <div className="text-[10px] text-text-muted italic">No tool calls yet.</div>
                              ) : (
                                m.toolCalls.slice(-12).map((tc, i) => (
                                  <div key={i} className="flex items-center gap-2 text-[11px]">
                                    <span className="text-accent/80">#{i + 1}</span>
                                    <code className="rounded bg-bg-tertiary px-1.5 py-0.5 text-[10px] text-text-secondary">{tc.tool}</code>
                                    {tc.args && (
                                      <span className="text-[10px] text-text-muted truncate max-w-[240px]">{tc.args}</span>
                                    )}
                                  </div>
                                ))
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          )}

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
                        📝 {t('fusion.roundLabel')} {ri + 1}/{teamFusionResult.all_rounds.length}
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
