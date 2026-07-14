/**
 * Decision trace panel — visualizes agent's decision-making process.
 *
 * Shows three layers:
 * 1. State machine: phase transitions with iteration count
 * 2. Action chain: each tool call with governance status (allowed/risk/verified)
 * 3. Predictability graph: local contributions decomposed per action
 *
 * Inspired by: "what action is executable, verifiable, traceable, controllable"
 */
import { useState, useMemo } from "react";
import {
  GitBranch, ShieldCheck, ShieldAlert, ShieldX,
  CheckCircle2, XCircle, AlertCircle, ChevronDown, ChevronRight,
  Activity, FileSearch, Cpu, FileEdit, Code, Network, BookOpen, MessageSquare
} from "lucide-react";

// Format predictability scores consistently across the panel
const fmtPred = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

interface GovernanceEntry {
  action_name: string;
  category: string;
  risk_level: string;
  allowed: boolean;
  reasons: string[];
  requires_approval: boolean;
  predictability: number;
  audit_id?: string;
  status?: string;
  verification_passed?: boolean;
  verification_message?: string;
  rollback_available?: boolean;
}

interface StateTransition {
  from_phase: string;
  to_phase: string;
  reason: string;
  iteration: number;
}

interface Props {
  governanceEvents: GovernanceEntry[];
  stateTransitions: StateTransition[];
  currentPhase: string;
  activeTraceId?: string;
}

const CATEGORY_ICONS: Record<string, typeof Cpu> = {
  query: FileSearch,
  analyze: Activity,
  simulate: Cpu,
  file_ops: FileEdit,
  code: Code,
  network: Network,
  learn: BookOpen,
  communicate: MessageSquare,
};

const RISK_COLORS: Record<string, string> = {
  none: "text-emerald-400",
  low: "text-blue-400",
  medium: "text-yellow-400",
  high: "text-orange-400",
  critical: "text-red-400",
};

const RISK_BG: Record<string, string> = {
  none: "bg-emerald-500/10 border-emerald-500/30",
  low: "bg-blue-500/10 border-blue-500/30",
  medium: "bg-yellow-500/10 border-yellow-500/30",
  high: "bg-orange-500/10 border-orange-500/30",
  critical: "bg-red-500/10 border-red-500/30",
};

const PHASE_FLOW = ["literature", "hypothesis", "planning", "execution", "validation", "reporting"];

export default function DecisionTracePanel({ governanceEvents, stateTransitions, currentPhase, activeTraceId }: Props) {
  const [tab, setTab] = useState<"actions" | "state" | "predictability">("actions");
  const [expanded, setExpanded] = useState<string | null>(null);

  // Aggregate stats
  const stats = useMemo(() => {
    const total = governanceEvents.length;
    const allowed = governanceEvents.filter(e => e.allowed).length;
    const verified = governanceEvents.filter(e => e.verification_passed).length;
    const blocked = governanceEvents.filter(e => !e.allowed).length;
    const avgPred = total > 0
      ? governanceEvents.reduce((s, e) => s + e.predictability, 0) / total
      : 0;
    return { total, allowed, verified, blocked, avgPred };
  }, [governanceEvents]);

  return (
    <div className="border-t border-border bg-bg-secondary/80 backdrop-blur-sm">
      {/* Header — collapsed summary */}
      <div className="flex items-center gap-2 px-4 py-1.5 border-b border-border">
        <GitBranch size={14} className="text-accent shrink-0" aria-hidden="true" />
        <span className="text-xs font-semibold text-text-secondary">Decision Trace</span>

        {/* Quick stats */}
        <div className="flex items-center gap-3 ml-auto text-[10px] text-text-muted">
          {/* OAK 启发: trace_id 贯穿展示, 让用户知道当前事件属于哪个研究分支 */}
          {activeTraceId && (
            <span className="flex items-center gap-1 font-mono text-cyan-400" title={`Active trace: ${activeTraceId}`}>
              <GitBranch size={10} aria-hidden="true" />
              {activeTraceId.length > 12 ? `${activeTraceId.slice(0, 8)}…` : activeTraceId}
            </span>
          )}
          <span className="flex items-center gap-1">
            <CheckCircle2 size={10} className="text-emerald-400" aria-hidden="true" />
            {stats.allowed}/{stats.total} allowed
          </span>
          <span className="flex items-center gap-1">
            <ShieldCheck size={10} className="text-blue-400" aria-hidden="true" />
            {stats.verified} verified
          </span>
          {stats.blocked > 0 && (
            <span className="flex items-center gap-1">
              <ShieldX size={10} className="text-red-400" aria-hidden="true" />
              {stats.blocked} blocked
            </span>
          )}
          <span className="flex items-center gap-1">
            <Activity size={10} className="text-accent" aria-hidden="true" />
            P=<span className="tabular-nums">{stats.avgPred.toFixed(2)}</span>
          </span>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center gap-0.5 ml-2" role="tablist" aria-label="Decision trace views">
          {(["actions", "state", "predictability"] as const).map(t => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={`px-2 py-0.5 text-[10px] rounded transition-colors focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:outline-none ${
                tab === t
                  ? "bg-accent/20 text-accent"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {t === "actions" ? "Actions" : t === "state" ? "State" : "Predict"}
            </button>
          ))}
        </div>
      </div>

      {/* Content — tab-dependent */}
      <div className="max-h-48 overflow-y-auto">
        {tab === "actions" && (
          <ActionChain
            events={governanceEvents}
            expanded={expanded}
            setExpanded={setExpanded}
          />
        )}
        {tab === "state" && (
          <StateMachineView
            transitions={stateTransitions}
            currentPhase={currentPhase}
          />
        )}
        {tab === "predictability" && (
          <PredictabilityView events={governanceEvents} />
        )}
      </div>
    </div>
  );
}

// ── Action chain view ─────────────────────────────────────────

function ActionChain({
  events,
  expanded,
  setExpanded,
}: {
  events: GovernanceEntry[];
  expanded: string | null;
  setExpanded: (v: string | null) => void;
}) {
  if (events.length === 0) {
    return (
      <div className="px-4 py-3 text-xs text-text-muted italic">
        No governance events yet — actions will appear here as the agent executes.
      </div>
    );
  }

  return (
    <div className="px-3 py-1.5 space-y-1">
      {events.slice(-20).map((evt, i) => {
        const Icon = CATEGORY_ICONS[evt.category] || Activity;
        const isExpanded = expanded === `${i}-${evt.action_name}`;
        const riskColor = RISK_COLORS[evt.risk_level] || "text-text-muted";
        const riskBg = RISK_BG[evt.risk_level] || "bg-bg-tertiary border-border";

        return (
          <div key={`${i}-${evt.action_name}`} className={`rounded border ${riskBg} px-2 py-1`}>
            <button
              type="button"
              className="flex items-center gap-2 cursor-pointer w-full text-left focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:outline-none rounded"
              onClick={() => setExpanded(isExpanded ? null : `${i}-${evt.action_name}`)}
              aria-expanded={isExpanded}
            >
              {isExpanded
                ? <ChevronDown size={12} className="text-text-muted shrink-0" aria-hidden="true" />
                : <ChevronRight size={12} className="text-text-muted shrink-0" aria-hidden="true" />
              }
              <Icon size={12} className={`${riskColor} shrink-0`} aria-hidden="true" />
              <span className="text-xs font-medium text-text-primary truncate">
                {evt.action_name}
              </span>

              {/* Status badges */}
              <div className="flex items-center gap-1.5 ml-auto shrink-0">
                {evt.allowed ? (
                  <CheckCircle2 size={11} className="text-emerald-400" aria-hidden="true" />
                ) : (
                  <XCircle size={11} className="text-red-400" aria-hidden="true" />
                )}
                {evt.verification_passed && (
                  <ShieldCheck size={11} className="text-blue-400" aria-hidden="true" />
                )}
                {evt.requires_approval && (
                  <ShieldAlert size={11} className="text-yellow-400" aria-hidden="true" />
                )}
                <span className={`text-[10px] font-mono ${riskColor}`}>
                  {evt.risk_level}
                </span>
                <span className="text-[10px] font-mono text-text-muted tabular-nums">
                  P={evt.predictability.toFixed(2)}
                </span>
              </div>
            </button>

            {/* Expanded detail */}
            {isExpanded && (
              <div className="mt-1 ml-5 space-y-1 text-[11px] text-text-muted">
                {evt.audit_id && (
                  <div className="font-mono text-[10px] text-text-muted/60">
                    audit_id: {evt.audit_id}
                  </div>
                )}
                {evt.reasons.length > 0 && (
                  <div>
                    <span className="text-text-secondary">Reasons:</span>
                    {evt.reasons.map((r, j) => (
                      <div key={j} className="ml-2 flex items-start gap-1">
                        <AlertCircle size={10} className="mt-0.5 shrink-0" aria-hidden="true" />
                        <span>{r}</span>
                      </div>
                    ))}
                  </div>
                )}
                {evt.verification_message && (
                  <div className="text-yellow-400/80">
                    <ShieldAlert size={10} className="inline mr-1" />
                    {evt.verification_message}
                  </div>
                )}
                {evt.rollback_available && (
                  <div className="text-orange-400/80">
                    <ShieldAlert size={10} className="inline mr-1" />
                    Rollback available
                  </div>
                )}
                {evt.status && (
                  <div>
                    <span className="text-text-secondary">Status:</span>{" "}
                    <span className={
                      evt.status === "verified" ? "text-emerald-400" :
                      evt.status === "failed" ? "text-red-400" :
                      evt.status === "rolled_back" ? "text-orange-400" :
                      "text-text-muted"
                    }>
                      {evt.status}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── State machine view ────────────────────────────────────────

function StateMachineView({
  transitions,
  currentPhase,
}: {
  transitions: StateTransition[];
  currentPhase: string;
}) {
  const currentIdx = PHASE_FLOW.indexOf(currentPhase);

  return (
    <div className="px-4 py-2">
      {/* Phase flow diagram */}
      <div className="flex items-center gap-1 mb-3">
        {PHASE_FLOW.map((phase, i) => {
          const isCurrent = phase === currentPhase;
          const isDone = currentIdx >= 0 && i < currentIdx;
          const isFuture = currentIdx >= 0 && i > currentIdx;

          return (
            <div key={phase} className="flex items-center">
              <div
                className={`px-2 py-0.5 text-[10px] rounded transition-all ${
                  isCurrent
                    ? "bg-accent/20 text-accent ring-1 ring-accent/40"
                    : isDone
                    ? "bg-accent/10 text-accent/60"
                    : isFuture
                    ? "bg-bg-tertiary text-text-muted/50"
                    : "bg-bg-tertiary text-text-muted"
                }`}
              >
                {phase}
              </div>
              {i < PHASE_FLOW.length - 1 && (
                <div className={`w-3 h-px ${isDone ? "bg-accent/40" : "bg-border"}`} />
              )}
            </div>
          );
        })}
      </div>

      {/* Transition history */}
      {transitions.length > 0 && (
        <div className="space-y-0.5">
          {transitions.slice(-15).map((t, i) => (
            <div key={i} className="flex items-center gap-2 text-[10px] text-text-muted">
              <span className="font-mono text-text-muted/60">
                iter {t.iteration}
              </span>
              <span className="text-text-secondary">{t.from_phase}</span>
              <span className="text-accent">→</span>
              <span className="text-text-secondary">{t.to_phase}</span>
              <span className="text-text-muted/60 truncate">{t.reason}</span>
            </div>
          ))}
        </div>
      )}
      {transitions.length === 0 && (
        <div className="text-[10px] text-text-muted italic">
          No state transitions recorded yet.
        </div>
      )}
    </div>
  );
}

// ── Predictability decomposition ──────────────────────────────

function PredictabilityView({ events }: { events: GovernanceEntry[] }) {
  if (events.length === 0) {
    return (
      <div className="px-4 py-3 text-xs text-text-muted italic">
        No predictability data yet. Actions will be scored once executed.
      </div>
    );
  }

  const sorted = [...events].sort((a, b) => b.predictability - a.predictability);
  const maxPred = Math.max(...sorted.map(e => e.predictability), 0.01);

  return (
    <div className="px-3 py-2 space-y-1">
      {sorted.slice(-15).map((evt, i) => {
        const pct = (evt.predictability / maxPred) * 100;
        const barColor =
          evt.predictability > 0.7 ? "bg-emerald-500" :
          evt.predictability > 0.4 ? "bg-yellow-500" :
          "bg-red-500";

        return (
          <div key={i} className="flex items-center gap-2">
            <span className="text-[10px] text-text-secondary truncate w-28">
              {evt.action_name}
            </span>
            <div className="flex-1 h-2 bg-bg-tertiary rounded-full overflow-hidden">
              <div
                className={`h-full ${barColor} rounded-full transition-[width] duration-300`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-text-muted w-10 text-right tabular-nums">
              {fmtPred.format(evt.predictability)}
            </span>
          </div>
        );
      })}
      <div className="text-[10px] text-text-muted/60 italic pt-1">
        PNAS insight: predictability decomposes into local contributions.
        Lower scores indicate unmet preconditions or constraint violations.
      </div>
    </div>
  );
}
