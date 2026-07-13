/**
 * Enhanced autoloop phase progress — 7-step stepper with labels,
 * click-to-expand detail, and overall percentage.
 */
import { useState } from "react";
import { ChevronDown, ChevronRight, Activity } from "lucide-react";

const PHASES = ["perceive", "hypothesize", "plan", "execute", "validate", "learn", "report"] as const;

const PHASE_ICONS: Record<string, string> = {
  perceive: "👁",
  hypothesize: "💡",
  plan: "📋",
  execute: "⚙",
  validate: "✓",
  learn: "📚",
  report: "📝",
};

interface Props {
  currentPhase: string;
  progress: number | string;
}

export default function AutoloopProgress({ currentPhase, progress }: Props) {
  const [expanded, setExpanded] = useState(false);
  const currentIdx = PHASES.indexOf(currentPhase as typeof PHASES[number]);
  const pct = typeof progress === "number" ? progress : parseInt(String(progress), 10) || 0;

  return (
    <div className="border-b border-border bg-bg-secondary">
      {/* Summary row — always visible */}
      <button
        type="button"
        className="flex items-center gap-3 px-6 py-2 cursor-pointer select-none w-full text-left focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:outline-none"
        onClick={() => setExpanded(v => !v)}
        aria-expanded={expanded}
        aria-label="Toggle autoloop progress details"
      >
        <Activity size={14} className="text-accent shrink-0" aria-hidden="true" />
        <span className="text-xs font-semibold text-text-secondary whitespace-nowrap">
          Autoloop
        </span>

        {/* Stepper — horizontal dots with connecting bars */}
        <div className="flex items-center gap-0.5 flex-1 min-w-0">
          {PHASES.map((phase, i) => {
            const isCurrent = i === currentIdx;
            const isDone = i < currentIdx;
            const isLast = i === PHASES.length - 1;
            return (
              <div key={phase} className="flex items-center gap-0.5 min-w-0">
                <div className="flex flex-col items-center gap-0.5 shrink-0">
                  <div
                    className={`h-2.5 w-2.5 rounded-full transition-[background-color,transform] duration-300 ${
                      isCurrent
                        ? "bg-accent ring-2 ring-accent/30 scale-125"
                        : isDone
                        ? "bg-accent/50"
                        : "bg-bg-tertiary"
                    }`}
                    title={phase}
                  />
                </div>
                {!isLast && (
                  <div
                    className={`h-0.5 w-6 transition-colors duration-300 ${
                      isDone ? "bg-accent/30" : "bg-border"
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>

        <span className="text-xs font-medium text-accent whitespace-nowrap">
          {currentPhase}
        </span>
        <span className="text-xs text-text-muted tabular-nums whitespace-nowrap">
          {pct}%
        </span>

        {expanded ? (
          <ChevronDown size={14} className="text-text-muted shrink-0" aria-hidden="true" />
        ) : (
          <ChevronRight size={14} className="text-text-muted shrink-0" aria-hidden="true" />
        )}
      </button>

      {/* Expanded detail — phase list with status */}
      {expanded && (
        <div className="px-6 pb-3 pt-1 grid grid-cols-7 gap-1">
          {PHASES.map((phase, i) => {
            const isCurrent = i === currentIdx;
            const isDone = i < currentIdx;
            return (
              <div
                key={phase}
                className={`flex flex-col items-center gap-1 rounded-md py-1.5 transition-colors ${
                  isCurrent
                    ? "bg-accent/10 border border-accent/20"
                    : "bg-transparent"
                }`}
              >
                <span className="text-base leading-none" aria-hidden="true">{PHASE_ICONS[phase]}</span>
                <span
                  className={`text-[10px] capitalize ${
                    isCurrent
                      ? "text-accent font-semibold"
                      : isDone
                      ? "text-text-secondary"
                      : "text-text-muted"
                  }`}
                >
                  {phase}
                </span>
                {isCurrent && (
                  <span className="text-[9px] text-accent animate-pulse motion-reduce:animate-none">●</span>
                )}
                {isDone && (
                  <span className="text-[9px] text-accent/50">✓</span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
