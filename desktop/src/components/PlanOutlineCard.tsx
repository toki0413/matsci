import { Loader2, Check } from 'lucide-react';

export interface PlanStepView {
  name?: string;
  description?: string;
}

interface PlanOutlineCardProps {
  steps?: PlanStepView[];
  criteria?: string[];
  tools?: string[];
  status?: 'executing' | 'done' | null;
  confirmed?: boolean;
  onRevise?: () => void;
}

/**
 * Renders an agent-produced plan as a visible, numbered outline checklist —
 * the OpenMAIC-style "show me what you'll build, then let me approve" moment.
 *
 * The agent's structured steps (already carried on plan messages as `planData`)
 * become scannable rows with a status cue, acceptance criteria, and suggested
 * tools, plus a "Revise" affordance to steer before execution kicks off.
 */
export default function PlanOutlineCard({
  steps = [],
  criteria = [],
  tools = [],
  status,
  confirmed = false,
  onRevise,
}: PlanOutlineCardProps) {
  const overridden = status === 'executing' || status === 'done';
  return (
    <div className="mb-3 rounded-xl border border-border bg-bg-tertiary/40 p-3" data-testid="plan-outline">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-bold text-text-primary">📋 Outline</span>
        {status === 'executing' && (
          <span className="flex items-center gap-1 text-xs text-accent">
            <Loader2 size={12} className="animate-spin" aria-hidden="true" /> Executing…
          </span>
        )}
        {status === 'done' && (
          <span className="flex items-center gap-1 text-xs text-success">
            <Check size={12} aria-hidden="true" /> Executed
          </span>
        )}
        {overridden && (
          <span className="ml-auto rounded bg-bg-tertiary px-1.5 py-0.5 font-mono text-[10px] text-text-muted">
            {status}
          </span>
        )}
      </div>

      {steps.length === 0 ? (
        <p className="text-xs text-text-muted">No steps in this plan yet.</p>
      ) : (
        <ol className="space-y-1.5">
          {steps.map((step, i) => (
            <li key={`${i}-${step.name ?? 'step'}`} className="flex items-start gap-2">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent/15 font-mono text-[11px] font-bold text-accent">
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <span className="block text-sm font-medium text-text-primary">
                  {step.name || `Step ${i + 1}`}
                </span>
                {step.description && (
                  <span className="block text-xs text-text-secondary">{step.description}</span>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}

      {criteria.length > 0 && (
        <div className="mt-2 border-t border-border/50 pt-2">
          <span className="mb-1 block text-[11px] font-bold uppercase tracking-widest text-text-muted">
            Acceptance
          </span>
          {criteria.map((c, i) => (
            <div key={`${i}-${c}`} className="text-xs text-text-secondary">✅ {c}</div>
          ))}
        </div>
      )}

      {tools.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 border-t border-border/50 pt-2">
          <span className="text-[11px] font-bold uppercase tracking-widest text-text-muted">
            Tools
          </span>
          {tools.map((tool) => (
            <span key={tool} className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">
              🔧 {tool}
            </span>
          ))}
        </div>
      )}

      {onRevise && !overridden && !confirmed && (
        <div className="mt-3 flex justify-end">
          <button
            onClick={onRevise}
            className="rounded-lg border border-border px-3 py-1.5 text-xs text-text-secondary transition-colors hover:bg-bg-tertiary hover:text-text-primary"
          >
            ✏️ Revise outline
          </button>
        </div>
      )}
    </div>
  );
}