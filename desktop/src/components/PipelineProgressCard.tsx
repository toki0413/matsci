import { useState } from "react";
import { Check, Loader2, Circle, AlertCircle, ChevronDown, FlaskConical, Cpu } from "lucide-react";
import type { Message } from "../hooks/useChatAndConnection";

// stage status → icon + color
function StageIcon({ status }: { status: string }) {
  switch (status) {
    case "done":
      return <Check className="w-4 h-4 text-emerald-500 shrink-0" />;
    case "running":
      return <Loader2 className="w-4 h-4 text-accent shrink-0 animate-spin" />;
    case "error":
      return <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />;
    default:
      return <Circle className="w-4 h-4 text-text-muted/40 shrink-0" />;
  }
}

const PIPELINE_LABELS: Record<string, string> = {
  deli_research: "DeliAutoResearch",
  computational_loop: "Computational Loop",
};

export function PipelineProgressCard({ msg }: { msg: Message }) {
  const [expanded, setExpanded] = useState(true);
  const stages = msg.pipelineStages || [];
  const pct = msg.pipelineProgressPct || 0;
  const isDone = stages.length > 0 && stages.every((s) => s.status === "done");

  const pipelineLabel = PIPELINE_LABELS[msg.pipelineName || ""] || msg.pipelineName || "Pipeline";
  const PipelineIcon = msg.pipelineName === "computational_loop" ? Cpu : FlaskConical;

  const doneCount = stages.filter((s) => s.status === "done").length;
  const currentStage = stages.find((s) => s.status === "running");

  return (
    <div
      className={`rounded-lg border ${
        isDone ? "border-emerald-500/30 bg-emerald-500/5" : "border-accent/30 bg-accent/5"
      } my-2 overflow-hidden`}
    >
      {/* header */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-bg-secondary/50 transition-colors"
      >
        <PipelineIcon className="w-4 h-4 text-accent shrink-0" />
        <span className="font-semibold text-sm flex-1 text-left">
          {pipelineLabel}
          {msg.pipelineTopic && (
            <span className="text-text-muted font-normal ml-2">
              {msg.pipelineTopic.length > 50
                ? msg.pipelineTopic.slice(0, 50) + "…"
                : msg.pipelineTopic}
            </span>
          )}
        </span>
        {/* progress bar */}
        <div className="flex items-center gap-2">
          <div className="w-24 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${
                isDone ? "bg-emerald-500" : "bg-accent"
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-xs text-text-muted tabular-nums w-8 text-right">
            {pct}%
          </span>
        </div>
        {/* stage count badge */}
        <span className="text-xs text-text-muted px-1.5 py-0.5 rounded bg-bg-tertiary tabular-nums">
          {doneCount}/{stages.length}
        </span>
        <ChevronDown
          className={`w-4 h-4 text-text-muted transition-transform ${
            expanded ? "rotate-180" : ""
          }`}
        />
      </button>

      {/* stages list */}
      {expanded && (
        <div className="px-3 pb-2 space-y-0.5">
          {stages.map((stage, idx) => (
            <div
              key={idx}
              className={`flex items-start gap-2 py-1 px-2 rounded ${
                stage.status === "running"
                  ? "bg-accent/10"
                  : ""
              }`}
            >
              <div className="mt-0.5">
                <StageIcon status={stage.status} />
              </div>
              <div className="flex-1 min-w-0">
                <div
                  className={`text-sm ${
                    stage.status === "pending"
                      ? "text-text-muted/50"
                      : stage.status === "done"
                      ? "text-text-muted"
                      : "text-text-primary"
                  }`}
                >
                  {stage.label || stage.name || `Stage ${idx + 1}`}
                </div>
                {stage.detail && (
                  <div className="text-xs text-text-muted/70 truncate">
                    {stage.detail}
                  </div>
                )}
              </div>
            </div>
          ))}
          {/* current status line */}
          {currentStage && (
            <div className="text-xs text-accent/70 italic px-2 pt-1">
              → {currentStage.detail || currentStage.label}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
