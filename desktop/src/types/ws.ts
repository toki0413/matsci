/**
 * WebSocket message shapes pushed by the Huginn backend.
 *
 * Every frame is a JSON object tagged with a `type` string. Modeling the
 * whole set as a discriminated union lets the UI `switch` on `type` and
 * reach into the payload without `any` casts — the compiler narrows the
 * variant in each case branch.
 */

// ── Shared sub-types ──────────────────────────────────────────────

export interface PlanStep {
  name?: string;
  description?: string;
}

export interface PlanData {
  summary?: string;
  steps?: PlanStep[];
  // criteria can arrive as bare strings or as { criterion } objects
  acceptance_criteria?: Array<string | { criterion: string }>;
  tools_needed?: string[];
}

export interface CriterionResult {
  criterion: string;
  passed: boolean;
}

export interface ClarificationQuestion {
  question_id?: string;
  question?: string;
  options?: string[];
}

export interface Citation {
  ref: string | number;
  filename: string;
  distance?: number;
}

export interface ExplorationResult {
  best_branch?: { name: string };
  convergence_reason?: string;
}

// ── Discriminated union ───────────────────────────────────────────

export type WSMessage =
  | { type: "text_delta"; text: string }
  | { type: "reasoning_delta"; text: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; id: string; content: string }
  | { type: "plan"; plan_id: string; plan: PlanData }
  | { type: "plan_confirm"; plan_id: string; confirmed: boolean }
  | { type: "plan_result"; plan_id: string; criteria: CriterionResult[]; all_passed: boolean }
  | {
      type: "clarification_request";
      questions: ClarificationQuestion[];
      thread_id: string;
    }
  | { type: "clarification_response"; question_id: string; answer: string }
  | { type: "citations"; sources: Citation[] }
  | {
      type: "task_progress";
      task_type: string;
      job_id?: string;
      status?: string;
      completed?: number;
      total?: number;
      progress_pct?: number;
    }
  | { type: "sediment"; stored: boolean; preview?: string }
  | {
      type: "approval_request";
      request_id: string;
      tool_name: string;
      reason: string;
      auto_approved: boolean;
      dangerous: boolean;
    }
  | { type: "auto_approve_set"; enabled?: boolean; scope?: string }
  | {
      type: "hook_warning";
      tool_name: string;
      warnings: Array<{ severity: string; message: string }>;
    }
  | { type: "exploration_result"; data?: ExplorationResult }
  | { type: "auto_checkpoint"; id: string; base: string; files: number }
  | {
      type: "agent_status";
      task_id?: string;
      agent_id?: string;
      status?: string;
      output?: string;
    }
  | { type: "ping" }
  | { type: "pong" }
  | { type: "done" }
  | { type: "error"; error: string };

/**
 * Minimal runtime guard so we can hand `unknown` WS frames to the typed
 * handler without an `as` cast. Only asserts the presence of `type`; the
 * switch in the handler silently ignores anything it does not recognize,
 * matching the previous behaviour.
 */
export function isWSMessage(data: unknown): data is WSMessage {
  return typeof data === "object" && data !== null && "type" in data;
}
