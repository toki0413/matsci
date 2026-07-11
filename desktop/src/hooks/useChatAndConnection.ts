/**
 * useChatAndConnection — Chat messages, WebSocket connection, backend lifecycle,
 * threads, personas, and notification management.
 *
 * This is the most complex hook: it owns the WebSocket connection, handles all
 * incoming message types, manages chat state, and coordinates backend startup.
 * External state updates (checkpoints, tools, skills) are delegated via callbacks.
 */
import { useState, useRef, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";
import { playTaskComplete, playError as playErrorSound } from "../sounds";
import { formatTime } from "../lib/constants";
import { ReconnectingWebSocket } from "../lib/ws-client";
import { getAuthToken, getApiBase } from "../lib/api-client";
import { api } from "../lib/api";
import {
  API_BASE, WS_URL, syncBackendUrl, PERSONAS_FALLBACK, wsUrlVersion,
} from "../lib/config-store";
import { isWSMessage, type WSMessage } from "../types/ws";
import type { AppConfig, PersonaSeed, PersonaEmotionResponse } from "../types/domain";
import type { PetStatusState } from "../components/PetStatusWidget";
import i18n from "../i18n";

// ── Types ──────────────────────────────────────────────────────
export interface PipelineStage {
  name: string;
  label: string;
  status: "pending" | "running" | "done" | "error";
  detail?: string;
}

export interface Message {
  role: "user" | "assistant" | "tool";
  content: string;
  timestamp: string;
  tool_name?: string;
  tool_args?: any;
  tool_status?: "running" | "done" | "error";
  tool_result?: string;
  tool_call_id?: string;
  isPlan?: boolean;
  planId?: string;
  planData?: any;
  planConfirmed?: boolean;
  isClarification?: boolean;
  clarifications?: any[];
  isCitation?: boolean;
  citationSources?: any[];
  isTaskProgress?: boolean;
  taskType?: string;
  jobId?: string;
  reasoning?: string;
  isCompacted?: boolean;
  compactBefore?: number;
  compactAfter?: number;
  // pipeline progress
  pipelineName?: string;
  pipelineTopic?: string;
  pipelineStages?: PipelineStage[];
  pipelineProgressPct?: number;
}

export interface Thread {
  id: string;
  label: string;
  created_at: string;
  last_active: string;
}

// ── Hook parameters ────────────────────────────────────────────
interface UseChatAndConnectionParams {
  config: AppConfig;
  activeTab: string;
  pushConfig: (cfg: AppConfig) => Promise<boolean>;
  onAutoCheckpoint: (cp: { id: string; base: string; files: number }) => void;
  onExplorationResult: (data: any) => void;
  toolsLength: number;
  skillsLength: number;
  setTools: (t: any[]) => void;
  setSkills: (s: any[]) => void;
}

export function useChatAndConnection(params: UseChatAndConnectionParams) {
  const { config, activeTab, onAutoCheckpoint, onExplorationResult,
    toolsLength, skillsLength, setTools, setSkills } = params;

  // ── Per-thread message cache (reactive) ─────────────────────
  // AstrBot pattern: messagesBySession Map, each thread has its own list
  const WELCOME_MSG = (): Message => ({
    role: "assistant",
    content: i18n.t('chat.welcomeMsg') || "Welcome to **Huginn**.\n\n*Magic springs from the wellspring of imagination.*\n\nI'm your materials-science research assistant. Set your LLM provider and API key in **Settings** on the left, then start a chat.",
    timestamp: formatTime(),
  });
  const [messagesByThread, setMessagesByThread] = useState<Record<string, Message[]>>({
    desktop: [WELCOME_MSG()],
  });

  // ── Chat message state (derived from activeThread) ───────────
  const [messages, setMessages] = useState<Message[]>([WELCOME_MSG()]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"chat" | "plan" | "build">("chat");
  const [pendingPlan, setPendingPlan] = useState<string>("");
  const [status, setStatus] = useState<string>("connecting…");
  const [isConnected, setIsConnected] = useState(false);
  const [wsReconnecting, setWsReconnecting] = useState(false);
  const [wsFailed, setWsFailed] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [undoWindow, setUndoWindow] = useState(false);
  const undoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pendingResponseRef = useRef<string>("");
  const streamingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [chatSearchOpen, setChatSearchOpen] = useState(false);
  const [chatSearchQuery, setChatSearchQuery] = useState("");

  // ── Guide state ──────────────────────────────────────────────
  const GUIDE_KEY = "huginn:guide:v1";
  const [showGuide, setShowGuide] = useState(false);
  useEffect(() => {
    if (!localStorage.getItem(GUIDE_KEY)) {
      setShowGuide(true);
    }
  }, []);
  const closeGuide = () => {
    localStorage.setItem(GUIDE_KEY, "1");
    setShowGuide(false);
  };

  // ── Approval state ───────────────────────────────────────────
  const [pendingApproval, setPendingApproval] = useState<{
    request_id: string;
    tool_name: string;
    reason: string;
    dangerous: boolean;
  } | null>(null);
  const [autoApprove, setAutoApprove] = useState<boolean>(true);

  // ── Autoloop progress (SSE) ──────────────────────────────────
  const [autoloopPhase, setAutoloopPhase] = useState<string>("");
  const [autoloopProgress, setAutoloopProgress] = useState<number>(0);

  // ── Thinking intensity (per-request override sent to backend) ──
  const [thinkingIntensity, setThinkingIntensity] = useState<"low" | "medium" | "high">("medium");

  // ── Message queue (Kimi-style: send while streaming, drain on done) ──
  const [pendingMessages, setPendingMessages] = useState<string[]>([]);
  const pendingMessagesRef = useRef<string[]>([]);

  // ── Research mode toggle (prepends /research to outgoing messages) ──
  const [researchMode, setResearchMode] = useState(false);

  // ── Sound toggle (persisted in localStorage) ──
  const [soundEnabled, setSoundEnabled] = useState(() => localStorage.getItem('huginn:sound') !== 'false');

  // ── Persona state ────────────────────────────────────────────
  const [personaList, setPersonaList] = useState<{ id: string; label: string; description?: string; avatar?: string }[]>(PERSONAS_FALLBACK);
  const [personaEmotion, setPersonaEmotion] = useState<{ mood: string; valence: number; arousal: number; trust: number } | null>(null);
  const [pendingClarifications, setPendingClarifications] = useState<{ question_id?: string; question: string; options?: string[]; thread_id?: string }[]>([]);

  // ── Pet state (pushed via pet_update WS messages) ────────────
  const [petState, setPetState] = useState<PetStatusState | null>(null);

  // ── Thread state ─────────────────────────────────────────────
  const [threads, setThreads] = useState<Thread[]>([
    { id: "desktop", label: "Default", created_at: "", last_active: "" },
  ]);
  const [activeThread, setActiveThread] = useState<string>("desktop");
  const activeThreadRef = useRef(activeThread);
  useEffect(() => { activeThreadRef.current = activeThread; }, [activeThread]);

  // switch thread: cache current messages, restore target thread's messages
  const switchThread = (threadId: string) => {
    if (threadId === activeThread) return;
    // save current thread's messages to reactive store
    setMessagesByThread((prev) => ({ ...prev, [activeThread]: messages }));
    // restore target thread's messages
    const cached = messagesByThread[threadId];

    // After a page refresh the in-memory cache is gone, so pull from the
    // backend checkpointer when we only have the welcome stub (or nothing).
    if (cached && cached.length > 1) {
      setMessages(cached);
    } else {
      setMessages(cached && cached.length > 0 ? cached : [WELCOME_MSG()]);
      api.get<{ messages?: any[] }>(`/threads/${threadId}/messages`)
        .then((data) => {
          if (data.messages && data.messages.length > 0) {
            const restored: Message[] = data.messages.map((m: any) => ({
              role: m.role as Message["role"],
              content: m.content,
              timestamp: m.timestamp || formatTime(),
              ...(m.tool_name ? { tool_name: m.tool_name } : {}),
              ...(m.tool_call_id ? { tool_call_id: m.tool_call_id } : {}),
            }));
            setMessages(restored);
            setMessagesByThread((prev) => ({ ...prev, [threadId]: restored }));
          }
        })
        .catch(() => { /* backend offline or no history — keep welcome */ });
    }
    setActiveThread(threadId);
  };

  const loadThreads = async () => {
    try {
      const data = await api.get<{ threads?: Thread[] }>("/threads");
      setThreads(data.threads || []);
    } catch (e: any) {
      console.error("[threads] load failed:", e);
    }
  };

  const createThread = async () => {
    try {
      const data = await api.post<{ id: string; label: string }>("/threads", { title: "New thread" });
      // cache current thread before switching
      setMessagesByThread((prev) => ({ ...prev, [activeThread]: messages }));
      setActiveThread(data.id);
      setMessages([
        {
          role: "assistant",
          content: `Started new thread **${data.label}**.`,
          timestamp: formatTime(),
        },
      ]);
      loadThreads();
    } catch (e: any) {
      console.error("[threads] create failed:", e);
    }
  };

  const renameThread = async (id: string, label: string) => {
    try {
      await api.patch(`/threads/${id}`, { label });
      loadThreads();
    } catch (e: any) {
      console.error("[threads] rename failed:", e);
    }
  };

  const deleteThread = async (id: string) => {
    try {
      await api.del(`/threads/${id}`);
      setMessagesByThread((prev) => { const next = { ...prev }; delete next[id]; return next; });
      if (activeThread === id) {
        setActiveThread("desktop");
      }
      loadThreads();
    } catch (e: any) {
      console.error("[threads] delete failed:", e);
    }
  };

  // ── WebSocket ref ────────────────────────────────────────────
  const wsClientRef = useRef<ReconnectingWebSocket | null>(null);

  // ── Notification ─────────────────────────────────────────────
  const notify = useCallback((title: string, body: string) => {
    try {
      if (document.hidden) {
        sendNotification({ title, body });
      }
    } catch {
      // ignore
    }
  }, []);

  // Request OS notification permission once
  useEffect(() => {
    (async () => {
      try {
        let permitted = await isPermissionGranted();
        if (!permitted) {
          const permission = await requestPermission();
          permitted = permission === "granted";
        }
      } catch {
        // notification plugin may not be available in web builds
      }
    })();
  }, []);

  // ── Backend lifecycle ────────────────────────────────────────
  const startBackend = useCallback(async () => {
    setStatus("starting backend…");
    try {
      const result = await invoke("start_backend");
      await syncBackendUrl();
      setStatus(`${result} • waiting for health…`);
    } catch (e: any) {
      // Not in Tauri — backend should already be running externally
      await syncBackendUrl();
      setStatus("external backend • waiting for health…");
    }
  }, []);

  useEffect(() => {
    let alive = true;

    const check = async () => {
      try {
        const s: any = await invoke("get_agent_status");
        if (alive) {
          await syncBackendUrl();
          setStatus(`${s.status} • v${s.version || "0.1.0"}`);
          if (s.status === "ok") return true;
        }
      } catch {
        // Not in Tauri (browser/dev) — try direct HTTP health check
        try {
          const resp = await fetch(`${getApiBase()}/health`, { signal: AbortSignal.timeout(3000) });
          if (resp.ok) {
            const s = await resp.json();
            if (alive) {
              setStatus(`${s.status} • v${s.version || "0.1.0"}`);
              if (s.status === "ok") return true;
            }
          }
        } catch { /* still down */ }
      }
      return false;
    };

    const run = async () => {
      const online = await check();
      if (online) return;
      await startBackend();
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 500));
        if (await check()) return;
      }
      if (alive) setStatus("backend did not come online");
    };

    run();
    return () => { alive = false; };
  }, [startBackend]);

  // ── Stream batching (assistant-UI pattern: batch tokens to reduce renders) ─
  const streamBufferRef = useRef<{ text: string; reasoning: string }>({ text: "", reasoning: "" });
  const rafScheduledRef = useRef(false);

  const flushStreamBuffer = useCallback(() => {
    rafScheduledRef.current = false;
    const buf = streamBufferRef.current;
    if (!buf.text && !buf.reasoning) return;
    // content arrived — cancel the connection-loss watchdog
    if (streamingTimeoutRef.current) {
      clearTimeout(streamingTimeoutRef.current);
      streamingTimeoutRef.current = null;
    }
    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (last && last.role === "assistant" && last.timestamp === "streaming") {
        updated[updated.length - 1] = {
          ...last,
          content: last.content + buf.text,
          reasoning: (last.reasoning || "") + buf.reasoning,
        };
      } else {
        updated.push({
          role: "assistant",
          content: buf.text,
          reasoning: buf.reasoning,
          timestamp: "streaming",
        });
      }
      return updated;
    });
    streamBufferRef.current = { text: "", reasoning: "" };
  }, []);

  const scheduleFlush = useCallback(() => {
    if (!rafScheduledRef.current) {
      rafScheduledRef.current = true;
      requestAnimationFrame(flushStreamBuffer);
    }
  }, [flushStreamBuffer]);

  // ── WebSocket message handler ────────────────────────────────
  const handleWsMessage = (data: WSMessage) => {
    // Route messages to their thread's cache. Non-active thread messages
    // are buffered so switching back doesn't lose them.
    const _BROADCAST_TYPES = new Set(["pet_update", "ping", "auto_approve_set", "context_compacted"]);
    // Backend injects thread_id on all messages at runtime, but TS union
    // only declares it on some variants — cast to avoid narrowing.
    const _tid = (data as any).thread_id as string | undefined;
    if (_tid && _tid !== activeThreadRef.current
        && !_BROADCAST_TYPES.has(data.type)) {
      // Buffer for the other thread instead of dropping
      const otherTid = _tid;
      setMessagesByThread((prev) => {
        const existing = prev[otherTid] || [];
        // Only buffer text/done/error — tool calls are too complex to merge
        if (data.type === "text_delta") {
          const last = existing[existing.length - 1];
          if (last && last.role === "assistant" && last.timestamp === "streaming") {
            last.content += (data as any).text || "";
            return { ...prev, [otherTid]: [...existing] };
          }
          existing.push({ role: "assistant", content: (data as any).text || "", timestamp: "streaming" });
          return { ...prev, [otherTid]: [...existing] };
        }
        if (data.type === "done" || data.type === "error") {
          const last = existing[existing.length - 1];
          if (last && last.timestamp === "streaming") {
            last.timestamp = formatTime();
            if (data.type === "error") last.content += "\n\n[error]";
            return { ...prev, [otherTid]: [...existing] };
          }
        }
        return prev;
      });
      return;
    }
    switch (data.type) {
      case "reasoning_delta":
        streamBufferRef.current.reasoning += data.text;
        scheduleFlush();
        setIsStreaming(true);
        break;
      case "text_delta":
        streamBufferRef.current.text += data.text;
        scheduleFlush();
        pendingResponseRef.current += data.text;
        setIsStreaming(true);
        break;
      case "done":
        // flush any remaining buffered tokens before finalizing
        flushStreamBuffer();
        if (streamingTimeoutRef.current) {
          clearTimeout(streamingTimeoutRef.current);
          streamingTimeoutRef.current = null;
        }
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.timestamp === "streaming") {
            updated[updated.length - 1] = { ...last, timestamp: formatTime() };
          }
          return updated;
        });
        setIsStreaming(false);
        if (soundEnabled) playTaskComplete();
        if (document.hidden) notify("Huginn", pendingResponseRef.current.slice(0, 120) || "Agent finished");
        break;
      case "error":
        if (streamingTimeoutRef.current) {
          clearTimeout(streamingTimeoutRef.current);
          streamingTimeoutRef.current = null;
        }
        if (soundEnabled) playErrorSound();
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `❌ ${data.error}`, timestamp: formatTime() },
        ]);
        setIsStreaming(false);
        if (document.hidden) notify("Huginn", `Error: ${data.error?.slice(0, 120) || "unknown"}`);
        break;
      case "tool_call":
        setMessages((prev) => [
          ...prev,
          {
            role: "tool",
            content: `Using tool **${data.name}**…`,
            timestamp: formatTime(),
            tool_call_id: data.id,
            tool_name: data.name,
            tool_args: data.args,
            tool_status: "running",
          },
        ]);
        setIsStreaming(true);
        break;
      case "tool_result":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = updated.findIndex(
            (m) => m.role === "tool" && m.tool_call_id === data.id && m.tool_status === "running"
          );
          if (idx !== -1) {
            updated[idx] = {
              ...updated[idx],
              content: `Tool **${updated[idx].tool_name}** finished`,
              tool_status: "done",
              tool_result: data.content,
            };
          }
          return updated;
        });
        break;
      case "auto_checkpoint":
        onAutoCheckpoint({ id: data.id, base: data.base, files: data.files });
        break;
      case "agent_status":
        setMessages((prev) => {
          const updated = [...prev];
          const key = `agent:${data.task_id}`;
          const idx = updated.findIndex((m) => m.role === "tool" && m.tool_call_id === key);
          const text = data.output
            ? `**${data.agent_id}** ${data.status}: ${data.output.slice(0, 200)}`
            : `**${data.agent_id}** ${data.status}…`;
          const entry: Message = {
            role: "tool",
            content: text,
            timestamp: formatTime(),
            tool_call_id: key,
            tool_name: data.agent_id,
            tool_status: data.status === "done" ? "done" : "running",
          };
          if (idx !== -1) {
            updated[idx] = entry;
          } else {
            updated.push(entry);
          }
          return updated;
        });
        break;
      case "exploration_result":
        if (data.data) {
          onExplorationResult(data.data);
          const best = data.data.best_branch;
          const summary = best
            ? `Exploration complete — best: **${best.name}** (${data.data.convergence_reason})`
            : `Exploration complete (${data.data.convergence_reason})`;
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: summary, timestamp: formatTime() },
          ]);
        }
        break;
      case "pong":
        break;
      case "context_compacted": {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant" as const,
            content: "",
            timestamp: new Date().toISOString(),
            isCompacted: true,
            compactBefore: data.before_pct,
            compactAfter: data.after_pct,
          },
        ]);
        break;
      }
      case "plan": {
        const planData = data.plan;
        const planId = data.plan_id;
        const steps = (planData?.steps || []).map((s: any, i: number) =>
          `${i + 1}. **${s.name || "Step"}**: ${s.description || ""}`
        ).join("\n");
        const criteria = (planData?.acceptance_criteria || []).map((c: any) =>
          `✅ ${c.criterion || c}`
        ).join("\n");
        const tools = (planData?.tools_needed || []).join(", ");
        const planContent = `📋 **Plan**: ${planData?.summary || ""}\n\n${steps}\n${criteria ? `\n${criteria}\n` : ""}${tools ? `\n🔧 Tools: ${tools}` : ""}`;
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: planContent,
            timestamp: formatTime(),
            isPlan: true,
            planId,
            planData,
          } as Message,
        ]);
        break;
      }
      case "plan_result": {
        const criteria = (data.criteria || []).map((c: any) =>
          `${c.passed ? "✅" : "❌"} ${c.criterion}`
        ).join("\n");
        const allPassed = data.all_passed;
        const resultContent = `${allPassed ? "🎉 All criteria passed!" : "⚠️ Some criteria failed"}\n${criteria}`;
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: resultContent, timestamp: formatTime() },
        ]);
        break;
      }
      case "clarification_request": {
        const questions = data.questions || [];
        const questionText = questions.map((q: any, i: number) => {
          const opts = q.options ? q.options.map((o: string) => `[${o}]`).join(" ") : "";
          return `${i + 1}. ${q.question || q}${opts ? " " + opts : ""}`;
        }).join("\n");
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `❓ Please clarify:\n${questionText}`,
            timestamp: formatTime(),
            isClarification: true,
            clarifications: questions,
          } as Message,
        ]);
        setPendingClarifications(questions.map((q: any) => ({
          question_id: q.question_id,
          question: q.question || q,
          options: q.options || [],
          thread_id: data.thread_id,
        })));
        break;
      }
      case "side_question_pending": {
        // Backend sends a single question, not an array
        const q = data.question;
        if (!q) break;
        setPendingClarifications((prev) => [
          ...prev,
          {
            question_id: data.question_id || "",
            question: q,
            options: [],
            thread_id: data.thread_id || activeThreadRef.current,
          },
        ]);
        break;
      }
      case "citations": {
        const sources = data.sources || [];
        const sourceText = sources.map((s: any) =>
          `[${s.ref}] ${s.filename}${s.distance ? ` (score: ${(1 - s.distance).toFixed(2)})` : ""}`
        ).join("\n");
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `📎 Sources:\n${sourceText}`,
            timestamp: formatTime(),
            isCitation: true,
            citationSources: sources,
          } as Message,
        ]);
        break;
      }
      case "task_progress": {
        const tp = data;

        // Pipeline progress (deli_research, computational_loop, etc.)
        if (tp.task_type === "pipeline") {
          const pipelineKey = tp.pipeline || "pipeline";
          const sIdx = tp.stage_index ?? 0;
          const sTotal = tp.total_stages ?? 1;
          setMessages((prev) => {
            for (let i = prev.length - 1; i >= Math.max(0, prev.length - 20); i--) {
              const m = prev[i];
              if (m.isTaskProgress && m.pipelineName === pipelineKey) {
                const stages = [...(m.pipelineStages || [])];
                while (stages.length < sTotal) {
                  stages.push({ name: "", label: "", status: "pending" });
                }
                stages[sIdx] = {
                  name: tp.message || "",
                  label: tp.stage_label || tp.message || "",
                  status: (tp.status as PipelineStage["status"]) || "pending",
                  detail: tp.detail,
                };
                for (let j = 0; j < sIdx; j++) {
                  if (stages[j].status !== "error" && stages[j].status !== "done") {
                    stages[j] = { ...stages[j], status: "done" };
                  }
                }
                for (let j = sIdx + 1; j < sTotal; j++) {
                  if (stages[j].status === "running") {
                    stages[j] = { ...stages[j], status: "pending" };
                  }
                }
                const updated = [...prev];
                updated[i] = {
                  ...m,
                  pipelineStages: stages,
                  pipelineProgressPct: tp.progress_pct,
                  pipelineTopic: tp.topic || m.pipelineTopic,
                  content: tp.detail || m.content,
                  timestamp: formatTime(),
                };
                return updated;
              }
            }
            // first message for this pipeline — create the card
            const stages: PipelineStage[] = Array.from(
              { length: sTotal },
              (_, idx) => ({
                name: idx === sIdx ? (tp.message || "") : "",
                label: idx === sIdx ? (tp.stage_label || tp.message || "") : "",
                status: idx === sIdx ? ((tp.status as PipelineStage["status"]) || "pending") : "pending",
                detail: idx === sIdx ? tp.detail : undefined,
              })
            );
            return [...prev, {
              role: "assistant",
              content: tp.detail || "",
              timestamp: formatTime(),
              isTaskProgress: true,
              taskType: "pipeline",
              pipelineName: pipelineKey,
              pipelineTopic: tp.topic,
              pipelineStages: stages,
              pipelineProgressPct: tp.progress_pct,
            } as Message];
          });
          break;
        }

        // HPC job / sweep / generic progress
        const taskKey = tp.task_type + (tp.job_id ? `_${tp.job_id}` : "");
        let progressText = "";
        if (tp.task_type === "hpc_job") {
          const icon = tp.status === "completed" ? "✅" : tp.status === "failed" ? "❌" : tp.status === "running" ? "🔄" : "⏳";
          progressText = `${icon} HPC Job ${tp.job_id}: ${tp.status}`;
        } else if (tp.task_type === "sweep") {
          progressText = `📊 Sweep: ${tp.completed}/${tp.total} (${tp.progress_pct}%)`;
        } else {
          progressText = `📈 Progress: ${tp.progress_pct}%`;
        }
        setMessages((prev) => {
          for (let i = prev.length - 1; i >= Math.max(0, prev.length - 10); i--) {
            const m = prev[i];
            if (m.isTaskProgress && (m.taskType + (m.jobId ? `_${m.jobId}` : "")) === taskKey) {
              const updated = [...prev];
              updated[i] = { ...m, content: progressText, timestamp: formatTime() };
              return updated;
            }
          }
          return [...prev, {
            role: "assistant",
            content: progressText,
            timestamp: formatTime(),
            isTaskProgress: true,
            taskType: tp.task_type,
            jobId: tp.job_id,
          } as Message];
        });
        break;
      }
      case "sediment": {
        if (data.stored) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `💾 Result saved to knowledge base`, timestamp: formatTime() },
          ]);
        }
        break;
      }
      case "approval_request": {
        if (data.auto_approved) {
          // Already approved by backend — just log it, no dialog needed
          const icon = data.dangerous ? "🔴" : "⚠️";
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: `${icon} Auto-approved: **${data.tool_name}** — ${data.reason}`,
              timestamp: formatTime(),
            },
          ]);
        } else {
          // Need explicit user approval
          setPendingApproval({
            request_id: data.request_id,
            tool_name: data.tool_name,
            reason: data.reason,
            dangerous: data.dangerous,
          });
        }
        break;
      }
      case "auto_approve_set":
        setAutoApprove(data.enabled ?? true);
        break;
      case "hook_warning": {
        const warningText = data.warnings
          .map((w) => `⚠️ **${data.tool_name}**: ${w.message}`)
          .join("\n");
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: warningText, timestamp: formatTime() },
        ]);
        break;
      }
      case "ping":
        if (wsClientRef.current) {
          wsClientRef.current.send(JSON.stringify({ type: "pong" }));
        }
        break;
      case "pet_update":
        setPetState({
          mood: data.mood,
          xp: data.xp,
          level: data.level,
          hunger: data.hunger,
          happiness: data.happiness,
        });
        break;
    }
  };

  // ── WebSocket initialization ─────────────────────────────────
  // Re-connect when wsUrlVersion changes (dynamic port assignment).
  useEffect(() => {
    const wsUrl = WS_URL || `${API_BASE.replace("http", "ws")}/ws/agent`;

    const ws = new ReconnectingWebSocket({
      url: wsUrl,
      authToken: () => getAuthToken(),
      pingInterval: 30_000,
      maxDelay: 30_000,
      onStatus: (wsStatus) => {
        setIsConnected(wsStatus === "connected");
        if (wsStatus === "reconnecting") {
          setIsConnected(false);
          setWsReconnecting(true);
          setWsFailed(false);
        } else if (wsStatus === "connected") {
          setWsReconnecting(false);
          setWsFailed(false);
        } else if (wsStatus === "failed") {
          setWsReconnecting(false);
          setWsFailed(true);
          // Release isStreaming — backend can't send 'done' if WS is dead
          setIsStreaming(false);
          setMessages((prev) =>
            prev.map((m) =>
              m.timestamp === "streaming"
                ? { ...m, timestamp: formatTime(), content: m.content + "\n\n[连接断开]" }
                : m
            )
          );
        }
      },
      onMessage: (data) => {
        if (typeof data === "string") return;
        if (isWSMessage(data)) handleWsMessage(data);
      },
      onConnected: () => {
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((m) =>
            m.timestamp === "streaming"
              ? { ...m, timestamp: formatTime(), content: m.content + "\n\n[连接中断，回复未完成]" }
              : m
          )
        );
        // Restore chat history from backend on (re)connect
        loadThreads();
        const tid = activeThreadRef.current;
        api.get<{ messages?: any[] }>(`/threads/${tid}/messages`)
          .then((data) => {
            if (data.messages && data.messages.length > 0) {
              const restored: Message[] = data.messages.map((m: any) => ({
                role: m.role as Message["role"],
                content: m.content || "",
                timestamp: m.timestamp || formatTime(),
                ...(m.tool_name ? { tool_name: m.tool_name } : {}),
                ...(m.tool_call_id ? { tool_call_id: m.tool_call_id } : {}),
              }));
              setMessages(restored);
              setMessagesByThread((prev) => ({ ...prev, [tid]: restored }));
            }
          })
          .catch(() => { /* backend offline — keep welcome */ });
      },
    });

    ws.connect();
    wsClientRef.current = ws;

    return () => {
      ws.close();
      wsClientRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsUrlVersion]);

  // ── Dynamic tab title (unread count when tab hidden) ──────────
  const lastSeenMsgCountRef = useRef(messages.length);
  useEffect(() => {
    const onVis = () => {
      if (!document.hidden) {
        lastSeenMsgCountRef.current = messages.length;
        document.title = "Huginn";
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [messages.length]);

  useEffect(() => {
    if (document.hidden && messages.length > lastSeenMsgCountRef.current) {
      const unread = messages.length - lastSeenMsgCountRef.current;
      document.title = `(${unread}) Huginn`;
    }
  }, [messages.length]);

  // ── Load tools and skills on tab switch ──────────────────────
  useEffect(() => {
    if (activeTab === "tools" && toolsLength === 0) {
      api.get<any[]>('/tools')
        .then(setTools)
        .catch((e) => console.error("Failed to load tools:", e));
    }
    if (activeTab === "skills" && skillsLength === 0) {
      api.get<any[]>('/skills')
        .then(setSkills)
        .catch((e) => console.error("Failed to load skills:", e));
    }
  }, [activeTab, toolsLength, skillsLength]);

  // ── Dynamic persona loading ──────────────────────────────────
  useEffect(() => {
    api.get<{ personas?: PersonaSeed[] } | PersonaSeed[]>('/personas')
      .then((data) => {
        const list = Array.isArray(data) ? data : (data.personas || []);
        const personas = list.map((p) => ({
          id: p.name || p.id || '',
          label: p.name || p.id || '',
          description: p.description || '',
          avatar: p.avatar || '',
        }));
        if (personas.length > 0) {
          setPersonaList(personas);
        }
      })
      .catch(() => { /* keep fallback list */ });
  }, []);

  // ── Load persona emotion when persona changes ────────────────
  useEffect(() => {
    if (!config.persona || config.persona === 'default') {
      setPersonaEmotion(null);
      return;
    }
    api.get<PersonaEmotionResponse>(`/personas/${config.persona}/emotion`)
      .then((d) => {
        setPersonaEmotion({
          mood: d.context_prompt?.slice(0, 80) || '',
          valence: d.state?.valence ?? 0,
          arousal: d.state?.arousal ?? 0,
          trust: d.state?.trust ?? 0.5,
        });
      })
      .catch(() => { /* emotion not available */ });
  }, [config.persona]);

  // ── Send message ─────────────────────────────────────────────
  const sendMessage = async () => {
    if (!input.trim()) return;

    let content = input.trim();

    // Research mode: auto-prefix /research so the backend triggers autoloop
    if (researchMode && !content.startsWith("/research ")) {
      content = "/research " + content;
    }

    // Plan mode: prefix /plan so the WS handler routes it as a plan request
    if (mode === "plan" && !content.startsWith("/plan ")) {
      content = "/plan " + content;
    }

    // Queue the message if the agent is still streaming (Kimi-style)
    if (isStreaming) {
      pendingMessagesRef.current.push(content);
      setPendingMessages([...pendingMessagesRef.current]);
      setInput("");
      return;
    }

    if (!wsClientRef.current) return;

    // AstrBot pattern: optimistic update — immediately add user msg + empty bot placeholder
    const userMsg: Message = { role: "user", content, timestamp: formatTime() };
    const botPlaceholder: Message = { role: "assistant", content: "", timestamp: "streaming", reasoning: "" };
    setMessages((prev) => [...prev, userMsg, botPlaceholder]);
    setInput("");
    pendingResponseRef.current = "";
    streamBufferRef.current = { text: "", reasoning: "" };

    // Undo window: 5 seconds to cancel the send (before streaming response arrives)
    setUndoWindow(true);
    if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
    undoTimerRef.current = setTimeout(() => setUndoWindow(false), 5000);

    const payload = JSON.stringify({ type: "user_input", content: userMsg.content, thread_id: activeThread, thinking: thinkingIntensity });
    try {
      wsClientRef.current!.send(payload);
    } catch {
      // WS dropped mid-send — roll back the optimistic user msg + bot placeholder
      setMessages((prev) => prev.slice(0, -2));
      setMessages((prev) => [...prev, { role: "assistant", content: "⚠️ Failed to send — WebSocket disconnected. Please wait for reconnection.", timestamp: formatTime() }]);
      return;
    }

    // watchdog: if no tokens arrive in 30s the WS likely dropped silently
    if (streamingTimeoutRef.current) clearTimeout(streamingTimeoutRef.current);
    streamingTimeoutRef.current = setTimeout(() => {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.timestamp === "streaming" && !last.content) {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...last,
            content: "Connection lost. Please try again.",
            timestamp: formatTime(),
          };
          return updated;
        }
        return prev;
      });
      setIsStreaming(false);
      streamingTimeoutRef.current = null;
    }, 30_000);
  };

  // Undo last send — removes the last user message + bot placeholder, restores input
  const undoSend = useCallback(() => {
    if (!undoWindow) return;
    setMessages((prev) => {
      if (prev.length < 2) return prev;
      const lastUser = [...prev].reverse().find(m => m.role === "user");
      if (lastUser) setInput(lastUser.content);
      return prev.slice(0, -2); // remove user msg + bot placeholder
    });
    setUndoWindow(false);
    if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
    if (streamingTimeoutRef.current) clearTimeout(streamingTimeoutRef.current);
    setIsStreaming(false);
  }, [undoWindow]);

  // ── Drain queued messages when streaming finishes ─────────────
  // Fires on every isStreaming transition; only acts when streaming
  // just stopped and there are pending messages to flush.
  useEffect(() => {
    if (!isStreaming && pendingMessagesRef.current.length > 0 && wsClientRef.current) {
      const nextContent = pendingMessagesRef.current.shift()!;
      setPendingMessages([...pendingMessagesRef.current]);

      let content = nextContent;
      if (researchMode && !content.startsWith("/research ")) {
        content = "/research " + content;
      }

      const userMsg: Message = { role: "user", content, timestamp: formatTime() };
      setMessages((prev) => [...prev, userMsg]);
      pendingResponseRef.current = "";

      const payload = JSON.stringify({ type: "user_input", content, thread_id: activeThread, thinking: thinkingIntensity });
      try {
        wsClientRef.current.send(payload);
      } catch {
        setMessages((prev) => prev.slice(0, -1));
        setMessages((prev) => [...prev, { role: "assistant", content: "⚠️ Failed to send queued message — WebSocket disconnected. Please wait for reconnection.", timestamp: formatTime() }]);
        return;
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming]);

  // ── Stop generation ──────────────────────────────────────────
  const stopGeneration = useCallback(async () => {
    try {
      await api.post('/agents/default/interrupt', {
        type: 'cancel',
        thread_id: activeThreadRef.current,
      });
    } catch {
      // Fallback: close WS to force-stop
      wsClientRef.current?.close();
    }
    setIsStreaming(false);
    setPendingMessages([]);
    setMessages((prev) => {
      if (prev.length > 0 && prev[prev.length - 1].role === 'assistant') {
        const last = prev[prev.length - 1];
        if (last.content === '' || last.content === '…') {
          return prev.slice(0, -1);
        }
        return [...prev.slice(0, -1), { ...last, content: last.content + '\n\n*[stopped]*' }];
      }
      return prev;
    });
  }, []);

  // ── Answer clarification ─────────────────────────────────────
  const answerClarification = (questionId: string | undefined, answer: string) => {
    const payload = JSON.stringify({
      type: "clarification_response",
      question_id: questionId,
      answer,
      thread_id: activeThread,
    });
    if (wsClientRef.current) {
      wsClientRef.current.send(payload);
    }
    setMessages((prev) => [
      ...prev,
      { role: "user", content: answer, timestamp: formatTime() },
    ]);
    setPendingClarifications([]);
  };

  // ── Approval response ───────────────────────────────────────
  const respondToApproval = (requestId: string, approved: boolean) => {
    if (wsClientRef.current) {
      wsClientRef.current.send(JSON.stringify({
        type: "approval_response",
        request_id: requestId,
        approved,
      }));
    }
    setPendingApproval(null);
  };

  const toggleAutoApprove = (enabled: boolean) => {
    setAutoApprove(enabled);
    if (wsClientRef.current) {
      wsClientRef.current.send(JSON.stringify({
        type: "set_auto_approve",
        enabled,
      }));
    }
  };

  // ── Autoloop SSE subscription ────────────────────────────────
  useEffect(() => {
    const es = new EventSource(`${API_BASE}/tasks/stream`);
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "task" && data.task_type === "autoloop") {
          setAutoloopPhase(data.status || "");
          setAutoloopProgress(data.progress_pct || 0);
        }
      } catch {
        // ignore malformed frames
      }
    };
    return () => es.close();
  }, []);

  return {
    // Chat state
    messages, input, mode, pendingPlan,
    chatSearchOpen, chatSearchQuery,
    isStreaming,
    messagesEndRef,
    // Connection
    isConnected, status, wsReconnecting, wsFailed,
    undoWindow, undoSend,
    wsClientRef,
    // Persona
    personaList, personaEmotion, pendingClarifications,
    // Threads
    threads, activeThread,
    // Guide
    showGuide, closeGuide,
    // Setters
    setInput, setMode, setMessages, setPendingPlan, setChatSearchOpen, setChatSearchQuery,
    setActiveThread, setThreads, setShowGuide, switchThread,
    // Functions
    sendMessage, answerClarification,
    loadThreads, createThread, renameThread, deleteThread,
    startBackend, notify,
    // Approval
    pendingApproval, autoApprove, respondToApproval, toggleAutoApprove,
    // Autoloop
    autoloopPhase, autoloopProgress,
    // Thinking intensity
    thinkingIntensity, setThinkingIntensity,
    // Message queue
    pendingMessages,
    // Stop generation
    stopGeneration,
    // Research mode
    researchMode, setResearchMode,
    // Sound toggle
    soundEnabled, setSoundEnabled,
    // Pet state
    petState,
  };
}
