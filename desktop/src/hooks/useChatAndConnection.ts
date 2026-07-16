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
import { toast } from "../components/Toast";

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
  // multi-agent persona
  persona?: string;
}

export interface Thread {
  id: string;
  label: string;
  created_at: string;
  last_active: string;
  archived?: boolean;
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
  const [status, setStatus] = useState<string>("connecting…");
  const [isConnected, setIsConnected] = useState(false);
  const [wsReconnecting, setWsReconnecting] = useState(false);
  const [wsFailed, setWsFailed] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [undoWindow, setUndoWindow] = useState(false);
  const undoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pendingResponseRef = useRef<string>("");
  const streamingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Always-current snapshot of messages for the active thread — avoids
  // stale closure captures in switchThread/createThread.
  const messagesRef = useRef<Message[]>([WELCOME_MSG()]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  // Tracks which thread we last sent a user_input to, so WS messages
  // without an explicit thread_id can still be routed correctly.
  const pendingThreadIdRef = useRef<string>("desktop");
  const [chatSearchOpen, setChatSearchOpen] = useState(false);
  const [chatSearchQuery, setChatSearchQuery] = useState("");

  // ── Guide state ──────────────────────────────────────────────
  const GUIDE_KEY = "muninn:guide:v2";
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

  // ── Agent mode banner (HRI: situation awareness) ────────────
  // 后端每次 chat() 入口 emit mode_banner, 前端展示当前 agent 工作模式
  const [agentMode, setAgentMode] = useState<{
    exec_mode: string; user_mode: string; flags: string[]; trace_id?: string;
  }>({ exec_mode: "tool_call", user_mode: "chat", flags: [] });

  // OAK 启发: trace_id 贯穿 — 当前活跃 trace, 前端按 trace 聚合事件
  const [activeTraceId, setActiveTraceId] = useState<string>("");

  // ── Trust score (HRI: trust calibration) ────────────────────
  const [trustScore, setTrustScore] = useState<number>(0.5);

  // ── Approval budget (HRI: alert fatigue avoidance) ──────────
  const [approvalBudget, setApprovalBudget] = useState<number>(10);

  // ── SUGGEST mode (HRI: LoA Level 4-6, 代码先展示给用户编辑) ──
  const [suggestMode, setSuggestMode] = useState<boolean>(false);
  const [pendingSuggestCode, setPendingSuggestCode] = useState<{
    code: string; risk: string; reason: string; turn: number;
  } | null>(null);

  // ── Dynamic risk threshold (HRI: trust-adaptive risk classification) ──
  const [riskThreshold, setRiskThreshold] = useState<number>(0.5);

  // ── Autoloop progress (SSE) ──────────────────────────────────
  const [autoloopPhase, setAutoloopPhase] = useState<string>("");
  const [autoloopProgress, setAutoloopProgress] = useState<number>(0);
  // campaign 事件: hypothesis / retry / suspect / refine, 从 SSE /tasks/stream 收
  // 之前前端只能正则刮消息文本, retry/suspect/refine 根本到不了. 现在走结构化 SSE.
  const [campaignEvents, setCampaignEvents] = useState<
    Array<{ event: string; data: Record<string, unknown>; ts: number; task_id: string }>
  >([]);
  // 当前 thread 的任务状态: goal / mode / iteration / 进度
  // switchThread 时从 /threads/{id}/state 拉, 恢复 research mode + 显示研究进度
  const [threadTaskState, setThreadTaskState] = useState<{
    goal: string;
    mode: string;
    iteration: number;
    steps_done: number;
    steps_total: number;
    key_findings: string[];
  }>({ goal: "", mode: "chat", iteration: 0, steps_done: 0, steps_total: 0, key_findings: [] });
  // plan 执行状态: plan_id -> "executing" | "done". 跟 plan 卡片绑定显示状态徽标.
  // 之前用户点 Confirm 后整个流式期间没任何指示, 只能盯着光标.
  const [planExecState, setPlanExecState] = useState<Record<string, "executing" | "done">>({});

  // ── Decision trace: governance events + state transitions ──
  const [governanceEvents, setGovernanceEvents] = useState<any[]>([]);
  const [stateTransitions, setStateTransitions] = useState<any[]>([]);

  // ── Context window usage (from context_compacted WS) ─────────
  const [contextPct, setContextPct] = useState<number>(0);

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

  // ── Forest result (随机森林多 engine 并行探索的 DS 合成结果) ──
  const [forestResult, setForestResult] = useState<any>(null);

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
    // save current thread's messages to reactive store — use ref to avoid stale closure
    const currentMsgs = messagesRef.current;
    setMessagesByThread((prev) => ({ ...prev, [activeThread]: currentMsgs }));
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
    // 拉 task_state 恢复 mode/goal/iteration 显示 (之前切回 research thread 后 mode 全丢)
    api.get<any>(`/threads/${threadId}/state`)
      .then((s) => {
        if (s && s.mode) {
          setThreadTaskState({
            goal: s.goal || "",
            mode: s.mode,
            iteration: s.iteration || 0,
            steps_done: s.steps_done || 0,
            steps_total: s.steps_total || 0,
            key_findings: s.key_findings || [],
          });
          setResearchMode(s.mode === "research");
        }
      })
      .catch(() => { /* no task state yet */ });
    setActiveThread(threadId);
  };

  const loadThreads = async (includeArchived = false) => {
    try {
      const data = await api.get<{ threads?: Thread[] }>(
        `/threads?include_archived=${includeArchived ? "true" : "false"}`
      );
      setThreads(data.threads || []);
    } catch (e: any) {
      console.error("[threads] load failed:", e);
    }
  };

  const createThread = async () => {
    try {
      const data = await api.post<{ id: string; label: string }>("/threads", { title: "New thread" });
      // cache current thread before switching — use ref to avoid stale closure
      setMessagesByThread((prev) => ({ ...prev, [activeThread]: messagesRef.current }));
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

  // fork: 在当前线程某节点分叉出新线程, 复制历史消息. 后端 POST /threads/{id}/fork.
  // 之前函数定义了但没有任何 UI 调 — 现在接到 ThreadsPanel 菜单.
  const forkThread = async (id: string) => {
    try {
      const data = await api.post<{ thread_id: string; label: string }>(`/threads/${id}/fork`);
      setMessagesByThread((prev) => ({ ...prev, [activeThread]: messagesRef.current }));
      setActiveThread(data.thread_id);
      setMessages([
        {
          role: "assistant",
          content: `Forked from **${id}** — new thread **${data.label}**.`,
          timestamp: formatTime(),
        },
      ]);
      loadThreads();
    } catch (e: any) {
      console.error("[threads] fork failed:", e);
    }
  };

  // archive: 后端 POST /threads/{id}/archive 标记 archived=true, 本地从列表移除.
  const archiveThread = async (id: string) => {
    try {
      await api.post(`/threads/${id}/archive`);
      setThreads((prev) => prev.filter((t) => t.id !== id));
      if (activeThread === id) {
        setActiveThread("desktop");
      }
    } catch (e: any) {
      console.error("[threads] archive failed:", e);
    }
  };

  // unarchive: 后端 POST /threads/{id}/unarchive 恢复, 重载列表.
  const unarchiveThread = async (id: string) => {
    try {
      await api.post(`/threads/${id}/unarchive`);
      loadThreads();
    } catch (e: any) {
      console.error("[threads] unarchive failed:", e);
    }
  };

  // ── WebSocket ref ────────────────────────────────────────────
  const wsClientRef = useRef<ReconnectingWebSocket | null>(null);

  // ── Notification ─────────────────────────────────────────────
  // alwaysShow=true 时即使窗口在前台也弹通知，用于权限/澄清等需要用户立刻注意的事件
  const notify = useCallback((title: string, body: string, alwaysShow: boolean = false) => {
    try {
      if (alwaysShow || document.hidden) {
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
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let backoff = 1000; // start at 1s, double on each failure

    const check = async () => {
      // Skip Tauri invoke in browser/dev — go straight to HTTP fetch
      const inTauri = "__TAURI_INTERNALS__" in window;
      if (inTauri) {
        try {
          const s: any = await invoke("get_agent_status");
          if (alive) {
            await syncBackendUrl();
            setStatus(`${s.status} • v${s.version || "0.1.0"}`);
            if (s.status === "ok") return true;
          }
        } catch {
          // Tauri invoke failed — fall through to HTTP check
        }
      }
      try {
        const resp = await fetch(`${getApiBase()}/health`, { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
          const s = await resp.json();
          if (alive) {
            await syncBackendUrl();
            setStatus(`${s.status} • v${s.version || "0.1.0"}`);
            if (s.status === "ok") return true;
          }
        }
      } catch { /* still down */ }
      return false;
    };

    const run = async () => {
      const online = await check();
      if (online) {
        // Healthy — poll every 30s to catch disconnects
        pollTimer = setTimeout(run, 30000);
        return;
      }
      // Not online yet — try startBackend, then poll with backoff
      await startBackend();
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, backoff));
        if (!alive) return;
        if (await check()) {
          backoff = 1000; // reset on success
          pollTimer = setTimeout(run, 30000);
          return;
        }
        backoff = Math.min(backoff * 2, 16000); // cap at 16s
      }
      if (alive) setStatus("backend did not come online");
    };

    run();
    return () => { alive = false; if (pollTimer) clearTimeout(pollTimer); };
  }, [startBackend]);

  // ── Stream batching (assistant-UI pattern: batch tokens to reduce renders) ─
  const streamBufferRef = useRef<{ text: string; reasoning: string }>({ text: "", reasoning: "" });
  const rafScheduledRef = useRef(false);

  // watchdog: 流式过程中如果 token 间隔超时, 判定 WS 静默断开.
  // 首次 arm 用 180s (DeepSeek Reasoner 首 token 60s+ + 弱网容忍),
  // 后续 token 间用 60s (后续 token 应秒级到达).
  // 触发时: 无 content 显示 "Connection lost", 有 content 保留部分结果标记完成.
  // 必须在 flushStreamBuffer 之前定义 — const 不 hoist, 闭包要拿到引用.
  const armWatchdog = useCallback((timeoutMs: number) => {
    if (streamingTimeoutRef.current) clearTimeout(streamingTimeoutRef.current);
    streamingTimeoutRef.current = setTimeout(() => {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.timestamp === "streaming") {
          const updated = [...prev];
          if (!last.content) {
            updated[updated.length - 1] = {
              ...last,
              content: "Connection lost. Please try again.",
              timestamp: formatTime(),
            };
          } else {
            // 部分内容: 保留, 标记完成 (不丢弃已收到的 token)
            updated[updated.length - 1] = { ...last, timestamp: formatTime() };
          }
          return updated;
        }
        return prev;
      });
      setIsStreaming(false);
      streamingTimeoutRef.current = null;
    }, timeoutMs);
  }, []);

  const flushStreamBuffer = useCallback(() => {
    rafScheduledRef.current = false;
    const buf = streamBufferRef.current;
    if (!buf.text && !buf.reasoning) return;
    // token 到了 — 清掉首 token 期 watchdog, 重设为后续 token 间隔 (60s).
    // 之前是直接 clear 不重设, 后续 token 中断时 watchdog 已失效, 静默 hang.
    armWatchdog(60_000);
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
  }, [armWatchdog]);

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
    // Backend injects thread_id on most messages, but some (tool_call,
    // task_progress, etc.) may omit it. Fall back to the thread we last
    // sent a user_input to so messages never cross into the wrong thread.
    const _tid = (data as any).thread_id as string | undefined;
    const _effectiveTid = _tid || pendingThreadIdRef.current;
    if (_effectiveTid !== activeThreadRef.current
        && !_BROADCAST_TYPES.has(data.type)) {
      // Buffer for the other thread — handle all message types, not just text
      const otherTid = _effectiveTid;
      setMessagesByThread((prev) => {
        const existing = prev[otherTid] ? [...prev[otherTid]] : [];
        if (data.type === "text_delta") {
          const last = existing[existing.length - 1];
          if (last && last.role === "assistant" && last.timestamp === "streaming") {
            // immutable update — copy the object, don't mutate
            existing[existing.length - 1] = { ...last, content: last.content + ((data as any).text || "") };
          } else {
            existing.push({ role: "assistant", content: (data as any).text || "", timestamp: "streaming" });
          }
          return { ...prev, [otherTid]: existing };
        }
        if (data.type === "reasoning_delta") {
          // buffer reasoning silently — it'll be re-fetched on switch if needed
          return prev;
        }
        if (data.type === "done" || data.type === "error") {
          const last = existing[existing.length - 1];
          if (last && last.timestamp === "streaming") {
            existing[existing.length - 1] = {
              ...last,
              timestamp: formatTime(),
              ...(data.type === "error" ? { content: last.content + "\n\n[error]" } : {}),
            };
          }
          return { ...prev, [otherTid]: existing };
        }
        if (data.type === "tool_call") {
          existing.push({
            role: "tool",
            content: `Using tool **${(data as any).name}**…`,
            timestamp: formatTime(),
            tool_call_id: (data as any).id,
            tool_name: (data as any).name,
            tool_args: (data as any).args,
            tool_status: "running",
          });
          return { ...prev, [otherTid]: existing };
        }
        if (data.type === "tool_result") {
          const idx = existing.findIndex(
            (m) => m.role === "tool" && m.tool_call_id === (data as any).id && m.tool_status === "running"
          );
          if (idx !== -1) {
            existing[idx] = {
              ...existing[idx],
              content: `Tool **${existing[idx].tool_name}** finished`,
              tool_status: "done",
              tool_result: (data as any).content,
            };
          }
          return { ...prev, [otherTid]: existing };
        }
        if (data.type === "task_progress") {
          existing.push({
            role: "assistant",
            content: `📊 ${(data as any).stage || "pipeline"}: ${(data as any).status || ""}`,
            timestamp: formatTime(),
          });
          return { ...prev, [otherTid]: existing };
        }
        // other types (plan, citations, etc.) — store as assistant text
        if (data.type === "plan" || data.type === "citations") {
          existing.push({
            role: "assistant",
            content: JSON.stringify(data).slice(0, 500),
            timestamp: formatTime(),
          });
          return { ...prev, [otherTid]: existing };
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
        else toast.success("Agent 已完成");
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
        setContextPct(data.after_pct || 0);
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
        // 澄清请求需要用户立刻注意，即使窗口在前台也弹通知
        notify("Huginn 需要澄清", "Agent 有问题需要你回答", true);
        break;
      }
      case "mode_banner": {
        const tid = data.trace_id || "";
        setAgentMode({
          exec_mode: data.exec_mode || "tool_call",
          user_mode: data.user_mode || "chat",
          flags: data.flags || [],
          trace_id: tid,
        });
        if (tid) setActiveTraceId(tid);
        break;
      }
      case "trust_update": {
        setTrustScore(data.trust);
        break;
      }
      case "budget_update": {
        setApprovalBudget(data.remaining);
        break;
      }
      case "budget_escalation": {
        setApprovalBudget(data.remaining);
        break;
      }
      case "suggest_code": {
        setPendingSuggestCode({
          code: data.code || "",
          risk: data.risk || "medium",
          reason: data.reason || "",
          turn: data.turn ?? 0,
        });
        break;
      }
      case "suggest_mode_set": {
        setSuggestMode(data.enabled);
        break;
      }
      case "risk_threshold": {
        setRiskThreshold(data.threshold);
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
          // 权限请求需要用户立刻注意，即使窗口在前台也弹通知
          notify("Huginn 需要权限", `工具 ${data.tool_name} 请求审批`, true);
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
      case "governance":
        setGovernanceEvents((prev) => [...prev, data].slice(-50));
        break;
      case "state_transition":
        setStateTransitions((prev) => [...prev, data].slice(-30));
        if (data.to_phase) setAutoloopPhase(data.to_phase);
        break;
      case "forest_result":
        setForestResult(data);
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
    // Track which thread this request belongs to, so WS messages
    // without an explicit thread_id can still be routed correctly.
    pendingThreadIdRef.current = activeThread;
    try {
      wsClientRef.current!.send(payload);
    } catch {
      // WS dropped mid-send — roll back the optimistic user msg + bot placeholder
      setMessages((prev) => prev.slice(0, -2));
      setMessages((prev) => [...prev, { role: "assistant", content: "⚠️ Failed to send — WebSocket disconnected. Please wait for reconnection.", timestamp: formatTime() }]);
      return;
    }

    // First message in a new thread → auto-generate a title from the user's input
    if (messagesRef.current.length <= 1 && activeThread !== "desktop") {
      const raw = input.trim();
      renameThread(activeThread, raw.length > 40 ? raw.slice(0, 40) + "..." : raw);
    }

    // watchdog: 首 token 期 180s 宽限 (DeepSeek Reasoner 首 token 60s+ + 弱网).
    // 后续 token 期 60s (在 flushStreamBuffer 里重设).
    // 之前固定 120s 在弱网下误杀, 60s 首 token 又太短.
    armWatchdog(180_000);
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
      pendingThreadIdRef.current = activeThread;
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

  // ── Pause / resume generation ───────────────────────────────
  const pauseGeneration = useCallback(async () => {
    try {
      await api.post('/agents/default/interrupt', {
        type: 'pause',
        thread_id: activeThreadRef.current,
      });
      setIsPaused(true);
    } catch {
      // backend might not support pause yet — ignore silently
    }
  }, []);

  const resumeGeneration = useCallback(async () => {
    try {
      await api.post('/agents/default/interrupt', {
        type: 'resume',
        thread_id: activeThreadRef.current,
      });
      setIsPaused(false);
    } catch {
      // backend might not support resume yet — ignore silently
    }
  }, []);

  // ── Answer clarification ─────────────────────────────────────
  const answerClarification = (questionId: string | undefined, answer: string) => {
    const payload = JSON.stringify({
      type: "clarification_response",
      question_id: questionId,
      answer,
      thread_id: activeThread,
    });
    pendingThreadIdRef.current = activeThread;
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

  // HRI #4: SUGGEST mode toggle — 所有 code_act 代码先展示给用户编辑
  const toggleSuggestMode = (enabled: boolean) => {
    setSuggestMode(enabled);
    if (wsClientRef.current) {
      wsClientRef.current.send(JSON.stringify({
        type: "set_suggest_mode",
        enabled,
        thread_id: activeThreadRef.current,
      }));
    }
  };

  // HRI #4: 用户对 suggest_code 的响应 (approve / edit / deny)
  const respondToSuggestCode = (action: "approve" | "edit" | "deny", editedCode?: string) => {
    if (wsClientRef.current) {
      wsClientRef.current.send(JSON.stringify({
        type: "suggest_response",
        thread_id: activeThreadRef.current,
        action,
        edited_code: editedCode || "",
      }));
    }
    setPendingSuggestCode(null);
  };

  // ── Autoloop SSE subscription ────────────────────────────────
  // Backend emits named events (snapshot/update/campaign), not unnamed messages.
  // The old es.onmessage handler never fired — autoloop progress was dead.
  useEffect(() => {
    const es = new EventSource(`${API_BASE}/tasks/stream`);
    const handleProgress = (e: MessageEvent) => {
      try {
        const t = JSON.parse(e.data);
        if (t.engine_kind === "autoloop") {
          setAutoloopPhase(t.current_label || t.status || "");
          setAutoloopProgress(t.percentage || 0);
        }
      } catch {
        // ignore malformed frames
      }
    };
    // campaign 事件: hypothesis / retry / suspect / refine, 结构化数据
    const handleCampaign = (e: MessageEvent) => {
      try {
        const t = JSON.parse(e.data);
        if (t && t._kind === "campaign") {
          setCampaignEvents((prev) =>
            [...prev, {
              event: t.event,
              data: t.data || {},
              ts: t.ts,
              task_id: t.task_id || "",
            }].slice(-200)
          );
          // plan 执行状态: 把 plan_id 关联到卡片
          const pid = t.data?.plan_id;
          if (typeof pid === "string") {
            if (t.event === "plan.exec_start") {
              setPlanExecState((prev) => ({ ...prev, [pid]: "executing" }));
            } else if (t.event === "plan.exec_complete") {
              setPlanExecState((prev) => ({ ...prev, [pid]: "done" }));
            }
          }
        }
      } catch {
        // ignore malformed frames
      }
    };
    es.addEventListener("snapshot", handleProgress);
    es.addEventListener("update", handleProgress);
    es.addEventListener("campaign", handleCampaign);
    return () => es.close();
  }, []);

  return {
    // Chat state
    messages, input, mode,
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
    setInput, setMode, setMessages, setChatSearchOpen, setChatSearchQuery,
    setActiveThread, setThreads, setShowGuide, switchThread,
    // Functions
    sendMessage, answerClarification,
    loadThreads, createThread, renameThread, deleteThread,
    forkThread, archiveThread, unarchiveThread,
    startBackend, notify,
    // Approval
    pendingApproval, autoApprove, respondToApproval, toggleAutoApprove,
    // Autoloop
    autoloopPhase, autoloopProgress,
    campaignEvents,
    threadTaskState,
    planExecState,
    // Context window
    contextPct,
    // Thinking intensity
    thinkingIntensity, setThinkingIntensity,
    // Message queue
    pendingMessages,
    // Stop generation
    stopGeneration,
    // Pause / resume generation
    pauseGeneration, resumeGeneration, isPaused,
    // Research mode
    researchMode, setResearchMode,
    // Sound toggle
    soundEnabled, setSoundEnabled,
    // Pet state
    petState,
    // Forest result (随机森林 DS 合成)
    forestResult,
    // Decision trace
    governanceEvents, stateTransitions,
    // Agent mode banner
    agentMode,
    // OAK: trace_id 贯穿
    activeTraceId,
    // Trust score
    trustScore,
    // Approval budget
    approvalBudget,
    // SUGGEST mode + pending suggest code
    suggestMode,
    pendingSuggestCode,
    toggleSuggestMode,
    respondToSuggestCode,
    // Dynamic risk threshold
    riskThreshold,
  };
}
