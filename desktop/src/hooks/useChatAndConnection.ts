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
import { getAuthToken } from "../lib/api-client";
import { api } from "../lib/api";
import {
  API_BASE, WS_URL, syncBackendUrl, PERSONAS_FALLBACK, loadStoredConfig,
} from "../lib/config-store";
import { isWSMessage, type WSMessage } from "../types/ws";
import type { AppConfig, PersonaSeed, PersonaEmotionResponse } from "../types/domain";

// ── Types ──────────────────────────────────────────────────────
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
  const { config, activeTab, pushConfig, onAutoCheckpoint, onExplorationResult,
    toolsLength, skillsLength, setTools, setSkills } = params;

  // ── Chat message state ───────────────────────────────────────
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Welcome to **Huginn**.\n\n*Magic springs from the wellspring of imagination.*\n\nI'm your materials-science research assistant. Set your LLM provider and API key in **Settings** on the left, then start a chat. I can help with DFT, molecular dynamics, packing, symbolic math, UQ/GP, and formal Lean verification.",
      timestamp: formatTime(),
    },
  ]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"chat" | "plan" | "build">("chat");
  const [pendingPlan, setPendingPlan] = useState<string>("");
  const [planLoading, setPlanLoading] = useState(false);
  const [status, setStatus] = useState<string>("connecting…");
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pendingResponseRef = useRef<string>("");
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

  // ── Persona state ────────────────────────────────────────────
  const [personaList, setPersonaList] = useState<{ id: string; label: string; description?: string; avatar?: string }[]>(PERSONAS_FALLBACK);
  const [personaEmotion, setPersonaEmotion] = useState<{ mood: string; valence: number; arousal: number; trust: number } | null>(null);
  const [pendingClarifications, setPendingClarifications] = useState<{ question_id?: string; question: string; options?: string[]; thread_id?: string }[]>([]);

  // ── Thread state ─────────────────────────────────────────────
  const [threads, setThreads] = useState<Thread[]>([
    { id: "desktop", label: "Default", created_at: "", last_active: "" },
  ]);
  const [activeThread, setActiveThread] = useState<string>("desktop");

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
      const data = await api.post<{ id: string; label: string }>("/threads", { label: "New thread" });
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
      setStatus(`backend start failed: ${e}`);
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
        if (alive) setStatus("desktop ready");
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

  // ── WebSocket message handler ────────────────────────────────
  const handleWsMessage = (data: WSMessage) => {
    switch (data.type) {
      case "text_delta":
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant" && last.timestamp === "streaming") {
            const updated = [...prev];
            updated[updated.length - 1] = { ...last, content: last.content + data.text };
            return updated;
          }
          return [...prev, { role: "assistant", content: data.text, timestamp: "streaming" }];
        });
        pendingResponseRef.current += data.text;
        setIsStreaming(true);
        break;
      case "done":
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.timestamp === "streaming") {
            updated[updated.length - 1] = { ...last, timestamp: formatTime() };
          }
          return updated;
        });
        setIsStreaming(false);
        playTaskComplete();
        notify("Huginn", pendingResponseRef.current.slice(0, 120) || "Agent finished");
        break;
      case "error":
        playErrorSound();
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `❌ ${data.error}`, timestamp: formatTime() },
        ]);
        setIsStreaming(false);
        notify("Huginn", `Error: ${data.error?.slice(0, 120) || "unknown"}`);
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
      case "approval_request":
        console.log("[WS] Approval request:", data.tool_name, data.reason);
        break;
      case "auto_approve_set":
        console.log("[WS] Auto-approve set:", data.enabled, data.scope);
        break;
      case "ping":
        if (wsClientRef.current) {
          wsClientRef.current.send(JSON.stringify({ type: "pong" }));
        }
        break;
    }
  };

  // ── WebSocket initialization ─────────────────────────────────
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
        }
      },
      onMessage: (data) => {
        if (typeof data === "string") return;
        if (isWSMessage(data)) handleWsMessage(data);
      },
      onConnected: () => {
        pushConfig(loadStoredConfig());
        setPendingClarifications([]);
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((m) =>
            m.timestamp === "streaming"
              ? { ...m, timestamp: formatTime(), content: m.content + "\n\n[连接中断，回复未完成]" }
              : m
          )
        );
      },
    });

    ws.connect();
    wsClientRef.current = ws;

    return () => {
      ws.close();
      wsClientRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Auto-scroll on new messages ──────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
    if (!input.trim() || isStreaming) return;

    const content = input.trim();

    if (mode === "plan") {
      setPlanLoading(true);
      setPendingPlan("");
      try {
        const data = await api.post<{ error?: string; plan?: string } & Record<string, any>>(
          "/plan",
          { content, thread_id: activeThread }
        );
        if (data.error) {
          setPendingPlan(`❌ ${data.error}`);
        } else {
          setPendingPlan(data.plan || "No plan returned.");
        }
      } catch (e: any) {
        setPendingPlan(`❌ Plan request failed: ${e.message}`);
      } finally {
        setPlanLoading(false);
      }
      return;
    }

    if (!wsClientRef.current) return;

    const userMsg: Message = { role: "user", content, timestamp: formatTime() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    pendingResponseRef.current = "";

    const payload = JSON.stringify({ type: "user_input", content: userMsg.content, thread_id: activeThread });
    if (wsClientRef.current) {
      wsClientRef.current.send(payload);
    }
  };

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

  return {
    // Chat state
    messages, input, mode, pendingPlan, planLoading,
    chatSearchOpen, chatSearchQuery,
    isStreaming,
    messagesEndRef,
    // Connection
    isConnected, status,
    wsClientRef,
    // Persona
    personaList, personaEmotion, pendingClarifications,
    // Threads
    threads, activeThread,
    // Guide
    showGuide, closeGuide,
    // Setters
    setInput, setMode, setMessages, setPendingPlan, setChatSearchOpen, setChatSearchQuery,
    setActiveThread, setThreads, setShowGuide,
    // Functions
    sendMessage, answerClarification,
    loadThreads, createThread, renameThread, deleteThread,
    startBackend, notify,
  };
}
