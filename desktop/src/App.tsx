import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

interface Message {
  role: "user" | "assistant" | "tool";
  content: string;
  timestamp: string;
  tool_name?: string;
  tool_args?: any;
  tool_status?: "running" | "done" | "error";
  tool_result?: string;
  tool_call_id?: string;
}

interface ToolInfo {
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
}

interface SkillInfo {
  name: string;
  description: string;
  category: string;
  parameters: Array<{
    name: string;
    type: string;
    description: string;
    required?: boolean;
    default?: any;
  }>;
  tags: string[];
}

interface AppConfig {
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
  ollama_host: string;
  persona: string;
}

interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

const API_BASE = "http://localhost:8000";
const WS_URL =
  ((import.meta as any).env?.VITE_WS_URL as string | undefined) ||
  `${API_BASE.replace("http", "ws")}/ws/agent`;

const CONFIG_KEY = "matsci:config:v1";

const DEFAULT_CONFIG: AppConfig = {
  provider: "openai",
  model: "gpt-4o",
  api_key: "",
  base_url: "",
  ollama_host: "http://localhost:11434",
  persona: "default",
};

const PERSONAS = [
  { id: "default", label: "Default Materials Scientist" },
  { id: "dft_expert", label: "DFT Expert" },
  { id: "md_expert", label: "MD Expert" },
  { id: "reviewer", label: "Critical Reviewer" },
  { id: "tutor", label: "Patient Tutor" },
];

const PROVIDERS = [
  { id: "openai", label: "OpenAI", keyVar: "OPENAI_API_KEY" },
  { id: "anthropic", label: "Anthropic", keyVar: "ANTHROPIC_API_KEY" },
  { id: "deepseek", label: "DeepSeek", keyVar: "DEEPSEEK_API_KEY" },
  { id: "google-genai", label: "Google GenAI", keyVar: "GOOGLE_API_KEY" },
  { id: "openrouter", label: "OpenRouter", keyVar: "OPENROUTER_API_KEY" },
  { id: "nvidia", label: "NVIDIA", keyVar: "NVIDIA_API_KEY" },
  { id: "ollama", label: "Ollama (local)", keyVar: "" },
  { id: "vllm", label: "vLLM / LM Studio", keyVar: "OPENAI_API_KEY" },
  { id: "local", label: "Local OpenAI-compatible", keyVar: "OPENAI_API_KEY" },
];

function loadStoredConfig(): AppConfig {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return { ...DEFAULT_CONFIG, ...parsed };
    }
  } catch {
    // ignore
  }
  return { ...DEFAULT_CONFIG };
}

function saveStoredConfig(config: AppConfig) {
  localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
}

function formatTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function CodeBlock({ code, language = "" }: { code: string; language?: string }) {
  return (
    <div className="my-3 rounded-xl border border-border bg-bg-secondary overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-bg-tertiary">
        <span className="text-xs text-text-muted font-mono uppercase">{language || "code"}</span>
        <button
          className="text-xs text-text-secondary hover:text-text-primary transition-colors"
          onClick={() => navigator.clipboard.writeText(code)}
        >
          Copy
        </button>
      </div>
      <pre className="p-3 m-0 bg-transparent border-0">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function MessageContent({ content }: { content: string }) {
  // Very simple markdown-like splitter: code blocks vs plain text
  const parts = content.split(/(```[\s\S]*?```)/g);
  return (
    <>
      {parts.map((part, i) => {
        const match = part.match(/^```(\w*)\n([\s\S]*?)```$/);
        if (match) {
          return <CodeBlock key={i} language={match[1]} code={match[2].trim()} />;
        }
        return (
          <div key={i} className="whitespace-pre-wrap leading-relaxed">
            {part}
          </div>
        );
      })}
    </>
  );
}

function defaultForSchema(prop: any): any {
  if (prop && "default" in prop) return prop.default;
  const type = prop?.type;
  if (type === "boolean") return false;
  if (type === "integer" || type === "number") return 0;
  if (type === "array") return [];
  if (type === "object") return buildDefaultArgs(prop);
  return "";
}

function buildDefaultArgs(schema: any): Record<string, any> {
  if (!schema || schema.type !== "object") return {};
  const out: Record<string, any> = {};
  for (const [key, prop] of Object.entries(schema.properties || {})) {
    out[key] = defaultForSchema(prop);
  }
  return out;
}

function JsonSchemaForm({
  schema,
  value,
  onChange,
}: {
  schema: any;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
}) {
  if (!schema || schema.type !== "object") return null;
  const required = new Set(schema.required || []);
  const update = (key: string, val: any) => {
    onChange({ ...value, [key]: val });
  };
  return (
    <div className="space-y-4">
      {Object.entries(schema.properties || {}).map(([key, propRaw]) => {
        const prop = propRaw as any;
        const label = (
          <span className="text-xs font-medium text-text-secondary">
            {key}
            {required.has(key) && <span className="ml-1 text-error">*</span>}
          </span>
        );
        const desc = prop.description ? (
          <p className="mt-1 text-xs text-text-muted">{prop.description}</p>
        ) : null;
        let input: React.ReactNode;
        if (prop.type === "boolean") {
          input = (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={!!value[key]}
                onChange={(e) => update(key, e.target.checked)}
                className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
              />
              <span className="text-sm text-text-primary">{value[key] ? "true" : "false"}</span>
            </label>
          );
        } else if (prop.type === "integer" || prop.type === "number") {
          input = (
            <input
              type="number"
              value={value[key] ?? ""}
              onChange={(e) => update(key, e.target.value === "" ? "" : Number(e.target.value))}
              className="input font-mono text-sm"
            />
          );
        } else if (Array.isArray(prop.enum) && prop.enum.length > 0) {
          input = (
            <select
              value={value[key] ?? ""}
              onChange={(e) => update(key, e.target.value)}
              className="input text-sm"
            >
              {prop.enum.map((opt: string) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          );
        } else {
          input = (
            <input
              type="text"
              value={value[key] ?? ""}
              onChange={(e) => update(key, e.target.value)}
              className="input font-mono text-sm"
            />
          );
        }
        return (
          <div key={key}>
            {label}
            <div className="mt-1.5">{input}</div>
            {desc}
          </div>
        );
      })}
    </div>
  );
}

function SkillForm({
  params,
  value,
  onChange,
}: {
  params: SkillInfo["parameters"];
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
}) {
  const update = (key: string, val: any) => onChange({ ...value, [key]: val });
  return (
    <div className="space-y-4">
      {params.map((p) => {
        const label = (
          <span className="text-xs font-medium text-text-secondary">
            {p.name}
            {p.required && <span className="ml-1 text-error">*</span>}
            <span className="ml-2 font-mono text-[10px] text-text-muted">{p.type}</span>
          </span>
        );
        const desc = p.description ? (
          <p className="mt-1 text-xs text-text-muted">{p.description}</p>
        ) : null;
        let input: React.ReactNode;
        if (p.type === "boolean") {
          input = (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={!!value[p.name]}
                onChange={(e) => update(p.name, e.target.checked)}
                className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
              />
              <span className="text-sm text-text-primary">{value[p.name] ? "true" : "false"}</span>
            </label>
          );
        } else if (p.type === "integer" || p.type === "number") {
          input = (
            <input
              type="number"
              value={value[p.name] ?? ""}
              onChange={(e) =>
                update(p.name, e.target.value === "" ? "" : Number(e.target.value))
              }
              className="input font-mono text-sm"
            />
          );
        } else {
          input = (
            <input
              type="text"
              value={value[p.name] ?? ""}
              onChange={(e) => update(p.name, e.target.value)}
              className="input font-mono text-sm"
            />
          );
        }
        return (
          <div key={p.name}>
            {label}
            <div className="mt-1.5">{input}</div>
            {desc}
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Welcome to **MatSci-Agent**. I'm your materials-science research assistant.\n\nSet your LLM provider and API key in **Settings** on the left, then start a chat. I can help with DFT, molecular dynamics, packing, symbolic math, UQ/GP, and formal Lean verification.",
      timestamp: formatTime(),
    },
  ]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<string>("connecting…");
  const [activeTab, setActiveTab] = useState<
    "chat" | "tools" | "memory" | "skills" | "settings" | "files" | "terminal" | "review"
  >("chat");
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [selectedTool, setSelectedTool] = useState<ToolInfo | null>(null);
  const [toolArgs, setToolArgs] = useState<Record<string, any>>({});
  const [toolResult, setToolResult] = useState<string>("");
  const [toolLoading, setToolLoading] = useState(false);

  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [skillArgs, setSkillArgs] = useState<Record<string, any>>({});
  const [skillResult, setSkillResult] = useState<string>("");
  const [skillLoading, setSkillLoading] = useState(false);

  const [config, setConfig] = useState<AppConfig>(loadStoredConfig());
  const [configDirty, setConfigDirty] = useState(false);
  const [configSavedMsg, setConfigSavedMsg] = useState<string>("");

  // Workspace / file explorer state
  const [cwd, setCwd] = useState<string>("");
  const [dirCache, setDirCache] = useState<Record<string, FileEntry[]>>({});
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState<string>("");
  const [editorDirty, setEditorDirty] = useState(false);
  const [editorMsg, setEditorMsg] = useState<string>("");

  const [terminalOutput, setTerminalOutput] = useState<string>("");
  const [terminalInput, setTerminalInput] = useState<string>("");
  const terminalEndRef = useRef<HTMLDivElement>(null);

  interface DiffEntry {
    path: string;
    status: string;
    diff: string;
    old: string;
    new: string;
  }
  interface Checkpoint {
    id: string;
    base: string;
    files: number;
  }
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [activeCp, setActiveCp] = useState<string | null>(null);
  const [diffs, setDiffs] = useState<DiffEntry[]>([]);
  const [selectedDiff, setSelectedDiff] = useState<DiffEntry | null>(null);

  const GUIDE_KEY = "matsci:guide:v1";
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

  const startBackend = useCallback(async () => {
    setStatus("starting backend…");
    try {
      const result = await invoke("start_backend");
      setStatus(`${result} • waiting for health…`);
    } catch (e: any) {
      setStatus(`backend start failed: ${e}`);
    }
  }, []);

  // Native Tauri status check + auto-start backend if needed
  useEffect(() => {
    let alive = true;

    const check = async () => {
      try {
        const s: any = await invoke("get_agent_status");
        if (alive) {
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
      // Try to start the bundled backend once.
      await startBackend();
      // Poll health for up to ~30s.
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 500));
        if (await check()) return;
      }
      if (alive) setStatus("backend did not come online");
    };

    run();
    return () => {
      alive = false;
    };
  }, [startBackend]);

  // Push config to backend whenever it changes or we come online
  const pushConfig = useCallback(async (cfg: AppConfig) => {
    try {
      const resp = await fetch(`${API_BASE}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return true;
    } catch (e) {
      console.warn("[config] failed to push:", e);
      return false;
    }
  }, []);

  const saveConfig = useCallback(
    async (next: AppConfig) => {
      saveStoredConfig(next);
      setConfig(next);
      setConfigDirty(false);
      const ok = await pushConfig(next);
      setConfigSavedMsg(ok ? "Saved and applied to backend." : "Saved locally; will apply once backend is online.");
      setTimeout(() => setConfigSavedMsg(""), 4000);
    },
    [pushConfig]
  );

  // Workspace file explorer helpers
  const loadDir = useCallback(async (path: string) => {
    try {
      const entries = (await invoke("read_dir", { path })) as FileEntry[];
      setDirCache((prev) => ({ ...prev, [path]: entries }));
    } catch (e: any) {
      console.error("[files] read_dir failed:", e);
    }
  }, []);

  const toggleDir = (path: string) => {
    setExpandedDirs((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
        loadDir(path);
      }
      return next;
    });
  };

  const openFile = async (path: string) => {
    try {
      const content = (await invoke("read_file", { path })) as string;
      setSelectedFile(path);
      setEditorContent(content);
      setEditorDirty(false);
      setEditorMsg("");
    } catch (e: any) {
      setEditorMsg(`Failed to open file: ${e}`);
    }
  };

  const saveFile = async () => {
    if (!selectedFile) return;
    try {
      await invoke("write_file", { path: selectedFile, content: editorContent });
      setEditorDirty(false);
      setEditorMsg("Saved.");
      setTimeout(() => setEditorMsg(""), 2000);
    } catch (e: any) {
      setEditorMsg(`Save failed: ${e}`);
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const path = (await invoke("get_cwd")) as string;
        setCwd(path);
        await loadDir(path);
        setExpandedDirs((prev) => new Set(prev).add(path));
      } catch (e) {
        console.error("[files] get_cwd failed:", e);
      }
    })();
  }, [loadDir]);

  // Listen to integrated terminal output
  useEffect(() => {
    let unlisten: UnlistenFn | undefined;
    (async () => {
      unlisten = await listen("terminal-output", (event) => {
        const payload = event.payload as { source: string; text: string };
        setTerminalOutput((prev) => prev + payload.text);
      });
    })();
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "auto" });
  }, [terminalOutput]);

  const createCheckpoint = async () => {
    if (!cwd) return;
    try {
      const cp = (await fetch(`${API_BASE}/checkpoints`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: cwd }),
      }).then((r) => r.json())) as Checkpoint;
      setCheckpoints((prev) => [cp, ...prev]);
      setActiveCp(cp.id);
      loadDiffs(cp.id);
    } catch (e: any) {
      console.error("[review] create checkpoint failed:", e);
    }
  };

  const loadDiffs = async (cpId: string) => {
    try {
      const data = await fetch(`${API_BASE}/checkpoints/${cpId}/diff`).then((r) => r.json());
      setDiffs((data.diffs as DiffEntry[]) || []);
      setSelectedDiff((data.diffs?.[0] as DiffEntry) || null);
      setActiveCp(cpId);
    } catch (e: any) {
      console.error("[review] load diffs failed:", e);
    }
  };

  const acceptCheckpoint = async (cpId: string) => {
    try {
      await fetch(`${API_BASE}/checkpoints/${cpId}/accept`, { method: "POST" });
      setCheckpoints((prev) => prev.filter((c) => c.id !== cpId));
      if (activeCp === cpId) {
        setActiveCp(null);
        setDiffs([]);
        setSelectedDiff(null);
      }
    } catch (e: any) {
      console.error("[review] accept failed:", e);
    }
  };

  const rejectCheckpoint = async (cpId: string) => {
    try {
      await fetch(`${API_BASE}/checkpoints/${cpId}/reject`, { method: "POST" });
      setCheckpoints((prev) => prev.filter((c) => c.id !== cpId));
      if (activeCp === cpId) {
        setActiveCp(null);
        setDiffs([]);
        setSelectedDiff(null);
      }
    } catch (e: any) {
      console.error("[review] reject failed:", e);
    }
  };

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log("[WS] connected");
        // Sync local config to backend on reconnect
        pushConfig(loadStoredConfig());
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWsMessage(data);
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (err) => {
        console.error("[WS] error:", err);
        setIsConnected(false);
      };
    } catch (e) {
      console.error("[WS] failed to connect:", e);
      setIsConnected(false);
    }
  }, [pushConfig]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connectWebSocket]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (activeTab === "tools" && tools.length === 0) {
      fetch(`${API_BASE}/tools`)
        .then((r) => r.json())
        .then((data) => setTools(data))
        .catch((e) => console.error("Failed to load tools:", e));
    }
    if (activeTab === "skills" && skills.length === 0) {
      fetch(`${API_BASE}/skills`)
        .then((r) => r.json())
        .then((data) => setSkills(data))
        .catch((e) => console.error("Failed to load skills:", e));
    }
  }, [activeTab, tools.length, skills.length]);

  const handleWsMessage = (data: any) => {
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
        break;
      case "error":
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `❌ ${data.error}`, timestamp: formatTime() },
        ]);
        setIsStreaming(false);
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
      case "pong":
        break;
    }
  };

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current || isStreaming) return;

    const userMsg: Message = { role: "user", content: input.trim(), timestamp: formatTime() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    wsRef.current.send(
      JSON.stringify({ type: "user_input", content: userMsg.content, thread_id: "desktop" })
    );
  };

  const runTool = async () => {
    if (!selectedTool) return;
    setToolLoading(true);
    setToolResult("");
    try {
      const name = selectedTool.function.name;
      const resp = await fetch(`${API_BASE}/tools/${name}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(toolArgs),
      });
      const data = await resp.json();
      setToolResult(JSON.stringify(data, null, 2));
    } catch (e: any) {
      setToolResult(`Error: ${e.message}`);
    } finally {
      setToolLoading(false);
    }
  };

  const runSkill = async () => {
    if (!selectedSkill) return;
    setSkillLoading(true);
    setSkillResult("");
    try {
      const resp = await fetch(`${API_BASE}/skills/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ skill: selectedSkill.name, args: skillArgs }),
      });
      const data = await resp.json();
      setSkillResult(JSON.stringify(data, null, 2));
    } catch (e: any) {
      setSkillResult(`Error: ${e.message}`);
    } finally {
      setSkillLoading(false);
    }
  };

  const providerLabel = PROVIDERS.find((p) => p.id === config.provider)?.label || config.provider;

  const tabs: { id: typeof activeTab; label: string; icon: string }[] = [
    { id: "chat", label: "Chat", icon: "💬" },
    { id: "files", label: "Files", icon: "📁" },
    { id: "terminal", label: "Terminal", icon: "🖥️" },
    { id: "review", label: "Review", icon: "📝" },
    { id: "tools", label: "Tools", icon: "🔧" },
    { id: "skills", label: "Skills", icon: "⚡" },
    { id: "memory", label: "Memory", icon: "🧠" },
    { id: "settings", label: "Settings", icon: "⚙️" },
  ];

  const renderTree = (path: string, depth = 0) => {
    const entries = dirCache[path];
    if (!entries) return null;
    return (
      <div>
        {entries.map((entry) => (
          <div key={entry.path}>
            <button
              onClick={() => (entry.is_dir ? toggleDir(entry.path) : openFile(entry.path))}
              className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm ${
                selectedFile === entry.path
                  ? "bg-accent text-white"
                  : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
              }`}
              style={{ paddingLeft: depth * 12 + 8 }}
            >
              <span>{entry.is_dir ? (expandedDirs.has(entry.path) ? "📂" : "📁") : "📄"}</span>
              <span className="truncate">{entry.name}</span>
            </button>
            {entry.is_dir &&
              expandedDirs.has(entry.path) &&
              renderTree(entry.path, depth + 1)}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-primary text-text-primary">
      {/* Sidebar */}
      <aside className="flex w-64 flex-col border-r border-border bg-bg-secondary">
        <div className="flex items-center gap-3 px-5 py-4 border-b border-border">
          <span className="text-2xl">🔬</span>
          <div>
            <div className="text-base font-bold tracking-tight">MatSci-Agent</div>
            <div className="text-xs text-text-muted">Research assistant</div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? "bg-accent text-white shadow-glow"
                  : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>

        <div className="border-t border-border p-4">
          <div className="mb-2 flex items-center gap-2 text-xs text-text-muted">
            <span className={`h-2 w-2 rounded-full ${isConnected ? "bg-success" : "bg-error"}`} />
            <span>{isConnected ? "Backend online" : "Backend offline"}</span>
          </div>
          <div className="text-xs text-text-muted truncate">{status}</div>
          <button
            onClick={() => setShowGuide(true)}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-bg-tertiary px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary"
          >
            ❓ Help / Guide
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex flex-1 flex-col min-w-0 bg-bg-primary">
        {/* Header */}
        <header className="flex h-14 items-center justify-between border-b border-border bg-bg-secondary px-6">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium">
              {tabs.find((t) => t.id === activeTab)?.label}
            </span>
            {activeTab === "chat" && (
              <span className="badge border border-border bg-bg-tertiary text-text-secondary">
                {providerLabel} / {config.model || "default"}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {!isConnected && !status.includes("starting") && (
              <button
                onClick={startBackend}
                className="badge bg-error/10 text-error border border-error/20 hover:bg-error/20"
              >
                ▶ Start backend
              </button>
            )}
            {status.includes("starting") && (
              <span className="badge bg-warning/10 text-warning border border-warning/20">
                Starting backend…
              </span>
            )}
            <button
              onClick={() => setActiveTab("settings")}
              className="btn-secondary px-3 py-1.5 text-xs"
            >
              ⚙️ Settings
            </button>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === "chat" && (
            <div className="flex h-full flex-col">
              <div className="flex-1 overflow-y-auto p-6 space-y-5">
                {messages.map((msg, i) => {
                  if (msg.role === "tool") {
                    return (
                      <div key={i} className="flex justify-center">
                        <div className="w-full max-w-2xl rounded-xl border border-border bg-bg-secondary p-4 shadow-sm">
                          <div className="flex items-center gap-2 text-sm font-semibold text-accent">
                            <span>🔧</span>
                            <span>{msg.tool_name}</span>
                            {msg.tool_status === "running" && (
                              <span className="ml-2 inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                            )}
                            {msg.tool_status === "done" && (
                              <span className="ml-2 text-xs text-success">done</span>
                            )}
                          </div>
                          <div className="mt-2 text-xs text-text-secondary">
                            Arguments
                          </div>
                          <pre className="mt-1 max-h-40 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs">
                            {JSON.stringify(msg.tool_args, null, 2)}
                          </pre>
                          {msg.tool_status === "done" && msg.tool_result !== undefined && (
                            <>
                              <div className="mt-3 text-xs text-text-secondary">
                                Result
                              </div>
                              <pre className="mt-1 max-h-60 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs">
                                {msg.tool_result}
                              </pre>
                            </>
                          )}
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div
                      key={i}
                      className={`flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                    >
                      <div
                        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm ${
                          msg.role === "user" ? "bg-accent text-white" : "bg-bg-tertiary text-text-secondary"
                        }`}
                      >
                        {msg.role === "user" ? "You" : "AI"}
                      </div>
                      <div
                        className={`max-w-[75%] rounded-2xl px-5 py-3 ${
                          msg.role === "user"
                            ? "bg-accent text-white rounded-br-none"
                            : "bg-bg-secondary border border-border rounded-bl-none"
                        }`}
                      >
                        <div className="mb-1 flex items-center gap-2 text-xs opacity-70">
                          <span>{msg.role === "user" ? "You" : "Assistant"}</span>
                          <span>
                            {msg.timestamp === "streaming" ? "typing…" : msg.timestamp}
                          </span>
                        </div>
                        <div className="text-[15px] leading-relaxed">
                          <MessageContent content={msg.content} />
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div ref={messagesEndRef} />
              </div>

              <div className="border-t border-border bg-bg-secondary p-4">
                {!isConnected && (
                  <div className="mb-3 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
                    Backend is not connected. Start the server to send messages, or configure it in Settings.
                  </div>
                )}
                <div className="flex items-end gap-3">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    placeholder={
                      isConnected
                        ? "Ask about materials science, DFT, MD, packing, UQ/GP…"
                        : "Backend offline — start server.py"
                    }
                    rows={2}
                    disabled={!isConnected || isStreaming}
                    className="input min-h-[56px] resize-none flex-1"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!isConnected || isStreaming || !input.trim()}
                    className="btn-primary h-11 px-5"
                  >
                    {isStreaming ? "…" : "Send"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "files" && (
            <div className="flex h-full">
              {/* File tree sidebar */}
              <aside className="flex w-72 flex-col border-r border-border bg-bg-secondary">
                <div className="flex h-12 items-center justify-between border-b border-border px-4">
                  <span className="text-sm font-semibold">Workspace</span>
                  <button
                    onClick={() => cwd && loadDir(cwd)}
                    className="text-xs text-text-secondary hover:text-text-primary"
                  >
                    Refresh
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2">
                  {cwd ? (
                    renderTree(cwd)
                  ) : (
                    <div className="p-4 text-xs text-text-muted">Loading workspace…</div>
                  )}
                </div>
                <div className="border-t border-border p-3 text-xs text-text-muted truncate">
                  {cwd}
                </div>
              </aside>

              {/* Editor */}
              <div className="flex flex-1 flex-col bg-bg-primary">
                <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-4">
                  <span className="text-sm font-medium truncate">
                    {selectedFile || "No file selected"}
                  </span>
                  <div className="flex items-center gap-3">
                    {editorDirty && (
                      <span className="text-xs text-warning">Unsaved changes</span>
                    )}
                    {editorMsg && (
                      <span className="text-xs text-success">{editorMsg}</span>
                    )}
                    <button
                      onClick={saveFile}
                      disabled={!selectedFile || !editorDirty}
                      className="btn-primary px-3 py-1.5 text-xs"
                    >
                      Save
                    </button>
                  </div>
                </div>
                {selectedFile ? (
                  <textarea
                    value={editorContent}
                    onChange={(e) => {
                      setEditorContent(e.target.value);
                      setEditorDirty(true);
                    }}
                    className="flex-1 resize-none bg-bg-primary p-4 font-mono text-sm text-text-primary focus:outline-none"
                    spellCheck={false}
                  />
                ) : (
                  <div className="flex flex-1 items-center justify-center text-sm text-text-muted">
                    Select a file from the workspace to edit
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "terminal" && (
            <div className="flex h-full flex-col bg-black">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-4">
                <span className="text-sm font-semibold">Integrated Terminal</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setTerminalOutput("")}
                    className="btn-secondary px-3 py-1.5 text-xs"
                  >
                    Clear
                  </button>
                  <button
                    onClick={() => invoke("stop_terminal")}
                    className="btn-secondary px-3 py-1.5 text-xs"
                  >
                    Stop
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-3 font-mono text-sm">
                <pre className="whitespace-pre-wrap break-all text-text-primary">
                  {terminalOutput}
                </pre>
                <div ref={terminalEndRef} />
              </div>
              <div className="flex items-center gap-2 border-t border-border bg-bg-secondary p-3">
                <span className="font-mono text-sm text-accent">&gt;</span>
                <input
                  type="text"
                  value={terminalInput}
                  onChange={(e) => setTerminalInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && terminalInput.trim()) {
                      const cmd = terminalInput + "\r\n";
                      setTerminalOutput((prev) => prev + "> " + terminalInput + "\n");
                      invoke("write_terminal", { text: cmd }).catch((err) =>
                        setTerminalOutput((prev) => prev + "[error] " + err + "\n")
                      );
                      setTerminalInput("");
                    }
                  }}
                  placeholder="Type a command and press Enter"
                  className="input flex-1 bg-black font-mono text-sm"
                  spellCheck={false}
                />
              </div>
            </div>
          )}

          {activeTab === "review" && (
            <div className="flex h-full">
              {/* Checkpoint list */}
              <aside className="flex w-72 flex-col border-r border-border bg-bg-secondary">
                <div className="flex h-12 items-center justify-between border-b border-border px-4">
                  <span className="text-sm font-semibold">Checkpoints</span>
                  <button
                    onClick={createCheckpoint}
                    disabled={!cwd}
                    className="btn-primary px-3 py-1.5 text-xs"
                  >
                    + New
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto p-3 space-y-2">
                  {checkpoints.length === 0 && (
                    <div className="text-xs text-text-muted">
                      Create a checkpoint before asking the agent to edit files. After the agent
                      runs, come back here to review changes.
                    </div>
                  )}
                  {checkpoints.map((cp) => (
                    <div
                      key={cp.id}
                      onClick={() => loadDiffs(cp.id)}
                      className={`cursor-pointer rounded-lg border border-border p-3 transition-colors ${
                        activeCp === cp.id
                          ? "bg-accent/10 border-accent"
                          : "bg-bg-tertiary hover:bg-bg-primary"
                      }`}
                    >
                      <div className="text-xs font-semibold text-accent">{cp.id}</div>
                      <div className="mt-1 truncate text-xs text-text-muted">{cp.base}</div>
                      <div className="mt-1 text-xs text-text-secondary">{cp.files} files</div>
                    </div>
                  ))}
                </div>
              </aside>

              {/* Diff viewer */}
              <div className="flex flex-1 flex-col bg-bg-primary">
                <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-4">
                  <span className="text-sm font-semibold">
                    {activeCp ? `Checkpoint ${activeCp}` : "Review"}
                  </span>
                  {activeCp && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => rejectCheckpoint(activeCp)}
                        className="btn-secondary px-3 py-1.5 text-xs text-error hover:bg-error/10"
                      >
                        Reject all
                      </button>
                      <button
                        onClick={() => acceptCheckpoint(activeCp)}
                        className="btn-primary px-3 py-1.5 text-xs"
                      >
                        Accept all
                      </button>
                    </div>
                  )}
                </div>

                <div className="flex flex-1 overflow-hidden">
                  {/* File list */}
                  <div className="w-64 overflow-y-auto border-r border-border bg-bg-secondary p-2">
                    {diffs.map((d) => (
                      <button
                        key={d.path}
                        onClick={() => setSelectedDiff(d)}
                        className={`mb-1 flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-xs ${
                          selectedDiff?.path === d.path
                            ? "bg-accent text-white"
                            : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
                        }`}
                      >
                        <span className="truncate">{d.path}</span>
                        <span
                          className={`ml-2 shrink-0 rounded px-1 text-[10px] ${
                            d.status === "added"
                              ? "bg-success/20 text-success"
                              : d.status === "deleted"
                              ? "bg-error/20 text-error"
                              : "bg-warning/20 text-warning"
                          }`}
                        >
                          {d.status}
                        </span>
                      </button>
                    ))}
                    {activeCp && diffs.length === 0 && (
                      <div className="p-2 text-xs text-text-muted">No changes</div>
                    )}
                  </div>

                  {/* Diff content */}
                  <div className="flex-1 overflow-auto bg-bg-primary p-4">
                    {selectedDiff ? (
                      <div>
                        <div className="mb-2 text-sm font-semibold">{selectedDiff.path}</div>
                        <pre className="rounded-lg border border-border bg-bg-secondary p-3 font-mono text-xs whitespace-pre-wrap">
                          {selectedDiff.diff || "(binary or no diff)"}
                        </pre>
                      </div>
                    ) : (
                      <div className="flex h-full items-center justify-center text-sm text-text-muted">
                        Select a changed file to review
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "tools" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Available Tools</h2>
                {selectedTool && (
                  <button onClick={() => setSelectedTool(null)} className="btn-secondary text-xs">
                    ← Back
                  </button>
                )}
              </div>

              {!selectedTool ? (
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {tools.map((tool) => (
                    <button
                      key={tool.function.name}
                      onClick={() => {
                        setSelectedTool(tool);
                        setToolArgs(buildDefaultArgs(tool.function.parameters));
                        setToolResult("");
                      }}
                      className="card text-left transition-colors hover:border-accent"
                    >
                      <div className="text-xs font-semibold uppercase text-accent">Tool</div>
                      <div className="mt-1 text-sm font-semibold">{tool.function.name}</div>
                      <div className="mt-1 text-xs text-text-secondary line-clamp-2">
                        {tool.function.description}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="max-w-3xl space-y-4">
                  <div className="card">
                    <div className="text-xs uppercase text-accent font-semibold">Tool</div>
                    <h3 className="mt-1 text-base font-semibold">{selectedTool.function.name}</h3>
                    <p className="mt-1 text-sm text-text-secondary">
                      {selectedTool.function.description}
                    </p>
                  </div>
                  <div className="card">
                    <div className="mb-3 flex items-center justify-between">
                      <label className="text-xs font-medium text-text-secondary">Arguments</label>
                      <button
                        onClick={() =>
                          setToolArgs(buildDefaultArgs(selectedTool.function.parameters))
                        }
                        className="text-xs text-accent hover:underline"
                      >
                        Reset defaults
                      </button>
                    </div>
                    <JsonSchemaForm
                      schema={selectedTool.function.parameters}
                      value={toolArgs}
                      onChange={setToolArgs}
                    />
                  </div>
                  <button onClick={runTool} disabled={toolLoading} className="btn-primary">
                    {toolLoading ? "Running…" : "Run Tool"}
                  </button>
                  {toolResult && (
                    <div className="card border-accent/20 bg-bg-secondary">
                      <div className="mb-2 text-xs font-semibold text-accent">Result</div>
                      <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">
                        {toolResult}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === "skills" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Declarative Skills</h2>
                {selectedSkill && (
                  <button onClick={() => setSelectedSkill(null)} className="btn-secondary text-xs">
                    ← Back
                  </button>
                )}
              </div>

              {!selectedSkill ? (
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {skills.map((skill) => (
                    <div key={skill.name} className="card flex flex-col">
                      <div className="text-xs font-semibold uppercase text-accent">{skill.category}</div>
                      <div className="mt-1 text-sm font-semibold">{skill.name}</div>
                      <div className="mt-1 flex-1 text-xs text-text-secondary line-clamp-3">
                        {skill.description}
                      </div>
                      <div className="mt-3 text-xs text-text-muted">
                        Tags: {skill.tags.join(", ")}
                      </div>
                      <button
                        onClick={() => {
                          setSelectedSkill(skill);
                          const defaults: Record<string, any> = {};
                          skill.parameters.forEach((p) => {
                            if (p.default !== undefined && p.default !== null)
                              defaults[p.name] = p.default;
                            else defaults[p.name] = p.type === "boolean" ? false : p.type === "number" || p.type === "integer" ? 0 : "";
                          });
                          setSkillArgs(defaults);
                          setSkillResult("");
                        }}
                        className="btn-secondary mt-3 w-full text-xs"
                      >
                        Execute
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="max-w-3xl space-y-4">
                  <div className="card">
                    <div className="text-xs uppercase text-accent font-semibold">Skill</div>
                    <h3 className="mt-1 text-base font-semibold">{selectedSkill.name}</h3>
                    <p className="mt-1 text-sm text-text-secondary">
                      {selectedSkill.description}
                    </p>
                    <div className="mt-2 text-xs text-text-muted">
                      Tags: {selectedSkill.tags.join(", ")}
                    </div>
                  </div>
                  <div className="card">
                    <div className="mb-3 flex items-center justify-between">
                      <label className="text-xs font-medium text-text-secondary">Arguments</label>
                      <button
                        onClick={() => {
                          const defaults: Record<string, any> = {};
                          selectedSkill.parameters.forEach((p) => {
                            defaults[p.name] =
                              p.default !== undefined && p.default !== null
                                ? p.default
                                : p.type === "boolean"
                                ? false
                                : p.type === "number" || p.type === "integer"
                                ? 0
                                : "";
                          });
                          setSkillArgs(defaults);
                        }}
                        className="text-xs text-accent hover:underline"
                      >
                        Reset defaults
                      </button>
                    </div>
                    <SkillForm
                      params={selectedSkill.parameters}
                      value={skillArgs}
                      onChange={setSkillArgs}
                    />
                  </div>
                  <button onClick={runSkill} disabled={skillLoading} className="btn-primary">
                    {skillLoading ? "Running…" : "Run Skill"}
                  </button>
                  {skillResult && (
                    <div className="card border-accent/20 bg-bg-secondary">
                      <div className="mb-2 text-xs font-semibold text-accent">Result</div>
                      <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">
                        {skillResult}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === "memory" && (
            <div className="h-full overflow-y-auto p-6">
              <h2 className="mb-4 text-lg font-semibold">Memory</h2>
              <div className="grid max-w-3xl grid-cols-1 gap-4 md:grid-cols-2">
                <div className="card">
                  <h3 className="text-sm font-semibold">Session Memory</h3>
                  <p className="mt-1 text-xs text-text-secondary">
                    Current conversation context and working memory.
                  </p>
                  <div className="mt-3 flex items-center justify-between text-sm">
                    <span className="text-text-muted">Messages</span>
                    <span className="font-medium">{messages.filter((m) => m.role !== "tool").length}</span>
                  </div>
                </div>
                <div className="card">
                  <h3 className="text-sm font-semibold">Long-term Memory</h3>
                  <p className="mt-1 text-xs text-text-secondary">
                    Stored facts, calculations, and insights from past sessions.
                  </p>
                  <div className="mt-3 flex items-center justify-between text-sm">
                    <span className="text-text-muted">Entries</span>
                    <span className="font-medium">{isConnected ? "—" : "connect to backend"}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "settings" && (
            <div className="h-full overflow-y-auto p-6">
              <h2 className="mb-1 text-lg font-semibold">Settings</h2>
              <p className="mb-6 text-sm text-text-secondary">
                Configure your LLM provider. Settings are saved locally and pushed to the backend when it is online.
              </p>

              <div className="max-w-2xl space-y-5">
                <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      Provider
                    </label>
                    <select
                      value={config.provider}
                      onChange={(e) => {
                        const next = { ...config, provider: e.target.value };
                        setConfig(next);
                        setConfigDirty(true);
                      }}
                      className="input"
                    >
                      {PROVIDERS.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      Model
                    </label>
                    <input
                      type="text"
                      value={config.model}
                      onChange={(e) => {
                        setConfig({ ...config, model: e.target.value });
                        setConfigDirty(true);
                      }}
                      placeholder="e.g. gpt-4o"
                      className="input"
                    />
                  </div>

                  <div className="md:col-span-2">
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      Persona
                    </label>
                    <select
                      value={config.persona}
                      onChange={(e) => {
                        const next = { ...config, persona: e.target.value };
                        setConfig(next);
                        setConfigDirty(true);
                      }}
                      className="input"
                    >
                      {PERSONAS.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-text-muted">
                      Changes the system prompt role used by the agent.
                    </p>
                  </div>

                  <div className="md:col-span-2">
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      API Key
                    </label>
                    <input
                      type="password"
                      value={config.api_key}
                      onChange={(e) => {
                        setConfig({ ...config, api_key: e.target.value });
                        setConfigDirty(true);
                      }}
                      placeholder={
                        PROVIDERS.find((p) => p.id === config.provider)?.keyVar || "API key"
                      }
                      className="input"
                    />
                    <p className="mt-1 text-xs text-text-muted">
                      Stored locally in the app. Sent to the backend when you save.
                    </p>
                  </div>

                  <div className="md:col-span-2">
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      Base URL (optional)
                    </label>
                    <input
                      type="text"
                      value={config.base_url}
                      onChange={(e) => {
                        setConfig({ ...config, base_url: e.target.value });
                        setConfigDirty(true);
                      }}
                      placeholder="https://api.openai.com/v1"
                      className="input"
                    />
                  </div>

                  <div className="md:col-span-2">
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      Ollama Host (for Ollama provider)
                    </label>
                    <input
                      type="text"
                      value={config.ollama_host}
                      onChange={(e) => {
                        setConfig({ ...config, ollama_host: e.target.value });
                        setConfigDirty(true);
                      }}
                      placeholder="http://localhost:11434"
                      className="input"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-3 pt-2">
                  <button
                    onClick={() => saveConfig(config)}
                    disabled={!configDirty}
                    className="btn-primary"
                  >
                    Save Settings
                  </button>
                  {configSavedMsg && (
                    <span className="text-sm text-success">{configSavedMsg}</span>
                  )}
                </div>

                <div className="card mt-6 border-accent/20 bg-accent/5">
                  <h3 className="text-sm font-semibold text-accent">Backend</h3>
                  <p className="mt-1 text-xs text-text-secondary">
                    The desktop app normally starts the Python backend automatically. If it didn't,
                    you can start or stop it here.
                  </p>
                  <div className="mt-3 flex items-center gap-2">
                    <button onClick={startBackend} className="btn-primary text-xs">
                      ▶ Start backend
                    </button>
                    <button
                      onClick={() => invoke("stop_backend")}
                      className="btn-secondary text-xs"
                    >
                      ⏹ Stop backend
                    </button>
                  </div>
                  <p className="mt-2 text-xs text-text-muted">
                    Status: {status}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {showGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl border border-border bg-bg-secondary p-6 shadow-2xl">
            <h2 className="mb-2 text-xl font-bold">Welcome to MatSci-Agent</h2>
            <p className="mb-5 text-sm text-text-secondary">
              A few quick tips to get you started:
            </p>
            <ol className="mb-6 space-y-3 text-sm text-text-primary">
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  1
                </span>
                <span>
                  Open <strong>Settings</strong> and enter your LLM provider / API key. The app
                  saves it locally and pushes it to the backend automatically.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  2
                </span>
                <span>
                  The Python backend starts automatically. If it doesn't, use the{" "}
                  <strong>▶ Start backend</strong> button in the header or Settings.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  3
                </span>
                <span>
                  Switch to <strong>Files</strong> to browse and edit scripts, or use{" "}
                  <strong>Tools</strong> / <strong>Skills</strong> to run capabilities directly.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  4
                </span>
                <span>
                  In chat, tool calls appear as expandable cards so you can see exactly what
                  the agent is doing.
                </span>
              </li>
            </ol>
            <div className="flex justify-end">
              <button onClick={closeGuide} className="btn-primary px-5 py-2">
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
