import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

interface Message {
  role: "user" | "assistant" | "tool";
  content: string;
  timestamp: string;
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
};

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
  const [activeTab, setActiveTab] = useState<"chat" | "tools" | "memory" | "skills" | "settings">("chat");
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [selectedTool, setSelectedTool] = useState<ToolInfo | null>(null);
  const [toolArgs, setToolArgs] = useState("{}");
  const [toolResult, setToolResult] = useState<string>("");
  const [toolLoading, setToolLoading] = useState(false);

  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [skillArgs, setSkillArgs] = useState("{}");
  const [skillResult, setSkillResult] = useState<string>("");
  const [skillLoading, setSkillLoading] = useState(false);

  const [config, setConfig] = useState<AppConfig>(loadStoredConfig());
  const [configDirty, setConfigDirty] = useState(false);
  const [configSavedMsg, setConfigSavedMsg] = useState<string>("");

  // Native Tauri status check
  useEffect(() => {
    invoke("get_agent_status")
      .then((s: any) => setStatus(`${s.status} • v${s.version || "0.1.0"}`))
      .catch(() => setStatus("desktop ready"));
  }, []);

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
      const args = JSON.parse(toolArgs);
      const resp = await fetch(`${API_BASE}/tools/${name}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
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
      const args = JSON.parse(skillArgs);
      const resp = await fetch(`${API_BASE}/skills/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ skill: selectedSkill.name, args }),
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
    { id: "tools", label: "Tools", icon: "🔧" },
    { id: "skills", label: "Skills", icon: "⚡" },
    { id: "memory", label: "Memory", icon: "🧠" },
    { id: "settings", label: "Settings", icon: "⚙️" },
  ];

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
            {!isConnected && (
              <span className="badge bg-error/10 text-error border border-error/20">
                Start backend: cd agent && matsci-agent server
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
                {messages.map((msg, i) => (
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
                ))}
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
                        setToolArgs("{}");
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
                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      Arguments (JSON)
                    </label>
                    <textarea
                      value={toolArgs}
                      onChange={(e) => setToolArgs(e.target.value)}
                      rows={10}
                      className="input font-mono text-sm"
                    />
                  </div>
                  <button onClick={runTool} disabled={toolLoading} className="btn-primary">
                    {toolLoading ? "Running…" : "Run Tool"}
                  </button>
                  {toolResult && (
                    <div className="card border-accent/20 bg-bg-secondary">
                      <pre className="max-h-96 overflow-auto text-xs">{toolResult}</pre>
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
                          });
                          setSkillArgs(JSON.stringify(defaults, null, 2));
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
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                      Arguments (JSON)
                    </label>
                    <textarea
                      value={skillArgs}
                      onChange={(e) => setSkillArgs(e.target.value)}
                      rows={12}
                      className="input font-mono text-sm"
                    />
                  </div>
                  <button onClick={runSkill} disabled={skillLoading} className="btn-primary">
                    {skillLoading ? "Running…" : "Run Skill"}
                  </button>
                  {skillResult && (
                    <div className="card border-accent/20 bg-bg-secondary">
                      <pre className="max-h-96 overflow-auto text-xs">{skillResult}</pre>
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
                  <h3 className="text-sm font-semibold text-accent">How to start the backend</h3>
                  <p className="mt-1 text-xs text-text-secondary">
                    The desktop UI talks to the Python server on port 8000. In a terminal run:
                  </p>
                  <pre className="mt-3 text-xs">cd agent
matsci-agent server</pre>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
