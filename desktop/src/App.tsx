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

const API_BASE = "http://localhost:8000";
const WS_URL = ((import.meta as any).env?.VITE_WS_URL as string | undefined) || `${API_BASE.replace("http", "ws")}/ws/agent`;

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Welcome to MatSci-Agent! I'm your material science research assistant. I can help with DFT calculations, molecular dynamics, structure analysis, and more.",
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<string>("connecting...");
  const [activeTab, setActiveTab] = useState<"chat" | "tools" | "memory" | "skills">("chat");
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // Tools state
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [selectedTool, setSelectedTool] = useState<ToolInfo | null>(null);
  const [toolArgs, setToolArgs] = useState("{}");
  const [toolResult, setToolResult] = useState<string>("");
  const [toolLoading, setToolLoading] = useState(false);

  // Skills state
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [skillArgs, setSkillArgs] = useState("{}");
  const [skillResult, setSkillResult] = useState<string>("");
  const [skillLoading, setSkillLoading] = useState(false);

  // Tauri native status check
  useEffect(() => {
    invoke("get_agent_status")
      .then((s: any) => setStatus(`${s.status} \u2022 v${s.version || "0.1.0"}`))
      .catch(() => setStatus("desktop ready"));
  }, []);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log("[WS] Connected to", WS_URL);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWsMessage(data);
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;
        // Auto-reconnect after 3s
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (err) => {
        console.error("[WS] Error:", err);
        setIsConnected(false);
      };
    } catch (e) {
      console.error("[WS] Failed to connect:", e);
      setIsConnected(false);
    }
  }, []);

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

  // Fetch tools and skills when their tabs are active
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
            updated[updated.length - 1] = {
              ...last,
              content: last.content + data.text,
            };
            return updated;
          } else {
            return [...prev, {
              role: "assistant",
              content: data.text,
              timestamp: "streaming",
            }];
          }
        });
        setIsStreaming(true);
        break;

      case "done":
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.timestamp === "streaming") {
            updated[updated.length - 1] = { ...last, timestamp: new Date().toLocaleTimeString() };
          }
          return updated;
        });
        setIsStreaming(false);
        break;

      case "error":
        setMessages((prev) => [...prev, {
          role: "assistant",
          content: `Error: ${data.error}`,
          timestamp: new Date().toLocaleTimeString(),
        }]);
        setIsStreaming(false);
        break;

      case "pong":
        break;
    }
  };

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current || isStreaming) return;

    const userMsg: Message = {
      role: "user",
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    wsRef.current.send(JSON.stringify({
      type: "user_input",
      content: userMsg.content,
      thread_id: "desktop",
    }));
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

  const renderSidebar = () => (
    <div className="sidebar">
      <div className="logo">
        <span className="logo-icon">🔬</span>
        <span className="logo-text">MatSci-Agent</span>
      </div>
      <nav className="nav">
        {[
          { id: "chat" as const, label: "Chat", icon: "💬" },
          { id: "tools" as const, label: "Tools", icon: "🔧" },
          { id: "memory" as const, label: "Memory", icon: "🧠" },
          { id: "skills" as const, label: "Skills", icon: "⚡" },
        ].map((tab) => (
          <button
            key={tab.id}
            className={`nav-item ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>
      <div className="status-bar">
        <span className={`status-dot ${isConnected ? "online" : "offline"}`} />
        <span className="status-text">{isConnected ? "live" : "offline"} • {status}</span>
      </div>
    </div>
  );

  const renderChat = () => (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-header">
              <span className="message-role">{msg.role}</span>
              <span className="message-time">{msg.timestamp === "streaming" ? "typing..." : msg.timestamp}</span>
            </div>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder={isConnected ? "Ask about materials science..." : "Backend offline — start server.py"}
          rows={2}
          disabled={!isConnected || isStreaming}
        />
        <button className="primary" onClick={sendMessage} disabled={!isConnected || isStreaming}>
          {isStreaming ? "..." : "Send"}
        </button>
      </div>
    </div>
  );

  const renderTools = () => (
    <div className="panel">
      <h2>Available Tools</h2>
      {!selectedTool ? (
        <div className="tool-grid">
          {tools.map((tool) => (
            <div
              key={tool.function.name}
              className="card tool-card"
              onClick={() => {
                setSelectedTool(tool);
                setToolArgs("{}");
                setToolResult("");
              }}
            >
              <div className="tool-category">Tool</div>
              <div className="tool-name">{tool.function.name}</div>
              <div className="tool-desc">{tool.function.description}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="form-panel">
          <button className="secondary" onClick={() => setSelectedTool(null)}>← Back to tools</button>
          <h3>{selectedTool.function.name}</h3>
          <p className="muted">{selectedTool.function.description}</p>
          <label>Arguments (JSON)</label>
          <textarea
            value={toolArgs}
            onChange={(e) => setToolArgs(e.target.value)}
            rows={8}
            className="json-input"
          />
          <button className="primary" onClick={runTool} disabled={toolLoading}>
            {toolLoading ? "Running..." : "Run Tool"}
          </button>
          {toolResult && (
            <div className="result-box">
              <pre>{toolResult}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderMemory = () => (
    <div className="panel">
      <h2>Memory</h2>
      <div className="memory-sections">
        <div className="card">
          <h3>Session Memory</h3>
          <p className="muted">Current conversation context and working memory</p>
          <div className="stat-row">
            <span>Messages:</span>
            <span>{messages.filter((m) => m.role !== "tool").length}</span>
          </div>
        </div>
        <div className="card">
          <h3>Long-term Memory</h3>
          <p className="muted">Stored facts, calculations, and insights</p>
          <div className="stat-row">
            <span>Entries:</span>
            <span>0 (connect to backend)</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSkills = () => (
    <div className="panel">
      <h2>Skills</h2>
      {!selectedSkill ? (
        <div className="skill-list">
          {skills.map((skill) => (
            <div key={skill.name} className="card skill-card">
              <div className="skill-name">{skill.name}</div>
              <div className="skill-desc">{skill.description}</div>
              <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
                Tags: {skill.tags.join(", ")}
              </div>
              <button
                className="secondary"
                onClick={() => {
                  setSelectedSkill(skill);
                  const defaults: Record<string, any> = {};
                  skill.parameters.forEach((p) => {
                    if (p.default !== undefined && p.default !== null) defaults[p.name] = p.default;
                  });
                  setSkillArgs(JSON.stringify(defaults, null, 2));
                  setSkillResult("");
                }}
              >
                Execute
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="form-panel">
          <button className="secondary" onClick={() => setSelectedSkill(null)}>← Back to skills</button>
          <h3>{selectedSkill.name}</h3>
          <p className="muted">{selectedSkill.description}</p>
          <label>Arguments (JSON)</label>
          <textarea
            value={skillArgs}
            onChange={(e) => setSkillArgs(e.target.value)}
            rows={10}
            className="json-input"
          />
          <button className="primary" onClick={runSkill} disabled={skillLoading}>
            {skillLoading ? "Running..." : "Run Skill"}
          </button>
          {skillResult && (
            <div className="result-box">
              <pre>{skillResult}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div className="app">
      {renderSidebar()}
      <main className="main">
        {activeTab === "chat" && renderChat()}
        {activeTab === "tools" && renderTools()}
        {activeTab === "memory" && renderMemory()}
        {activeTab === "skills" && renderSkills()}
      </main>
      <style>{`
        .app {
          display: flex;
          height: 100vh;
          width: 100vw;
          overflow: hidden;
        }
        .sidebar {
          width: 240px;
          background: var(--bg-secondary);
          border-right: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          padding: var(--spacing);
        }
        .logo {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 24px;
          padding-bottom: 16px;
          border-bottom: 1px solid var(--border);
        }
        .logo-icon { font-size: 24px; }
        .logo-text { font-size: 18px; font-weight: 700; }
        .nav { display: flex; flex-direction: column; gap: 4px; flex: 1; }
        .nav-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 12px;
          border-radius: var(--radius);
          background: transparent;
          color: var(--text-secondary);
          text-align: left;
          font-weight: 500;
          border: none;
          cursor: pointer;
        }
        .nav-item:hover { background: var(--bg-tertiary); color: var(--text-primary); }
        .nav-item.active { background: var(--accent); color: white; }
        .status-bar {
          display: flex;
          align-items: center;
          gap: 8px;
          padding-top: 12px;
          border-top: 1px solid var(--border);
          font-size: 12px;
          color: var(--text-secondary);
        }
        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }
        .status-dot.online { background: var(--success); }
        .status-dot.offline { background: var(--error); }
        .main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
        .chat-container { display: flex; flex-direction: column; height: 100%; }
        .messages { flex: 1; overflow-y: auto; padding: var(--spacing); display: flex; flex-direction: column; gap: 12px; }
        .message { max-width: 80%; padding: 12px; border-radius: var(--radius); }
        .message.user { align-self: flex-end; background: var(--accent); color: white; }
        .message.assistant { align-self: flex-start; background: var(--bg-secondary); border: 1px solid var(--border); }
        .message-header { display: flex; gap: 8px; margin-bottom: 4px; font-size: 12px; opacity: 0.7; }
        .message-content { white-space: pre-wrap; word-break: break-word; }
        .input-area { display: flex; gap: 8px; padding: var(--spacing); border-top: 1px solid var(--border); }
        .input-area textarea { flex: 1; resize: none; }
        .input-area textarea:disabled { opacity: 0.5; }
        .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
        .panel { padding: var(--spacing); overflow-y: auto; height: 100%; }
        .panel h2 { margin-bottom: 16px; font-size: 20px; }
        .tool-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
        .tool-card { cursor: pointer; }
        .tool-card:hover { border-color: var(--accent); }
        .tool-card .tool-category { font-size: 11px; text-transform: uppercase; color: var(--accent); margin-bottom: 4px; }
        .tool-card .tool-name { font-weight: 600; margin-bottom: 4px; }
        .tool-card .tool-desc { font-size: 13px; color: var(--text-secondary); }
        .memory-sections { display: flex; flex-direction: column; gap: 12px; max-width: 600px; }
        .stat-row { display: flex; justify-content: space-between; margin-top: 8px; font-size: 14px; }
        .skill-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
        .skill-card .skill-name { font-weight: 600; margin-bottom: 4px; }
        .skill-card .skill-desc { font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; }
        .muted { color: var(--text-secondary); font-size: 13px; }
        .form-panel { max-width: 700px; display: flex; flex-direction: column; gap: 12px; }
        .form-panel label { font-weight: 500; font-size: 14px; }
        .json-input { font-family: monospace; font-size: 13px; }
        .result-box {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 12px;
          max-height: 400px;
          overflow: auto;
        }
        .result-box pre { margin: 0; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
      `}</style>
    </div>
  );
}

export default App;
