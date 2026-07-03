import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";
import Pet from "./Pet";
import { playTaskComplete, playError as playErrorSound } from "./sounds";
import PeriodicTable from "./components/PeriodicTable";
import SandboxPanel from "./components/SandboxPanel";
import Notebook from "./components/Notebook";
import DiffViewer from "./components/DiffViewer";
import SweepDashboard from "./components/SweepDashboard";
import StructureViewer from "./components/StructureViewer";
import EmotionTrackerPanel from "./components/EmotionTracker";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import {
  MessageSquare, Wrench, Zap, FolderTree, Terminal, Settings,
  Users, Code2, FlaskConical, Brain, BookOpen, GitBranch,
  MessageCircle, Puzzle, FileText, Bird, Briefcase, HelpCircle,
  Dna, Play, Compass, Stethoscope, Monitor, ChevronDown, Sparkles,
  Search, X,
  Atom, Notebook as NotebookIcon, TerminalSquare, BarChart3, Box, Activity,
} from 'lucide-react';

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
  destructive?: boolean;
  read_only?: boolean;
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

interface ModelConfig {
  alias: string;
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
  temperature: number;
  enabled: boolean;
}

interface AgentProfile {
  id: string;
  name: string;
  model_alias: string;
  persona: string;
  tools: string[];
  enabled: boolean;
  max_steps: number;
}

interface AppConfig {
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
  ollama_host: string;
  persona: string;
  rag_enabled: boolean;
  models: ModelConfig[];
  agents: AgentProfile[];
  team_mode_enabled: boolean;
  max_concurrent_subagents: number;
  privacy_redact_secrets: boolean;
  privacy_block_on_secrets: boolean;
  local_only_mode: boolean;
  max_tool_output_tokens: number;
  context_budget_tokens: number;
  pet_name: string;
  pet_personality: "cheerful" | "nerdy" | "calm" | "sassy";
  pet_accessories: string[];
  encrypt_config: boolean;
  encryption_password: string;
  encryption_key_file: string;
}

interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

interface BackendLogEvent {
  source: "stdout" | "stderr";
  text: string;
  time: string;
}

let API_BASE = "http://localhost:8000";
let WS_URL =
  ((import.meta as any).env?.VITE_WS_URL as string | undefined) ||
  `${API_BASE.replace("http", "ws")}/ws/agent`;

/** Update API_BASE and WS_URL from the actual backend port reported by Tauri. */
async function syncBackendUrl() {
  try {
    const port: number = await invoke("get_backend_port");
    if (port && port > 0) {
      API_BASE = `http://localhost:${port}`;
      WS_URL = `${API_BASE.replace("http", "ws")}/ws/agent`;
      console.log(`[API] synced to port ${port}`);
    }
  } catch {
    // Tauri IPC not available (dev/browser) — keep defaults
  }
}

const CONFIG_KEY = "huginn:config:v1";

const DEFAULT_CONFIG: AppConfig = {
  provider: "openai",
  model: "gpt-4o",
  api_key: "",
  base_url: "",
  ollama_host: "http://localhost:11434",
  persona: "default",
  rag_enabled: false,
  models: [],
  agents: [],
  team_mode_enabled: false,
  max_concurrent_subagents: 3,
  privacy_redact_secrets: true,
  privacy_block_on_secrets: false,
  local_only_mode: false,
  max_tool_output_tokens: 25000,
  context_budget_tokens: 0,
  pet_name: "Muninn",
  pet_personality: "cheerful",
  pet_accessories: [],
  encrypt_config: false,
  encryption_password: "",
  encryption_key_file: "",
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

const IS_PET_MODE = window.location.search.includes("pet=1");

async function openPetWindow() {
  try {
    const existing = await WebviewWindow.getByLabel("pet");
    if (existing) {
      await existing.setFocus();
      return;
    }
    const pet = new WebviewWindow("pet", {
      url: "index.html?pet=1",
      width: 180,
      height: 220,
      transparent: true,
      decorations: false,
      alwaysOnTop: true,
      skipTaskbar: true,
      resizable: false,
      center: false,
      x: window.screen.width - 200,
      y: window.screen.height - 260,
    });
    pet.once("tauri://error", (e) => {
      console.error("[pet] failed to create window:", e);
    });
  } catch (err) {
    console.error("[pet] open failed:", err);
  }
}

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

function MessageContent({ content }: { content: string }) {
  return (
    <div className="chat-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const isInline = !match && !className;
            if (isInline) {
              return <code {...props}>{children}</code>;
            }
            return (
              <div className="code-block-wrapper">
                {match && <span className="code-block-lang">{match[1]}</span>}
                <button
                  className="code-block-copy"
                  onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                  title="Copy"
                >📋</button>
                <code className={className} {...props}>{children}</code>
              </div>
            );
          },
          pre({ children }) {
            return <pre>{children}</pre>;
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
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

// 凭据管理面板: SSH 连接 + LLM API key 的长时存储 / 编辑 / 删除 / 测试。
// 自包含组件, 通过模块级 API_BASE 与后端 /credentials 交互; 明文密钥从不出后端,
// 列表只展示脱敏值, 编辑时密钥留空表示"不改"。
function CredentialsPanel() {
  const [sshCreds, setSshCreds] = useState<any[]>([]);
  const [llmCreds, setLlmCreds] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);
  const [editing, setEditing] = useState<{ kind: string; id?: string } | null>(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, any>>({});

  const [sshForm, setSshForm] = useState({
    name: "", host: "", username: "", port: "22", scheduler: "slurm",
    key_path: "", password: "", remote_work_dir: "~/huginn_jobs", strict_host_key_checking: true,
  });
  const [llmForm, setLlmForm] = useState({
    name: "", provider: "openai", model: "", base_url: "", api_key: "", alias: "",
  });

  const flash = (text: string, ok = true) => {
    setMsg({ text, ok });
    setTimeout(() => setMsg(null), 3500);
  };

  const load = async () => {
    try {
      const [ssh, llm] = await Promise.all([
        fetch(`${API_BASE}/credentials?kind=ssh`).then((r) => r.json()),
        fetch(`${API_BASE}/credentials?kind=llm`).then((r) => r.json()),
      ]);
      setSshCreds(ssh.credentials || []);
      setLlmCreds(llm.credentials || []);
    } catch (e: any) {
      flash("加载凭据失败: " + e.message, false);
    }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const startNew = (kind: string) => {
    setEditing({ kind });
    setTestResult({});
    if (kind === "ssh") {
      setSshForm({ name: "", host: "", username: "", port: "22", scheduler: "slurm", key_path: "", password: "", remote_work_dir: "~/huginn_jobs", strict_host_key_checking: true });
    } else {
      setLlmForm({ name: "", provider: "openai", model: "", base_url: "", api_key: "", alias: "" });
    }
  };

  const startEdit = (c: any) => {
    setEditing({ kind: c.kind, id: c.id });
    setTestResult({});
    if (c.kind === "ssh") {
      const m = c.metadata || {};
      setSshForm({
        name: c.name, host: m.host || "", username: m.username || "",
        port: String(m.port || 22), scheduler: m.scheduler || "slurm",
        key_path: m.key_path || "", password: "",
        remote_work_dir: m.remote_work_dir || "~/huginn_jobs",
        strict_host_key_checking: m.strict_host_key_checking !== false,
      });
    } else {
      const m = c.metadata || {};
      setLlmForm({ name: c.name, provider: m.provider || "openai", model: m.model || "", base_url: m.base_url || "", api_key: "", alias: m.alias || "" });
    }
  };

  const saveSsh = async () => {
    if (!sshForm.name.trim() || !sshForm.host.trim() || !sshForm.username.trim()) {
      flash("name / host / username 必填", false);
      return;
    }
    const body: any = {
      kind: "ssh",
      name: sshForm.name,
      metadata: {
        host: sshForm.host, username: sshForm.username, port: Number(sshForm.port) || 22,
        scheduler: sshForm.scheduler, key_path: sshForm.key_path,
        remote_work_dir: sshForm.remote_work_dir,
        strict_host_key_checking: sshForm.strict_host_key_checking,
      },
    };
    if (!editing?.id) {
      body.secret = sshForm.password; // 新建: 密码可空 (密钥认证)
    } else if (sshForm.password) {
      body.secret = sshForm.password; // 编辑: 只有填了才覆盖, 留空=不改
    }
    try {
      const url = editing?.id ? `${API_BASE}/credentials/${editing.id}` : `${API_BASE}/credentials`;
      const method = editing?.id ? "PUT" : "POST";
      const data = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
      if (data.success) { flash(editing?.id ? "SSH 凭据已更新" : "SSH 凭据已创建"); setEditing(null); load(); }
      else flash(data.error || "保存失败", false);
    } catch (e: any) { flash("保存出错: " + e.message, false); }
  };

  const saveLlm = async () => {
    if (!llmForm.name.trim() || !llmForm.provider.trim() || !llmForm.model.trim()) {
      flash("name / provider / model 必填", false);
      return;
    }
    const body: any = {
      kind: "llm",
      name: llmForm.name,
      metadata: { provider: llmForm.provider, model: llmForm.model, base_url: llmForm.base_url, alias: llmForm.alias },
    };
    if (!editing?.id) {
      body.secret = llmForm.api_key;
    } else if (llmForm.api_key) {
      body.secret = llmForm.api_key;
    }
    try {
      const url = editing?.id ? `${API_BASE}/credentials/${editing.id}` : `${API_BASE}/credentials`;
      const method = editing?.id ? "PUT" : "POST";
      const data = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
      if (data.success) { flash(editing?.id ? "LLM 凭据已更新" : "LLM 凭据已创建"); setEditing(null); load(); }
      else flash(data.error || "保存失败", false);
    } catch (e: any) { flash("保存出错: " + e.message, false); }
  };

  const remove = async (id: string, name: string) => {
    if (!confirm(`删除凭据 "${name}"？此操作不可撤销。`)) return;
    try {
      const data = await fetch(`${API_BASE}/credentials/${id}`, { method: "DELETE" }).then((r) => r.json());
      if (data.success) { flash("已删除"); load(); } else flash(data.error || "删除失败", false);
    } catch (e: any) { flash("删除出错: " + e.message, false); }
  };

  const setDef = async (id: string) => {
    try {
      const data = await fetch(`${API_BASE}/credentials/${id}/set-default`, { method: "POST" }).then((r) => r.json());
      if (data.success) { flash("已设为默认"); load(); } else flash(data.error || "设置失败", false);
    } catch (e: any) { flash("出错: " + e.message, false); }
  };

  const test = async (id: string) => {
    setTesting(id);
    setTestResult((p) => ({ ...p, [id]: { loading: true } }));
    try {
      const data = await fetch(`${API_BASE}/credentials/${id}/test`, { method: "POST" }).then((r) => r.json());
      setTestResult((p) => ({ ...p, [id]: data }));
    } catch (e: any) {
      setTestResult((p) => ({ ...p, [id]: { success: false, error: e.message } }));
    }
    setTesting(null);
  };

  const btn = "rounded px-2 py-1 text-xs transition-colors";
  const btnGhost = `${btn} text-text-secondary hover:bg-bg-tertiary hover:text-text-primary`;
  const btnDanger = `${btn} text-error hover:bg-error/10`;

  const renderTestBadge = (id: string) => {
    const r = testResult[id];
    if (!r) return null;
    if (r.loading) return <span className="text-xs text-text-muted">测试中…</span>;
    if (r.success) return <span className="text-xs text-success">✓ {r.hostname ? `hostname=${r.hostname}` : r.model_response ? `回复: ${r.model_response.slice(0, 40)}` : "连通"} {r.latency_ms != null && `· ${r.latency_ms}ms`}</span>;
    return <span className="text-xs text-error">✗ {r.error || "失败"}</span>;
  };

  const sshFormEl = (
    <div className="card space-y-3 border-accent/20 bg-accent/5">
      <h4 className="text-sm font-semibold">{editing?.id ? "编辑 SSH 连接" : "新增 SSH 连接"}</h4>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs text-text-secondary">名称 *</label>
          <input className="input" value={sshForm.name} onChange={(e) => setSshForm({ ...sshForm, name: e.target.value })} placeholder="如: 实验室集群" />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">Host *</label>
          <input className="input" value={sshForm.host} onChange={(e) => setSshForm({ ...sshForm, host: e.target.value })} placeholder="hpc.univ.edu" />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">用户名 *</label>
          <input className="input" value={sshForm.username} onChange={(e) => setSshForm({ ...sshForm, username: e.target.value })} />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">端口</label>
          <input className="input" value={sshForm.port} onChange={(e) => setSshForm({ ...sshForm, port: e.target.value })} />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">调度器</label>
          <select className="input" value={sshForm.scheduler} onChange={(e) => setSshForm({ ...sshForm, scheduler: e.target.value })}>
            <option value="slurm">slurm</option>
            <option value="pbs">pbs</option>
            <option value="local">local</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">SSH 私钥路径 (可选)</label>
          <input className="input" value={sshForm.key_path} onChange={(e) => setSshForm({ ...sshForm, key_path: e.target.value })} placeholder="~/.ssh/id_rsa" />
        </div>
        <div className="md:col-span-2">
          <label className="mb-1 block text-xs text-text-secondary">密码 (可选; {editing?.id ? "留空=不修改" : "密钥认证可留空"})</label>
          <input type="password" className="input" value={sshForm.password} onChange={(e) => setSshForm({ ...sshForm, password: e.target.value })} placeholder="••••••••" />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">远程工作目录</label>
          <input className="input" value={sshForm.remote_work_dir} onChange={(e) => setSshForm({ ...sshForm, remote_work_dir: e.target.value })} />
        </div>
        <div>
          <label className="flex cursor-pointer items-center gap-2 pt-5">
            <input type="checkbox" className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent" checked={sshForm.strict_host_key_checking} onChange={(e) => setSshForm({ ...sshForm, strict_host_key_checking: e.target.checked })} />
            <span className="text-xs text-text-primary">Strict host key checking</span>
          </label>
        </div>
      </div>
      <div className="flex gap-2">
        <button onClick={saveSsh} className="btn-primary text-xs">保存</button>
        <button onClick={() => setEditing(null)} className="btn-secondary text-xs">取消</button>
      </div>
    </div>
  );

  const llmFormEl = (
    <div className="card space-y-3 border-accent/20 bg-accent/5">
      <h4 className="text-sm font-semibold">{editing?.id ? "编辑 LLM 凭据" : "新增 LLM 凭据"}</h4>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs text-text-secondary">名称 *</label>
          <input className="input" value={llmForm.name} onChange={(e) => setLlmForm({ ...llmForm, name: e.target.value })} placeholder="如: DeepSeek 主 key" />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">Provider *</label>
          <select className="input" value={llmForm.provider} onChange={(e) => setLlmForm({ ...llmForm, provider: e.target.value })}>
            {["openai", "anthropic", "deepseek", "google", "openrouter", "nvidia", "ollama", "vllm", "local"].map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">Model *</label>
          <input className="input" value={llmForm.model} onChange={(e) => setLlmForm({ ...llmForm, model: e.target.value })} placeholder="deepseek-chat / gpt-4o / ..." />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">Base URL (可选)</label>
          <input className="input" value={llmForm.base_url} onChange={(e) => setLlmForm({ ...llmForm, base_url: e.target.value })} placeholder="https://api.deepseek.com" />
        </div>
        <div className="md:col-span-2">
          <label className="mb-1 block text-xs text-text-secondary">API Key ({editing?.id ? "留空=不修改" : "必填, 本地 provider 可填占位"})</label>
          <input type="password" className="input" value={llmForm.api_key} onChange={(e) => setLlmForm({ ...llmForm, api_key: e.target.value })} placeholder="sk-..." />
        </div>
      </div>
      <div className="flex gap-2">
        <button onClick={saveLlm} className="btn-primary text-xs">保存</button>
        <button onClick={() => setEditing(null)} className="btn-secondary text-xs">取消</button>
      </div>
    </div>
  );

  const renderCard = (c: any, subtitle: string) => (
    <div key={c.id} className="card">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium text-text-primary">{c.name}</span>
            {c.is_default && <span className="rounded bg-accent/15 px-1.5 py-0.5 text-[10px] font-semibold text-accent">默认</span>}
          </div>
          <div className="mt-0.5 truncate text-xs text-text-secondary">{subtitle}</div>
          <div className="mt-0.5 text-xs text-text-muted">{c.has_secret ? `密钥: ${c.secret_masked}` : "无密钥 (密钥认证)"}</div>
          <div className="mt-1">{renderTestBadge(c.id)}</div>
        </div>
        <div className="flex flex-shrink-0 flex-wrap justify-end gap-1">
          {!c.is_default && <button onClick={() => setDef(c.id)} className={btnGhost}>设默认</button>}
          <button onClick={() => test(c.id)} disabled={testing === c.id} className={btnGhost}>{testing === c.id ? "…" : "测试"}</button>
          <button onClick={() => startEdit(c)} className={btnGhost}>编辑</button>
          <button onClick={() => remove(c.id, c.name)} className={btnDanger}>删除</button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h3 className="text-base font-semibold text-text-primary">凭据管理</h3>
        <p className="mt-1 text-sm text-text-secondary">
          长期保存 SSH 连接与 LLM API Key, 加密存储于本地。明文不会回传前端, 可随时编辑 / 删除 / 设默认 / 测试连通性。
        </p>
      </div>
      {msg && <div className={`rounded-lg border px-3 py-2 text-xs ${msg.ok ? "border-success/20 bg-success/10 text-success" : "border-error/20 bg-error/10 text-error"}`}>{msg.text}</div>}

      {/* SSH 连接 */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-text-primary">SSH 连接</h4>
          {editing?.kind !== "ssh" && <button onClick={() => startNew("ssh")} className="btn-secondary text-xs">+ 新增 SSH</button>}
        </div>
        {editing?.kind === "ssh" && sshFormEl}
        {loading ? <p className="text-xs text-text-muted">加载中…</p> : sshCreds.length === 0 && !editing ? <p className="text-xs text-text-muted">暂无 SSH 连接, 点击"新增 SSH"添加。</p> : null}
        <div className="space-y-2">{sshCreds.map((c) => renderCard(c, `${c.metadata?.host || ""} · ${c.metadata?.username || ""} · ${c.metadata?.scheduler || ""}`))}</div>
      </section>

      {/* LLM API Key */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-text-primary">LLM API Key</h4>
          {editing?.kind !== "llm" && <button onClick={() => startNew("llm")} className="btn-secondary text-xs">+ 新增 LLM</button>}
        </div>
        {editing?.kind === "llm" && llmFormEl}
        {loading ? <p className="text-xs text-text-muted">加载中…</p> : llmCreds.length === 0 && !editing ? <p className="text-xs text-text-muted">暂无 LLM 凭据, 点击"新增 LLM"添加。</p> : null}
        <div className="space-y-2">{llmCreds.map((c) => renderCard(c, `${c.metadata?.provider || ""} / ${c.metadata?.model || ""}${c.metadata?.base_url ? " · " + c.metadata.base_url : ""}`))}</div>
      </section>
    </div>
  );
}

export default function App() {
  if (IS_PET_MODE) {
    return <Pet />;
  }

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
  const [activeTab, setActiveTab] = useState<
    | "chat"
    | "tools"
    | "memory"
    | "skills"
    | "settings"
    | "files"
    | "terminal"
    | "review"
    | "knowledge"
    | "logs"
    | "plugins"
    | "threads"
    | "project"
    | "team"
    | "coder"
    | "benchmark"
    | "evolution"
    | "execute"
    | "workflows"
    | "explore"
    | "diagnose"
    | "hpc"
    | "periodic"
    | "notebook"
    | "sandbox"
    | "sweep"
    | "structure"
    | "emotion"
  >("chat");
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  // Sidebar group collapse state (progressive disclosure)
  const [sidebarGroups, setSidebarGroups] = useState<Record<string, boolean>>({
    core: true,
    research: true,
    workspace: false,
    system: false,
  });
  const toggleSidebarGroup = (group: string) =>
    setSidebarGroups((prev) => ({ ...prev, [group]: !prev[group] }));
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
  const [settingsTab, setSettingsTab] = useState<"general" | "models" | "agents" | "privacy" | "pet" | "security" | "credentials">("general");

  // Multi-agent team state
  const [teamObjective, setTeamObjective] = useState("");
  const [teamPlan, setTeamPlan] = useState<any[] | null>(null);
  const [teamRunning, setTeamRunning] = useState(false);
  const [teamResult, setTeamResult] = useState<any>(null);
  const [teamError, setTeamError] = useState("");

  // Coder mode state
  const [coderTask, setCoderTask] = useState("");
  const [coderAutoApprove, setCoderAutoApprove] = useState(false);
  const [coderMaxIters, setCoderMaxIters] = useState<number | "">("");
  const [coderRunning, setCoderRunning] = useState(false);
  const [coderResult, setCoderResult] = useState<string>("");
  const [coderError, setCoderError] = useState("");

  // Workbench features are now independent sidebar tabs (no more workbenchTab state)

  // Collapsible cards in Settings (models/agents)
  const [expandedModels, setExpandedModels] = useState<Set<number>>(new Set());
  const [expandedAgents, setExpandedAgents] = useState<Set<number>>(new Set());
  const toggleModelExpanded = (i: number) => setExpandedModels((prev) => {
    const next = new Set(prev);
    next.has(i) ? next.delete(i) : next.add(i);
    return next;
  });
  const toggleAgentExpanded = (i: number) => setExpandedAgents((prev) => {
    const next = new Set(prev);
    next.has(i) ? next.delete(i) : next.add(i);
    return next;
  });

  // Memory panel view toggle
  const [memoryView, setMemoryView] = useState<"browse" | "add">("browse");

  // Chat message search
  const [chatSearchOpen, setChatSearchOpen] = useState(false);
  const [chatSearchQuery, setChatSearchQuery] = useState("");

  const [benchEvolve, setBenchEvolve] = useState(false);
  const [benchCategories, setBenchCategories] = useState("");
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchResult, setBenchResult] = useState<any>(null);
  const [benchError, setBenchError] = useState("");

  const [evolveRunning, setEvolveRunning] = useState(false);
  const [evolveResult, setEvolveResult] = useState<any>(null);
  const [evolveError, setEvolveError] = useState("");

  const [executeStages, setExecuteStages] = useState("");
  const [executeWorkingDir, setExecuteWorkingDir] = useState(".");
  const [executeName, setExecuteName] = useState("execute");
  const [executeRunning, setExecuteRunning] = useState(false);
  const [executeResult, setExecuteResult] = useState<any>(null);
  const [executeError, setExecuteError] = useState("");

  const [workflowTemplates, setWorkflowTemplates] = useState<string[]>([]);
  const [workflowTemplate, setWorkflowTemplate] = useState("");
  const [workflowArgs, setWorkflowArgs] = useState("");
  const [workflowRunning, setWorkflowRunning] = useState(false);
  const [workflowResult, setWorkflowResult] = useState<any>(null);
  const [workflowError, setWorkflowError] = useState("");

  const [exploreObjective, setExploreObjective] = useState("");
  const [exploreMaxIters, setExploreMaxIters] = useState(20);
  const [exploreMaxBranches, setExploreMaxBranches] = useState(10);
  const [exploreRunning, setExploreRunning] = useState(false);
  const [exploreResult, setExploreResult] = useState<any>(null);
  const [exploreError, setExploreError] = useState("");

  const [diagnoseError, setDiagnoseError] = useState("");
  const [diagnoseSoftware, setDiagnoseSoftware] = useState("");
  const [diagnoseCalcType, setDiagnoseCalcType] = useState("");
  const [diagnoseContext, setDiagnoseContext] = useState("");
  const [diagnoseRunning, setDiagnoseRunning] = useState(false);
  const [diagnoseResult, setDiagnoseResult] = useState<any>(null);
  const [diagnoseErrorMsg, setDiagnoseErrorMsg] = useState("");

  const [hpcHost, setHpcHost] = useState("");
  const [hpcUsername, setHpcUsername] = useState("");
  const [hpcScheduler, setHpcScheduler] = useState<"slurm" | "pbs">("slurm");
  const [hpcKeyPath, setHpcKeyPath] = useState("");
  const [hpcCommand, setHpcCommand] = useState("");
  const [hpcJobName, setHpcJobName] = useState("huginn_job");
  const [hpcWalltime, setHpcWalltime] = useState("01:00:00");
  const [hpcNodes, setHpcNodes] = useState(1);
  const [hpcNtasks, setHpcNtasks] = useState(4);
  const [hpcQueue, setHpcQueue] = useState("");
  const [hpcJobId, setHpcJobId] = useState("");
  const [hpcRunning, setHpcRunning] = useState(false);
  const [hpcResult, setHpcResult] = useState<any>(null);
  const [hpcError, setHpcError] = useState("");

  // Project context + codebase search state
  const [projectContext, setProjectContext] = useState<string>("");
  const [projectContextSource, setProjectContextSource] = useState<string>("none");
  const [projectContextMsg, setProjectContextMsg] = useState<string>("");
  const [codebaseStatus, setCodebaseStatus] = useState<any>(null);
  const [codebaseQuery, setCodebaseQuery] = useState<string>("");
  const [codebaseResults, setCodebaseResults] = useState<any[]>([]);
  const [codebaseMsg, setCodebaseMsg] = useState<string>("");

  // Memory panel state
  interface MemoryEntry {
    id: string;
    category: string;
    content: string;
    tags: string;
    source: string;
    importance: number;
    tier: string;
    created_at: string;
    last_accessed: string;
    expires_at: string | null;
    access_count: number;
  }
  interface MemoryStats {
    longterm_entries: number;
    tier_counts: { short: number; mid: number; long: number };
  }
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [memorySearch, setMemorySearch] = useState<string>("");
  const [memoryFilter, setMemoryFilter] = useState<{ category: string; tier: string }>({ category: "", tier: "" });
  const [memoryForm, setMemoryForm] = useState<{ content: string; category: string; tags: string; importance: number; tier: string }>({
    content: "",
    category: "fact",
    tags: "",
    importance: 0.5,
    tier: "mid",
  });
  const [memoryMsg, setMemoryMsg] = useState<string>("");

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

  const [backendLogs, setBackendLogs] = useState<BackendLogEvent[]>([]);
  const [logFilter, setLogFilter] = useState<"all" | "stdout" | "stderr">("all");
  const backendLogEndRef = useRef<HTMLDivElement>(null);

  interface DiffEntry {
    path: string;
    status: "modified" | "added" | "deleted";
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

  const pendingResponseRef = useRef<string>("");

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

  const notify = useCallback((title: string, body: string) => {
    try {
      if (document.hidden) {
        sendNotification({ title, body });
      }
    } catch {
      // ignore
    }
  }, []);

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

  // Native Tauri status check + auto-start backend if needed
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

  const handleTeamPlan = async () => {
    if (!teamObjective.trim()) return;
    setTeamRunning(true);
    setTeamError("");
    setTeamResult(null);
    try {
      const resp = await fetch(`${API_BASE}/team/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objective: teamObjective }),
      });
      const data = await resp.json();
      if (data.success) {
        setTeamPlan(data.tasks || []);
      } else {
        setTeamError(data.error || "Planning failed.");
        setTeamPlan(null);
      }
    } catch (e: any) {
      setTeamError(e.message || "Network error");
    } finally {
      setTeamRunning(false);
    }
  };

  const handleTeamRun = async () => {
    if (!teamObjective.trim()) return;
    setTeamRunning(true);
    setTeamError("");
    setTeamResult(null);
    try {
      const resp = await fetch(`${API_BASE}/team/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objective: teamObjective }),
      });
      const data = await resp.json();
      if (data.success) {
        setTeamResult(data);
      } else {
        setTeamError(data.error || "Team run failed.");
      }
    } catch (e: any) {
      setTeamError(e.message || "Network error");
    } finally {
      setTeamRunning(false);
    }
  };

  const handleCoderRun = async () => {
    if (!coderTask.trim()) return;
    setCoderRunning(true);
    setCoderError("");
    setCoderResult("");
    try {
      const body: any = { task: coderTask, auto_approve: coderAutoApprove };
      if (coderMaxIters !== "") body.max_iterations = Number(coderMaxIters);
      const resp = await fetch(`${API_BASE}/coder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (data.success) {
        setCoderResult(data.final_answer || "Done.");
      } else {
        setCoderError(data.error || "Coder run failed.");
      }
    } catch (e: any) {
      setCoderError(e.message || "Network error");
    } finally {
      setCoderRunning(false);
    }
  };

  const handleBenchRun = async () => {
    setBenchRunning(true);
    setBenchError("");
    setBenchResult(null);
    try {
      const body: any = { evolve: benchEvolve };
      if (benchCategories.trim()) body.categories = benchCategories.split(",").map((s) => s.trim()).filter(Boolean);
      const data = await fetch(`${API_BASE}/bench/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json());
      if (data.success) {
        setBenchResult(data.report);
      } else {
        setBenchError(data.error || "Benchmark failed.");
      }
    } catch (e: any) {
      setBenchError(e.message || "Network error");
    } finally {
      setBenchRunning(false);
    }
  };

  const handleEvolveRun = async () => {
    setEvolveRunning(true);
    setEvolveError("");
    setEvolveResult(null);
    try {
      const data = await fetch(`${API_BASE}/evolve/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }).then((r) => r.json());
      if (data.success) {
        setEvolveResult(data.report);
      } else {
        setEvolveError(data.error || "Evolution failed.");
      }
    } catch (e: any) {
      setEvolveError(e.message || "Network error");
    } finally {
      setEvolveRunning(false);
    }
  };

  const handleExecuteRun = async () => {
    if (!executeStages.trim()) return;
    setExecuteRunning(true);
    setExecuteError("");
    setExecuteResult(null);
    try {
      let stages: any;
      try {
        stages = JSON.parse(executeStages);
      } catch {
        setExecuteError("Stages must be valid JSON.");
        setExecuteRunning(false);
        return;
      }
      const data = await fetch(`${API_BASE}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stages, working_dir: executeWorkingDir, name: executeName }),
      }).then((r) => r.json());
      if (data.success) {
        setExecuteResult(data);
      } else {
        setExecuteError(data.error || "Execution failed.");
      }
    } catch (e: any) {
      setExecuteError(e.message || "Network error");
    } finally {
      setExecuteRunning(false);
    }
  };

  const loadWorkflowTemplates = async () => {
    try {
      const data = await fetch(`${API_BASE}/workflows`).then((r) => r.json());
      setWorkflowTemplates(Array.isArray(data) ? data : []);
    } catch (e: any) {
      console.error("[workflows] load failed:", e);
    }
  };

  const handleWorkflowRun = async () => {
    if (!workflowTemplate) return;
    setWorkflowRunning(true);
    setWorkflowError("");
    setWorkflowResult(null);
    try {
      const args: any = {};
      workflowArgs.split(" ").forEach((a) => {
        if (!a.includes("=")) return;
        const [k, v] = a.split("=");
        try {
          args[k] = JSON.parse(v);
        } catch {
          args[k] = v;
        }
      });
      const data = await fetch(`${API_BASE}/workflows/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ template: workflowTemplate, args }),
      }).then((r) => r.json());
      if (data.error) {
        setWorkflowError(data.error);
      } else {
        setWorkflowResult(data);
      }
    } catch (e: any) {
      setWorkflowError(e.message || "Network error");
    } finally {
      setWorkflowRunning(false);
    }
  };

  const handleExploreRun = async () => {
    if (!exploreObjective.trim()) return;
    setExploreRunning(true);
    setExploreError("");
    setExploreResult(null);
    try {
      const data = await fetch(`${API_BASE}/explore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          objective: exploreObjective,
          max_iterations: exploreMaxIters,
          max_branches: exploreMaxBranches,
        }),
      }).then((r) => r.json());
      if (data.success) {
        setExploreResult(data);
      } else {
        setExploreError(data.error || "Exploration failed.");
      }
    } catch (e: any) {
      setExploreError(e.message || "Network error");
    } finally {
      setExploreRunning(false);
    }
  };

  const handleDiagnoseRun = async () => {
    if (!diagnoseError.trim()) return;
    setDiagnoseRunning(true);
    setDiagnoseErrorMsg("");
    setDiagnoseResult(null);
    try {
      const data = await fetch(`${API_BASE}/diagnose`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          error_message: diagnoseError,
          software: diagnoseSoftware || undefined,
          calculation_type: diagnoseCalcType || undefined,
          context: diagnoseContext || undefined,
        }),
      }).then((r) => r.json());
      if (data.success) {
        setDiagnoseResult(data.data);
      } else {
        setDiagnoseErrorMsg(data.error || "Diagnosis failed.");
      }
    } catch (e: any) {
      setDiagnoseErrorMsg(e.message || "Network error");
    } finally {
      setDiagnoseRunning(false);
    }
  };

  const handleHpcTest = async () => {
    setHpcRunning(true);
    setHpcError("");
    setHpcResult(null);
    try {
      const data = await fetch(`${API_BASE}/hpc/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host: hpcHost, username: hpcUsername, scheduler: hpcScheduler, key_path: hpcKeyPath || undefined }),
      }).then((r) => r.json());
      if (data.success) {
        setHpcResult(data);
      } else {
        setHpcError(data.error || "HPC test failed.");
      }
    } catch (e: any) {
      setHpcError(e.message || "Network error");
    } finally {
      setHpcRunning(false);
    }
  };

  const handleHpcSubmit = async () => {
    if (!hpcCommand.trim()) return;
    setHpcRunning(true);
    setHpcError("");
    setHpcResult(null);
    try {
      const data = await fetch(`${API_BASE}/hpc/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host: hpcHost,
          username: hpcUsername,
          scheduler: hpcScheduler,
          key_path: hpcKeyPath || undefined,
          command: hpcCommand,
          job_name: hpcJobName,
          walltime: hpcWalltime,
          nodes: hpcNodes,
          ntasks_per_node: hpcNtasks,
          queue: hpcQueue || undefined,
        }),
      }).then((r) => r.json());
      if (data.success) {
        setHpcJobId(data.job_id);
        setHpcResult(data);
      } else {
        setHpcError(data.error || "HPC submit failed.");
      }
    } catch (e: any) {
      setHpcError(e.message || "Network error");
    } finally {
      setHpcRunning(false);
    }
  };

  const handleHpcStatus = async () => {
    if (!hpcJobId.trim()) return;
    setHpcRunning(true);
    setHpcError("");
    try {
      const data = await fetch(`${API_BASE}/hpc/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host: hpcHost,
          username: hpcUsername,
          scheduler: hpcScheduler,
          key_path: hpcKeyPath || undefined,
          job_id: hpcJobId,
        }),
      }).then((r) => r.json());
      if (data.success) {
        setHpcResult(data);
      } else {
        setHpcError(data.error || "HPC status failed.");
      }
    } catch (e: any) {
      setHpcError(e.message || "Network error");
    } finally {
      setHpcRunning(false);
    }
  };

  const ensureDefaultModel = () => {
    if (config.models.length === 0) {
      return {
        ...config,
        models: [
          {
            alias: "default",
            provider: config.provider || "openai",
            model: config.model || "",
            api_key: config.api_key || "",
            base_url: config.base_url || "",
            temperature: 0.7,
            enabled: true,
          },
        ],
      };
    }
    return config;
  };

  const updateModel = (idx: number, patch: Partial<ModelConfig>) => {
    const base = ensureDefaultModel();
    const nextModels = base.models.map((m, i) => (i === idx ? { ...m, ...patch } : m));
    const next = { ...base, models: nextModels };
    setConfig(next);
    setConfigDirty(true);
  };

  const addModel = () => {
    const base = ensureDefaultModel();
    const next: AppConfig = {
      ...base,
      models: [
        ...base.models,
        { alias: `model${base.models.length + 1}`, provider: "openai", model: "", api_key: "", base_url: "", temperature: 0.7, enabled: true },
      ],
    };
    setConfig(next);
    setConfigDirty(true);
  };

  const removeModel = (idx: number) => {
    const next = { ...config, models: config.models.filter((_, i) => i !== idx) };
    setConfig(next);
    setConfigDirty(true);
  };

  const ensureDefaultAgents = () => {
    if (config.agents.length === 0) {
      const modelAlias = config.models[0]?.alias || "default";
      return {
        ...config,
        agents: [
          { id: "lead", name: "Lead", model_alias: modelAlias, persona: "default", tools: [], enabled: true, max_steps: 10 },
          { id: "researcher", name: "Researcher", model_alias: modelAlias, persona: "tutor", tools: [], enabled: true, max_steps: 10 },
          { id: "reviewer", name: "Reviewer", model_alias: modelAlias, persona: "reviewer", tools: [], enabled: true, max_steps: 10 },
        ],
      };
    }
    return config;
  };

  const updateAgent = (idx: number, patch: Partial<AgentProfile>) => {
    const base = ensureDefaultAgents();
    const nextAgents = base.agents.map((a, i) => (i === idx ? { ...a, ...patch } : a));
    const next = { ...base, agents: nextAgents };
    setConfig(next);
    setConfigDirty(true);
  };

  const addAgent = () => {
    const base = ensureDefaultAgents();
    const modelAlias = config.models[0]?.alias || "default";
    const next: AppConfig = {
      ...base,
      agents: [
        ...base.agents,
        { id: `agent${base.agents.length + 1}`, name: "", model_alias: modelAlias, persona: "default", tools: [], enabled: true, max_steps: 10 },
      ],
    };
    setConfig(next);
    setConfigDirty(true);
  };

  const removeAgent = (idx: number) => {
    const next = { ...config, agents: config.agents.filter((_, i) => i !== idx) };
    setConfig(next);
    setConfigDirty(true);
  };

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

  // Listen to backend stdout/stderr
  useEffect(() => {
    let unlisten: UnlistenFn | undefined;
    (async () => {
      unlisten = await listen("backend-log", (event) => {
        const payload = event.payload as { source: string; text: string };
        const source = payload.source === "stderr" ? "stderr" : "stdout";
        setBackendLogs((prev) => [
          ...prev,
          { source, text: payload.text, time: formatTime() },
        ]);
      });
    })();
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  useEffect(() => {
    backendLogEndRef.current?.scrollIntoView({ behavior: "auto" });
  }, [backendLogs, logFilter]);

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
      }
    } catch (e: any) {
      console.error("[review] reject failed:", e);
    }
  };

  // Knowledge base state
  interface KbDoc {
    doc_id: string;
    filename: string;
  }
  const [kbDocs, setKbDocs] = useState<KbDoc[]>([]);
  const [kbAvailable, setKbAvailable] = useState(false);
  const [kbMsg, setKbMsg] = useState<string>("");
  const [kbQuery, setKbQuery] = useState<string>("");
  const [kbChunks, setKbChunks] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadKnowledge = async () => {
    try {
      const data = await fetch(`${API_BASE}/knowledge`).then((r) => r.json());
      setKbDocs(data.documents || []);
      setKbAvailable(data.available);
    } catch (e: any) {
      setKbMsg(`Failed to load knowledge base: ${e.message}`);
    }
  };

  const uploadKnowledge = async (file: File) => {
    setKbMsg("Uploading…");
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await fetch(`${API_BASE}/knowledge/upload`, {
        method: "POST",
        body: form,
      }).then((r) => r.json());
      if (data.success) {
        setKbMsg(`Uploaded ${data.document.chunks} chunks from ${file.name}`);
        loadKnowledge();
      } else {
        setKbMsg(`Upload failed: ${data.error}`);
      }
    } catch (e: any) {
      setKbMsg(`Upload error: ${e.message}`);
    }
  };

  const deleteKnowledge = async (docId: string) => {
    try {
      await fetch(`${API_BASE}/knowledge/${docId}`, { method: "DELETE" });
      loadKnowledge();
    } catch (e: any) {
      setKbMsg(`Delete failed: ${e.message}`);
    }
  };

  const queryKnowledge = async () => {
    if (!kbQuery.trim()) return;
    setKbMsg("Querying…");
    try {
      const data = await fetch(`${API_BASE}/knowledge/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: kbQuery, top_k: 5 }),
      }).then((r) => r.json());
      setKbChunks(data.chunks || []);
      setKbMsg(data.chunks?.length ? `Found ${data.chunks.length} chunks` : "No results");
    } catch (e: any) {
      setKbMsg(`Query failed: ${e.message}`);
    }
  };

  // MCP / Plugin state
  interface McpServer {
    name: string;
    connected: boolean;
    tools: { name: string; description: string; input_schema?: any }[];
  }
  interface DiscoveredServer {
    name: string;
    path: string;
    command: string;
    args: string[];
  }
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);
  const [discoveredServers, setDiscoveredServers] = useState<DiscoveredServer[]>([]);
  const [mcpMsg, setMcpMsg] = useState<string>("");
  const [newMcp, setNewMcp] = useState<{ name: string; command: string; args: string }>({
    name: "",
    command: "python",
    args: "",
  });

  const loadMcp = async () => {
    try {
      const data = await fetch(`${API_BASE}/mcp/servers`).then((r) => r.json());
      setMcpServers(data.servers || []);
    } catch (e: any) {
      setMcpMsg(`Failed to load MCP servers: ${e.message}`);
    }
  };

  const discoverMcp = async () => {
    try {
      const data = await fetch(`${API_BASE}/mcp/servers/discover`).then((r) => r.json());
      setDiscoveredServers(data.servers || []);
    } catch (e: any) {
      setMcpMsg(`Discovery failed: ${e.message}`);
    }
  };

  const connectMcp = async (server: { name: string; command: string; args: string[] }) => {
    setMcpMsg(`Connecting ${server.name}…`);
    try {
      const data = await fetch(`${API_BASE}/mcp/servers/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(server),
      }).then((r) => r.json());
      if (data.success) {
        setMcpMsg(`Connected ${server.name} (${data.tools?.length || 0} tools)`);
        loadMcp();
      } else {
        setMcpMsg(`Connect failed: ${data.error}`);
      }
    } catch (e: any) {
      setMcpMsg(`Connect error: ${e.message}`);
    }
  };

  const disconnectMcp = async (name: string) => {
    setMcpMsg(`Disconnecting ${name}…`);
    try {
      const data = await fetch(`${API_BASE}/mcp/servers/${name}/disconnect`, {
        method: "POST",
      }).then((r) => r.json());
      if (data.success) {
        setMcpMsg(`Disconnected ${name}`);
        loadMcp();
      } else {
        setMcpMsg(`Disconnect failed: ${data.error}`);
      }
    } catch (e: any) {
      setMcpMsg(`Disconnect error: ${e.message}`);
    }
  };

  // Thread state
  interface Thread {
    id: string;
    label: string;
    created_at: string;
    last_active: string;
  }
  const [threads, setThreads] = useState<Thread[]>([
    { id: "desktop", label: "Default", created_at: "", last_active: "" },
  ]);
  const [activeThread, setActiveThread] = useState<string>("desktop");

  const loadThreads = async () => {
    try {
      const data = await fetch(`${API_BASE}/threads`).then((r) => r.json());
      setThreads(data.threads || []);
    } catch (e: any) {
      console.error("[threads] load failed:", e);
    }
  };

  const createThread = async () => {
    try {
      const data = await fetch(`${API_BASE}/threads`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: "New thread" }),
      }).then((r) => r.json());
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
      await fetch(`${API_BASE}/threads/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      }).then((r) => r.json());
      loadThreads();
    } catch (e: any) {
      console.error("[threads] rename failed:", e);
    }
  };

  const deleteThread = async (id: string) => {
    try {
      await fetch(`${API_BASE}/threads/${id}`, { method: "DELETE" });
      if (activeThread === id) {
        setActiveThread("desktop");
      }
      loadThreads();
    } catch (e: any) {
      console.error("[threads] delete failed:", e);
    }
  };

  const loadProjectContext = async () => {
    try {
      const data = await fetch(`${API_BASE}/project-context`).then((r) => r.json());
      setProjectContext(data.content || "");
      setProjectContextSource(data.source || "none");
      setProjectContextMsg("");
    } catch (e: any) {
      setProjectContextMsg(`Load failed: ${e.message}`);
    }
  };

  const saveProjectContext = async () => {
    setProjectContextMsg("Saving…");
    try {
      const data = await fetch(`${API_BASE}/project-context`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: projectContext }),
      }).then((r) => r.json());
      if (data.success) {
        setProjectContextMsg("Saved. Agent will reload on next message.");
      } else {
        setProjectContextMsg(`Save failed: ${data.error}`);
      }
    } catch (e: any) {
      setProjectContextMsg(`Save error: ${e.message}`);
    }
  };

  const loadCodebaseStatus = async () => {
    try {
      const data = await fetch(`${API_BASE}/codebase`).then((r) => r.json());
      setCodebaseStatus(data);
    } catch (e: any) {
      setCodebaseMsg(`Status failed: ${e.message}`);
    }
  };

  const indexCodebase = async () => {
    setCodebaseMsg("Indexing workspace…");
    try {
      const data = await fetch(`${API_BASE}/codebase/index`, { method: "POST" }).then((r) =>
        r.json()
      );
      if (data.success) {
        setCodebaseMsg(`Indexed ${data.indexed_files} files, ${data.chunks} chunks`);
        loadCodebaseStatus();
      } else {
        setCodebaseMsg(`Index failed: ${data.error}`);
      }
    } catch (e: any) {
      setCodebaseMsg(`Index error: ${e.message}`);
    }
  };

  const searchCodebase = async () => {
    if (!codebaseQuery.trim()) return;
    setCodebaseMsg("Searching…");
    try {
      const data = await fetch(`${API_BASE}/codebase/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: codebaseQuery, top_k: 8 }),
      }).then((r) => r.json());
      setCodebaseResults(data.results || []);
      setCodebaseMsg(data.results?.length ? `Found ${data.results.length} results` : "No results");
    } catch (e: any) {
      setCodebaseMsg(`Search error: ${e.message}`);
    }
  };

  const loadMemory = async () => {
    try {
      const params = new URLSearchParams();
      if (memoryFilter.category) params.set("category", memoryFilter.category);
      if (memoryFilter.tier) params.set("tier", memoryFilter.tier);
      params.set("limit", "200");
      const data = await fetch(`${API_BASE}/memory?${params.toString()}`).then((r) => r.json());
      setMemories(data.entries || []);
    } catch (e: any) {
      setMemoryMsg(`Load failed: ${e.message}`);
    }
  };

  const loadMemoryStats = async () => {
    try {
      const data = await fetch(`${API_BASE}/memory/stats`).then((r) => r.json());
      setMemoryStats(data);
    } catch {
      setMemoryStats(null);
    }
  };

  const searchMemory = async () => {
    if (!memorySearch.trim()) {
      loadMemory();
      return;
    }
    setMemoryMsg("Searching…");
    try {
      const data = await fetch(`${API_BASE}/memory/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: memorySearch, top_k: 10 }),
      }).then((r) => r.json());
      setMemories(data.results || []);
      setMemoryMsg(data.results?.length ? `Found ${data.results.length} results` : "No results");
    } catch (e: any) {
      setMemoryMsg(`Search error: ${e.message}`);
    }
  };

  const createMemory = async () => {
    setMemoryMsg("Saving…");
    try {
      const data = await fetch(`${API_BASE}/memory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: memoryForm.content,
          category: memoryForm.category,
          tags: memoryForm.tags.split(",").map((t) => t.trim()).filter(Boolean),
          importance: memoryForm.importance,
          tier: memoryForm.tier,
        }),
      }).then((r) => r.json());
      if (data.success) {
        setMemoryForm({ content: "", category: "fact", tags: "", importance: 0.5, tier: "mid" });
        setMemoryMsg("Memory saved.");
        loadMemory();
        loadMemoryStats();
      } else {
        setMemoryMsg(`Save failed: ${data.error}`);
      }
    } catch (e: any) {
      setMemoryMsg(`Save error: ${e.message}`);
    }
  };

  const deleteMemory = async (id: string) => {
    if (!confirm("Delete this memory?")) return;
    try {
      await fetch(`${API_BASE}/memory/${id}`, { method: "DELETE" });
      loadMemory();
      loadMemoryStats();
    } catch (e: any) {
      setMemoryMsg(`Delete error: ${e.message}`);
    }
  };

  const promoteMemory = async (id: string) => {
    try {
      const data = await fetch(`${API_BASE}/memory/promote/${id}`, { method: "POST" }).then((r) => r.json());
      if (data.success) {
        loadMemory();
        loadMemoryStats();
      } else {
        setMemoryMsg(`Promote failed: ${data.error}`);
      }
    } catch (e: any) {
      setMemoryMsg(`Promote error: ${e.message}`);
    }
  };

  const pruneMemory = async () => {
    if (!confirm("Prune expired and low-importance memories?")) return;
    try {
      const data = await fetch(`${API_BASE}/memory/prune`, { method: "POST" }).then((r) => r.json());
      setMemoryMsg(`Pruned ${data.expired ?? 0} expired, ${data.low_importance ?? 0} low-importance.`);
      loadMemory();
      loadMemoryStats();
    } catch (e: any) {
      setMemoryMsg(`Prune error: ${e.message}`);
    }
  };

  const syncMemoryMd = async () => {
    setMemoryMsg("Syncing MEMORY.md…");
    try {
      const data = await fetch(`${API_BASE}/memory/sync-md`, { method: "POST" }).then((r) => r.json());
      if (data.path) {
        setMemoryMsg(`Synced to ${data.path}`);
      } else {
        setMemoryMsg("Sync returned no path.");
      }
    } catch (e: any) {
      setMemoryMsg(`Sync error: ${e.message}`);
    }
  };

  // Auto-expand sidebar group when activeTab changes (e.g. from deep links)
  useEffect(() => {
    const group = sidebarGroupsData.find((g) => g.tabs.some((t) => t.id === activeTab));
    if (group) {
      setSidebarGroups((prev) => {
        if (prev[group.key]) return prev;
        return { ...prev, [group.key]: true };
      });
    }
  }, [activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (activeTab === "knowledge") {
      loadKnowledge();
    }
    if (activeTab === "plugins") {
      loadMcp();
      discoverMcp();
    }
    if (activeTab === "threads") {
      loadThreads();
    }
    if (activeTab === "project") {
      loadProjectContext();
      loadCodebaseStatus();
    }
    if (activeTab === "memory") {
      loadMemory();
      loadMemoryStats();
    }
    if (activeTab === "workflows" && workflowTemplates.length === 0) {
      loadWorkflowTemplates();
    }
  }, [activeTab, memoryFilter.category, memoryFilter.tier]);

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
        notify(
          "Huginn",
          pendingResponseRef.current.slice(0, 120) || "Agent finished"
        );
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
        setCheckpoints((prev) => [
          { id: data.id, base: data.base, files: data.files },
          ...prev,
        ]);
        setActiveCp(data.id);
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
          setExploreResult(data.data);
          setExploreRunning(false);
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
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    const content = input.trim();

    if (mode === "plan") {
      setPlanLoading(true);
      setPendingPlan("");
      try {
        const data = await fetch(`${API_BASE}/plan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content, thread_id: activeThread }),
        }).then((r) => r.json());
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

    if (!wsRef.current) return;

    const userMsg: Message = { role: "user", content, timestamp: formatTime() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    pendingResponseRef.current = "";

    wsRef.current.send(
      JSON.stringify({ type: "user_input", content: userMsg.content, thread_id: activeThread })
    );
  };

  const runTool = async () => {
    if (!selectedTool) return;
    if (selectedTool.destructive) {
      const ok = window.confirm(
        `⚠️ ${selectedTool.function.name} may overwrite files or run shell commands. Run it anyway?`
      );
      if (!ok) return;
    }
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

  const iconSize = 16;

  interface SidebarTab {
    id: typeof activeTab;
    label: string;
    icon: React.ReactNode;
    indented?: boolean;
  }

  interface SidebarGroup {
    key: string;
    label: string;
    tabs: SidebarTab[];
  }

  const sidebarGroupsData: SidebarGroup[] = [
    {
      key: "core",
      label: "CORE",
      tabs: [
        { id: "chat", label: "Chat", icon: <MessageSquare size={iconSize} /> },
        { id: "team", label: "Team", icon: <Users size={iconSize} /> },
        { id: "coder", label: "Coder", icon: <Code2 size={iconSize} /> },
      ],
    },
    {
      key: "research",
      label: "RESEARCH",
      tabs: [
        { id: "knowledge", label: "Knowledge", icon: <BookOpen size={iconSize} /> },
        { id: "periodic", label: "Periodic Table", icon: <Atom size={iconSize} /> },
        { id: "project", label: "Project", icon: <Briefcase size={iconSize} /> },
        { id: "notebook", label: "Notebook", icon: <NotebookIcon size={iconSize} />, indented: true },
        { id: "benchmark", label: "Benchmark", icon: <FlaskConical size={iconSize} />, indented: true },
        { id: "evolution", label: "Evolution", icon: <Dna size={iconSize} />, indented: true },
        { id: "execute", label: "Execute", icon: <Play size={iconSize} />, indented: true },
        { id: "workflows", label: "Workflows", icon: <Zap size={iconSize} />, indented: true },
        { id: "sweep", label: "Sweep", icon: <BarChart3 size={iconSize} />, indented: true },
        { id: "explore", label: "Explore", icon: <Compass size={iconSize} />, indented: true },
        { id: "diagnose", label: "Diagnose", icon: <Stethoscope size={iconSize} />, indented: true },
        { id: "structure", label: "Structure", icon: <Box size={iconSize} />, indented: true },
        { id: "hpc", label: "HPC", icon: <Monitor size={iconSize} />, indented: true },
      ],
    },
    {
      key: "workspace",
      label: "WORKSPACE",
      tabs: [
        { id: "files", label: "Files", icon: <FolderTree size={iconSize} /> },
        { id: "terminal", label: "Terminal", icon: <Terminal size={iconSize} /> },
        { id: "sandbox", label: "Sandbox", icon: <TerminalSquare size={iconSize} /> },
        { id: "review", label: "Review", icon: <GitBranch size={iconSize} /> },
        { id: "tools", label: "Tools", icon: <Wrench size={iconSize} /> },
        { id: "skills", label: "Skills", icon: <Sparkles size={iconSize} /> },
      ],
    },
    {
      key: "system",
      label: "SYSTEM",
      tabs: [
        { id: "memory", label: "Memory", icon: <Brain size={iconSize} /> },
        { id: "emotion", label: "Emotion", icon: <Activity size={iconSize} /> },
        { id: "plugins", label: "Plugins", icon: <Puzzle size={iconSize} /> },
        { id: "threads", label: "Threads", icon: <MessageCircle size={iconSize} /> },
        { id: "logs", label: "Logs", icon: <FileText size={iconSize} /> },
        { id: "settings", label: "Settings", icon: <Settings size={iconSize} /> },
      ],
    },
  ];

  const allTabs = sidebarGroupsData.flatMap((g) => g.tabs);

  const sectionAccent: Record<string, string> = {
    core: "var(--seed-primary)",
    research: "var(--seed-accent)",
    workspace: "var(--workspace-accent, #b8956a)",
    system: "var(--system-accent, #8a8680)",
  };
  const activeGroupKey = sidebarGroupsData.find((g) => g.tabs.some((t) => t.id === activeTab))?.key ?? "core";
  const activeAccent = sectionAccent[activeGroupKey] ?? sectionAccent.core;

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
            <div className="text-base font-bold tracking-tight">Huginn</div>
            <div className="text-xs text-text-muted">Research assistant</div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 py-2">
          {sidebarGroupsData.map((group, gi) => (
            <div key={group.key} className={gi > 0 ? "mt-1" : ""}>
              {/* Group header - clickable, collapsible */}
              <button
                onClick={() => toggleSidebarGroup(group.key)}
                className="sidebar-group-header flex w-full items-center gap-1.5 px-2 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-text-muted hover:text-text-secondary transition-colors"
              >
                <ChevronDown
                  size={12}
                  className={`transition-transform duration-200 ${
                    sidebarGroups[group.key] ? "rotate-0" : "-rotate-90"
                  }`}
                />
                {group.label}
              </button>

              {/* Collapsible items container */}
              <div
                className="sidebar-group-content overflow-hidden transition-all duration-200 ease-in-out"
                style={{
                  maxHeight: sidebarGroups[group.key] ? `${group.tabs.length * 36 + 4}px` : "0px",
                  opacity: sidebarGroups[group.key] ? 1 : 0,
                }}
              >
                {group.tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                      tab.indented ? "pl-6" : ""
                    } ${
                      activeTab === tab.id
                        ? "bg-accent text-white shadow-glow"
                        : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
                    }`}
                  >
                    <span className="flex-shrink-0">{tab.icon}</span>
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>
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
            <HelpCircle size={14} /> Help / Guide
          </button>
          <button
            onClick={openPetWindow}
            className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-bg-tertiary px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary"
          >
            <Bird size={14} /> Summon Pet
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex flex-1 flex-col min-w-0 bg-bg-primary">
        {/* Header */}
        <header className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
          <div className="flex items-center gap-2.5">
            <span style={{ color: activeAccent }}>
              {allTabs.find((t) => t.id === activeTab)?.icon}
            </span>
            <span className="text-sm font-semibold">
              {allTabs.find((t) => t.id === activeTab)?.label}
            </span>
            {activeTab === "chat" && (
              <>
                <span className="badge border border-border bg-bg-tertiary text-text-secondary">
                  {config.models.length > 0
                    ? `${config.models.filter((m) => m.enabled).length} models`
                    : `${providerLabel} / ${config.model || "default"}`}
                </span>
                <label className="flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
                  <input
                    type="checkbox"
                    checked={config.team_mode_enabled}
                    onChange={(e) => {
                      const next = { ...config, team_mode_enabled: e.target.checked };
                      setConfig(next);
                      setConfigDirty(true);
                      saveConfig(next);
                    }}
                  />
                  Team mode
                </label>
              </>
            )}
          </div>
          <div className="flex items-center gap-3">
            {activeTab === "chat" && (
              <button
                onClick={() => { setChatSearchOpen((p) => !p); if (chatSearchOpen) setChatSearchQuery(""); }}
                className={`rounded-lg p-1.5 transition-colors ${chatSearchOpen ? "bg-bg-tertiary text-accent" : "text-text-muted hover:text-text-secondary"}`}
                title="Search messages"
              >
                <Search size={16} />
              </button>
            )}
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
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-hidden border-l-[3px]" style={{ borderLeftColor: activeAccent, transition: "border-left-color 0.3s ease" }}>
          {activeTab === "chat" && (
            <div className="flex h-full flex-col">
              {chatSearchOpen && (
                <div className="flex items-center gap-2 border-b border-border bg-bg-secondary/50 px-6 py-2">
                  <Search size={14} className="shrink-0 text-text-muted" />
                  <input
                    type="text"
                    autoFocus
                    value={chatSearchQuery}
                    onChange={(e) => setChatSearchQuery(e.target.value)}
                    placeholder="Search messages…"
                    className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
                  />
                  {chatSearchQuery && (
                    <span className="shrink-0 text-[11px] text-text-muted">
                      {messages.filter((m) => m.content.toLowerCase().includes(chatSearchQuery.toLowerCase())).length} matches
                    </span>
                  )}
                  <button onClick={() => { setChatSearchOpen(false); setChatSearchQuery(""); }} className="shrink-0 text-text-muted hover:text-text-secondary">
                    <X size={14} />
                  </button>
                </div>
              )}
              <div className="flex-1 overflow-y-auto p-6 space-y-5">
                {(chatSearchQuery.trim()
                  ? messages.filter((m) => m.content.toLowerCase().includes(chatSearchQuery.toLowerCase()))
                  : messages
                ).map((msg, i) => {
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

                {pendingPlan && (
                  <div className="mb-3 rounded-xl border border-border bg-bg-tertiary p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs font-semibold text-accent">📋 Plan</span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setPendingPlan("")}
                          className="text-xs text-text-secondary hover:text-text-primary"
                        >
                          Dismiss
                        </button>
                        <button
                          onClick={() => {
                            setMode("build");
                            sendMessage();
                          }}
                          disabled={!input.trim() || planLoading}
                          className="btn-primary px-3 py-1 text-xs"
                        >
                          Run plan
                        </button>
                      </div>
                    </div>
                    <div className="max-h-48 overflow-y-auto whitespace-pre-wrap text-xs text-text-primary">
                      <MessageContent content={pendingPlan} />
                    </div>
                  </div>
                )}

                <div className="mb-3 flex items-center gap-2">
                  <div className="flex rounded-lg border border-border bg-bg-tertiary p-0.5 text-xs">
                    {(["chat", "plan", "build"] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => setMode(m)}
                        className={`rounded px-3 py-1 capitalize ${
                          mode === m
                            ? "bg-accent text-white"
                            : "text-text-secondary hover:text-text-primary"
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                  <span className="text-[10px] text-text-muted">
                    {mode === "plan" && "Generate a step-by-step plan without executing tools"}
                    {mode === "build" && "Execute tools and edit files"}
                    {mode === "chat" && "Normal assistant chat"}
                  </span>
                </div>

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
                      mode === "plan"
                        ? "Describe what you want to do. I'll generate a plan first."
                        : mode === "build"
                        ? "Run the plan with tool execution enabled…"
                        : isConnected
                        ? "Ask about materials science, DFT, MD, packing, UQ/GP…"
                        : "Backend offline — start server.py"
                    }
                    rows={2}
                    disabled={!isConnected || isStreaming || planLoading}
                    className="input min-h-[56px] resize-none flex-1"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!isConnected || isStreaming || !input.trim() || planLoading}
                    className="btn-primary h-11 px-5"
                  >
                    {planLoading ? "Planning…" : isStreaming ? "…" : mode === "plan" ? "Plan" : "Send"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "team" && (
            <div className="flex h-full flex-col">
              <div className="flex-1 overflow-y-auto p-6">
                <div className="mx-auto max-w-3xl space-y-5">
                  <div className="card">
                    <h2 className="mb-2 text-base font-semibold">👥 Multi-Agent Team</h2>
                    <p className="text-sm text-text-secondary">
                      The lead agent breaks your objective into subtasks and delegates them to the configured agent profiles.
                    </p>
                    <div className="mt-4 flex items-center gap-2">
                      <label className="flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
                        <input
                          type="checkbox"
                          checked={config.team_mode_enabled}
                          onChange={(e) => {
                            const next = { ...config, team_mode_enabled: e.target.checked };
                            setConfig(next);
                            setConfigDirty(true);
                            saveConfig(next);
                          }}
                        />
                        Enable team mode
                      </label>
                    </div>
                  </div>

                  <div className="card space-y-3">
                    <label className="block text-xs font-medium text-text-secondary">Objective</label>
                    <textarea
                      value={teamObjective}
                      onChange={(e) => setTeamObjective(e.target.value)}
                      placeholder="e.g. Compare VASP and Quantum ESPRESSO for silicon band structure, then suggest which is cheaper for a 50-atom cell."
                      rows={4}
                      disabled={teamRunning}
                      className="input resize-none"
                    />
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleTeamPlan}
                        disabled={!isConnected || teamRunning || !teamObjective.trim()}
                        className="btn-secondary px-3 py-1.5 text-xs"
                      >
                        📋 Plan
                      </button>
                      <button
                        onClick={handleTeamRun}
                        disabled={!isConnected || teamRunning || !teamObjective.trim()}
                        className="btn-primary px-3 py-1.5 text-xs"
                      >
                        {teamRunning ? "Running…" : "▶ Run team"}
                      </button>
                    </div>
                    {teamRunning && (
                      <div className="flex items-center gap-2 text-xs text-text-secondary">
                        <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                        Working with the team…
                      </div>
                    )}
                    {teamError && (
                      <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">
                        {teamError}
                      </div>
                    )}
                  </div>

                  {teamPlan && teamPlan.length > 0 && (
                    <div className="card space-y-3">
                      <h3 className="text-sm font-semibold">Planned tasks</h3>
                      <div className="space-y-2">
                        {teamPlan.map((t) => (
                          <div key={t.task_id} className="rounded-lg border border-border bg-bg-tertiary p-3">
                            <div className="flex items-center gap-2 text-xs font-semibold">
                              <span className="text-accent">{t.task_id}</span>
                              <span className="text-text-muted">→</span>
                              <span>{t.agent_id}</span>
                            </div>
                            <p className="mt-1 text-xs text-text-secondary">{t.prompt}</p>
                            {t.depends_on?.length > 0 && (
                              <p className="mt-1 text-[10px] text-text-muted">Depends on: {t.depends_on.join(", ")}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {teamResult && (
                    <div className="card space-y-3">
                      <h3 className="text-sm font-semibold">Result</h3>
                      <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-sm whitespace-pre-wrap">
                        {teamResult.summary}
                      </div>
                      {Object.keys(teamResult.outputs || {}).length > 0 && (
                        <div className="space-y-2">
                          <h4 className="text-xs font-semibold text-text-secondary">Sub-agent outputs</h4>
                          {Object.entries(teamResult.outputs).map(([taskId, output]: [string, any]) => (
                            <details key={taskId} className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs">
                              <summary className="cursor-pointer font-medium">{taskId}</summary>
                              <div className="mt-2 whitespace-pre-wrap text-text-secondary">{String(output)}</div>
                            </details>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === "coder" && (
            <div className="flex h-full flex-col">
              <div className="flex-1 overflow-y-auto p-6">
                <div className="mx-auto max-w-3xl space-y-5">
                  <div className="card">
                    <h2 className="mb-2 text-base font-semibold">💻 Coder Mode</h2>
                    <p className="text-sm text-text-secondary">
                      Give Muninn a coding task and let it read, write, edit, run shell commands, and commit changes autonomously.
                    </p>
                  </div>

                  <div className="card space-y-3">
                    <label className="block text-xs font-medium text-text-secondary">Task</label>
                    <textarea
                      value={coderTask}
                      onChange={(e) => setCoderTask(e.target.value)}
                      placeholder="e.g. Refactor the VASP parser to use the Rust extension, then add a test."
                      rows={5}
                      disabled={coderRunning}
                      className="input resize-none"
                    />
                    <div className="flex flex-wrap items-center gap-4">
                      <label className="flex cursor-pointer items-center gap-2 text-sm text-text-primary">
                        <input
                          type="checkbox"
                          checked={coderAutoApprove}
                          onChange={(e) => setCoderAutoApprove(e.target.checked)}
                          disabled={coderRunning}
                          className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                        />
                        Auto-approve destructive actions
                      </label>
                      <div className="flex items-center gap-2">
                        <label className="text-xs text-text-secondary">Max iterations</label>
                        <input
                          type="number"
                          min={1}
                          max={200}
                          value={coderMaxIters}
                          onChange={(e) => setCoderMaxIters(e.target.value === "" ? "" : parseInt(e.target.value, 10))}
                          disabled={coderRunning}
                          placeholder="default"
                          className="input w-24 px-2 py-1 text-xs"
                        />
                      </div>
                    </div>
                    <button
                      onClick={handleCoderRun}
                      disabled={!isConnected || coderRunning || !coderTask.trim()}
                      className="btn-primary px-4 py-1.5 text-xs"
                    >
                      {coderRunning ? "Coding…" : "▶ Run coder"}
                    </button>
                    {coderRunning && (
                      <div className="flex items-center gap-2 text-xs text-text-secondary">
                        <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                        Muninn is coding…
                      </div>
                    )}
                    {coderError && (
                      <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">
                        {coderError}
                      </div>
                    )}
                  </div>

                  {coderResult && (
                    <div className="card space-y-3">
                      <h3 className="text-sm font-semibold">Result</h3>
                      <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-sm whitespace-pre-wrap">
                        {coderResult}
                      </div>
                    </div>
                  )}
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
            <div className="flex h-full flex-col bg-bg-tertiary text-text-primary">
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
                  className="input flex-1 bg-bg-tertiary font-mono text-sm border-border text-text-primary"
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
                </div>

                <div className="flex flex-1 overflow-hidden">
                  {activeCp && diffs.length > 0 ? (
                    <DiffViewer
                      diffs={diffs}
                      onAcceptAll={() => acceptCheckpoint(activeCp)}
                      onRejectAll={() => rejectCheckpoint(activeCp)}
                    />
                  ) : activeCp ? (
                    <div className="flex h-full items-center justify-center text-sm text-text-muted">
                      No changes in this checkpoint
                    </div>
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-text-muted">
                      Select a checkpoint to review changes
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === "knowledge" && (
            <div className="flex h-full flex-col">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
                <span className="text-sm font-semibold">Knowledge Base</span>
                <label className="flex cursor-pointer items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.rag_enabled}
                    onChange={(e) => {
                      const next = { ...config, rag_enabled: e.target.checked };
                      setConfig(next);
                      saveConfig(next);
                    }}
                    className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                  />
                  <span className="text-xs text-text-secondary">Use RAG in chat</span>
                </label>
              </div>
              <div className="flex flex-1 overflow-hidden">
                {/* Upload / docs */}
                <aside className="flex w-80 flex-col border-r border-border bg-bg-secondary p-4">
                  <div className="mb-4 rounded-lg border border-dashed border-border bg-bg-tertiary p-4 text-center">
                    <input
                      ref={fileInputRef}
                      type="file"
                      className="hidden"
                      accept=".txt,.md,.pdf,.py,.json,.yaml,.yml"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) uploadKnowledge(file);
                        if (fileInputRef.current) fileInputRef.current.value = "";
                      }}
                    />
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="btn-primary w-full text-xs"
                    >
                      📤 Upload document
                    </button>
                    <p className="mt-2 text-xs text-text-muted">
                      Supports TXT, MD, PDF, code files
                    </p>
                  </div>

                  {kbMsg && (
                    <div className="mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
                      {kbMsg}
                    </div>
                  )}

                  <div className="flex-1 overflow-y-auto">
                    <div className="mb-2 text-xs font-medium text-text-secondary">
                      Documents ({kbDocs.length})
                    </div>
                    {!kbAvailable && (
                      <div className="text-xs text-text-muted">
                        Knowledge base backend is not available. Install chromadb and sentence-transformers.
                      </div>
                    )}
                    {kbDocs.map((doc) => (
                      <div
                        key={doc.doc_id}
                        className="mb-2 flex items-center justify-between rounded-lg border border-border bg-bg-tertiary p-2"
                      >
                        <span className="truncate text-xs text-text-primary">{doc.filename}</span>
                        <button
                          onClick={() => deleteKnowledge(doc.doc_id)}
                          className="text-xs text-error hover:underline"
                        >
                          Delete
                        </button>
                      </div>
                    ))}
                  </div>
                </aside>

                {/* Query tester */}
                <div className="flex flex-1 flex-col bg-bg-primary p-4">
                  <h3 className="mb-3 text-sm font-semibold">Test retrieval</h3>
                  <div className="mb-4 flex gap-2">
                    <input
                      type="text"
                      value={kbQuery}
                      onChange={(e) => setKbQuery(e.target.value)}
                      placeholder="Ask a question against the knowledge base…"
                      className="input flex-1"
                      onKeyDown={(e) => e.key === "Enter" && queryKnowledge()}
                    />
                    <button onClick={queryKnowledge} className="btn-primary">
                      Search
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto space-y-3">
                    {kbChunks.map((chunk, i) => (
                      <div key={i} className="rounded-lg border border-border bg-bg-secondary p-3">
                        <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                          <span>{chunk.metadata?.filename}</span>
                          <span>distance: {chunk.distance?.toFixed(3)}</span>
                        </div>
                        <p className="text-xs text-text-primary whitespace-pre-wrap">
                          {chunk.text}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "logs" && (
            <div className="flex h-full flex-col bg-bg-tertiary text-text-primary">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-4">
                <span className="text-sm font-semibold">Backend Logs</span>
                <div className="flex items-center gap-2">
                  <div className="flex rounded-lg border border-border bg-bg-tertiary p-0.5 text-xs">
                    {(["all", "stdout", "stderr"] as const).map((f) => (
                      <button
                        key={f}
                        onClick={() => setLogFilter(f)}
                        className={`rounded px-2.5 py-1 capitalize ${
                          logFilter === f
                            ? "bg-accent text-white"
                            : "text-text-secondary hover:text-text-primary"
                        }`}
                      >
                        {f}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() =>
                      navigator.clipboard.writeText(
                        backendLogs.map((l) => `[${l.time}][${l.source}] ${l.text}`).join("")
                      )
                    }
                    className="btn-secondary px-3 py-1.5 text-xs"
                  >
                    Copy
                  </button>
                  <button
                    onClick={() => setBackendLogs([])}
                    className="btn-secondary px-3 py-1.5 text-xs"
                  >
                    Clear
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-3 font-mono text-sm">
                {backendLogs
                  .filter((l) => logFilter === "all" || l.source === logFilter)
                  .map((l, i) => (
                    <div
                      key={i}
                      className={`whitespace-pre-wrap break-all ${
                        l.source === "stderr" ? "text-error" : "text-text-primary"
                      }`}
                    >
                      <span className="text-text-muted">[{l.time}]</span>{" "}
                      {l.text}
                    </div>
                  ))}
                <div ref={backendLogEndRef} />
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
                tools.length === 0 ? (
                  !isConnected ? (
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <Wrench size={40} className="text-text-muted opacity-40" />
                      <p className="mt-4 text-sm font-medium text-text-secondary">Backend not connected</p>
                      <p className="mt-1 max-w-xs text-xs text-text-muted">
                        Tools are loaded from the AI backend. Make sure the server is running and reconnect to see available tools.
                      </p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                      <p className="mt-4 text-sm text-text-muted">Loading tools…</p>
                    </div>
                  )
                ) : (
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
                      <div className="flex items-center justify-between">
                        <div className="text-xs font-semibold uppercase text-accent">Tool</div>
                        {tool.destructive && (
                          <span className="rounded bg-error/10 px-1.5 py-0.5 text-[10px] text-error">
                            destructive
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-sm font-semibold">{tool.function.name}</div>
                      <div className="mt-1 text-xs text-text-secondary line-clamp-2">
                        {tool.function.description}
                      </div>
                    </button>
                  ))}
                </div>
                )
              ) : (
                <div className="max-w-3xl space-y-4">
                  <div className="card">
                    <div className="flex items-center justify-between">
                      <div className="text-xs uppercase text-accent font-semibold">Tool</div>
                      {selectedTool.destructive && (
                        <span className="rounded bg-error/10 px-1.5 py-0.5 text-[10px] text-error">
                          destructive
                        </span>
                      )}
                    </div>
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
                    {toolLoading
                      ? "Running…"
                      : selectedTool.destructive
                      ? "⚠️ Run Tool"
                      : "Run Tool"}
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
                skills.length === 0 ? (
                  !isConnected ? (
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <Sparkles size={40} className="text-text-muted opacity-40" />
                      <p className="mt-4 text-sm font-medium text-text-secondary">Backend not connected</p>
                      <p className="mt-1 max-w-xs text-xs text-text-muted">
                        Skills are loaded from the AI backend. Make sure the server is running and reconnect to see available skills.
                      </p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                      <p className="mt-4 text-sm text-text-muted">Loading skills…</p>
                    </div>
                  )
                ) : (
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
                )
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

          {activeTab === "emotion" && (
            <EmotionTrackerPanel apiBase={API_BASE} />
          )}

          {activeTab === "memory" && (
            <div className="flex h-full flex-col">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
                <span className="text-sm font-semibold">Memory</span>
                <div className="flex items-center gap-2">
                  <button onClick={syncMemoryMd} className="btn-secondary px-3 py-1.5 text-xs">Sync MEMORY.md</button>
                  <button onClick={pruneMemory} className="btn-secondary px-3 py-1.5 text-xs">Prune</button>
                  <button onClick={loadMemory} className="btn-secondary px-3 py-1.5 text-xs">Refresh</button>
                </div>
              </div>
              <div className="flex flex-1 overflow-hidden">
                <div className="w-80 overflow-y-auto border-r border-border bg-bg-secondary p-4">
                  <div className="card mb-4">
                    <h3 className="text-sm font-semibold">Stats</h3>
                    <div className="mt-2 space-y-1 text-xs">
                      <div className="flex justify-between"><span className="text-text-muted">Total</span><span>{memoryStats?.longterm_entries ?? "—"}</span></div>
                      <div className="flex justify-between"><span className="text-text-muted">Short</span><span>{memoryStats?.tier_counts?.short ?? 0}</span></div>
                      <div className="flex justify-between"><span className="text-text-muted">Mid</span><span>{memoryStats?.tier_counts?.mid ?? 0}</span></div>
                      <div className="flex justify-between"><span className="text-text-muted">Long</span><span>{memoryStats?.tier_counts?.long ?? 0}</span></div>
                    </div>
                  </div>
                  <div className="card mb-4">
                    <button
                      onClick={() => setMemoryView(memoryView === "add" ? "browse" : "add")}
                      className="flex w-full items-center justify-between text-left"
                    >
                      <h3 className="text-sm font-semibold">
                        {memoryView === "add" ? "Add Memory" : "+ Add memory"}
                      </h3>
                      <ChevronDown size={14} className={`text-text-muted transition-transform duration-150 ${memoryView === "add" ? "rotate-0" : "-rotate-90"}`} />
                    </button>
                    {memoryView === "add" && (
                      <div className="mt-3">
                        <textarea
                          className="input-field mb-2 min-h-[80px] text-xs"
                          placeholder="Content..."
                          value={memoryForm.content}
                          onChange={(e) => setMemoryForm({ ...memoryForm, content: e.target.value })}
                        />
                        <div className="mb-2 grid grid-cols-2 gap-2">
                          <select className="input-field text-xs" value={memoryForm.category} onChange={(e) => setMemoryForm({ ...memoryForm, category: e.target.value })}>
                            <option value="fact">fact</option>
                            <option value="insight">insight</option>
                            <option value="conversation">conversation</option>
                            <option value="calculation">calculation</option>
                            <option value="error">error</option>
                            <option value="episode">episode</option>
                          </select>
                          <select className="input-field text-xs" value={memoryForm.tier} onChange={(e) => setMemoryForm({ ...memoryForm, tier: e.target.value })}>
                            <option value="short">short (6h)</option>
                            <option value="mid">mid (7d)</option>
                            <option value="long">long (perm)</option>
                          </select>
                        </div>
                        <input
                          className="input-field mb-2 text-xs"
                          placeholder="tags, comma separated"
                          value={memoryForm.tags}
                          onChange={(e) => setMemoryForm({ ...memoryForm, tags: e.target.value })}
                        />
                        <div className="mb-2 flex items-center gap-2 text-xs">
                          <span className="text-text-muted">Importance</span>
                          <input type="range" min={0} max={1} step={0.05} value={memoryForm.importance} onChange={(e) => setMemoryForm({ ...memoryForm, importance: parseFloat(e.target.value) })} />
                          <span>{memoryForm.importance.toFixed(2)}</span>
                        </div>
                        <button onClick={createMemory} className="btn-primary w-full py-1.5 text-xs" disabled={!memoryForm.content.trim()}>
                          Remember
                        </button>
                      </div>
                    )}
                  </div>
                  {memoryMsg && <p className="text-xs text-text-secondary">{memoryMsg}</p>}
                </div>
                <div className="flex flex-1 flex-col overflow-hidden bg-bg-primary p-4">
                  <div className="mb-3 flex items-center gap-2">
                    <input
                      className="input-field flex-1 text-xs"
                      placeholder="Search memory..."
                      value={memorySearch}
                      onChange={(e) => setMemorySearch(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && searchMemory()}
                    />
                    <button onClick={searchMemory} className="btn-primary px-3 py-1.5 text-xs">Search</button>
                    <select className="input-field text-xs" value={memoryFilter.category} onChange={(e) => setMemoryFilter({ ...memoryFilter, category: e.target.value })}>
                      <option value="">all categories</option>
                      <option value="fact">fact</option>
                      <option value="insight">insight</option>
                      <option value="conversation">conversation</option>
                      <option value="calculation">calculation</option>
                      <option value="error">error</option>
                      <option value="episode">episode</option>
                    </select>
                    <select className="input-field text-xs" value={memoryFilter.tier} onChange={(e) => setMemoryFilter({ ...memoryFilter, tier: e.target.value })}>
                      <option value="">all tiers</option>
                      <option value="short">short</option>
                      <option value="mid">mid</option>
                      <option value="long">long</option>
                    </select>
                  </div>
                  <div className="flex-1 overflow-y-auto space-y-2">
                    {memories.length === 0 && (
                      <div className="flex flex-col items-center justify-center py-16 text-center">
                        <Brain size={36} className="text-text-muted opacity-40" />
                        {memorySearch.trim() ? (
                          <>
                            <p className="mt-3 text-sm font-medium text-text-secondary">No matching memories</p>
                            <p className="mt-1 max-w-xs text-xs text-text-muted">
                              No memories match your search. Try different keywords or clear the filter.
                            </p>
                          </>
                        ) : (
                          <>
                            <p className="mt-3 text-sm font-medium text-text-secondary">No memories yet</p>
                            <p className="mt-1 max-w-xs text-xs text-text-muted">
                              Memories help Huginn remember context across conversations. Add your first memory using the panel on the left.
                            </p>
                          </>
                        )}
                      </div>
                    )}
                    {memories.map((m) => (
                      <div key={m.id} className="card">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 text-xs">
                              <span className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono">{m.tier}</span>
                              <span className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono">{m.category}</span>
                              <span className="text-text-muted">importance {m.importance}</span>
                            </div>
                            <p className="mt-1 whitespace-pre-wrap text-sm">{m.content}</p>
                            <p className="mt-1 text-xs text-text-muted">tags: {m.tags || "—"} · source: {m.source || "—"}</p>
                            <p className="text-xs text-text-muted">expires: {m.expires_at ? new Date(m.expires_at).toLocaleString() : "never"} · accessed {m.access_count ?? 0}</p>
                          </div>
                          <div className="flex flex-col gap-1">
                            {m.tier !== "long" && (
                              <button onClick={() => promoteMemory(m.id)} className="btn-secondary px-2 py-1 text-xs" title="Promote to long">
                                ⬆
                              </button>
                            )}
                            <button onClick={() => deleteMemory(m.id)} className="btn-secondary px-2 py-1 text-xs" title="Delete">
                              🗑
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "plugins" && (
            <div className="flex h-full flex-col">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
                <span className="text-sm font-semibold">Plugins / MCP Servers</span>
                <button onClick={() => { loadMcp(); discoverMcp(); }} className="btn-secondary px-3 py-1.5 text-xs">
                  Refresh
                </button>
              </div>
              <div className="flex flex-1 overflow-hidden">
                <aside className="flex w-80 flex-col border-r border-border bg-bg-secondary p-4">
                  <h3 className="mb-3 text-sm font-semibold">Connect manually</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="mb-1 block text-xs text-text-secondary">Name</label>
                      <input
                        type="text"
                        value={newMcp.name}
                        onChange={(e) => setNewMcp({ ...newMcp, name: e.target.value })}
                        placeholder="my-server"
                        className="input text-sm"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs text-text-secondary">Command</label>
                      <input
                        type="text"
                        value={newMcp.command}
                        onChange={(e) => setNewMcp({ ...newMcp, command: e.target.value })}
                        placeholder="python"
                        className="input text-sm"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs text-text-secondary">Args (space separated)</label>
                      <input
                        type="text"
                        value={newMcp.args}
                        onChange={(e) => setNewMcp({ ...newMcp, args: e.target.value })}
                        placeholder="server.py"
                        className="input text-sm"
                      />
                    </div>
                    <button
                      onClick={() => {
                        if (!newMcp.name.trim()) return;
                        const args = newMcp.args
                          .split(" ")
                          .map((s) => s.trim())
                          .filter(Boolean);
                        connectMcp({ name: newMcp.name.trim(), command: newMcp.command.trim() || "python", args });
                        setNewMcp({ name: "", command: "python", args: "" });
                      }}
                      className="btn-primary w-full text-xs"
                    >
                      Connect
                    </button>
                  </div>

                  {mcpMsg && (
                    <div className="mt-4 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
                      {mcpMsg}
                    </div>
                  )}

                  <h3 className="mb-2 mt-6 text-sm font-semibold">Discovered local</h3>
                  <div className="flex-1 overflow-y-auto space-y-2">
                    {discoveredServers.length === 0 && (
                      <div className="flex flex-col items-start rounded-lg border border-dashed border-border p-3">
                        <p className="text-xs font-medium text-text-secondary">No local servers found</p>
                        <p className="mt-1 text-[11px] leading-relaxed text-text-muted">
                          Start an MCP server on your network, or use the form above to connect manually by name and command.
                        </p>
                      </div>
                    )}
                    {discoveredServers.map((srv) => (
                      <div
                        key={srv.name}
                        className="rounded-lg border border-border bg-bg-tertiary p-2"
                      >
                        <div className="text-xs font-medium text-text-primary">{srv.name}</div>
                        <div className="mt-1 truncate text-[10px] text-text-muted">{srv.path}</div>
                        <button
                          onClick={() => connectMcp(srv)}
                          className="mt-2 w-full rounded bg-accent px-2 py-1 text-xs text-white hover:bg-accent/90"
                        >
                          Connect
                        </button>
                      </div>
                    ))}
                  </div>
                </aside>

                <div className="flex flex-1 flex-col bg-bg-primary p-4">
                  <h3 className="mb-3 text-sm font-semibold">Connected servers</h3>
                  <div className="flex-1 overflow-y-auto space-y-3">
                    {mcpServers.length === 0 && (
                      <div className="flex flex-col items-center justify-center py-16 text-center">
                        <Puzzle size={36} className="text-text-muted opacity-40" />
                        <p className="mt-3 text-sm font-medium text-text-secondary">No plugins connected</p>
                        <p className="mt-1 max-w-xs text-xs text-text-muted">
                          Connect an MCP server from the sidebar to extend Huginn with additional tools and capabilities.
                        </p>
                      </div>
                    )}
                    {mcpServers.map((srv) => (
                      <div key={srv.name} className="rounded-xl border border-border bg-bg-secondary p-4">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-semibold text-text-primary">{srv.name}</div>
                          <button
                            onClick={() => disconnectMcp(srv.name)}
                            className="btn-secondary px-2 py-1 text-xs text-error hover:bg-error/10"
                          >
                            Disconnect
                          </button>
                        </div>
                        <div className="mt-2 text-xs text-text-secondary">
                          {srv.tools.length} tool{srv.tools.length === 1 ? "" : "s"}
                        </div>
                        <div className="mt-2 space-y-1">
                          {srv.tools.map((t) => (
                            <div
                              key={t.name}
                              className="rounded bg-bg-tertiary px-2 py-1 text-xs text-text-primary"
                            >
                              <span className="font-mono text-accent">{t.name}</span>
                              <span className="ml-2 text-text-muted">{t.description}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "project" && (
            <div className="flex h-full flex-col">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
                <span className="text-sm font-semibold">Project Context & Codebase</span>
                <button onClick={loadProjectContext} className="btn-secondary px-3 py-1.5 text-xs">
                  Refresh
                </button>
              </div>
              <div className="flex flex-1 overflow-hidden">
                {/* Project context editor */}
                <aside className="flex w-1/2 flex-col border-r border-border bg-bg-secondary p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold">Project Instructions</h3>
                      <p className="text-[10px] text-text-muted">
                        Loaded from: <span className="text-text-secondary">{projectContextSource}</span>
                      </p>
                    </div>
                    <button onClick={saveProjectContext} className="btn-primary px-3 py-1.5 text-xs">
                      Save
                    </button>
                  </div>
                  {projectContextMsg && (
                    <div className="mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
                      {projectContextMsg}
                    </div>
                  )}
                  <textarea
                    value={projectContext}
                    onChange={(e) => setProjectContext(e.target.value)}
                    placeholder="Write project-level instructions here (coding style, conventions, important formulas, DFT preferences...). Saved to .huginn.md in the workspace."
                    className="input flex-1 resize-none font-mono text-sm"
                    spellCheck={false}
                  />
                </aside>

                {/* Codebase semantic search */}
                <div className="flex w-1/2 flex-col bg-bg-primary p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold">Codebase Search</h3>
                      <p className="text-[10px] text-text-muted">
                        {codebaseStatus?.available
                          ? `${codebaseStatus.indexed_files || 0} files indexed`
                          : "Not indexed"}
                      </p>
                    </div>
                    <button onClick={indexCodebase} className="btn-primary px-3 py-1.5 text-xs">
                      Re-index
                    </button>
                  </div>
                  <div className="mb-3 flex gap-2">
                    <input
                      type="text"
                      value={codebaseQuery}
                      onChange={(e) => setCodebaseQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && searchCodebase()}
                      placeholder="Search the codebase semantically…"
                      className="input flex-1 text-sm"
                    />
                    <button onClick={searchCodebase} className="btn-primary text-xs">
                      Search
                    </button>
                  </div>
                  {codebaseMsg && (
                    <div className="mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
                      {codebaseMsg}
                    </div>
                  )}
                  <div className="flex-1 overflow-y-auto space-y-3">
                    {codebaseResults.map((r, i) => (
                      <div key={i} className="rounded-lg border border-border bg-bg-secondary p-3">
                        <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                          <span className="font-mono">{r.path}</span>
                          <span>chunk {r.chunk}</span>
                        </div>
                        <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded bg-bg-tertiary p-2 text-xs text-text-primary">
                          {r.text}
                        </pre>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "threads" && (
            <div className="flex h-full flex-col">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
                <span className="text-sm font-semibold">Threads</span>
                <button onClick={createThread} className="btn-primary px-3 py-1.5 text-xs">
                  + New thread
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4">
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {threads.map((t) => (
                    <div
                      key={t.id}
                      className={`rounded-xl border p-4 transition-colors ${
                        activeThread === t.id
                          ? "border-accent bg-accent/10"
                          : "border-border bg-bg-secondary hover:bg-bg-tertiary"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <input
                          value={t.label}
                          onChange={(e) => {
                            const next = threads.map((th) =>
                              th.id === t.id ? { ...th, label: e.target.value } : th
                            );
                            setThreads(next);
                          }}
                          onBlur={(e) => renameThread(t.id, e.target.value)}
                          className="w-full bg-transparent text-sm font-semibold text-text-primary focus:outline-none"
                        />
                        <button
                          onClick={() => deleteThread(t.id)}
                          className="text-xs text-error hover:underline"
                        >
                          Delete
                        </button>
                      </div>
                      <div className="mt-2 text-[10px] text-text-muted">ID: {t.id}</div>
                      <button
                        onClick={() => {
                          setActiveThread(t.id);
                          setMessages([
                            {
                              role: "assistant",
                              content: `Switched to thread **${t.label}**.`,
                              timestamp: formatTime(),
                            },
                          ]);
                        }}
                        disabled={activeThread === t.id}
                        className="mt-3 w-full rounded-lg border border-border bg-bg-tertiary py-1.5 text-xs text-text-secondary hover:text-text-primary disabled:opacity-50"
                      >
                        {activeThread === t.id ? "Active" : "Switch"}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "settings" && (
            <div className="flex h-full flex-col">
              <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
                <span className="text-sm font-semibold">Settings</span>
                <div className="flex items-center gap-2">
                  {(["general", "models", "agents", "privacy", "pet", "security", "credentials"] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setSettingsTab(t)}
                      className={`rounded px-3 py-1 text-xs capitalize ${
                        settingsTab === t
                          ? "bg-accent text-white"
                          : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-6">
                {settingsTab === "general" && (
                  <div className="max-w-2xl space-y-5">
                    <p className="text-sm text-text-secondary">
                      Default single-model settings. For multi-LLM mode, switch to the Models tab.
                    </p>
                    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                      <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Provider</label>
                        <select
                          value={config.provider}
                          onChange={(e) => { const next = { ...config, provider: e.target.value }; setConfig(next); setConfigDirty(true); }}
                          className="input"
                        >
                          {PROVIDERS.map((p) => (
                            <option key={p.id} value={p.id}>{p.label}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Model</label>
                        <input
                          type="text"
                          value={config.model}
                          onChange={(e) => { setConfig({ ...config, model: e.target.value }); setConfigDirty(true); }}
                          placeholder="e.g. gpt-4o"
                          className="input"
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Persona</label>
                        <select
                          value={config.persona}
                          onChange={(e) => { const next = { ...config, persona: e.target.value }; setConfig(next); setConfigDirty(true); }}
                          className="input"
                        >
                          {PERSONAS.map((p) => (
                            <option key={p.id} value={p.id}>{p.label}</option>
                          ))}
                        </select>
                      </div>
                      <div className="md:col-span-2">
                        <label className="flex cursor-pointer items-center gap-2">
                          <input
                            type="checkbox"
                            checked={config.rag_enabled}
                            onChange={(e) => { const next = { ...config, rag_enabled: e.target.checked }; setConfig(next); setConfigDirty(true); }}
                            className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                          />
                          <span className="text-sm text-text-primary">Use knowledge base (RAG) in chat</span>
                        </label>
                      </div>
                      <div className="md:col-span-2">
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">API Key</label>
                        <input
                          type="password"
                          value={config.api_key}
                          onChange={(e) => { setConfig({ ...config, api_key: e.target.value }); setConfigDirty(true); }}
                          placeholder={PROVIDERS.find((p) => p.id === config.provider)?.keyVar || "API key"}
                          className="input"
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Base URL (optional)</label>
                        <input
                          type="text"
                          value={config.base_url}
                          onChange={(e) => { setConfig({ ...config, base_url: e.target.value }); setConfigDirty(true); }}
                          placeholder="https://api.openai.com/v1"
                          className="input"
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Ollama Host</label>
                        <input
                          type="text"
                          value={config.ollama_host}
                          onChange={(e) => { setConfig({ ...config, ollama_host: e.target.value }); setConfigDirty(true); }}
                          placeholder="http://localhost:11434"
                          className="input"
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Max concurrent sub-agents</label>
                        <input
                          type="number"
                          min={1}
                          max={10}
                          value={config.max_concurrent_subagents}
                          onChange={(e) => { const next = { ...config, max_concurrent_subagents: parseInt(e.target.value || "1", 10) }; setConfig(next); setConfigDirty(true); }}
                          className="input"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {settingsTab === "models" && (
                  <div className="max-w-3xl space-y-4">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-text-secondary">Configure multiple provider/model entries.</p>
                      <button onClick={addModel} className="btn-secondary px-3 py-1.5 text-xs">+ Add Model</button>
                    </div>
                    {config.models.length === 0 && (
                      <p className="text-sm text-text-muted">No model pool yet. Add a model or use the General tab for a single provider.</p>
                    )}
                    {config.models.map((m, i) => (
                      <div key={i} className="card">
                        <div className="flex items-center justify-between">
                          <button
                            onClick={() => toggleModelExpanded(i)}
                            className="flex flex-1 items-center gap-2 text-left min-w-0"
                          >
                            <ChevronDown size={14} className={`flex-shrink-0 text-text-muted transition-transform duration-150 ${expandedModels.has(i) ? "rotate-0" : "-rotate-90"}`} />
                            <input
                              className="input-field w-32 text-sm font-semibold"
                              value={m.alias}
                              onChange={(e) => updateModel(i, { alias: e.target.value })}
                              placeholder="alias"
                              onClick={(e) => e.stopPropagation()}
                            />
                            {!expandedModels.has(i) && (
                              <span className="text-xs text-text-muted truncate">{m.provider} / {m.model || "—"}</span>
                            )}
                          </button>
                          <div className="flex items-center gap-2">
                            <label className="flex items-center gap-1 text-xs">
                              <input type="checkbox" checked={m.enabled} onChange={(e) => updateModel(i, { enabled: e.target.checked })} />
                              Enabled
                            </label>
                            <button onClick={() => removeModel(i)} className="btn-secondary px-2 py-1 text-xs">🗑</button>
                          </div>
                        </div>
                        {expandedModels.has(i) && (
                          <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                            <select
                              className="input-field text-xs"
                              value={m.provider}
                              onChange={(e) => updateModel(i, { provider: e.target.value })}
                            >
                              {PROVIDERS.map((p) => (
                                <option key={p.id} value={p.id}>{p.label}</option>
                              ))}
                            </select>
                            <input className="input-field text-xs" value={m.model} onChange={(e) => updateModel(i, { model: e.target.value })} placeholder="model name" />
                            <input className="input-field text-xs" type="password" value={m.api_key} onChange={(e) => updateModel(i, { api_key: e.target.value })} placeholder="API key (optional)" />
                            <input className="input-field text-xs" value={m.base_url} onChange={(e) => updateModel(i, { base_url: e.target.value })} placeholder="base URL (optional)" />
                            <div className="flex items-center gap-2 text-xs">
                              <span className="text-text-muted">Temp</span>
                              <input type="range" min={0} max={2} step={0.05} value={m.temperature} onChange={(e) => updateModel(i, { temperature: parseFloat(e.target.value) })} />
                              <span>{m.temperature.toFixed(2)}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {settingsTab === "agents" && (
                  <div className="max-w-3xl space-y-4">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-text-secondary">Agent profiles used by Team mode and @ routing.</p>
                      <button onClick={addAgent} className="btn-secondary px-3 py-1.5 text-xs">+ Add Agent</button>
                    </div>
                    {config.agents.length === 0 && (
                      <p className="text-sm text-text-muted">No agent profiles yet. Add one or use the General tab for a single agent.</p>
                    )}
                    {config.agents.map((a, i) => (
                      <div key={i} className="card">
                        <div className="flex items-center justify-between gap-2">
                          <button
                            onClick={() => toggleAgentExpanded(i)}
                            className="flex flex-1 items-center gap-2 text-left min-w-0"
                          >
                            <ChevronDown size={14} className={`flex-shrink-0 text-text-muted transition-transform duration-150 ${expandedAgents.has(i) ? "rotate-0" : "-rotate-90"}`} />
                            <input
                              className="input-field w-28 text-sm font-semibold"
                              value={a.id}
                              onChange={(e) => updateAgent(i, { id: e.target.value })}
                              placeholder="id"
                              onClick={(e) => e.stopPropagation()}
                            />
                            <input
                              className="input-field flex-1 text-sm"
                              value={a.name}
                              onChange={(e) => updateAgent(i, { name: e.target.value })}
                              placeholder="display name"
                              onClick={(e) => e.stopPropagation()}
                            />
                            {!expandedAgents.has(i) && (
                              <span className="text-xs text-text-muted truncate">{a.model_alias || "default"} · {a.persona || "—"}</span>
                            )}
                          </button>
                          <div className="flex items-center gap-2">
                            <label className="flex items-center gap-1 text-xs">
                              <input type="checkbox" checked={a.enabled} onChange={(e) => updateAgent(i, { enabled: e.target.checked })} />
                              Enabled
                            </label>
                            <button onClick={() => removeAgent(i)} className="btn-secondary px-2 py-1 text-xs">🗑</button>
                          </div>
                        </div>
                        {expandedAgents.has(i) && (
                          <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                            <select
                              className="input-field text-xs"
                              value={a.model_alias}
                              onChange={(e) => updateAgent(i, { model_alias: e.target.value })}
                            >
                              <option value="">default model</option>
                              {config.models.filter((m) => m.enabled).map((m) => (
                                <option key={m.alias} value={m.alias}>{m.alias} ({m.provider})</option>
                              ))}
                            </select>
                            <select
                              className="input-field text-xs"
                              value={a.persona}
                              onChange={(e) => updateAgent(i, { persona: e.target.value })}
                            >
                              {PERSONAS.map((p) => (
                                <option key={p.id} value={p.id}>{p.label}</option>
                              ))}
                            </select>
                            <input
                              className="input-field text-xs md:col-span-2"
                              value={(a.tools || []).join(", ")}
                              onChange={(e) => updateAgent(i, { tools: e.target.value.split(",").map((t) => t.trim()).filter(Boolean) })}
                              placeholder="tool allowlist, comma separated (empty = all)"
                            />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {settingsTab === "privacy" && (
                  <div className="max-w-2xl space-y-5">
                    <p className="text-sm text-text-secondary">
                      Controls what local data can leave your machine when using a cloud LLM provider.
                    </p>
                    <div className="space-y-4">
                      <label className="flex cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={config.local_only_mode}
                          onChange={(e) => { const next = { ...config, local_only_mode: e.target.checked }; setConfig(next); setConfigDirty(true); }}
                          className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                        />
                        <span className="text-sm text-text-primary">Local-only / no-cloud mode (allow only Ollama, vLLM, local loopback endpoints)</span>
                      </label>
                      <label className="flex cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={config.privacy_redact_secrets}
                          onChange={(e) => { const next = { ...config, privacy_redact_secrets: e.target.checked }; setConfig(next); setConfigDirty(true); }}
                          className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                        />
                        <span className="text-sm text-text-primary">Redact secrets (API keys, private keys, tokens) before sending to LLM</span>
                      </label>
                      <label className="flex cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={config.privacy_block_on_secrets}
                          onChange={(e) => { const next = { ...config, privacy_block_on_secrets: e.target.checked }; setConfig(next); setConfigDirty(true); }}
                          className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                        />
                        <span className="text-sm text-text-primary">Block messages that contain detected secrets</span>
                      </label>
                      <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Max tool output tokens</label>
                        <input
                          type="number"
                          min={0}
                          value={config.max_tool_output_tokens}
                          onChange={(e) => { const next = { ...config, max_tool_output_tokens: parseInt(e.target.value || "0", 10) }; setConfig(next); setConfigDirty(true); }}
                          placeholder="0 = unlimited"
                          className="input"
                        />
                        <p className="mt-1 text-xs text-text-muted">Tool results longer than this are truncated before being sent to the LLM.</p>
                      </div>
                      <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Context budget tokens</label>
                        <input
                          type="number"
                          min={0}
                          value={config.context_budget_tokens}
                          onChange={(e) => { const next = { ...config, context_budget_tokens: parseInt(e.target.value || "0", 10) }; setConfig(next); setConfigDirty(true); }}
                          placeholder="0 = unlimited"
                          className="input"
                        />
                        <p className="mt-1 text-xs text-text-muted">Warn when the estimated prompt tokens exceed this budget.</p>
                      </div>
                    </div>
                  </div>
                )}

                {settingsTab === "pet" && (
                  <div className="max-w-2xl space-y-5">
                    <p className="text-sm text-text-secondary">
                      Customize your desktop companion.
                    </p>
                    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                      <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Pet name</label>
                        <input
                          type="text"
                          value={config.pet_name}
                          onChange={(e) => { const next = { ...config, pet_name: e.target.value }; setConfig(next); setConfigDirty(true); }}
                          placeholder="Muninn"
                          className="input"
                        />
                      </div>
                      <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">Personality</label>
                        <select
                          value={config.pet_personality}
                          onChange={(e) => { const next = { ...config, pet_personality: e.target.value as any }; setConfig(next); setConfigDirty(true); }}
                          className="input"
                        >
                          <option value="cheerful">Cheerful</option>
                          <option value="nerdy">Nerdy</option>
                          <option value="calm">Calm</option>
                          <option value="sassy">Sassy</option>
                        </select>
                      </div>
                    </div>

                    {/* Accessories */}
                    <div>
                      <label className="mb-1.5 block text-xs font-medium text-text-secondary">Accessories</label>
                      <div className="flex flex-wrap gap-2">
                        {([
                          { id: "crown", label: "Crown", minLevel: 5 },
                          { id: "glasses", label: "Glasses", minLevel: 3 },
                          { id: "scarf", label: "Scarf", minLevel: 7 },
                        ] as const).map((acc) => {
                          const active = config.pet_accessories.includes(acc.id);
                          return (
                            <button
                              key={acc.id}
                              onClick={() => {
                                const next = {
                                  ...config,
                                  pet_accessories: active
                                    ? config.pet_accessories.filter(a => a !== acc.id)
                                    : [...config.pet_accessories, acc.id],
                                };
                                setConfig(next);
                                setConfigDirty(true);
                              }}
                              className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                                active
                                  ? "border-accent/40 bg-accent/10 text-text-primary"
                                  : "border-border bg-bg-tertiary text-text-secondary hover:text-text-primary"
                              }`}
                            >
                              {acc.label}
                              <span className="text-text-muted text-[10px]">(Lv.{acc.minLevel}+)</span>
                            </button>
                          );
                        })}
                      </div>
                      <p className="mt-1 text-[10px] text-text-muted">
                        Accessories unlock as your pet levels up through completed tasks.
                      </p>
                    </div>

                    {/* Status overview (read-only from backend) */}
                    <div className="rounded-xl border border-border bg-bg-tertiary p-4">
                      <p className="mb-2 text-xs font-medium text-text-secondary">Pet vitals are managed by the backend.</p>
                      <p className="text-[11px] text-text-muted leading-relaxed">
                        Your raven gains XP when agent tasks succeed. Hunger and mood decay slowly over time.
                        Feed and pet your companion via the right-click menu on the pet window.
                      </p>
                    </div>

                    {/* Reset */}
                    <div className="flex items-center gap-3">
                      <button
                        onClick={async () => {
                          try {
                            await fetch(`${API_BASE}/pet/reset`, { method: "POST" });
                          } catch {
                            // backend may not have this endpoint yet
                          }
                        }}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-error/30 bg-error/5 px-3 py-1.5 text-xs font-medium text-error/80 transition-colors hover:bg-error/10 hover:text-error"
                      >
                        Reset pet progress
                      </button>
                      <span className="text-[10px] text-text-muted">
                        Resets level, XP, hunger, mood, and accessories.
                      </span>
                    </div>

                    <p className="text-xs text-text-muted">
                      The pet's greeting, idle tips, and click responses will match the chosen personality.
                    </p>
                  </div>
                )}

                {settingsTab === "security" && (
                  <div className="max-w-2xl space-y-5">
                    <p className="text-sm text-text-secondary">
                      Encrypt sensitive configuration files and key material at rest.
                    </p>
                    <label className="flex cursor-pointer items-center gap-2">
                      <input
                        type="checkbox"
                        checked={config.encrypt_config}
                        onChange={(e) => { const next = { ...config, encrypt_config: e.target.checked }; setConfig(next); setConfigDirty(true); }}
                        className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
                      />
                      <span className="text-sm text-text-primary">Encrypt config files</span>
                    </label>
                    <div>
                      <label className="mb-1.5 block text-xs font-medium text-text-secondary">Encryption password</label>
                      <input
                        type="password"
                        value={config.encryption_password}
                        onChange={(e) => { const next = { ...config, encryption_password: e.target.value }; setConfig(next); setConfigDirty(true); }}
                        placeholder="Leave empty to keep unchanged"
                        className="input"
                      />
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-medium text-text-secondary">Key file path (optional)</label>
                      <input
                        type="text"
                        value={config.encryption_key_file}
                        onChange={(e) => { const next = { ...config, encryption_key_file: e.target.value }; setConfig(next); setConfigDirty(true); }}
                        placeholder="Path to encrypted key file"
                        className="input"
                      />
                    </div>
                    <button
                      onClick={async () => {
                        try {
                          const data = await fetch(`${API_BASE}/config/encrypt`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ path: "huginn.toml", password: config.encryption_password }),
                          }).then((r) => r.json());
                          if (data.success) {
                            setConfigSavedMsg(`Encrypted config saved to ${data.path}`);
                            setTimeout(() => setConfigSavedMsg(""), 4000);
                          } else {
                            setConfigSavedMsg(`Encrypt failed: ${data.error}`);
                          }
                        } catch (e: any) {
                          setConfigSavedMsg(`Encrypt error: ${e.message}`);
                        }
                      }}
                      disabled={!config.encryption_password}
                      className="btn-secondary text-xs"
                    >
                      Encrypt huginn.toml now
                    </button>
                  </div>
                )}

                {settingsTab === "credentials" && (
                  <CredentialsPanel />
                )}

                <div className="mt-6 flex items-center gap-3 pt-2">
                  <button onClick={() => saveConfig(config)} disabled={!configDirty} className="btn-primary">
                    Save Settings
                  </button>
                  {configSavedMsg && <span className="text-sm text-success">{configSavedMsg}</span>}
                </div>

                <div className="card mt-6 border-accent/20 bg-accent/5">
                  <h3 className="text-sm font-semibold text-accent">Backend</h3>
                  <p className="mt-1 text-xs text-text-secondary">
                    The desktop app normally starts the Python backend automatically. If it didn't, you can start it here.
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

          {activeTab === "benchmark" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Benchmark</h2>
                  <p className="text-sm text-text-secondary">Run standardized tasks and measure pass rate.</p>
                </div>
                <div className="card space-y-3">
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <input type="checkbox" checked={benchEvolve} onChange={(e) => setBenchEvolve(e.target.checked)} className="h-4 w-4 rounded border-border" />
                    Run evolution cycle afterward
                  </label>
                  <input
                    type="text"
                    value={benchCategories}
                    onChange={(e) => setBenchCategories(e.target.value)}
                    placeholder="Categories, comma separated (empty = all)"
                    className="input text-sm"
                  />
                  <button onClick={handleBenchRun} disabled={benchRunning || !isConnected} className="btn-primary text-xs">
                    {benchRunning ? "Running…" : "▶ Run benchmark"}
                  </button>
                  {benchError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{benchError}</div>}
                </div>
                {benchResult && (
                  <div className="card space-y-3">
                    <h3 className="text-sm font-semibold">Report</h3>
                    <div className="text-xs text-text-secondary">
                      Pass rate: {(benchResult.metrics?.pass_rate * 100).toFixed(0)}% · Total: {benchResult.total} · Passed: {benchResult.passed} · Failed: {benchResult.failed} · Skipped: {benchResult.skipped}
                    </div>
                    <div className="space-y-2">
                      {(benchResult.results || []).map((r: any) => (
                        <div key={r.task_id} className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs">
                          <span className={`font-semibold ${r.passed ? "text-success" : "text-error"}`}>{r.passed ? "✓" : "✗"}</span>{" "}
                          <span className="font-mono">{r.task_id}</span> — {r.reason}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "evolution" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Evolution</h2>
                  <p className="text-sm text-text-secondary">Run a self-evolution cycle over recent execution logs to learn rules and skills.</p>
                </div>
                <div className="card space-y-3">
                  <button onClick={handleEvolveRun} disabled={evolveRunning || !isConnected} className="btn-primary text-xs">
                    {evolveRunning ? "Evolving…" : "▶ Run evolution cycle"}
                  </button>
                  {evolveError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{evolveError}</div>}
                </div>
                {evolveResult && (
                  <div className="card space-y-3">
                    <h3 className="text-sm font-semibold">Report</h3>
                    <div className="text-xs text-text-secondary">
                      Failure rules: {evolveResult.failure_rules?.length} · Success skills: {evolveResult.success_skills?.length} · Prompt patches: {evolveResult.prompt_patches?.length}
                    </div>
                    <div className="text-xs text-text-secondary">Total rules: {evolveResult.total_rules_after} · Total skills: {evolveResult.total_skills_after}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "execute" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Execute</h2>
                  <p className="text-sm text-text-secondary">Run raw workflow stages through the execution orchestrator.</p>
                </div>
                <div className="card space-y-3">
                  <textarea
                    value={executeStages}
                    onChange={(e) => setExecuteStages(e.target.value)}
                    placeholder={`[{"id":"stage1","tool":"diagnose_tool","action":"...","params":{}}]`}
                    rows={8}
                    className="input font-mono text-xs resize-none"
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <input type="text" value={executeWorkingDir} onChange={(e) => setExecuteWorkingDir(e.target.value)} placeholder="Working dir" className="input text-xs" />
                    <input type="text" value={executeName} onChange={(e) => setExecuteName(e.target.value)} placeholder="Workflow name" className="input text-xs" />
                  </div>
                  <button onClick={handleExecuteRun} disabled={executeRunning || !isConnected} className="btn-primary text-xs">
                    {executeRunning ? "Executing…" : "▶ Execute stages"}
                  </button>
                  {executeError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{executeError}</div>}
                </div>
                {executeResult && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Result</h3>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(executeResult, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "workflows" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Workflows</h2>
                  <p className="text-sm text-text-secondary">Run a workflow template with KEY=VALUE arguments.</p>
                </div>
                <div className="card space-y-3">
                  <select value={workflowTemplate} onChange={(e) => setWorkflowTemplate(e.target.value)} className="input text-sm">
                    <option value="">Select a template</option>
                    {workflowTemplates.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                  <input
                    type="text"
                    value={workflowArgs}
                    onChange={(e) => setWorkflowArgs(e.target.value)}
                    placeholder="key1=value1 key2=value2 ..."
                    className="input text-sm"
                  />
                  <button onClick={handleWorkflowRun} disabled={workflowRunning || !isConnected || !workflowTemplate} className="btn-primary text-xs">
                    {workflowRunning ? "Running…" : "▶ Run workflow"}
                  </button>
                  {workflowError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{workflowError}</div>}
                </div>
                {workflowResult && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Result</h3>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(workflowResult, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "explore" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Explore</h2>
                  <p className="text-sm text-text-secondary">Systematically search a design space.</p>
                </div>
                <div className="card space-y-3">
                  <input type="text" value={exploreObjective} onChange={(e) => setExploreObjective(e.target.value)} placeholder="Objective, e.g. find highest energy density cathode" className="input text-sm" />
                  <div className="grid grid-cols-2 gap-3">
                    <input type="number" min={1} value={exploreMaxIters} onChange={(e) => setExploreMaxIters(parseInt(e.target.value || "1", 10))} placeholder="Max iterations" className="input text-xs" />
                    <input type="number" min={1} value={exploreMaxBranches} onChange={(e) => setExploreMaxBranches(parseInt(e.target.value || "1", 10))} placeholder="Max branches" className="input text-xs" />
                  </div>
                  <button onClick={handleExploreRun} disabled={exploreRunning || !isConnected || !exploreObjective.trim()} className="btn-primary text-xs">
                    {exploreRunning ? "Exploring…" : "▶ Explore"}
                  </button>
                  {exploreError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{exploreError}</div>}
                </div>
                {exploreResult && (
                  <div className="card space-y-3">
                    <h3 className="text-sm font-semibold">Result</h3>
                    <div className="text-xs text-text-secondary">Explored: {exploreResult.n_branches_explored} · Pruned: {exploreResult.n_branches_pruned} · Convergence: {exploreResult.convergence_reason}</div>
                    {exploreResult.best_branch && <div className="text-xs text-text-secondary">Best branch: {exploreResult.best_branch.name}</div>}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "diagnose" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Diagnose</h2>
                  <p className="text-sm text-text-secondary">Diagnose computational chemistry / MD errors.</p>
                </div>
                <div className="card space-y-3">
                  <textarea value={diagnoseError} onChange={(e) => setDiagnoseError(e.target.value)} placeholder="Paste error message…" rows={4} className="input resize-none text-sm" />
                  <div className="grid grid-cols-3 gap-3">
                    <input type="text" value={diagnoseSoftware} onChange={(e) => setDiagnoseSoftware(e.target.value)} placeholder="Software" className="input text-xs" />
                    <input type="text" value={diagnoseCalcType} onChange={(e) => setDiagnoseCalcType(e.target.value)} placeholder="Calc type" className="input text-xs" />
                    <input type="text" value={diagnoseContext} onChange={(e) => setDiagnoseContext(e.target.value)} placeholder="Context" className="input text-xs" />
                  </div>
                  <button onClick={handleDiagnoseRun} disabled={diagnoseRunning || !isConnected || !diagnoseError.trim()} className="btn-primary text-xs">
                    {diagnoseRunning ? "Diagnosing…" : "▶ Diagnose"}
                  </button>
                  {diagnoseErrorMsg && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{diagnoseErrorMsg}</div>}
                </div>
                {diagnoseResult && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Findings</h3>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(diagnoseResult, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "hpc" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">HPC</h2>
                  <p className="text-sm text-text-secondary">Submit and monitor jobs on a remote cluster.</p>
                </div>
                <div className="card space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <input type="text" value={hpcHost} onChange={(e) => setHpcHost(e.target.value)} placeholder="Host" className="input text-xs" />
                    <input type="text" value={hpcUsername} onChange={(e) => setHpcUsername(e.target.value)} placeholder="Username" className="input text-xs" />
                    <select value={hpcScheduler} onChange={(e) => setHpcScheduler(e.target.value as any)} className="input text-xs">
                      <option value="slurm">SLURM</option>
                      <option value="pbs">PBS</option>
                    </select>
                    <input type="text" value={hpcKeyPath} onChange={(e) => setHpcKeyPath(e.target.value)} placeholder="SSH key path (optional)" className="input text-xs" />
                  </div>
                  <button onClick={handleHpcTest} disabled={hpcRunning || !isConnected || !hpcHost || !hpcUsername} className="btn-secondary text-xs">
                    Test connection
                  </button>
                  <hr className="border-border" />
                  <input type="text" value={hpcCommand} onChange={(e) => setHpcCommand(e.target.value)} placeholder="Command to run" className="input text-sm" />
                  <div className="grid grid-cols-3 gap-3">
                    <input type="text" value={hpcJobName} onChange={(e) => setHpcJobName(e.target.value)} placeholder="Job name" className="input text-xs" />
                    <input type="text" value={hpcWalltime} onChange={(e) => setHpcWalltime(e.target.value)} placeholder="Walltime" className="input text-xs" />
                    <input type="text" value={hpcQueue} onChange={(e) => setHpcQueue(e.target.value)} placeholder="Queue" className="input text-xs" />
                    <input type="number" min={1} value={hpcNodes} onChange={(e) => setHpcNodes(parseInt(e.target.value || "1", 10))} placeholder="Nodes" className="input text-xs" />
                    <input type="number" min={1} value={hpcNtasks} onChange={(e) => setHpcNtasks(parseInt(e.target.value || "1", 10))} placeholder="Tasks/node" className="input text-xs" />
                    <input type="text" value={hpcJobId} onChange={(e) => setHpcJobId(e.target.value)} placeholder="Job ID" className="input text-xs" />
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={handleHpcSubmit} disabled={hpcRunning || !isConnected || !hpcCommand.trim()} className="btn-primary text-xs">
                      Submit
                    </button>
                    <button onClick={handleHpcStatus} disabled={hpcRunning || !isConnected || !hpcJobId.trim()} className="btn-secondary text-xs">
                      Status
                    </button>
                  </div>
                  {hpcError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{hpcError}</div>}
                </div>
                {hpcResult && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Result</h3>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(hpcResult, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "periodic" && (
            <div className="h-full overflow-y-auto p-4">
              <PeriodicTable API_BASE={API_BASE} />
            </div>
          )}

          {activeTab === "notebook" && (
            <div className="h-full overflow-hidden p-4">
              <Notebook API_BASE={API_BASE} />
            </div>
          )}

          {activeTab === "sandbox" && (
            <div className="h-full overflow-hidden p-4">
              <SandboxPanel API_BASE={API_BASE} />
            </div>
          )}

          {activeTab === "sweep" && (
            <div className="h-full overflow-y-auto p-4">
              <SweepDashboard API_BASE={API_BASE} />
            </div>
          )}

          {activeTab === "structure" && (
            <div className="h-full overflow-hidden p-4">
              <StructureViewer API_BASE={API_BASE} />
            </div>
          )}
        </div>
      </main>

      {showGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-text-primary/40 p-4 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl border border-border bg-bg-secondary p-6 shadow-2xl">
            <h2 className="mb-1 text-xl font-bold">Welcome to Huginn</h2>
            <p className="mb-5 text-sm italic text-text-secondary">
              Magic springs from the wellspring of imagination.
            </p>
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
