import { useState, useEffect } from "react";
import { Eye, EyeOff } from "lucide-react";
import { api } from "../lib/api";
import { PROVIDERS } from "../lib/constants";

// /config/providers only hands back ids + defaults — the readable label and the
// 国际/国内/本地/自定义 grouping live here so the dropdown reads naturally.
// ponytail: ceiling — if the backend grows a new provider, add a label entry here
// (unknown ids fall back to the raw id under 自定义). upgrade path: have the
// backend return label/category directly and drop this map.
const PROVIDER_LABELS: Record<string, string> = {
  anthropic: "Anthropic (Claude)",
  openai: "OpenAI (GPT)",
  "google-genai": "Google (Gemini)",
  openrouter: "OpenRouter (聚合)",
  nvidia: "NVIDIA NIM",
  deepseek: "DeepSeek (深度求索)",
  siliconflow: "硅基流动 (SiliconFlow)",
  moonshot: "Kimi (月之暗面)",
  zhipu: "智谱 GLM",
  baichuan: "百川",
  dashscope: "阿里通义 (DashScope)",
  qianfan: "百度千帆",
  doubao: "字节豆包",
  hunyuan: "腾讯混元",
  minimax: "MiniMax (稀宇)",
  ollama: "Ollama (本地)",
  vllm: "vLLM (本地)",
  local: "本地 (OpenAI 兼容)",
  "lm-studio": "LM Studio (本地)",
  "llama-cpp": "llama.cpp (本地)",
  sglang: "SGLang (本地)",
  "openai-compatible": "自定义 (OpenAI 兼容)",
  default: "默认 (Default)",
};

const PROVIDER_CATEGORIES = ["国际", "国内", "本地", "自定义"] as const;

const DOMESTIC_PROVIDERS = new Set([
  "deepseek", "siliconflow", "moonshot", "zhipu", "baichuan",
  "dashscope", "qianfan", "doubao", "hunyuan", "minimax",
]);

// Keyless providers are all loopback-local (ollama/vllm/local/lm-studio/...);
// openai-compatible + default are user-supplied, so they land in 自定义.
function providerCategory(p: { provider: string; needs_api_key: boolean }): string {
  if (!p.needs_api_key) return "本地";
  if (p.provider === "openai-compatible" || p.provider === "default") return "自定义";
  return DOMESTIC_PROVIDERS.has(p.provider) ? "国内" : "国际";
}

export function CredentialsPanel() {
  const [sshCreds, setSshCreds] = useState<any[]>([]);
  const [llmCreds, setLlmCreds] = useState<any[]>([]);
  const [serviceCreds, setServiceCreds] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);
  const [editing, setEditing] = useState<{ kind: string; id?: string } | null>(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, any>>({});
  const [importing, setImporting] = useState(false);
  const [showSecret, setShowSecret] = useState(false);

  // Provider catalogue pulled from /config/providers (id + defaults). null while
  // loading / on fetch failure — the dropdown falls back to the flat PROVIDERS list.
  const [providers, setProviders] = useState<any[] | null>(null);
  // Inline connectivity-test result for the LLM form (before it's saved).
  const [formTest, setFormTest] = useState<any>(null);

  // External API key form
  const [apiKeyForm, setApiKeyForm] = useState({ service: "", api_key: "" });
  const [testingService, setTestingService] = useState<string | null>(null);
  const [serviceTestResult, setServiceTestResult] = useState<Record<string, any>>({});

  const SUPPORTED_SERVICES = [
    "openai", "anthropic", "google_ai", "deepseek", "qwen",
    "materials_project", "wiley", "scopus", "springer_nature",
    "elsevier_science_direct", "arxiv", "semantic_scholar",
    "nist_webbook", "pubchem", "chemspider",
  ];

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

  // Look up a fetched provider entry by id (null until /config/providers resolves).
  const providerInfo = (id: string) => providers?.find((p) => p.provider === id) || null;

  const load = async () => {
    try {
      const [ssh, llm, svc] = await Promise.all([
        api.get<{ credentials?: any[] }>("/credentials?kind=ssh"),
        api.get<{ credentials?: any[] }>("/credentials?kind=llm"),
        api.get<{ services?: any[] }>("/credentials"),
      ]);
      setSshCreds(ssh.credentials || []);
      setLlmCreds(llm.credentials || []);
      setServiceCreds(svc.services || []);
    } catch (e: any) {
      flash("加载凭据失败: " + e.message, false);
    }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  // Fetch the provider catalogue once. Silent on failure — the form just falls
  // back to the flat PROVIDERS list, so a flaky backend shouldn't block creds UI.
  useEffect(() => {
    api.get<{ providers?: any[] }>("/config/providers")
      .then((d) => setProviders(d.providers || []))
      .catch(() => setProviders(null));
  }, []);

  const startNew = (kind: string) => {
    setEditing({ kind });
    setTestResult({});
    setFormTest(null);
    if (kind === "ssh") {
      setSshForm({ name: "", host: "", username: "", port: "22", scheduler: "slurm", key_path: "", password: "", remote_work_dir: "~/huginn_jobs", strict_host_key_checking: true });
    } else {
      // prefill model/base_url from the openai defaults if the catalogue's loaded
      const pi = providerInfo("openai");
      setLlmForm({ name: "", provider: "openai", model: pi?.default_model || "", base_url: pi?.default_base_url || "", api_key: "", alias: "" });
    }
  };

  const startEdit = (c: any) => {
    setEditing({ kind: c.kind, id: c.id });
    setTestResult({});
    setFormTest(null);
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
      const data = editing?.id
        ? await api.put<any>(`/credentials/${editing.id}`, body)
        : await api.post<any>("/credentials", body);
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
      const data = editing?.id
        ? await api.put<any>(`/credentials/${editing.id}`, body)
        : await api.post<any>("/credentials", body);
      if (data.success) { flash(editing?.id ? "LLM 凭据已更新" : "LLM 凭据已创建"); setEditing(null); load(); }
      else flash(data.error || "保存失败", false);
    } catch (e: any) { flash("保存出错: " + e.message, false); }
  };

  // Fire an unsaved connectivity probe via the inline model-test endpoint.
  // Sending the raw api_key from the form; for edits with a blank key field the
  // backend falls back to the provider's env var (resolve_provider_key).
  const testForm = async () => {
    if (!llmForm.provider || !llmForm.model) {
      flash("provider / model 必填后再测试", false);
      return;
    }
    setFormTest({ loading: true });
    try {
      const data = await api.post<{ success?: boolean; error?: string | null; latency_ms?: number; model_response?: string }>(
        "/config/models/test",
        {
          alias: "__test__",
          provider: llmForm.provider,
          model: llmForm.model,
          api_key: llmForm.api_key || undefined,
          base_url: llmForm.base_url || undefined,
          temperature: 0.0,
          enabled: true,
        }
      );
      setFormTest(data);
    } catch (e: any) {
      setFormTest({ success: false, error: e.message });
    }
  };

  const remove = async (id: string, name: string) => {
    if (!confirm(`删除凭据 "${name}"？此操作不可撤销。`)) return;
    try {
      const data = await api.del<{ success?: boolean; error?: string }>(`/credentials/${id}`);
      if (data.success) { flash("已删除"); load(); } else flash(data.error || "删除失败", false);
    } catch (e: any) { flash("删除出错: " + e.message, false); }
  };

  const setDef = async (id: string) => {
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(`/credentials/${id}/set-default`);
      if (data.success) { flash("已设为默认"); load(); } else flash(data.error || "设置失败", false);
    } catch (e: any) { flash("出错: " + e.message, false); }
  };

  const test = async (id: string) => {
    setTesting(id);
    setTestResult((p) => ({ ...p, [id]: { loading: true } }));
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(`/credentials/${id}/test`);
      setTestResult((p) => ({ ...p, [id]: data }));
    } catch (e: any) {
      setTestResult((p) => ({ ...p, [id]: { success: false, error: e.message } }));
    }
    setTesting(null);
  };

  // Pull API keys already present in the runtime config (.env / config file)
  // and write them into the credential store so they can be reused / rotated.
  const importFromConfig = async () => {
    setImporting(true);
    try {
      const data = await api.post<{ success?: boolean; imported?: number; count?: number; error?: string }>(
        "/credentials/import-from-config"
      );
      if (data.success) {
        const n = data.imported ?? data.count ?? 0;
        flash(n > 0 ? `已从配置导入 ${n} 条凭据` : "配置中未发现可导入的密钥");
        load();
      } else {
        flash(data.error || "导入失败", false);
      }
    } catch (e: any) {
      flash("导入出错: " + e.message, false);
    }
    setImporting(false);
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
          <div className="relative">
            <input
              type={showSecret ? "text" : "password"}
              className="input pr-9"
              value={sshForm.password}
              onChange={(e) => setSshForm({ ...sshForm, password: e.target.value })}
              placeholder="••••••••"
            />
            <button
              type="button"
              onClick={() => setShowSecret((s) => !s)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
              tabIndex={-1}
            >
              {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
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

  const needsKey = providerInfo(llmForm.provider)?.needs_api_key ?? true;
  const grouped = providers
    ? PROVIDER_CATEGORIES.reduce<Record<string, any[]>>((acc, cat) => {
        acc[cat] = providers.filter((p) => providerCategory(p) === cat);
        return acc;
      }, {})
    : null;

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
          <select
            className="input"
            value={llmForm.provider}
            onChange={(e) => {
              const provider = e.target.value;
              const pi = providerInfo(provider);
              // switching provider resets model/base_url to its defaults so the
              // user sees a sane starting point; they can still override afterwards
              setLlmForm((f) => ({ ...f, provider, model: pi?.default_model || "", base_url: pi?.default_base_url || "" }));
              setFormTest(null);
            }}
          >
            {grouped
              ? PROVIDER_CATEGORIES.map((cat) => {
                  const items = grouped[cat];
                  if (!items?.length) return null;
                  return (
                    <optgroup key={cat} label={cat}>
                      {items.map((p) => (
                        <option key={p.provider} value={p.provider}>
                          {PROVIDER_LABELS[p.provider] || p.provider}
                        </option>
                      ))}
                    </optgroup>
                  );
                })
              : PROVIDERS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">Model *</label>
          <input className="input" value={llmForm.model} onChange={(e) => setLlmForm({ ...llmForm, model: e.target.value })} placeholder={providerInfo(llmForm.provider)?.default_model || "deepseek-chat / gpt-4o / ..."} />
        </div>
        <div>
          <label className="mb-1 block text-xs text-text-secondary">Base URL {needsKey ? "(可选)" : "*"}</label>
          <input className="input" value={llmForm.base_url} onChange={(e) => setLlmForm({ ...llmForm, base_url: e.target.value })} placeholder={providerInfo(llmForm.provider)?.default_base_url || "https://api.deepseek.com"} />
        </div>
        {needsKey && (
          <div className="md:col-span-2">
            <label className="mb-1 block text-xs text-text-secondary">API Key ({editing?.id ? "留空=不修改" : "必填"})</label>
            <div className="relative">
              <input
                type={showSecret ? "text" : "password"}
                className="input pr-9"
                value={llmForm.api_key}
                onChange={(e) => setLlmForm({ ...llmForm, api_key: e.target.value })}
                placeholder={providerInfo(llmForm.provider)?.env_var || "sk-..."}
              />
              <button
                type="button"
                onClick={() => setShowSecret((s) => !s)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
                tabIndex={-1}
              >
                {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </div>
        )}
        {formTest && !formTest.loading && (
          <div className={`md:col-span-2 text-xs ${formTest.success ? "text-success" : "text-error"}`}>
            {formTest.success
              ? `✓ 连通${formTest.latency_ms != null ? ` · ${formTest.latency_ms}ms` : ""}${formTest.model_response ? ` · ${String(formTest.model_response).slice(0, 40)}` : ""}`
              : `✗ ${formTest.error || "失败"}`}
          </div>
        )}
      </div>
      <div className="flex gap-2">
        <button onClick={saveLlm} className="btn-primary text-xs">保存</button>
        <button onClick={testForm} disabled={formTest?.loading} className="btn-secondary text-xs">{formTest?.loading ? "测试中…" : "测试连接"}</button>
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
          <div className="flex items-center gap-2">
            <button
              onClick={importFromConfig}
              disabled={importing}
              className="rounded-lg bg-accent px-2.5 py-1 text-xs text-white transition-colors hover:bg-accent/90 disabled:opacity-50"
            >
              {importing ? "导入中…" : "从配置导入"}
            </button>
            {editing?.kind !== "llm" && <button onClick={() => startNew("llm")} className="btn-secondary text-xs">+ 新增 LLM</button>}
          </div>
        </div>
        {editing?.kind === "llm" && llmFormEl}
        {loading ? <p className="text-xs text-text-muted">加载中…</p> : llmCreds.length === 0 && !editing ? <p className="text-xs text-text-muted">暂无 LLM 凭据, 点击"新增 LLM"添加。</p> : null}
        <div className="space-y-2">{llmCreds.map((c) => renderCard(c, `${c.metadata?.provider || ""} / ${c.metadata?.model || ""}${c.metadata?.base_url ? " · " + c.metadata.base_url : ""}`))}</div>
      </section>

      {/* 外部 API Keys (Materials Project, Scopus, etc.) */}
      <section className="mt-6">
        <div className="mb-2 flex items-center justify-between">
          <h4 className="text-sm font-semibold text-text-primary">外部 API Keys</h4>
          <span className="text-xs text-text-muted">材料数据库 / 文献检索等</span>
        </div>
        <p className="mb-3 text-xs text-text-muted">
          为外部数据源和文献检索服务配置 API Key。密钥加密存储, 不会明文返回。你自行申请后在此输入。
        </p>

        {/* Add new API key form */}
        <div className="mb-3 flex gap-2">
          <select
            value={apiKeyForm.service}
            onChange={(e) => setApiKeyForm({ ...apiKeyForm, service: e.target.value })}
            className="input flex-1 text-xs"
          >
            <option value="">选择服务…</option>
            {SUPPORTED_SERVICES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <div className="relative flex-1">
            <input
              type={showSecret ? "text" : "password"}
              value={apiKeyForm.api_key}
              onChange={(e) => setApiKeyForm({ ...apiKeyForm, api_key: e.target.value })}
              placeholder="API Key…"
              className="input w-full text-xs pr-9"
            />
            <button
              type="button"
              onClick={() => setShowSecret((s) => !s)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
              tabIndex={-1}
            >
              {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
          <button
            onClick={async () => {
              if (!apiKeyForm.service || !apiKeyForm.api_key) return;
              try {
                const data = await api.post<{ success?: boolean; error?: string }>(
                  `/credentials/${apiKeyForm.service}`,
                  { api_key: apiKeyForm.api_key }
                );
                if (data.success !== false) {
                  flash(`${apiKeyForm.service} API Key 已保存`);
                  setApiKeyForm({ service: "", api_key: "" });
                  load();
                } else {
                  flash(`保存失败: ${data.error || "未知错误"}`, false);
                }
              } catch (e: any) {
                flash(`保存失败: ${e.message}`, false);
              }
            }}
            disabled={!apiKeyForm.service || !apiKeyForm.api_key}
            className="btn-primary text-xs disabled:opacity-50"
          >
            保存
          </button>
        </div>

        {/* Service credentials list */}
        {serviceCreds.length === 0 ? (
          <p className="text-xs text-text-muted">暂无外部 API Key。选择服务并输入密钥后点击保存。</p>
        ) : (
          <div className="space-y-2">
            {serviceCreds.map((s: any) => (
              <div key={s.service} className="flex items-center justify-between rounded-lg border border-border bg-bg-tertiary p-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-text-primary">{s.service}</span>
                  {s.has_key ? (
                    <span className="rounded bg-success/20 px-1.5 py-0.5 text-xs text-success">✓ 已配置</span>
                  ) : (
                    <span className="rounded bg-warning/20 px-1.5 py-0.5 text-xs text-warning">未配置</span>
                  )}
                </div>
                <div className="flex gap-2">
                  {s.has_key && (
                    <>
                      <button
                        onClick={async () => {
                          setTestingService(s.service);
                          setServiceTestResult({});
                          try {
                            const data = await api.get<{ valid?: boolean; error?: string }>(
                              `/credentials/${s.service}/test`
                            );
                            setServiceTestResult({ [s.service]: data });
                          } catch (e: any) {
                            setServiceTestResult({ [s.service]: { valid: false, error: e.message } });
                          }
                          setTestingService(null);
                        }}
                        disabled={testingService === s.service}
                        className="text-xs text-accent hover:underline"
                      >
                        {testingService === s.service ? "测试中…" : "测试"}
                      </button>
                      {serviceTestResult[s.service] && (
                        <span className={`text-xs ${serviceTestResult[s.service].valid ? "text-success" : "text-error"}`}>
                          {serviceTestResult[s.service].valid ? "✓ 有效" : `✗ ${serviceTestResult[s.service].error || "无效"}`}
                        </span>
                      )}
                      <button
                        onClick={async () => {
                          try {
                            await api.del(`/credentials/${s.service}`);
                            flash(`${s.service} API Key 已删除`);
                            load();
                          } catch (e: any) {
                            flash(`删除失败: ${e.message}`, false);
                          }
                        }}
                        className="text-xs text-error hover:underline"
                      >
                        删除
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default CredentialsPanel;
