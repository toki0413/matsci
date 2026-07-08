/**
 * SettingsPanel — extracted from App.tsx.
 *
 * Renders the full settings UI: sub-tab navigation, general / models /
 * agents / privacy / pet / security / credentials / jobs / export / bot tabs,
 * plus the save button and backend start/stop card.
 */
import { useState, lazy, Suspense } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ChevronDown } from "lucide-react";
import { SettingsTabNav, ConfigField } from "../SettingsPanel";
import type { SettingsTab } from "../SettingsPanel";
import { PROVIDERS } from "../../lib/constants";
import { api } from "../../lib/api";
import type { ModelConfig, AgentProfile, AppConfig } from "../../types/domain";

// Lazy-load heavy sub-panels so their chunks stay out of the initial bundle.
const CredentialsPanel = lazy(() => import("../CredentialsPanel"));
const RemoteJobsPanel = lazy(() => import("../RemoteJobsPanel"));

// ── LocalModelDiscoverer (moved from App.tsx module scope) ──────────────
// Probes a local server (ollama / vllm / local / openai-compatible) and
// lists model names that can be picked with one click.
function LocalModelDiscoverer({
  model,
  onUpdate,
}: {
  model: ModelConfig;
  onUpdate: (patch: Partial<ModelConfig>) => void;
}) {
  const [discovered, setDiscovered] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const LOCAL_PROVIDERS = ["ollama", "vllm", "local", "openai-compatible"];
  if (!LOCAL_PROVIDERS.includes(model.provider)) return null;

  const discover = async () => {
    setLoading(true);
    setErr("");
    try {
      const params = new URLSearchParams({
        provider: model.provider,
        base_url: model.base_url || "",
      });
      const data = await api.get<{ success?: boolean; models?: string[]; error?: string }>(
        `/config/local-models?${params.toString()}`
      );
      if (data.success && Array.isArray(data.models) && data.models.length > 0) {
        setDiscovered(data.models);
      } else {
        setErr(data.error || "未发现可用模型, 请检查 base URL");
        setDiscovered([]);
      }
    } catch (e: any) {
      setErr(e.message || "请求失败");
      setDiscovered([]);
    }
    setLoading(false);
  };

  return (
    <div className="md:col-span-2 space-y-1.5">
      <button
        onClick={discover}
        disabled={loading}
        className="btn-secondary px-2.5 py-1 text-xs disabled:opacity-50"
      >
        {loading ? "探测中…" : "Discover Local Models"}
      </button>
      {err && <p className="text-xs text-error">{err}</p>}
      {discovered.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {discovered.map((m) => (
            <button
              key={m}
              onClick={() => onUpdate({ model: m })}
              className="rounded bg-accent/10 px-2 py-0.5 text-xs text-accent transition-colors hover:bg-accent/20"
            >
              {m}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Props ────────────────────────────────────────────────────────────────

export interface SettingsPanelProps {
  // From useConfig hook
  config: AppConfig;
  configDirty: boolean;
  configSavedMsg: string;
  settingsTab: SettingsTab;
  llmCredOptions: Array<{ id: string; name: string; provider?: string }>;
  expandedModels: Set<number>;
  expandedAgents: Set<number>;
  setConfig: React.Dispatch<React.SetStateAction<AppConfig>>;
  setConfigDirty: React.Dispatch<React.SetStateAction<boolean>>;
  setConfigSavedMsg: React.Dispatch<React.SetStateAction<string>>;
  setSettingsTab: React.Dispatch<React.SetStateAction<SettingsTab>>;
  saveConfig: (next: AppConfig) => Promise<void>;
  updateModel: (idx: number, patch: Partial<ModelConfig>) => void;
  addModel: () => void;
  removeModel: (idx: number) => void;
  updateAgent: (idx: number, patch: Partial<AgentProfile>) => void;
  addAgent: () => void;
  removeAgent: (idx: number) => void;
  toggleModelExpanded: (i: number) => void;
  toggleAgentExpanded: (i: number) => void;
  switchPersona: (personaName: string) => Promise<void>;

  // From App.tsx / useChatAndConnection
  startBackend: () => void;
  status: string;
  personaList: Array<{ id: string; label: string; description?: string; avatar?: string }>;
  personaEmotion: { mood: string; valence: number; arousal: number; trust: number } | null;
}

// ── Component ────────────────────────────────────────────────────────────

export function SettingsPanel(props: SettingsPanelProps) {
  const {
    config, configDirty, configSavedMsg, settingsTab, llmCredOptions,
    expandedModels, expandedAgents,
    setConfig, setConfigDirty, setConfigSavedMsg, setSettingsTab,
    saveConfig, updateModel, addModel, removeModel,
    updateAgent, addAgent, removeAgent,
    toggleModelExpanded, toggleAgentExpanded, switchPersona,
    startBackend, status, personaList, personaEmotion,
  } = props;

  return (
    <div className="flex h-full flex-col">
      <SettingsTabNav activeTab={settingsTab} onTabChange={setSettingsTab} />
      <div className="flex-1 overflow-y-auto p-6">
        {settingsTab === "general" && (
          <div className="max-w-2xl space-y-5">
            <p className="text-sm text-text-secondary">
              Default single-model settings. For multi-LLM mode, switch to the Models tab.
            </p>
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
              <ConfigField label="Provider">
                <select
                  value={config.provider}
                  onChange={(e) => { const next = { ...config, provider: e.target.value }; setConfig(next); setConfigDirty(true); }}
                  className="input"
                >
                  {PROVIDERS.map((p) => (
                    <option key={p.id} value={p.id}>{p.label}</option>
                  ))}
                </select>
              </ConfigField>
              <ConfigField label="Model">
                <input
                  type="text"
                  value={config.model}
                  onChange={(e) => { setConfig({ ...config, model: e.target.value }); setConfigDirty(true); }}
                  placeholder="e.g. gpt-4o"
                  className="input"
                />
              </ConfigField>
              <ConfigField label="Persona" full>
                <select
                  value={config.persona}
                  onChange={(e) => {
                    const next = { ...config, persona: e.target.value };
                    setConfig(next);
                    setConfigDirty(true);
                    switchPersona(e.target.value);
                  }}
                  className="input"
                >
                  {personaList.map((p) => (
                    <option key={p.id} value={p.id}>{p.label}{p.description ? ` — ${p.description.slice(0, 40)}` : ""}</option>
                  ))}
                </select>
                {personaEmotion && (
                  <div className="mt-1.5 text-xs text-text-tertiary">
                    <span className="inline-flex items-center gap-1">
                      <span className="inline-block h-2 w-2 rounded-full" style={{
                        backgroundColor: personaEmotion.valence > 0 ? "#7ee787" : personaEmotion.valence < -0.3 ? "#f85149" : "#8b949e"
                      }} />
                      {personaEmotion.mood || "neutral mood"}
                    </span>
                  </div>
                )}
              </ConfigField>
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
              <ConfigField label="API Key" full>
                <input
                  type="password"
                  value={config.api_key}
                  onChange={(e) => { setConfig((prev) => ({ ...prev, api_key: e.target.value })); setConfigDirty(true); }}
                  placeholder={PROVIDERS.find((p) => p.id === config.provider)?.keyVar || "API key"}
                  className="input"
                />
              </ConfigField>
              <ConfigField label="Base URL (optional)" full>
                <input
                  type="text"
                  value={config.base_url}
                  onChange={(e) => { setConfig({ ...config, base_url: e.target.value }); setConfigDirty(true); }}
                  placeholder="https://api.openai.com/v1"
                  className="input"
                />
              </ConfigField>
              <ConfigField label="Ollama Host" full>
                <input
                  type="text"
                  value={config.ollama_host}
                  onChange={(e) => { setConfig((prev) => ({ ...prev, ollama_host: e.target.value })); setConfigDirty(true); }}
                  placeholder="http://localhost:11434"
                  className="input"
                />
              </ConfigField>
              <ConfigField label="Max concurrent sub-agents" full>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={config.max_concurrent_subagents}
                  onChange={(e) => { const next = { ...config, max_concurrent_subagents: parseInt(e.target.value || "1", 10) }; setConfig(next); setConfigDirty(true); }}
                  className="input"
                />
              </ConfigField>
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

                    {/* Link to a stored credential, or probe a local server for model names */}
                    <div className="md:col-span-2 space-y-1.5 border-t border-border pt-3">
                      <label className="block text-xs font-medium text-text-secondary">Stored credential (optional)</label>
                      <select
                        className="input-field text-xs"
                        value={m.credential_id || ""}
                        onChange={(e) => updateModel(i, { credential_id: e.target.value || null })}
                      >
                        <option value="">— Direct API Key —</option>
                        {llmCredOptions.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.name}{c.provider ? ` (${c.provider})` : ""}
                          </option>
                        ))}
                      </select>
                      {m.credential_id ? (
                        <p className="text-xs text-text-secondary">
                          Using stored credential. API key field above can be left empty.
                        </p>
                      ) : llmCredOptions.length === 0 ? (
                        <p className="text-xs text-text-muted">
                          No stored LLM credentials yet — add some in the Credentials tab.
                        </p>
                      ) : null}
                      <LocalModelDiscoverer model={m} onUpdate={(patch) => updateModel(i, patch)} />
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
                      {personaList.map((p) => (
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
              <ConfigField label="Max tool output tokens">
                <input
                  type="number"
                  min={0}
                  value={config.max_tool_output_tokens}
                  onChange={(e) => { const next = { ...config, max_tool_output_tokens: parseInt(e.target.value || "0", 10) }; setConfig(next); setConfigDirty(true); }}
                  placeholder="0 = unlimited"
                  className="input"
                />
                <p className="mt-1 text-xs text-text-muted">Tool results longer than this are truncated before being sent to the LLM.</p>
              </ConfigField>
              <ConfigField label="Context budget tokens">
                <input
                  type="number"
                  min={0}
                  value={config.context_budget_tokens}
                  onChange={(e) => { const next = { ...config, context_budget_tokens: parseInt(e.target.value || "0", 10) }; setConfig(next); setConfigDirty(true); }}
                  placeholder="0 = unlimited"
                  className="input"
                />
                <p className="mt-1 text-xs text-text-muted">Warn when the estimated prompt tokens exceed this budget.</p>
              </ConfigField>
            </div>
          </div>
        )}

        {settingsTab === "pet" && (
          <div className="max-w-2xl space-y-5">
            <p className="text-sm text-text-secondary">
              Customize your desktop companion.
            </p>
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
              <ConfigField label="Pet name">
                <input
                  type="text"
                  value={config.pet_name}
                  onChange={(e) => { const next = { ...config, pet_name: e.target.value }; setConfig(next); setConfigDirty(true); }}
                  placeholder="Muninn"
                  className="input"
                />
              </ConfigField>
              <ConfigField label="Personality">
                <select
                  value={config.pet_personality}
                  onChange={(e) => { const next = { ...config, pet_personality: e.target.value as "cheerful" | "nerdy" | "calm" | "sassy" }; setConfig(next); setConfigDirty(true); }}
                  className="input"
                >
                  <option value="cheerful">Cheerful</option>
                  <option value="nerdy">Nerdy</option>
                  <option value="calm">Calm</option>
                  <option value="sassy">Sassy</option>
                </select>
              </ConfigField>
            </div>

            {/* Accessories */}
            <ConfigField label="Accessories">
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
            </ConfigField>

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
                    await api.post("/pet/reset");
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
            <ConfigField label="Encryption password">
              <input
                type="password"
                value={config.encryption_password}
                onChange={(e) => { const next = { ...config, encryption_password: e.target.value }; setConfig(next); setConfigDirty(true); }}
                placeholder="Leave empty to keep unchanged"
                className="input"
              />
            </ConfigField>
            <ConfigField label="Key file path (optional)">
              <input
                type="text"
                value={config.encryption_key_file}
                onChange={(e) => { const next = { ...config, encryption_key_file: e.target.value }; setConfig(next); setConfigDirty(true); }}
                placeholder="Path to encrypted key file"
                className="input"
              />
            </ConfigField>
            <button
              onClick={async () => {
                try {
                  const data = await api.post<{ success?: boolean; path?: string; error?: string }>(
                    "/config/encrypt",
                    { path: "huginn.toml", password: config.encryption_password }
                  );
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
          <Suspense fallback={<div className="flex h-32 items-center justify-center text-sm text-text-muted">Loading…</div>}>
            <CredentialsPanel />
          </Suspense>
        )}

        {settingsTab === "jobs" && (
          <Suspense fallback={<div className="flex h-32 items-center justify-center text-sm text-text-muted">Loading…</div>}>
            <RemoteJobsPanel />
          </Suspense>
        )}

        {/* Export / Import Panel */}
        {settingsTab === "export" && (
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">数据导出与共享</h3>
              <p className="mt-1 text-xs text-text-muted">
                将记忆、知识库、知识图谱等数据打包为压缩文件, 可用于备份或分享给其他用户。
              </p>
            </div>

            {/* Export status */}
            <div className="rounded-lg border border-border bg-bg-tertiary p-3">
              <div className="mb-2 text-xs font-medium text-text-secondary">可导出数据</div>
              <div id="export-status" className="space-y-1 text-xs text-text-muted">
                <span className="text-text-muted">点击下方按钮查看...</span>
              </div>
              <button
                onClick={async () => {
                  try {
                    const data = await api.get<{ available?: Record<string, boolean> }>("/export/status");
                    const el = document.getElementById("export-status");
                    if (el && data.available) {
                      // 用 textContent 代替 innerHTML, 避免 XSS
                      el.innerHTML = "";
                      Object.entries(data.available)
                        .filter(([, v]) => v)
                        .forEach(([k]) => {
                          const div = document.createElement("div");
                          div.textContent = `✓ ${k}`;
                          el.appendChild(div);
                        });
                    }
                  } catch (e: any) {
                    console.error("export status error:", e);
                  }
                }}
                className="btn-secondary text-xs mt-2"
              >
                刷新状态
              </button>
            </div>

            {/* Export all */}
            <div className="rounded-lg border border-border bg-bg-tertiary p-3">
              <div className="mb-2 text-xs font-medium text-text-secondary">全量导出</div>
              <div className="flex gap-2">
                <button
                  onClick={async () => {
                    try {
                      const blob = await api.getBlob("/export/all", {
                        method: "POST",
                        body: JSON.stringify({ format: "zip" }),
                        headers: { "Content-Type": "application/json" },
                      });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = "huginn_export.zip";
                      a.click();
                      URL.revokeObjectURL(url);
                    } catch (e: any) {
                      console.error("export error:", e);
                    }
                  }}
                  className="btn-primary text-xs"
                >
                  📦 导出全部 (ZIP)
                </button>
                <button
                  onClick={async () => {
                    try {
                      const blob = await api.getBlob("/export/memory", {
                        method: "POST",
                        body: JSON.stringify({ format: "json" }),
                        headers: { "Content-Type": "application/json" },
                      });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = "huginn_memory.json";
                      a.click();
                      URL.revokeObjectURL(url);
                    } catch (e: any) {
                      console.error("export memory error:", e);
                    }
                  }}
                  className="btn-secondary text-xs"
                >
                  🧠 仅记忆
                </button>
                <button
                  onClick={async () => {
                    try {
                      const blob = await api.getBlob("/export/knowledge", {
                        method: "POST",
                        body: JSON.stringify({ format: "json" }),
                        headers: { "Content-Type": "application/json" },
                      });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = "huginn_knowledge.json";
                      a.click();
                      URL.revokeObjectURL(url);
                    } catch (e: any) {
                      console.error("export knowledge error:", e);
                    }
                  }}
                  className="btn-secondary text-xs"
                >
                  📚 仅知识库
                </button>
              </div>
            </div>

            {/* Import */}
            <div className="rounded-lg border border-border bg-bg-tertiary p-3">
              <div className="mb-2 text-xs font-medium text-text-secondary">导入归档</div>
              <input
                type="file"
                accept=".zip,.tar.gz,.json"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  const formData = new FormData();
                  formData.append("file", file);
                  try {
                    const data = await api.upload<{ imported?: Record<string, any> }>(
                      "/import/all",
                      formData
                    );
                    if (data.imported) {
                      alert(`导入成功: ${JSON.stringify(data.imported)}`);
                    }
                  } catch (e: any) {
                    alert(`导入失败: ${e.message}`);
                  }
                }}
                className="hidden"
                id="import-file-input"
              />
              <button
                onClick={() => document.getElementById("import-file-input")?.click()}
                className="btn-secondary text-xs"
              >
                📥 选择文件导入
              </button>
              <p className="mt-1 text-xs text-text-muted">
                支持 ZIP / TAR.GZ / JSON 格式, 导入时自动合并到现有数据。
              </p>
            </div>
          </div>
        )}

        {/* Bot Management Panel */}
        {settingsTab === "bot" && (
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">机器人接入</h3>
              <p className="mt-1 text-xs text-text-muted">
                通过 OneBot v11 协议接入 QQ/微信机器人。需要先部署 go-cqhttp / Lagrange / NapCat。
              </p>
            </div>

            {/* Bot status */}
            <div className="rounded-lg border border-border bg-bg-tertiary p-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-medium text-text-secondary">状态: </span>
                  <span id="bot-status-text" className="text-xs text-text-muted">未运行</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={async () => {
                      try {
                        const data = await api.post<{ running?: boolean }>("/bot/start");
                        const el = document.getElementById("bot-status-text");
                        if (el) el.textContent = data.running ? "运行中" : "启动失败";
                      } catch (e: any) {
                        console.error("bot start error:", e);
                      }
                    }}
                    className="btn-primary text-xs"
                  >
                    ▶ 启动
                  </button>
                  <button
                    onClick={async () => {
                      try {
                        await api.post("/bot/stop");
                        const el = document.getElementById("bot-status-text");
                        if (el) el.textContent = "已停止";
                      } catch (e: any) {
                        console.error("bot stop error:", e);
                      }
                    }}
                    className="btn-secondary text-xs"
                  >
                    ⏹ 停止
                  </button>
                </div>
              </div>
              <button
                onClick={async () => {
                  try {
                    const data = await api.get<{ running?: boolean; platform?: string }>("/bot/status");
                    const el = document.getElementById("bot-status-text");
                    if (el) el.textContent = data.running ? `运行中 (${data.platform})` : "未运行";
                  } catch (e: any) {
                    console.error("bot status error:", e);
                  }
                }}
                className="btn-secondary text-xs mt-2"
              >
                刷新状态
              </button>
            </div>

            {/* Bot config */}
            <div className="rounded-lg border border-border bg-bg-tertiary p-3">
              <div className="mb-2 text-xs font-medium text-text-secondary">配置</div>
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-text-muted">平台</label>
                  <select
                    id="bot-platform"
                    className="input mt-1 w-full text-xs"
                    defaultValue="qq"
                  >
                    <option value="qq">QQ</option>
                    <option value="wechat">微信</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-text-muted">Bot ID (QQ号/微信号)</label>
                  <input
                    type="text"
                    id="bot-id"
                    className="input mt-1 w-full text-xs"
                    placeholder="如: 123456789"
                  />
                </div>
                <div>
                  <label className="text-xs text-text-muted">HTTP API 地址</label>
                  <input
                    type="text"
                    id="bot-api-url"
                    className="input mt-1 w-full text-xs"
                    placeholder="如: http://127.0.0.1:5700"
                  />
                </div>
                <div>
                  <label className="text-xs text-text-muted">事件监听端口</label>
                  <input
                    type="number"
                    id="bot-http-port"
                    className="input mt-1 w-full text-xs"
                    defaultValue={8080}
                  />
                </div>
                <div>
                  <label className="text-xs text-text-muted">允许的群号 (逗号分隔, 留空=全部)</label>
                  <input
                    type="text"
                    id="bot-allowed-groups"
                    className="input mt-1 w-full text-xs"
                    placeholder="如: 123456,789012"
                  />
                </div>
              </div>
              <button
                onClick={async () => {
                  const config: any = {
                    platform: (document.getElementById("bot-platform") as HTMLSelectElement)?.value || "qq",
                    bot_id: (document.getElementById("bot-id") as HTMLInputElement)?.value || "",
                    api_url: (document.getElementById("bot-api-url") as HTMLInputElement)?.value || "",
                    http_port: parseInt((document.getElementById("bot-http-port") as HTMLInputElement)?.value || "8080"),
                    enabled: true,
                  };
                  const groups = (document.getElementById("bot-allowed-groups") as HTMLInputElement)?.value;
                  if (groups) {
                    config.allowed_groups = groups.split(",").map((s: string) => s.trim()).filter(Boolean);
                  }
                  try {
                    await api.put("/bot/config", config);
                    alert("配置已保存");
                  } catch (e: any) {
                    alert(`保存失败: ${e.message}`);
                  }
                }}
                className="btn-primary text-xs mt-2"
              >
                保存配置
              </button>
            </div>

            <div className="rounded-lg border border-accent/20 bg-accent/5 p-3">
              <p className="text-xs text-accent">
                💡 使用说明: 先启动 go-cqhttp 或 Lagrange, 配置 HTTP POST 上报到
                <code className="mx-1 rounded bg-bg-tertiary px-1">http://your-host:8080/onebot/v11/event</code>
                然后在上方填写配置并启动。
              </p>
            </div>
          </div>
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
  );
}
