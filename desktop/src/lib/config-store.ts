/**
 * Central config store — API base URL, WS URL, localStorage persistence.
 *
 * Single source of truth for backend connection strings. Components import
 * API_BASE / WS_URL as reactive getters; syncBackendUrl() refreshes from
 * the backend / localStorage.
 */

import { getApiBase, setApiBase } from "./api-client";
import type { AppConfig } from "../types/domain";

/** Reactive API base — reads from api-client's module state on every access.
 *  Proxy intercepts property access so callers always get the latest value.
 *  Must handle toString / Symbol.toPrimitive for template-literal coercion. */
export const API_BASE = new Proxy(
  {},
  {
    get(_t, prop) {
      const val = getApiBase();
      if (prop === "toString" || prop === Symbol.toPrimitive || prop === "valueOf") {
        return () => val;
      }
      if (typeof (val as any)[prop] === "function") {
        return (val as any)[prop].bind(val);
      }
      return (val as any)[prop] ?? val;
    },
  }
) as unknown as string;

/** Derived WS URL. */
export let WS_URL: string = "";

/** Bumped every time syncBackendUrl() updates the port. Components that
 *  depend on WS_URL should track this counter to re-connect. */
export let wsUrlVersion: number = 0;

function recomputeWsUrl() {
  const base = getApiBase();
  const newUrl = base.replace(/^http/, "ws") + "/ws/agent";
  // Only bump version when URL actually changes — avoids reconnect storms
  if (newUrl !== WS_URL) {
    WS_URL = newUrl;
    wsUrlVersion++;
  }
}

export async function syncBackendUrl(): Promise<void> {
  // In Tauri, read the sidecar port. In browser/dev, skip — invoke hangs.
  if ("__TAURI_INTERNALS__" in window) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const port = await invoke<number>("get_backend_port").catch(() => null);
      if (port) {
        setApiBase(`http://127.0.0.1:${port}`);
        recomputeWsUrl();
        return;
      }
    } catch {
      // Tauri invoke failed — fall through to probe
    }
  }
  // Probe 8000-8010 — catches externally-started backends in browser mode
  // or when Tauri invoke returns 0. Stops at first /health that responds.
  // ponytail: linear scan is fine for 11 ports; switch to port file read
  // if range grows beyond 50.
  for (let p = 8000; p <= 8010; p++) {
    try {
      const r = await fetch(`http://127.0.0.1:${p}/health`, {
        signal: AbortSignal.timeout(500),
      });
      if (r.ok) {
        setApiBase(`http://127.0.0.1:${p}`);
        break;
      }
    } catch { /* not this port */ }
  }
  recomputeWsUrl();
}

// ── LocalStorage config persistence ──────────────────────────
// ponytail: config blob includes api_key and encryption_password —
// stored in localStorage for simplicity. Any XSS can exfiltrate them.
// Acceptable for local single-user desktop app; upgrade to
// tauri-plugin-stronghold (OS keychain) for sensitive fields.

const CONFIG_KEY = "huginn:config";

export function loadStoredConfig(): AppConfig {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* corrupt JSON — fall through to defaults */ }
  return defaultConfig();
}

export function saveStoredConfig(cfg: AppConfig): void {
  try {
    localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
  } catch { /* quota exceeded — best effort */ }
}

function defaultConfig(): AppConfig {
  return {
    provider: "openai",
    model: "",
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
    max_tool_output_tokens: 12000,
    context_budget_tokens: 100000,
    pet_name: "Muninn",
    pet_personality: "cheerful",
    pet_accessories: [],
    encrypt_config: false,
    encryption_password: "",
    encryption_key_file: "",
    // 极限模式 + 分层 memory 默认值 (跟后端 spec 对齐)
    extreme_dispatch: false,
    wm_summarize: "rule",
    wm_token_budget: 8192,
    em_recall_top_k: 5,
    pm_c_min: 0.2,
    wm_summarize_every_n: 5,
  };
}

// ── Persona fallback list ───────────────────────────────────

export const PERSONAS_FALLBACK: Array<{
  id: string;
  label: string;
  description?: string;
  avatar?: string;
}> = [
  { id: "default", label: "Default", description: "Balanced general-purpose assistant" },
  { id: "tutor", label: "Tutor", description: "Patient, educational, step-by-step" },
  { id: "reviewer", label: "Reviewer", description: "Critical, thorough, citation-focused" },
  { id: "researcher", label: "Researcher", description: "Exploratory, hypothesis-driven" },
  { id: "engineer", label: "Engineer", description: "Practical, code-first, concise" },
  { id: "analyst", label: "Analyst", description: "Data-driven, quantitative" },
];
