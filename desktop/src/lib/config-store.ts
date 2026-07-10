/**
 * Central config store — API base URL, WS URL, localStorage persistence.
 *
 * Single source of truth for backend connection strings. Components import
 * API_BASE / WS_URL as reactive getters; syncBackendUrl() refreshes from
 * the backend / localStorage.
 */

import { getApiBase, setApiBase } from "./api-client";
import type { AppConfig } from "../types/domain";

/** Reactive API base — reads from api-client's module state.
 *  Proxy intercepts property access so callers always get the latest value.
 *  Must handle toString / Symbol.toPrimitive for template-literal coercion. */
export const API_BASE = new Proxy(
  { _v: "" },
  {
    get(t, prop) {
      if (prop === "toString" || prop === Symbol.toPrimitive || prop === "valueOf") {
        return () => {
          if (!t._v) t._v = getApiBase();
          return t._v;
        };
      }
      if (!t._v) t._v = getApiBase();
      const val = t._v;
      // delegate string method calls (.replace, .startsWith, etc.)
      if (typeof (val as any)[prop] === "function") {
        return (val as any)[prop].bind(val);
      }
      return (val as any)[prop] ?? val;
    },
  }
) as unknown as string;

/** Derived WS URL. */
export let WS_URL: string = "";

function recomputeWsUrl() {
  const base = getApiBase();
  WS_URL = base.replace(/^http/, "ws") + "/ws/agent";
}
recomputeWsUrl();

export async function syncBackendUrl(): Promise<void> {
  // Try to read the actual port from the Tauri-managed sidecar
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const port = await invoke<number>("get_backend_port").catch(() => null);
    if (port) {
      const base = `http://127.0.0.1:${port}`;
      setApiBase(base);
      recomputeWsUrl();
    }
  } catch {
    // Not in Tauri context (dev mode) — use whatever api-client already has
  }
  recomputeWsUrl();
}

// ── LocalStorage config persistence ──────────────────────────

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
    pet_name: "Huginn",
    pet_personality: "cheerful",
    pet_accessories: [],
    encrypt_config: false,
    encryption_password: "",
    encryption_key_file: "",
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
