/**
 * useConfig — App configuration state management.
 *
 * Manages the AppConfig (models, agents, privacy, pet, encryption),
 * settings tab navigation, model/agent CRUD, and config save/push.
 */
import { useState, useCallback, useEffect } from "react";
import { api } from "../lib/api";
import {
  loadStoredConfig, saveStoredConfig,
} from "../lib/config-store";
import type { AppConfig, ModelConfig, AgentProfile } from "../types/domain";

type SettingsTab =
  | "general" | "models" | "agents" | "privacy" | "pet"
  | "security" | "credentials" | "jobs" | "export" | "bot" | "advanced";

export function useConfig() {
  // Lazy init: loadStoredConfig() hits localStorage, only run once on mount.
  const [config, setConfig] = useState<AppConfig>(() => loadStoredConfig());
  const [configDirty, setConfigDirty] = useState(false);
  const [configSavedMsg, setConfigSavedMsg] = useState<string>("");
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("general");
  const [llmCredOptions, setLlmCredOptions] = useState<Array<{ id: string; name: string; provider?: string }>>([]);
  // 当前生效模型 (lead agent 绑定的 alias)。由 /config/active-model 读写。
  const [activeModel, setActiveModel] = useState<string>("");
  const [activeModelSavedMsg, setActiveModelSavedMsg] = useState<string>("");
  // 模型档位 (极简模式)。由 /config/model-tier 读取/切换。
  const [modelTier, setModelTier] = useState<string>("");
  const [availableModelTiers, setAvailableModelTiers] = useState<string[]>([]);
  const [modelTierSavedMsg, setModelTierSavedMsg] = useState<string>("");

  // Collapse state for Settings model/agent cards
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

  // ── Config push / save ──────────────────────────────────────
  const pushConfig = useCallback(async (cfg: AppConfig) => {
    try {
      const resp = await api.post<{ success?: boolean; error?: string }>("/config", cfg);
      if (resp.success === false) {
        console.warn("[config] backend rejected:", resp.error);
        return false;
      }
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

  // ── Model CRUD ─────────────────────────────────────────────
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

  // ── Agent CRUD ─────────────────────────────────────────────
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

  // ── Active model (runtime provider switch) ────────────────────
  const loadActiveModel = useCallback(async () => {
    try {
      const data = await api.get<{ active_alias?: string }>("/config/active-model");
      if (data.active_alias) setActiveModel(data.active_alias);
    } catch (e) {
      // backend offline — keep previous value
    }
  }, []);

  const switchActiveModel = useCallback(async (alias: string) => {
    try {
      const resp = await api.post<{ success?: boolean; error?: string }>("/config/active-model", { alias });
      if (resp.success) {
        setActiveModel(alias);
        setActiveModelSavedMsg(`已切换到 ${alias}`);
      } else {
        setActiveModelSavedMsg(resp.error || "切换失败");
      }
    } catch (e: any) {
      setActiveModelSavedMsg(`切换失败: ${e.message}`);
    }
    setTimeout(() => setActiveModelSavedMsg(""), 4000);
  }, []);

  // ── Model tier (极简模式) ────────────────────────────────────
  const loadModelTier = useCallback(async () => {
    try {
      const data = await api.get<{
        model_tier?: string;
        available_model_tiers?: string[];
      }>("/config/model-tier");
      if (data.model_tier) setModelTier(data.model_tier);
      if (data.available_model_tiers?.length) setAvailableModelTiers(data.available_model_tiers);
    } catch (e) {
      // backend offline — keep previous value
    }
  }, []);

  const switchModelTier = useCallback(async (tier: string) => {
    try {
      const resp = await api.post<{
        success?: boolean;
        error?: string;
        model_tier?: string;
      }>("/config/model-tier", { tier });
      if (resp.success && resp.model_tier) {
        setModelTier(resp.model_tier);
        setModelTierSavedMsg(`已切换档位: ${resp.model_tier}`);
      } else {
        setModelTierSavedMsg(resp.error || "切换失败");
      }
    } catch (e: any) {
      setModelTierSavedMsg(`切换失败: ${e.message}`);
    }
    setTimeout(() => setModelTierSavedMsg(""), 4000);
  }, []);

  // ── Persona switch ─────────────────────────────────────────
  const switchPersona = async (personaName: string) => {
    try {
      await api.post(`/personas/${personaName}/switch`, {});
      setConfig((prev) => {
        const next = { ...prev, persona: personaName };
        saveStoredConfig(next);
        return next;
      });
    } catch (e) {
      console.error("Failed to switch persona:", e);
    }
  };

  // ── Lazy-load LLM credentials + active model when Models tab opens ──
  useEffect(() => {
    if (settingsTab !== "models") return;
    let alive = true;
    loadActiveModel();
    api
      .get<{ credentials?: any[] }>("/credentials?kind=llm")
      .then((data) => {
        if (!alive) return;
        const list = (data.credentials || []).map((c: any) => ({
          id: c.id,
          name: c.name,
          provider: c.metadata?.provider,
        }));
        setLlmCredOptions(list);
      })
      .catch(() => {
        // backend may be offline; just leave the list empty
      });
    return () => { alive = false; };
  }, [settingsTab]);

  // ── Lazy-load model tier when Advanced tab opens ──
  useEffect(() => {
    if (settingsTab !== "advanced") return;
    loadModelTier();
  }, [settingsTab, loadModelTier]);

  return {
    config, configDirty, configSavedMsg, settingsTab, llmCredOptions,
    expandedModels, expandedAgents, activeModel, activeModelSavedMsg,
    modelTier, availableModelTiers, modelTierSavedMsg,
    setConfig, setConfigDirty, setConfigSavedMsg, setSettingsTab,
    pushConfig, saveConfig,
    ensureDefaultModel, updateModel, addModel, removeModel,
    ensureDefaultAgents, updateAgent, addAgent, removeAgent,
    toggleModelExpanded, toggleAgentExpanded,
    switchPersona, switchActiveModel, loadActiveModel,
    switchModelTier,
  };
}
