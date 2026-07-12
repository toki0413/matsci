/**
 * Shared domain types used across hooks and panel components.
 *
 * Extracted from App.tsx to allow independent hook modules.
 */

export interface MemoryEntry {
  id: string;
  category: string;
  content: string;
  tags: string[];
  source: string;
  importance: number;
  tier: string;
  created_at: string;
  last_accessed: string;
  expires_at: string | null;
  access_count: number;
}

export interface MemoryStats {
  longterm_entries: number;
  tier_counts: { short: number; mid: number; long: number };
}

export interface KbDoc {
  doc_id: string;
  filename: string;
}

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface BackendLogEvent {
  source: "stdout" | "stderr";
  text: string;
  time: string;
}

export interface DiffEntry {
  path: string;
  status: "modified" | "added" | "deleted";
  diff: string;
  old: string;
  new: string;
}

export interface Checkpoint {
  id: string;
  base: string;
  files: number;
}

export interface DocumentParseResult {
  info_packages?: number;
  graph?: { nodes?: unknown[]; edges?: unknown[] };
}

export interface DocumentGraph {
  nodes?: unknown[];
  edges?: unknown[];
}

export interface McpServer {
  name: string;
  connected: boolean;
  tools: { name: string; description: string; input_schema?: any }[];
}

export interface DiscoveredServer {
  name: string;
  path: string;
  command: string;
  args: string[];
}

export interface ModelConfig {
  alias: string;
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
  temperature: number;
  enabled: boolean;
  credential_id?: string | null;
  // reasoning intensity: null = off, "low"/"medium"/"high" = on
  thinking?: string | null;
  // max output tokens; null = provider default
  max_tokens?: number | null;
}

export interface AgentProfile {
  id: string;
  name: string;
  model_alias: string;
  persona: string;
  tools: string[];
  enabled: boolean;
  max_steps: number;
}

export interface AppConfig {
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

export interface PersonaSeed {
  name?: string;
  id?: string;
  description?: string;
  avatar?: string;
}

export interface PersonaEmotionResponse {
  context_prompt?: string;
  state?: { valence?: number; arousal?: number; trust?: number };
}

export interface ToolInfo {
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
  destructive?: boolean;
  read_only?: boolean;
}

export interface SkillInfo {
  name: string;
  description: string;
  category: string;
  parameters: Array<{
    name: string;
    type: string;
    description: string;
    required?: boolean;
    default?: any;
    enum?: string[];
  }>;
  tags: string[];
}
