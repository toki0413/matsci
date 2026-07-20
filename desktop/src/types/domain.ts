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

// 4 层 memory 状态, 对应后端 GET /memory/layers 返回结构.
// 每层都可能 available=false (后端 try/except 隔离, 单层失败不阻塞其他).
export interface MemoryLayers {
  wm?: {
    token_used?: number;
    token_budget?: number;
    messages_count?: number;
    tool_calls_count?: number;
    summaries_count?: number;
    last_summarize_at?: string | null;
    extreme_dispatch?: boolean;
    available?: boolean;
    error?: string;
  };
  em?: {
    total_entries?: number;
    tier_counts?: { short?: number; mid?: number; long?: number };
    recent_episodes?: Array<{
      id: string;
      content: string;
      last_accessed: string | null;
      importance: number | null;
      source: string | null;
    }>;
    available?: boolean;
    error?: string;
  };
  sm?: {
    kb_chunks?: number;
    kg_nodes?: number;
    kg_edges?: number;
    kg_node_types?: Record<string, number>;
    recent_patterns?: Array<{
      doc_id: string;
      task_pattern: string;
      run_id: string;
      objective: string;
      confidence: number;
      doc_preview: string;
    }>;
    available?: boolean;
    error?: string;
  };
  pm?: {
    stable_principles_count?: number;
    stable_principles_preview?: string[];
    top_patterns_by_confidence?: Array<{
      doc_id: string;
      task_pattern: string;
      confidence: number;
      objective: string;
    }>;
    available?: boolean;
    error?: string;
  };
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
  // 极限模式 + 分层 memory (Settings Advanced tab)
  extreme_dispatch: boolean;
  wm_summarize: "rule" | "ngram" | "llm" | "hybrid";
  wm_token_budget: number;
  em_recall_top_k: number;
  pm_c_min: number;
  wm_summarize_every_n: number;
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

export type SearchResultType = "thread" | "memory" | "knowledge" | "provenance";

export interface GlobalSearchResult {
  type: SearchResultType;
  id: string;
  title: string;
  snippet: string;
  score: number;
  metadata: Record<string, any>;
}

export interface GlobalSearchResponse {
  results: GlobalSearchResult[];
  total: number;
  sources: string[];
  error?: string;
}

// v7 G59: 认知热机健康状态. 后端 cognitive_heat_engine.health_check() 返回.
// 每轮 darwin_ratchet 后通过 SSE campaign event 推送一次.
export interface HeatEngineHealth {
  eta_cog: number;
  Re_cog: number;
  Re_crit: number;
  T_hot: number;
  T_cold: number;
  U: number;
  L: number;
  nu: number;
  intermittency_kurtosis: number;
  cumulative_work: number;
  cumulative_entropy_produced: number;
  status: "healthy" | "stagnant" | "chaotic" | "conservative";
  warnings: string[];
}
