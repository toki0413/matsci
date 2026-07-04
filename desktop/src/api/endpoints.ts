/**
 * Huginn REST API Endpoints
 *
 * Thin typed wrappers around the production api-client, matching
 * the backend route contracts.  Usage:
 *
 *   import { getHealth, getKnowledge } from '../api/endpoints';
 *   const health = await getHealth();
 */
import { api, getApiBase } from '../lib/api-client';

/* ── Health ──────────────────────────────────────── */

export interface HealthData {
  status: string;
  provider?: string;
  model?: string;
  version?: string;
  configured?: boolean;
}

export const getHealth = () => api.get<HealthData>('/health');
export const getReady = () => api.get<{ status: string }>('/ready');

/* ── Knowledge Base ──────────────────────────────── */

export interface KnowledgeDoc {
  id: string;
  name: string;
  size: number;
  chunks: number;
  createdAt: string;
}

export interface KnowledgeListData {
  available: boolean;
  documents: KnowledgeDoc[];
}

export const getKnowledge = () => api.get<KnowledgeListData>('/knowledge');

export const uploadKnowledge = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api.upload<{ success: boolean; error?: string; id?: string }>(
    '/knowledge/upload',
    form,
  );
};

export const deleteKnowledge = (docId: string) =>
  api.delete<{ success: boolean }>(`/knowledge/${docId}`);

export const queryKnowledge = (query: string, topK = 5) =>
  api.post<{ chunks: Array<{ text: string; score: number; source: string }> }>(
    '/knowledge/query',
    { query, top_k: topK },
  );

/* ── Threads ─────────────────────────────────────── */

export interface Thread {
  id: string;
  label: string;
  createdAt: string;
  messageCount?: number;
}

export const getThreads = () => api.get<Thread[]>('/threads');
export const createThread = (title: string) =>
  api.post<Thread>('/threads', { title });
export const deleteThread = (id: string) =>
  api.delete<{ success: boolean }>(`/threads/${id}`);
export const renameThread = (id: string, label: string) =>
  api.patch<{ success: boolean }>(`/threads/${id}`, { label });

/* ── Tools ───────────────────────────────────────── */

export interface ToolInfo {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
}

export const getTools = () => api.get<ToolInfo[]>('/tools');
export const executeTool = (name: string, args: Record<string, unknown>) =>
  api.post(`/tools/${name}`, args);

/* ── Pet ─────────────────────────────────────────── */

export interface PetStatus {
  name: string;
  hunger: number;
  happiness: number;
  xp: number;
  level: number;
}

export const getPetStatus = () => api.get<PetStatus>('/pet/status');
export const feedPet = (amount = 25) =>
  api.post(`/pet/feed?amount=${amount}`);
export const petPet = (amount = 15) =>
  api.post(`/pet/pet?amount=${amount}`);

/* ── Config ──────────────────────────────────────── */

export interface ActiveModel {
  provider: string;
  model: string;
}

export interface AppConfig {
  [key: string]: unknown;
}

export interface Provider {
  id: string;
  name: string;
  models: string[];
}

export const getActiveModel = () => api.get<ActiveModel>('/config/active-model');
export const getConfig = () => api.get<AppConfig>('/config');
export const getProviders = () => api.get<Provider[]>('/config/providers');

/* ── HPC ─────────────────────────────────────────── */

export interface HpcJob {
  id: string;
  name: string;
  status: string;
  submittedAt: string;
  nodes?: number;
  cores?: number;
}

export const getHpcJobs = () => api.get<HpcJob[]>('/hpc/jobs');
export const submitHpcJob = (params: Record<string, unknown>) =>
  api.post<HpcJob>('/hpc/submit', params);

/* ── Auth ────────────────────────────────────────── */

export const login = (apiKey: string) => api.login(apiKey);

/* ── Memory ──────────────────────────────────────── */

export interface Memory {
  id: string;
  content: string;
  createdAt: string;
}

export const getMemories = () => api.get<Memory[]>('/memory');
export const searchMemories = (query: string) =>
  api.post<Memory[]>('/memory/search', { query });

/* ── Events (SSE) ────────────────────────────────── */

export function subscribeEvents(onEvent: (event: unknown) => void): () => void {
  const source = new EventSource(`${getApiBase()}/events`);
  source.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch {
      // ignore parse errors
    }
  };
  return () => source.close();
}
