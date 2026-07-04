/**
 * Unified API client for Huginn backend.
 *
 * Replaces the 68 scattered fetch() calls in App.tsx with a single
 * client that handles:
 *   - Base URL management (synced with Tauri IPC)
 *   - Authentication (JWT token, auto-refresh, API key fallback)
 *   - Unified error response parsing ({error_code, message, request_id})
 *   - Request timeout (30s default, AbortController)
 *   - Automatic retry on 5xx (max 2 retries, 1s backoff)
 *   - Request deduplication (in-flight GET dedup)
 *
 * Usage:
 *   import { api } from './lib/api-client';
 *   const tools = await api.get('/tools');
 *   const result = await api.post('/agents/default/chat', { content: 'hi' });
 */

// ── Types ────────────────────────────────────────────────────────

export interface ApiError {
  error_code: string;
  message: string;
  request_id?: string;
  details?: Record<string, unknown>;
  status: number;
}

export interface ApiResponse<T = unknown> {
  ok: boolean;
  data: T | null;
  error: ApiError | null;
}

// ── Config ───────────────────────────────────────────────────────

const DEFAULT_TIMEOUT = 30_000; // 30s
const MAX_RETRIES = 2;
const RETRY_DELAY = 1_000; // 1s
const TOKEN_REFRESH_THRESHOLD = 5 * 60; // refresh if < 5 min left

// ── State ────────────────────────────────────────────────────────

let apiBase: string = 'http://localhost:8000';
let authToken: string | null = null;
let apiKey: string | null = null;
let tokenExpiry: number | null = null;
let isRefreshing = false;
let refreshPromise: Promise<string | null> | null = null;

// In-flight GET request dedup
const inflightGets = new Map<string, Promise<ApiResponse>>();

// ── Token management ─────────────────────────────────────────────

export function setApiBase(base: string): void {
  apiBase = base.replace(/\/$/, ''); // strip trailing slash
}

export function getApiBase(): string {
  return apiBase;
}

export function setAuthToken(token: string | null, expiresIn?: number): void {
  authToken = token;
  if (expiresIn) {
    tokenExpiry = Date.now() + expiresIn * 1000;
  } else {
    tokenExpiry = null;
  }
  if (token) {
    localStorage.setItem('huginn:auth_token', token);
    if (tokenExpiry) localStorage.setItem('huginn:token_expiry', String(tokenExpiry));
  } else {
    localStorage.removeItem('huginn:auth_token');
    localStorage.removeItem('huginn:token_expiry');
  }
}

export function setApiKey(key: string | null): void {
  apiKey = key;
  if (key) {
    localStorage.setItem('huginn:api_key', key);
  } else {
    localStorage.removeItem('huginn:api_key');
  }
}

export function getAuthToken(): string | null {
  if (!authToken) {
    // Restore from localStorage
    const stored = localStorage.getItem('huginn:auth_token');
    if (stored) {
      authToken = stored;
      const expStr = localStorage.getItem('huginn:token_expiry');
      if (expStr) tokenExpiry = Number(expStr);
    }
  }
  if (!apiKey) {
    apiKey = localStorage.getItem('huginn:api_key');
  }
  return authToken;
}

export function isTokenExpired(): boolean {
  if (!tokenExpiry) return false;
  return Date.now() > tokenExpiry - TOKEN_REFRESH_THRESHOLD * 1000;
}

export function clearAuth(): void {
  setAuthToken(null);
  setApiKey(null);
}

function buildHeaders(custom?: Record<string, string>): HeadersInit {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...custom,
  };
  // JWT takes priority over API key
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  } else if (apiKey) {
    headers['X-HUGINN-API-KEY'] = apiKey;
  }
  return headers;
}

// ── Token refresh ────────────────────────────────────────────────

async function ensureToken(): Promise<void> {
  if (!authToken || !isTokenExpired()) return;
  if (isRefreshing && refreshPromise) {
    await refreshPromise;
    return;
  }
  isRefreshing = true;
  refreshPromise = refreshToken();
  try {
    await refreshPromise;
  } finally {
    isRefreshing = false;
    refreshPromise = null;
  }
}

async function refreshToken(): Promise<string | null> {
  try {
    const resp = await fetch(`${apiBase}/auth/refresh`, {
      method: 'POST',
      headers: buildHeaders(),
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) {
      // Token is invalid, clear it
      clearAuth();
      return null;
    }
    const data = await resp.json();
    if (data.token) {
      setAuthToken(data.token, data.expires_in);
      return data.token;
    }
    return null;
  } catch {
    return null;
  }
}

// ── Core request ─────────────────────────────────────────────────

async function request<T = unknown>(
  method: string,
  path: string,
  body?: unknown,
  options?: {
    timeout?: number;
    retry?: boolean;
    isFormData?: boolean;
    headers?: Record<string, string>;
  },
): Promise<ApiResponse<T>> {
  await ensureToken();

  const url = path.startsWith('http') ? path : `${apiBase}${path}`;
  const timeout = options?.timeout ?? DEFAULT_TIMEOUT;
  const shouldRetry = options?.retry ?? true;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  const init: RequestInit = {
    method,
    headers: options?.isFormData
      ? { ...(buildHeaders(options?.headers) as Record<string, string>) }
      : buildHeaders(options?.headers),
    signal: controller.signal,
  };

  if (body !== undefined) {
    if (options?.isFormData && body instanceof FormData) {
      init.body = body;
      // Remove Content-Type so browser sets multipart boundary
      const h = init.headers as Record<string, string>;
      delete h['Content-Type'];
    } else {
      init.body = JSON.stringify(body);
    }
  }

  let lastError: ApiError | null = null;
  const maxAttempts = shouldRetry ? MAX_RETRIES + 1 : 1;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const resp = await fetch(url, init);
      clearTimeout(timeoutId);

      // Parse response
      const contentType = resp.headers.get('content-type') || '';
      let data: unknown = null;
      if (contentType.includes('application/json')) {
        data = await resp.json();
      } else if (contentType.includes('text/')) {
        data = await resp.text();
      }

      if (resp.ok) {
        return { ok: true, data: data as T, error: null };
      }

      // Parse unified error response
      const errorBody = data as Record<string, unknown> | null;
      lastError = {
        error_code: (errorBody?.error_code as string) || 'HTTP_ERROR',
        message: (errorBody?.message as string) || (errorBody?.detail as string) || `HTTP ${resp.status}`,
        request_id: errorBody?.request_id as string | undefined,
        details: errorBody?.details as Record<string, unknown> | undefined,
        status: resp.status,
      };

      // Don't retry on 4xx (except 429)
      if (resp.status >= 400 && resp.status < 500 && resp.status !== 429) break;

      // Don't retry on auth errors — clear token
      if (resp.status === 401) {
        if (lastError.error_code === 'TOKEN_REVOKED') {
          clearAuth();
        }
        break;
      }

      // Retry on 5xx or 429
      if (attempt < maxAttempts - 1) {
        await new Promise((r) => setTimeout(r, RETRY_DELAY * (attempt + 1)));
        continue;
      }
    } catch (err) {
      clearTimeout(timeoutId);
      const isAbort = err instanceof DOMException && err.name === 'AbortError';
      lastError = {
        error_code: isAbort ? 'TIMEOUT' : 'NETWORK_ERROR',
        message: isAbort ? `Request timed out after ${timeout}ms` : (err as Error).message,
        status: 0,
      };
      if (attempt < maxAttempts - 1 && !isAbort) {
        await new Promise((r) => setTimeout(r, RETRY_DELAY * (attempt + 1)));
        continue;
      }
    }
  }

  return { ok: false, data: null, error: lastError };
}

// ── Public API ───────────────────────────────────────────────────

export const api = {
  /** GET request with dedup */
  async get<T = unknown>(
    path: string,
    options?: { timeout?: number; dedup?: boolean },
  ): Promise<ApiResponse<T>> {
    const shouldDedup = options?.dedup ?? true;
    if (shouldDedup) {
      const existing = inflightGets.get(path);
      if (existing) return existing as Promise<ApiResponse<T>>;
    }
    const promise = request<T>('GET', path, undefined, {
      timeout: options?.timeout,
      retry: true,
    });
    if (shouldDedup) {
      inflightGets.set(path, promise);
      promise.finally(() => inflightGets.delete(path));
    }
    return promise;
  },

  /** POST request */
  async post<T = unknown>(
    path: string,
    body?: unknown,
    options?: { timeout?: number },
  ): Promise<ApiResponse<T>> {
    return request<T>('POST', path, body, { timeout: options?.timeout });
  },

  /** PUT request */
  async put<T = unknown>(
    path: string,
    body?: unknown,
    options?: { timeout?: number },
  ): Promise<ApiResponse<T>> {
    return request<T>('PUT', path, body, { timeout: options?.timeout });
  },

  /** PATCH request */
  async patch<T = unknown>(
    path: string,
    body?: unknown,
    options?: { timeout?: number },
  ): Promise<ApiResponse<T>> {
    return request<T>('PATCH', path, body, { timeout: options?.timeout });
  },

  /** DELETE request */
  async delete<T = unknown>(
    path: string,
    options?: { timeout?: number },
  ): Promise<ApiResponse<T>> {
    return request<T>('DELETE', path, undefined, { timeout: options?.timeout });
  },

  /** File upload (multipart/form-data) */
  async upload<T = unknown>(
    path: string,
    file: File | FormData,
    options?: { timeout?: number },
  ): Promise<ApiResponse<T>> {
    const formData = file instanceof FormData ? file : new FormData();
    if (file instanceof File) formData.append('file', file);
    return request<T>('POST', path, formData, {
      timeout: options?.timeout ?? 120_000, // 2 min for uploads
      isFormData: true,
    });
  },

  /** Get the WebSocket URL for the current base */
  getWsUrl(): string {
    return apiBase.replace(/^http/, 'ws') + '/ws/agent';
  },

  /** Login with API key and store JWT */
  async login(key: string): Promise<ApiResponse<{ token: string; expires_in: number; role: string }>> {
    const resp = await this.post<{ token: string; expires_in: number; role: string }>(
      '/auth/login',
      { api_key: key },
    );
    if (resp.ok && resp.data?.token) {
      setAuthToken(resp.data.token, resp.data.expires_in);
      setApiKey(key);
    }
    return resp;
  },

  /** Logout and revoke token */
  async logout(): Promise<void> {
    try {
      await this.post('/auth/logout');
    } catch {
      // Ignore — token will expire anyway
    }
    clearAuth();
  },

  /** Check if authenticated */
  isAuthenticated(): boolean {
    return getAuthToken() !== null || apiKey !== null;
  },
};

// ── Tauri IPC sync ────────────────────────────────────────────────

/**
 * Sync API base URL from Tauri backend (if running in Tauri).
 * Called on app startup.
 */
export async function syncApiBaseFromTauri(): Promise<void> {
  try {
    // @ts-ignore — Tauri IPC
    if (typeof window !== 'undefined' && window.__TAURI__) {
      // @ts-ignore
      const { invoke } = await import('@tauri-apps/api/core');
      const port = await invoke('get_backend_port');
      if (port && typeof port === 'number') {
        setApiBase(`http://localhost:${port}`);
      }
    }
  } catch {
    // Not in Tauri or port not available — keep default
  }
}

// ── Error helper ──────────────────────────────────────────────────

export function formatError(resp: ApiResponse): string {
  if (!resp.error) return 'Unknown error';
  const { error_code, message, request_id } = resp.error;
  let text = message;
  if (error_code !== 'HTTP_ERROR' && error_code !== 'NETWORK_ERROR') {
    text = `[${error_code}] ${message}`;
  }
  if (request_id && request_id !== 'unknown') {
    text += ` (req: ${request_id.slice(0, 8)})`;
  }
  return text;
}
