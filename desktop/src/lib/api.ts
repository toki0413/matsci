/**
 * Thin fetch wrapper used across the desktop UI.
 *
 * Centralises the backend base URL (kept in sync with the Tauri-managed
 * port via api-client), JSON (de)serialisation and auth headers so call
 * sites can stay one-liners. Non-2xx responses reject the promise, which
 * lets callers lean on try/catch / .catch().
 *
 * ponytail: one retry on 5xx with 1s backoff — covers transient remote
 * backend hiccups without adding circuit-breaker complexity.
 * upgrade path: add jitter / circuit-breaker if thundering-herd shows up.
 */

import { getApiBase, getAuthToken } from "./api-client";

async function fetchWithRetry(input: string, init: RequestInit): Promise<Response> {
  const resp = await fetch(input, init);
  if (resp.status >= 500) {
    await new Promise((r) => setTimeout(r, 1000));
    return fetch(input, init);
  }
  return resp;
}

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    // fall back to the raw API key for endpoints that accept it
    const apiKey = localStorage.getItem("huginn:api_key");
    if (apiKey) headers["X-HUGINN-API-KEY"] = apiKey;
  }
  return headers;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const isFormData = options.body instanceof FormData;

  const headers: Record<string, string> = { ...authHeaders() };
  const callerHeaders = options.headers as Record<string, string> | undefined;
  if (callerHeaders) Object.assign(headers, callerHeaders);

  if (isFormData) {
    // the browser must set the multipart boundary itself
    delete headers["Content-Type"];
  } else if (!headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const resp = await fetchWithRetry(`${getApiBase()}${path}`, { ...options, headers });

  if (!resp.ok) {
    let detail = `API error: ${resp.status}`;
    try {
      const body = await resp.json();
      if (body && (body.message || body.detail || body.error)) {
        detail = body.message || body.detail || body.error;
      }
    } catch {
      // non-JSON error body — stick with the status text
    }
    throw new Error(detail);
  }

  const contentType = resp.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return (await resp.json()) as T;
  if (contentType.startsWith("text/")) return (await resp.text()) as unknown as T;
  // 204 / empty body — resolve to undefined rather than throwing on .json()
  return (await resp.text().catch(() => "")) as unknown as T;
}

export const api = {
  get: <T = unknown>(path: string, options?: RequestInit) =>
    request<T>(path, { method: "GET", ...options }),

  post: <T = unknown>(path: string, body?: unknown, options?: RequestInit) =>
    request<T>(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
      ...options,
    }),

  put: <T = unknown>(path: string, body?: unknown, options?: RequestInit) =>
    request<T>(path, {
      method: "PUT",
      body: body === undefined ? undefined : JSON.stringify(body),
      ...options,
    }),

  patch: <T = unknown>(path: string, body?: unknown, options?: RequestInit) =>
    request<T>(path, {
      method: "PATCH",
      body: body === undefined ? undefined : JSON.stringify(body),
      ...options,
    }),

  del: <T = unknown>(path: string, options?: RequestInit) =>
    request<T>(path, { method: "DELETE", ...options }),

  /** Blob downloads (exports, generated files). */
  getBlob: async (path: string, options?: RequestInit): Promise<Blob> => {
    const resp = await fetchWithRetry(`${getApiBase()}${path}`, {
      ...options,
      headers: {
        ...authHeaders(),
        ...((options?.headers as Record<string, string> | undefined) || {}),
      },
    });
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.blob();
  },

  /** Multipart upload — pass a single File or a pre-built FormData. */
  upload: <T = unknown>(path: string, data: File | FormData, options?: RequestInit) => {
    const form = data instanceof FormData ? data : new FormData();
    if (data instanceof File) form.append("file", data);
    return request<T>(path, { ...options, method: "POST", body: form });
  },
};
