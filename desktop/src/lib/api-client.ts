/**
 * API base URL + auth token state accessors.
 *
 * ponytail: was 419 lines with retry/dedup/token-refresh machinery —
 * none of it was used (api.ts handles all actual fetch calls).
 * Trimmed to the 3 functions that App.tsx / Pet.tsx / CredentialsPanel
 * actually import.
 */

let apiBase: string = 'http://localhost:8000';
let authToken: string | null = null;
let apiKey: string | null = null;

export function setApiBase(base: string): void {
  apiBase = base.replace(/\/$/, '');
}

export function getApiBase(): string {
  return apiBase;
}

export function getAuthToken(): string | null {
  if (!authToken) {
    authToken = localStorage.getItem('huginn:auth_token');
  }
  if (!apiKey) {
    apiKey = localStorage.getItem('huginn:api_key');
  }
  return authToken;
}
