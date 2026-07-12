/**
 * API base URL + auth token state accessors.
 *
 * ponytail: was 419 lines with retry/dedup/token-refresh machinery —
 * none of it was used (api.ts handles all actual fetch calls).
 * Trimmed to the 3 functions that App.tsx / Pet.tsx / CredentialsPanel
 * actually import.
 */

let apiBase: string = 'http://127.0.0.1:8000';
let authToken: string | null = null;

export function setApiBase(base: string): void {
  apiBase = base.replace(/\/$/, '');
}

export function getApiBase(): string {
  return apiBase;
}

export function getAuthToken(): string | null {
  if (!authToken) {
    // ponytail: JWT stored in localStorage — readable by any script on the
    // page (XSS). Acceptable for local single-user desktop app; upgrade to
    // tauri-plugin-stronghold (OS keychain) when multi-user / remote mode lands.
    authToken = localStorage.getItem('huginn:auth_token');
  }
  return authToken;
}
