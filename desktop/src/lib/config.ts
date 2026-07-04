/**
 * Shared configuration between App.tsx and Pet.tsx.
 *
 * Previously both files had their own `let API_BASE` copy, each
 * independently synced via Tauri IPC. This module ensures a single
 * source of truth.
 */

import { getApiBase, setApiBase, syncApiBaseFromTauri } from './api-client';

export { getApiBase, setApiBase, syncApiBaseFromTauri };

/**
 * Sync API base from Tauri and notify all subscribers.
 * Returns the synced base URL.
 */
export async function syncBackendUrl(): Promise<string> {
  await syncApiBaseFromTauri();
  return getApiBase();
}
