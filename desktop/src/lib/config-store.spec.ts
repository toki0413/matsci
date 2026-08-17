import { describe, it, expect, beforeEach } from 'vitest';
import { loadStoredConfig, saveStoredConfig } from './config-store';
import type { AppConfig } from '../types/domain';

// Keep in sync with CONFIG_KEY in config-store.ts (private by design —
// don't export internals just for a test).
const CONFIG_KEY = 'huginn:config';
describe('config-store persistence', () => {
  beforeEach(() => localStorage.clear());

  const sample: Partial<AppConfig> = {
    provider: 'deepseek',
    model: 'deepseek-chat',
    api_key: 'sk-test',
    max_concurrent_subagents: 5,
    context_budget_tokens: 50000,
  };

  it('returns defaults when nothing is stored', () => {
    const cfg = loadStoredConfig();
    // a couple of pins to prove we hit the defaults path rather than garbage
    expect(cfg.provider).toBe('openai');
    expect(cfg.pet_name).toBe('Muninn');
  });

  it('round-trips a stored config through localStorage', () => {
    saveStoredConfig(sample as AppConfig);
    const loaded = loadStoredConfig();
    expect(loaded.provider).toBe('deepseek');
    expect(loaded.model).toBe('deepseek-chat');
    expect(loaded.api_key).toBe('sk-test');
    expect(loaded.max_concurrent_subagents).toBe(5);
    expect(loaded.context_budget_tokens).toBe(50000);
  });

  it('falls back to defaults on corrupt JSON', () => {
    localStorage.setItem(CONFIG_KEY, '{not valid json');
    const cfg = loadStoredConfig();
    expect(cfg.provider).toBe('openai');
    expect(cfg.pet_name).toBe('Muninn');
  });
});