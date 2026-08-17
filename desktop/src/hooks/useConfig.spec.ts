import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// ---- mock the api module so no real network happens ---------------------
const apiGet = vi.fn();
const apiPost = vi.fn();
const apiPut = vi.fn();
const apiPatch = vi.fn();
const apiDel = vi.fn();

vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: (...a: unknown[]) => apiPost(...a),
    put: (...a: unknown[]) => apiPut(...a),
    patch: (...a: unknown[]) => apiPatch(...a),
    del: (...a: unknown[]) => apiDel(...a),
  },
}));

// re-import after the mock is installed
const { useConfig } = await import('./useConfig');

beforeEach(() => {
  localStorage.clear();
  apiGet.mockReset();
  apiPost.mockReset();
  apiPut.mockReset();
});

afterEach(() => {
  if (vi.isFakeTimers()) {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  }
});

describe('useConfig', () => {
  it('loads stored config from localStorage on mount', () => {
    localStorage.setItem('huginn:config', JSON.stringify({ provider: 'deepseek', model: 'deepseek-chat' }));
    const { result } = renderHook(() => useConfig());
    expect(result.current.config.provider).toBe('deepseek');
    expect(result.current.config.model).toBe('deepseek-chat');
  });

  it('falls back to defaults when nothing is stored', () => {
    const { result } = renderHook(() => useConfig());
    expect(result.current.config.provider).toBe('openai');
    expect(result.current.config.pet_name).toBe('Muninn');
  });

  it('marks config dirty on model edits', () => {
    localStorage.setItem(
      'huginn:config',
      JSON.stringify({
        provider: 'openai',
        model: 'gpt-4o',
        api_key: '',
        base_url: '',
        ollama_host: '',
        persona: 'default',
        rag_enabled: false,
        models: [{ alias: 'default', provider: 'openai', model: 'gpt-4o', api_key: '', base_url: '', temperature: 0.7, enabled: true }],
        agents: [],
        team_mode_enabled: false,
        max_concurrent_subagents: 3,
        privacy_redact_secrets: true,
        privacy_block_on_secrets: false,
        local_only_mode: false,
        max_tool_output_tokens: 12000,
        context_budget_tokens: 100000,
        pet_name: 'Muninn',
        pet_personality: 'cheerful',
        pet_accessories: [],
        encrypt_config: false,
        encryption_password: '',
        encryption_key_file: '',
        mp_api_key: '',
        oqmd_api_key: '',
        mineru_api_keys: '',
        wecom_token: '',
        extreme_dispatch: false,
        wm_summarize: 'rule',
        wm_token_budget: 8192,
        em_recall_top_k: 5,
        pm_c_min: 0.2,
        wm_summarize_every_n: 5,
        perm_cost_budget_hours: 0,
        perm_trust_adaptive: false,
        perm_auto_approve_all: false,
        perm_plan_mode: false,
        perm_sandbox_mode: false,
      }),
    );
    const { result } = renderHook(() => useConfig());
    act(() => {
      result.current.updateModel(0, { temperature: 0.1 });
    });
    expect(result.current.configDirty).toBe(true);
    expect(result.current.config.models[0].temperature).toBe(0.1);
  });

  it('adds and removes models', () => {
    const { result } = renderHook(() => useConfig());
    // addModel seeds a default model first (ensureDefaultModel), so the first
    // add produces 2 entries: [default, model2]
    act(() => result.current.addModel());
    expect(result.current.config.models.length).toBe(2);
    expect(result.current.config.models[1].alias).toMatch(/model\d+/);

    act(() => result.current.addModel());
    expect(result.current.config.models.length).toBe(3);

    act(() => result.current.removeModel(0));
    expect(result.current.config.models.length).toBe(2);
  });

  it('ensureDefaultModel creates a default model when models is empty', () => {
    const { result } = renderHook(() => useConfig());
    act(() => result.current.updateModel(0, { model: 'x' }));
    // updateModel calls ensureDefaultModel internally -> seeds the default first
    expect(result.current.config.models.length).toBe(1);
    expect(result.current.config.models[0].model).toBe('x');
  });

  it('saveConfig persists to localStorage and pushes to the backend', async () => {
    vi.useFakeTimers();
    apiPost.mockResolvedValue({ success: true });
    localStorage.setItem('huginn:config', JSON.stringify({ provider: 'openai', model: '' }));
    const { result } = renderHook(() => useConfig());

    const next = { ...result.current.config, provider: 'anthropic' };
    let promise: Promise<void>;
    act(() => {
      promise = result.current.saveConfig(next);
    });
    await act(async () => {
      await promise!;
    });
    expect(apiPost).toHaveBeenCalledWith('/config', { ...next, provider: 'anthropic' });
    expect(result.current.config.provider).toBe('anthropic');
    expect(result.current.configDirty).toBe(false);
    expect(localStorage.getItem('huginn:config')).toContain('anthropic');
  });

  it('toggles model expanded state', () => {
    const { result } = renderHook(() => useConfig());
    act(() => result.current.toggleModelExpanded(0));
    expect(result.current.expandedModels.has(0)).toBe(true);
    act(() => result.current.toggleModelExpanded(0));
    expect(result.current.expandedModels.has(0)).toBe(false);
  });

  it('switchPersona posts and updates config + persistence', async () => {
    apiPost.mockResolvedValue({});
    localStorage.setItem('huginn:config', JSON.stringify({ provider: 'openai', model: '' }));
    const { result } = renderHook(() => useConfig());
    await act(async () => {
      await result.current.switchPersona('researcher');
    });
    expect(apiPost).toHaveBeenCalledWith('/personas/researcher/switch', {});
    expect(result.current.config.persona).toBe('researcher');
  });

  it('lazy-loads active model when the models tab is opened', async () => {
    apiGet.mockImplementation((url: string) => {
      if (url === '/config/active-model') return Promise.resolve({ active_alias: 'fast' });
      if (url === '/credentials?kind=llm') return Promise.resolve({ credentials: [] });
      return Promise.resolve({});
    });
    const { result } = renderHook(() => useConfig());
    act(() => result.current.setSettingsTab('models'));
    await waitFor(() => expect(result.current.activeModel).toBe('fast'));
  });
});