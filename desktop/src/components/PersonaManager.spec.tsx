import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PersonaManager from './PersonaManager';

// useTranslation returns a tiny key->label map so the rendered text is readable;
// unmapped keys fall back to the key itself. The t reference must be STABLE
// across renders: PersonaManager wraps loadDetail in useCallback([t]), and a
// fresh t every render would retrigger the detail effect into an infinite loop.
const { tStable } = vi.hoisted(() => {
  const MAP: Record<string, string> = {
    'persona.title': 'Personas',
    'persona.create': 'New Persona',
    'persona.setActive': 'Set Active',
    'persona.setDefault': 'Set Default',
    'persona.delete': 'Delete',
    'persona.builtin': 'Builtin',
    'persona.empty': 'No personas yet',
    'persona.selectPrompt': 'Select a persona',
    'persona.systemPrompt': 'System Prompt',
    'persona.createTitle': 'Create Persona',
    'persona.save': 'Save',
    'persona.cancel': 'Cancel',
  };
  return { tStable: (key: string) => MAP[key] ?? key };
});

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: tStable }),
}));

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiPatch = vi.fn();
const apiDel = vi.fn();
vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: (...a: unknown[]) => apiPost(...a),
    patch: (...a: unknown[]) => apiPatch(...a),
    del: (...a: unknown[]) => apiDel(...a),
  },
}));

const LIST = {
  default: 'dft_expert',
  personas: [
    {
      name: 'dft_expert',
      system_prompt: 'You are a DFT expert.',
      description: 'Default DFT expert',
      when_to_use: ['vasp', 'qe'],
    },
    {
      name: 'custom',
      system_prompt: 'A custom prompt.',
      description: 'Made by the user',
      when_to_use: [],
    },
  ],
};

const detailOf = (name: string) => ({
  name,
  system_prompt: name === 'dft_expert' ? 'You are a DFT expert.' : 'A custom prompt.',
  begin_dialogs: [],
  mood_dialogs: [],
});

beforeEach(() => {
  apiGet.mockReset();
  apiPost.mockReset();
  apiPatch.mockReset();
  apiDel.mockReset();
  apiGet.mockImplementation((url: string) => {
    if (url === '/personas') return Promise.resolve(LIST);
    if (url.startsWith('/personas/')) return Promise.resolve(detailOf(url.split('/')[2]));
    return Promise.resolve({});
  });
});

describe('PersonaManager', () => {
  it('loads the list and shows detail for the default persona', async () => {
    render(<PersonaManager />);
    // head detail title comes from the auto-selected default
    expect(await screen.findByRole('heading', { name: 'dft_expert' })).toBeTruthy();
    expect(apiGet).toHaveBeenCalledWith('/personas/dft_expert');
    // both personas are listed on the left (scoped to the card, not the <option>)
    expect(screen.getByRole('button', { name: /custom/ })).toBeTruthy();
  });

  it('shows an error banner when the list fails to load', async () => {
    apiGet.mockRejectedValueOnce(new Error('boom'));
    render(<PersonaManager />);
    expect(await screen.findByText('boom')).toBeTruthy();
  });

  it('switches the active persona', async () => {
    apiPost.mockResolvedValue({ success: true });
    render(<PersonaManager />);
    fireEvent.change(await screen.findByTitle('persona.switchTo'), {
      target: { value: 'custom' },
    });
    await waitFor(() =>
      expect(apiPost).toHaveBeenCalledWith('/personas/custom/switch'),
    );
  });

  it('creates a persona from the modal', async () => {
    apiPost.mockResolvedValue({ success: true });
    render(<PersonaManager />);
    await screen.findByRole('heading', { name: 'dft_expert' });
    fireEvent.click(screen.getByText('New Persona'));
    fireEvent.change(screen.getByPlaceholderText('persona.namePlaceholder'), {
      target: { value: 'physicist' },
    });
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() =>
      expect(apiPost).toHaveBeenCalledWith(
        '/personas',
        expect.objectContaining({
          name: 'physicist',
          when_to_use: [],
        }),
      ),
    );
  });

  it('clone pre-fills the create form with <name>_copy', async () => {
    render(<PersonaManager />);
    // 'custom' shows up in both the left card and the <option> switcher, so
    // scope to the card button (its accessible name is "custom Made by the user")
    fireEvent.click(await screen.findByRole('button', { name: /custom/ }));
    fireEvent.click(await screen.findByText('Clone'));
    const nameInput = screen.getByPlaceholderText('persona.namePlaceholder') as HTMLInputElement;
    expect(nameInput.value).toBe('custom_copy');
  });

  it('sets a persona as default via the patch endpoint', async () => {
    apiPatch.mockResolvedValue({ default: 'custom' });
    render(<PersonaManager />);
    fireEvent.click(await screen.findByRole('button', { name: /custom/ }));
    fireEvent.click(await screen.findByText('Set Default'));
    await waitFor(() =>
      expect(apiPatch).toHaveBeenCalledWith('/personas/custom/default'),
    );
  });
});