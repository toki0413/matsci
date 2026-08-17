import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Notebook from './Notebook';

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiDel = vi.fn();
vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: (...a: unknown[]) => apiPost(...a),
    put: vi.fn(),
    patch: vi.fn(),
    del: (...a: unknown[]) => apiDel(...a),
  },
}));

const ENTRY = {
  id: 'm1',
  content: JSON.stringify({
    title: 'Band structure of MoS2',
    material: 'MoS2',
    calc_type: 'DFT',
    parameters: { kpoints: '12x12' },
    results: '# Results\n**gap** = 1.8 eV',
    conclusion: 'Direct gap confirmed.',
    tags: ['dft', 'band'],
  }),
  category: 'notebook',
  tier: 'mid',
  tags: ['dft', 'band'],
  importance: 7,
  created_at: '2026-01-01T00:00:00Z',
};

beforeEach(() => {
  apiGet.mockReset();
  apiPost.mockReset();
  apiDel.mockReset();
});

describe('Notebook', () => {
  it('shows empty state when there are no entries', async () => {
    apiGet.mockResolvedValue([]);
    render(<Notebook API_BASE="/api" />);
    expect(await screen.findByText('No entries yet.')).toBeTruthy();
  });

  it('fetches the entry list from the memory category endpoint', async () => {
    apiGet.mockResolvedValue([ENTRY]);
    render(<Notebook API_BASE="/api" />);
    expect(apiGet).toHaveBeenCalledWith('/memory?category=notebook&limit=200');
    expect(await screen.findByText('Band structure of MoS2')).toBeTruthy();
  });

  it('selecting an entry shows its viewer with markdown results', async () => {
    apiGet.mockResolvedValue([ENTRY]);
    render(<Notebook API_BASE="/api" />);
    fireEvent.click(await screen.findByText('Band structure of MoS2'));
    expect(await screen.findByRole('heading', { name: 'Band structure of MoS2' })).toBeTruthy();
    // material badge appears in the viewer; markdown-split bold value is rendered
    expect(screen.getAllByText('MoS2').length).toBeGreaterThan(0);
    expect(screen.getByText(/1\.8 eV/)).toBeTruthy();
  });

  it('debounces a search and calls the search endpoint', async () => {
    apiGet.mockResolvedValue([]);
    render(<Notebook API_BASE="/api" />);
    await screen.findByText('No entries yet.');
    apiPost.mockResolvedValue([]);
    fireEvent.change(screen.getByPlaceholderText(/Search entries/), {
      target: { value: 'MoS2' },
    });
    await waitFor(
      () =>
        expect(apiPost).toHaveBeenCalledWith('/memory/search', {
          query: 'MoS2',
          category: 'notebook',
        }),
      { timeout: 1000 },
    );
  });

  it('creating a new entry posts to /memory and reloads the list', async () => {
    apiGet.mockResolvedValue([]);
    render(<Notebook API_BASE="/api" />);
    await screen.findByText('No entries yet.');
    fireEvent.click(screen.getAllByText('New Entry')[0]);
    fireEvent.change(screen.getByPlaceholderText(/Band structure of monolayer/), {
      target: { value: 'New Test Entry' },
    });
    fireEvent.click(screen.getByText('Save Entry'));
    await waitFor(() =>
      expect(apiPost).toHaveBeenCalledWith(
        '/memory',
        expect.objectContaining({ category: 'notebook' }),
      ),
    );
    // initial load + the refresh after save
    expect(apiGet).toHaveBeenCalledTimes(2);
  });

  it('deletes an entry after confirmation', async () => {
    apiGet.mockResolvedValue([ENTRY]);
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    render(<Notebook API_BASE="/api" />);
    fireEvent.click(await screen.findByText('Band structure of MoS2'));
    fireEvent.click(screen.getByText('Delete'));
    await waitFor(() => expect(apiDel).toHaveBeenCalledWith('/memory/m1'));
  });
});