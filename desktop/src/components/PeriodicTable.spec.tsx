import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import PeriodicTable from './PeriodicTable';

const apiPost = vi.fn();
vi.mock('../lib/api', () => ({
  api: { get: vi.fn(), post: (...a: unknown[]) => apiPost(...a), put: vi.fn(), patch: vi.fn(), del: vi.fn() },
}));

// cells are <button title="Hydrogen (H) — nonmetal"> so select by title
const cell = (title: string) => screen.getByTitle(new RegExp(`^${title} \\(`));

beforeEach(() => apiPost.mockReset());

describe('PeriodicTable', () => {
  it('starts with the empty detail prompt', () => {
    render(<PeriodicTable API_BASE="/api" />);
    expect(screen.getByText('Select an element')).toBeTruthy();
  });

  it('shows element detail after a single click', () => {
    render(<PeriodicTable API_BASE="/api" />);
    fireEvent.click(cell('Hydrogen'));
    expect(screen.getByRole('heading', { name: 'Hydrogen' })).toBeTruthy();
    expect(screen.getByText('1.008 u')).toBeTruthy();
    expect(screen.getByText(/Query Materials Project — H/)).toBeTruthy();
  });

  it('dims non-matching cells while searching', () => {
    render(<PeriodicTable API_BASE="/api" />);
    fireEvent.change(screen.getByPlaceholderText(/Search name, symbol, or number/), {
      target: { value: 'Iron' },
    });
    // Iron's detail appears once selected; its cell is unmasked (no opacity-20)
    const ironCell = screen.getByTitle(/^Iron \(Fe\)/);
    expect(ironCell.className).not.toContain('opacity-20');
  });

  it('calls the MP endpoint and renders the JSON result', async () => {
    apiPost.mockResolvedValue({ formula: 'Fe', spacegroup: 229 });
    render(<PeriodicTable API_BASE="/api" />);
    fireEvent.click(cell('Iron'));
    fireEvent.click(screen.getByRole('button', { name: /Query Materials Project — Fe/ }));
    // wait for the promise to resolve and re-render
    expect(await screen.findByText(/spacegroup/)).toBeTruthy();
    expect(apiPost).toHaveBeenCalledWith('/tools/materials_database_tool', {
      action: 'mp_summary',
      element: 'Fe',
    });
  });

  it('caps compare-mode selection at 4 elements', () => {
    render(<PeriodicTable API_BASE="/api" />);
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }));
    ['Hydrogen', 'Helium', 'Carbon', 'Iron', 'Oxygen'].forEach((name) => {
      fireEvent.click(cell(name));
    });
    // cap at 4: the grouped detail line and toggle counter both reflect it
    expect(screen.getByText(/Comparing 4 elements?/)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Compare (4/4)' })).toBeTruthy();
  });
});