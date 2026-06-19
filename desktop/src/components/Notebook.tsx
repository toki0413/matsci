import { useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NotebookEntry {
  id?: string;
  title: string;
  material: string;
  calc_type: string;
  parameters: Record<string, string>;
  results: string;
  conclusion: string;
  tags: string[];
  created_at?: string;
}

interface MemoryEntry {
  id: string;
  content: string;
  category: string;
  tier: string;
  tags: string[];
  importance: number;
  created_at: string;
  updated_at?: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CALC_TYPES = ['DFT', 'MD', 'FEA', 'CFD', 'ML', 'Other'] as const;

const EMPTY_ENTRY: NotebookEntry = {
  title: '',
  material: '',
  calc_type: 'DFT',
  parameters: {},
  results: '',
  conclusion: '',
  tags: [],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseEntry(mem: MemoryEntry): NotebookEntry {
  try {
    const data = JSON.parse(mem.content) as NotebookEntry;
    return { ...data, id: mem.id, created_at: mem.created_at };
  } catch {
    return {
      id: mem.id,
      title: '(corrupt entry)',
      material: '',
      calc_type: 'Other',
      parameters: {},
      results: '',
      conclusion: '',
      tags: [],
      created_at: mem.created_at,
    };
  }
}

function formatDate(iso: string | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Notebook({ API_BASE }: { API_BASE: string }) {
  // ---- State --------------------------------------------------------------
  const [entries, setEntries] = useState<NotebookEntry[]>([]);
  const [activeEntry, setActiveEntry] = useState<NotebookEntry | null>(null);
  const [editing, setEditing] = useState(false);
  const [isNew, setIsNew] = useState(false);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  // Editor-local draft state (so edits don't mutate the list until save)
  const [draft, setDraft] = useState<NotebookEntry>({ ...EMPTY_ENTRY });

  // ---- Fetch entries ------------------------------------------------------
  const fetchEntries = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/memory?category=notebook&limit=200`,
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const raw: MemoryEntry[] = await res.json();
      const parsed = raw.map(parseEntry);
      // newest first
      parsed.sort((a, b) => {
        const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
        const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
        return tb - ta;
      });
      setEntries(parsed);
    } catch (err) {
      console.error('[Notebook] fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);

  // ---- Search (debounced server-side) ------------------------------------
  useEffect(() => {
    if (!search.trim()) return;
    const id = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/memory/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: search, category: 'notebook' }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const raw: MemoryEntry[] = await res.json();
        const parsed = raw.map(parseEntry);
        parsed.sort((a, b) => {
          const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
          const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
          return tb - ta;
        });
        setEntries(parsed);
      } catch (err) {
        console.error('[Notebook] search error:', err);
      }
    }, 400);
    return () => clearTimeout(id);
  }, [search, API_BASE]);

  // ---- New entry ----------------------------------------------------------
  const handleNewEntry = () => {
    setDraft({ ...EMPTY_ENTRY, created_at: new Date().toISOString() });
    setActiveEntry(null);
    setEditing(true);
    setIsNew(true);
  };

  // ---- Select entry from list --------------------------------------------
  const handleSelectEntry = (entry: NotebookEntry) => {
    if (editing && !window.confirm('Discard unsaved changes?')) return;
    setActiveEntry(entry);
    setDraft({ ...entry });
    setEditing(false);
    setIsNew(false);
  };

  // ---- Save entry --------------------------------------------------------
  const handleSave = async () => {
    setSaving(true);
    try {
      const payload = {
        content: JSON.stringify({
          title: draft.title,
          material: draft.material,
          calc_type: draft.calc_type,
          parameters: draft.parameters,
          results: draft.results,
          conclusion: draft.conclusion,
          tags: draft.tags,
        }),
        category: 'notebook',
        tier: 'mid',
        tags: draft.tags,
        importance: 7,
      };

      const res = await fetch(`${API_BASE}/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await fetchEntries();
      // Refresh the active entry from the latest list
      setActiveEntry(null);
      setEditing(false);
      setIsNew(false);
    } catch (err) {
      console.error('[Notebook] save error:', err);
      alert('Failed to save entry. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  // ---- Delete entry ------------------------------------------------------
  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this entry permanently?')) return;
    try {
      const res = await fetch(`${API_BASE}/memory/${id}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setActiveEntry(null);
      setEditing(false);
      setIsNew(false);
      fetchEntries();
    } catch (err) {
      console.error('[Notebook] delete error:', err);
      alert('Failed to delete entry.');
    }
  };

  // ---- Editor draft helpers ----------------------------------------------
  const setDraftField = <K extends keyof NotebookEntry>(
    field: K,
    value: NotebookEntry[K],
  ) => {
    setDraft((prev) => ({ ...prev, [field]: value }));
  };

  const addParameter = () => {
    setDraft((prev) => ({
      ...prev,
      parameters: { ...prev.parameters, '': '' },
    }));
  };

  const removeParameter = (key: string) => {
    setDraft((prev) => {
      const { [key]: _removed, ...rest } = prev.parameters;
      return { ...prev, parameters: rest };
    });
  };

  const renameParameter = (oldKey: string, newKey: string) => {
    setDraft((prev) => {
      const params: Record<string, string> = {};
      for (const [k, v] of Object.entries(prev.parameters)) {
        params[k === oldKey ? newKey : k] = v;
      }
      return { ...prev, parameters: params };
    });
  };

  const updateParameterValue = (key: string, value: string) => {
    setDraft((prev) => ({
      ...prev,
      parameters: { ...prev.parameters, [key]: value },
    }));
  };

  const handleTagsInput = (value: string) => {
    setDraftField(
      'tags',
      value
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean),
    );
  };

  const startEdit = () => {
    if (activeEntry) setDraft({ ...activeEntry });
    setEditing(true);
    setIsNew(false);
  };

  const cancelEdit = () => {
    if (isNew) {
      setActiveEntry(null);
    } else if (activeEntry) {
      setDraft({ ...activeEntry });
    }
    setEditing(false);
    setIsNew(false);
  };

  // ---- Filtered list (client-side filter on top of search results) -------
  const filteredEntries = entries;

  // ---- Render ------------------------------------------------------------
  return (
    <div className="flex h-full w-full text-text-primary">
      {/* ── Left pane: entry list ────────────────────────────────────────── */}
      <aside className="flex w-72 flex-shrink-0 flex-col border-r border-[#262320] bg-bg-secondary">
        {/* Header */}
        <div className="border-b border-[#262320] p-3 space-y-2">
          <h2 className="text-sm font-semibold tracking-wide text-text-secondary uppercase">
            Notebook
          </h2>

          {/* Search */}
          <div className="relative">
            <svg
              className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-text-muted pointer-events-none"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M21 21l-4.35-4.35M11 19a8 8 0 100-16 8 8 0 000 16z"
              />
            </svg>
            <input
              type="text"
              placeholder="Search entries…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded bg-bg-tertiary pl-8 pr-2 py-1.5 text-xs text-text-primary placeholder-text-muted outline-none border border-transparent focus:border-accent/50 transition"
            />
          </div>

          {/* New entry button */}
          <button
            onClick={handleNewEntry}
            className="w-full flex items-center justify-center gap-1.5 rounded bg-accent/15 hover:bg-accent/25 text-accent text-xs font-medium py-1.5 border border-accent/30 transition"
          >
            <svg
              className="h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Entry
          </button>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto">
          {loading && entries.length === 0 && (
            <p className="p-4 text-center text-xs text-text-muted">Loading…</p>
          )}

          {!loading && filteredEntries.length === 0 && (
            <p className="p-4 text-center text-xs text-text-muted">
              {search ? 'No matches found.' : 'No entries yet.'}
            </p>
          )}

          {filteredEntries.map((entry) => {
            const isActive =
              !isNew && activeEntry?.id === entry.id;
            return (
              <button
                key={entry.id}
                onClick={() => handleSelectEntry(entry)}
                className={`w-full text-left px-3 py-2.5 border-l-2 transition group ${
                  isActive
                    ? 'border-accent bg-bg-tertiary'
                    : 'border-transparent hover:border-accent/40 hover:bg-bg-tertiary/60'
                }`}
              >
                <p className="text-sm font-medium text-text-primary truncate leading-snug">
                  {entry.title || 'Untitled'}
                </p>
                <div className="flex flex-wrap items-center gap-1 mt-1">
                  {entry.material && (
                    <span className="inline-flex items-center rounded bg-success/15 text-success text-[10px] font-medium px-1.5 py-0.5 leading-none">
                      {entry.material}
                    </span>
                  )}
                  {entry.calc_type && (
                    <span className="inline-flex items-center rounded bg-accent/15 text-accent text-[10px] font-medium px-1.5 py-0.5 leading-none">
                      {entry.calc_type}
                    </span>
                  )}
                </div>
                <div className="flex items-center justify-between mt-1.5">
                  <span className="text-[10px] text-text-muted">
                    {formatDate(entry.created_at)}
                  </span>
                  {entry.tags.length > 0 && (
                    <span className="text-[10px] text-text-muted truncate max-w-[8rem]">
                      {entry.tags.slice(0, 2).join(', ')}
                      {entry.tags.length > 2 && '…'}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* ── Right pane: editor / viewer / empty ─────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-hidden bg-bg-primary">
        {/* Empty state */}
        {!activeEntry && !editing && (
          <div className="flex flex-1 flex-col items-center justify-center text-center p-8 gap-4">
            <div className="flex items-center justify-center h-16 w-16 rounded-2xl bg-bg-tertiary border border-[#262320]">
              <svg
                className="h-8 w-8 text-text-muted"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
                />
              </svg>
            </div>
            <div>
              <p className="text-sm text-text-secondary font-medium">
                No notebook entries yet.
              </p>
              <p className="text-xs text-text-muted mt-1">
                Create your first experiment log.
              </p>
            </div>
            <button
              onClick={handleNewEntry}
              className="mt-1 flex items-center gap-1.5 rounded bg-accent hover:bg-accent/80 text-white text-xs font-medium px-4 py-2 transition"
            >
              <svg
                className="h-3.5 w-3.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
              New Entry
            </button>
          </div>
        )}

        {/* ── Editor ─────────────────────────────────────────────────── */}
        {editing && (
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wide">
              {isNew ? 'New Notebook Entry' : 'Edit Notebook Entry'}
            </h3>

            {/* Title */}
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">Title</label>
              <input
                type="text"
                value={draft.title}
                onChange={(e) => setDraftField('title', e.target.value)}
                placeholder="e.g. Band structure of monolayer MoS₂"
                className="w-full rounded bg-bg-secondary px-3 py-2 text-sm text-text-primary placeholder-text-muted outline-none border border-[#262320] focus:border-accent/60 transition"
              />
            </div>

            {/* Material system */}
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">
                Material System
              </label>
              <input
                type="text"
                value={draft.material}
                onChange={(e) => setDraftField('material', e.target.value)}
                placeholder="e.g. Si, GaAs, Fe-Cr alloy"
                className="w-full rounded bg-bg-secondary px-3 py-2 text-sm text-text-primary placeholder-text-muted outline-none border border-[#262320] focus:border-accent/60 transition"
              />
            </div>

            {/* Calculation type */}
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">
                Calculation Type
              </label>
              <select
                value={draft.calc_type}
                onChange={(e) => setDraftField('calc_type', e.target.value)}
                className="w-full rounded bg-bg-secondary px-3 py-2 text-sm text-text-primary outline-none border border-[#262320] focus:border-accent/60 transition appearance-none cursor-pointer"
              >
                {CALC_TYPES.map((ct) => (
                  <option key={ct} value={ct}>
                    {ct}
                  </option>
                ))}
              </select>
            </div>

            {/* Parameters (dynamic key-value) */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-text-secondary">
                  Parameters
                </label>
                <button
                  type="button"
                  onClick={addParameter}
                  className="text-[10px] text-accent hover:text-accent/80 font-medium flex items-center gap-0.5 transition"
                >
                  <svg
                    className="h-3 w-3"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 4v16m8-8H4"
                    />
                  </svg>
                  Add row
                </button>
              </div>

              {Object.entries(draft.parameters).length === 0 && (
                <p className="text-xs text-text-muted italic">
                  No parameters yet.
                </p>
              )}

              {Object.entries(draft.parameters).map(([key, value], idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <input
                    type="text"
                    value={key}
                    onChange={(e) => renameParameter(key, e.target.value)}
                    placeholder="key"
                    className="flex-1 rounded bg-bg-secondary px-2.5 py-1.5 text-xs text-text-primary placeholder-text-muted outline-none border border-[#262320] focus:border-accent/60 transition"
                  />
                  <input
                    type="text"
                    value={value}
                    onChange={(e) =>
                      updateParameterValue(key, e.target.value)
                    }
                    placeholder="value"
                    className="flex-1 rounded bg-bg-secondary px-2.5 py-1.5 text-xs text-text-primary placeholder-text-muted outline-none border border-[#262320] focus:border-accent/60 transition"
                  />
                  <button
                    type="button"
                    onClick={() => removeParameter(key)}
                    className="flex-shrink-0 text-text-muted hover:text-error transition p-1"
                    title="Remove parameter"
                  >
                    <svg
                      className="h-3.5 w-3.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </div>
              ))}
            </div>

            {/* Results */}
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">
                Results{' '}
                <span className="text-text-muted font-normal">
                  (Markdown supported)
                </span>
              </label>
              <textarea
                value={draft.results}
                onChange={(e) => setDraftField('results', e.target.value)}
                placeholder="Paste or type your results here…"
                rows={8}
                className="w-full rounded bg-bg-secondary px-3 py-2 text-sm text-text-primary placeholder-text-muted outline-none border border-[#262320] focus:border-accent/60 transition resize-y font-mono leading-relaxed"
              />
            </div>

            {/* Conclusion */}
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">Conclusion</label>
              <textarea
                value={draft.conclusion}
                onChange={(e) => setDraftField('conclusion', e.target.value)}
                placeholder="Key takeaways or next steps…"
                rows={4}
                className="w-full rounded bg-bg-secondary px-3 py-2 text-sm text-text-primary placeholder-text-muted outline-none border border-[#262320] focus:border-accent/60 transition resize-y"
              />
            </div>

            {/* Tags */}
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">
                Tags{' '}
                <span className="text-text-muted font-normal">
                  (comma-separated)
                </span>
              </label>
              <input
                type="text"
                value={draft.tags.join(', ')}
                onChange={(e) => handleTagsInput(e.target.value)}
                placeholder="e.g. band-structure, spin-orbit, convergence"
                className="w-full rounded bg-bg-secondary px-3 py-2 text-sm text-text-primary placeholder-text-muted outline-none border border-[#262320] focus:border-accent/60 transition"
              />
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-3 pt-2 border-t border-[#262320]">
              <button
                onClick={handleSave}
                disabled={saving || !draft.title.trim()}
                className="flex items-center gap-1.5 rounded bg-accent hover:bg-accent/80 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium px-4 py-2 transition"
              >
                {saving ? (
                  <>
                    <svg
                      className="h-3.5 w-3.5 animate-spin"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth={4}
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                    Saving…
                  </>
                ) : (
                  'Save Entry'
                )}
              </button>
              <button
                onClick={cancelEdit}
                className="rounded bg-bg-tertiary hover:bg-bg-tertiary/80 text-text-secondary text-xs font-medium px-4 py-2 border border-[#262320] transition"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* ── Viewer ─────────────────────────────────────────────────── */}
        {activeEntry && !editing && (
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            {/* Header row */}
            <div className="flex items-start justify-between gap-4">
              <h3 className="text-lg font-semibold text-text-primary leading-snug">
                {activeEntry.title || 'Untitled Entry'}
              </h3>
              <div className="flex items-center gap-2 flex-shrink-0">
                <button
                  onClick={startEdit}
                  className="flex items-center gap-1.5 rounded bg-bg-tertiary hover:bg-bg-tertiary/80 text-text-secondary text-xs font-medium px-3 py-1.5 border border-[#262320] transition"
                >
                  <svg
                    className="h-3.5 w-3.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M16.862 3.487a2.15 2.15 0 113.042 3.042L8.96 17.473l-4.098.924.924-4.098L16.862 3.487z"
                    />
                  </svg>
                  Edit
                </button>
                {activeEntry.id && (
                  <button
                    onClick={() => handleDelete(activeEntry.id as string)}
                    className="flex items-center gap-1.5 rounded bg-error/10 hover:bg-error/20 text-error text-xs font-medium px-3 py-1.5 border border-error/30 transition"
                  >
                    <svg
                      className="h-3.5 w-3.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5-4h4a1 1 0 011 1v0H9V4a1 1 0 011-1z"
                      />
                    </svg>
                    Delete
                  </button>
                )}
              </div>
            </div>

            {/* Metadata badges */}
            <div className="flex flex-wrap items-center gap-2">
              {activeEntry.material && (
                <span className="inline-flex items-center gap-1 rounded-full bg-success/15 text-success text-xs font-medium px-2.5 py-1">
                  <svg
                    className="h-3 w-3"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
                    />
                  </svg>
                  {activeEntry.material}
                </span>
              )}
              {activeEntry.calc_type && (
                <span className="inline-flex items-center gap-1 rounded-full bg-accent/15 text-accent text-xs font-medium px-2.5 py-1">
                  {activeEntry.calc_type}
                </span>
              )}
              {activeEntry.created_at && (
                <span className="inline-flex items-center gap-1 rounded-full bg-bg-tertiary text-text-muted text-xs font-medium px-2.5 py-1">
                  <svg
                    className="h-3 w-3"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  {formatDate(activeEntry.created_at)}
                </span>
              )}
              {activeEntry.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center rounded-full bg-bg-tertiary text-text-secondary text-xs font-medium px-2.5 py-1 border border-[#262320]"
                >
                  #{tag}
                </span>
              ))}
            </div>

            {/* Parameters table */}
            {Object.keys(activeEntry.parameters).length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  Parameters
                </h4>
                <div className="rounded border border-[#262320] overflow-hidden">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-bg-tertiary">
                        <th className="text-left text-text-muted font-medium px-3 py-1.5">
                          Key
                        </th>
                        <th className="text-left text-text-muted font-medium px-3 py-1.5">
                          Value
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(activeEntry.parameters).map(
                        ([k, v], i) => (
                          <tr
                            key={k}
                            className={
                              i % 2 === 0 ? 'bg-bg-secondary' : 'bg-bg-primary'
                            }
                          >
                            <td className="px-3 py-1.5 text-text-secondary font-mono">
                              {k}
                            </td>
                            <td className="px-3 py-1.5 text-text-primary font-mono">
                              {v}
                            </td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Results */}
            {activeEntry.results && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  Results
                </h4>
                <div className="prose prose-invert prose-sm max-w-none rounded bg-bg-secondary border border-[#262320] p-4 text-sm leading-relaxed text-text-primary [&_code]:text-accent [&_pre]:bg-bg-tertiary [&_pre]:rounded [&_pre]:p-3 [&_a]:text-accent [&_a]:underline [&_table]:border-collapse [&_th]:border [&_th]:border-[#262320] [&_th]:px-2 [&_th]:py-1 [&_td]:border [&_td]:border-[#262320] [&_td]:px-2 [&_td]:py-1">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {activeEntry.results}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {/* Conclusion */}
            {activeEntry.conclusion && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  Conclusion
                </h4>
                <div className="prose prose-invert prose-sm max-w-none rounded bg-bg-secondary border border-[#262320] p-4 text-sm leading-relaxed text-text-primary [&_code]:text-accent [&_pre]:bg-bg-tertiary [&_pre]:rounded [&_pre]:p-3 [&_a]:text-accent [&_a]:underline">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {activeEntry.conclusion}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
