import { useState, useMemo, useCallback } from "react";
import elementsData from "../data/elements.json";

/* ────────────────────────────── Types ────────────────────────────── */

interface Element {
  atomic_number: number;
  symbol: string;
  name: string;
  category: string;
  block: string;
  group: number | null;
  period: number;
  row: number;
  col: number;
  atomic_mass: number;
  electron_configuration: string;
  electronegativity: number | null;
  melting_point: number | null;
  boiling_point: number | null;
  density: number | null;
  crystal_structure?: string;
  common_oxidation_states?: number[];
  cpk_color?: string;
  atomic_radius?: number;
}

type ColorMode = "category" | "block";

interface MpResult {
  [key: string]: unknown;
}

const elements = elementsData as Element[];

/* ──────────────────────── Color Palettes ─────────────────────────── */

const CATEGORY_COLORS: Record<string, string> = {
  "alkali-metal":          "#c9544e",
  "alkaline-earth-metal":  "#d4874a",
  "transition-metal":      "#c9a84e",
  "post-transition-metal": "#6b9e8a",
  "metalloid":             "#8a7bb5",
  "nonmetal":              "#6ba874",
  "halogen":               "#5a8ec4",
  "noble-gas":             "#9b6bb5",
  "lanthanide":            "#c47a8a",
  "actinide":              "#c47a6a",
};

const BLOCK_COLORS: Record<string, string> = {
  s: "#c9544e",
  p: "#5a8ec4",
  d: "#c9a84e",
  f: "#9b6bb5",
};

const CATEGORY_LABELS = Object.keys(CATEGORY_COLORS);
const BLOCK_LABELS = Object.keys(BLOCK_COLORS);

/* ──────────────────── Helper: cell background ────────────────────── */

function cellBg(el: Element, mode: ColorMode): string {
  const map = mode === "category" ? CATEGORY_COLORS : BLOCK_COLORS;
  const key = mode === "category" ? el.category : el.block;
  const hex = map[key] ?? "#9a9590";
  // return a muted fill (30 % opacity) so text stays readable on light bg
  return hex + "4d";
}

function cellBorder(el: Element, mode: ColorMode): string {
  const map = mode === "category" ? CATEGORY_COLORS : BLOCK_COLORS;
  const key = mode === "category" ? el.category : el.block;
  return map[key] ?? "#9a9590";
}

/* ──────────────────────── Main Component ─────────────────────────── */

export default function PeriodicTable({ API_BASE }: { API_BASE: string }) {
  const [selectedElements, setSelectedElements] = useState<Set<number>>(new Set());
  const [colorMode, setColorMode] = useState<ColorMode>("category");
  const [search, setSearch] = useState("");
  const [compareMode, setCompareMode] = useState(false);
  const [mpResult, setMpResult] = useState<MpResult | null>(null);
  const [mpLoading, setMpLoading] = useState(false);
  const [mpError, setMpError] = useState<string | null>(null);

  /* ── Search matching ── */
  const searchLower = search.trim().toLowerCase();
  const matchesSearch = useCallback(
    (el: Element) => {
      if (!searchLower) return true;
      return (
        el.name.toLowerCase().includes(searchLower) ||
        el.symbol.toLowerCase().includes(searchLower) ||
        String(el.atomic_number) === searchLower
      );
    },
    [searchLower],
  );

  /* ── Selected elements resolved ── */
  const selectedList = useMemo(
    () => elements.filter((el) => selectedElements.has(el.atomic_number)),
    [selectedElements],
  );

  /* ── Click handler ── */
  const handleElementClick = useCallback(
    (el: Element) => {
      if (compareMode) {
        setSelectedElements((prev) => {
          const next = new Set(prev);
          if (next.has(el.atomic_number)) {
            next.delete(el.atomic_number);
          } else if (next.size < 4) {
            next.add(el.atomic_number);
          }
          return next;
        });
      } else {
        setSelectedElements(new Set([el.atomic_number]));
        setMpResult(null);
        setMpError(null);
      }
    },
    [compareMode],
  );

  /* ── Materials Project query ── */
  const queryMp = useCallback(
    async (symbol: string) => {
      setMpLoading(true);
      setMpResult(null);
      setMpError(null);
      try {
        const res = await fetch(`${API_BASE}/tools/materials_database_tool`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "mp_summary", element: symbol }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setMpResult(data);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setMpError(msg);
      } finally {
        setMpLoading(false);
      }
    },
    [API_BASE],
  );

  /* ── Build grid cells (with placeholders for lanthanide/actinide markers) ── */
  const gridCells = useMemo(() => {
    const cells: Array<{ el: Element; isPlaceholder?: boolean }> = [];
    elements.forEach((el) => cells.push({ el }));
    return cells;
  }, []);

  /* ──────────────────────── Render helpers ─────────────────────────── */

  function renderElementCell(el: Element) {
    const isSelected = selectedElements.has(el.atomic_number);
    const isMatch = matchesSearch(el);
    const bg = cellBg(el, colorMode);
    const borderColor = cellBorder(el, colorMode);

    return (
      <button
        key={el.atomic_number}
        onClick={() => handleElementClick(el)}
        className={`
          relative flex flex-col items-center justify-center
          rounded-lg border cursor-pointer
          transition-all duration-150
          hover:border-accent hover:z-10 hover:scale-110
          focus:outline-none focus:ring-2 focus:ring-accent
          ${isSelected ? "ring-2 ring-accent z-10" : ""}
          ${!isMatch ? "opacity-20 pointer-events-none" : ""}
        `}
        style={{
          gridRow: el.row,
          gridColumn: el.col,
          backgroundColor: bg,
          borderColor: isSelected ? "#d4884a" : borderColor + "66",
          minHeight: 44,
          minWidth: 0,
        }}
        title={`${el.name} (${el.symbol}) — ${el.category}`}
      >
        <span className="absolute top-0.5 left-1 text-[8px] leading-none text-text-muted font-mono">
          {el.atomic_number}
        </span>
        <span className="text-sm font-semibold leading-tight text-text-primary mt-1">
          {el.symbol}
        </span>
        <span className="text-[7px] leading-none text-text-secondary truncate w-full text-center px-0.5">
          {el.name}
        </span>
      </button>
    );
  }

  /* ── Detail panel: single element ── */
  function renderDetailSingle(el: Element) {
    return (
      <div className="space-y-3">
        {/* Header */}
        <div className="flex items-start gap-3">
          <div
            className="flex items-center justify-center w-14 h-14 rounded-xl text-2xl font-bold text-text-primary shrink-0"
            style={{ backgroundColor: cellBg(el, colorMode), borderColor: cellBorder(el, colorMode), borderWidth: 1 }}
          >
            {el.symbol}
          </div>
          <div className="min-w-0">
            <h2 className="text-lg font-semibold text-text-primary leading-tight">{el.name}</h2>
            <p className="text-xs text-text-secondary">
              #{el.atomic_number} &middot; {el.category} &middot; {el.block}-block
            </p>
            {el.group !== null && (
              <p className="text-xs text-text-muted">
                Group {el.group}, Period {el.period}
              </p>
            )}
          </div>
        </div>

        {/* Properties grid */}
        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
          <PropertyRow label="Atomic Mass" value={`${el.atomic_mass} u`} />
          <PropertyRow
            label="Electronegativity"
            value={el.electronegativity != null ? String(el.electronegativity) : "—"}
          />
          <PropertyRow
            label="Melting Point"
            value={el.melting_point != null ? `${el.melting_point} °C` : "—"}
          />
          <PropertyRow
            label="Boiling Point"
            value={el.boiling_point != null ? `${el.boiling_point} °C` : "—"}
          />
          <PropertyRow
            label="Density"
            value={el.density != null ? `${el.density} g/cm³` : "—"}
          />
          <PropertyRow
            label="Crystal"
            value={el.crystal_structure ?? "—"}
          />
        </div>

        {/* Electron configuration */}
        <div>
          <span className="text-[10px] uppercase tracking-wider text-text-muted">Electron Config</span>
          <p className="text-xs font-mono text-text-primary mt-0.5 break-all">
            {el.electron_configuration}
          </p>
        </div>

        {/* Query Materials Project */}
        <button
          onClick={() => queryMp(el.symbol)}
          disabled={mpLoading}
          className="btn-primary w-full text-xs py-1.5"
        >
          {mpLoading ? (
            <span className="inline-flex items-center gap-1.5">
              <Spinner /> Querying…
            </span>
          ) : (
            `Query Materials Project — ${el.symbol}`
          )}
        </button>

        {/* MP result */}
        {mpError && (
          <p className="text-xs text-error">{mpError}</p>
        )}
        {mpResult && (
          <pre className="text-[10px] font-mono text-text-secondary bg-bg-tertiary border border-border rounded-lg p-2 overflow-auto max-h-48 whitespace-pre-wrap">
            {JSON.stringify(mpResult, null, 2)}
          </pre>
        )}
      </div>
    );
  }

  /* ── Detail panel: compare mode ── */
  function renderDetailCompare() {
    if (selectedList.length === 0) {
      return (
        <p className="text-xs text-text-muted text-center py-8">
          Click up to 4 elements to compare them side-by-side.
        </p>
      );
    }
    return (
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-text-primary">
          Comparing {selectedList.length} element{selectedList.length > 1 ? "s" : ""}
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] border-collapse">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-1 pr-2 text-text-muted font-medium">Property</th>
                {selectedList.map((el) => (
                  <th key={el.atomic_number} className="text-center py-1 px-1 text-text-primary font-semibold">
                    <span
                      className="inline-block w-2 h-2 rounded-full mr-1"
                      style={{ backgroundColor: cellBorder(el, colorMode) }}
                    />
                    {el.symbol}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-text-secondary">
              <CompareRow label="Number" values={selectedList.map((el) => String(el.atomic_number))} />
              <CompareRow label="Category" values={selectedList.map((el) => el.category)} />
              <CompareRow label="Block" values={selectedList.map((el) => el.block)} />
              <CompareRow label="Mass" values={selectedList.map((el) => `${el.atomic_mass}`)} />
              <CompareRow
                label="EN"
                values={selectedList.map((el) => (el.electronegativity != null ? String(el.electronegativity) : "—"))}
              />
              <CompareRow
                label="M.P."
                values={selectedList.map((el) => (el.melting_point != null ? `${el.melting_point} °C` : "—"))}
              />
              <CompareRow
                label="B.P."
                values={selectedList.map((el) => (el.boiling_point != null ? `${el.boiling_point} °C` : "—"))}
              />
              <CompareRow
                label="Density"
                values={selectedList.map((el) => (el.density != null ? `${el.density}` : "—"))}
              />
              <CompareRow label="Config" values={selectedList.map((el) => el.electron_configuration)} />
            </tbody>
          </table>
        </div>

        {/* MP query for first selected */}
        {selectedList.length > 0 && (
          <button
            onClick={() => queryMp(selectedList[0].symbol)}
            disabled={mpLoading}
            className="btn-primary w-full text-xs py-1.5"
          >
            {mpLoading ? (
              <span className="inline-flex items-center gap-1.5">
                <Spinner /> Querying…
              </span>
            ) : (
              `Query Materials Project — ${selectedList[0].symbol}`
            )}
          </button>
        )}

        {mpError && <p className="text-xs text-error">{mpError}</p>}
        {mpResult && (
          <pre className="text-[10px] font-mono text-text-secondary bg-bg-tertiary border border-border rounded-lg p-2 overflow-auto max-h-48 whitespace-pre-wrap">
            {JSON.stringify(mpResult, null, 2)}
          </pre>
        )}
      </div>
    );
  }

  /* ── Legend ── */
  function renderLegend() {
    const items = colorMode === "category" ? CATEGORY_LABELS : BLOCK_LABELS;
    const colors = colorMode === "category" ? CATEGORY_COLORS : BLOCK_COLORS;
    return (
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2">
        {items.map((key) => (
          <span key={key} className="inline-flex items-center gap-1 text-[9px] text-text-muted">
            <span
              className="inline-block w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: colors[key] }}
            />
            {key}
          </span>
        ))}
      </div>
    );
  }

  /* ──────────────────────────── Main JSX ──────────────────────────── */

  return (
    <div className="flex h-full w-full gap-3 p-3 overflow-hidden">
      {/* ── Left: Periodic Table ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center gap-2 mb-2 shrink-0 flex-wrap">
          {/* Search */}
          <div className="relative flex-1 min-w-[140px] max-w-xs">
            <SearchIcon className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted" />
            <input
              type="text"
              placeholder="Search name, symbol, or number…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="input pl-7 py-1.5 text-xs"
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                className="absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
              >
                <XIcon className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Color mode toggle */}
          <div className="flex rounded-lg border border-border overflow-hidden text-[10px]">
            <button
              onClick={() => setColorMode("category")}
              className={`px-2.5 py-1.5 transition-colors ${
                colorMode === "category"
                  ? "bg-accent text-white"
                  : "bg-bg-tertiary text-text-secondary hover:text-text-primary"
              }`}
            >
              Category
            </button>
            <button
              onClick={() => setColorMode("block")}
              className={`px-2.5 py-1.5 transition-colors ${
                colorMode === "block"
                  ? "bg-accent text-white"
                  : "bg-bg-tertiary text-text-secondary hover:text-text-primary"
              }`}
            >
              Block
            </button>
          </div>

          {/* Compare toggle */}
          <button
            onClick={() => {
              setCompareMode((prev) => {
                if (prev) {
                  // leaving compare mode — keep only first selected
                  setSelectedElements((s) => {
                    const first = s.values().next().value;
                    return first != null ? new Set([first]) : new Set();
                  });
                }
                return !prev;
              });
              setMpResult(null);
              setMpError(null);
            }}
            className={`text-[10px] px-2.5 py-1.5 rounded-lg border transition-colors ${
              compareMode
                ? "bg-accent/20 border-accent text-accent"
                : "bg-bg-tertiary border-border text-text-secondary hover:text-text-primary"
            }`}
          >
            {compareMode ? `Compare (${selectedElements.size}/4)` : "Compare"}
          </button>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-auto">
          <div
            className="grid gap-[3px] w-full"
            style={{
              gridTemplateColumns: "repeat(18, minmax(0, 1fr))",
              gridTemplateRows: "repeat(7, minmax(44px, auto)) 12px repeat(2, minmax(44px, auto))",
            }}
          >
            {/* Lanthanide / Actinide placeholder markers */}
            <PlaceholderMarker row={6} col={3} label="57-71" />
            <PlaceholderMarker row={7} col={3} label="89-103" />

            {gridCells.map(({ el }) => renderElementCell(el))}
          </div>

          {/* Legend */}
          {renderLegend()}
        </div>
      </div>

      {/* ── Right: Detail / Compare Panel ── */}
      <div className="w-80 shrink-0 overflow-y-auto">
        <div className="bg-bg-secondary border border-border rounded-xl p-4 h-full">
          {compareMode ? (
            renderDetailCompare()
          ) : selectedList.length === 1 ? (
            renderDetailSingle(selectedList[0])
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center py-16">
              <FlaskIcon className="w-10 h-10 text-text-muted mb-3 opacity-40" />
              <p className="text-sm text-text-secondary">Select an element</p>
              <p className="text-xs text-text-muted mt-1">
                Click any element to view its properties
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────── Sub-components ──────────────────────────── */

function PropertyRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <span className="text-text-muted">{label}</span>
      <span className="text-text-primary font-mono">{value}</span>
    </>
  );
}

function CompareRow({ label, values }: { label: string; values: string[] }) {
  return (
    <tr className="border-b border-border/50">
      <td className="py-1 pr-2 text-text-muted font-medium whitespace-nowrap">{label}</td>
      {values.map((v, i) => (
        <td key={i} className="py-1 px-1 text-center font-mono break-all">
          {v}
        </td>
      ))}
    </tr>
  );
}

function PlaceholderMarker({ row, col, label }: { row: number; col: number; label: string }) {
  return (
    <div
      className="flex items-center justify-center text-[7px] text-text-muted rounded border border-dashed border-border/60"
      style={{ gridRow: row, gridColumn: col }}
    >
      {label}
    </div>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ── Tiny inline SVG icons (avoid extra deps) ── */

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function XIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}

function FlaskIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 3h6" />
      <path d="M10 3v6.5L4 19a1 1 0 0 0 .87 1.5h14.26A1 1 0 0 0 20 19l-6-9.5V3" />
      <path d="M7 15h10" />
    </svg>
  );
}
