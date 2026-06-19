import { useState, useRef, useCallback, useEffect } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ExecutionResult {
  stdout: string;
  stderr: string;
  return_value: any;
  execution_time: number;
}

interface HistoryEntry {
  id: number;
  code: string;
  timestamp: Date;
  success: boolean;
  executionTime: number;
}

// ---------------------------------------------------------------------------
// Template snippets
// ---------------------------------------------------------------------------

const TEMPLATES: { label: string; code: string }[] = [
  {
    label: 'Birch-Murnaghan EOS fit',
    code: `import numpy as np
from scipy.optimize import curve_fit

def birch_murnaghan(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta)
    )

V = np.linspace(0.9 * 60, 1.1 * 60, 20)
E = birch_murnaghan(V, -5.0, 60.0, 0.8, 4.0) + np.random.normal(0, 0.01, 20)
popt, _ = curve_fit(birch_murnaghan, V, E, p0=[-5.0, 60.0, 0.8, 4.0])
print(f"Fitted E0={popt[0]:.4f}  V0={popt[1]:.2f}  B0={popt[2]:.4f}  B'={popt[3]:.2f}")
`,
  },
  {
    label: 'Convergence test plot',
    code: `import base64, io, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cutoffs = [300, 400, 500, 600, 700, 800]
energies = [-42.31, -42.87, -43.12, -43.19, -43.21, -43.22]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cutoffs, energies, 'o-', color='#d4884a', linewidth=2, markersize=6)
ax.set_xlabel('Cutoff energy (eV)', fontsize=12)
ax.set_ylabel('Total energy (eV)', fontsize=12)
ax.set_title('Plane-wave convergence test')
ax.grid(True, alpha=0.3)
fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=120)
buf.seek(0)
print('data:image/png;base64,' + base64.b64encode(buf.read()).decode())
plt.close(fig)
`,
  },
  {
    label: 'Lennard-Jones potential curve',
    code: `import numpy as np

def lennard_jones(r, epsilon=1.0, sigma=1.0):
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

r = np.linspace(0.8, 3.0, 50)
V = lennard_jones(r)
r_min = 2 ** (1.0 / 6.0)

print("Lennard-Jones potential curve")
print(f"Equilibrium distance r_min = {r_min:.4f} sigma")
print(f"Well depth epsilon = 1.0")
print()
for ri, vi in zip(r[::5], V[::5]):
    bar = '#' * max(0, int((vi + 2) * 5))
    print(f"  r={ri:.2f}  V={vi:+8.4f}  {bar}")
`,
  },
  {
    label: 'Stress-strain analysis',
    code: `import numpy as np

strains = np.linspace(0, 0.15, 100)
E_modulus = 200.0  # GPa (steel-like)
yield_strain = 0.002
ut_strain = 0.05

stresses = np.where(
    strains < yield_strain,
    E_modulus * strains,
    E_modulus * yield_strain + 50 * np.log(1 + (strains - yield_strain) / 0.01),
)

max_stress = np.max(stresses)
print(f"Young's modulus: {E_modulus} GPa")
print(f"Yield strength: {E_modulus * yield_strain:.1f} GPa")
print(f"Ultimate tensile strength: {max_stress:.1f} GPa")
print(f"Number of data points: {len(strains)}")
print(f"Strain at UTS: {strains[np.argmax(stresses)]:.4f}")
`,
  },
  {
    label: 'Crystal structure info',
    code: `structures = {
    'FCC (Al)': {'a': 4.05, 'atoms': 4, 'pf': 0.74},
    'BCC (Fe)': {'a': 2.87, 'atoms': 2, 'pf': 0.68},
    'HCP (Ti)': {'a': 2.95, 'c': 4.68, 'atoms': 2, 'pf': 0.74},
    'Diamond (Si)': {'a': 5.43, 'atoms': 8, 'pf': 0.34},
}

print(f"{'Structure':<16} {'a (A)':<8} {'Atoms':<7} {'PF':<6} {'Density proxy'}")
print('-' * 56)
for name, p in structures.items():
    c_str = f"c={p['c']:.2f}" if 'c' in p else ''
    rho = p['atoms'] / p['a'] ** 3 * 100
    print(f"{name:<16} {p['a']:<8.2f} {p['atoms']:<7} {p['pf']:<6.2f} {rho:.2f}")
`,
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Try to detect a base64 PNG embedded in stdout text. */
function extractImages(text: string): { cleaned: string; images: string[] } {
  const images: string[] = [];
  const lines = text.split('\n');
  const cleanedLines: string[] = [];

  for (const line of lines) {
    // Full data URI on its own line
    const dataUriMatch = line.match(/^(data:image\/png;base64,[A-Za-z0-9+/=]+)$/);
    if (dataUriMatch) {
      images.push(dataUriMatch[1]);
      continue;
    }
    // Raw base64 that starts with the PNG magic bytes (iVBOR)
    const rawMatch = line.match(/^([A-Za-z0-9+/=]{64,})$/);
    if (rawMatch && rawMatch[1].length > 100) {
      try {
        const decoded = atob(rawMatch[1].slice(0, 8));
        if (decoded.startsWith('\x89PNG') || decoded.includes('PNG')) {
          images.push('data:image/png;base64,' + rawMatch[1]);
          continue;
        }
      } catch {
        // not valid base64 — treat as normal text
      }
    }
    cleanedLines.push(line);
  }

  return { cleaned: cleanedLines.join('\n'), images };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SandboxPanel({ API_BASE }: { API_BASE: string }) {
  const [code, setCode] = useState<string>(TEMPLATES[0].code);
  const [output, setOutput] = useState<ExecutionResult | null>(null);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyOpen, setHistoryOpen] = useState(true);
  const [selectedTemplate, setSelectedTemplate] = useState(0);
  const [templateMenuOpen, setTemplateMenuOpen] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const historyIdRef = useRef(0);
  const templateMenuRef = useRef<HTMLDivElement>(null);

  // Close template dropdown when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (templateMenuRef.current && !templateMenuRef.current.contains(e.target as Node)) {
        setTemplateMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  // ----- Tab indentation -----
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const ta = e.currentTarget;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const value = ta.value;
      const newValue = value.substring(0, start) + '    ' + value.substring(end);
      setCode(newValue);
      // Restore cursor position after React re-render
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 4;
      });
    }
  }, []);

  // ----- Execute code -----
  const runCode = useCallback(async () => {
    if (running || !code.trim()) return;
    setRunning(true);
    setOutput(null);

    try {
      const res = await fetch(`${API_BASE}/sandbox/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });

      if (!res.ok) {
        const errText = await res.text();
        setOutput({
          stdout: '',
          stderr: `HTTP ${res.status}: ${errText}`,
          return_value: null,
          execution_time: 0,
        });
        setHistory((h) => [
          { id: ++historyIdRef.current, code, timestamp: new Date(), success: false, executionTime: 0 },
          ...h,
        ]);
        return;
      }

      const data: ExecutionResult = await res.json();
      setOutput(data);
      setHistory((h) => [
        {
          id: ++historyIdRef.current,
          code,
          timestamp: new Date(),
          success: !data.stderr,
          executionTime: data.execution_time,
        },
        ...h,
      ]);
    } catch (err: any) {
      setOutput({
        stdout: '',
        stderr: `Network error: ${err.message ?? String(err)}`,
        return_value: null,
        execution_time: 0,
      });
      setHistory((h) => [
        { id: ++historyIdRef.current, code, timestamp: new Date(), success: false, executionTime: 0 },
        ...h,
      ]);
    } finally {
      setRunning(false);
    }
  }, [API_BASE, code, running]);

  // ----- Template selection -----
  const selectTemplate = useCallback(
    (idx: number) => {
      setSelectedTemplate(idx);
      setCode(TEMPLATES[idx].code);
      setTemplateMenuOpen(false);
    },
    [],
  );

  // ----- Load history entry -----
  const loadHistory = useCallback((entry: HistoryEntry) => {
    setCode(entry.code);
  }, []);

  // ----- Line numbers -----
  const lineCount = code.split('\n').length;

  // ----- Render output -----
  const renderOutput = () => {
    if (!output) {
      return (
        <div className="flex h-full items-center justify-center text-[var(--text-muted,#706b64)]">
          <p className="text-sm">Run your code to see output here</p>
        </div>
      );
    }

    const { cleaned: stdoutCleaned, images: stdoutImages } = extractImages(output.stdout);
    const { cleaned: stderrCleaned, images: stderrImages } = extractImages(output.stderr);
    const allImages = [...stdoutImages, ...stderrImages];

    return (
      <div className="space-y-3 font-mono text-sm leading-relaxed">
        {/* stdout */}
        {stdoutCleaned && (
          <pre className="whitespace-pre-wrap break-words text-[var(--text-secondary,#a19b94)]">
            {stdoutCleaned}
          </pre>
        )}

        {/* Inline images */}
        {allImages.map((src, i) => (
          <img
            key={i}
            src={src}
            alt={`Output plot ${i + 1}`}
            className="max-w-full rounded border border-[var(--border,#262320)]"
          />
        ))}

        {/* return_value */}
        {output.return_value !== null && output.return_value !== undefined && (
          <div className="rounded border border-[var(--border,#262320)] bg-[var(--bg-tertiary,#282521)] p-3">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[var(--text-muted,#706b64)]">
              Return value
            </span>
            <pre className="whitespace-pre-wrap break-words text-[var(--success,#6b9e8a)]">
              {typeof output.return_value === 'object'
                ? JSON.stringify(output.return_value, null, 2)
                : String(output.return_value)}
            </pre>
          </div>
        )}

        {/* stderr */}
        {stderrCleaned && (
          <pre className="whitespace-pre-wrap break-words text-[var(--error,#d4645a)]">
            {stderrCleaned}
          </pre>
        )}

        {/* Execution time */}
        {output.execution_time > 0 && (
          <p className="text-xs text-[var(--text-muted,#706b64)]">
            Executed in {output.execution_time.toFixed(3)}s
          </p>
        )}
      </div>
    );
  };

  // -----------------------------------------------------------------------
  // JSX
  // -----------------------------------------------------------------------
  return (
    <div className="flex h-full flex-col bg-[var(--bg-primary,#181614)] text-[var(--text-primary,#faf6f1)]">
      {/* Toolbar */}
      <div className="flex items-center gap-3 border-b border-[var(--border,#262320)] px-4 py-2">
        <h2 className="text-sm font-semibold tracking-wide text-[var(--text-secondary,#a19b94)]">
          Python Sandbox
        </h2>

        {/* Template dropdown */}
        <div ref={templateMenuRef} className="relative">
          <button
            type="button"
            onClick={() => setTemplateMenuOpen((v) => !v)}
            className="flex items-center gap-1.5 rounded border border-[var(--border,#262320)] bg-[var(--bg-tertiary,#282521)] px-3 py-1 text-xs text-[var(--text-secondary,#a19b94)] transition hover:bg-[var(--bg-secondary,#1e1b18)] hover:text-[var(--text-primary,#faf6f1)]"
          >
            {TEMPLATES[selectedTemplate].label}
            <svg
              className={`h-3 w-3 transition-transform ${templateMenuOpen ? 'rotate-180' : ''}`}
              viewBox="0 0 12 12"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M3 4.5L6 7.5L9 4.5" />
            </svg>
          </button>

          {templateMenuOpen && (
            <div className="absolute left-0 top-full z-50 mt-1 w-60 overflow-hidden rounded border border-[var(--border,#262320)] bg-[var(--bg-secondary,#1e1b18)] shadow-lg">
              {TEMPLATES.map((t, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => selectTemplate(idx)}
                  className={`block w-full px-3 py-2 text-left text-xs transition hover:bg-[var(--bg-tertiary,#282521)] ${
                    idx === selectedTemplate
                      ? 'text-[var(--accent,#d4884a)]'
                      : 'text-[var(--text-secondary,#a19b94)]'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Run button */}
        <button
          type="button"
          onClick={runCode}
          disabled={running || !code.trim()}
          className="ml-auto flex items-center gap-2 rounded bg-[var(--accent,#d4884a)] px-4 py-1.5 text-xs font-semibold text-[var(--bg-primary,#181614)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? (
            <>
              <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 16 16" fill="none">
                <circle
                  cx="8"
                  cy="8"
                  r="6"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeDasharray="28"
                  strokeDashoffset="8"
                />
              </svg>
              Running...
            </>
          ) : (
            <>
              <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="currentColor">
                <path d="M4 2.5v11l9-5.5L4 2.5z" />
              </svg>
              Run
            </>
          )}
        </button>
      </div>

      {/* Main split */}
      <div className="flex flex-1 overflow-hidden">
        {/* Code editor (55%) */}
        <div className="flex w-[55%] flex-col border-r border-[var(--border,#262320)]">
          <div className="flex flex-1 overflow-auto">
            {/* Line numbers */}
            <div
              aria-hidden
              className="select-none border-r border-[var(--border,#262320)] bg-[var(--bg-secondary,#1e1b18)] px-2 py-3 text-right font-mono text-xs leading-[1.625rem] text-[var(--text-muted,#706b64)]"
            >
              {Array.from({ length: lineCount }, (_, i) => (
                <div key={i}>{i + 1}</div>
              ))}
            </div>

            {/* Textarea */}
            <textarea
              ref={textareaRef}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onKeyDown={handleKeyDown}
              spellCheck={false}
              className="flex-1 resize-none bg-[var(--bg-primary,#181614)] p-3 font-mono text-xs leading-[1.625rem] text-[var(--text-primary,#faf6f1)] caret-[var(--accent,#d4884a)] outline-none placeholder:text-[var(--text-muted,#706b64)]"
              placeholder="# Write your Python code here..."
            />
          </div>
        </div>

        {/* Output panel (45%) */}
        <div className="flex w-[45%] flex-col overflow-hidden">
          {/* Output area */}
          <div className="flex-1 overflow-auto p-4">{renderOutput()}</div>

          {/* History panel (collapsible) */}
          <div className="border-t border-[var(--border,#262320)]">
            <button
              type="button"
              onClick={() => setHistoryOpen((v) => !v)}
              className="flex w-full items-center gap-2 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-[var(--text-muted,#706b64)] transition hover:text-[var(--text-secondary,#a19b94)]"
            >
              <svg
                className={`h-3 w-3 transition-transform ${historyOpen ? 'rotate-0' : '-rotate-90'}`}
                viewBox="0 0 12 12"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M3 4.5L6 7.5L9 4.5" />
              </svg>
              History ({history.length})
            </button>

            {historyOpen && (
              <div className="max-h-44 overflow-auto px-4 pb-3">
                {history.length === 0 ? (
                  <p className="text-xs text-[var(--text-muted,#706b64)]">No executions yet</p>
                ) : (
                  <ul className="space-y-1">
                    {history.map((entry) => {
                      const firstLine = entry.code.trim().split('\n')[0].slice(0, 50);
                      const timeStr = entry.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      });
                      return (
                        <li key={entry.id}>
                          <button
                            type="button"
                            onClick={() => loadHistory(entry)}
                            className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left transition hover:bg-[var(--bg-tertiary,#282521)]"
                          >
                            {/* Success / fail badge */}
                            <span
                              className={`inline-block h-2 w-2 flex-shrink-0 rounded-full ${
                                entry.success
                                  ? 'bg-[var(--success,#6b9e8a)]'
                                  : 'bg-[var(--error,#d4645a)]'
                              }`}
                            />
                            <span className="flex-1 truncate font-mono text-xs text-[var(--text-secondary,#a19b94)]">
                              {firstLine}
                            </span>
                            <span className="flex-shrink-0 text-[10px] text-[var(--text-muted,#706b64)]">
                              {timeStr}
                            </span>
                            {entry.executionTime > 0 && (
                              <span className="flex-shrink-0 text-[10px] text-[var(--text-muted,#706b64)]">
                                {entry.executionTime.toFixed(2)}s
                              </span>
                            )}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
