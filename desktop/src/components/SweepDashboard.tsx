import { useState, useEffect, useCallback, useMemo } from "react";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Play,
  X,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Table,
  BarChart3,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SweepParam {
  name: string;
  values: number[];
}

interface Job {
  params: Record<string, number>;
  status: "pending" | "running" | "done" | "failed";
  result: any;
  duration: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function cartesianProduct(arrays: number[][]): number[][] {
  return arrays.reduce<number[][]>(
    (acc, curr) => acc.flatMap((a) => curr.map((v) => [...a, v])),
    [[]],
  );
}

function parseValues(raw: string): number[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((n) => !Number.isNaN(n));
}

function generateRange(min: number, max: number, step: number): number[] {
  if (step <= 0 || min > max) return [];
  const vals: number[] = [];
  for (let v = min; v <= max + step * 0.001; v += step) {
    vals.push(parseFloat(v.toFixed(10)));
  }
  return vals;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}m`;
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: Job["status"] }) {
  switch (status) {
    case "pending":
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-border px-2.5 py-0.5 text-xs text-text-muted">
          <Clock size={12} />
          Pending
        </span>
      );
    case "running":
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-accent/15 px-2.5 py-0.5 text-xs text-accent">
          <Loader2 size={12} className="animate-spin" />
          Running
        </span>
      );
    case "done":
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-success/15 px-2.5 py-0.5 text-xs text-success">
          <CheckCircle2 size={12} />
          Done
        </span>
      );
    case "failed":
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-error/15 px-2.5 py-0.5 text-xs text-error">
          <XCircle size={12} />
          Failed
        </span>
      );
  }
}

// ---------------------------------------------------------------------------
// Chart tooltip
// ---------------------------------------------------------------------------

function ChartTooltipContent({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg border border-border bg-bg-secondary px-3 py-2 shadow-lg"
      style={{ fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
    >
      {payload.map((entry, i) => (
        <p key={i} className="text-text-secondary">
          <span className="text-text-primary">{entry.name}:</span>{" "}
          {typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function SweepDashboard({ API_BASE }: { API_BASE: string }) {
  // -- Templates
  const [templates, setTemplates] = useState<string[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState("");

  // -- Sweep parameters
  const [sweepParams, setSweepParams] = useState<SweepParam[]>([]);
  const [newParamName, setNewParamName] = useState("");
  const [newParamMin, setNewParamMin] = useState("");
  const [newParamMax, setNewParamMax] = useState("");
  const [newParamStep, setNewParamStep] = useState("");
  const [newParamExplicit, setNewParamExplicit] = useState("");
  const [inputMode, setInputMode] = useState<"range" | "explicit">("range");

  // -- Jobs
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sweepRunning, setSweepRunning] = useState(false);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  // -- Chart
  const [chartXAxis, setChartXAxis] = useState("");
  const [chartYAxis, setChartYAxis] = useState("");
  const [viewMode, setViewMode] = useState<"chart" | "table">("chart");

  // -- Load templates on mount
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/workflows`)
      .then((r) => r.json())
      .then((data: { templates: string[] }) => {
        if (!cancelled) {
          setTemplates(data.templates ?? []);
          if (data.templates?.length) setSelectedTemplate(data.templates[0]);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [API_BASE]);

  // -- Derived: total combinations
  const totalCombinations = useMemo(() => {
    if (sweepParams.length === 0) return 0;
    return sweepParams.reduce((acc, p) => acc * (p.values.length || 1), 1);
  }, [sweepParams]);

  // -- Add parameter
  const handleAddParam = useCallback(() => {
    const name = newParamName.trim();
    if (!name) return;

    let values: number[];
    if (inputMode === "explicit") {
      values = parseValues(newParamExplicit);
    } else {
      const min = parseFloat(newParamMin);
      const max = parseFloat(newParamMax);
      const step = parseFloat(newParamStep);
      if (Number.isNaN(min) || Number.isNaN(max) || Number.isNaN(step)) return;
      values = generateRange(min, max, step);
    }
    if (values.length === 0) return;

    setSweepParams((prev) => [...prev, { name, values }]);
    setNewParamName("");
    setNewParamMin("");
    setNewParamMax("");
    setNewParamStep("");
    setNewParamExplicit("");
  }, [newParamName, newParamMin, newParamMax, newParamStep, newParamExplicit, inputMode]);

  // -- Remove parameter
  const handleRemoveParam = useCallback((index: number) => {
    setSweepParams((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // -- Summary stats
  const summaryStats = useMemo(() => {
    const done = jobs.filter((j) => j.status === "done");
    const failed = jobs.filter((j) => j.status === "failed");
    const avgTime =
      done.length > 0
        ? done.reduce((s, j) => s + j.duration, 0) / done.length
        : 0;
    return { done: done.length, failed: failed.length, total: jobs.length, avgTime };
  }, [jobs]);

  // -- Submit sweep
  const handleSubmitSweep = useCallback(async () => {
    if (!selectedTemplate || sweepParams.length === 0 || sweepRunning) return;

    const paramNames = sweepParams.map((p) => p.name);
    const paramValueArrays = sweepParams.map((p) => p.values);
    const combos = cartesianProduct(paramValueArrays);

    const initialJobs: Job[] = combos.map((combo) => {
      const params: Record<string, number> = {};
      paramNames.forEach((name, i) => {
        params[name] = combo[i];
      });
      return { params, status: "pending", result: null, duration: 0 };
    });

    setJobs(initialJobs);
    setSweepRunning(true);
    setExpandedRow(null);

    // Set chart axes if not set
    if (!chartXAxis && paramNames.length > 0) setChartXAxis(paramNames[0]);

    // Run jobs sequentially to avoid overwhelming the server
    for (let idx = 0; idx < initialJobs.length; idx++) {
      // Check if cancelled (we rely on sweepRunning state via ref-like pattern)
      setJobs((prev) => {
        const next = [...prev];
        next[idx] = { ...next[idx], status: "running" };
        return next;
      });

      const start = performance.now();
      try {
        const resp = await fetch(`${API_BASE}/workflows/execute`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            template: selectedTemplate,
            args: initialJobs[idx].params,
          }),
        });
        const result = await resp.json();
        const duration = performance.now() - start;

        setJobs((prev) => {
          const next = [...prev];
          next[idx] = {
            ...next[idx],
            status: "done",
            result,
            duration: Math.round(duration),
          };
          return next;
        });
      } catch {
        const duration = performance.now() - start;
        setJobs((prev) => {
          const next = [...prev];
          next[idx] = {
            ...next[idx],
            status: "failed",
            result: null,
            duration: Math.round(duration),
          };
          return next;
        });
      }
    }

    setSweepRunning(false);
  }, [selectedTemplate, sweepParams, sweepRunning, API_BASE, chartXAxis]);

  // -- Cancel sweep
  const handleCancel = useCallback(() => {
    setSweepRunning(false);
    // Mark remaining pending/running jobs as failed
    setJobs((prev) =>
      prev.map((j) =>
        j.status === "pending" || j.status === "running"
          ? { ...j, status: "failed", result: null }
          : j,
      ),
    );
  }, []);

  // -- Chart data
  const chartData = useMemo(() => {
    const doneJobs = jobs.filter((j) => j.status === "done" && j.result);
    return doneJobs.map((j) => {
      const point: Record<string, number> = { ...j.params };
      // Flatten result output into point
      if (j.result?.output && typeof j.result.output === "object") {
        Object.entries(j.result.output).forEach(([k, v]) => {
          if (typeof v === "number") point[k] = v;
        });
      }
      // Also try top-level result fields
      if (typeof j.result === "object") {
        Object.entries(j.result).forEach(([k, v]) => {
          if (typeof v === "number" && !(k in point)) point[k] = v;
        });
      }
      return point;
    });
  }, [jobs]);

  // -- Available Y-axis keys (result metric names from completed jobs)
  const resultMetricKeys = useMemo(() => {
    const keys = new Set<string>();
    const paramNames = new Set(sweepParams.map((p) => p.name));
    chartData.forEach((point) => {
      Object.keys(point).forEach((k) => {
        if (!paramNames.has(k)) keys.add(k);
      });
    });
    return Array.from(keys);
  }, [chartData, sweepParams]);

  const paramNames = useMemo(() => sweepParams.map((p) => p.name), [sweepParams]);
  const isMultiParam = paramNames.length > 1;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex min-h-screen flex-col gap-6 bg-bg-primary p-6 text-text-primary">
      {/* ================================================================== */}
      {/* DEFINE PANEL                                                        */}
      {/* ================================================================== */}
      <section className="rounded-xl border border-border bg-bg-secondary p-5">
        <h2 className="mb-4 text-lg font-semibold tracking-tight text-text-primary">
          Define Sweep
        </h2>

        {/* Template selector */}
        <div className="mb-4">
          <label className="mb-1.5 block text-xs font-medium text-text-secondary">
            Workflow Template
          </label>
          <select
            value={selectedTemplate}
            onChange={(e) => setSelectedTemplate(e.target.value)}
            disabled={sweepRunning}
            className="w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary outline-none transition focus:border-accent disabled:opacity-50"
          >
            {templates.length === 0 && (
              <option value="" disabled>
                No templates loaded
              </option>
            )}
            {templates.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        {/* Parameter input */}
        <div className="mb-4 rounded-lg border border-border bg-bg-primary p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-medium text-text-secondary">Add Parameter</h3>
            <div className="flex rounded-md border border-border text-xs">
              <button
                onClick={() => setInputMode("range")}
                className={`px-2.5 py-1 transition ${
                  inputMode === "range"
                    ? "bg-accent/15 text-accent"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                Range
              </button>
              <button
                onClick={() => setInputMode("explicit")}
                className={`px-2.5 py-1 transition ${
                  inputMode === "explicit"
                    ? "bg-accent/15 text-accent"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                Explicit
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-end gap-3">
            {/* Name */}
            <div className="min-w-[120px] flex-1">
              <label className="mb-1 block text-xs text-text-muted">Name</label>
              <input
                type="text"
                value={newParamName}
                onChange={(e) => setNewParamName(e.target.value)}
                placeholder="e.g. temperature"
                disabled={sweepRunning}
                className="w-full rounded-md border border-border bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-accent disabled:opacity-50"
              />
            </div>

            {inputMode === "range" ? (
              <>
                <div className="w-24">
                  <label className="mb-1 block text-xs text-text-muted">Min</label>
                  <input
                    type="number"
                    value={newParamMin}
                    onChange={(e) => setNewParamMin(e.target.value)}
                    disabled={sweepRunning}
                    className="w-full rounded-md border border-border bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-accent disabled:opacity-50"
                  />
                </div>
                <div className="w-24">
                  <label className="mb-1 block text-xs text-text-muted">Max</label>
                  <input
                    type="number"
                    value={newParamMax}
                    onChange={(e) => setNewParamMax(e.target.value)}
                    disabled={sweepRunning}
                    className="w-full rounded-md border border-border bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-accent disabled:opacity-50"
                  />
                </div>
                <div className="w-24">
                  <label className="mb-1 block text-xs text-text-muted">Step</label>
                  <input
                    type="number"
                    value={newParamStep}
                    onChange={(e) => setNewParamStep(e.target.value)}
                    disabled={sweepRunning}
                    className="w-full rounded-md border border-border bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-accent disabled:opacity-50"
                  />
                </div>
              </>
            ) : (
              <div className="min-w-[200px] flex-1">
                <label className="mb-1 block text-xs text-text-muted">
                  Values (comma-separated)
                </label>
                <input
                  type="text"
                  value={newParamExplicit}
                  onChange={(e) => setNewParamExplicit(e.target.value)}
                  placeholder="100, 200, 300, 500"
                  disabled={sweepRunning}
                  className="w-full rounded-md border border-border bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-accent disabled:opacity-50"
                />
              </div>
            )}

            <button
              onClick={handleAddParam}
              disabled={sweepRunning}
              className="flex items-center gap-1.5 rounded-md border border-border bg-bg-tertiary px-3 py-1.5 text-sm text-text-secondary transition hover:border-accent hover:text-accent disabled:opacity-50"
            >
              <Plus size={14} />
              Add
            </button>
          </div>
        </div>

        {/* Added parameters list */}
        {sweepParams.length > 0 && (
          <div className="mb-4 space-y-2">
            {sweepParams.map((p, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded-lg border border-border bg-bg-tertiary px-3 py-2"
              >
                <div className="text-sm">
                  <span className="font-medium text-text-primary">{p.name}</span>
                  <span className="ml-2 text-text-muted">
                    [{p.values.join(", ")}]
                  </span>
                  <span className="ml-2 text-xs text-text-muted">
                    ({p.values.length} values)
                  </span>
                </div>
                <button
                  onClick={() => handleRemoveParam(i)}
                  disabled={sweepRunning}
                  className="text-text-muted transition hover:text-error disabled:opacity-50"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Summary + actions */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-text-secondary">
            Total combinations:{" "}
            <span className="font-mono font-semibold text-text-primary">
              {totalCombinations}
            </span>
          </p>
          <div className="flex gap-2">
            {sweepRunning && (
              <button
                onClick={handleCancel}
                className="flex items-center gap-1.5 rounded-lg border border-error/30 bg-error/10 px-4 py-2 text-sm font-medium text-error transition hover:bg-error/20"
              >
                <X size={14} />
                Cancel
              </button>
            )}
            <button
              onClick={handleSubmitSweep}
              disabled={
                sweepRunning ||
                !selectedTemplate ||
                sweepParams.length === 0 ||
                totalCombinations === 0
              }
              className="flex items-center gap-1.5 rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-40 disabled:hover:bg-accent"
            >
              <Play size={14} />
              Submit Sweep
            </button>
          </div>
        </div>
      </section>

      {/* ================================================================== */}
      {/* PROGRESS MATRIX                                                     */}
      {/* ================================================================== */}
      {jobs.length > 0 && (
        <section className="rounded-xl border border-border bg-bg-secondary p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold tracking-tight text-text-primary">
              Progress Matrix
            </h2>
            <div className="flex items-center gap-4 text-xs text-text-secondary">
              <span>
                <span className="font-mono font-semibold text-success">
                  {summaryStats.done}
                </span>
                /{summaryStats.total} done
              </span>
              <span>
                <span className="font-mono font-semibold text-error">
                  {summaryStats.failed}
                </span>{" "}
                failed
              </span>
              {summaryStats.avgTime > 0 && (
                <span>
                  avg{" "}
                  <span className="font-mono font-semibold text-text-primary">
                    {formatDuration(Math.round(summaryStats.avgTime))}
                  </span>
                </span>
              )}
            </div>
          </div>

          {/* Progress bar */}
          <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-border">
            <div
              className="h-full rounded-full bg-success transition-all duration-500"
              style={{
                width: `${summaryStats.total > 0 ? (summaryStats.done / summaryStats.total) * 100 : 0}%`,
              }}
            />
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-text-muted">
                  <th className="w-10 py-2 pr-3 font-medium">#</th>
                  {paramNames.map((name) => (
                    <th key={name} className="px-3 py-2 font-medium">
                      {name}
                    </th>
                  ))}
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Result</th>
                  <th className="px-3 py-2 font-medium">Duration</th>
                </tr>
              </thead>
              <tbody className="cv-list">
                {jobs.map((job, idx) => (
                  <>
                    <tr
                      key={idx}
                      onClick={() =>
                        setExpandedRow(expandedRow === idx ? null : idx)
                      }
                      className="cursor-pointer border-b border-border/50 transition hover:bg-bg-tertiary/60"
                    >
                      <td className="py-2.5 pr-3 font-mono text-xs text-text-muted">
                        {expandedRow === idx ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                        {idx + 1}
                      </td>
                      {paramNames.map((name) => (
                        <td
                          key={name}
                          className="px-3 py-2.5 font-mono text-xs text-text-primary"
                        >
                          {job.params[name]}
                        </td>
                      ))}
                      <td className="px-3 py-2.5">
                        <StatusBadge status={job.status} />
                      </td>
                      <td className="max-w-[200px] truncate px-3 py-2.5 text-xs text-text-secondary">
                        {job.status === "done" && job.result
                          ? typeof job.result === "object"
                            ? JSON.stringify(job.result).slice(0, 60) +
                              (JSON.stringify(job.result).length > 60
                                ? "..."
                                : "")
                            : String(job.result).slice(0, 60)
                          : job.status === "failed"
                            ? "—"
                            : "—"}
                      </td>
                      <td className="px-3 py-2.5 font-mono text-xs text-text-secondary">
                        {job.duration > 0 ? formatDuration(job.duration) : "—"}
                      </td>
                    </tr>
                    {/* Expanded detail row */}
                    {expandedRow === idx && job.result && (
                      <tr key={`${idx}-detail`} className="border-b border-border/50">
                        <td
                          colSpan={paramNames.length + 4}
                          className="bg-bg-primary px-4 py-3"
                        >
                          <pre className="max-h-48 overflow-auto rounded-lg bg-bg-tertiary p-3 font-mono text-xs text-text-secondary">
                            {JSON.stringify(job.result, null, 2)}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ================================================================== */}
      {/* RESULTS CHART                                                        */}
      {/* ================================================================== */}
      {jobs.some((j) => j.status === "done") && (
        <section className="rounded-xl border border-border bg-bg-secondary p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold tracking-tight text-text-primary">
              Results
            </h2>
            <div className="flex rounded-md border border-border text-xs">
              <button
                onClick={() => setViewMode("chart")}
                className={`flex items-center gap-1 px-2.5 py-1 transition ${
                  viewMode === "chart"
                    ? "bg-accent/15 text-accent"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                <BarChart3 size={12} />
                Chart
              </button>
              <button
                onClick={() => setViewMode("table")}
                className={`flex items-center gap-1 px-2.5 py-1 transition ${
                  viewMode === "table"
                    ? "bg-accent/15 text-accent"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                <Table size={12} />
                Table
              </button>
            </div>
          </div>

          {/* Axis selectors */}
          <div className="mb-4 flex flex-wrap gap-4">
            <div>
              <label className="mb-1 block text-xs text-text-muted">X-Axis</label>
              <select
                value={chartXAxis}
                onChange={(e) => setChartXAxis(e.target.value)}
                className="rounded-md border border-border bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none focus:border-accent"
              >
                <option value="" disabled>
                  Select...
                </option>
                {paramNames.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-muted">Y-Axis</label>
              <select
                value={chartYAxis}
                onChange={(e) => setChartYAxis(e.target.value)}
                className="rounded-md border border-border bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none focus:border-accent"
              >
                <option value="" disabled>
                  Select...
                </option>
                {resultMetricKeys.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Chart or table */}
          {viewMode === "chart" ? (
            <div className="rounded-lg bg-bg-primary p-4">
              {!chartXAxis || !chartYAxis ? (
                <p className="flex h-64 items-center justify-center text-sm text-text-muted">
                  Select X and Y axes to visualize results
                </p>
              ) : isMultiParam ? (
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid stroke="#d5cfc6" strokeDasharray="3 3" />
                    <XAxis
                      dataKey={chartXAxis}
                      name={chartXAxis}
                      tick={{ fill: "#9a9590", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                      axisLine={{ stroke: "#d5cfc6" }}
                      tickLine={{ stroke: "#d5cfc6" }}
                      label={{
                        value: chartXAxis,
                        position: "bottom",
                        fill: "#6b665f",
                        fontFamily: "Arial",
                        fontSize: 20,
                        fontWeight: "bold",
                        offset: 0,
                      }}
                    />
                    <YAxis
                      dataKey={chartYAxis}
                      name={chartYAxis}
                      tick={{ fill: "#9a9590", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                      axisLine={{ stroke: "#d5cfc6" }}
                      tickLine={{ stroke: "#d5cfc6" }}
                      label={{
                        value: chartYAxis,
                        angle: -90,
                        position: "insideLeft",
                        fill: "#6b665f",
                        fontFamily: "Arial",
                        fontSize: 20,
                        fontWeight: "bold",
                      }}
                    />
                    <Tooltip
                      content={<ChartTooltipContent />}
                      cursor={{ strokeDasharray: "3 3" }}
                    />
                    <Scatter
                      data={chartData}
                      fill="#d4884a"
                      fillOpacity={0.85}
                      strokeWidth={0}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart
                    data={chartData.sort(
                      (a, b) => (a[chartXAxis] ?? 0) - (b[chartXAxis] ?? 0),
                    )}
                    margin={{ top: 10, right: 20, bottom: 20, left: 10 }}
                  >
                    <CartesianGrid stroke="#d5cfc6" strokeDasharray="3 3" />
                    <XAxis
                      dataKey={chartXAxis}
                      tick={{ fill: "#9a9590", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                      axisLine={{ stroke: "#d5cfc6" }}
                      tickLine={{ stroke: "#d5cfc6" }}
                      label={{
                        value: chartXAxis,
                        position: "bottom",
                        fill: "#6b665f",
                        fontFamily: "Arial",
                        fontSize: 20,
                        fontWeight: "bold",
                        offset: 0,
                      }}
                    />
                    <YAxis
                      dataKey={chartYAxis}
                      tick={{ fill: "#9a9590", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                      axisLine={{ stroke: "#d5cfc6" }}
                      tickLine={{ stroke: "#d5cfc6" }}
                      label={{
                        value: chartYAxis,
                        angle: -90,
                        position: "insideLeft",
                        fill: "#6b665f",
                        fontFamily: "Arial",
                        fontSize: 20,
                        fontWeight: "bold",
                      }}
                    />
                    <Tooltip
                      content={<ChartTooltipContent />}
                      cursor={{ stroke: "#9a9590", strokeDasharray: "3 3" }}
                    />
                    <Line
                      type="monotone"
                      dataKey={chartYAxis}
                      stroke="#d4884a"
                      strokeWidth={2}
                      dot={{ fill: "#6b9e8a", r: 4, strokeWidth: 0 }}
                      activeDot={{ fill: "#6b9e8a", r: 6, strokeWidth: 0 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          ) : (
            /* Data table view */
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-border bg-bg-tertiary text-xs text-text-muted">
                    <th className="px-3 py-2 font-medium">#</th>
                    {paramNames.map((n) => (
                      <th key={n} className="px-3 py-2 font-medium">
                        {n}
                      </th>
                    ))}
                    {resultMetricKeys.map((k) => (
                      <th key={k} className="px-3 py-2 font-medium">
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {chartData.map((point, i) => (
                    <tr
                      key={i}
                      className="border-b border-border/50 hover:bg-bg-tertiary/40"
                    >
                      <td className="px-3 py-2 font-mono text-xs text-text-muted">
                        {i + 1}
                      </td>
                      {paramNames.map((n) => (
                        <td
                          key={n}
                          className="px-3 py-2 font-mono text-xs text-text-primary"
                        >
                          {point[n] ?? "—"}
                        </td>
                      ))}
                      {resultMetricKeys.map((k) => (
                        <td
                          key={k}
                          className="px-3 py-2 font-mono text-xs text-text-secondary"
                        >
                          {point[k] != null
                            ? typeof point[k] === "number"
                              ? Number(point[k]).toFixed(4)
                              : String(point[k])
                            : "—"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
