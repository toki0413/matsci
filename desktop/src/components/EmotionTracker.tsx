import { useState, useEffect, useCallback } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { RefreshCw, Activity } from "lucide-react";

// ---------------------------------------------------------------------------
// Types — mirrors backend EmotionState / EmotionEvent
// ---------------------------------------------------------------------------

interface EmotionState {
  valence: number;
  arousal: number;
  trust: number;
  affection: number;
  fatigue: number;
  loneliness: number;
  interest: number;
  timestamp: string;
  events: EmotionEvent[];
}

interface EmotionEvent {
  timestamp: string;
  source: string;
  type: string;
  deltas: Record<string, number>;
  note: string;
}

interface EmotionResponse {
  success: boolean;
  persona: string;
  state: EmotionState;
  context_prompt: string;
  trajectory: EmotionEvent[];
}

// Seven dimensions with display metadata
const DIMENSIONS = [
  { key: "valence", label: "Valence", desc: "Pleasant ↔ Unpleasant", min: -1, max: 1, baseline: 0 },
  { key: "arousal", label: "Arousal", desc: "Excited ↔ Calm", min: -1, max: 1, baseline: 0.1 },
  { key: "trust", label: "Trust", desc: "Trusting ↔ Suspicious", min: 0, max: 1, baseline: 0.5 },
  { key: "affection", label: "Affection", desc: "Attached ↔ Distant", min: 0, max: 1, baseline: 0.2 },
  { key: "fatigue", label: "Fatigue", desc: "Exhausted ↔ Energised", min: 0, max: 1, baseline: 0 },
  { key: "loneliness", label: "Loneliness", desc: "Lonely ↔ Connected", min: 0, max: 1, baseline: 0 },
  { key: "interest", label: "Interest", desc: "Curious ↔ Bored", min: 0, max: 1, baseline: 0.5 },
] as const;

// Map event type to a display colour class
const EVENT_TYPE_COLORS: Record<string, string> = {
  praise: "text-green-400",
  criticism: "text-red-400",
  task_success: "text-green-300",
  task_failure: "text-red-300",
  greeting: "text-blue-300",
  farewell: "text-purple-300",
  silence: "text-gray-400",
  message: "text-text-secondary",
  manual: "text-yellow-300",
};

// Normalise a dimension value to 0-1 for the radar chart
function normalise(value: number, min: number, max: number): number {
  if (max === min) return 0.5;
  return (value - min) / (max - min);
}

function formatTimeAgo(iso: string): string {
  const then = new Date(iso);
  const now = new Date();
  const diffSec = Math.floor((now.getTime() - then.getTime()) / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86400)}d ago`;
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// ---------------------------------------------------------------------------
// Component props — receives API_BASE from parent
// ---------------------------------------------------------------------------

interface EmotionTrackerProps {
  apiBase: string;
  personaName?: string;
}

// Chart line colours — keep consistent with the 7 dimensions
const LINE_COLORS: string[] = [
  "#a78bfa", // valence — purple
  "#38bdf8", // arousal — cyan
  "#4ade80", // trust — green
  "#fb923c", // affection — orange
  "#f87171", // fatigue — red
  "#94a3b8", // loneliness — slate
  "#facc15", // interest — yellow
];

export default function EmotionTrackerPanel({
  apiBase,
  personaName = "default",
}: EmotionTrackerProps) {
  const [data, setData] = useState<EmotionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchEmotion = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${apiBase}/personas/${personaName}/emotion`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json: EmotionResponse = await res.json();
      if (json.success) {
        setData(json);
      } else {
        setError("Backend returned failure");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch emotion data");
    } finally {
      setLoading(false);
    }
  }, [apiBase, personaName]);

  useEffect(() => {
    fetchEmotion();
  }, [fetchEmotion]);

  // Auto-refresh every 5 seconds when toggled
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchEmotion, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchEmotion]);

  // Build radar chart data — each axis normalised to 0-1
  const radarData = DIMENSIONS.map((dim) => {
    const raw = data?.state[dim.key] ?? dim.baseline;
    const normalised = normalise(raw, dim.min, dim.max);
    return {
      dimension: dim.label,
      value: Math.round(normalised * 100),
      rawValue: raw,
      desc: dim.desc,
    };
  });

  // Build timeline data from trajectory events
  // Convert each event's deltas into a series of points over time
  const timelineEvents = data?.trajectory ?? [];

  // Build a cumulative trajectory for the line chart
  // Start from baseline, apply each event's deltas cumulatively
  type TrajectoryPoint = Record<string, number | string>;
  const cumulativePoints: TrajectoryPoint[] = [];
  const currentVals: Record<string, number> = {};
  for (const dim of DIMENSIONS) {
    currentVals[dim.key] = dim.baseline;
  }

  for (const evt of timelineEvents.slice(-30)) {
    for (const dim of DIMENSIONS) {
      const delta = evt.deltas[dim.key] || 0;
      const min = dim.min;
      const max = dim.max;
      currentVals[dim.key] = Math.max(min, Math.min(max, currentVals[dim.key] + delta));
    }
    cumulativePoints.push({
      time: formatTime(evt.timestamp),
      ...currentVals,
    });
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-accent" />
          <span className="text-sm font-semibold">Emotion Tracker</span>
          <span className="text-xs text-text-muted">— {personaName}</span>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex cursor-pointer items-center gap-1.5">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-border bg-bg-tertiary text-accent"
            />
            <span className="text-xs text-text-secondary">Auto-refresh (5s)</span>
          </label>
          <button
            onClick={fetchEmotion}
            disabled={loading}
            className="btn-secondary flex items-center gap-1.5 px-3 py-1.5 text-xs"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {error && (
          <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
            {error}
          </div>
        )}

        {!data && !error && (
          <div className="flex h-40 items-center justify-center text-sm text-text-muted">
            Loading emotion data…
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {/* Context prompt */}
            <div className="rounded-lg border border-border bg-bg-secondary p-4">
              <h3 className="mb-2 text-xs font-semibold text-text-muted">Current Mood Context</h3>
              <p className="text-sm italic text-text-primary">{data.context_prompt}</p>
            </div>

            {/* Radar chart + dimension breakdown */}
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border border-border bg-bg-secondary p-4">
                <h3 className="mb-3 text-xs font-semibold text-text-muted">
                  7-Dimension Emotion State
                </h3>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={radarData} outerRadius="75%">
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis
                      dataKey="dimension"
                      tick={{ fill: "var(--text-secondary)", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                    />
                    <PolarRadiusAxis
                      domain={[0, 100]}
                      tick={{ fill: "var(--text-muted)", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                      tickCount={5}
                    />
                    <Radar
                      name="Current"
                      dataKey="value"
                      stroke="var(--accent)"
                      fill="var(--accent)"
                      fillOpacity={0.35}
                      strokeWidth={2}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--bg-tertiary)",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        fontFamily: "Arial",
                        fontSize: 20,
                        fontWeight: "bold",
                      }}
                      formatter={(_value: any, _name: any, props: any) => {
                        const raw = props.payload.rawValue;
                        return [raw.toFixed(3), props.payload.desc];
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              <div className="rounded-lg border border-border bg-bg-secondary p-4">
                <h3 className="mb-3 text-xs font-semibold text-text-muted">
                  Dimension Breakdown
                </h3>
                <div className="space-y-3">
                  {DIMENSIONS.map((dim) => {
                    const raw = data.state[dim.key] ?? dim.baseline;
                    const norm = normalise(raw, dim.min, dim.max);
                    const displayPct = Math.round(norm * 100);
                    const baselinePct = Math.round(normalise(dim.baseline, dim.min, dim.max) * 100);
                    const diff = displayPct - baselinePct;
                    return (
                      <div key={dim.key}>
                        <div className="mb-1 flex items-center justify-between">
                          <div>
                            <span className="text-xs font-medium text-text-primary">
                              {dim.label}
                            </span>
                            <span className="ml-2 text-xs text-text-muted">{dim.desc}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono text-text-primary">
                              {raw.toFixed(3)}
                            </span>
                            <span
                              className={`text-xs ${
                                diff > 0
                                  ? "text-green-400"
                                  : diff < 0
                                    ? "text-red-400"
                                    : "text-text-muted"
                              }`}
                            >
                              {diff > 0 ? "▲" : diff < 0 ? "▼" : "—"} {Math.abs(diff)}%
                            </span>
                          </div>
                        </div>
                        <div className="relative h-2 overflow-hidden rounded-full bg-bg-tertiary">
                          {/* Baseline marker */}
                          <div
                            className="absolute top-0 h-full w-px bg-text-muted opacity-50"
                            style={{ left: `${baselinePct}%` }}
                          />
                          {/* Current value bar */}
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${displayPct}%`,
                              background:
                                diff > 0
                                  ? "var(--green-400, #4ade80)"
                                  : diff < 0
                                    ? "var(--red-400, #f87171)"
                                    : "var(--accent, #6366f1)",
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Cumulative trajectory line chart */}
            {cumulativePoints.length > 1 && (
              <div className="rounded-lg border border-border bg-bg-secondary p-4">
                <h3 className="mb-3 text-xs font-semibold text-text-muted">
                  Emotion Trajectory (last {cumulativePoints.length} events)
                </h3>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={cumulativePoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis
                      dataKey="time"
                      tick={{ fill: "var(--text-muted)", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                      angle={-20}
                      textAnchor="end"
                      height={50}
                    />
                    <YAxis
                      domain={[-1, 1]}
                      tick={{ fill: "var(--text-muted)", fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--bg-tertiary)",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        fontFamily: "Arial",
                        fontSize: 20,
                        fontWeight: "bold",
                      }}
                    />
                    <Legend
                      wrapperStyle={{ fontFamily: "Arial", fontSize: 20, fontWeight: "bold" }}
                    />
                    {DIMENSIONS.map((dim, i) => (
                      <Line
                        key={dim.key}
                        type="monotone"
                        dataKey={dim.key}
                        name={dim.label}
                        stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        strokeWidth={1.5}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Event log */}
            <div className="rounded-lg border border-border bg-bg-secondary p-4">
              <h3 className="mb-3 text-xs font-semibold text-text-muted">
                Event Log ({timelineEvents.length})
              </h3>
              <div className="max-h-64 space-y-2 overflow-y-auto">
                {timelineEvents.length === 0 && (
                  <p className="text-xs text-text-muted">No events recorded yet.</p>
                )}
                {[...timelineEvents].reverse().map((evt, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 rounded-md border border-border bg-bg-tertiary p-2"
                  >
                    <div className="flex w-20 flex-shrink-0 flex-col">
                      <span className="text-xs font-mono text-text-muted">
                        {formatTime(evt.timestamp)}
                      </span>
                      <span className="text-xs text-text-muted">
                        {formatTimeAgo(evt.timestamp)}
                      </span>
                    </div>
                    <div className="flex flex-1 flex-col gap-1">
                      <div className="flex items-center gap-2">
                        <span
                          className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                            EVENT_TYPE_COLORS[evt.type] || "text-text-secondary"
                          } bg-bg-secondary`}
                        >
                          {evt.type}
                        </span>
                        <span className="text-xs text-text-muted">from {evt.source}</span>
                      </div>
                      {evt.note && evt.note !== "neutral" && (
                        <span className="text-xs text-text-secondary">{evt.note}</span>
                      )}
                      {Object.keys(evt.deltas).length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(evt.deltas).map(([k, v]) => (
                            <span
                              key={k}
                              className={`text-xs font-mono ${
                                v > 0 ? "text-green-400" : "text-red-400"
                              }`}
                            >
                              {k}: {v > 0 ? "+" : ""}{v.toFixed(3)}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
