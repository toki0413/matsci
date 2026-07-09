/**
 * Thin status bar that polls /metrics (Prometheus) and /health every 5s
 * and shows token count, active connections, and the current model.
 */
import { useState, useEffect } from "react";
import { api } from "../lib/api";

interface MetricsState {
  tokens: number;
  activeConnections: number;
  model: string;
  provider: string;
}

// Parse a single numeric value out of a Prometheus line like:
//   huginn_llm_tokens_total 12345
function parsePrometheusValue(text: string, metricName: string): number {
  const re = new RegExp(`^${metricName}\\s+([\\d.]+)`, "m");
  const m = text.match(re);
  return m ? parseFloat(m[1]) : 0;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

export function MetricsBar() {
  const [metrics, setMetrics] = useState<MetricsState>({
    tokens: 0,
    activeConnections: 0,
    model: "",
    provider: "",
  });
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const [metricsText, health] = await Promise.all([
          api.get<string>("/metrics").catch(() => ""),
          api.get<any>("/health").catch(() => null),
        ]);

        if (cancelled) return;

        const next: MetricsState = {
          tokens: parsePrometheusValue(metricsText, "huginn_llm_tokens_total"),
          activeConnections: parsePrometheusValue(metricsText, "huginn_active_websocket_connections"),
          model: health?.model || health?.config?.model || "",
          provider: health?.provider || health?.config?.provider || "",
        };

        setMetrics(next);
        setVisible(true);
      } catch {
        // backend might not be up yet — keep the bar hidden
      }
    };

    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!visible) return null;

  return (
    <div className="flex items-center gap-4 border-b border-border bg-bg-tertiary/50 px-6 py-1 text-[11px] text-text-muted">
      {metrics.model && (
        <span className="flex items-center gap-1">
          <span>🤖</span>
          <span className="font-medium text-text-secondary">
            {metrics.provider && `${metrics.provider}/`}
            {metrics.model}
          </span>
        </span>
      )}
      {metrics.tokens > 0 && (
        <span className="flex items-center gap-1">
          <span>📊</span>
          <span>{formatTokens(metrics.tokens)} tokens</span>
        </span>
      )}
      <span className="flex items-center gap-1">
        <span>🔌</span>
        <span>{metrics.activeConnections} active</span>
      </span>
    </div>
  );
}
