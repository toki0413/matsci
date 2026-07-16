import { useState, useEffect } from "react";
import { api } from "../lib/api";

interface StatusBarProps {
  isConnected: boolean;
  status: string;
  wsReconnecting?: boolean;
  wsFailed?: boolean;
  cwd?: string;
}

interface Metrics {
  tokens: number;
  model: string;
  provider: string;
  tps: number | null;
  ttft: number | null;
}

function parsePrometheusValue(text: string, metricName: string): number {
  const re = new RegExp(`^${metricName}\\s+([\\d.]+)`, "m");
  const m = text.match(re);
  return m ? parseFloat(m[1]) : 0;
}

function parseHistogramAvg(text: string, metricName: string): number | null {
  const sumRe = new RegExp(`^${metricName}_sum(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)`, "gm");
  const countRe = new RegExp(`^${metricName}_count(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)`, "gm");
  let sum = 0, count = 0;
  let m: RegExpExecArray | null;
  while ((m = sumRe.exec(text)) !== null) sum += parseFloat(m[1]);
  while ((m = countRe.exec(text)) !== null) count += parseFloat(m[1]);
  return count > 0 ? sum / count : null;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

export function StatusBar({ isConnected, wsReconnecting, wsFailed, cwd }: StatusBarProps) {
  const [metrics, setMetrics] = useState<Metrics>({
    tokens: 0,
    model: "",
    provider: "",
    tps: null,
    ttft: null,
  });
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const [metricsText, health] = await Promise.all([
          api.get<string>("/metrics").catch(() => ""),
          api.get<any>("/health").catch(() => null),
        ]);
        if (cancelled) return;
        setMetrics({
          tokens: parsePrometheusValue(metricsText, "huginn_llm_tokens_total"),
          model: health?.model || health?.config?.model || "",
          provider: health?.provider || health?.config?.provider || "",
          tps: parseHistogramAvg(metricsText, "huginn_llm_tps"),
          ttft: parseHistogramAvg(metricsText, "huginn_llm_ttft_seconds"),
        });
      } catch {
        // backend might not be up yet
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const connectionStatus = wsFailed 
    ? { color: 'bg-error', text: 'Backend stopped', icon: '🔴' }
    : wsReconnecting 
      ? { color: 'bg-warning animate-pulse', text: 'Reconnecting…', icon: '🟡' }
      : isConnected 
        ? { color: 'bg-success', text: 'Connected', icon: '🟢' }
        : { color: 'bg-error', text: 'Offline', icon: '🔴' };

  return (
    <div className="flex items-center justify-between border-t border-border bg-bg-secondary px-4 py-1 text-[11px]">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 rounded-full ${connectionStatus.color}`} />
          <span className="text-text-muted">{connectionStatus.text}</span>
        </div>
        
        {metrics.model && (
          <div className="flex items-center gap-1.5" title={`${metrics.provider}/${metrics.model}`}>
            <span>🤖</span>
            <span className="text-text-secondary">
              {metrics.provider && `${metrics.provider}/`}
              {metrics.model}
            </span>
          </div>
        )}

        {metrics.tokens > 0 && (
          <div className="flex items-center gap-1.5" title={`Total tokens: ${metrics.tokens.toLocaleString()}`}>
            <span>📊</span>
            <span className="text-text-muted">{formatTokens(metrics.tokens)} tokens</span>
          </div>
        )}

        {metrics.tps != null && (
          <div className="flex items-center gap-1.5" title="Avg streaming tokens/sec">
            <span>⚡</span>
            <span className="text-text-muted">{metrics.tps.toFixed(1)} tok/s</span>
          </div>
        )}

        {metrics.ttft != null && (
          <div className="flex items-center gap-1.5" title="Avg time-to-first-token">
            <span>⏱</span>
            <span className="text-text-muted">{(metrics.ttft * 1000).toFixed(0)}ms TTFT</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        {cwd && (
          <div className="flex items-center gap-1.5" title={cwd}>
            <span>📁</span>
            <span className="text-text-muted truncate max-w-[200px]">
              {cwd.length > 50 ? `…${cwd.slice(-47)}` : cwd}
            </span>
          </div>
        )}

        <div className="text-text-muted">
          {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </div>
      </div>
    </div>
  );
}