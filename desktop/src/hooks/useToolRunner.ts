import { useState, useCallback, useRef, useEffect } from "react";
import { api } from "../lib/api";

// ponytail: collapses the run/setRunning/setError/finally boilerplate that
// bench, evolve, explore (and the other panel runners) all duplicated.
// setResult/setRunning are exposed because the explore WebSocket handler
// pushes exploration_result events from outside the normal HTTP run path;
// ceiling: any panel that needs richer external updates grows its own state.

interface ToolRunnerOptions<T> {
  endpoint: string;
  buildPayload: () => Record<string, any>;
  extractResult: (data: any) => T;
  // Default: data.success === true. Override for panels using a different check.
  isSuccess?: (data: any) => boolean;
  defaultError?: string;
  // Optional pre-run guard (e.g. skip when objective is empty)
  inputGuard?: () => boolean;
}

interface ToolRunnerState<T> {
  running: boolean;
  result: T | null;
  error: string;
  run: () => Promise<void>;
  reset: () => void;
  // out-of-band updates (e.g. WebSocket exploration_result)
  setResult: (r: T | null) => void;
  setRunning: (r: boolean) => void;
}

export function useToolRunner<T>(opts: ToolRunnerOptions<T>): ToolRunnerState<T> {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<T | null>(null);
  const [error, setError] = useState("");
  const mountedRef = useRef(true);

  useEffect(() => () => { mountedRef.current = false; }, []);

  const run = useCallback(async () => {
    if (opts.inputGuard && !opts.inputGuard()) return;
    setRunning(true);
    setError("");
    setResult(null);
    try {
      const data = await api.post<any>(opts.endpoint, opts.buildPayload());
      const success = opts.isSuccess ? opts.isSuccess(data) : data.success === true;
      if (success) {
        setResult(opts.extractResult(data));
      } else {
        setError(data.error || opts.defaultError || "Request failed.");
      }
    } catch (e: any) {
      setError(e.message || "Network error");
    } finally {
      if (mountedRef.current) setRunning(false);
    }
  }, [opts.endpoint, opts.buildPayload, opts.extractResult, opts.isSuccess, opts.defaultError, opts.inputGuard]);

  const reset = useCallback(() => {
    setResult(null);
    setError("");
  }, []);

  return { running, result, error, run, reset, setResult, setRunning };
}
