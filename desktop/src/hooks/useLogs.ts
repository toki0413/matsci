/**
 * useLogs — Backend log listener and filtering.
 *
 * Subscribes to Tauri "backend-log" events and accumulates
 * log entries. Provides auto-scroll ref for the log panel.
 */
import { useState, useRef, useEffect } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { formatTime } from "../lib/constants";
import type { BackendLogEvent } from "../types/domain";

export function useLogs() {
  const [backendLogs, setBackendLogs] = useState<BackendLogEvent[]>([]);
  const [logFilter, setLogFilter] = useState<"all" | "stdout" | "stderr">("all");
  const backendLogEndRef = useRef<HTMLDivElement>(null);

  // Listen to backend stdout/stderr
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    let unlisten: UnlistenFn | undefined;
    (async () => {
      unlisten = await listen("backend-log", (event) => {
        const payload = event.payload as { source: string; text: string };
        const source = payload.source === "stderr" ? "stderr" : "stdout";
        setBackendLogs((prev) => [
          ...prev,
          { source, text: payload.text, time: formatTime() },
        ]);
      });
    })();
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  // Auto-scroll to bottom when logs change
  useEffect(() => {
    backendLogEndRef.current?.scrollIntoView({ behavior: "auto" });
  }, [backendLogs, logFilter]);

  return {
    backendLogs, logFilter, backendLogEndRef,
    setBackendLogs, setLogFilter,
  };
}
