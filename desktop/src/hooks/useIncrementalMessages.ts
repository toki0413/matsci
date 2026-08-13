/**
 * useIncrementalMessages — block-level session view for the incremental
 * frontend engine (T-BCSE-05/06).
 *
 * The backend serves a block model via ``GET /threads/{id}/events?after=<seq>``
 * (see huginn/routes/threads.py + huginn/events/projection.py). Each block is
 * ``{kind, text, frozen, rev, seq}``. A ``frozen`` block is final and must not
 * re-render; compaction/branch summaries arrive as their own divider blocks so
 * history is never dropped.
 *
 * This hook is intentionally *additive*: it does not touch the existing
 * streaming/optimistic-update path in useChatAndConnection. It only provides
 * (a) pure block→Message conversion, and (b) an incremental fetch that keeps a
 * per-thread ``next_seq`` cursor. Callers may use it to hydrate persisted
 * history (including compaction dividers) on thread switch / reconnect.
 */

import { useCallback, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Message } from "./useChatAndConnection";

// ── Block model (mirrors huginn.events.projection.UiBlock) ─────────
export type UiBlockKind = "text" | "compaction" | "tool";

export interface UiBlock {
  kind: UiBlockKind;
  text: string;
  frozen: boolean;
  rev: number;
  seq: number;
}

export interface ThreadEventsResponse {
  thread_id: string;
  blocks: UiBlock[];
  next_seq: number;
  leaf_id: string | null;
}

// ── Pure conversion: blocks → renderable Messages ──────────────────
// A `compaction` block becomes an `isCompacted` divider Message (rendered by
// ChatPanel as `── Context compressed ──`). A `text`/`tool` block becomes an
// assistant message. Pure and side-effect free so it is unit-testable.
export function blocksToMessages(blocks: UiBlock[]): Message[] {
  const out: Message[] = [];
  for (const b of blocks) {
    if (b.kind === "compaction") {
      out.push({
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        isCompacted: true,
        compactBefore: undefined,
        compactAfter: undefined,
      });
    } else {
      out.push({
        role: "assistant",
        content: b.text || "",
        timestamp: new Date().toISOString(),
      });
    }
  }
  return out;
}

// Extract only the compaction divider blocks already present in `blocks`.
export function compactionBlocks(blocks: UiBlock[]): UiBlock[] {
  return blocks.filter((b) => b.kind === "compaction");
}

// ── Hook ───────────────────────────────────────────────────────────
export function useIncrementalMessages() {
  // Per-thread incremental cursor (last seq we've applied).
  const cursorsRef = useRef<Record<string, number>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  const fetchEvents = useCallback(
    async (threadId: string, after?: number): Promise<ThreadEventsResponse> => {
      setLoading((prev) => ({ ...prev, [threadId]: true }));
      try {
        const params = new URLSearchParams();
        if (after !== undefined && after >= 0) {
          params.set("after", String(after));
        }
        const qs = params.toString();
        const data = await api.get<ThreadEventsResponse>(
          `/threads/${threadId}/events${qs ? `?${qs}` : ""}`
        );
        // advance the cursor to the backend's latest seq
        cursorsRef.current[threadId] = data.next_seq;
        return data;
      } finally {
        setLoading((prev) => ({ ...prev, [threadId]: false }));
      }
    },
    []
  );

  /** Fetch the full block model for a thread (for hydration). */
  const fetchThreadBlocks = useCallback(
    async (threadId: string): Promise<UiBlock[]> => {
      const data = await fetchEvents(threadId);
      return data.blocks || [];
    },
    [fetchEvents]
  );

  /** Fetch only blocks newer than the stored cursor (incremental). */
  const fetchIncremental = useCallback(
    async (threadId: string): Promise<UiBlock[]> => {
      const after = cursorsRef.current[threadId];
      const data = await fetchEvents(threadId, after ?? -1);
      return data.blocks || [];
    },
    [fetchEvents]
  );

  return {
    loading,
    fetchEvents,
    fetchThreadBlocks,
    fetchIncremental,
    blocksToMessages,
    compactionBlocks,
  };
}