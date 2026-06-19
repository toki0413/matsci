import { useState, useMemo, useCallback } from "react";
import { parsePatch } from "diff";
import {
  Check,
  X,
  CheckCheck,
  XCircle,
  ChevronLeft,
  ChevronRight,
  Columns,
  Rows,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DiffFile {
  path: string;
  status: "modified" | "added" | "deleted";
  diff: string; // unified diff format string
}

interface DiffViewerProps {
  diffs: DiffFile[];
  onAcceptAll?: () => void;
  onRejectAll?: () => void;
  onAcceptFile?: (path: string) => void;
  onRejectFile?: (path: string) => void;
}

type ViewMode = "inline" | "split";

// Internal representation of a processed line for rendering
interface DiffLine {
  type: "add" | "remove" | "context";
  content: string;
  oldLineNo: number | null;
  newLineNo: number | null;
}

// For side-by-side rendering
interface SplitLine {
  left: { lineNo: number | null; content: string; type: "remove" | "context" | "empty" } | null;
  right: { lineNo: number | null; content: string; type: "add" | "context" | "empty" } | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATUS_BADGE: Record<DiffFile["status"], { label: string; cls: string }> = {
  modified: { label: "M", cls: "bg-accent/20 text-accent" },
  added: { label: "A", cls: "bg-success/20 text-success" },
  deleted: { label: "D", cls: "bg-error/20 text-error" },
};

/** Context window size: how many unchanged lines to keep visible around changes */
const CONTEXT_WINDOW = 3;

/**
 * Parse a unified diff string into structured lines using the `diff` package.
 * Returns a flat array of DiffLine objects for inline rendering.
 */
function parseUnifiedLines(diffStr: string): DiffLine[] {
  const parsed = parsePatch(diffStr);
  const result: DiffLine[] = [];

  for (const file of parsed) {
    for (const hunk of file.hunks) {
      let oldNo = hunk.oldStart;
      let newNo = hunk.newStart;

      const lines: string[] = hunk.lines ?? [];
      for (const raw of lines) {
        // Each line from parsePatch starts with '+', '-', or ' ' (context)
        // There can also be '\ No newline at end of file' markers
        const prefix = raw[0];
        const content = raw.slice(1);

        if (prefix === "+") {
          result.push({ type: "add", content, oldLineNo: null, newLineNo: newNo });
          newNo++;
        } else if (prefix === "-") {
          result.push({ type: "remove", content, oldLineNo: oldNo, newLineNo: null });
          oldNo++;
        } else if (prefix === " ") {
          result.push({ type: "context", content, oldLineNo: oldNo, newLineNo: newNo });
          oldNo++;
          newNo++;
        }
        // skip '\ No newline...' and unknown prefixes
      }
    }
  }

  return result;
}

/**
 * Build aligned side-by-side rows from flat diff lines.
 * Groups consecutive removes (left only) and adds (right only) together.
 */
function buildSplitRows(lines: DiffLine[]): SplitLine[] {
  const rows: SplitLine[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.type === "context") {
      rows.push({
        left: { lineNo: line.oldLineNo, content: line.content, type: "context" },
        right: { lineNo: line.newLineNo, content: line.content, type: "context" },
      });
      i++;
    } else {
      // Collect consecutive removes and adds
      const removes: DiffLine[] = [];
      const adds: DiffLine[] = [];

      while (i < lines.length && lines[i].type === "remove") {
        removes.push(lines[i]);
        i++;
      }
      while (i < lines.length && lines[i].type === "add") {
        adds.push(lines[i]);
        i++;
      }

      const maxLen = Math.max(removes.length, adds.length);
      for (let j = 0; j < maxLen; j++) {
        const rem = removes[j];
        const add = adds[j];
        rows.push({
          left: rem
            ? { lineNo: rem.oldLineNo, content: rem.content, type: "remove" }
            : { lineNo: null, content: "", type: "empty" },
          right: add
            ? { lineNo: add.newLineNo, content: add.content, type: "add" }
            : { lineNo: null, content: "", type: "empty" },
        });
      }
    }
  }

  return rows;
}

/**
 * Determine which line indices are "near" a change (within CONTEXT_WINDOW).
 * Lines that are context and NOT near a change should be collapsible.
 */
function computeVisibleIndices(lines: DiffLine[]): Set<number> {
  const visible = new Set<number>();
  const changeIndices: number[] = [];

  lines.forEach((l, i) => {
    if (l.type !== "context") changeIndices.push(i);
  });

  // If there are no changes, show everything
  if (changeIndices.length === 0) {
    lines.forEach((_, i) => visible.add(i));
    return visible;
  }

  for (const ci of changeIndices) {
    const lo = Math.max(0, ci - CONTEXT_WINDOW);
    const hi = Math.min(lines.length - 1, ci + CONTEXT_WINDOW);
    for (let j = lo; j <= hi; j++) visible.add(j);
  }

  return visible;
}

/**
 * Build segments: groups of visible lines and collapsed gaps.
 * Each segment is either { kind: 'lines', indices: number[] } or { kind: 'gap', count: number, beforeIndex: number }.
 */
interface LinesSegment {
  kind: "lines";
  indices: number[];
}
interface GapSegment {
  kind: "gap";
  count: number;
  /** The index of the first visible line after the gap */
  afterIndex: number;
}
type Segment = LinesSegment | GapSegment;

function buildSegments(lines: DiffLine[], visible: Set<number>, expandedGaps: Set<number>): Segment[] {
  const segments: Segment[] = [];
  let currentLineGroup: number[] = [];
  let gapCount = 0;
  let gapStartIndex = -1;
  let gapId = 0;

  const flushLines = () => {
    if (currentLineGroup.length > 0) {
      segments.push({ kind: "lines", indices: [...currentLineGroup] });
      currentLineGroup = [];
    }
  };

  const flushGap = () => {
    if (gapCount > 0) {
      // Find the next visible line index after the gap
      let afterIdx = gapStartIndex;
      while (afterIdx < lines.length && !visible.has(afterIdx)) afterIdx++;
      if (expandedGaps.has(gapId)) {
        // Expanded: include all gap lines as visible
        for (let g = gapStartIndex; g < gapStartIndex + gapCount; g++) {
          currentLineGroup.push(g);
        }
      } else {
        flushLines();
        segments.push({ kind: "gap", count: gapCount, afterIndex: afterIdx });
      }
      gapCount = 0;
      gapId++;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    if (visible.has(i)) {
      flushGap();
      currentLineGroup.push(i);
    } else {
      if (gapCount === 0) {
        gapStartIndex = i;
        flushLines();
      }
      gapCount++;
    }
  }

  flushGap();
  flushLines();

  return segments;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: DiffFile["status"] }) {
  const badge = STATUS_BADGE[status];
  return (
    <span
      className={`inline-flex items-center justify-center w-5 h-5 rounded text-[10px] font-bold font-mono ${badge.cls}`}
    >
      {badge.label}
    </span>
  );
}

function LineNumber({ value, className = "" }: { value: number | null; className?: string }) {
  return (
    <span
      className={`select-none text-text-muted text-right pr-2 inline-block w-12 shrink-0 ${className}`}
    >
      {value ?? ""}
    </span>
  );
}

function InlineDiffLine({ line }: { line: DiffLine }) {
  const bgClass =
    line.type === "add"
      ? "bg-success/15 text-success"
      : line.type === "remove"
        ? "bg-error/15 text-error"
        : "text-text-primary";

  const prefix = line.type === "add" ? "+" : line.type === "remove" ? "-" : " ";

  return (
    <div className={`flex font-mono text-[13px] leading-5 ${bgClass} hover:brightness-110`}>
      <LineNumber value={line.oldLineNo} />
      <LineNumber value={line.newLineNo} />
      <span className="select-none text-text-muted w-4 shrink-0 text-center">{prefix}</span>
      <span className="pl-2 whitespace-pre break-all">{line.content}</span>
    </div>
  );
}

function SplitDiffRow({ row }: { row: SplitLine }) {
  const leftBg =
    row.left?.type === "remove"
      ? "bg-error/15 text-error"
      : row.left?.type === "empty"
        ? "bg-bg-secondary/50"
        : "text-text-primary";

  const rightBg =
    row.right?.type === "add"
      ? "bg-success/15 text-success"
      : row.right?.type === "empty"
        ? "bg-bg-secondary/50"
        : "text-text-primary";

  const leftPrefix = row.left?.type === "remove" ? "-" : row.left?.type === "context" ? " " : "";
  const rightPrefix = row.right?.type === "add" ? "+" : row.right?.type === "context" ? " " : "";

  return (
    <div className="flex">
      {/* Left (old) column */}
      <div className={`flex flex-1 min-w-0 border-r border-border ${leftBg}`}>
        <LineNumber value={row.left?.lineNo ?? null} />
        <span className="select-none text-text-muted w-4 shrink-0 text-center">{leftPrefix}</span>
        <span className="pl-2 pr-2 whitespace-pre break-all font-mono text-[13px] leading-5">
          {row.left?.content ?? ""}
        </span>
      </div>
      {/* Right (new) column */}
      <div className={`flex flex-1 min-w-0 ${rightBg}`}>
        <LineNumber value={row.right?.lineNo ?? null} />
        <span className="select-none text-text-muted w-4 shrink-0 text-center">{rightPrefix}</span>
        <span className="pl-2 pr-2 whitespace-pre break-all font-mono text-[13px] leading-5">
          {row.right?.content ?? ""}
        </span>
      </div>
    </div>
  );
}

function GapIndicator({
  count,
  gapId,
  onExpand,
  isExpanded,
}: {
  count: number;
  gapId: number;
  onExpand: (id: number) => void;
  isExpanded: boolean;
}) {
  return (
    <button
      onClick={() => onExpand(gapId)}
      className="flex items-center justify-center gap-2 w-full py-1.5 bg-bg-tertiary/60 hover:bg-bg-tertiary border-y border-border text-text-muted text-xs transition-colors cursor-pointer"
      title={isExpanded ? "Collapse unchanged lines" : `Expand ${count} hidden lines`}
    >
      {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      <span className="font-mono">
        {isExpanded ? "Collapse" : `\u22ef ${count} line${count !== 1 ? "s" : ""} hidden`}
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function DiffViewer({
  diffs,
  onAcceptAll,
  onRejectAll,
  onAcceptFile,
  onRejectFile,
}: DiffViewerProps) {
  const [activeFileIndex, setActiveFileIndex] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>("inline");
  const [expandedGaps, setExpandedGaps] = useState<Map<string, number[]>>(new Map());

  const activeFile = diffs[activeFileIndex] ?? null;

  // Parse the active file's diff into structured lines
  const parsedLines = useMemo(() => {
    if (!activeFile) return [];
    return parseUnifiedLines(activeFile.diff);
  }, [activeFile]);

  // Compute which line indices should be visible (context window around changes)
  const visibleIndices = useMemo(() => computeVisibleIndices(parsedLines), [parsedLines]);

  // Build segments (lines + gaps) for context collapsing
  // Key expandedGaps per file so expanding in one file doesn't affect another
  const fileExpandedGaps = useMemo(() => {
    if (!activeFile) return new Set<number>();
    const key = `${activeFileIndex}`;
    const stored = expandedGaps.get(key);
    return stored ? new Set<number>(stored) : new Set<number>();
  }, [expandedGaps, activeFileIndex, activeFile]);

  const segments = useMemo(
    () => buildSegments(parsedLines, visibleIndices, fileExpandedGaps),
    [parsedLines, visibleIndices, fileExpandedGaps],
  );

  // For split view, also compute visibility
  const splitVisibleIndices = useMemo(() => {
    return computeVisibleIndices(parsedLines);
  }, [parsedLines]);

  const splitSegments = useMemo(() => {
    return buildSegments(parsedLines, splitVisibleIndices, fileExpandedGaps);
  }, [parsedLines, splitVisibleIndices, fileExpandedGaps]);

  const toggleExpandGap = useCallback(
    (gapId: number) => {
      const key = `${activeFileIndex}`;
      setExpandedGaps((prev) => {
        const next = new Map(prev);
        const existing = next.get(key) ?? [];
        const set = new Set<number>(existing);
        if (set.has(gapId)) {
          set.delete(gapId);
        } else {
          set.add(gapId);
        }
        next.set(key, [...set]);
        return next;
      });
    },
    [activeFileIndex],
  );

  const goToPrevFile = useCallback(() => {
    setActiveFileIndex((i) => Math.max(0, i - 1));
  }, []);

  const goToNextFile = useCallback(() => {
    setActiveFileIndex((i) => Math.min(diffs.length - 1, i + 1));
  }, [diffs.length]);

  // -------------------------------------------------------------------------
  // Render: inline diff
  // -------------------------------------------------------------------------
  const renderInlineDiff = () => {
    if (parsedLines.length === 0) {
      return (
        <div className="flex items-center justify-center h-32 text-text-muted text-sm">
          No diff content available
        </div>
      );
    }

    let gapCounter = 0;

    return segments.map((seg, segIdx) => {
      if (seg.kind === "gap") {
        const gid = gapCounter++;
        return (
          <GapIndicator
            key={`gap-${segIdx}`}
            count={seg.count}
            gapId={gid}
            onExpand={toggleExpandGap}
            isExpanded={false}
          />
        );
      }

      return (
        <div key={`lines-${segIdx}`}>
          {seg.indices.map((lineIdx) => (
            <InlineDiffLine key={lineIdx} line={parsedLines[lineIdx]} />
          ))}
        </div>
      );
    });
  };

  // -------------------------------------------------------------------------
  // Render: side-by-side diff
  // -------------------------------------------------------------------------
  const renderSplitDiff = () => {
    if (parsedLines.length === 0) {
      return (
        <div className="flex items-center justify-center h-32 text-text-muted text-sm">
          No diff content available
        </div>
      );
    }

    // Map from line index to split row: we rebuild rows per segment
    let gapCounter = 0;

    return splitSegments.map((seg, segIdx) => {
      if (seg.kind === "gap") {
        const gid = gapCounter++;
        return (
          <GapIndicator
            key={`split-gap-${segIdx}`}
            count={seg.count}
            gapId={gid}
            onExpand={toggleExpandGap}
            isExpanded={false}
          />
        );
      }

      // Convert the line indices in this segment to split rows
      const rows = buildSplitRows(seg.indices.map((i) => parsedLines[i]));
      return (
        <div key={`split-lines-${segIdx}`}>
          {rows.map((row, rIdx) => (
            <SplitDiffRow key={rIdx} row={row} />
          ))}
        </div>
      );
    });
  };

  // -------------------------------------------------------------------------
  // Main render
  // -------------------------------------------------------------------------

  if (diffs.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted font-mono text-sm bg-bg-primary rounded-lg border border-border">
        No file changes to display
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-bg-primary rounded-lg border border-border overflow-hidden">
      {/* ---- Top toolbar ---- */}
      <div className="flex items-center justify-between px-4 py-2 bg-bg-secondary border-b border-border shrink-0">
        {/* Left: file nav */}
        <div className="flex items-center gap-2">
          <button
            onClick={goToPrevFile}
            disabled={activeFileIndex === 0}
            className="p-1 rounded hover:bg-bg-tertiary text-text-secondary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Previous file"
          >
            <ChevronLeft size={16} />
          </button>
          <span className="text-text-secondary text-xs font-mono whitespace-nowrap">
            {activeFileIndex + 1} / {diffs.length}
          </span>
          <button
            onClick={goToNextFile}
            disabled={activeFileIndex === diffs.length - 1}
            className="p-1 rounded hover:bg-bg-tertiary text-text-secondary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Next file"
          >
            <ChevronRight size={16} />
          </button>
        </div>

        {/* Center: view mode toggle */}
        <div className="flex items-center bg-bg-tertiary rounded overflow-hidden border border-border">
          <button
            onClick={() => setViewMode("inline")}
            className={`flex items-center gap-1.5 px-3 py-1 text-xs transition-colors ${
              viewMode === "inline"
                ? "bg-accent/20 text-accent"
                : "text-text-muted hover:text-text-secondary"
            }`}
            title="Inline (unified) view"
          >
            <Rows size={14} />
            Inline
          </button>
          <button
            onClick={() => setViewMode("split")}
            className={`flex items-center gap-1.5 px-3 py-1 text-xs transition-colors ${
              viewMode === "split"
                ? "bg-accent/20 text-accent"
                : "text-text-muted hover:text-text-secondary"
            }`}
            title="Side-by-side view"
          >
            <Columns size={14} />
            Split
          </button>
        </div>

        {/* Right: global actions */}
        <div className="flex items-center gap-1.5">
          {onAcceptAll && (
            <button
              onClick={onAcceptAll}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium bg-success/15 text-success hover:bg-success/25 transition-colors"
              title="Accept all changes"
            >
              <CheckCheck size={14} />
              Accept All
            </button>
          )}
          {onRejectAll && (
            <button
              onClick={onRejectAll}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium bg-error/15 text-error hover:bg-error/25 transition-colors"
              title="Reject all changes"
            >
              <XCircle size={14} />
              Reject All
            </button>
          )}
        </div>
      </div>

      {/* ---- File tabs ---- */}
      <div className="flex items-center gap-0 px-2 bg-bg-secondary/60 border-b border-border overflow-x-auto shrink-0 scrollbar-thin">
        {diffs.map((file, idx) => (
          <button
            key={file.path}
            onClick={() => setActiveFileIndex(idx)}
            className={`flex items-center gap-2 px-3 py-2 text-xs whitespace-nowrap border-b-2 transition-colors ${
              idx === activeFileIndex
                ? "border-accent text-text-primary bg-bg-primary/50"
                : "border-transparent text-text-muted hover:text-text-secondary hover:bg-bg-tertiary/40"
            }`}
          >
            <StatusBadge status={file.status} />
            <span className="font-mono truncate max-w-[200px]" title={file.path}>
              {file.path.split("/").pop()}
            </span>
          </button>
        ))}
      </div>

      {/* ---- Active file header ---- */}
      {activeFile && (
        <div className="flex items-center justify-between px-4 py-1.5 bg-bg-secondary/40 border-b border-border shrink-0">
          <span className="text-text-secondary text-xs font-mono truncate" title={activeFile.path}>
            {activeFile.path}
          </span>
          <div className="flex items-center gap-1.5 shrink-0 ml-3">
            {onAcceptFile && (
              <button
                onClick={() => onAcceptFile(activeFile.path)}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-success/15 text-success hover:bg-success/25 transition-colors"
                title={`Accept changes to ${activeFile.path}`}
              >
                <Check size={12} />
                Accept
              </button>
            )}
            {onRejectFile && (
              <button
                onClick={() => onRejectFile(activeFile.path)}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-error/15 text-error hover:bg-error/25 transition-colors"
                title={`Reject changes to ${activeFile.path}`}
              >
                <X size={12} />
                Reject
              </button>
            )}
          </div>
        </div>
      )}

      {/* ---- Diff content ---- */}
      <div className="flex-1 overflow-y-auto overflow-x-auto min-h-0">
        {viewMode === "inline" ? renderInlineDiff() : renderSplitDiff()}
      </div>
    </div>
  );
}
