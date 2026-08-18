import { useState, useEffect, useRef, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Search, FolderOpen, Terminal, X } from "lucide-react";
import { api } from "../lib/api";

export type PaletteMode = "commands" | "files";

interface PaletteItem {
  id: string;
  label: string;
  detail?: string;
  icon?: React.ReactNode;
  handler: () => void;
}

interface CommandPaletteProps {
  isOpen: boolean;
  mode: PaletteMode;
  cwd: string;
  commands: PaletteItem[];
  onClose: () => void;
  onOpenFile: (path: string) => void;
}

/** 命令面板（Ctrl+Shift+P）与快速打开（Ctrl+P）共用的下拉弹层。 */
export function CommandPalette({
  isOpen, mode, cwd, commands, onClose, onOpenFile,
}: CommandPaletteProps) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<{ name: string; path: string }[]>([]);
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // 打开时聚焦；关闭时清空状态
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelected(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    } else {
      setFiles([]);
    }
  }, [isOpen]);

  // 快速打开：进入时拉取工作区文件清单（只拉一次，可全量模糊过滤）
  useEffect(() => {
    if (!isOpen || mode !== "files") return;
    let cancelled = false;
    api
      .get<{ files: { name: string; path: string }[] }>("/v1/fs/search", {
        params: new URLSearchParams({ path: cwd || "." }),
      })
      .then((data) => { if (!cancelled) setFiles(data.files || []); })
      .catch(() => { /* 后端未起来时保持空列表 */ });
    return () => { cancelled = true; };
  }, [isOpen, mode, cwd]);

  useEffect(() => setSelected(0), [query]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (mode === "commands") {
      if (!q) return commands;
      return commands.filter((c) => (c.label + (c.detail || "")).toLowerCase().includes(q));
    }
    if (!q) return files.map((f) => ({ id: f.path, label: f.name, detail: f.path, icon: undefined, handler: () => onOpenFile(f.path) }));
    // 文件名优先于全路径的模糊匹配，最贴近 VS Code 的 Quick Open 行为
    return files
      .map((f) => ({ item: f, score: matchScore(f, q) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 80)
      .map(({ item }) => ({ id: item.path, label: item.name, detail: item.path, icon: undefined, handler: () => onOpenFile(item.path) }));
  }, [mode, query, commands, files, onOpenFile]);

  if (!isOpen) return null;

  const active = matches[selected];

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); setSelected((p) => Math.min(p + 1, matches.length - 1)); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); setSelected((p) => Math.max(p - 1, 0)); return; }
    if (e.key === "Enter" && active) { e.preventDefault(); active.handler(); }
  };

  const title = mode === "commands" ? (t("cmd.commandPalette") || "Command Palette") : (t("cmd.quickOpen") || "Quick Open");
  const placeholder = mode === "commands" ? (t("cmd.palettePlaceholder") || "Run a command…") : (t("cmd.quickOpenPlaceholder") || "Open a file…");

  return (
    <div
      className="fixed inset-0 z-[200] flex items-start justify-center bg-black/40 p-4 pt-[15vh] backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-xl border border-border bg-bg-secondary shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-border px-4 py-3">
          {mode === "commands" ? <Search size={18} className="text-text-muted" /> : <FolderOpen size={18} className="text-text-muted" />}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKey}
            placeholder={placeholder}
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted outline-none"
          />
          <button onClick={onClose} className="text-text-muted hover:text-text-primary" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="max-h-[50vh] overflow-y-auto py-1.5">
          {mode === "files" && files.length === 0 && query === "" && (
            <div className="px-4 py-6 text-center text-xs text-text-muted">
              {t("cmd.scanning") || "Scanning workspace…"}
            </div>
          )}
          {matches.length === 0 && (
            <div className="px-4 py-6 text-center text-xs text-text-muted">
              {t("cmd.noMatch") || "No matching results"}
            </div>
          )}
          {matches.map((m, i) => (
            <button
              key={m.id}
              onClick={() => m.handler()}
              onMouseEnter={() => setSelected(i)}
              className={`flex w-full items-center gap-3 px-4 py-2 text-left transition-colors ${
                i === selected ? "bg-accent/10" : "hover:bg-bg-tertiary"
              }`}
            >
              <span className={`shrink-0 ${m.icon ? "" : "text-text-muted"}`}>{m.icon || <Terminal size={14} aria-hidden="true" />}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm text-text-primary">{m.label}</span>
                {m.detail && <span className="block truncate text-xs text-text-muted">{m.detail}</span>}
              </span>
            </button>
          ))}
        </div>

        <div className="flex items-center gap-4 border-t border-border px-4 py-2 text-[11px] text-text-muted">
          <span className="flex items-center gap-1">
            <kbd className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono">Enter</kbd> Select
          </span>
          <span className="flex items-center gap-1">
            <kbd className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono">↑↓</kbd> Navigate
          </span>
          <span className="flex items-center gap-1">
            <kbd className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono">Esc</kbd> Close
          </span>
        </div>
      </div>
    </div>
  );
}

/** 文件名：前缀命中最强，其次子串；全路径弱匹配兜底。数值越低越靠前。 */
function matchScore(f: { name: string; path: string }, q: string): number {
  const name = f.name.toLowerCase();
  if (name === q) return 0;
  if (name.startsWith(q)) return 1;
  if (name.includes(q)) return 2;
  const path = f.path.toLowerCase();
  if (path.includes(q)) return 3;
  return 0;
}