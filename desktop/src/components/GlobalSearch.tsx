import { useState, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../lib/api";
import { GlobalSearchResult, GlobalSearchResponse } from "../types/domain";
import {
  Search,
  MessageSquare,
  Brain,
  BookOpen,
  FileCode,
  ArrowRight,
  Loader2,
  X,
} from "lucide-react";

interface GlobalSearchProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (result: GlobalSearchResult) => void;
}

const TYPE_ICONS: Record<string, React.ReactNode> = {
  thread: <MessageSquare size={16} aria-hidden="true" />,
  memory: <Brain size={16} aria-hidden="true" />,
  knowledge: <BookOpen size={16} aria-hidden="true" />,
  provenance: <FileCode size={16} aria-hidden="true" />,
};

const TYPE_LABELS: Record<string, string> = {
  thread: "Thread",
  memory: "Memory",
  knowledge: "Knowledge",
  provenance: "File",
};

export function GlobalSearch({ isOpen, onClose, onSelect }: GlobalSearchProps) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<GlobalSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
    if (!isOpen) {
      setQuery("");
      setResults([]);
      setError("");
      setSelectedIndex(0);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      setError("");
      return;
    }

    if (query.length < 2) {
      return;
    }

    setLoading(true);
    setError("");

    const debounce = setTimeout(async () => {
      try {
        const data = await api.search<GlobalSearchResponse>(query, 20);
        if (data.error) {
          setError(data.error);
          setResults([]);
        } else {
          setResults(data.results || []);
          setError("");
        }
      } catch (e: any) {
        setError(e.message || "Search failed");
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => clearTimeout(debounce);
  }, [query]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [results]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter" && results[selectedIndex]) {
      e.preventDefault();
      onSelect(results[selectedIndex]);
      onClose();
    } else if (e.key === "Escape") {
      onClose();
    }
  };

  const handleResultClick = (result: GlobalSearchResult) => {
    onSelect(result);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-xl overflow-hidden rounded-xl border border-border bg-bg-secondary shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-border px-4 py-3">
          <Search size={18} className="text-text-muted" aria-hidden="true" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t("search.placeholder") || "Search across threads, memory, knowledge…"}
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted outline-none"
            aria-label="Global search"
          />
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary transition-colors"
            aria-label="Close search"
          >
            <X size={16} />
          </button>
        </div>

        <div className="max-h-[50vh] overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 size={20} className="animate-spin text-accent" aria-label="Loading" />
            </div>
          )}

          {error && !loading && (
            <div className="px-4 py-8 text-center text-sm text-error">
              {error}
            </div>
          )}

          {!loading && !error && query.trim() && results.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-text-muted">
              {t("search.noResults") || "No results found"}
            </div>
          )}

          {!loading && !error && results.length > 0 && (
            <div className="py-2">
              {results.map((result, index) => (
                <button
                  key={`${result.type}-${result.id}-${index}`}
                  onClick={() => handleResultClick(result)}
                  className={`flex w-full items-start gap-3 px-4 py-3 text-left transition-colors focus-visible:outline-none ${
                    index === selectedIndex
                      ? "bg-accent/10"
                      : "hover:bg-bg-tertiary"
                  }`}
                >
                  <span className={`mt-0.5 shrink-0 ${
                    result.type === "thread" ? "text-accent" :
                    result.type === "memory" ? "text-purple-400" :
                    result.type === "knowledge" ? "text-amber-400" :
                    "text-emerald-400"
                  }`}>
                    {TYPE_ICONS[result.type]}
                  </span>
                  <div className="flex flex-1 flex-col min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-text-primary truncate">
                        {result.title}
                      </span>
                      <span className="shrink-0 rounded-full bg-bg-tertiary px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
                        {TYPE_LABELS[result.type]}
                      </span>
                    </div>
                    {result.snippet && (
                      <p className="mt-1 line-clamp-2 text-xs text-text-secondary truncate">
                        {result.snippet}
                      </p>
                    )}
                  </div>
                  <ArrowRight
                    size={14}
                    className={`mt-0.5 shrink-0 transition-opacity ${
                      index === selectedIndex ? "opacity-100 text-accent" : "opacity-0"
                    }`}
                    aria-hidden="true"
                  />
                </button>
              ))}
            </div>
          )}

          {!query.trim() && (
            <div className="px-4 py-6">
              <p className="text-xs text-text-muted">
                {t("search.hint") || "Type at least 2 characters to search"}
              </p>
              <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2 rounded-md bg-bg-tertiary px-3 py-2">
                  <MessageSquare size={14} className="text-accent" aria-hidden="true" />
                  <span className="text-text-secondary">Search conversations</span>
                </div>
                <div className="flex items-center gap-2 rounded-md bg-bg-tertiary px-3 py-2">
                  <Brain size={14} className="text-purple-400" aria-hidden="true" />
                  <span className="text-text-secondary">Search memories</span>
                </div>
                <div className="flex items-center gap-2 rounded-md bg-bg-tertiary px-3 py-2">
                  <BookOpen size={14} className="text-amber-400" aria-hidden="true" />
                  <span className="text-text-secondary">Search knowledge base</span>
                </div>
                <div className="flex items-center gap-2 rounded-md bg-bg-tertiary px-3 py-2">
                  <FileCode size={14} className="text-emerald-400" aria-hidden="true" />
                  <span className="text-text-secondary">Search files</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-border px-4 py-2 text-[11px] text-text-muted">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono">Enter</kbd>
              Select
            </span>
            <span className="flex items-center gap-1">
              <kbd className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono">Esc</kbd>
              Close
            </span>
          </div>
          {results.length > 0 && (
            <span>{results.length} {t("search.results") || "results"}</span>
          )}
        </div>
      </div>
    </div>
  );
}
