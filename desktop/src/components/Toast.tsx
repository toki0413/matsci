/**
 * Minimal toast system — no dependency, no context provider.
 * Call `toast(msg)` from anywhere; the container self-mounts on first use.
 * Each toast auto-dismisses independently (error lasts longer than success).
 */
import { createRoot } from "react-dom/client";

type ToastKind = "info" | "success" | "error";
interface ToastItem { id: number; msg: string; kind: ToastKind; timer?: ReturnType<typeof setTimeout>; }

let _id = 0;
const _queue: ToastItem[] = [];
let _container: HTMLDivElement | null = null;
let _root: ReturnType<typeof createRoot> | null = null;

function _ensureContainer() {
  if (_container) return;
  _container = document.createElement("div");
  _container.style.cssText = "position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none";
  document.body.appendChild(_container);
  _root = createRoot(_container);
}

function _render() {
  if (!_root) return;
  _root.render(
    <>
      {_queue.map((t) => (
        <div
          key={t.id}
          role={t.kind === "error" ? "alert" : "status"}
          aria-live={t.kind === "error" ? "assertive" : "polite"}
          onClick={() => _dismiss(t.id)}
          style={{
            pointerEvents: "auto",
            cursor: "pointer",
            borderRadius: "8px",
            padding: "8px 12px 8px 16px",
            fontSize: "13px",
            fontWeight: 500,
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            animation: "toast-in 0.2s ease-out",
            maxWidth: "400px",
            wordBreak: "break-word",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            ...(
              t.kind === "success" ? { background: "var(--success, #22c55e)", color: "#fff" } :
              t.kind === "error" ? { background: "var(--error, #ef4444)", color: "#fff" } :
              { background: "var(--bg-tertiary, #f0ede8)", color: "var(--text-primary, #1a1815)" }
            ),
          }}
        >
          <span style={{ flex: 1 }}>{t.msg}</span>
          <button
            onClick={(e) => { e.stopPropagation(); _dismiss(t.id); }}
            aria-label="Dismiss notification"
            style={{
              background: "none",
              border: "none",
              color: "currentColor",
              cursor: "pointer",
              padding: "2px",
              display: "flex",
              alignItems: "center",
              borderRadius: "4px",
              opacity: 0.6,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </>
  );
}

function _dismiss(id: number) {
  const idx = _queue.findIndex((t) => t.id === id);
  if (idx !== -1) {
    const item = _queue[idx];
    if (item.timer) clearTimeout(item.timer);
    _queue.splice(idx, 1);
    _render();
  }
}

export function toast(msg: string, kind: ToastKind = "info") {
  _ensureContainer();
  const item: ToastItem = { id: ++_id, msg, kind };
  _queue.push(item);
  // keep max 5 visible
  while (_queue.length > 5) {
    const removed = _queue.shift();
    if (removed?.timer) clearTimeout(removed.timer);
  }
  // Auto-dismiss: error=5s, success=3s, info=3s
  const duration = kind === "error" ? 5000 : 3000;
  item.timer = setTimeout(() => _dismiss(item.id), duration);
  _render();
}

toast.success = (msg: string) => toast(msg, "success");
toast.error = (msg: string) => toast(msg, "error");
