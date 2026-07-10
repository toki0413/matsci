/**
 * Minimal toast system — no dependency, no context provider.
 * Call `toast(msg)` from anywhere; the container self-mounts on first use.
 */
import { createRoot } from "react-dom/client";

type ToastKind = "info" | "success" | "error";
interface ToastItem { id: number; msg: string; kind: ToastKind; }

let _id = 0;
const _queue: ToastItem[] = [];
let _container: HTMLDivElement | null = null;
let _root: ReturnType<typeof createRoot> | null = null;
let _hideTimer: ReturnType<typeof setTimeout> | null = null;

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
          style={{
            pointerEvents: "auto",
            borderRadius: "8px",
            padding: "8px 16px",
            fontSize: "13px",
            fontWeight: 500,
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            animation: "toast-in 0.2s ease-out",
            ...(
              t.kind === "success" ? { background: "#22c55e", color: "#fff" } :
              t.kind === "error" ? { background: "#ef4444", color: "#fff" } :
              { background: "var(--bg-tertiary, #f0ede8)", color: "var(--text-primary, #1a1815)" }
            ),
          }}
        >
          {t.msg}
        </div>
      ))}
    </>
  );
}

function _scheduleHide() {
  if (_hideTimer) clearTimeout(_hideTimer);
  _hideTimer = setTimeout(() => {
    _queue.length = 0;
    _render();
  }, 3000);
}

export function toast(msg: string, kind: ToastKind = "info") {
  _ensureContainer();
  _queue.push({ id: ++_id, msg, kind });
  // keep max 3 visible
  while (_queue.length > 3) _queue.shift();
  _render();
  _scheduleHide();
}

toast.success = (msg: string) => toast(msg, "success");
toast.error = (msg: string) => toast(msg, "error");
