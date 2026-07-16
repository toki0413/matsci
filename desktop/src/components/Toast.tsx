/**
 * Minimal toast system — no dependency, no context provider.
 * Call `toast(msg)` from anywhere; the container self-mounts on first use.
 * Each toast auto-dismisses independently (error lasts longer than success).
 *
 * Extended with progress bar support for long-running tasks.
 */
import { createRoot } from "react-dom/client";

type ToastKind = "info" | "success" | "error" | "progress";

interface ToastItem {
  id: number;
  msg: string;
  kind: ToastKind;
  timer?: ReturnType<typeof setTimeout>;
  progress?: number;
  progressId?: string;
  cancelable?: boolean;
  onCancel?: () => void;
}

let _id = 0;
const _queue: ToastItem[] = [];
let _container: HTMLDivElement | null = null;
let _root: ReturnType<typeof createRoot> | null = null;
const _progressToasts: Set<string> = new Set();

function _ensureContainer() {
  if (_container) return;
  _container = document.createElement("div");
  _container.style.cssText = "position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none";
  document.body.appendChild(_container);
  _root = createRoot(_container);
}

function _render() {
  if (!_root) return;
  const progressToasts = _queue.filter(t => t.kind === "progress");
  const normalToasts = _queue.filter(t => t.kind !== "progress");
  
  // 最多显示 2 个进度 toast
  const visibleProgress = progressToasts.slice(-2);
  // 普通 toast 保持 5 个上限
  const visibleNormal = normalToasts.slice(-5);
  const visible = [...visibleProgress, ...visibleNormal];
  
  _root.render(
    <>
      {visible.map((t) => (
        <div
          key={t.id}
          role={t.kind === "error" ? "alert" : "status"}
          aria-live={t.kind === "error" ? "assertive" : "polite"}
          onClick={() => {
            if (t.kind !== "progress") _dismiss(t.id);
          }}
          style={{
            pointerEvents: "auto",
            cursor: t.kind === "progress" ? "default" : "pointer",
            borderRadius: "8px",
            padding: "8px 12px 8px 16px",
            fontSize: "13px",
            fontWeight: 500,
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            animation: "toast-in 0.2s ease-out",
            maxWidth: "400px",
            wordBreak: "break-word",
            display: "flex",
            flexDirection: "column",
            gap: t.kind === "progress" ? "6px" : "0",
            ...(
              t.kind === "success" ? { background: "var(--success, #22c55e)", color: "#fff" } :
              t.kind === "error" ? { background: "var(--error, #ef4444)", color: "#fff" } :
              t.kind === "progress" ? { background: "var(--bg-tertiary, #f0ede8)", color: "var(--text-primary, #1a1815)" } :
              { background: "var(--bg-tertiary, #f0ede8)", color: "var(--text-primary, #1a1815)" }
            ),
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ flex: 1 }}>{t.msg}</span>
            {t.kind === "progress" && t.progress !== undefined && (
              <span style={{ fontSize: "12px", fontWeight: 600, opacity: 0.8 }}>
                {Math.round(t.progress)}%
              </span>
            )}
            {t.kind === "progress" && t.cancelable && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (t.progressId) {
                    toast.cancel(t.progressId);
                  }
                }}
                aria-label="Cancel task"
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
            )}
            {t.kind !== "progress" && (
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
            )}
          </div>
          {t.kind === "progress" && t.progress !== undefined && (
            <div
              style={{
                width: "100%",
                height: "4px",
                borderRadius: "2px",
                background: "color-mix(in srgb, var(--seed-fg, #1a1815) 12%, transparent)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${t.progress}%`,
                  height: "100%",
                  borderRadius: "2px",
                  background: t.progress >= 100 ? "var(--success, #22c55e)" : "var(--accent, #4285f4)",
                  transition: "width 0.3s ease, background 0.3s ease",
                }}
              />
            </div>
          )}
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
    if (item.progressId) _progressToasts.delete(item.progressId);
    _queue.splice(idx, 1);
    _render();
  }
}

export function toast(msg: string, kind: "info" | "success" | "error" = "info") {
  _ensureContainer();
  const item: ToastItem = { id: ++_id, msg, kind };
  _queue.push(item);
  while (_queue.filter(t => t.kind !== "progress").length > 5) {
    const idx = _queue.findIndex(t => t.kind !== "progress");
    if (idx !== -1) {
      const removed = _queue[idx];
      if (removed?.timer) clearTimeout(removed.timer);
      _queue.splice(idx, 1);
    }
  }
  const duration = kind === "error" ? 5000 : 3000;
  item.timer = setTimeout(() => _dismiss(item.id), duration);
  _render();
}

toast.success = (msg: string) => toast(msg, "success");
toast.error = (msg: string) => toast(msg, "error");

/**
 * 显示进度条 toast
 */
toast.progress = (msg: string, options?: {
  progress?: number;
  id?: string;
  cancelable?: boolean;
  onCancel?: () => void;
}) => {
  _ensureContainer();
  const progressId = options?.id || `progress-${Date.now()}`;
  
  // 如果已存在同 ID 的进度 toast，更新它
  const existing = _queue.find(t => t.progressId === progressId);
  if (existing) {
    existing.msg = msg;
    existing.progress = options?.progress ?? 0;
    existing.cancelable = options?.cancelable ?? false;
    existing.onCancel = options?.onCancel;
    _render();
    return progressId;
  }
  
  // 限制进度 toast 数量
  if (_progressToasts.size >= 2) {
    // 移除最早的一个
    const oldestProgress = _queue.find(t => t.kind === "progress");
    if (oldestProgress && oldestProgress.progressId) {
      toast.cancel(oldestProgress.progressId);
    }
  }
  
  const item: ToastItem = {
    id: ++_id,
    msg,
    kind: "progress",
    progress: options?.progress ?? 0,
    progressId,
    cancelable: options?.cancelable ?? false,
    onCancel: options?.onCancel,
  };
  _queue.push(item);
  _progressToasts.add(progressId);
  _render();
  return progressId;
};

/**
 * 更新进度
 */
toast.updateProgress = (id: string, progress: number) => {
  const item = _queue.find(t => t.progressId === id);
  if (item) {
    item.progress = Math.min(100, Math.max(0, progress));
    _render();
  }
};

/**
 * 完成进度，转为 success toast
 */
toast.complete = (id: string, msg?: string) => {
  const idx = _queue.findIndex(t => t.progressId === id);
  if (idx !== -1) {
    _queue.splice(idx, 1);
    _progressToasts.delete(id);
    _render();
    
    // 显示成功 toast
    if (msg) {
      toast.success(msg);
    }
  }
};

/**
 * 取消进度 toast
 */
toast.cancel = (id: string) => {
  const idx = _queue.findIndex(t => t.progressId === id);
  if (idx !== -1) {
    const item = _queue[idx];
    if (item.onCancel) item.onCancel();
    _queue.splice(idx, 1);
    _progressToasts.delete(id);
    _render();
  }
};