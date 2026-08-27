/**
 * EmbeddingDownloadBanner — 首次用到知识库时, 模型权重需联网拉取(桌面版不内置).
 * 用一条细横幅让用户感知下载进度; 失败时改错误色并提示重试, 避免"首次提问莫名卡住".
 */
import { CloudDownload, CheckCircle, AlertTriangle, X } from "lucide-react";
import type { EmbeddingDownloadState } from "../hooks/useChatAndConnection";

interface Props {
  state: EmbeddingDownloadState;
  onDismiss: () => void;
}

export default function EmbeddingDownloadBanner({ state, onDismiss }: Props) {
  if (state.status === "idle") return null;

  const downloading = state.status === "downloading";
  const done = state.status === "done";
  const error = state.status === "error";
  // 后端现在会推 embedding.download.progress 事件带真实 percent; percent 仍为 0
  // (常见于刚 start、还没来得及算字节) 时退回 y不确定动画, 避免误判为卡死.
  const indeterminate = downloading && !state.percent;
  const pct = downloading ? state.percent ?? 0 : 0;

  return (
    <div
      className={`border-b bg-bg-secondary ${
        error ? "border-error/30" : "border-border"
      }`}
      role={error ? "alert" : "status"}
      aria-live="polite"
    >
      <div className="flex items-center gap-3 px-6 py-2">
        {error ? (
          <AlertTriangle size={16} className="text-error shrink-0" aria-hidden="true" />
        ) : done ? (
          <CheckCircle size={16} className="text-success shrink-0" aria-hidden="true" />
        ) : (
          <CloudDownload size={16} className="text-accent shrink-0 animate-bounce motion-reduce:animate-none" aria-hidden="true" />
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3">
            <span className={`text-xs font-semibold whitespace-nowrap ${
              error ? "text-error" : done ? "text-success" : "text-text-secondary"
            }`}>
              {error
                ? "知识库模型下载失败"
                : done
                  ? "知识库模型已就绪"
                  : "正在下载知识库模型（首次使用需联网，约 470MB）"}
            </span>
            <button
              type="button"
              onClick={onDismiss}
              className="p-0.5 rounded text-text-muted hover:text-text-primary focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:outline-none"
              aria-label="关闭提示"
            >
              <X size={14} aria-hidden="true" />
            </button>
          </div>

          {indeterminate && (
            <div className="mt-1 h-1.5 w-full rounded-full bg-bg-tertiary overflow-hidden">
              <div className="h-full w-1/3 rounded-full bg-accent animate-slide-indeterminate" />
            </div>
          )}
          {downloading && !indeterminate && (
            <div
              className="mt-1 flex items-center gap-2"
              role="progressbar"
              aria-valuenow={Math.round(pct)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div className="h-1.5 flex-1 rounded-full bg-bg-tertiary overflow-hidden">
                <div
                  className="h-full rounded-full bg-accent transition-[width] duration-300"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="w-9 text-right text-[10px] tabular-nums text-text-muted">
                {Math.round(pct)}%
              </span>
            </div>
          )}
          {downloading && (
            <span className="mt-0.5 block text-[10px] text-text-muted">
              {indeterminate
                ? "连接下载源中… · 可继续使用其他功能，视网速可能需要几分钟"
                : "下载中，可继续使用其他功能 · 视网速可能需要几分钟"}
            </span>
          )}
          {error && (
            <span className="mt-0.5 block text-[10px] text-text-muted">
              {state.error} · 可稍后重试，或检查网络/镜像设置后再次提问
            </span>
          )}
          {done && (
            <span className="mt-0.5 block text-[10px] text-text-muted">
              模型已缓存，后续使用不再联网
            </span>
          )}
        </div>
      </div>
    </div>
  );
}