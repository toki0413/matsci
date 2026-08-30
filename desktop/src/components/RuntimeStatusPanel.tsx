/**
 * Runtime Status 抽屉 — "按需感知入口"的最小实现.
 *
 * Ctrl+K 触发, 不进侧栏, 不常驻. 打开时拉一次后台任务 + 待决策, 用完即关.
 * 只做"看"不做"改" — 治↳自服务, 不塞进去管理/审批这类活(那些留给对应面板).
 *
 * 数据源全是现有端点, 不加新后端:
 *   GET /tasks?active_only=true   → 后台正在跑的 objective
 *   GET /inbox?state=pending       → 待决策 (approval / question / notification)
 *
 * ponytail: 打开时才 fetch, 无轮询/无 WS. ceiling: 要看实时进度得去 /tasks/stream.
 */

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../lib/api";
import {
  Loader2,
  X,
  ListTodo,
  MailOpen,
  CheckCircle2,
  Clock,
  AlertTriangle,
  RotateCcw,
} from "lucide-react";

interface RuntimeTask {
  task_id: string;
  description: string;
  status: string;
  percentage: number;
  current_label?: string;
  stage_labels?: string[];
  error?: string;
  engine_kind?: string;
}

interface InboxEntry {
  id: string;
  kind: string;
  title: string;
  body?: string;
  state: string;
  created_at?: string;
}

interface ResumableRun {
  run_id: string;
  objective: string;
  iteration: number;
  saved_at?: number;
  resumable?: boolean;
}

interface RuntimeStatusPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ToolEconomy {
  calls: number;
  cache_hits: number;
  dedupe_hits: number;
  chars_saved: number;
  tokens_saved: number;
}

const KIND_LABELS: Record<string, string> = {
  approval: "批准",
  question: "提问",
  directory: "授权目录",
  plan: "计划待批",
  notification: "通知",
};

export function RuntimeStatusPanel({ isOpen, onClose }: RuntimeStatusPanelProps) {
  const { t } = useTranslation();
  const [tasks, setTasks] = useState<RuntimeTask[]>([]);
  const [inbox, setInbox] = useState<InboxEntry[]>([]);
  const [resumable, setResumable] = useState<ResumableRun[]>([]);
  const [economy, setEconomy] = useState<ToolEconomy | null>(null);
  const [loading, setLoading] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError("");

    const fetchState = async () => {
      try {
        const [taskResp, inboxResp, resumableResp, econResp] = await Promise.all([
          api.get<{ tasks?: RuntimeTask[] }>("/tasks", {
            params: new URLSearchParams({ active_only: "true" }),
          }),
          api.get<{ items?: InboxEntry[] }>("/inbox", {
            params: new URLSearchParams({ state: "pending" }),
          }),
          api.get<{ runs?: ResumableRun[] }>("/autoloop/resumable").catch(() => null),
          api.get<ToolEconomy>("/tool-economy"),
        ]);
        setTasks(taskResp.tasks || []);
        setInbox(inboxResp.items || []);
        setResumable((resumableResp?.runs || []).filter((r) => r.resumable));
        setEconomy(econResp);
      } catch (e: any) {
        setError(e.message || "加载失败");
      } finally {
        setLoading(false);
      }
    };

    fetchState();
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      setTasks([]);
      setInbox([]);
      setResumable([]);
      setEconomy(null);
      setError("");
    }
  }, [isOpen]);

  const resumeRun = async () => {
    if (resuming) return;
    setResuming(true);
    setError("");
    try {
      await api.post("/autoloop/resume");
      // 续跑已作为后台任务拉起, 关掉抽屉让进度面板接管展示.
      onClose();
    } catch (e: any) {
      setError(e.message || "续跑失败");
    } finally {
      setResuming(false);
    }
  };

  const [resolvingId, setResolvingId] = useState<string | null>(null);

  const resolveInbox = async (item: InboxEntry, decision: string) => {
    if (resolvingId) return;
    setResolvingId(item.id);
    setError("");
    try {
      await api.post(`/inbox/${encodeURIComponent(item.id)}/resolve`, {
        resolution: decision,
      });
      // 已决的项从列表移除, 引擎侧 await store.wait 会收到答案恢复.
      setInbox((prev) => prev.filter((x) => x.id !== item.id));
    } catch (e: any) {
      setError(e.message || "提交失败");
    } finally {
      setResolvingId(null);
    }
  };

  if (!isOpen) return null;

  const runningTasks = tasks.filter((t) =>
    ["pending", "running"].includes(t.status)
  );

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-xl overflow-hidden rounded-xl border border-border bg-bg-secondary shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={t("runtime.title", "运行状态")}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <ListTodo size={16} className="text-text-muted" aria-hidden="true" />
            <span className="text-sm font-semibold text-text-primary">
              {t("runtime.title", "运行状态")}
            </span>
            {(runningTasks.length > 0 || inbox.length > 0) && (
              <span className="rounded-full bg-accent/10 px-2 py-0.5 text-xs text-accent">
                {runningTasks.length + inbox.length}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary transition-colors"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </div>

        <div className="max-h-[60vh] overflow-y-auto p-4 space-y-5">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-8 text-text-muted">
              <Loader2 size={18} className="animate-spin" aria-hidden="true" />
              <span className="text-sm">{t("runtime.loading", "加载中…")}</span>
            </div>
          )}

          {error && !loading && (
            <div className="flex items-center gap-2 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-muted">
              <AlertTriangle size={14} className="shrink-0" aria-hidden="true" />
              {error}
            </div>
          )}

          {!loading && !error && (
            <>
              {/* 后台任务 */}
              <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                  {t("runtime.tasks", "后台任务")}
                  {runningTasks.length > 0 && ` · ${runningTasks.length}`}
                </h3>
                {runningTasks.length === 0 ? (
                  <p className="py-4 text-center text-sm text-text-muted">
                    {t("runtime.noTasks", "暂无运行中的任务")}
                  </p>
                ) : (
                  <ul className="space-y-2">
                    {runningTasks.map((task) => (
                      <li
                        key={task.task_id}
                        className="rounded-lg border border-border bg-bg-tertiary/60 px-3 py-2"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="min-w-0 flex-1 truncate text-sm text-text-primary">
                            {task.description || task.task_id}
                          </span>
                          <span className="flex shrink-0 items-center gap-1 text-xs text-text-muted">
                            <Clock size={12} aria-hidden="true" />
                            {task.percentage}%
                          </span>
                        </div>
                        {(task.current_label || task.stage_labels?.length) && (
                          <p className="mt-1 truncate text-xs text-text-muted">
                            {task.current_label ||
                              (task.stage_labels || []).join(" › ")}
                          </p>
                        )}
                        <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-bg-tertiary">
                          <div
                            className="h-full rounded-full bg-accent transition-all"
                            style={{ width: `${Math.min(100, task.percentage)}%` }}
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              {/* 待决策 */}
              <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                  {t("runtime.inbox", "待决策")}
                  {inbox.length > 0 && ` · ${inbox.length}`}
                </h3>
                {inbox.length === 0 ? (
                  <p className="py-4 text-center text-sm text-text-muted">
                    {t("runtime.noInbox", "没有待你决定的项")}
                  </p>
                ) : (
                  <ul className="space-y-2">
                    {inbox.map((item) => (
                      <li
                        key={item.id}
                        className="rounded-lg border border-border bg-bg-tertiary/60 px-3 py-2"
                      >
                        <div className="flex items-center gap-2">
                          <MailOpen size={14} className="shrink-0 text-text-muted" aria-hidden="true" />
                          <span className="text-xs text-accent">
                            {item.kind && (KIND_LABELS[item.kind] || item.kind)}
                          </span>
                        </div>
                        <p className="mt-1 text-sm text-text-primary">{item.title}</p>
                        {item.body && (
                          <p className="mt-0.5 line-clamp-2 text-xs text-text-muted">
                            {item.body}
                          </p>
                        )}
                        <div className="mt-2 flex items-center gap-2">
                          <button
                            onClick={() => resolveInbox(item, "approve: 批准续投")}
                            disabled={resolvingId != null}
                            className="inline-flex items-center gap-1.5 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-contrast transition-opacity hover:opacity-90 disabled:opacity-50"
                          >
                            <CheckCircle2 size={12} aria-hidden="true" />
                            {t("runtime.approve", "批准")}
                          </button>
                          <button
                            onClick={() => resolveInbox(item, "deny: 停止")}
                            disabled={resolvingId != null}
                            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-bg-tertiary px-2.5 py-1 text-xs font-medium text-text-primary transition-opacity hover:opacity-90 disabled:opacity-50"
                          >
                            <X size={12} aria-hidden="true" />
                            {t("runtime.deny", "拒绝")}
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              {/* 可续跑 */}
              {resumable.length > 0 && (
                <section>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                    {t("runtime.resumable", "可续跑")}
                    {` · ${resumable.length}`}
                  </h3>
                  <ul className="space-y-2">
                    {resumable.map((run) => (
                      <li
                        key={run.run_id}
                        className="rounded-lg border border-border bg-bg-tertiary/60 px-3 py-2"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="min-w-0 flex-1 truncate text-sm text-text-primary">
                            {run.objective || run.run_id}
                          </span>
                          <span className="shrink-0 text-xs text-text-muted">
                            #{run.iteration}
                          </span>
                        </div>
                        <button
                          onClick={resumeRun}
                          disabled={resuming}
                          className="mt-2 flex items-center gap-1.5 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-contrast transition-opacity hover:opacity-90 disabled:opacity-50"
                        >
                          <RotateCcw size={12} aria-hidden="true" />
                          {resuming
                            ? t("runtime.resuming", "续跑中…")
                            : t("runtime.resume", "一键续跑")}
                        </button>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {/* 工具经济 */}
              {economy && (
                <section>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                    {t("runtime.toolEconomy", "工具经济")}
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: t("runtime.realCalls", "真实调用"), value: economy.calls ?? 0 },
                      { label: t("runtime.cacheHits", "缓存命中"), value: economy.cache_hits ?? 0 },
                      { label: t("runtime.dedupeHits", "去重命中"), value: economy.dedupe_hits ?? 0 },
                      { label: t("runtime.tokensSaved", "省下 tokens"), value: economy.tokens_saved ?? 0 },
                    ].map((m) => (
                      <div
                        key={m.label}
                        className="rounded-lg border border-border bg-bg-tertiary/60 px-3 py-2"
                      >
                        <p className="text-xs text-text-muted">{m.label}</p>
                        <p className="mt-0.5 text-lg font-semibold text-text-primary">
                          {m.value.toLocaleString()}
                        </p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* 空态 */}
              {runningTasks.length === 0 && inbox.length === 0 && (
                <div className="flex flex-col items-center gap-2 py-8 text-text-muted">
                  <CheckCircle2 size={24} aria-hidden="true" />
                  <p className="text-sm">
                    {t("runtime.allClear", "一切正常，没有后台任务或待决策")}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}