import { useState, useEffect, useRef } from "react";
import { api } from "../lib/api";
import { ReconnectingWebSocket } from "../lib/ws-client";
import { getApiBase, getAuthToken } from "../lib/api-client";

// 远程作业面板: 展示通过 SSH 提交到 HPC 调度器的作业, 支持轮询状态 / 取消。
// 自包含组件, 挂载即拉 /hpc/jobs, 默认每 30s 自动刷新一次。
export function RemoteJobsPanel() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  // 记录正在 refresh / cancel 的作业 id, 用来禁用按钮避免重复点击
  const [busy, setBusy] = useState<string | null>(null);
  const [liveJobId, setLiveJobId] = useState<string | null>(null);
  const [liveOutput, setLiveOutput] = useState('');
  const outputWsRef = useRef<ReconnectingWebSocket | null>(null);

  const toggleLiveOutput = (localId: string) => {
    if (liveJobId === localId) {
      outputWsRef.current?.close();
      outputWsRef.current = null;
      setLiveJobId(null);
      return;
    }
    outputWsRef.current?.close();
    setLiveOutput('');
    const wsUrl = getApiBase().replace(/^http/, 'ws') + `/ws/hpc/jobs/${localId}/output`;
    const ws = new ReconnectingWebSocket({
      url: wsUrl,
      authToken: getAuthToken,
      onMessage: (data) => {
        const text = typeof data === 'string' ? data
          : (data as any)?.data ?? (data as any)?.output ?? JSON.stringify(data);
        setLiveOutput((prev) => prev + text);
      },
    });
    ws.connect();
    outputWsRef.current = ws;
    setLiveJobId(localId);
    setLiveOutput(`[connecting to ${wsUrl}]\n`);
  };

  useEffect(() => () => { outputWsRef.current?.close(); }, []);

  const load = async () => {
    try {
      const data = await api.get<{ success?: boolean; jobs?: any[] }>("/hpc/jobs");
      if (data.success) setJobs(data.jobs || []);
    } catch {
      // 后端还没起来就先静默, 下个轮询周期再试
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
    if (!autoRefresh) return;
    const timer = setInterval(load, 30000);
    return () => clearInterval(timer);
  }, [autoRefresh]);

  const refreshJob = async (localId: string) => {
    setBusy(localId);
    try {
      await api.post(`/hpc/jobs/${localId}/refresh`);
      await load();
    } catch {
      // 网络抖动之类的, 忽略
    }
    setBusy(null);
  };

  const cancelJob = async (localId: string) => {
    if (!confirm("取消该作业? 正在运行的任务会被终止。")) return;
    setBusy(localId);
    try {
      await api.post(`/hpc/jobs/${localId}/cancel`);
      await load();
    } catch {
      // 同上
    }
    setBusy(null);
  };

  const statusColor = (s: string) => {
    switch (s) {
      case "PENDING": return "text-yellow-600";
      case "RUNNING": return "text-blue-600";
      case "COMPLETED": return "text-success";
      case "FAILED": return "text-error";
      case "CANCELLED": return "text-gray-500";
      case "TIMEOUT": return "text-orange-600";
      default: return "text-text-secondary";
    }
  };

  // 算作业耗时, 没结束的拿当前时间凑
  const fmtDuration = (job: any) => {
    if (!job.submitted_at) return "-";
    const end = job.completed_at ? job.completed_at * 1000 : Date.now();
    const ms = end - job.submitted_at * 1000;
    if (ms < 0) return "-";
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${s % 60}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
  };

  const fmtTime = (ts: any) => {
    if (!ts) return "-";
    return new Date(ts * 1000).toLocaleString();
  };

  const cmdStr = (cmd: any) =>
    Array.isArray(cmd) ? cmd.join(" ") : String(cmd ?? "");

  const btnGhost = "rounded px-2 py-1 text-xs text-text-secondary transition-colors hover:bg-bg-tertiary hover:text-text-primary disabled:opacity-50";
  const btnDanger = "rounded px-2 py-1 text-xs text-error transition-colors hover:bg-error/10 disabled:opacity-50";

  return (
    <div className="max-w-4xl space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-text-primary">Remote Jobs</h3>
          <p className="mt-1 text-sm text-text-secondary">
            通过 SSH 提交到 HPC 调度器的作业, 可手动刷新状态或取消未完成的作业。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-border bg-bg-tertiary text-accent"
            />
            Auto-refresh (30s)
          </label>
          <button onClick={load} className="btn-secondary text-xs">
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
      </div>

      {loading ? (
        <p className="text-xs text-text-muted">加载中…</p>
      ) : jobs.length === 0 ? (
        <div className="card py-10 text-center text-sm text-text-secondary">
          No remote jobs yet. Submit a job via SSH to see it here.
        </div>
      ) : (
        <div className="cv-list space-y-2">
          {jobs.map((job) => {
            const active = job.status === "PENDING" || job.status === "RUNNING";
            return (
              <div key={job.local_id} className="card">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={`font-mono text-sm font-bold ${statusColor(job.status)}`}>
                        {job.status}
                      </span>
                      <span className="text-xs text-text-secondary">ID: {job.local_id}</span>
                      {job.scheduler_id && (
                        <span className="text-xs text-text-muted">Scheduler: {job.scheduler_id}</span>
                      )}
                      {job.queue && (
                        <span className="rounded bg-bg-tertiary px-1.5 py-0.5 text-[10px] text-text-secondary">{job.queue}</span>
                      )}
                    </div>
                    <div className="mt-1.5 text-xs text-text-secondary">
                      <span className="text-text-muted">Command:</span>{" "}
                      <code className="break-all text-xs">{cmdStr(job.command)}</code>
                    </div>
                    {job.message && (
                      <div className="mt-1 text-xs text-text-muted">Message: {job.message}</div>
                    )}
                    <div className="mt-1 text-xs text-text-muted">
                      Submitted: {fmtTime(job.submitted_at)}
                      {job.completed_at && ` | Completed: ${fmtTime(job.completed_at)}`}
                      {` | Duration: ${fmtDuration(job)}`}
                      {job.exit_code !== null && job.exit_code !== undefined && ` | Exit: ${job.exit_code}`}
                    </div>
                  </div>
                  <div className="flex flex-shrink-0 gap-1">
                    {active && (
                      <>
                        <button
                          onClick={() => refreshJob(job.local_id)}
                          disabled={busy === job.local_id}
                          className={btnGhost}
                        >
                          {busy === job.local_id ? "…" : "Refresh"}
                        </button>
                        <button
                          onClick={() => cancelJob(job.local_id)}
                          disabled={busy === job.local_id}
                          className={btnDanger}
                        >
                          Cancel
                        </button>
                      </>
                    )}
                    <button
                      onClick={() => toggleLiveOutput(job.local_id)}
                      className={`${btnGhost} ${liveJobId === job.local_id ? 'text-accent' : ''}`}
                    >
                      {liveJobId === job.local_id ? 'Stop' : 'Output'}
                    </button>
                  </div>
                </div>
                {liveJobId === job.local_id && (
                  <div className="mt-2 rounded-lg bg-bg-tertiary p-3">
                    <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-all text-xs font-mono text-text-secondary">{liveOutput}</pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default RemoteJobsPanel;
