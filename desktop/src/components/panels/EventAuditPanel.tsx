/**
 * EventAuditPanel — 后台事件审计视图.
 *
 * 后端 EventBus 会为 agent 生命周期的一切结构化事件(工具调用/编译/团队/
 * 决策/沉淀/embedding 下载...)留一份滑窗历史. 这里通过轮询 /events/recent
 * 回放并实时刷新, 让用户"看得见后台在干什么". 轮询而非订阅 SSE:
 * /events/recent 返回全部类型无需前端挂一长串 addEventListener,
 * 而且断连一次也能自愈, 跟 reflection 的做法一致.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { PanelHeader } from "../settings-shared";
import { api } from "../../lib/api";

interface AuditEvent {
  type: string;
  ts?: number;
  thread_id?: string;
  source?: string;
  data?: Record<string, any>;
}

// 顶层域 → 主题色, 让不同类型一眼可分. 未列出的用灰.
const DOMAIN_COLORS: Record<string, string> = {
  tool: "bg-accent",
  team: "bg-pink-500",
  compact: "bg-amber-500",
  decision: "bg-purple-500",
  sediment: "bg-emerald-500",
  embedding: "bg-sky-500",
  pipeline: "bg-orange-500",
  session: "bg-cyan-500",
  governance: "bg-red-500",
};

const domainOf = (type: string) => type.split(".")[0] ?? "other";
const colorOf = (type: string) => DOMAIN_COLORS[domainOf(type)] ?? "bg-bg-tertiary";

const fmtTime = (ts?: number) =>
  ts ? new Date(ts * 1000).toLocaleTimeString() : "--:--:--";

export function EventAuditPanel() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [filter, setFilter] = useState("");
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState("");
  const seenRef = useRef<Set<string>>(new Set());
  const filterRef = useRef(filter);
  filterRef.current = filter;

  useEffect(() => {
    let stopped = false;
    const poll = async () => {
      if (stopped || paused) return;
      try {
        const d = await api.get<{ events?: AuditEvent[] }>("/events/recent?n=400");
        const fresh = d.events || [];
        const add: AuditEvent[] = [];
        for (const e of fresh) {
          const key = `${e.type}:${e.ts}:${e.thread_id}`;
          if (!seenRef.current.has(key)) {
            seenRef.current.add(key);
            add.push(e);
          }
        }
        if (add.length) {
          setEvents((prev) => [...add.reverse(), ...prev].slice(0, 500));
        }
        setError("");
      } catch (e: any) {
        if (!stopped) setError(`审计接口暂不可用: ${e.message}`);
      }
      setTimeout(poll, 2000);
    };
    poll();
    return () => {
      stopped = true;
    };
  }, [paused]);

  const { filtered, tallies } = useMemo(() => {
    const f = filter.trim().toLowerCase();
    const shown = f ? events.filter((e) =>
      (e.type + " " + (e.source || "")).toLowerCase().includes(f)
    ) : events;
    const counts: Record<string, number> = {};
    for (const e of shown) {
      counts[domainOf(e.type)] = (counts[domainOf(e.type)] || 0) + 1;
    }
    return { filtered: shown, tallies: counts };
  }, [events, filter]);

  // 摘要: 展示各顶层域的事件条数
  const tallySummary = Object.entries(tallies)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);

  return (
    <div data-component="event-audit-panel" className="flex h-full flex-col">
      <PanelHeader title="Event Audit" className="px-6">
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by type / source…"
            className="input h-7 w-56 text-xs"
            aria-label="Filter events"
          />
          <button
            onClick={() => setPaused((v) => !v)}
            className="rounded-lg border border-border px-2 py-1 text-xs text-text-secondary hover:text-text-primary"
            title={paused ? "继续自动刷新" : "暂停自动刷新"}
          >
            {paused ? "▶ Live" : "⏸ Pause"}
          </button>
        </div>
      </PanelHeader>

      {/* 顶层域汇总条 */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border bg-bg-secondary px-6 py-2">
        {tallySummary.length === 0 && (
          <span className="text-xs text-text-muted">暂无事件</span>
        )}
        {tallySummary.map(([dom, n]) => (
          <span
            key={dom}
            className="flex items-center gap-1.5 rounded-full bg-bg-tertiary px-2 py-0.5 text-[11px] text-text-secondary"
          >
            <span className={`h-2 w-2 rounded-full ${colorOf(dom)}`} aria-hidden="true" />
            {dom} · {n}
          </span>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-3 space-y-1.5">
        {error && <div className="text-xs text-error">{error}</div>}
        {filtered.length === 0 && !error && (
          <div className="text-xs text-text-muted">等待后台事件…</div>
        )}
        {filtered.map((e, i) => {
          const dom = domainOf(e.type);
          const detail = e.data && Object.keys(e.data).length
            ? JSON.stringify(e.data).slice(0, 220)
            : "";
          return (
            <div
              key={`${e.ts}-${i}`}
              className="flex items-start gap-2 rounded-lg border border-border bg-bg-secondary p-2"
            >
              <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${colorOf(e.type)}`} aria-hidden="true" />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="truncate font-mono text-xs font-semibold text-text-primary">{e.type}</span>
                  <span className="shrink-0 text-[10px] tabular-nums text-text-muted">{fmtTime(e.ts)}</span>
                  {e.source && (
                    <span className="shrink-0 rounded bg-bg-tertiary px-1 text-[10px] text-text-muted">{e.source}</span>
                  )}
                </div>
                {detail ? (
                  <pre className="mt-0.5 whitespace-pre-wrap break-all font-mono text-[10px] leading-snug text-text-secondary">{detail}</pre>
                ) : null}
                <div className="text-[10px] text-text-muted">domain: {dom}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}