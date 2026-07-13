/**
 * IterationTimeline — vertical timeline of autoloop iterations.
 *
 * 优先用结构化 campaign SSE 事件 (campaign.iteration / hypothesis / retry /
 * suspect / refine) 构建 timeline. 之前靠正则刮消息文本, retry/suspect/refine
 * 事件根本到不了前端. 现在 SSE /tasks/stream 的 'campaign' event 直达.
 *
 * fallback: 没有 campaign 事件时 (旧会话 / 非 autoloop), 退回正则刮文本,
 * 保持向后兼容.
 */
import { useMemo } from 'react';
import { FlaskConical, CheckCircle2, XCircle, ChevronRight, RefreshCw, AlertTriangle } from 'lucide-react';

interface IterationEntry {
  iteration: number;
  hypothesis?: string;
  mode?: string;
  status: 'running' | 'done' | 'error';
  r_phys?: number;
  surprise?: number;
  persona?: string;
  // 研究循环节点标记 (来自 campaign.retry / suspect / refine)
  flags?: string[];
}

export interface CampaignEventEntry {
  event: string;
  data: Record<string, unknown>;
  ts: number;
  task_id: string;
}

interface Props {
  messages: Array<Record<string, unknown>>;
  campaignEvents?: CampaignEventEntry[];
}

/** 从结构化 campaign SSE 事件构建 timeline. */
function buildFromCampaign(events: CampaignEventEntry[]): IterationEntry[] {
  const byIter = new Map<number, IterationEntry>();
  for (const ev of events) {
    const it = ev.data.iteration as number | undefined;
    if (it == null) continue;
    let entry = byIter.get(it);
    if (!entry) {
      entry = { iteration: it, status: 'running' };
      byIter.set(it, entry);
    }
    switch (ev.event) {
      case 'campaign.iteration': {
        // 新一轮开始, 之前还没结束的标记 done
        for (const e of byIter.values()) {
          if (e.iteration < it && e.status === 'running') e.status = 'done';
        }
        const max = ev.data.max as number | undefined;
        if (max) entry.mode = `1/${max}`;
        break;
      }
      case 'campaign.hypothesis':
        entry.hypothesis = ev.data.hypothesis as string | undefined;
        break;
      case 'campaign.retry':
        entry.flags = [...(entry.flags || []), 'retry'];
        entry.status = 'error';
        break;
      case 'campaign.suspect':
        entry.flags = [...(entry.flags || []), 'suspect'];
        entry.status = 'error';
        break;
      case 'campaign.refine':
        entry.flags = [...(entry.flags || []), ev.data.pivot ? 'pivot' : 'refine'];
        break;
    }
  }
  return [...byIter.values()].sort((a, b) => a.iteration - b.iteration);
}

/** fallback: 没有结构化事件时, 从消息文本正则刮. */
function buildFromMessages(messages: Array<Record<string, unknown>>): IterationEntry[] {
  const entries: IterationEntry[] = [];
  let current: IterationEntry | null = null;

  for (const msg of messages) {
    const content = String(msg.content || '');

    const iterMatch = content.match(/iter\s+(\d+)/i);
    if (iterMatch) {
      const iterNum = parseInt(iterMatch[1]);
      if (!current || current.iteration !== iterNum) {
        if (current) entries.push(current);
        current = { iteration: iterNum, status: 'running' };
      }
    }
    if (current && !current.hypothesis) {
      const hypMatch = content.match(/hypothes[i]s[:\s]+(.{10,120})/i);
      if (hypMatch) current.hypothesis = hypMatch[1].trim();
    }
    if (current && !current.mode) {
      const modeMatch = content.match(/(?:mode|plan)[:\s]+(coder|workflow|explore|skill|visual_inspect)/i);
      if (modeMatch) current.mode = modeMatch[1];
    }
    if (current && !current.persona) {
      const personaMatch = content.match(/persona[:\s]+(dft_expert|md_expert|reviewer)/i);
      if (personaMatch) current.persona = personaMatch[1];
    }
    if (current) {
      const rMatch = content.match(/r_phys[:\s]+([\d.]+)/i);
      if (rMatch) current.r_phys = parseFloat(rMatch[1]);
      const sMatch = content.match(/surprise[:\s]+([\d.]+)/i);
      if (sMatch) current.surprise = parseFloat(sMatch[1]);
    }
    if (current && /done|complet|finish/i.test(content)) {
      current.status = 'done';
    }
    if (current && /error|fail/i.test(content) && msg.role === 'tool') {
      current.status = 'error';
    }
  }
  if (current) entries.push(current);
  return entries;
}

const FLAG_LABEL: Record<string, string> = {
  retry: 'retry',
  suspect: 'suspect',
  refine: 'refine',
  pivot: 'pivot',
};

export function IterationTimeline({ messages, campaignEvents }: Props) {
  const iterations = useMemo(() => {
    if (campaignEvents && campaignEvents.length > 0) {
      return buildFromCampaign(campaignEvents);
    }
    return buildFromMessages(messages);
  }, [messages, campaignEvents]);

  if (iterations.length === 0) {
    return null;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {iterations.map((iter, i) => (
        <div key={i} style={{ display: 'flex', gap: 8, position: 'relative' }}>
          {/* Timeline line + node */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 20 }}>
            <div style={{
              width: 20, height: 20, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: iter.status === 'done' ? 'var(--success-bg, #10b98120)' :
                         iter.status === 'error' ? 'var(--error-bg, #ef444420)' :
                         'var(--info-bg, #3b82f620)',
              border: `1px solid ${iter.status === 'done' ? '#10b981' : iter.status === 'error' ? '#ef4444' : '#3b82f6'}`,
            }}>
              {iter.status === 'done' ? <CheckCircle2 size={12} color="#10b981" /> :
               iter.status === 'error' ? <XCircle size={12} color="#ef4444" /> :
               <FlaskConical size={12} color="#3b82f6" />}
            </div>
            {i < iterations.length - 1 && (
              <div style={{ width: 1, flex: 1, background: 'var(--border-default, #333)' }} />
            )}
          </div>

          {/* Content */}
          <div style={{ flex: 1, paddingBottom: 12, fontSize: 11 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600, color: 'var(--fg-primary)' }}>Iter {iter.iteration}</span>
              {iter.persona && (
                <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 4, background: 'var(--bg-tertiary)', color: 'var(--fg-muted)' }}>
                  {iter.persona}
                </span>
              )}
              {iter.mode && (
                <span style={{ fontSize: 9, color: 'var(--fg-muted)' }}>· {iter.mode}</span>
              )}
              {/* 研究循环节点标记 */}
              {iter.flags?.map((f, idx) => (
                <span key={idx} style={{
                  fontSize: 9, padding: '1px 5px', borderRadius: 3,
                  display: 'inline-flex', alignItems: 'center', gap: 2,
                  background: f === 'pivot' ? '#f59e0b20' : f === 'suspect' ? '#ef444420' : 'var(--bg-tertiary)',
                  color: f === 'pivot' ? '#f59e0b' : f === 'suspect' ? '#ef4444' : 'var(--fg-muted)',
                }}>
                  {f === 'suspect' ? <AlertTriangle size={9} /> : <RefreshCw size={9} />}
                  {FLAG_LABEL[f] || f}
                </span>
              ))}
            </div>
            {iter.hypothesis && (
              <div style={{ color: 'var(--fg-secondary)', display: 'flex', alignItems: 'flex-start', gap: 4 }}>
                <ChevronRight size={10} style={{ marginTop: 2, flexShrink: 0 }} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                  {iter.hypothesis}
                </span>
              </div>
            )}
            <div style={{ display: 'flex', gap: 8, marginTop: 2 }}>
              {iter.r_phys != null && (
                <span style={{ fontSize: 9, color: iter.r_phys > 0.7 ? '#10b981' : iter.r_phys > 0.4 ? '#f59e0b' : '#ef4444' }}>
                  r={iter.r_phys.toFixed(2)}
                </span>
              )}
              {iter.surprise != null && (
                <span style={{ fontSize: 9, color: iter.surprise > 0.5 ? '#f59e0b' : 'var(--fg-muted)' }}>
                  surprise={iter.surprise.toFixed(2)}
                </span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
