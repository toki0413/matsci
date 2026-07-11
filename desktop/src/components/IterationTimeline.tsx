/**
 * IterationTimeline — vertical timeline of autoloop iterations.
 *
 * ponytail: no new data source. Reuses existing task_progress WS messages
 * that already flow through useChatAndConnection. Just accumulates them
 * by iteration and renders as a vertical list instead of a horizontal bar.
 *
 * Also scans the message history for hypothesis/plan/result summaries to
 * build the timeline without requiring a new backend endpoint.
 */
import { useMemo } from 'react';
import { FlaskConical, CheckCircle2, XCircle, ChevronRight } from 'lucide-react';

interface IterationEntry {
  iteration: number;
  hypothesis?: string;
  mode?: string;
  status: 'running' | 'done' | 'error';
  r_phys?: number;
  surprise?: number;
  persona?: string;
}

interface Props {
  messages: Array<Record<string, unknown>>;
}

export function IterationTimeline({ messages }: Props) {
  const iterations = useMemo(() => {
    // Scan messages for autoloop iteration markers
    const entries: IterationEntry[] = [];
    let current: IterationEntry | null = null;

    for (const msg of messages) {
      const content = String(msg.content || '');

      // Detect iteration start: "iter N:" in content or task_progress
      const iterMatch = content.match(/iter\s+(\d+)/i);
      if (iterMatch) {
        const iterNum = parseInt(iterMatch[1]);
        if (!current || current.iteration !== iterNum) {
          if (current) entries.push(current);
          current = { iteration: iterNum, status: 'running' };
        }
      }

      // Detect hypothesis
      if (current && !current.hypothesis) {
        const hypMatch = content.match(/hypothes[i]s[:\s]+(.{10,120})/i);
        if (hypMatch) current.hypothesis = hypMatch[1].trim();
      }

      // Detect mode
      if (current && !current.mode) {
        const modeMatch = content.match(/(?:mode|plan)[:\s]+(coder|workflow|explore|skill|visual_inspect)/i);
        if (modeMatch) current.mode = modeMatch[1];
      }

      // Detect persona
      if (current && !current.persona) {
        const personaMatch = content.match(/persona[:\s]+(dft_expert|md_expert|reviewer)/i);
        if (personaMatch) current.persona = personaMatch[1];
      }

      // Detect r_phys
      if (current) {
        const rMatch = content.match(/r_phys[:\s]+([\d.]+)/i);
        if (rMatch) current.r_phys = parseFloat(rMatch[1]);
        const sMatch = content.match(/surprise[:\s]+([\d.]+)/i);
        if (sMatch) current.surprise = parseFloat(sMatch[1]);
      }

      // Detect completion
      if (current && /done|complet|finish/i.test(content)) {
        current.status = 'done';
      }
      if (current && /error|fail/i.test(content) && msg.role === 'tool') {
        current.status = 'error';
      }
    }
    if (current) entries.push(current);
    return entries;
  }, [messages]);

  if (iterations.length === 0) {
    return null; // Don't render if no iterations detected
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
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <span style={{ fontWeight: 600, color: 'var(--fg-primary)' }}>Iter {iter.iteration}</span>
              {iter.persona && (
                <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 4, background: 'var(--bg-tertiary)', color: 'var(--fg-muted)' }}>
                  {iter.persona}
                </span>
              )}
              {iter.mode && (
                <span style={{ fontSize: 9, color: 'var(--fg-muted)' }}>· {iter.mode}</span>
              )}
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
