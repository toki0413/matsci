/**
 * ConvergenceChart — SCF/optimization convergence renderer.
 * ponytail: recharts already in deps. Parses {energies: [...]} or {convergence: {scf_energies: [...]}}.
 */
import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

interface Props {
  content: string;
}

export default function ConvergenceChart({ content }: Props) {
  const data = useMemo(() => {
    try {
      const parsed = JSON.parse(content);
      // Try multiple shapes
      const energies = parsed.energies || parsed.scf_energies ||
                       parsed.convergence?.scf_energies || parsed.convergence?.energies || [];
      const forces = parsed.forces || parsed.convergence?.forces || [];
      const threshold = parsed.threshold ?? parsed.convergence?.threshold ?? null;

      if (!Array.isArray(energies) || energies.length === 0) return null;

      const chartData = energies.map((e: number, i: number) => ({
        step: i,
        energy: e,
        force: forces[i] ?? null,
      }));

      return { chartData, threshold, hasForces: forces.length > 0 };
    } catch {
      return null;
    }
  }, [content]);

  if (!data) {
    return <div className="p-4 text-xs text-text-muted">No convergence data found in result.</div>;
  }

  return (
    <div style={{ width: '100%', height: 240 }}>
      <ResponsiveContainer>
        <LineChart data={data.chartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" opacity={0.3} />
          <XAxis dataKey="step" tick={{ fontSize: 9, fill: 'var(--fg-muted)' }} />
          <YAxis tick={{ fontSize: 9, fill: 'var(--fg-muted)' }} />
          <Tooltip
            contentStyle={{ fontSize: 10, background: 'var(--bg-secondary)', border: '1px solid var(--border-default)' }}
          />
          <Line type="monotone" dataKey="energy" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
          {data.hasForces && (
            <Line type="monotone" dataKey="force" stroke="#ef4444" strokeWidth={1.5} dot={{ r: 2 }} isAnimationActive={false} />
          )}
          {data.threshold != null && (
            <ReferenceLine y={data.threshold} stroke="#10b981" strokeDasharray="5 5" label={{ value: 'threshold', fontSize: 9, fill: '#10b981' }} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
