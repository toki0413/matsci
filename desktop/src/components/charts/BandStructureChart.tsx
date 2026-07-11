/**
 * BandStructureChart — inline band structure + DOS renderer.
 * ponytail: recharts already in deps. No plotly, no d3.
 * Parses common tool output shapes: {bands: [[...]], kpath: [...], dos: [...]}
 */
import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

interface Props {
  content: string;
}

export default function BandStructureChart({ content }: Props) {
  const data = useMemo(() => {
    try {
      const parsed = JSON.parse(content);
      const bands = parsed.bands || parsed.band_structure || [];
      const kpath = parsed.kpath || parsed.k_points || [];
      const efermi = parsed.efermi ?? parsed.fermi_energy ?? 0;

      if (!Array.isArray(bands) || bands.length === 0) return null;

      // Build chart data: x = k-path index, y = energy
      const nPoints = Math.max(...bands.map((b: unknown[]) => Array.isArray(b) ? b.length : 0));
      const chartData = Array.from({ length: nPoints }, (_, i) => {
        const point: Record<string, number> = { k: i };
        bands.forEach((band: number[], bi: number) => {
          point[`band${bi}`] = band[i] ?? null;
        });
        return point;
      });

      // High-symmetry points from kpath
      const hspLabels = kpath.length > 0
        ? kpath.map((kp: unknown, i: number) => ({ x: i, label: typeof kp === 'string' ? kp : `K${i}` }))
        : [];

      return { chartData, nBands: bands.length, efermi, hspLabels };
    } catch {
      return null;
    }
  }, [content]);

  if (!data) {
    return <div className="p-4 text-xs text-text-muted">No band data found in result.</div>;
  }

  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <div style={{ width: '100%', height: 240 }}>
      <ResponsiveContainer>
        <LineChart data={data.chartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" opacity={0.3} />
          <XAxis
            dataKey="k"
            tick={{ fontSize: 9, fill: 'var(--fg-muted)' }}
            tickFormatter={(v: number) => data.hspLabels.find((h: { x: number; label: string }) => h.x === v)?.label ?? ''}
          />
          <YAxis tick={{ fontSize: 9, fill: 'var(--fg-muted)' }} />
          <Tooltip
            contentStyle={{ fontSize: 10, background: 'var(--bg-secondary)', border: '1px solid var(--border-default)' }}
          />
          <ReferenceLine y={data.efermi} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: 'E_F', fontSize: 9, fill: '#f59e0b' }} />
          {Array.from({ length: data.nBands }, (_, i) => (
            <Line
              key={i}
              type="monotone"
              dataKey={`band${i}`}
              stroke={colors[i % colors.length]}
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
