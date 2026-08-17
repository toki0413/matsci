import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const apiPost = vi.fn();

vi.mock('../lib/api', () => ({
  api: {
    get: vi.fn(),
    post: (...a: unknown[]) => apiPost(...a),
    put: vi.fn(),
    patch: vi.fn(),
    del: vi.fn(),
  },
}));

const { useTeam } = await import('./useTeam');

beforeEach(() => apiPost.mockReset());

describe('useTeam', () => {
  it('plans via the v2 endpoint when it succeeds', async () => {
    apiPost.mockResolvedValue({ success: true, tasks: [{ id: 't1' }] });
    const { result } = renderHook(() => useTeam());
    await act(async () => result.current.setTeamObjective('设计一个电池材料'));
    await act(async () => {
      await result.current.handleTeamPlan();
    });
    expect(apiPost).toHaveBeenCalledWith('/team/v2/plan', { objective: '设计一个电池材料' });
    expect(result.current.teamPlan).toEqual([{ id: 't1' }]);
    expect(result.current.teamRunning).toBe(false);
  });

  it('falls back to the legacy /team/plan when v2 is unavailable', async () => {
    apiPost.mockRejectedValueOnce(null); // v2 throws -> catch -> null
    apiPost.mockResolvedValueOnce({ success: true, tasks: [{ id: 'legacy' }] });
    const { result } = renderHook(() => useTeam());
    await act(async () => result.current.setTeamObjective('研究高温合金'));
    await act(async () => {
      await result.current.handleTeamPlan();
    });
    expect(apiPost).toHaveBeenNthCalledWith(2, '/team/plan', { objective: '研究高温合金' });
    expect(result.current.teamPlan).toEqual([{ id: 'legacy' }]);
  });

  it('records an error when planning fails on both endpoints', async () => {
    const { result } = renderHook(() => useTeam());
    await act(async () => result.current.setTeamObjective('x'));
    apiPost.mockRejectedValueOnce(null);
    apiPost.mockResolvedValueOnce({ success: false, error: 'planning unavailable' });
    await act(async () => {
      await result.current.handleTeamPlan();
    });
    expect(result.current.teamError).toContain('planning unavailable');
    expect(result.current.teamPlan).toBeNull();
  });

  it('does nothing on plan/run when the objective is blank', async () => {
    const { result } = renderHook(() => useTeam());
    await act(async () => {
      await result.current.handleTeamPlan();
      await result.current.handleTeamRun();
    });
    expect(apiPost).not.toHaveBeenCalled();
  });

  it('runs a team job and stores the full result', async () => {
    apiPost.mockResolvedValueOnce(null); // v2 throws
    apiPost.mockResolvedValueOnce({ success: true, output: 'ok' });
    const { result } = renderHook(() => useTeam());
    await act(async () => result.current.setTeamObjective('多智能体协作'));
    await act(async () => {
      await result.current.handleTeamRun();
    });
    expect(apiPost).toHaveBeenCalledWith('/team/run', { objective: '多智能体协作' });
    expect(result.current.teamResult).toEqual(expect.objectContaining({ success: true, output: 'ok' }));
  });

  it('fuses opinions via /team/v2/fusion', async () => {
    apiPost.mockResolvedValue({ success: true, consensus: 'x' });
    const { result } = renderHook(() => useTeam());
    await act(async () => result.current.setTeamObjective('评估两种建模路径'));
    await act(async () => {
      await result.current.handleTeamFusion(3);
    });
    expect(apiPost).toHaveBeenCalledWith('/team/v2/fusion', { query: '评估两种建模路径', rounds: 3 });
    expect(result.current.teamFusionResult).toEqual(expect.objectContaining({ success: true }));
  });
});