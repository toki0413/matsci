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

const { useHPC } = await import('./useHPC');

beforeEach(() => apiPost.mockReset());

describe('useHPC', () => {
  it('tests the HPC connection and stores the result', async () => {
    apiPost.mockResolvedValue({ success: true, latency_ms: 12 });
    const { result } = renderHook(() => useHPC());
    await act(async () => result.current.setHpcHost('login'));
    await act(async () => result.current.setHpcUsername('alice'));
    await act(async () => {
      await result.current.handleHpcTest();
    });
    expect(apiPost).toHaveBeenCalledWith('/hpc/test', expect.objectContaining({ host: 'login', username: 'alice', scheduler: 'slurm' }));
    expect(result.current.hpcResult).toEqual(expect.objectContaining({ success: true }));
    expect(result.current.hpcRunning).toBe(false);
  });

  it('records an error when the HPC test returns success:false', async () => {
    apiPost.mockResolvedValue({ success: false, error: 'auth failed' });
    const { result } = renderHook(() => useHPC());
    await act(async () => {
      await result.current.handleHpcTest();
    });
    expect(result.current.hpcError).toContain('auth failed');
  });

  it('submits a job and stores the returned job id', async () => {
    apiPost.mockResolvedValue({ success: true, job_id: 'slurm-77' });
    const { result } = renderHook(() => useHPC());
    await act(async () => result.current.setHpcCommand('sbatch run.sh'));
    await act(async () => {
      await result.current.handleHpcSubmit();
    });
    expect(apiPost).toHaveBeenCalledWith(
      '/hpc/submit',
      expect.objectContaining({ command: 'sbatch run.sh', job_name: 'huginn_job', nodes: 1, ntasks_per_node: 4 }),
    );
    expect(result.current.hpcJobId).toBe('slurm-77');
  });

  it('skips submission when the command is blank', async () => {
    const { result } = renderHook(() => useHPC());
    await act(async () => {
      await result.current.handleHpcSubmit();
    });
    expect(apiPost).not.toHaveBeenCalled();
  });

  it('checks job status only when a job id exists', async () => {
    apiPost.mockResolvedValue({ success: true, state: 'RUNNING' });
    const { result } = renderHook(() => useHPC());
    await act(async () => result.current.setHpcJobId('j1'));
    await act(async () => {
      await result.current.handleHpcStatus();
    });
    expect(apiPost).toHaveBeenCalledWith('/hpc/status', expect.objectContaining({ job_id: 'j1' }));
    expect(result.current.hpcResult.state).toBe('RUNNING');

    const { result: empty } = renderHook(() => useHPC());
    apiPost.mockClear();
    await act(async () => {
      await empty.current.handleHpcStatus();
    });
    expect(apiPost).not.toHaveBeenCalled();
  });
});