/**
 * useHPC — Manages HPC (High-Performance Computing) state and API calls.
 *
 * Encapsulates connection testing, job submission, and status checking
 * for Slurm/PBS schedulers.
 */
import { useState } from 'react';
import { api } from '../lib/api';

export function useHPC() {
  const [hpcHost, setHpcHost] = useState('');
  const [hpcUsername, setHpcUsername] = useState('');
  const [hpcScheduler, setHpcScheduler] = useState<'slurm' | 'pbs'>('slurm');
  const [hpcKeyPath, setHpcKeyPath] = useState('');
  const [hpcCommand, setHpcCommand] = useState('');
  const [hpcJobName, setHpcJobName] = useState('huginn_job');
  const [hpcWalltime, setHpcWalltime] = useState('01:00:00');
  const [hpcNodes, setHpcNodes] = useState(1);
  const [hpcNtasks, setHpcNtasks] = useState(4);
  const [hpcQueue, setHpcQueue] = useState('');
  const [hpcJobId, setHpcJobId] = useState('');
  const [hpcRunning, setHpcRunning] = useState(false);
  const [hpcResult, setHpcResult] = useState<any>(null);
  const [hpcError, setHpcError] = useState('');

  const handleHpcTest = async () => {
    setHpcRunning(true);
    setHpcError('');
    setHpcResult(null);
    try {
      const data = await api.post<{ success?: boolean; error?: string } & Record<string, any>>(
        '/hpc/test',
        { host: hpcHost, username: hpcUsername, scheduler: hpcScheduler, key_path: hpcKeyPath || undefined }
      );
      if (data.success) {
        setHpcResult(data);
      } else {
        setHpcError(data.error || 'HPC test failed.');
      }
    } catch (e: any) {
      setHpcError(e.message || 'Network error');
    } finally {
      setHpcRunning(false);
    }
  };

  const handleHpcSubmit = async () => {
    if (!hpcCommand.trim()) return;
    setHpcRunning(true);
    setHpcError('');
    setHpcResult(null);
    try {
      const data = await api.post<{ success?: boolean; job_id?: string; error?: string } & Record<string, any>>(
        '/hpc/submit',
        {
          host: hpcHost,
          username: hpcUsername,
          scheduler: hpcScheduler,
          key_path: hpcKeyPath || undefined,
          command: hpcCommand,
          job_name: hpcJobName,
          walltime: hpcWalltime,
          nodes: hpcNodes,
          ntasks_per_node: hpcNtasks,
          queue: hpcQueue || undefined,
        }
      );
      if (data.success) {
        setHpcJobId(data.job_id ?? '');
        setHpcResult(data);
      } else {
        setHpcError(data.error || 'HPC submit failed.');
      }
    } catch (e: any) {
      setHpcError(e.message || 'Network error');
    } finally {
      setHpcRunning(false);
    }
  };

  const handleHpcStatus = async () => {
    if (!hpcJobId.trim()) return;
    setHpcRunning(true);
    setHpcError('');
    try {
      const data = await api.post<{ success?: boolean; error?: string } & Record<string, any>>(
        '/hpc/status',
        {
          host: hpcHost,
          username: hpcUsername,
          scheduler: hpcScheduler,
          key_path: hpcKeyPath || undefined,
          job_id: hpcJobId,
        }
      );
      if (data.success) {
        setHpcResult(data);
      } else {
        setHpcError(data.error || 'HPC status failed.');
      }
    } catch (e: any) {
      setHpcError(e.message || 'Network error');
    } finally {
      setHpcRunning(false);
    }
  };

  return {
    hpcHost, hpcUsername, hpcScheduler, hpcKeyPath, hpcCommand,
    hpcJobName, hpcWalltime, hpcNodes, hpcNtasks, hpcQueue,
    hpcJobId, hpcRunning, hpcResult, hpcError,
    setHpcHost, setHpcUsername, setHpcScheduler, setHpcKeyPath,
    setHpcCommand, setHpcJobName, setHpcWalltime, setHpcNodes,
    setHpcNtasks, setHpcQueue, setHpcJobId,
    handleHpcTest, handleHpcSubmit, handleHpcStatus,
  };
}
