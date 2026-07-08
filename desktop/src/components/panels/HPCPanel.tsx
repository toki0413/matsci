interface HPCPanelProps {
  isConnected: boolean;
  hpcHost: string;
  hpcUsername: string;
  hpcScheduler: string;
  hpcKeyPath: string;
  hpcCommand: string;
  hpcJobName: string;
  hpcWalltime: string;
  hpcNodes: number;
  hpcNtasks: number;
  hpcQueue: string;
  hpcJobId: string;
  hpcRunning: boolean;
  hpcResult: any;
  hpcError: string;
  setHpcHost: (v: string) => void;
  setHpcUsername: (v: string) => void;
  setHpcScheduler: (v: "slurm" | "pbs") => void;
  setHpcKeyPath: (v: string) => void;
  setHpcCommand: (v: string) => void;
  setHpcJobName: (v: string) => void;
  setHpcWalltime: (v: string) => void;
  setHpcNodes: (v: number) => void;
  setHpcNtasks: (v: number) => void;
  setHpcQueue: (v: string) => void;
  setHpcJobId: (v: string) => void;
  handleHpcTest: () => void;
  handleHpcSubmit: () => void;
  handleHpcStatus: () => void;
}

export function HPCPanel({
  isConnected, hpcHost, hpcUsername, hpcScheduler, hpcKeyPath, hpcCommand,
  hpcJobName, hpcWalltime, hpcNodes, hpcNtasks, hpcQueue, hpcJobId,
  hpcRunning, hpcResult, hpcError,
  setHpcHost, setHpcUsername, setHpcScheduler, setHpcKeyPath,
  setHpcCommand, setHpcJobName, setHpcWalltime, setHpcNodes,
  setHpcNtasks, setHpcQueue, setHpcJobId,
  handleHpcTest, handleHpcSubmit, handleHpcStatus,
}: HPCPanelProps) {
  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mx-auto max-w-3xl space-y-5">
        <div className="card">
          <h2 className="mb-2 text-base font-semibold">HPC</h2>
          <p className="text-sm text-text-secondary">Submit and monitor jobs on a remote cluster.</p>
        </div>
        <div className="card space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <input type="text" value={hpcHost} onChange={(e) => setHpcHost(e.target.value)} placeholder="Host" className="input text-xs" />
            <input type="text" value={hpcUsername} onChange={(e) => setHpcUsername(e.target.value)} placeholder="Username" className="input text-xs" />
            <select value={hpcScheduler} onChange={(e) => setHpcScheduler(e.target.value as "slurm" | "pbs")} className="input text-xs">
              <option value="slurm">SLURM</option>
              <option value="pbs">PBS</option>
            </select>
            <input type="text" value={hpcKeyPath} onChange={(e) => setHpcKeyPath(e.target.value)} placeholder="SSH key path (optional)" className="input text-xs" />
          </div>
          <button onClick={handleHpcTest} disabled={hpcRunning || !isConnected || !hpcHost || !hpcUsername} className="btn-secondary text-xs">
            Test connection
          </button>
          <hr className="border-border" />
          <input type="text" value={hpcCommand} onChange={(e) => setHpcCommand(e.target.value)} placeholder="Command to run" className="input text-sm" />
          <div className="grid grid-cols-3 gap-3">
            <input type="text" value={hpcJobName} onChange={(e) => setHpcJobName(e.target.value)} placeholder="Job name" className="input text-xs" />
            <input type="text" value={hpcWalltime} onChange={(e) => setHpcWalltime(e.target.value)} placeholder="Walltime" className="input text-xs" />
            <input type="text" value={hpcQueue} onChange={(e) => setHpcQueue(e.target.value)} placeholder="Queue" className="input text-xs" />
            <input type="number" min={1} value={hpcNodes} onChange={(e) => setHpcNodes(parseInt(e.target.value || "1", 10))} placeholder="Nodes" className="input text-xs" />
            <input type="number" min={1} value={hpcNtasks} onChange={(e) => setHpcNtasks(parseInt(e.target.value || "1", 10))} placeholder="Tasks/node" className="input text-xs" />
            <input type="text" value={hpcJobId} onChange={(e) => setHpcJobId(e.target.value)} placeholder="Job ID" className="input text-xs" />
          </div>
          <div className="flex items-center gap-2">
            <button onClick={handleHpcSubmit} disabled={hpcRunning || !isConnected || !hpcCommand.trim()} className="btn-primary text-xs">
              Submit
            </button>
            <button onClick={handleHpcStatus} disabled={hpcRunning || !isConnected || !hpcJobId.trim()} className="btn-secondary text-xs">
              Status
            </button>
          </div>
          {hpcError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{hpcError}</div>}
        </div>
        {hpcResult && (
          <div className="card">
            <h3 className="text-sm font-semibold mb-2">Result</h3>
            <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(hpcResult, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
