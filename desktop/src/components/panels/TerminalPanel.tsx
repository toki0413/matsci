import { invoke } from '@tauri-apps/api/core';
import { PanelHeader } from '../settings-shared';

interface TerminalPanelProps {
  terminalOutput: string;
  terminalInput: string;
  terminalEndRef: React.RefObject<HTMLDivElement>;
  setTerminalOutput: (v: string | ((prev: string) => string)) => void;
  setTerminalInput: (v: string) => void;
}

export function TerminalPanel({
  terminalOutput, terminalInput, terminalEndRef,
  setTerminalOutput, setTerminalInput,
}: TerminalPanelProps) {
  return (
    <div className="flex h-full flex-col bg-bg-tertiary text-text-primary">
      <PanelHeader title="Integrated Terminal">
        <button
          onClick={() => setTerminalOutput("")}
          className="btn-secondary px-3 py-1.5 text-xs"
        >
          Clear
        </button>
        <button
          onClick={() => invoke("stop_terminal")}
          className="btn-secondary px-3 py-1.5 text-xs"
        >
          Stop
        </button>
      </PanelHeader>
      <div className="flex-1 overflow-y-auto p-3 font-mono text-sm">
        <pre className="whitespace-pre-wrap break-all text-text-primary">
          {terminalOutput}
        </pre>
        <div ref={terminalEndRef} />
      </div>
      <div className="flex items-center gap-2 border-t border-border bg-bg-secondary p-3">
        <span className="font-mono text-sm text-accent">&gt;</span>
        <input
          type="text"
          value={terminalInput}
          onChange={(e) => setTerminalInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && terminalInput.trim()) {
              const cmd = terminalInput + "\r\n";
              setTerminalOutput((prev) => prev + "> " + terminalInput + "\n");
              invoke("write_terminal", { text: cmd }).catch((err) =>
                setTerminalOutput((prev) => prev + "[error] " + err + "\n")
              );
              setTerminalInput("");
            }
          }}
          placeholder="Type a command and press Enter"
          className="input flex-1 bg-bg-tertiary font-mono text-sm border-border text-text-primary"
          spellCheck={false}
        />
      </div>
    </div>
  );
}
