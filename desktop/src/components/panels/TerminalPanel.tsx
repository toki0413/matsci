import { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { PanelHeader } from '../settings-shared';
import { ReconnectingWebSocket, type WsStatus } from '../../lib/ws-client';
import { getApiBase, getAuthToken } from '../../lib/api-client';

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
  const { t } = useTranslation();
  const [isRemote, setIsRemote] = useState(false);
  const [remoteStatus, setRemoteStatus] = useState<WsStatus>('idle');
  const wsRef = useRef<ReconnectingWebSocket | null>(null);

  const toggleRemote = () => {
    if (isRemote) {
      wsRef.current?.close();
      wsRef.current = null;
      setRemoteStatus('idle');
      setIsRemote(false);
      return;
    }
    const wsUrl = getApiBase().replace(/^http/, 'ws') + '/ws/terminal';
    const ws = new ReconnectingWebSocket({
      url: wsUrl,
      authToken: getAuthToken,
      onStatus: (s) => setRemoteStatus(s),
      onMessage: (data) => {
        const text = typeof data === 'string' ? data
          : (data as any)?.data ?? (data as any)?.output ?? JSON.stringify(data);
        setTerminalOutput((prev) => prev + text);
      },
    });
    ws.connect();
    wsRef.current = ws;
    setIsRemote(true);
    setTerminalOutput((prev) => prev + `[remote] connecting to ${wsUrl}...\n`);
  };

  useEffect(() => () => { wsRef.current?.close(); }, []);

  const handleSend = () => {
    if (!terminalInput.trim()) return;
    const cmd = terminalInput + "\r\n";
    setTerminalOutput((prev) => prev + "> " + terminalInput + "\n");
    if (isRemote && wsRef.current) {
      wsRef.current.send({ type: 'input', data: cmd });
    } else {
      invoke("write_terminal", { text: cmd }).catch((err) =>
        setTerminalOutput((prev) => prev + "[error] " + err + "\n")
      );
    }
    setTerminalInput("");
  };

  return (
    <div className="flex h-full flex-col bg-bg-tertiary text-text-primary">
      <PanelHeader title={t('terminal.title')}>
        <button
          onClick={toggleRemote}
          className={`px-3 py-1.5 text-xs ${isRemote ? 'btn-primary' : 'btn-secondary'}`}
        >
          {isRemote ? `${t('terminal.remote')}: ${remoteStatus}` : t('terminal.remote')}
        </button>
        <button
          onClick={() => setTerminalOutput("")}
          className="btn-secondary px-3 py-1.5 text-xs"
        >
          {t('terminal.clear')}
        </button>
        {!isRemote && (
          <button
            onClick={() => invoke("stop_terminal")}
            className="btn-secondary px-3 py-1.5 text-xs"
          >
            {t('terminal.stop')}
          </button>
        )}
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
          onKeyDown={(e) => { if (e.key === "Enter") handleSend(); }}
          placeholder={isRemote ? t('terminal.remotePh') : t('terminal.localPh')}
          className="input flex-1 bg-bg-tertiary font-mono text-sm border-border text-text-primary"
          spellCheck={false}
        />
      </div>
    </div>
  );
}
