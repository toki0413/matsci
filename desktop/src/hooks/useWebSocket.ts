/**
 * useWebSocket — React hook for Huginn WebSocket agent connection.
 *
 * Wraps the production ReconnectingWebSocket from lib/ws-client.ts,
 * dispatches server messages to typed callbacks, and exposes a send()
 * method for client messages.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  ReconnectingWebSocket,
  type WsStatus,
} from '../lib/ws-client';
import { api, getAuthToken } from '../lib/api-client';

/** Map internal WS states to simpler public connection states */
function mapStatus(wsStatus: WsStatus): 'connected' | 'connecting' | 'disconnected' {
  switch (wsStatus) {
    case 'connected':
      return 'connected';
    case 'connecting':
    case 'reconnecting':
      return 'connecting';
    case 'failed':
    case 'idle':
    default:
      return 'disconnected';
  }
}

export interface WsCallbacks {
  onTextDelta?: (text: string) => void;
  onToolCall?: (id: string, name: string, args: unknown) => void;
  onToolResult?: (id: string, content: unknown) => void;
  onDone?: () => void;
  onError?: (error: string) => void;
  onAgentStatus?: (taskId: string, agentId: string, status: string, output?: unknown) => void;
  onAutoCheckpoint?: (id: string, base: string, files: string[]) => void;
}

/**
 * @param callbacks - Message dispatch callbacks (stable ref internally)
 * @param threadId  - Conversation thread ID (default "default")
 */
export function useWebSocket(callbacks: WsCallbacks, threadId = 'default') {
  const [status, setStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const wsRef = useRef<ReconnectingWebSocket | null>(null);
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  // Stable message handler
  const handleMessage = useCallback((data: unknown) => {
    const msg = data as Record<string, unknown>;
    if (!msg || typeof msg.type !== 'string') return;

    const cb = cbRef.current;
    switch (msg.type) {
      case 'text_delta':
        cb.onTextDelta?.(msg.text as string);
        break;
      case 'tool_call':
        cb.onToolCall?.(msg.id as string, msg.name as string, msg.args);
        break;
      case 'tool_result':
        cb.onToolResult?.(msg.id as string, msg.content);
        break;
      case 'done':
        cb.onDone?.();
        break;
      case 'error':
        cb.onError?.(msg.error as string);
        break;
      case 'agent_status':
        cb.onAgentStatus?.(
          msg.task_id as string,
          msg.agent_id as string,
          msg.status as string,
          msg.output,
        );
        break;
      case 'auto_checkpoint':
        cb.onAutoCheckpoint?.(
          msg.id as string,
          msg.base as string,
          msg.files as string[],
        );
        break;
      default:
        // unhandled message types (approval_request, exploration_result, etc.)
        break;
    }
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    const wsUrl = api.getWsUrl();

    const ws = new ReconnectingWebSocket({
      url: wsUrl,
      authToken: () => getAuthToken(),
      onStatus: (state) => setStatus(mapStatus(state)),
      onMessage: handleMessage,
    });

    ws.connect();
    wsRef.current = ws;

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [handleMessage]);

  /** Send a user_input message to the agent. */
  const send = useCallback(
    (content: string, opts: Record<string, unknown> = {}): boolean => {
      if (!wsRef.current) return false;
      return wsRef.current.send({
        type: 'user_input',
        content,
        thread_id: threadId,
        ...opts,
      });
    },
    [threadId],
  );

  return { status, send };
}
