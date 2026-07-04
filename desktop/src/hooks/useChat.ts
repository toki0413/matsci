/**
 * useChat — Chat state management hook.
 *
 * Manages the message list, integrates with useWebSocket for streaming,
 * handles text_delta accumulation and tool_call/tool_result matching.
 */
import { useState, useCallback, useRef } from 'react';
import { useWebSocket, type WsCallbacks } from './useWebSocket';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  isError?: boolean;
  toolName?: string;
  toolArgs?: unknown;
  toolStatus?: 'running' | 'done';
  toolResult?: string;
  toolCallId?: string;
}

export interface UseChatOptions {
  /** Conversation thread ID (default "default") */
  threadId?: string;
  /** Pre-built sample conversation for demo mode */
  sampleMessages?: ChatMessage[];
}

export function useChat(opts: UseChatOptions = {}) {
  const { threadId = 'default', sampleMessages } = opts;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  // Track the latest assistant message index for text_delta accumulation
  const assistantIdxRef = useRef(-1);
  // Track tool_call IDs for matching tool_result
  const toolMapRef = useRef(new Map<string, number>());

  const onTextDelta = useCallback((text: string) => {
    setMessages((prev) => {
      const idx = assistantIdxRef.current;
      if (idx < 0 || idx >= prev.length) {
        // Create a new assistant message
        const next = [...prev, { role: 'assistant' as const, content: text }];
        assistantIdxRef.current = next.length - 1;
        return next;
      }
      // Append to existing assistant message
      const updated = [...prev];
      updated[idx] = { ...updated[idx], content: updated[idx].content + text };
      return updated;
    });
  }, []);

  const onToolCall = useCallback((id: string, name: string, args: unknown) => {
    setMessages((prev) => {
      const msg: ChatMessage = {
        role: 'tool',
        content: '',
        toolName: name,
        toolArgs: args,
        toolStatus: 'running',
        toolResult: '',
        toolCallId: id,
      };
      toolMapRef.current.set(id, prev.length);
      return [...prev, msg];
    });
  }, []);

  const onToolResult = useCallback((id: string, content: unknown) => {
    setMessages((prev) => {
      const idx = toolMapRef.current.get(id);
      if (idx === undefined || idx >= prev.length) return prev;
      const updated = [...prev];
      updated[idx] = {
        ...updated[idx],
        toolStatus: 'done',
        toolResult: typeof content === 'string' ? content : JSON.stringify(content),
      };
      toolMapRef.current.delete(id);
      return updated;
    });
  }, []);

  const onDone = useCallback(() => {
    setIsStreaming(false);
    assistantIdxRef.current = -1;
  }, []);

  const onError = useCallback((error: string) => {
    setMessages((prev) => [
      ...prev,
      { role: 'assistant' as const, content: `Error: ${error}`, isError: true },
    ]);
    setIsStreaming(false);
    assistantIdxRef.current = -1;
  }, []);

  const callbacks: WsCallbacks = {
    onTextDelta,
    onToolCall,
    onToolResult,
    onDone,
    onError,
  };

  const { status: connectionStatus, send } = useWebSocket(callbacks, threadId);

  /** Send a user message. Adds user message to the list and triggers streaming. */
  const sendMessage = useCallback(
    (text: string) => {
      if (!text.trim()) return;

      const userMsg: ChatMessage = { role: 'user', content: text };
      setMessages((prev) => {
        const next = [...prev, userMsg];
        // Reset for incoming assistant response
        assistantIdxRef.current = next.length;
        return next;
      });
      setIsStreaming(true);
      send(text);
    },
    [send],
  );

  /** Load the sample/demo conversation. */
  const showSample = useCallback(() => {
    if (sampleMessages) {
      setMessages(sampleMessages);
    }
  }, [sampleMessages]);

  /** Clear messages (new chat). */
  const clearMessages = useCallback(() => {
    setMessages([]);
    assistantIdxRef.current = -1;
    toolMapRef.current.clear();
    setIsStreaming(false);
  }, []);

  return {
    messages,
    isStreaming,
    connectionStatus,
    sendMessage,
    showSample,
    clearMessages,
  };
}
