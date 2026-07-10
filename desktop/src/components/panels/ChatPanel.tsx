import { Search, X } from 'lucide-react';
import { Virtuoso } from 'react-virtuoso';
import { useTranslation } from 'react-i18next';
import { ToolResultRenderer } from '../ToolResultRenderer';
import { SaveToMemoryButton } from '../SaveToMemoryButton';
import MessageContent from '../MessageContent';
import type { Message } from '../../hooks/useChatAndConnection';
import type { ReconnectingWebSocket } from '../../lib/ws-client';

interface ChatPanelProps {
  messages: Message[];
  chatSearchOpen: boolean;
  chatSearchQuery: string;
  setChatSearchOpen: (v: boolean | ((p: boolean) => boolean)) => void;
  setChatSearchQuery: (v: string) => void;
  wsClientRef: React.RefObject<ReconnectingWebSocket | null>;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  answerClarification: (questionId: string | undefined, answer: string) => void;
  pendingClarifications: any[];
  isConnected: boolean;
  sendMessage: () => void;
  pendingPlan: string;
  setPendingPlan: (v: string) => void;
  planLoading: boolean;
  setMode: (v: "chat" | "plan" | "build") => void;
  input: string;
  setInput: (v: string) => void;
  mode: "chat" | "plan" | "build";
  isStreaming: boolean;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  pendingApproval: {
    request_id: string;
    tool_name: string;
    reason: string;
    dangerous: boolean;
  } | null;
  respondToApproval: (requestId: string, approved: boolean) => void;
  autoApprove: boolean;
  toggleAutoApprove: (enabled: boolean) => void;
  thinkingIntensity: "low" | "medium" | "high";
  setThinkingIntensity: (v: "low" | "medium" | "high") => void;
  pendingMessages: string[];
  researchMode: boolean;
  setResearchMode: (v: boolean) => void;
  autoloopPhase: string;
  autoloopProgress: number;
}

export function ChatPanel(props: ChatPanelProps) {
  const { t } = useTranslation();
  const {
    messages, chatSearchOpen, chatSearchQuery, setChatSearchOpen, setChatSearchQuery,
    wsClientRef, setMessages, answerClarification, pendingClarifications,
    isConnected, sendMessage, pendingPlan, setPendingPlan, planLoading,
    setMode, input, setInput, mode, isStreaming, messagesEndRef,
    pendingApproval, respondToApproval, autoApprove, toggleAutoApprove,
    thinkingIntensity, setThinkingIntensity,
    pendingMessages, researchMode, setResearchMode,
    autoloopPhase, autoloopProgress,
  } = props;

  // Cline pattern: combine consecutive tool calls into groups
  function groupMessages(msgs: Message[]): Message[] {
    const result: Message[] = [];
    let toolGroup: Message[] = [];

    for (const msg of msgs) {
      if (msg.role === "tool") {
        toolGroup.push(msg);
      } else {
        if (toolGroup.length > 0) {
          if (toolGroup.length === 1) {
            result.push(toolGroup[0]);
          } else {
            result.push({
              role: "tool_group" as any,
              tool_calls: toolGroup,
              timestamp: toolGroup[0].timestamp,
            } as any);
          }
          toolGroup = [];
        }
        result.push(msg);
      }
    }
    // flush remaining
    if (toolGroup.length > 0) {
      if (toolGroup.length === 1) {
        result.push(toolGroup[0]);
      } else {
        result.push({
          role: "tool_group" as any,
          tool_calls: toolGroup,
          timestamp: toolGroup[0].timestamp,
        } as any);
      }
    }
    return result;
  }

  const filteredMessages = chatSearchQuery.trim()
    ? messages.filter((m) => m.content.toLowerCase().includes(chatSearchQuery.toLowerCase()))
    : messages;
  const groupedMessages = groupMessages(filteredMessages);

  const AUTOLOOP_PHASES = ["perceive", "hypothesize", "plan", "execute", "validate", "learn", "report"];

  return (
    <div className="flex h-full flex-col">
      {chatSearchOpen && (
        <div className="flex items-center gap-2 border-b border-border bg-bg-secondary/50 px-6 py-2">
          <Search size={14} className="shrink-0 text-text-muted" />
          <input
            type="text"
            autoFocus
            value={chatSearchQuery}
            onChange={(e) => setChatSearchQuery(e.target.value)}
            placeholder={t('chat.searchPlaceholder')}
            className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
          />
          {chatSearchQuery && (
            <span className="shrink-0 text-[11px] text-text-muted">
              {messages.filter((m) => m.content.toLowerCase().includes(chatSearchQuery.toLowerCase())).length} {t('chat.matches')}
            </span>
          )}
          <button onClick={() => { setChatSearchOpen(false); setChatSearchQuery(""); }} className="shrink-0 text-text-muted hover:text-text-secondary" aria-label="Close search">
            <X size={14} />
          </button>
        </div>
      )}
      <Virtuoso
        data={groupedMessages}
        className="cv-list flex-1"
        style={{ height: '100%' }}
        itemContent={(index, msg) => {
          if ((msg as any).role === "tool_group" && (msg as any).tool_calls) {
            return (
              <div key={index} className="flex justify-center">
                <div className="w-full max-w-2xl rounded-xl border border-border bg-bg-secondary p-4 shadow-sm">
                  <div className="flex items-center gap-2 text-sm font-semibold text-accent">
                    <span>🔧</span>
                    <span>{(msg as any).tool_calls.length} tool calls</span>
                  </div>
                  <div className="mt-2 space-y-2">
                    {(msg as any).tool_calls.map((tc: Message, ti: number) => (
                      <details key={ti} className="rounded-lg bg-bg-tertiary p-2">
                        <summary className="cursor-pointer text-xs font-medium text-text-secondary">
                          {tc.tool_name} {tc.tool_status === "running" && "⟳"}
                          {tc.tool_status === "done" && "✓"}
                        </summary>
                        <pre className="mt-1 max-h-40 overflow-auto text-xs">
                          {JSON.stringify(tc.tool_args, null, 2)}
                        </pre>
                        {tc.tool_status === "done" && tc.tool_result !== undefined && (
                          <ToolResultRenderer content={tc.tool_result} toolName={tc.tool_name} />
                        )}
                      </details>
                    ))}
                  </div>
                </div>
              </div>
            );
          }
          if (msg.role === "tool") {
            return (
              <div key={index} className="flex justify-center">
                <div className="w-full max-w-2xl rounded-xl border border-border bg-bg-secondary p-4 shadow-sm">
                  <div className="flex items-center gap-2 text-sm font-semibold text-accent">
                    <span>🔧</span>
                    <span>{msg.tool_name}</span>
                    {msg.tool_status === "running" && (
                      <span className="ml-2 inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                    )}
                    {msg.tool_status === "done" && (
                      <span className="ml-2 text-xs text-success">{t('chat.done')}</span>
                    )}
                  </div>
                  <div className="mt-2 text-xs text-text-secondary">
                    {t('chat.arguments')}
                  </div>
                  <pre className="mt-1 max-h-40 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs">
                    {JSON.stringify(msg.tool_args, null, 2)}
                  </pre>
                  {msg.tool_status === "done" && msg.tool_result !== undefined && (
                    <>
                      <div className="mt-3 text-xs text-text-secondary">
                        {t('chat.result')}
                      </div>
                      <ToolResultRenderer content={msg.tool_result} toolName={msg.tool_name} />
                    </>
                  )}
                </div>
              </div>
            );
          }
          return (
            <div
              key={index}
              className={`flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
            >
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm ${
                  msg.role === "user" ? "bg-accent text-white" : "bg-bg-tertiary text-text-secondary"
                }`}
              >
                {msg.role === "user" ? t('chat.you') : t('chat.ai')}
              </div>
              <div
                className={`max-w-[75%] px-5 py-3 ${
                  msg.role === "user"
                    ? "bg-accent text-white rounded-2xl rounded-br-none"
                    : "rounded-2xl rounded-bl-none"
                }`}
              >
                <div className="mb-1 flex items-center gap-2 text-xs opacity-70">
                  <span>{msg.role === "user" ? t('chat.you') : t('chat.assistant')}</span>
                  <span>
                    {msg.timestamp === "streaming" ? t('chat.typing') : msg.timestamp}
                  </span>
                </div>
                {msg.reasoning && (
                  <details
                    className={`mb-2 rounded-lg ${msg.reasoning && !msg.content ? "" : ""}`}
                    open={msg.timestamp === "streaming" && !msg.content}
                  >
                    <summary className="cursor-pointer select-none text-xs font-medium text-text-muted hover:text-text-secondary">
                      {msg.timestamp === "streaming" && !msg.content
                        ? t('chat.thinking')
                        : t('chat.thoughtProcess')}
                    </summary>
                    <div className="mt-1.5 max-h-60 overflow-y-auto whitespace-pre-wrap border-l-2 border-border pl-3 text-xs italic leading-relaxed text-text-muted opacity-80">
                      {msg.reasoning}
                    </div>
                  </details>
                )}
                <div className="text-[15px] leading-relaxed">
                  {msg.content && <MessageContent content={msg.content} />}
                </div>
                {msg.role === "assistant" && msg.content && msg.timestamp !== "streaming" && (
                  <div className="mt-1.5 flex justify-end">
                    <SaveToMemoryButton content={msg.content} />
                  </div>
                )}
                {/* Plan confirm/cancel buttons */}
                {msg.isPlan && msg.planId && (
                  <div className="mt-3 flex gap-2 border-t border-border/50 pt-3">
                    <button
                      onClick={() => {
                        if (wsClientRef.current) {
                          wsClientRef.current.send(JSON.stringify({
                            type: "plan_confirm",
                            plan_id: msg.planId,
                            confirmed: true,
                          }));
                        }
                        setMessages((prev) => prev.map((m) =>
                          m === msg ? { ...m, planConfirmed: true } : m
                        ));
                      }}
                      disabled={msg.planConfirmed}
                      className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {msg.planConfirmed ? t('chat.confirmed') : t('chat.confirmExecute')}
                    </button>
                    <button
                      onClick={() => {
                        if (wsClientRef.current) {
                          wsClientRef.current.send(JSON.stringify({
                            type: "plan_confirm",
                            plan_id: msg.planId,
                            confirmed: false,
                          }));
                        }
                        setMessages((prev) => prev.map((m) =>
                          m === msg ? { ...m, planConfirmed: true } : m
                        ));
                      }}
                      disabled={msg.planConfirmed}
                      className="rounded-lg border border-border px-4 py-2 text-sm text-text-secondary hover:bg-bg-tertiary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                )}
                {/* Interactive clarification question cards */}
                {msg.isClarification && msg.clarifications && (
                  <div className="mt-3 space-y-2 border-t border-border/50 pt-3">
                    {msg.clarifications.map((q: any, qi: number) => (
                      <div key={qi} className="rounded-lg bg-bg-tertiary p-3">
                        <div className="text-sm font-medium text-text-primary">
                          {q.question || q}
                        </div>
                        {q.options && q.options.length > 0 ? (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {q.options.map((opt: string) => (
                              <button
                                key={opt}
                                onClick={() => answerClarification(q.question_id, opt)}
                                disabled={pendingClarifications.length === 0}
                                className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-1.5 text-sm text-accent hover:bg-accent/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                              >
                                {opt}
                              </button>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    ))}
                    <div className="text-xs text-text-tertiary">
                      {t('chat.clarificationHint')}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        }}
        components={{
          Footer: () => <div ref={messagesEndRef} style={{ height: 1 }} />,
        }}
        followOutput={'auto'}
      />

      {autoloopPhase && (
        <div className="flex items-center gap-2 border-b border-border bg-accent/5 px-6 py-1.5">
          <span className="text-xs">🔬</span>
          <div className="flex items-center gap-1 text-[11px]">
            {AUTOLOOP_PHASES.map((phase, i, arr) => (
              <span key={phase} className="flex items-center gap-1">
                <span
                  className={
                    phase === autoloopPhase
                      ? "font-bold text-accent"
                      : AUTOLOOP_PHASES.indexOf(phase) < AUTOLOOP_PHASES.indexOf(autoloopPhase)
                      ? "text-accent/50"
                      : "text-text-muted"
                  }
                >
                  {phase}
                </span>
                {i < arr.length - 1 && <span className="text-text-muted">→</span>}
              </span>
            ))}
          </div>
          <span className="ml-auto text-[11px] text-text-muted">{autoloopProgress}%</span>
        </div>
      )}

      <div className="border-t border-border bg-bg-secondary p-4">
        {!isConnected && (
          <div className="mb-3 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
            {t('chat.backendNotConnected')}
          </div>
        )}

        {pendingClarifications.length > 0 && (
          <div className="mb-3 rounded-lg border border-accent/20 bg-accent/5 px-3 py-2 text-xs text-accent">
            💡 Agent is waiting for your clarification — answer above or type below
          </div>
        )}

        {pendingPlan && (
          <div className="mb-3 rounded-xl border border-border bg-bg-tertiary p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-semibold text-accent">📋 Plan</span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPendingPlan("")}
                  className="text-xs text-text-secondary hover:text-text-primary"
                >
                  Dismiss
                </button>
                <button
                  onClick={() => {
                    setMode("build");
                    sendMessage();
                  }}
                  disabled={!input.trim() || planLoading}
                  className="btn-primary px-3 py-1 text-xs"
                >
                  {t('chat.runPlan')}
                </button>
              </div>
            </div>
            <div className="max-h-48 overflow-y-auto whitespace-pre-wrap text-xs text-text-primary">
              <MessageContent content={pendingPlan} />
            </div>
          </div>
        )}

        {pendingApproval && (
          <div
            className={`mb-3 rounded-xl border-2 p-3 ${
              pendingApproval.dangerous
                ? "border-error bg-error/5"
                : "border-warning bg-warning/5"
            }`}
          >
            <div className="mb-1.5 flex items-center gap-2">
              <span className={`text-sm font-bold ${pendingApproval.dangerous ? "text-error" : "text-warning"}`}>
                {pendingApproval.dangerous ? `🔴 ${t('chat.approvalRequired')}` : `⚠️ ${t('chat.approvalRequired')}`}
              </span>
              <span className="rounded bg-bg-tertiary px-2 py-0.5 text-xs font-mono text-text-secondary">
                {pendingApproval.tool_name}
              </span>
            </div>
            <p className="mb-3 text-xs text-text-secondary">{pendingApproval.reason}</p>
            <div className="flex gap-2">
              <button
                onClick={() => respondToApproval(pendingApproval.request_id, true)}
                className="rounded-lg bg-success px-4 py-2 text-sm font-medium text-white hover:bg-success/80 transition-colors"
              >
                {t('chat.approve')}
              </button>
              <button
                onClick={() => respondToApproval(pendingApproval.request_id, false)}
                className="rounded-lg bg-error px-4 py-2 text-sm font-medium text-white hover:bg-error/80 transition-colors"
              >
                {t('chat.deny')}
              </button>
            </div>
          </div>
        )}

        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-text-muted">
              <input
                type="checkbox"
                checked={autoApprove}
                onChange={(e) => toggleAutoApprove(e.target.checked)}
                className="h-3 w-3"
              />
              Auto-approve
            </label>

            {/* Thinking intensity segmented control */}
            <div className="flex items-center gap-1 text-[10px] text-text-muted">
              <span>🧠</span>
              {(["low", "medium", "high"] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setThinkingIntensity(level)}
                  className={`rounded px-1.5 py-0.5 transition-colors ${
                    thinkingIntensity === level
                      ? "bg-accent/20 text-accent font-medium"
                      : "text-text-muted hover:text-text-secondary"
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>

            {/* Plan mode toggle */}
            <button
              onClick={() => {
                if (mode === "plan") {
                  setMode("chat");
                } else {
                  setMode("plan");
                  setResearchMode(false);
                }
              }}
              disabled={researchMode}
              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-colors ${
                mode === "plan"
                  ? "bg-accent/20 text-accent font-medium"
                  : "text-text-muted hover:text-text-secondary"
              } ${researchMode ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              📋 Plan Mode
            </button>

            {/* Research mode toggle */}
            <button
              onClick={() => {
                const next = !researchMode;
                setResearchMode(next);
                if (next && mode === "plan") setMode("chat");
              }}
              disabled={mode === "plan"}
              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-colors ${
                researchMode
                  ? "bg-accent/20 text-accent font-medium"
                  : "text-text-muted hover:text-text-secondary"
              } ${mode === "plan" ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              🔬 Research Mode
            </button>
          </div>

          {mode !== "chat" && (
            <button
              onClick={() => setMode("chat")}
              className="text-[10px] text-accent hover:underline"
            >
              {t('chat.backToChat')}
            </button>
          )}
        </div>

        {pendingMessages.length > 0 && (
          <div className="mb-2 rounded-lg border border-accent/20 bg-accent/5 px-3 py-1.5 text-[11px] text-accent">
            📋 {pendingMessages.length} message{pendingMessages.length > 1 ? "s" : ""} queued — will send after current response
          </div>
        )}

        <div className="flex items-end gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder={
              mode === "plan"
                ? t('chat.placeholderPlan')
                : mode === "build"
                ? t('chat.placeholderBuild')
                : isConnected
                ? t('chat.placeholderConnected')
                : t('chat.placeholderOffline')
            }
            rows={2}
            disabled={!isConnected || planLoading}
            className="input min-h-[56px] resize-none flex-1"
          />
          <button
            onClick={sendMessage}
            disabled={!isConnected || !input.trim() || planLoading}
            className="btn-primary h-11 px-5"
          >
            {planLoading ? t('chat.planning') : isStreaming ? t('chat.streaming') : mode === "plan" ? t('chat.mode.plan') : t('chat.send')}
          </button>
        </div>
      </div>
    </div>
  );
}
