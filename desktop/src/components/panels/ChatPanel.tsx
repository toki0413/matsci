import { Search, X } from 'lucide-react';
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
}

export function ChatPanel(props: ChatPanelProps) {
  const { t } = useTranslation();
  const {
    messages, chatSearchOpen, chatSearchQuery, setChatSearchOpen, setChatSearchQuery,
    wsClientRef, setMessages, answerClarification, pendingClarifications,
    isConnected, sendMessage, pendingPlan, setPendingPlan, planLoading,
    setMode, input, setInput, mode, isStreaming, messagesEndRef,
    pendingApproval, respondToApproval, autoApprove, toggleAutoApprove,
  } = props;

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
            placeholder="Search messages…"
            className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
          />
          {chatSearchQuery && (
            <span className="shrink-0 text-[11px] text-text-muted">
              {messages.filter((m) => m.content.toLowerCase().includes(chatSearchQuery.toLowerCase())).length} matches
            </span>
          )}
          <button onClick={() => { setChatSearchOpen(false); setChatSearchQuery(""); }} className="shrink-0 text-text-muted hover:text-text-secondary" aria-label="Close search">
            <X size={14} />
          </button>
        </div>
      )}
      <div className="cv-list flex-1 overflow-y-auto p-6 space-y-5">
        {(chatSearchQuery.trim()
          ? messages.filter((m) => m.content.toLowerCase().includes(chatSearchQuery.toLowerCase()))
          : messages
        ).map((msg, i) => {
          if (msg.role === "tool") {
            return (
              <div key={i} className="flex justify-center">
                <div className="w-full max-w-2xl rounded-xl border border-border bg-bg-secondary p-4 shadow-sm">
                  <div className="flex items-center gap-2 text-sm font-semibold text-accent">
                    <span>🔧</span>
                    <span>{msg.tool_name}</span>
                    {msg.tool_status === "running" && (
                      <span className="ml-2 inline-flex h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                    )}
                    {msg.tool_status === "done" && (
                      <span className="ml-2 text-xs text-success">done</span>
                    )}
                  </div>
                  <div className="mt-2 text-xs text-text-secondary">
                    Arguments
                  </div>
                  <pre className="mt-1 max-h-40 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs">
                    {JSON.stringify(msg.tool_args, null, 2)}
                  </pre>
                  {msg.tool_status === "done" && msg.tool_result !== undefined && (
                    <>
                      <div className="mt-3 text-xs text-text-secondary">
                        Result
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
              key={i}
              className={`flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
            >
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm ${
                  msg.role === "user" ? "bg-accent text-white" : "bg-bg-tertiary text-text-secondary"
                }`}
              >
                {msg.role === "user" ? "You" : "AI"}
              </div>
              <div
                className={`max-w-[75%] rounded-2xl px-5 py-3 ${
                  msg.role === "user"
                    ? "bg-accent text-white rounded-br-none"
                    : "bg-bg-secondary border border-border rounded-bl-none"
                }`}
              >
                <div className="mb-1 flex items-center gap-2 text-xs opacity-70">
                  <span>{msg.role === "user" ? "You" : "Assistant"}</span>
                  <span>
                    {msg.timestamp === "streaming" ? "typing…" : msg.timestamp}
                  </span>
                </div>
                {msg.reasoning && (
                  <details
                    className={`mb-2 rounded-lg ${msg.reasoning && !msg.content ? "" : ""}`}
                    open={msg.timestamp === "streaming" && !msg.content}
                  >
                    <summary className="cursor-pointer select-none text-xs font-medium text-text-muted hover:text-text-secondary">
                      {msg.timestamp === "streaming" && !msg.content
                        ? "💭 thinking…"
                        : "💭 thought process"}
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
                      {msg.planConfirmed ? "✓ Confirmed" : "Confirm & Execute"}
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
                      Type your answer or click an option above
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-border bg-bg-secondary p-4">
        {!isConnected && (
          <div className="mb-3 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
            Backend is not connected. Start the server to send messages, or configure it in Settings.
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
                  Run plan
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
                {pendingApproval.dangerous ? "🔴 Approval Required" : "⚠️ Approval Required"}
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
                Approve
              </button>
              <button
                onClick={() => respondToApproval(pendingApproval.request_id, false)}
                className="rounded-lg bg-error px-4 py-2 text-sm font-medium text-white hover:bg-error/80 transition-colors"
              >
                Deny
              </button>
            </div>
          </div>
        )}

        <div className="mb-2 flex items-center justify-between">
          <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-text-muted">
            <input
              type="checkbox"
              checked={autoApprove}
              onChange={(e) => toggleAutoApprove(e.target.checked)}
              className="h-3 w-3"
            />
            Auto-approve
          </label>
          {mode !== "chat" && (
            <button
              onClick={() => setMode("chat")}
              className="text-[10px] text-accent hover:underline"
            >
              ← Back to chat
            </button>
          )}
        </div>

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
                ? "Describe what you want to do. I'll generate a plan first."
                : mode === "build"
                ? "Run the plan with tool execution enabled…"
                : isConnected
                ? "Ask about materials science, DFT, MD, packing, UQ/GP…"
                : "Backend offline — start server.py"
            }
            rows={2}
            disabled={!isConnected || isStreaming || planLoading}
            className="input min-h-[56px] resize-none flex-1"
          />
          <button
            onClick={sendMessage}
            disabled={!isConnected || isStreaming || !input.trim() || planLoading}
            className="btn-primary h-11 px-5"
          >
            {planLoading ? "Planning…" : isStreaming ? "…" : mode === "plan" ? "Plan" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
