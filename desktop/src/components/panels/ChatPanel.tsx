import { useState } from 'react';
import { Search, X, Settings, Archive } from 'lucide-react';
import { Virtuoso } from 'react-virtuoso';
import { useTranslation } from 'react-i18next';
import { ToolResultRenderer } from '../ToolResultRenderer';
import { SaveToMemoryButton } from '../SaveToMemoryButton';
import MessageContent from '../MessageContent';
import type { Message } from '../../hooks/useChatAndConnection';
import type { ReconnectingWebSocket } from '../../lib/ws-client';

const INLINE_COMMANDS = [
  { cmd: '/plan', desc: 'Switch to Plan mode — generate a plan before executing' },
  { cmd: '/research', desc: 'Switch to Research mode — autonomous research loop' },
  { cmd: '/clear', desc: 'Clear all messages in this thread' },
  { cmd: '/new', desc: 'Create a new thread' },
  { cmd: '/help', desc: 'Show available commands and shortcuts' },
  { cmd: '/tools', desc: 'Open the tools panel' },
  { cmd: '/settings', desc: 'Open settings' },
];

// extensions we treat as plain text on drag-drop (materials-science friendly)
const TEXT_FILE_EXTS = [
  '.txt', '.py', '.json', '.cif', '.yaml', '.yml', '.toml', '.md',
  '.csv', '.log', '.incar', '.poscar', '.potcar', '.in', '.out', '.dat',
];

// dev-only guard: make sure the domain extensions route to the text branch
if (import.meta.env.DEV) {
  const _ext = (name: string) => name.substring(name.lastIndexOf('.')).toLowerCase();
  console.assert(TEXT_FILE_EXTS.includes(_ext('structure.cif')), 'cif should be text');
  console.assert(TEXT_FILE_EXTS.includes(_ext('INCAR')), 'incar should be text');
  console.assert(!TEXT_FILE_EXTS.includes(_ext('image.png')), 'png should NOT be text');
}

function CollapsibleMessageContent({ content, isStreaming }: { content: string; isStreaming: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = content.length > 1500;
  if (!isLong || isStreaming || expanded) {
    return <MessageContent content={content} />;
  }
  return (
    <div>
      <div className="max-h-[400px] overflow-hidden relative">
        <MessageContent content={content} />
        <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-bg-secondary to-transparent pointer-events-none" />
      </div>
      <button
        onClick={() => setExpanded(true)}
        className="mt-1 text-xs text-accent hover:underline"
      >
        Show more ({content.length.toLocaleString()} chars)
      </button>
    </div>
  );
}

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
  // real type from useState — allows functional updates (e.g. drag-drop appends)
  setInput: React.Dispatch<React.SetStateAction<string>>;
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
  } = props;

  const [showCommands, setShowCommands] = useState(false);
  const [cmdSelectIdx, setCmdSelectIdx] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const filteredCommands = showCommands
    ? INLINE_COMMANDS.filter(c => c.cmd.startsWith(input))
    : [];

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    // only clear when actually leaving the container, not when crossing into a child
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    for (const file of files) {
      const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      if (TEXT_FILE_EXTS.includes(ext) || file.type.startsWith('text/')) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          const text = ev.target?.result as string;
          setInput((prev) => prev + '\n\n--- ' + file.name + ' ---\n' + text);
        };
        reader.readAsText(file);
      } else {
        setInput((prev) => prev + '\n\n[Attached: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)]');
      }
    }
  };

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
          if (msg.isCompacted) {
            return (
              <div key={index} className="flex justify-center py-1">
                <div className="inline-flex items-center gap-2 rounded-full border border-border bg-bg-secondary px-3 py-1 text-xs text-text-muted">
                  <Archive size={12} />
                  <span>{t('chat.contextCompacted', { before: msg.compactBefore ?? '?', after: msg.compactAfter ?? '?' })}</span>
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
                  {msg.content && (
                    <CollapsibleMessageContent content={msg.content} isStreaming={msg.timestamp === "streaming"} />
                  )}
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

        <div className="mb-2 flex items-center justify-between gap-2">
          {/* Mode selector — segmented control */}
          <div className="flex items-center gap-1 rounded-lg bg-bg-tertiary p-0.5">
            <button
              onClick={() => { setMode("chat"); setResearchMode(false); }}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                mode === "chat" && !researchMode ? "bg-accent text-white" : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {t('chat.mode.chat')}
            </button>
            <button
              onClick={() => { setMode("plan"); setResearchMode(false); }}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                mode === "plan" ? "bg-accent text-white" : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {t('chat.mode.plan')}
            </button>
            <button
              onClick={() => { setResearchMode(!researchMode); if (!researchMode) setMode("chat"); }}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                researchMode ? "bg-accent text-white" : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {t('chat.mode.research') || 'Research'}
            </button>
          </div>

          {/* Options popover */}
          <details className="relative">
            <summary className="flex cursor-pointer list-none items-center gap-1 rounded-md px-2 py-1 text-xs text-text-muted hover:text-text-secondary transition-colors" title="Options">
              <Settings size={14} />
            </summary>
            <div className="absolute right-0 top-full z-50 mt-1 w-56 rounded-lg border border-border bg-bg-secondary p-3 shadow-lg">
              <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
                <input type="checkbox" checked={autoApprove} onChange={(e) => toggleAutoApprove(e.target.checked)} className="h-3.5 w-3.5" />
                {t('chat.autoApprove')}
              </label>
              <div className="mt-3 border-t border-border pt-3">
                <div className="mb-1.5 text-xs text-text-muted">🧠 Thinking intensity</div>
                <div className="flex gap-1">
                  {(["low", "medium", "high"] as const).map((level) => (
                    <button
                      key={level}
                      onClick={() => setThinkingIntensity(level)}
                      className={`flex-1 rounded px-2 py-1 text-xs capitalize transition-colors ${
                        thinkingIntensity === level ? "bg-accent/20 text-accent font-medium" : "text-text-muted hover:text-text-secondary"
                      }`}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </details>
        </div>

        {pendingMessages.length > 0 && (
          <div className="mb-2 rounded-lg border border-accent/20 bg-accent/5 px-3 py-1.5 text-[11px] text-accent">
            📋 {pendingMessages.length} message{pendingMessages.length > 1 ? "s" : ""} queued — will send after current response
          </div>
        )}

        {showCommands && filteredCommands.length > 0 && (
          <div className="mb-2 rounded-lg border border-border bg-bg-secondary shadow-lg overflow-hidden">
            {filteredCommands.map((c, i) => (
              <button
                key={c.cmd}
                onClick={() => {
                  if (c.cmd === '/plan') { setMode('plan'); setInput(''); }
                  else if (c.cmd === '/research') { setResearchMode(true); setInput(''); }
                  else if (c.cmd === '/clear') { setMessages([]); setInput(''); }
                  else { setInput(c.cmd + ' '); }
                  setShowCommands(false);
                }}
                onMouseEnter={() => setCmdSelectIdx(i)}
                className={`flex w-full items-center gap-3 px-4 py-2 text-left transition-colors ${
                  i === cmdSelectIdx ? 'bg-bg-tertiary' : 'hover:bg-bg-tertiary'
                }`}
              >
                <code className="text-xs font-mono text-accent">{c.cmd}</code>
                <span className="text-xs text-text-muted">{c.desc}</span>
              </button>
            ))}
          </div>
        )}

        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`flex items-end gap-3 rounded-lg transition-colors ${isDragOver ? 'border-2 border-dashed border-accent bg-accent/5 p-1' : 'border-2 border-transparent p-1'}`}
        >
          <textarea
            value={input}
            onChange={(e) => {
              const val = e.target.value;
              setInput(val);
              setShowCommands(val.startsWith('/') && !val.includes(' '));
            }}
            onKeyDown={(e) => {
              if (showCommands && filteredCommands.length > 0) {
                if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  setCmdSelectIdx((prev) => (prev + 1) % filteredCommands.length);
                  return;
                }
                if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  setCmdSelectIdx((prev) => (prev - 1 + filteredCommands.length) % filteredCommands.length);
                  return;
                }
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  const selected = filteredCommands[cmdSelectIdx];
                  if (selected) {
                    if (selected.cmd === '/plan') { setMode('plan'); setInput(''); }
                    else if (selected.cmd === '/research') { setResearchMode(true); setInput(''); }
                    else if (selected.cmd === '/clear') { setMessages([]); setInput(''); }
                    else { setInput(selected.cmd + ' '); }
                    setShowCommands(false);
                    return;
                  }
                }
                if (e.key === 'Escape') {
                  e.preventDefault();
                  setShowCommands(false);
                  return;
                }
              }
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
