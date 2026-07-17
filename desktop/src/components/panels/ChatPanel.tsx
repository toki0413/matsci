import { useState, useRef, useEffect, useMemo, type ReactNode, useCallback } from 'react';
import { Search, X, Settings, Archive, Copy, RotateCw, Trash2, ArrowDown, Check, Pencil, CornerUpLeft, Download, BarChart3, Volume2, ChevronUp, ChevronDown, ChevronRight, Clock, Wrench, Loader2, AlertCircle, Paperclip, Pause, Play, FolderTree, Pin, MessageSquare, Code2, BookOpen, ThumbsUp, ThumbsDown, FileDown } from 'lucide-react';
import { Virtuoso, type VirtuosoHandle } from 'react-virtuoso';
import { useTranslation } from 'react-i18next';
import { formatTimeAgo } from '../../lib/constants';
import { api } from '../../lib/api';
import { toast } from '../Toast';
import { ToolResultRenderer } from '../ToolResultRenderer';
import { IterationTimeline } from '../IterationTimeline';
import { SaveToMemoryButton } from '../SaveToMemoryButton';
import { PipelineProgressCard } from '../PipelineProgressCard';
import MessageContent from '../MessageContent';
import type { Message } from '../../hooks/useChatAndConnection';
import type { HeatEngineHealth } from '../../types/domain';
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

// 这些工具默认展开 — 命令执行/文件编辑类操作, 用户一般想直接看清楚干了什么
const AUTO_EXPAND_TOOLS = new Set([
  "shell", "bash", "terminal",
  "edit", "write", "patch", "replace", "str_replace", "file_edit",
]);
const shouldAutoExpand = (toolName?: string) =>
  AUTO_EXPAND_TOOLS.has((toolName || "").toLowerCase());

function CollapsibleMessageContent({ content, isStreaming }: { content: string; isStreaming: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = content.length > 800;
  const isVeryLong = content.length > 3000;
  if (!isLong || isStreaming || expanded) {
    return (
      <>
        <MessageContent content={content} />
        {isLong && expanded && !isStreaming && (
          <button
            onClick={() => setExpanded(false)}
            className="mt-1 text-xs text-accent hover:underline"
          >
            Show less
          </button>
        )}
      </>
    );
  }
  const preview = content.slice(0, isVeryLong ? 500 : 800);
  const lastNewline = preview.lastIndexOf('\n');
  const trimmedPreview = lastNewline > 0 ? preview.slice(0, lastNewline) : preview;
  return (
    <div>
      <MessageContent content={trimmedPreview + '\n\n'} />
      <div className="relative rounded-lg border border-border bg-bg-tertiary/50 px-4 py-3">
        <div className="absolute inset-0 overflow-hidden rounded-lg">
          <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-bg-tertiary/50 to-transparent pointer-events-none" />
        </div>
        <button
          onClick={() => setExpanded(true)}
          className="relative flex items-center gap-2 text-xs text-accent hover:underline"
        >
          <ChevronDown size={14} />
          Show more ({(content.length - trimmedPreview.length).toLocaleString()} more chars)
        </button>
      </div>
    </div>
  );
}

// HRI #4: SUGGEST mode editor — agent 代码先展示给用户编辑, Ctrl+Enter 执行
function SuggestCodeEditor({
  code, risk, reason, onRespond,
}: {
  code: string;
  risk: string;
  reason: string;
  onRespond: (action: "approve" | "edit" | "deny", editedCode?: string) => void;
}) {
  const { t } = useTranslation();
  const [editedCode, setEditedCode] = useState(code);
  const isHighRisk = risk === "high";

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      // 有改动 → edit, 没改动 → approve
      const changed = editedCode !== code;
      onRespond(changed ? "edit" : "approve", changed ? editedCode : undefined);
    }
  };

  return (
    <div className={`mb-3 rounded-xl border-2 p-3 ${isHighRisk ? "border-error bg-error/5" : "border-accent bg-accent/5"}`}>
      <div className="mb-1.5 flex items-center gap-2">
        <span className={`text-sm font-bold ${isHighRisk ? "text-error" : "text-accent"}`}>
          {isHighRisk ? "🔴 SUGGEST (high risk)" : "💡 SUGGEST mode"}
        </span>
        <span className={`rounded px-1.5 py-0.5 text-[10px] font-mono ${
          risk === "high" ? "bg-red-500/15 text-red-400" : risk === "medium" ? "bg-yellow-500/15 text-yellow-400" : "bg-green-500/15 text-green-400"
        }`}>
          {risk}
        </span>
      </div>
      {reason && <p className="mb-2 text-xs text-text-secondary">{reason}</p>}
      <textarea
        value={editedCode}
        onChange={(e) => setEditedCode(e.target.value)}
        onKeyDown={handleKeyDown}
        spellCheck={false}
        className="mb-2 h-48 w-full resize-y rounded-lg border border-border bg-bg-tertiary p-2 font-mono text-xs text-text-primary focus:border-accent focus:outline-none"
        placeholder="Edit code here..."
      />
      <div className="flex items-center gap-2">
        <button
          onClick={() => {
            const changed = editedCode !== code;
            onRespond(changed ? "edit" : "approve", changed ? editedCode : undefined);
          }}
          className="btn-success px-4 py-2 text-sm"
        >
          {editedCode !== code ? t('chat.approveEdit') : t('chat.approveWithShortcut')}
        </button>
        <button
          onClick={() => onRespond("deny")}
          className="btn-danger px-4 py-2 text-sm"
        >
          {t('chat.deny')}
        </button>
        <button
          onClick={() => setEditedCode(code)}
          className="rounded px-3 py-2 text-xs text-text-muted hover:text-text-secondary transition-colors"
          title="Reset to original code"
        >
          {t('chat.reset')}
        </button>
        <span className="ml-auto text-[10px] text-text-muted">
          {editedCode !== code ? t('chat.edited') : t('chat.unchanged')}
        </span>
      </div>
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
  wsReconnecting?: boolean;
  wsFailed?: boolean;
  undoWindow?: boolean;
  undoSend?: () => void;
  sendMessage: () => void;
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
  stopGeneration: () => void;
  pauseGeneration: () => void;
  resumeGeneration: () => void;
  isPaused: boolean;
  researchMode: boolean;
  setResearchMode: (v: boolean) => void;
  contextBudgetTokens?: number;
  onExpandResult?: (content: string, toolName?: string) => void;
  campaignEvents?: Array<{
    event: string;
    data: Record<string, unknown>;
    ts: number;
    task_id: string;
  }>;
  threadTaskState?: {
    goal: string;
    mode: string;
    iteration: number;
    steps_done: number;
    steps_total: number;
    key_findings: string[];
  };
  planExecState?: Record<string, "executing" | "done">;
  agentMode?: { exec_mode: string; user_mode: string; flags: string[]; trace_id?: string };
  trustScore?: number;
  approvalBudget?: number;
  suggestMode?: boolean;
  pendingSuggestCode?: {
    code: string; risk: string; reason: string; turn: number;
  } | null;
  toggleSuggestMode?: (enabled: boolean) => void;
  respondToSuggestCode?: (action: "approve" | "edit" | "deny", editedCode?: string) => void;
  riskThreshold?: number;
  // v7 G59: 认知热机健康 (从 SSE /tasks/stream 'campaign' 推送)
  heatEngineHealth?: HeatEngineHealth | null;
  // 跳到 files tab — FilesPanel 自己管 cwd/editor 那堆状态, 不在这里重做
  onOpenFiles?: () => void;
  // multi-agent persona
  personas?: Array<{ name: string; description?: string }>;
  currentPersona?: string;
  setCurrentPersona?: (name: string) => void;
}

export function ChatPanel(props: ChatPanelProps) {
  const { t } = useTranslation();
  const {
    messages, chatSearchOpen, chatSearchQuery, setChatSearchOpen, setChatSearchQuery,
    wsClientRef, setMessages, answerClarification, pendingClarifications,
    isConnected, wsReconnecting, wsFailed, undoWindow, undoSend, sendMessage,
    setMode, input, setInput, mode, isStreaming, messagesEndRef,
    pendingApproval, respondToApproval, autoApprove, toggleAutoApprove,
    thinkingIntensity, setThinkingIntensity,
    pendingMessages, stopGeneration, pauseGeneration, resumeGeneration, isPaused, researchMode, setResearchMode,
    contextBudgetTokens, onExpandResult, campaignEvents, threadTaskState, planExecState,
    agentMode,
    trustScore,
    approvalBudget,
    suggestMode,
    pendingSuggestCode,
    toggleSuggestMode,
    respondToSuggestCode,
    riskThreshold,
    heatEngineHealth,
    onOpenFiles,
    personas = [],
    currentPersona,
    setCurrentPersona,
  } = props;

  const [showCommands, setShowCommands] = useState(false);
  const [showTimeline, setShowTimeline] = useState(true);
  const [cmdSelectIdx, setCmdSelectIdx] = useState(0);
  const [showMentions, setShowMentions] = useState(false);
  const [mentionIdx, setMentionIdx] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [mentionQuery, setMentionQuery] = useState('');
  const [editHistory, setEditHistory] = useState<Record<number, string[]>>({});
  const [viewingHistory, setViewingHistory] = useState<number | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [copyingId, setCopyingId] = useState<number | null>(null);
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; msg: Message; index: number } | null>(null);
  const [quotedMsg, setQuotedMsg] = useState<string | null>(null);
  const [showStats, setShowStats] = useState(false);
  const [notifSound, setNotifSound] = useState(() => localStorage.getItem('chat-notif-sound') !== 'off');
  const [streamingWasActive, setStreamingWasActive] = useState(false);
  // reasoning 折叠状态: 用户手动 toggle 后记住, 否则跟随 streaming 自动开/关
  const [reasoningOpen, setReasoningOpen] = useState<Record<number, boolean>>({});
  // 搜索范围
  const [chatSearchScope, setChatSearchScope] = useState<'messages' | 'files' | 'code' | 'knowledge'>('messages');
  // 消息管理：选中状态、置顶状态
  const [selectedMsgIds, setSelectedMsgIds] = useState<Set<number>>(new Set());
  const [pinnedMsgIds, setPinnedMsgIds] = useState<Set<number>>(new Set());
  const [lastClickedIdx, setLastClickedIdx] = useState<number | null>(null);
  // persona selector
  const [showPersonaSelector, setShowPersonaSelector] = useState(false);
  // message reactions: index → 'up' | 'down'
  const [reactions, setReactions] = useState<Record<number, 'up' | 'down'>>({});
  // export menu
  const [showExportMenu, setShowExportMenu] = useState(false);
  // command history popover
  const [showCmdHistory, setShowCmdHistory] = useState(false);

  // ponytail: dev-only progress toast test button — remove before prod
  const testProgressToast = () => {
    const id = `test-${Date.now()}`;
    toast.progress('Processing...', { progress: 0, id, cancelable: true });
    let p = 0;
    const timer = setInterval(() => {
      p += 10;
      toast.updateProgress(id, p);
      if (p >= 100) {
        clearInterval(timer);
        toast.complete(id, 'Done!');
      }
    }, 500);
  };

  // Play notification sound when streaming completes
  useEffect(() => {
    if (isStreaming) {
      setStreamingWasActive(true);
    } else if (streamingWasActive && notifSound) {
      // Simple beep using Web Audio API
      try {
        const ctx = new AudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 800;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
        osc.start();
        osc.stop(ctx.currentTime + 0.3);
      } catch {}
      setStreamingWasActive(false);
    }
  }, [isStreaming, streamingWasActive, notifSound]);

  const toggleNotifSound = () => {
    const next = !notifSound;
    setNotifSound(next);
    localStorage.setItem('chat-notif-sound', next ? 'on' : 'off');
  };

  const exportConversation = (format: 'md' | 'json') => {
    const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    const userMsgs = messages.filter(m => m.role === 'user').length;
    const aiMsgs = messages.filter(m => m.role === 'assistant').length;
    let content: string;
    let mime: string;

    if (format === 'json') {
      content = JSON.stringify({ exported_at: new Date().toISOString(), message_count: messages.length, messages }, null, 2);
      mime = 'application/json';
    } else {
      content = `# Huginn Conversation\n\nExported: ${new Date().toLocaleString()}\nMessages: ${messages.length} (${userMsgs} user, ${aiMsgs} assistant)\n\n---\n\n`;
      for (const m of messages) {
        if (m.role === 'user') content += `## 👤 You\n\n${m.content}\n\n`;
        else if (m.role === 'assistant') content += `## 🤖 Assistant\n\n${m.content}\n\n`;
        else if (m.role === 'tool') content += `### 🔧 ${m.tool_name}\n\n\`\`\`json\n${JSON.stringify(m.tool_args, null, 2)}\n\`\`\`\n\n`;
        if (m.reasoning) content += `<details><summary>💭 Reasoning</summary>\n\n${m.reasoning}\n\n</details>\n\n`;
      }
      mime = 'text/markdown';
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `huginn-conversation-${ts}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${messages.length} messages as ${format.toUpperCase()}`);
  };

  // Close context menu on any click outside
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, [ctxMenu]);

  // Input history — up arrow cycles through previous user messages
  const inputHistoryRef = useRef<string[]>([]);
  const historyIdxRef = useRef(-1);
  const savedInputRef = useRef("");
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, []);

  const filteredCommands = showCommands
    ? INLINE_COMMANDS.filter(c => c.cmd.startsWith(input))
    : [];

  // ── @mention system ──────────────────────────────────────────
  const MENTION_ITEMS = [
    { id: 'agent', label: 'Agent', desc: 'Autonomous agent team' },
    { id: 'coder', label: 'Coder', desc: 'Code execution assistant' },
    { id: 'knowledge', label: 'Knowledge Base', desc: 'Search uploaded documents' },
    { id: 'web', label: 'Web Search', desc: 'Search the internet' },
    { id: 'periodic', label: 'Periodic Table', desc: 'Element properties' },
    { id: 'structure', label: 'Structure Viewer', desc: '3D crystal structures' },
    { id: 'notebook', label: 'Notebook', desc: 'Jupyter notebook' },
    { id: 'sandbox', label: 'Sandbox', desc: 'Code sandbox' },
    { id: 'sweep', label: 'Sweep Dashboard', desc: 'Parameter sweep' },
    { id: 'memory', label: 'Memory', desc: 'Persistent memory store' },
  ];

  const filteredMentions = showMentions
    ? MENTION_ITEMS.filter(m => m.label.toLowerCase().includes(mentionQuery.toLowerCase()))
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

  const handleDrop = async (e: React.DragEvent) => {
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
        // Binary file — actually upload to backend and notify
        try {
          setInput((prev) => prev + `\n\n[Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…]`);
          const result = await api.uploadWithProgress<{ success?: boolean; path?: string; error?: string }>(
            '/transfer/upload', file,
          );
          if (result.success) {
            toast.success(`Uploaded ${file.name}`);
            setInput((prev) => prev.replace(`[Uploading ${file.name}…`, `[Attached: ${file.name}]`));
          } else {
            toast.error(`Upload failed: ${result.error}`);
            setInput((prev) => prev.replace(`[Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…`, `[Upload failed: ${file.name}]`));
          }
        } catch (err: any) {
          toast.error(`Upload failed: ${err.message}`);
        }
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

  const [searchMatchIdx, setSearchMatchIdx] = useState(0);
  const searchActive = chatSearchQuery.trim().length > 0;
  const filteredMessages = searchActive
    ? messages.filter((m) => m.content.toLowerCase().includes(chatSearchQuery.toLowerCase()))
    : messages;
  const groupedMessages = groupMessages(filteredMessages);

  // ── Turn auto-collapse (Trae style) ───────────────────────
  // A "turn" = user message + all subsequent assistant/tool messages
  // until the next user message. Old turns auto-collapse to keep the
  // conversation scannable.
  interface TurnInfo {
    idx: number;
    startIdx: number;
    endIdx: number;
    userExcerpt: string;
    toolCount: number;
    assistantExcerpt: string;
    msgCount: number;
  }

  const turns = useMemo<TurnInfo[]>(() => {
    const result: TurnInfo[] = [];
    let current: TurnInfo | null = null;
    for (let i = 0; i < groupedMessages.length; i++) {
      const msg = groupedMessages[i];
      if (msg.role === "user") {
        if (current) { current.endIdx = i - 1; result.push(current); }
        current = {
          idx: result.length, startIdx: i, endIdx: i,
          userExcerpt: msg.content.slice(0, 60),
          toolCount: 0, assistantExcerpt: "", msgCount: 1,
        };
      } else if (current) {
        current.endIdx = i;
        current.msgCount++;
        if (msg.role === "tool") current.toolCount++;
        if ((msg as any).role === "tool_group")
          current.toolCount += (msg as any).tool_calls?.length || 0;
        if (msg.role === "assistant" && msg.content && !current.assistantExcerpt)
          current.assistantExcerpt = msg.content.slice(0, 80);
      }
    }
    if (current) result.push(current);
    return result;
  }, [groupedMessages]);

  const [collapsedTurns, setCollapsedTurns] = useState<Set<number>>(new Set());
  // auto-collapse old turns when there are more than 5
  useEffect(() => {
    if (turns.length > 5) {
      setCollapsedTurns(prev => {
        const next = new Set(prev);
        for (let i = 0; i < turns.length - 3; i++) {
          if (!prev.has(i)) next.add(i);
        }
        return next;
      });
    }
  }, [turns.length]);

  // build display list: replace collapsed turns with summary entries
  const displayMessages = useMemo(() => {
    if (collapsedTurns.size === 0) return groupedMessages;
    const result: any[] = [];
    let turnIdx = -1;
    const inserted = new Set<number>();
    for (let i = 0; i < groupedMessages.length; i++) {
      const msg = groupedMessages[i];
      if (msg.role === "user") {
        turnIdx++;
        if (collapsedTurns.has(turnIdx) && !inserted.has(turnIdx)) {
          result.push({ isTurnSummary: true, turn: turns[turnIdx] });
          inserted.add(turnIdx);
          continue;
        }
      }
      if (turnIdx >= 0 && collapsedTurns.has(turnIdx)) continue;
      result.push(msg);
    }
    return result;
  }, [groupedMessages, collapsedTurns, turns]);

  // map displayMessages index → turn index (for collapse button)
  const displayTurnMap = useMemo(() => {
    const map: number[] = [];
    let turnIdx = -1;
    for (let i = 0; i < displayMessages.length; i++) {
      if (displayMessages[i]?.role === "user") turnIdx++;
      map.push(turnIdx);
    }
    return map;
  }, [displayMessages]);

  const searchRegex = searchActive
    ? new RegExp(`(${chatSearchQuery.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi")
    : null;

  // Flatten match indices for prev/next navigation
  const searchMatchCount = searchActive
    ? filteredMessages.reduce((sum, m) => {
        const matches = m.content.toLowerCase().match(new RegExp(chatSearchQuery.trim().toLowerCase(), "g"));
        return sum + (matches?.length || 0);
      }, 0)
    : 0;

  function highlightText(text: string): ReactNode {
    if (!searchRegex) return text;
    const parts = text.split(searchRegex);
    return parts.map((part, i) =>
      searchRegex.test(part)
        ? <mark key={i} className="search-hl">{part}</mark>
        : part
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* ponytail: dev-only test button for progress toast — remove before prod */}
      {import.meta.env.DEV && (
        <button
          onClick={testProgressToast}
          className="absolute top-2 left-2 z-50 rounded bg-accent/20 px-2 py-1 text-xs font-medium text-accent hover:bg-accent/30"
        >
          Test Progress Toast
        </button>
      )}
      {chatSearchOpen && (
        <div className="flex flex-col border-b border-border bg-bg-secondary/50">
          <div className="flex items-center gap-2 px-6 py-2">
            <Search size={14} className="shrink-0 text-text-muted" aria-hidden="true" />
            <div className="flex rounded-lg border border-border bg-bg-tertiary p-0.5">
              {[
                { id: 'messages' as const, label: 'Messages', icon: <MessageSquare size={12} /> },
                { id: 'files' as const, label: 'Files', icon: <FolderTree size={12} /> },
                { id: 'code' as const, label: 'Code', icon: <Code2 size={12} /> },
                { id: 'knowledge' as const, label: 'KB', icon: <BookOpen size={12} /> },
              ].map((scope) => (
                <button
                  key={scope.id}
                  onClick={() => setChatSearchScope(scope.id)}
                  className={`flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] transition-colors ${
                    chatSearchScope === scope.id
                      ? 'bg-accent text-white'
                      : 'text-text-muted hover:text-text-secondary'
                  }`}
                  aria-label={`Search in ${scope.label}`}
                >
                  {scope.icon}
                  {scope.label}
                </button>
              ))}
            </div>
            <input
              type="text"
              autoFocus
              value={chatSearchQuery}
              onChange={(e) => setChatSearchQuery(e.target.value)}
              placeholder={`Search ${chatSearchScope === 'messages' ? 'messages' : chatSearchScope === 'files' ? 'files' : chatSearchScope === 'code' ? 'code' : 'knowledge'}…`}
              aria-label="Search"
              className="flex-1 bg-transparent text-sm text-text-primary outline-none focus-visible:ring-2 focus-visible:ring-accent/40 rounded placeholder:text-text-muted"
            />
            {chatSearchQuery && chatSearchScope === 'messages' && (
              <div className="flex shrink-0 items-center gap-1">
                <span className="text-[11px] text-text-muted">
                  {searchMatchCount > 0 ? `${Math.min(searchMatchIdx + 1, searchMatchCount)}/${searchMatchCount}` : '0'}
                </span>
                <button
                  onClick={() => setSearchMatchIdx(prev => (prev - 1 + searchMatchCount) % Math.max(searchMatchCount, 1))}
                  disabled={searchMatchCount === 0}
                  className="rounded p-0.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary disabled:opacity-30"
                  aria-label="Previous match"
                  title="Previous match"
                >
                  <ChevronUp size={14} />
                </button>
                <button
                  onClick={() => setSearchMatchIdx(prev => (prev + 1) % Math.max(searchMatchCount, 1))}
                  disabled={searchMatchCount === 0}
                  className="rounded p-0.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary disabled:opacity-30"
                  aria-label="Next match"
                  title="Next match"
                >
                  <ChevronDown size={14} aria-hidden="true" />
                </button>
              </div>
            )}
            <button onClick={() => { setChatSearchOpen(false); setChatSearchQuery(""); }} className="shrink-0 text-text-muted hover:text-text-secondary" aria-label="Close search">
              <X size={14} aria-hidden="true" />
            </button>
          </div>
        </div>
      )}
      {/* Multi-select toolbar */}
      {selectedMsgIds.size > 0 && (
        <div className="flex items-center justify-between border-b border-border bg-accent/10 px-6 py-1.5">
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-accent">{selectedMsgIds.size} selected</span>
            <button
              onClick={() => {
                setMessages(prev => prev.filter((_, i) => !selectedMsgIds.has(i)));
                toast.success(`Deleted ${selectedMsgIds.size} messages`);
                setSelectedMsgIds(new Set());
              }}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-error hover:bg-error/10 transition-colors"
              aria-label="Delete selected"
            >
              <Trash2 size={12} /> Delete
            </button>
            <button
              onClick={() => {
                const text = Array.from(selectedMsgIds)
                  .sort((a, b) => a - b)
                  .map(i => messages[i]?.content || '')
                  .join('\n\n');
                navigator.clipboard.writeText(text);
                toast.success('Copied selected');
              }}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-colors"
              aria-label="Copy selected"
            >
              <Copy size={12} /> Copy
            </button>
          </div>
          <button
            onClick={() => setSelectedMsgIds(new Set())}
            className="text-xs text-text-muted hover:text-text-secondary"
          >
            Deselect all
          </button>
        </div>
      )}
      {/* Chat action toolbar — export, stats, sound */}
      <div className="flex items-center gap-1 border-b border-border bg-bg-secondary/30 px-6 py-1">
        <button
          onClick={() => exportConversation('md')}
          disabled={messages.length === 0}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-colors disabled:opacity-30"
          title={t('chat.exportMarkdown')}
        >
          <Download size={12} aria-hidden="true" /> MD
        </button>
        <button
          onClick={() => exportConversation('json')}
          disabled={messages.length === 0}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-colors disabled:opacity-30"
          title={t('chat.exportJson')}
        >
          <Download size={12} aria-hidden="true" /> JSON
        </button>
        <div className="mx-1 h-3 w-px bg-border" />
        <button
          onClick={() => setShowStats(s => !s)}
          className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] transition-colors ${showStats ? 'text-accent' : 'text-text-muted hover:text-text-primary'} hover:bg-bg-tertiary`}
          title="Chat statistics"
        >
          <BarChart3 size={12} aria-hidden="true" /> Stats
        </button>
        <div className="mx-1 h-3 w-px bg-border" />
        <button
          onClick={toggleNotifSound}
          className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] transition-colors ${notifSound ? 'text-accent' : 'text-text-muted'} hover:bg-bg-tertiary`}
          title={notifSound ? 'Sound on (click to mute)' : 'Sound off (click to enable)'}
        >
          <Volume2 size={12} aria-hidden="true" /> {notifSound ? 'On' : 'Off'}
        </button>
        <div className="ml-auto flex items-center gap-2 text-[10px] text-text-muted">
          {agentMode && (
            <div className="flex items-center gap-1" title="Agent execution mode (HRI situation awareness)">
              <span className={`rounded px-1.5 py-0.5 font-medium ${
                agentMode.exec_mode === 'code_act'
                  ? 'bg-purple-500/15 text-purple-400'
                  : 'bg-blue-500/15 text-blue-400'
              }`}>
                {agentMode.exec_mode === 'code_act' ? 'Code' : 'Tool'}
              </span>
              {agentMode.user_mode !== 'chat' && (
                <span className="rounded bg-orange-500/15 px-1.5 py-0.5 font-medium text-orange-400">
                  {agentMode.user_mode === 'research' ? 'Research' : 'Plan'}
                </span>
              )}
              {agentMode.flags.includes('plan_mode') && (
                <span className="rounded bg-yellow-500/15 px-1.5 py-0.5 font-medium text-yellow-400">
                  plan_mode
                </span>
              )}
            </div>
          )}
          {trustScore !== undefined && (
            <div className="flex items-center gap-1" title={`Trust: ${trustScore.toFixed(2)} (< 0.3 forces ask, > 0.7 auto-medium)`}>
              <span className={`h-1.5 w-1.5 rounded-full ${
                trustScore > 0.7 ? 'bg-green-400' : trustScore < 0.3 ? 'bg-red-400' : 'bg-yellow-400'
              }`} />
              <span className={
                trustScore > 0.7 ? 'text-green-400' : trustScore < 0.3 ? 'text-red-400' : 'text-yellow-400'
              }>
                {trustScore.toFixed(2)}
              </span>
            </div>
          )}
          {approvalBudget !== undefined && (
            <div className="flex items-center gap-1" title={`Approval budget: ${approvalBudget} remaining (auto-escalates at ≤ 3)`}>
              <span className={
                approvalBudget <= 3 ? 'text-red-400' : approvalBudget <= 6 ? 'text-yellow-400' : 'text-text-muted'
              }>
                ◐{approvalBudget}
              </span>
            </div>
          )}
          {riskThreshold !== undefined && (
            <div className="flex items-center gap-1" title={`Risk threshold: ${riskThreshold.toFixed(2)} (>0.7 lenient, <0.3 strict)`}>
              <span className={
                riskThreshold > 0.7 ? 'text-green-400' : riskThreshold < 0.3 ? 'text-red-400' : 'text-text-muted'
              }>
                ⚖{riskThreshold.toFixed(2)}
              </span>
            </div>
          )}
          {heatEngineHealth && (
            <div
              className="flex items-center gap-1"
              title={`Cognitive heat engine (v7 G59)
Re=${heatEngineHealth.Re_cog.toFixed(1)} (crit ${heatEngineHealth.Re_crit.toFixed(1)})  U=${heatEngineHealth.U.toFixed(1)} L=${heatEngineHealth.L.toFixed(1)} nu=${heatEngineHealth.nu.toFixed(2)}
eta=${heatEngineHealth.eta_cog.toFixed(3)}  T_hot=${heatEngineHealth.T_hot.toFixed(2)} T_cold=${heatEngineHealth.T_cold.toFixed(2)}
intermittency(kurt)=${heatEngineHealth.intermittency_kurtosis.toFixed(2)}  cum_work=${heatEngineHealth.cumulative_work.toFixed(1)}  cum_entropy=${heatEngineHealth.cumulative_entropy_produced.toFixed(1)}
status: ${heatEngineHealth.status}${heatEngineHealth.warnings.length ? '\nwarnings:\n- ' + heatEngineHealth.warnings.join('\n- ') : '\nno warnings'}`}
            >
              <span className={
                heatEngineHealth.status === 'healthy' ? 'text-green-400'
                  : heatEngineHealth.status === 'chaotic' ? 'text-red-400'
                  : heatEngineHealth.status === 'stagnant' ? 'text-yellow-400'
                  : 'text-orange-400'
              }>
                🜂{heatEngineHealth.Re_cog.toFixed(0)}
              </span>
              <span className="text-text-muted">η{heatEngineHealth.eta_cog.toFixed(2)}</span>
            </div>
          )}
          {suggestMode && (
            <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 font-medium text-cyan-400" title="SUGGEST mode: all code shown for editing before execution">
              Suggest
            </span>
          )}
          {/* OAK 启发: trace_id 贯穿 — 当前研究分支标识 */}
          {agentMode?.trace_id && (
            <span className="font-mono text-[10px] text-cyan-400/70" title={`Trace: ${agentMode.trace_id}`}>
              ⟡{agentMode.trace_id.length > 10 ? agentMode.trace_id.slice(0, 8) : agentMode.trace_id}
            </span>
          )}
          {(() => {
            const budget = contextBudgetTokens || 32000;
            const totalChars = messages.reduce((sum, m) => sum + (m.content?.length || 0) + (m.reasoning?.length || 0), 0) + input.length;
            const estTokens = Math.round(totalChars / 4);
            const pct = Math.min(estTokens / budget, 1);
            const r = 8, c = 2 * Math.PI * r;
            const color = pct > 0.8 ? 'var(--error, #ef4444)' : pct > 0.5 ? 'var(--warning, #f59e0b)' : 'var(--success, #22c55e)';
            return (
              <div className="flex items-center gap-1.5" title={`~${estTokens.toLocaleString()} / ${budget.toLocaleString()} tokens (${Math.round(pct * 100)}%)`}>
                <svg width="20" height="20" viewBox="0 0 20 20" className="shrink-0">
                  <circle cx="10" cy="10" r={r} fill="none" stroke="var(--border)" strokeWidth="2" />
                  <circle cx="10" cy="10" r={r} fill="none" stroke={color} strokeWidth="2"
                    strokeDasharray={c} strokeDashoffset={c * (1 - pct)}
                    strokeLinecap="round" transform="rotate(-90 10 10)" style={{ transition: 'stroke-dashoffset 0.3s ease' }} />
                </svg>
                <span style={{ color }}>{Math.round(pct * 100)}%</span>
              </div>
            );
          })()}
          <span>{messages.length} {messages.length === 1 ? 'message' : 'messages'}</span>
          {(() => {
            const shellCmds = messages.filter(m => m.role === 'tool' && (m.tool_name?.toLowerCase().includes('shell') || m.tool_name?.toLowerCase().includes('bash')));
            if (shellCmds.length === 0) return null;
            return (
              <div className="relative">
                <button
                  onClick={() => setShowCmdHistory(!showCmdHistory)}
                  className="flex items-center gap-1 text-text-muted hover:text-text-primary"
                  title="Shell command history"
                >
                  <Clock size={12} />
                  {shellCmds.length} cmds
                </button>
                {showCmdHistory && (
                  <div className="absolute top-full right-0 mt-1 w-96 max-h-80 overflow-y-auto rounded-lg border border-border bg-bg-secondary shadow-lg z-20">
                    <div className="sticky top-0 bg-bg-secondary border-b border-border px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-text-muted">
                      {t('chat.shellCommands')} ({shellCmds.length})
                    </div>
                    {shellCmds.map((m, i) => (
                      <div key={i} className="border-b border-border/50 px-3 py-2 hover:bg-bg-tertiary">
                        <div className="flex items-center justify-between mb-0.5">
                          <span className={`text-[10px] font-medium ${m.tool_status === 'error' ? 'text-red-400' : 'text-green-400'}`}>
                            {m.tool_status === 'error' ? '✗' : '✓'} {m.tool_status || 'done'}
                          </span>
                          <span className="text-[10px] text-text-muted">{m.timestamp}</span>
                        </div>
                        <code className="text-xs text-text-primary font-mono">
                          {typeof m.tool_args === 'object' ? m.tool_args?.command || m.tool_args?.cmd || JSON.stringify(m.tool_args).slice(0, 120) : String(m.tool_args || '').slice(0, 120)}
                        </code>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })()}
          {messages.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setShowExportMenu(!showExportMenu)}
                className="flex items-center gap-1 text-text-muted hover:text-text-primary"
                title={t('chat.exportConversation')}
              >
                <FileDown size={12} />
                {t('chat.export')}
              </button>
              {showExportMenu && (
                <div className="absolute top-full right-0 mt-1 w-36 rounded-lg border border-border bg-bg-secondary shadow-lg z-20">
                  <button
                    onClick={() => {
                      const md = messages.map(m => {
                        if (m.role === 'user') return `## 🧑 User\n\n${m.content}`;
                        if (m.role === 'assistant') return `## 🤖 Assistant\n\n${m.content}`;
                        if (m.role === 'tool') return `### 🔧 ${m.tool_name || 'Tool'}\n\n\`\`\`\n${m.tool_result || m.content}\n\`\`\``;
                        return m.content;
                      }).join('\n\n---\n\n');
                      const blob = new Blob([`# Conversation Export\n\n${new Date().toISOString()}\n\n---\n\n${md}`], { type: 'text/markdown' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url; a.download = `chat-${Date.now()}.md`; a.click();
                      URL.revokeObjectURL(url);
                      setShowExportMenu(false);
                      toast.success(t('chat.exportedAsMarkdown'));
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
                  >
                    📝 Markdown
                  </button>
                  <button
                    onClick={() => {
                      const json = JSON.stringify({ exported_at: new Date().toISOString(), messages }, null, 2);
                      const blob = new Blob([json], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url; a.download = `chat-${Date.now()}.json`; a.click();
                      URL.revokeObjectURL(url);
                      setShowExportMenu(false);
                      toast.success(t('chat.exportedAsJson'));
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
                  >
                    {'{ }'} JSON
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Stats panel */}
      {showStats && messages.length > 0 && (
        <div className="border-b border-border bg-bg-tertiary/50 px-6 py-2">
          <div className="flex flex-wrap gap-4 text-xs">
            {(() => {
              const userMsgs = messages.filter(m => m.role === 'user');
              const aiMsgs = messages.filter(m => m.role === 'assistant');
              const toolMsgs = messages.filter(m => m.role === 'tool');
              const totalChars = messages.reduce((sum, m) => sum + (m.content?.length || 0), 0);
              const estTokens = Math.round(totalChars / 4);
              return (
                <>
                  <span className="text-text-secondary">👤 User: <strong className="text-text-primary">{userMsgs.length}</strong></span>
                  <span className="text-text-secondary">🤖 AI: <strong className="text-text-primary">{aiMsgs.length}</strong></span>
                  <span className="text-text-secondary">🔧 Tools: <strong className="text-text-primary">{toolMsgs.length}</strong></span>
                  <span className="text-text-secondary">📝 Chars: <strong className="text-text-primary">{totalChars.toLocaleString()}</strong></span>
                  <span className="text-text-secondary">🪙 ~Tokens: <strong className="text-text-primary">{estTokens.toLocaleString()}</strong></span>
                </>
              );
            })()}
          </div>
        </div>
      )}

      {/* Iteration timeline — collapsible bar, reuses existing messages */}
      <div className="border-b border-border" data-timeline-wrapper>
        <button
          onClick={() => setShowTimeline(v => !v)}
          className="flex w-full items-center gap-1 px-3 py-1 text-xs text-text-muted hover:text-text-primary"
        >
          {showTimeline ? <ChevronDown size={12} aria-hidden="true" /> : <ChevronRight size={12} aria-hidden="true" />}
          Iterations
          {threadTaskState?.goal && (
            <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--fg-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, textAlign: 'left' }}>
              {threadTaskState.goal}
            </span>
          )}
          {threadTaskState && threadTaskState.iteration > 0 && (
            <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 3, background: 'var(--bg-tertiary)', color: 'var(--fg-muted)' }}>
              iter {threadTaskState.iteration}
            </span>
          )}
        </button>
        {showTimeline && (
          <div className="max-h-40 overflow-y-auto px-3 pb-2">
            <IterationTimeline messages={messages as unknown as Array<Record<string, unknown>>} campaignEvents={campaignEvents} />
          </div>
        )}
      </div>

      <div className="relative flex-1 min-h-0">
      <Virtuoso
        ref={virtuosoRef}
        data={displayMessages}
        className="cv-list"
        style={{ height: '100%' }}
        atBottomStateChange={(atBottom) => setShowScrollBtn(!atBottom)}
        itemContent={(index, msg) => {
          // collapsed turn summary bar (Trae style)
          if ((msg as any).isTurnSummary) {
            const turn = (msg as any).turn;
            return (
              <div key={index} className="flex justify-center py-1">
                <button
                  onClick={() => {
                    setCollapsedTurns(prev => {
                      const next = new Set(prev);
                      next.delete(turn.idx);
                      return next;
                    });
                  }}
                  className="flex items-center gap-2 rounded-full border border-border bg-bg-tertiary px-3 py-1 text-xs text-text-muted hover:bg-bg-secondary hover:text-text-primary transition-colors max-w-2xl w-full"
                >
                  <ChevronRight className="w-3 h-3 shrink-0" aria-hidden="true" />
                  <span className="truncate">
                    <span className="text-text-secondary font-medium">#{turn.idx + 1}</span>
                    {" "}
                    {turn.userExcerpt}
                  </span>
                  <span className="text-text-muted/50 shrink-0">
                    {turn.msgCount} msgs · {turn.toolCount} tools
                  </span>
                </button>
              </div>
            );
          }
          if ((msg as any).role === "tool_group" && (msg as any).tool_calls) {
            return (
              <div key={index} className="flex justify-center">
                <div className="w-full max-w-2xl rounded-xl border border-border bg-bg-secondary p-4 shadow-sm">
                  <div className="flex items-center gap-2 text-sm font-semibold text-accent">
                    <Wrench size={14} aria-hidden="true" />
                    <span>{(msg as any).tool_calls.length} tool calls</span>
                  </div>
                  <div className="mt-2 space-y-2">
                    {(msg as any).tool_calls.map((tc: Message, ti: number) => (
                      <details key={ti} open={shouldAutoExpand(tc.tool_name)} className={`rounded-lg border p-2 ${
                        tc.tool_status === "running" ? "border-accent/30 bg-accent/5"
                        : tc.tool_status === "error" ? "border-red-500/30 bg-red-500/5"
                        : "border-border bg-bg-tertiary"
                      }`}>
                        <summary className="cursor-pointer text-xs font-medium text-text-secondary flex items-center gap-1.5">
                          {tc.tool_status === "running" ? (
                            <Loader2 size={12} className="text-accent animate-spin motion-reduce:animate-none" aria-hidden="true" />
                          ) : tc.tool_status === "error" ? (
                            <AlertCircle size={12} className="text-red-500" aria-hidden="true" />
                          ) : (
                            <Check size={12} className="text-emerald-500" aria-hidden="true" />
                          )}
                          {tc.tool_name}
                        </summary>
                        <pre className="mt-1 max-h-40 overflow-auto text-xs">
                          {JSON.stringify(tc.tool_args, null, 2)}
                        </pre>
                        {tc.tool_status === "done" && tc.tool_result !== undefined && (() => {
                          const result = tc.tool_result;
                          return <ToolResultRenderer content={result} toolName={tc.tool_name} onExpand={onExpandResult ? () => onExpandResult(result, tc.tool_name) : undefined} />;
                        })()}
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
                <div className={`w-full max-w-2xl rounded-xl border p-4 shadow-sm ${
                  msg.tool_status === "running" ? "border-accent/30 bg-accent/5"
                  : msg.tool_status === "error" ? "border-red-500/30 bg-red-500/5"
                  : "border-border bg-bg-secondary"
                }`}>
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <Wrench size={14} className={msg.tool_status === "running" ? "text-accent" : "text-text-muted"} aria-hidden="true" />
                    <span className={msg.tool_status === "running" ? "text-accent" : "text-text-primary"}>{msg.tool_name}</span>
                    {msg.tool_status === "running" && (
                      <Loader2 size={14} className="text-accent animate-spin motion-reduce:animate-none ml-1" aria-hidden="true" />
                    )}
                    {msg.tool_status === "done" && (
                      <span className="ml-1 inline-flex items-center gap-1 text-xs text-emerald-500">
                        <Check size={12} aria-hidden="true" />
                        {t('chat.done')}
                      </span>
                    )}
                  </div>
                  <details className="mt-2" open={shouldAutoExpand(msg.tool_name)}>
                    <summary className="cursor-pointer text-xs text-text-secondary">{t('chat.arguments')}</summary>
                    <pre className="mt-1 max-h-40 overflow-auto rounded-lg bg-bg-tertiary p-2 text-xs">
                      {JSON.stringify(msg.tool_args, null, 2)}
                    </pre>
                  </details>
                  {msg.tool_status === "done" && msg.tool_result !== undefined && (() => {
                    const result = msg.tool_result;
                    return (
                    <>
                      <div className="mt-3 text-xs text-text-secondary">
                        {t('chat.result')}
                      </div>
                      <ToolResultRenderer content={result} toolName={msg.tool_name} onExpand={onExpandResult ? () => onExpandResult(result, msg.tool_name) : undefined} />
                    </>
                    );
                  })()}
                </div>
              </div>
            );
          }
          if (msg.isCompacted) {
            return (
              <div key={index} className="flex items-center gap-3 py-2">
                <div className="h-px flex-1 bg-border" />
                <div className="inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/5 px-3 py-1 text-[11px] text-accent" title={`Context compacted: ${msg.compactBefore}% → ${msg.compactAfter}%`}>
                  <Archive size={11} />
                  <span>Context compressed ({msg.compactBefore ?? '?'}% → {msg.compactAfter ?? '?'}%)</span>
                </div>
                <div className="h-px flex-1 bg-border" />
              </div>
            );
          }
          // pipeline progress card (deli_research, computational_loop)
          if (msg.isTaskProgress && msg.taskType === "pipeline") {
            return (
              <div key={index} className="flex justify-center px-4">
                <div className="w-full max-w-2xl">
                  <PipelineProgressCard msg={msg} />
                </div>
              </div>
            );
          }
          return (
            <div
              key={index}
              className={`group flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : ""} ${
                selectedMsgIds.has(index) ? "bg-accent/5" : ""
              }`}
              onClick={(e) => {
                if (e.shiftKey && lastClickedIdx !== null) {
                  const start = Math.min(index, lastClickedIdx);
                  const end = Math.max(index, lastClickedIdx);
                  const newSelected = new Set(selectedMsgIds);
                  for (let i = start; i <= end; i++) {
                    newSelected.add(i);
                  }
                  setSelectedMsgIds(newSelected);
                } else if (e.ctrlKey || e.metaKey) {
                  setSelectedMsgIds(prev => {
                    const next = new Set(prev);
                    if (next.has(index)) next.delete(index);
                    else next.add(index);
                    return next;
                  });
                } else {
                  setSelectedMsgIds(new Set([index]));
                }
                setLastClickedIdx(index);
              }}
              onContextMenu={(e) => {
                e.preventDefault();
                setCtxMenu({ x: e.clientX, y: e.clientY, msg, index });
              }}
            >
              <div className="flex flex-col items-center gap-0.5">
                {pinnedMsgIds.has(index) && (
                  <span className="text-[10px] text-yellow-500" title={t('chat.pinned')}>📌</span>
                )}
                <div
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm ${
                    msg.role === "user" ? "bg-accent text-white" : msg.persona ? "bg-purple-500 text-white" : "bg-bg-tertiary text-text-secondary"
                  }`}
                >
                  {msg.role === "user" ? t('chat.you') : msg.persona ? msg.persona[0].toUpperCase() : t('chat.ai')}
                </div>
              </div>
              <div
                className={`max-w-[75%] px-5 py-3 ${
                  msg.role === "user"
                    ? "bg-accent text-white rounded-2xl rounded-br-none"
                    : "rounded-2xl rounded-bl-none"
                }`}
              >
                <div className="mb-1 flex items-center gap-2 text-xs opacity-70">
                  <span>{msg.role === "user" ? t('chat.you') : msg.persona || t('chat.assistant')}</span>
                  {msg.persona && (
                    <span className="rounded bg-purple-500/15 px-1.5 py-0.5 text-[10px] font-medium text-purple-400">
                      {msg.persona}
                    </span>
                  )}
                  <span>
                    {msg.timestamp === "streaming" ? t('chat.typing') : formatTimeAgo(msg.timestamp)}
                  </span>
                </div>
                {msg.reasoning && (
                  <details
                    className="mb-2 rounded-lg"
                    open={reasoningOpen[index] ?? (msg.timestamp === "streaming")}
                    onToggle={(e) => setReasoningOpen(prev => ({ ...prev, [index]: (e.target as HTMLDetailsElement).open }))}
                  >
                    <summary className="cursor-pointer select-none text-xs font-medium text-text-muted hover:text-text-secondary flex items-center gap-1.5">
                      {msg.timestamp === "streaming" && !msg.content && (
                        <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400 animate-pulse motion-reduce:animate-none" aria-hidden="true" />
                      )}
                      <span>
                        {msg.timestamp === "streaming" && !msg.content
                          ? t('chat.thinking')
                          : t('chat.thoughtProcess')}
                      </span>
                      <span className="opacity-60">
                        · {msg.reasoning.length > 999 ? `${(msg.reasoning.length / 1000).toFixed(1)}k` : msg.reasoning.length}
                      </span>
                    </summary>
                    <div className="mt-1.5 max-h-60 overflow-y-auto whitespace-pre-wrap border-l-2 border-border pl-3 text-xs italic leading-relaxed text-text-muted opacity-80">
                      {msg.reasoning}
                    </div>
                  </details>
                )}
                <div className="text-[15px] leading-relaxed">
                  {msg.content && (
                    searchActive && msg.role === "user" ? (
                      <div className="whitespace-pre-wrap">{highlightText(msg.content)}</div>
                    ) : (
                      <CollapsibleMessageContent content={msg.content} isStreaming={msg.timestamp === "streaming"} />
                    )
                  )}
                  {/* Typing dots — shown when streaming hasn't produced content yet */}
                  {msg.timestamp === "streaming" && !msg.content && !msg.reasoning && (
                    <div className="flex items-center gap-1 py-2" aria-label="Assistant is typing">
                      <span className="h-2 w-2 rounded-full bg-text-muted animate-bounce motion-reduce:animate-none" style={{ animationDelay: '0ms' }} aria-hidden="true" />
                      <span className="h-2 w-2 rounded-full bg-text-muted animate-bounce motion-reduce:animate-none" style={{ animationDelay: '150ms' }} aria-hidden="true" />
                      <span className="h-2 w-2 rounded-full bg-text-muted animate-bounce motion-reduce:animate-none" style={{ animationDelay: '300ms' }} aria-hidden="true" />
                    </div>
                  )}
                </div>
                {msg.role === "assistant" && msg.content && msg.timestamp !== "streaming" && (
                  <div className="mt-1.5 flex items-center justify-end gap-1.5">
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-0.5">
                      <button
                        onClick={() => {
                          setReactions(prev => {
                            const next = { ...prev };
                            if (next[index] === 'up') delete next[index];
                            else next[index] = 'up';
                            return next;
                          });
                        }}
                        className={`rounded p-1 transition-colors ${
                          reactions[index] === 'up' ? 'text-green-500' : 'text-text-muted hover:text-green-500'
                        }`}
                        aria-label="Good response"
                        title="Good response"
                      >
                        <ThumbsUp size={13} aria-hidden="true" />
                      </button>
                      <button
                        onClick={() => {
                          setReactions(prev => {
                            const next = { ...prev };
                            if (next[index] === 'down') delete next[index];
                            else next[index] = 'down';
                            return next;
                          });
                        }}
                        className={`rounded p-1 transition-colors ${
                          reactions[index] === 'down' ? 'text-red-500' : 'text-text-muted hover:text-red-500'
                        }`}
                        aria-label="Bad response"
                        title="Bad response"
                      >
                        <ThumbsDown size={13} aria-hidden="true" />
                      </button>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(msg.content).then(() => {
                            setCopyingId(index);
                            setTimeout(() => setCopyingId(null), 1500);
                            toast.success(t('chat.copied'));
                          });
                        }}
                        className="rounded p-1 text-text-muted hover:text-text-primary transition-colors"
                        aria-label="Copy message"
                        title="Copy"
                      >
                        {copyingId === index ? <Check size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}
                      </button>
                      {index === displayMessages.length - 1 && !isStreaming && (
                        <button
                          onClick={() => {
                            // Remove last assistant message and resend previous user message
                            const lastUserMsg = [...messages].reverse().find(m => m.role === "user");
                            if (lastUserMsg) {
                              setMessages(prev => {
                                const lastAssistantIdx = [...prev].reverse().findIndex(m => m.role === "assistant");
                                if (lastAssistantIdx === -1) return prev;
                                const realIdx = prev.length - 1 - lastAssistantIdx;
                                return prev.slice(0, realIdx);
                              });
                              setTimeout(() => {
                                setInput(lastUserMsg.content);
                                sendMessage();
                              }, 50);
                            }
                          }}
                          className="rounded p-1 text-text-muted hover:text-text-primary transition-colors"
                          aria-label="Regenerate response"
                          title="Regenerate"
                        >
                          <RotateCw size={13} aria-hidden="true" />
                        </button>
                      )}
                    </div>
                    <SaveToMemoryButton content={msg.content} />
                  </div>
                )}
                {msg.role === "user" && msg.timestamp !== "streaming" && (
                  <div className="relative mt-1.5 flex justify-end gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    {(() => {
                      const tIdx = displayTurnMap[index];
                      const canCollapse = turns.length > 3 && tIdx < turns.length - 2 && tIdx >= 0;
                      if (!canCollapse) return null;
                      return (
                        <button
                          onClick={() => {
                            setCollapsedTurns(prev => {
                              const next = new Set(prev);
                              next.add(tIdx);
                              return next;
                            });
                          }}
                          className="rounded p-1 text-text-muted hover:text-accent transition-colors"
                          aria-label="Collapse turn"
                          title="Collapse this turn"
                        >
                          <ChevronDown size={13} />
                        </button>
                      );
                    })()}
                    <button
                      onClick={() => {
                        // Edit: save current content to history, fill input, remove message
                        setEditHistory(prev => ({
                          ...prev,
                          [index]: [...(prev[index] || []), msg.content],
                        }));
                        setInput(msg.content);
                        setMessages(prev => prev.filter((_, i) => i !== index));
                        setTimeout(() => textareaRef.current?.focus(), 0);
                      }}
                      className="rounded p-1 text-text-muted hover:text-text-primary transition-colors"
                      aria-label="Edit message"
                      title="Edit"
                    >
                      <Pencil size={13} aria-hidden="true" />
                    </button>
                    {editHistory[index] && editHistory[index].length > 0 && (
                      <button
                        onClick={() => setViewingHistory(viewingHistory === index ? null : index)}
                        className="text-[10px] italic text-text-muted hover:text-accent transition-colors cursor-pointer"
                        title={`Edited ${editHistory[index].length}x — click to view history`}
                      >
                        {t('chat.editedXTimes', { count: editHistory[index].length })}
                      </button>
                    )}
                    {viewingHistory === index && editHistory[index] && (
                      <div className="absolute right-0 top-full z-20 mt-1 w-80 rounded-lg border border-border bg-bg-secondary p-3 shadow-xl">
                        <div className="mb-2 text-[10px] font-bold uppercase tracking-widest text-text-muted">{t('chat.editHistory')}</div>
                        {editHistory[index].map((old, hi) => {
                          const prev = hi === 0 ? '' : editHistory[index][hi - 1];
                          const oldLines = old.split('\n');
                          const prevLines = prev.split('\n');
                          const maxLines = Math.max(oldLines.length, prevLines.length);
                          return (
                            <div key={hi} className="mb-2 border-b border-border pb-2 last:border-0">
                              <div className="text-[10px] text-text-muted">v{hi + 1}</div>
                              <div className="mt-1 rounded border border-border/50 bg-bg-tertiary/50 p-1.5 font-mono text-[10px] leading-relaxed">
                                {Array.from({ length: maxLines }).map((_, li) => {
                                  const oldL = oldLines[li] || '';
                                  const prevL = prevLines[li] || '';
                                  if (oldL === prevL) return <div key={li} className="text-text-muted/50">{oldL || '\u00a0'}</div>;
                                  if (prevL && !oldL) return <div key={li} className="text-red-400/60 line-through">{prevL}</div>;
                                  if (!prevL && oldL) return <div key={li} className="text-green-400/80">+ {oldL}</div>;
                                  return <div key={li} className="text-green-400/80">+ {oldL}</div>;
                                })}
                              </div>
                            </div>
                          );
                        })}
                        <button
                          onClick={() => setViewingHistory(null)}
                          className="mt-1 text-[10px] text-accent hover:underline"
                        >
                          Close
                        </button>
                      </div>
                    )}
                    <button
                      onClick={() => {
                        setMessages(prev => prev.filter((_, i) => i !== index));
                        toast.success('Message deleted');
                      }}
                      className="rounded p-1 text-text-muted hover:text-error transition-colors"
                      aria-label="Delete message"
                      title="Delete"
                    >
                      <Trash2 size={13} aria-hidden="true" />
                    </button>
                  </div>
                )}
                {/* Plan confirm/cancel buttons */}
                {msg.isPlan && msg.planId && (
                  <div className="mt-3 flex gap-2 items-center border-t border-border/50 pt-3">
                    {(() => {
                      const st = planExecState?.[msg.planId!];
                      if (st === "executing") {
                        return (
                          <span className="mr-auto flex items-center gap-1 text-xs text-accent">
                            <Loader2 size={12} className="animate-spin" aria-hidden="true" />
                            Executing…
                          </span>
                        );
                      }
                      if (st === "done") {
                        return (
                          <span className="mr-auto flex items-center gap-1 text-xs text-success">
                            <Check size={12} aria-hidden="true" />
                            Executed
                          </span>
                        );
                      }
                      return null;
                    })()}
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
          Footer: () => (
            <>
              {isStreaming && !isPaused && (
                <div className="flex items-center gap-1.5 px-4 py-3">
                  <span className="block h-1.5 w-1.5 rounded-full bg-accent animate-bounce-dot motion-reduce:animate-none" style={{ animationDelay: '0ms' }} aria-hidden="true" />
                  <span className="block h-1.5 w-1.5 rounded-full bg-accent animate-bounce-dot motion-reduce:animate-none" style={{ animationDelay: '160ms' }} aria-hidden="true" />
                  <span className="block h-1.5 w-1.5 rounded-full bg-accent animate-bounce-dot motion-reduce:animate-none" style={{ animationDelay: '320ms' }} aria-hidden="true" />
                </div>
              )}
              <div ref={messagesEndRef} style={{ height: 1 }} />
            </>
          ),
          EmptyPlaceholder: () => (
            <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-accent/10">
                <span className="text-3xl">🦅</span>
              </div>
              <div className="space-y-1">
                <div className="text-lg font-bold text-text-primary">{t('chat.welcome') || 'Start a conversation'}</div>
                <div className="max-w-sm text-sm text-text-muted">
                  Ask about materials science, run simulations, analyze data, or use <kbd className="rounded border border-border bg-bg-tertiary px-1 text-xs">Ctrl+K</kbd> to explore tools.
                </div>
              </div>
              <div className="flex flex-wrap justify-center gap-2 pt-2">
                {['Explain XRD patterns', 'Optimize a supercell', 'Calculate band structure', 'Find materials with E_g > 2eV'].map((s) => (
                  <button
                    key={s}
                    onClick={() => { setInput(s); textareaRef.current?.focus(); }}
                    className="rounded-lg border border-border bg-bg-secondary px-3 py-1.5 text-xs text-text-secondary hover:border-accent/50 hover:text-accent transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ),
        }}
        followOutput={'auto'}
      />
      {/* Scroll to bottom button */}
      {showScrollBtn && (
        <button
          onClick={() => virtuosoRef.current?.scrollToIndex({ index: displayMessages.length - 1, behavior: 'smooth' })}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 rounded-full border border-border bg-bg-secondary p-2 shadow-lg hover:bg-bg-tertiary transition-colors"
          aria-label="Scroll to latest"
          title="Scroll to bottom"
        >
          <ArrowDown size={16} />
        </button>
      )}
      </div>

      {/* ARIA live region for screen readers */}
      <div aria-live="polite" className="sr-only">
        {isStreaming ? t('chat.typing') : ''}
      </div>

      {/* Context menu */}
      {ctxMenu && (
        <div
          className="fixed z-50 min-w-[160px] overflow-hidden rounded-lg border border-border bg-bg-secondary py-1 shadow-xl"
          style={{ left: ctxMenu.x, top: ctxMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {ctxMenu.msg.content && (
            <button
              onClick={() => {
                navigator.clipboard.writeText(ctxMenu.msg.content);
                toast.success(t('chat.copied'));
                setCtxMenu(null);
              }}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-primary hover:bg-bg-tertiary"
            >
              <Copy size={12} /> Copy message
            </button>
          )}
          {ctxMenu.msg.content && (
            <button
              onClick={() => {
                const snippet = ctxMenu.msg.content.substring(0, 200);
                setQuotedMsg(`${ctxMenu.msg.role === 'user' ? 'You' : 'Assistant'}: ${snippet}${ctxMenu.msg.content.length > 200 ? '…' : ''}`);
                setCtxMenu(null);
                setTimeout(() => textareaRef.current?.focus(), 0);
              }}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-primary hover:bg-bg-tertiary"
            >
              <CornerUpLeft size={12} aria-hidden="true" /> Quote reply
            </button>
          )}
          {ctxMenu.msg.role === "user" && (
            <button
              onClick={() => {
                setEditHistory(prev => ({
                  ...prev,
                  [ctxMenu.index]: [...(prev[ctxMenu.index] || []), ctxMenu.msg.content],
                }));
                setInput(ctxMenu.msg.content);
                setMessages(prev => prev.filter((_, i) => i !== ctxMenu.index));
                setCtxMenu(null);
                setTimeout(() => textareaRef.current?.focus(), 0);
              }}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-primary hover:bg-bg-tertiary"
            >
              <Pencil size={12} aria-hidden="true" /> Edit
            </button>
          )}
          {ctxMenu.index === displayMessages.length - 1 && ctxMenu.msg.role === "assistant" && (
            <button
              onClick={() => {
                const lastUserMsg = [...messages].reverse().find(m => m.role === "user");
                if (lastUserMsg) {
                  setMessages(prev => {
                    const idx = [...prev].reverse().findIndex(m => m.role === "assistant");
                    if (idx === -1) return prev;
                    return prev.slice(0, prev.length - 1 - idx);
                  });
                  setTimeout(() => { setInput(lastUserMsg.content); sendMessage(); }, 50);
                }
                setCtxMenu(null);
              }}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-primary hover:bg-bg-tertiary"
            >
              <RotateCw size={12} /> Regenerate
            </button>
          )}
          <button
            onClick={() => {
              setPinnedMsgIds(prev => {
                const next = new Set(prev);
                if (next.has(ctxMenu.index)) next.delete(ctxMenu.index);
                else next.add(ctxMenu.index);
                return next;
              });
              toast.success(pinnedMsgIds.has(ctxMenu.index) ? t('chat.unpinned') : t('chat.pinned'));
              setCtxMenu(null);
            }}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-primary hover:bg-bg-tertiary"
          >
            <Pin size={12} aria-hidden="true" /> {pinnedMsgIds.has(ctxMenu.index) ? t('chat.unpin') : t('chat.pin')}
          </button>
          <button
            onClick={() => {
              setMessages(prev => prev.filter((_, i) => i !== ctxMenu.index));
              toast.success('Message deleted');
              setCtxMenu(null);
            }}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-error hover:bg-bg-tertiary"
          >
            <Trash2 size={12} aria-hidden="true" /> Delete
          </button>
        </div>
      )}

      <div className="shrink-0 border-t border-border bg-bg-secondary p-4">
        {!isConnected && (wsFailed || wsReconnecting) && (
            <div className="mb-3 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning" role="alert">
              {wsFailed ? (
                <div className="flex items-center gap-2">
                  <span>Backend appears to have stopped. Reconnection attempts exhausted.</span>
                  <button
                    onClick={() => { try { (window as any).__TAURI__.invoke("start_backend"); } catch {} }}
                    className="rounded bg-warning/20 px-2 py-0.5 text-xs font-medium text-warning hover:bg-warning/30"
                  >
                    Restart Backend
                  </button>
                </div>
              ) : (
                <>
                  <span>{wsReconnecting ? t('chat.reconnecting') : t('chat.backendNotConnected')}</span>
                  {wsReconnecting && (
                    <span className="ml-2 inline-flex items-center gap-1">
                      <span className="h-1.5 w-1.5 rounded-full bg-warning animate-pulse" />
                      retrying…
                    </span>
                  )}
                </>
              )}
            </div>
          )}

        {undoWindow && undoSend && (
          <div className="mb-3 flex items-center justify-between rounded-lg border border-accent/20 bg-accent/5 px-3 py-2 text-xs">
            <span className="text-accent">Message sent · Undo available for 5s</span>
            <button
              onClick={undoSend}
              className="rounded bg-accent/20 px-2 py-0.5 font-medium text-accent hover:bg-accent/30"
            >
              Undo
            </button>
          </div>
        )}

        {pendingClarifications.length > 0 && (
          <div className="mb-3 rounded-lg border border-accent/20 bg-accent/5 px-3 py-2 text-xs text-accent">
            💡 Agent is waiting for your clarification — answer above or type below
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
                className="btn-success px-4 py-2 text-sm"
              >
                {t('chat.approve')}
              </button>
              <button
                onClick={() => respondToApproval(pendingApproval.request_id, false)}
                className="btn-danger px-4 py-2 text-sm"
              >
                {t('chat.deny')}
              </button>
            </div>
          </div>
        )}

        {/* HRI #4: SUGGEST mode — 可编辑代码块, 用户 Approve/Edit/Deny */}
        {pendingSuggestCode && respondToSuggestCode && (
          <SuggestCodeEditor
            code={pendingSuggestCode.code}
            risk={pendingSuggestCode.risk}
            reason={pendingSuggestCode.reason}
            onRespond={respondToSuggestCode}
          />
        )}

        {/* @mention popover */}
        {showMentions && filteredMentions.length > 0 && (
          <div className="mb-2 rounded-lg border border-border bg-bg-secondary shadow-lg overflow-hidden">
            <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-text-muted border-b border-border">
              Mention
            </div>
            {filteredMentions.map((m, i) => (
              <button
                key={m.id}
                onClick={() => {
                  const atIdx = input.lastIndexOf('@');
                  if (atIdx !== -1) {
                    setInput(input.slice(0, atIdx) + `@${m.label} `);
                  }
                  setShowMentions(false);
                  setTimeout(() => textareaRef.current?.focus(), 0);
                }}
                onMouseEnter={() => setMentionIdx(i)}
                className={`flex w-full items-center gap-3 px-4 py-2 text-left transition-colors ${
                  i === mentionIdx ? 'bg-bg-tertiary' : 'hover:bg-bg-tertiary'
                }`}
              >
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/10 text-xs font-bold text-accent">
                  {m.label[0]}
                </span>
                <div className="flex-1">
                  <div className="text-xs font-medium text-text-primary">{m.label}</div>
                  <div className="text-[11px] text-text-muted">{m.desc}</div>
                </div>
              </button>
            ))}
          </div>
        )}

        <div className="mb-2 flex items-center justify-between gap-2">
          {/* Mode selector — segmented control */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 rounded-lg bg-bg-tertiary p-0.5" role="radiogroup" aria-label="Chat mode">
              <button
                onClick={() => { setMode("chat"); setResearchMode(false); }}
                aria-checked={mode === "chat" && !researchMode}
                role="radio"
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  mode === "chat" && !researchMode ? "bg-accent text-white" : "text-text-muted hover:text-text-secondary"
                }`}
              >
                {t('chat.mode.chat')}
              </button>
              <button
                onClick={() => { setMode("plan"); setResearchMode(false); }}
                aria-checked={mode === "plan"}
                role="radio"
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  mode === "plan" ? "bg-accent text-white" : "text-text-muted hover:text-text-secondary"
                }`}
              >
                {t('chat.mode.plan')}
              </button>
              <button
                onClick={() => { setResearchMode(!researchMode); if (!researchMode) setMode("chat"); }}
                aria-checked={researchMode}
                role="radio"
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  researchMode ? "bg-accent text-white" : "text-text-muted hover:text-text-secondary"
                }`}
              >
                {t('chat.mode.research') || 'Research'}
              </button>
            </div>
            {/* Persona selector */}
            {currentPersona && (
              <div className="relative">
                <button
                  onClick={() => setShowPersonaSelector(!showPersonaSelector)}
                  className="flex items-center gap-1 rounded-lg border border-border bg-bg-tertiary px-2 py-1 text-xs text-text-secondary hover:text-text-primary"
                  aria-label="Select persona"
                >
                  <span className="flex h-4 w-4 items-center justify-center rounded-full bg-purple-500 text-[10px] text-white">
                    {currentPersona[0].toUpperCase()}
                  </span>
                  {currentPersona}
                  <ChevronDown size={12} />
                </button>
                {showPersonaSelector && (
                  <div className="absolute bottom-full left-0 mb-2 w-48 overflow-hidden rounded-lg border border-border bg-bg-secondary shadow-lg">
                    <div className="px-2 py-1 text-[10px] font-bold uppercase tracking-widest text-text-muted border-b border-border">
                      Personas
                    </div>
                    {personas.map((p) => (
                      <button
                        key={p.name}
                        onClick={() => {
                          setCurrentPersona?.(p.name);
                          setShowPersonaSelector(false);
                        }}
                        className={`flex w-full items-center gap-2 px-3 py-1.5 text-xs text-left transition-colors ${
                          currentPersona === p.name ? 'bg-bg-tertiary text-text-primary' : 'text-text-secondary hover:bg-bg-tertiary'
                        }`}
                      >
                        <span className={`flex h-4 w-4 items-center justify-center rounded-full text-[10px] ${
                          currentPersona === p.name ? 'bg-purple-500 text-white' : 'bg-bg-tertiary text-text-muted'
                        }`}>
                          {p.name[0].toUpperCase()}
                        </span>
                        <span className="truncate">{p.name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Thinking intensity — always visible */}
          <div className="flex items-center gap-1">
            <span className="text-xs text-text-muted">🧠</span>
            {(["low", "medium", "high"] as const).map((level) => (
              <button
                key={level}
                onClick={() => setThinkingIntensity(level)}
                className={`rounded px-1.5 py-0.5 text-[10px] capitalize transition-colors ${
                  thinkingIntensity === level ? "bg-accent/20 text-accent font-medium" : "text-text-muted hover:text-text-secondary"
                }`}
                title={`Thinking: ${level}`}
              >
                {level[0].toUpperCase()}
              </button>
            ))}
          </div>
          {onOpenFiles && (
            <button
              onClick={onOpenFiles}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-text-muted hover:text-text-secondary transition-colors"
              title={t('chat.openFiles') || 'Open files panel'}
              aria-label="Open files panel"
            >
              <FolderTree size={14} />
            </button>
          )}
          <details className="relative">
            <summary className="flex cursor-pointer list-none items-center gap-1 rounded-md px-2 py-1 text-xs text-text-muted hover:text-text-secondary transition-colors" title="Options" aria-label="Chat options" role="button" aria-expanded="false">
              <Settings size={14} />
            </summary>
            <div className="absolute right-0 top-full z-50 mt-1 w-56 rounded-lg border border-border bg-bg-secondary p-3 shadow-lg">
              <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
                <input type="checkbox" checked={autoApprove} onChange={(e) => toggleAutoApprove(e.target.checked)} className="h-3.5 w-3.5" />
                {t('chat.autoApprove')}
              </label>
              {toggleSuggestMode && (
                <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-text-secondary" title="SUGGEST mode: all CodeAct code shown for editing before execution (LoA Level 4-6)">
                  <input type="checkbox" checked={!!suggestMode} onChange={(e) => toggleSuggestMode(e.target.checked)} className="h-3.5 w-3.5" />
                  SUGGEST mode
                  <span className="text-[10px] text-text-muted">(review every code)</span>
                </label>
              )}
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
          <div className="mb-2 space-y-1">
            {pendingMessages.map((pmsg, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg border border-accent/20 bg-accent/5 px-3 py-1.5 text-xs"
              >
                <Clock className="w-3 h-3 text-accent shrink-0" />
                <span className="text-accent/80 truncate flex-1">
                  {pmsg.length > 80 ? pmsg.slice(0, 80) + "…" : pmsg}
                </span>
                <span className="text-text-muted/50 shrink-0">
                  #{i + 1}
                </span>
              </div>
            ))}
            <div className="text-[10px] text-text-muted pl-1">
              {pendingMessages.length} queued — will send after current response
            </div>
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

        {quotedMsg && (
          <div className="mb-2 flex items-start gap-2 rounded-lg border-l-2 border-accent bg-accent/5 px-3 py-2">
            <CornerUpLeft size={14} className="mt-0.5 shrink-0 text-accent" aria-hidden="true" />
            <div className="flex-1 truncate text-xs text-text-secondary">{quotedMsg}</div>
            <button
              onClick={() => setQuotedMsg(null)}
              className="shrink-0 text-text-muted hover:text-text-primary"
              aria-label="Remove quote"
            >
              <X size={14} aria-hidden="true" />
            </button>
          </div>
        )}

        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`flex items-end gap-3 rounded-lg transition-colors ${isDragOver ? 'border-2 border-dashed border-accent bg-accent/5 p-1' : 'border-2 border-transparent p-1'}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) {
                const files = Array.from(e.target.files);
                const fakeEvent = { dataTransfer: { files } } as unknown as React.DragEvent;
                handleDrop(fakeEvent);
                e.target.value = '';
              }
            }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn-ghost h-9 w-9 shrink-0 rounded-lg"
            aria-label={t('chat.attachFile') || 'Attach file'}
            title={t('chat.attachFile') || 'Attach file'}
            disabled={!isConnected}
          >
            <Paperclip size={18} aria-hidden="true" />
          </button>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => {
              const val = e.target.value;
              setInput(val);
              setShowCommands(val.startsWith('/') && !val.includes(' '));
              // @mention detection: check if user just typed @
              const atIdx = val.lastIndexOf('@');
              if (atIdx !== -1 && atIdx === val.length - 1) {
                setShowMentions(true);
                setMentionQuery('');
                setMentionIdx(0);
              } else if (showMentions && atIdx !== -1) {
                // Extract text after @ for filtering
                const afterAt = val.slice(atIdx + 1);
                if (afterAt.includes(' ') || afterAt.includes('\n')) {
                  setShowMentions(false);
                } else {
                  setMentionQuery(afterAt);
                }
              } else if (showMentions && atIdx === -1) {
                setShowMentions(false);
              }
              autoResize();
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
              // --- @mention navigation ---
              if (showMentions && filteredMentions.length > 0) {
                if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  setMentionIdx(prev => (prev + 1) % filteredMentions.length);
                  return;
                }
                if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  setMentionIdx(prev => (prev - 1 + filteredMentions.length) % filteredMentions.length);
                  return;
                }
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  const selected = filteredMentions[mentionIdx];
                  if (selected) {
                    const atIdx = input.lastIndexOf('@');
                    if (atIdx !== -1) {
                      setInput(input.slice(0, atIdx) + `@${selected.label} `);
                    }
                    setShowMentions(false);
                    return;
                  }
                }
                if (e.key === 'Escape') {
                  e.preventDefault();
                  setShowMentions(false);
                  return;
                }
              }
              // Up arrow in empty input → recall last user message
              if (e.key === 'ArrowUp' && !showCommands && !showMentions && input === '') {
                const history = inputHistoryRef.current;
                if (history.length > 0) {
                  e.preventDefault();
                  if (historyIdxRef.current === -1) {
                    savedInputRef.current = input;
                    historyIdxRef.current = history.length - 1;
                  } else if (historyIdxRef.current > 0) {
                    historyIdxRef.current--;
                  }
                  setInput(history[historyIdxRef.current]);
                  setTimeout(autoResize, 0);
                }
              }
              if (e.key === 'ArrowDown' && !showCommands && !showMentions && historyIdxRef.current !== -1) {
                e.preventDefault();
                const history = inputHistoryRef.current;
                if (historyIdxRef.current < history.length - 1) {
                  historyIdxRef.current++;
                  setInput(history[historyIdxRef.current]);
                } else {
                  // back to saved input
                  historyIdxRef.current = -1;
                  setInput(savedInputRef.current);
                }
                setTimeout(autoResize, 0);
              }
              if (e.key === "Enter" && (!e.shiftKey || (e.ctrlKey || e.metaKey))) {
                e.preventDefault();
                const val = input.trim();
                if (val) {
                  inputHistoryRef.current.push(val);
                  historyIdxRef.current = -1;
                }
                sendMessage();
                setTimeout(autoResize, 0);
              }
            }}
            placeholder={
              mode === "plan"
                ? t('chat.placeholderPlan')
                : isConnected
                ? t('chat.placeholderConnected')
                : wsFailed
                ? t('chat.placeholderOffline')
                : 'Connecting…'
            }
            rows={2}
            className="input min-h-[56px] max-h-[200px] resize-none flex-1 overflow-y-auto"
            aria-label="Message input"
          />
          <div className="flex flex-col items-center gap-1">
            {isStreaming ? (
              <div className="flex items-center gap-1">
                {isPaused ? (
                  <button
                    onClick={resumeGeneration}
                    className="btn-ghost h-11 px-3"
                    aria-label="Resume"
                    title="Resume generation"
                  >
                    <Play className="h-4 w-4" />
                  </button>
                ) : (
                  <button
                    onClick={pauseGeneration}
                    className="btn-ghost h-11 px-3"
                    aria-label="Pause"
                    title="Pause generation"
                  >
                    <Pause className="h-4 w-4" />
                  </button>
                )}
                <button
                  onClick={stopGeneration}
                  className="btn-danger h-11 px-5"
                  aria-label={t('chat.stop') || 'Stop'}
                  title={t('chat.stop') || 'Stop generation'}
                >
                  <span className="flex items-center gap-1.5">
                    <span className="h-3 w-3 rounded-sm bg-current" />
                    {t('chat.stop') || 'Stop'}
                  </span>
                </button>
              </div>
            ) : (
              <button
                onClick={() => { setQuotedMsg(null); sendMessage(); }}
                disabled={!isConnected || !input.trim()}
                className="btn-primary h-11 px-5"
                aria-label={mode === "plan" ? t('chat.mode.plan') : t('chat.send')}
              >
                {mode === "plan" ? t('chat.mode.plan') : t('chat.send')}
              </button>
            )}
          </div>
        </div>
        <div className="mt-1 flex items-center justify-between px-1 text-[10px] text-text-muted">
          <span>
            <kbd className="rounded border border-border bg-bg-tertiary px-1">Enter</kbd> send · <kbd className="rounded border border-border bg-bg-tertiary px-1">Shift+Enter</kbd> newline · <kbd className="rounded border border-border bg-bg-tertiary px-1">↑</kbd> history
          </span>
          {input.length > 0 && (
            <span className="flex items-center gap-2">
              <span className={input.length > 4000 ? 'text-warning' : ''}>{input.length} chars</span>
              <span className="text-text-muted/60">·</span>
              <span>~{Math.round(input.length / 4)} tokens</span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
