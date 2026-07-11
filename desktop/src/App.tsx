import { useState, useEffect, useRef, lazy, Suspense, Fragment } from "react";
import { useTranslation } from "react-i18next";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { listen } from "@tauri-apps/api/event";
import Pet from "./Pet";
import ErrorBoundary from "./components/ErrorBoundary";
import { LanguageSwitcher } from "./components/LanguageSwitcher";
const EmotionTrackerPanel = lazy(() => import("./components/EmotionTracker"));
const SandboxPanel = lazy(() => import("./components/SandboxPanel"));
const PeriodicTable = lazy(() => import("./components/PeriodicTable"));
const Notebook = lazy(() => import("./components/Notebook"));
const SweepDashboard = lazy(() => import("./components/SweepDashboard"));
const StructureViewer = lazy(() => import("./components/StructureViewer"));
const PersonaManager = lazy(() => import("./components/PersonaManager"));
import { PROVIDERS, formatTimeAgo } from "./lib/constants";
import { api } from "./lib/api";
import { getApiBase } from "./lib/api-client";
import { useToolRunner } from "./hooks/useToolRunner";
import { useMemory } from "./hooks/useMemory";
import { useKnowledge } from "./hooks/useKnowledge";
import { useWorkspace } from "./hooks/useWorkspace";
import { useHPC } from "./hooks/useHPC";
import { useTeam } from "./hooks/useTeam";
import { usePlugins } from "./hooks/usePlugins";
import { useProject } from "./hooks/useProject";
import { useLogs } from "./hooks/useLogs";
import { useConfig } from "./hooks/useConfig";
import { useChatAndConnection } from "./hooks/useChatAndConnection";
import { PetStatusWidget } from "./components/PetStatusWidget";
import AutoloopProgress from "./components/AutoloopProgress";
import { ContextBar } from "./components/ContextBar";
import { ChatPanel } from "./components/panels/ChatPanel";
import { MetricsBar } from "./components/MetricsBar";
import { MemoryPanel } from "./components/panels/MemoryPanel";
import { SettingsPanel } from "./components/panels/SettingsPanel";
import { KnowledgePanel } from "./components/panels/KnowledgePanel";
import { PluginsPanel } from "./components/panels/PluginsPanel";
import { ToolsPanel } from "./components/panels/ToolsPanel";
import { SkillsPanel } from "./components/panels/SkillsPanel";
import { TeamPanel } from "./components/panels/TeamPanel";
import { ProjectPanel } from "./components/panels/ProjectPanel";
import { ResearchProjectPanel } from "./components/panels/ResearchProjectPanel";
import { FilesPanel } from "./components/panels/FilesPanel";
import { ThreadsPanel } from "./components/panels/ThreadsPanel";
import { ReviewPanel } from "./components/panels/ReviewPanel";
import { CoderPanel } from "./components/panels/CoderPanel";
import { BenchmarkPanel } from "./components/panels/BenchmarkPanel";
import { ResultPanel } from "./components/panels/ResultPanel";
import { CodeSearchPanel } from "./components/panels/CodeSearchPanel";
import { GitPanel } from "./components/panels/GitPanel";
import { HPCPanel } from "./components/panels/HPCPanel";
import { useTheme } from "./hooks/useTheme";
import { useFocusTrap } from "./hooks/useFocusTrap";
import { toast } from "./components/Toast";
import { LogsPanel } from "./components/panels/LogsPanel";
import { TerminalPanel } from "./components/panels/TerminalPanel";
import { PanelHeader } from "./components/settings-shared";
import type { DiffEntry, Checkpoint, ToolInfo, SkillInfo } from "./types/domain";
import {
  MessageSquare, Wrench, FolderTree, Terminal, Settings,
  Users, Code2, BookOpen,
  MessageCircle, Bird, Briefcase, HelpCircle,
  ChevronDown, Sparkles,
  Search, Grid, Sun, Moon, Plus, Trash2,
  Maximize2, GitBranch,
} from 'lucide-react';

const IS_PET_MODE = window.location.search.includes("pet=1");

async function openPetWindow() {
  try {
    const existing = await WebviewWindow.getByLabel("pet");
    if (existing) {
      await existing.setFocus();
      return;
    }
    const pet = new WebviewWindow("pet", {
      url: "index.html?pet=1",
      width: 180,
      height: 220,
      transparent: true,
      decorations: false,
      alwaysOnTop: true,
      skipTaskbar: true,
      resizable: false,
      center: false,
      x: window.screen.width - 200,
      y: window.screen.height - 260,
    });
    pet.once("tauri://error", (e) => {
      console.error("[pet] failed to create window:", e);
    });
  } catch (err) {
    console.error("[pet] open failed:", err);
  }
}

function LoadingFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-accent border-t-transparent" />
    </div>
  );
}

export default function App() {
  const { theme, toggleTheme } = useTheme();

  if (IS_PET_MODE) {
    return <Pet />;
  }

  const { t } = useTranslation();

  // Focus trap for modals
  const toolPaletteRef = useRef<HTMLDivElement>(null);
  const guideModalRef = useRef<HTMLDivElement>(null);

  // ── Sidebar state ────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<
    | "chat" | "tools" | "memory" | "skills" | "settings" | "files"
    | "terminal" | "review" | "knowledge" | "logs" | "plugins"
    | "threads" | "project" | "projects" | "team" | "coder" | "benchmark"
    | "evolution" | "execute" | "workflows" | "explore" | "diagnose"
    | "hpc" | "periodic" | "notebook" | "sandbox" | "sweep"
    | "structure" | "emotion" | "provenance" | "side" | "solver"
    | "persona" | "result" | "code" | "git"
  >("chat");
  const [sidebarHidden, setSidebarHidden] = useState(false);
  // Result panel state: content + toolName for the expanded view
  const [resultContent, setResultContent] = useState("");
  const [resultToolName, setResultToolName] = useState<string | undefined>(undefined);
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = localStorage.getItem('sidebar-width');
    return saved ? parseInt(saved) : 224;
  });

  // Sidebar drag-to-resize
  const sidebarDraggingRef = useRef(false);
  const handleSidebarResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    sidebarDraggingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!sidebarDraggingRef.current) return;
      const w = Math.max(180, Math.min(400, e.clientX));
      setSidebarWidth(w);
    };
    const onUp = () => {
      if (sidebarDraggingRef.current) {
        sidebarDraggingRef.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        localStorage.setItem('sidebar-width', String(sidebarWidth));
      }
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [sidebarWidth]);
  const [toolPaletteOpen, setToolPaletteOpen] = useState(false);
  const [shortcutHelpOpen, setShortcutHelpOpen] = useState(false);
  const [toolSearch, setToolSearch] = useState("");
  const [draggedTab, setDraggedTab] = useState<string | null>(null);
  const [dragOverTab, setDragOverTab] = useState<string | null>(null);
  const [customTabOrder, setCustomTabOrder] = useState<any[]>(() => {
    try {
      const saved = localStorage.getItem('huginn:tab-order');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  // ── Tools / Skills state arrays ──────────────────────────────
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [skills, setSkills] = useState<SkillInfo[]>([]);

  // ── Hook calls ───────────────────────────────────────────────
  const {
    config, configDirty, configSavedMsg, settingsTab, llmCredOptions,
    expandedModels, expandedAgents,
    setConfig, setConfigDirty, setConfigSavedMsg, setSettingsTab,
    pushConfig, saveConfig,
    updateModel, addModel, removeModel,
    updateAgent, addAgent, removeAgent,
    toggleModelExpanded, toggleAgentExpanded,
    switchPersona,
  } = useConfig();

  const {
    teamObjective, teamPlan, teamRunning, teamResult, teamFusionResult, teamError,
    setTeamObjective,
    handleTeamPlan, handleTeamRun, handleTeamFusion,
  } = useTeam();

  const {
    hpcHost, hpcUsername, hpcScheduler, hpcKeyPath, hpcCommand,
    hpcJobName, hpcWalltime, hpcNodes, hpcNtasks, hpcQueue,
    hpcJobId, hpcRunning, hpcResult, hpcError,
    setHpcHost, setHpcUsername, setHpcScheduler, setHpcKeyPath,
    setHpcCommand, setHpcJobName, setHpcWalltime, setHpcNodes,
    setHpcNtasks, setHpcQueue, setHpcJobId,
    handleHpcTest, handleHpcSubmit, handleHpcStatus,
  } = useHPC();

  const {
    projectContext, projectContextSource, projectContextMsg,
    codebaseStatus, codebaseQuery, codebaseResults, codebaseMsg,
    setProjectContext, setCodebaseQuery,
    loadProjectContext, saveProjectContext,
    loadCodebaseStatus, indexCodebase, searchCodebase,
  } = useProject();

  const memory = useMemory();
  const {
    memories, memoriesLoading, memoryHasMore, memoryStats, memorySearch, memoryFilter, memoryForm, memoryMsg, memoryView,
    setMemorySearch, setMemoryFilter, setMemoryForm, setMemoryView,
    loadMemory, loadMemoryStats, searchMemory, createMemory, deleteMemory,
    updateMemory, promoteMemory, pruneMemory, syncMemoryMd,
  } = memory;

  const {
    cwd, selectedFile,
    editorContent, editorDirty, editorMsg,
    terminalOutput, terminalInput, terminalEndRef,
    setEditorContent, setEditorDirty,
    setTerminalOutput, setTerminalInput,
    loadDir, saveFile, renderTree,
  } = useWorkspace();

  const {
    backendLogs, logFilter, backendLogEndRef,
    setBackendLogs, setLogFilter,
  } = useLogs();

  const {
    kbDocs, kbAvailable, kbLoading, kbMsg, kbQuery, kbChunks, parseLoading, uploadPct,
    fileInputRef, parseFileInputRef,
    setKbQuery,
    loadKnowledge, uploadKnowledge, parseDocument, loadDocumentGraph,
    deleteKnowledge, queryKnowledge, ingestUrl, loadProvenanceDag,
  } = useKnowledge();

  const {
    mcpServers, discoveredServers, mcpMsg, newMcp,
    setNewMcp,
    loadMcp, discoverMcp, connectMcp, disconnectMcp,
  } = usePlugins();

  // ── Checkpoint / review state ────────────────────────────────
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [activeCp, setActiveCp] = useState<string | null>(null);
  const [diffs, setDiffs] = useState<DiffEntry[]>([]);

  // ── Provenance state ─────────────────────────────────────────
  const [provenanceRecords, setProvenanceRecords] = useState<any[]>([]);
  const [provSortCol, setProvSortCol] = useState<'tool' | 'file' | 'format' | 'time' | null>(null);
  const [provSortDir, setProvSortDir] = useState<'asc' | 'desc'>('asc');

  const toggleProvSort = (col: 'tool' | 'file' | 'format' | 'time') => {
    if (provSortCol === col) {
      setProvSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setProvSortCol(col);
      setProvSortDir('asc');
    }
  };

  const sortedProvRecords = (() => {
    if (!provSortCol) return provenanceRecords;
    const sorted = [...provenanceRecords].sort((a, b) => {
      const av = (provSortCol === 'time') ? (a.timestamp || a.time || '') : (a[provSortCol] || a.path || '');
      const bv = (provSortCol === 'time') ? (b.timestamp || b.time || '') : (b[provSortCol] || b.path || '');
      return String(av).localeCompare(String(bv));
    });
    return provSortDir === 'asc' ? sorted : sorted.reverse();
  })();
  const [provenanceExpanded, setProvenanceExpanded] = useState<number | null>(null);

  const loadProvenance = async () => {
    try {
      const data = await api.get<any[]>("/provenance/recent?n=50");
      setProvenanceRecords(Array.isArray(data) ? data : []);
    } catch (e: any) {
      console.error("[provenance] load failed:", e);
    }
  };

  const createCheckpoint = async () => {
    if (!cwd) return;
    try {
      const cp = await api.post<Checkpoint>("/checkpoints", { path: cwd });
      setCheckpoints((prev) => [cp, ...prev]);
      setActiveCp(cp.id);
      loadDiffs(cp.id);
    } catch (e: any) {
      console.error("[review] create checkpoint failed:", e);
    }
  };

  const loadDiffs = async (cpId: string) => {
    try {
      const data = await api.get<{ diffs?: DiffEntry[] }>(`/checkpoints/${cpId}/diff`);
      setDiffs((data.diffs as DiffEntry[]) || []);
      setActiveCp(cpId);
    } catch (e: any) {
      console.error("[review] load diffs failed:", e);
    }
  };

  const acceptCheckpoint = async (cpId: string) => {
    try {
      await api.post(`/checkpoints/${cpId}/accept`);
      setCheckpoints((prev) => prev.filter((c) => c.id !== cpId));
      if (activeCp === cpId) {
        setActiveCp(null);
        setDiffs([]);
      }
    } catch (e: any) {
      console.error("[review] accept failed:", e);
    }
  };

  const rejectCheckpoint = async (cpId: string) => {
    try {
      await api.post(`/checkpoints/${cpId}/reject`);
      setCheckpoints((prev) => prev.filter((c) => c.id !== cpId));
      if (activeCp === cpId) {
        setActiveCp(null);
        setDiffs([]);
      }
    } catch (e: any) {
      console.error("[review] reject failed:", e);
    }
  };

  // ── Tool runner instances ────────────────────────────────────
  const [coderTask, setCoderTask] = useState("");
  const [coderAutoApprove, setCoderAutoApprove] = useState(false);
  const [coderMaxIters, setCoderMaxIters] = useState<number | "">("");
  const coder = useToolRunner<string>({
    endpoint: "/coder",
    buildPayload: () => {
      const body: Record<string, any> = { task: coderTask, auto_approve: coderAutoApprove };
      if (coderMaxIters !== "") body.max_iterations = Number(coderMaxIters);
      return body;
    },
    extractResult: (data) => data.final_answer || "Done.",
    inputGuard: () => !!coderTask.trim(),
    defaultError: "Coder run failed.",
  });

  const [benchEvolve, setBenchEvolve] = useState(false);
  const [benchCategories, setBenchCategories] = useState("");
  const bench = useToolRunner<any>({
    endpoint: "/bench/run",
    buildPayload: () => {
      const body: any = { evolve: benchEvolve };
      if (benchCategories.trim()) body.categories = benchCategories.split(",").map((s) => s.trim()).filter(Boolean);
      return body;
    },
    extractResult: (data) => data.report,
    defaultError: "Benchmark failed.",
  });

  const evolve = useToolRunner<any>({
    endpoint: "/evolve/run",
    buildPayload: () => ({}),
    extractResult: (data) => data.report,
    defaultError: "Evolution failed.",
  });

  const [executeStages, setExecuteStages] = useState("");
  const [executeWorkingDir, setExecuteWorkingDir] = useState(".");
  const [executeName, setExecuteName] = useState("execute");
  const execute = useToolRunner<any>({
    endpoint: "/execute",
    buildPayload: () => {
      let stages: any;
      try { stages = JSON.parse(executeStages); } catch { throw new Error("Stages must be valid JSON."); }
      return { stages, working_dir: executeWorkingDir, name: executeName };
    },
    extractResult: (data) => data,
    inputGuard: () => !!executeStages.trim(),
    defaultError: "Execution failed.",
  });

  const [workflowTemplates, setWorkflowTemplates] = useState<string[]>([]);
  const [workflowTemplate, setWorkflowTemplate] = useState("");
  const [workflowArgs, setWorkflowArgs] = useState("");
  const workflow = useToolRunner<any>({
    endpoint: "/workflows/execute",
    buildPayload: () => {
      const args: Record<string, any> = {};
      workflowArgs.split(" ").forEach((a) => {
        if (!a.includes("=")) return;
        const [k, v] = a.split("=");
        try { args[k] = JSON.parse(v); } catch { args[k] = v; }
      });
      return { template: workflowTemplate, args };
    },
    extractResult: (data) => data,
    isSuccess: (data) => !data.error,
    inputGuard: () => !!workflowTemplate,
    defaultError: "Workflow execution failed.",
  });

  const [exploreObjective, setExploreObjective] = useState("");
  const [exploreMaxIters, setExploreMaxIters] = useState(20);
  const [exploreMaxBranches, setExploreMaxBranches] = useState(10);
  const explore = useToolRunner<any>({
    endpoint: "/explore",
    buildPayload: () => ({
      objective: exploreObjective,
      max_iterations: exploreMaxIters,
      max_branches: exploreMaxBranches,
    }),
    extractResult: (data) => data,
    defaultError: "Exploration failed.",
    inputGuard: () => !!exploreObjective.trim(),
  });

  const [diagnoseError, setDiagnoseError] = useState("");
  const [diagnoseSoftware, setDiagnoseSoftware] = useState("");
  const [diagnoseCalcType, setDiagnoseCalcType] = useState("");
  const [diagnoseContext, setDiagnoseContext] = useState("");
  const diagnose = useToolRunner<any>({
    endpoint: "/diagnose",
    buildPayload: () => ({
      error_message: diagnoseError,
      software: diagnoseSoftware || undefined,
      calculation_type: diagnoseCalcType || undefined,
      context: diagnoseContext || undefined,
    }),
    extractResult: (data) => data.data,
    inputGuard: () => !!diagnoseError.trim(),
    defaultError: "Diagnosis failed.",
  });

  // ── Side dialogue state ──────────────────────────────────────
  const [sideInput, setSideInput] = useState("");
  const [sideQuestions, setSideQuestions] = useState<any[]>([]);
  const [sideAnswerId, setSideAnswerId] = useState<string | null>(null);
  const [sideAnswer, setSideAnswer] = useState("");
  const [sideMsg, setSideMsg] = useState("");

  const loadSidePending = async () => {
    try {
      const data = await api.get<any[]>("/side/pending");
      setSideQuestions(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setSideMsg(`Load failed: ${e.message}`);
    }
  };

  const sendSideQuestion = async () => {
    if (!sideInput.trim()) return;
    try {
      await api.post("/side", { question: sideInput });
      setSideInput("");
      loadSidePending();
    } catch (e: any) {
      setSideMsg(`Send failed: ${e.message}`);
    }
  };

  const answerSideQuestion = async (id: string) => {
    if (!sideAnswer.trim()) return;
    try {
      await api.post("/side", { id, answer: sideAnswer });
      setSideAnswer("");
      setSideAnswerId(null);
      loadSidePending();
    } catch (e: any) {
      setSideMsg(`Answer failed: ${e.message}`);
    }
  };

  const clearSide = async () => {
    try {
      await api.del("/side");
      setSideQuestions([]);
    } catch (e: any) {
      setSideMsg(`Clear failed: ${e.message}`);
    }
  };

  // ── Unified solver state ─────────────────────────────────────
  const [solverModels, setSolverModels] = useState<string[]>([]);
  const [solverModel, setSolverModel] = useState("");
  const [solverInput, setSolverInput] = useState("");
  const [solverDerived, setSolverDerived] = useState("");
  const [solverSolution, setSolverSolution] = useState("");
  const [solverPlotUrl, setSolverPlotUrl] = useState("");
  const [solverRunning, setSolverRunning] = useState(false);
  const [solverError, setSolverError] = useState("");

  const loadSolverModels = async () => {
    try {
      const data = await api.get<string[]>("/unified/models");
      const models = Array.isArray(data) ? data : [];
      setSolverModels(models);
      if (models.length > 0 && !solverModel) setSolverModel(models[0]);
    } catch (e: any) {
      setSolverError(`Load models failed: ${e.message}`);
    }
  };

  const solverDerive = async () => {
    setSolverRunning(true); setSolverError(""); setSolverDerived(""); setSolverSolution(""); setSolverPlotUrl("");
    try {
      const data = await api.post<{ derived?: string }>("/unified/derive", { model: solverModel, input: solverInput });
      setSolverDerived(data.derived || JSON.stringify(data, null, 2));
    } catch (e: any) { setSolverError(e.message); }
    setSolverRunning(false);
  };

  const solverSolve = async () => {
    setSolverRunning(true); setSolverError(""); setSolverSolution(""); setSolverPlotUrl("");
    try {
      const data = await api.post<{ solution?: string }>("/unified/solve", { model: solverModel, derived: solverDerived, input: solverInput });
      setSolverSolution(data.solution || JSON.stringify(data, null, 2));
    } catch (e: any) { setSolverError(e.message); }
    setSolverRunning(false);
  };

  const solverPlot = async () => {
    setSolverRunning(true); setSolverError(""); setSolverPlotUrl("");
    try {
      const data = await api.post<{ plot_url?: string; image?: string }>("/unified/plot", { model: solverModel, solution: solverSolution });
      setSolverPlotUrl(data.plot_url || (data.image ? `data:image/png;base64,${data.image}` : "") || JSON.stringify(data));
    } catch (e: any) { setSolverError(e.message); }
    setSolverRunning(false);
  };

  // ── Chat and connection hook ─────────────────────────────────
  const {
    messages, input, mode, pendingPlan,
    chatSearchOpen, chatSearchQuery,
    isStreaming,
    messagesEndRef,
    isConnected, status, wsReconnecting, wsFailed, undoWindow, undoSend,
    wsClientRef,
    personaList, personaEmotion, pendingClarifications,
    threads, activeThread,
    showGuide, closeGuide,
    setInput, setMode, setMessages, setPendingPlan, setChatSearchOpen, setChatSearchQuery,
    setThreads, setShowGuide, switchThread,
    sendMessage, answerClarification,
    loadThreads, createThread, renameThread, deleteThread,
    startBackend,
    pendingApproval, autoApprove, respondToApproval, toggleAutoApprove,
    autoloopPhase, autoloopProgress,
    contextPct,
    thinkingIntensity, setThinkingIntensity,
    pendingMessages,
    stopGeneration,
    pauseGeneration, resumeGeneration, isPaused,
    researchMode, setResearchMode,
    petState,
  } = useChatAndConnection({
    config,
    activeTab,
    pushConfig,
    onAutoCheckpoint: (cp) => {
      setCheckpoints((prev) => [{ id: cp.id, base: cp.base, files: cp.files }, ...prev]);
      setActiveCp(cp.id);
    },
    onExplorationResult: (data) => {
      explore.setResult(data);
      explore.setRunning(false);
    },
    toolsLength: tools.length,
    skillsLength: skills.length,
    setTools,
    setSkills,
  });

  // Focus trap for modals — must be after toolPaletteOpen/showGuide are available
  useFocusTrap(toolPaletteRef, toolPaletteOpen);
  useFocusTrap(guideModalRef, showGuide);

  // Dynamic favicon based on connection/streaming state
  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 32; canvas.height = 32;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let raf: number;

    const draw = (progress = 0) => {
      ctx.clearRect(0, 0, 32, 32);
      // Circle background
      ctx.beginPath();
      ctx.arc(16, 16, 14, 0, Math.PI * 2);
      if (isStreaming) {
        ctx.fillStyle = '#3b82f6'; // accent blue
      } else if (!isConnected) {
        ctx.fillStyle = '#ef4444'; // error red
      } else {
        ctx.fillStyle = '#22c55e'; // success green
      }
      ctx.fill();
      // Streaming: draw rotating arc overlay
      if (isStreaming) {
        ctx.beginPath();
        ctx.arc(16, 16, 12, progress * Math.PI * 2, progress * Math.PI * 2 + Math.PI * 1.2);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      // Letter H
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 18px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('H', 16, 16);
      setFavicon(canvas.toDataURL());
      if (isStreaming) {
        raf = requestAnimationFrame(() => draw(progress + 0.02));
      }
    };
    draw();

    function setFavicon(href: string) {
      let link = document.querySelector<HTMLLinkElement>("link[rel='icon']");
      if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.head.appendChild(link);
      }
      link.href = href;
    }
    return () => cancelAnimationFrame(raf);
  }, [isStreaming, isConnected]);

  // ── loadWorkflowTemplates ────────────────────────────────────
  const loadWorkflowTemplates = async () => {
    try {
      const data = await api.get<any[]>("/workflows");
      setWorkflowTemplates(Array.isArray(data) ? data : []);
    } catch (e: any) {
      console.error("[workflows] load failed:", e);
    }
  };

  // ── useEffect: sidebar auto-expand ───────────────────────────
  type SidebarTabId = typeof activeTab;
  interface SidebarTabItem {
    id: SidebarTabId;
    label: string;
    icon: React.ReactNode;
    indented?: boolean;
  }
  interface SidebarGroupData {
    key: string;
    label: string;
    tabs: SidebarTabItem[];
  }
  const sidebarGroupsData: SidebarGroupData[] = [
    {
      key: "core",
      label: t('nav.core'),
      tabs: [
        { id: "chat" as const, label: t('tab.chat'), icon: <MessageSquare size={16} /> },
        { id: "team" as const, label: t('tab.team'), icon: <Users size={16} /> },
        { id: "coder" as const, label: t('tab.coder'), icon: <Code2 size={16} /> },
      ],
    },
    {
      key: "workspace",
      label: t('nav.workspace'),
      tabs: [
        { id: "files" as const, label: t('tab.files'), icon: <FolderTree size={16} /> },
        { id: "code" as const, label: t('tab.code'), icon: <Search size={16} /> },
        { id: "git" as const, label: t('tab.git'), icon: <GitBranch size={16} /> },
        { id: "terminal" as const, label: t('tab.terminal'), icon: <Terminal size={16} /> },
        { id: "tools" as const, label: t('tab.tools'), icon: <Wrench size={16} /> },
        { id: "skills" as const, label: t('tab.skills'), icon: <Sparkles size={16} /> },
      ],
    },
    {
      key: "system",
      label: t('nav.system'),
      tabs: [
        { id: "threads" as const, label: t('tab.threads'), icon: <MessageCircle size={16} /> },
        { id: "settings" as const, label: t('tab.settings'), icon: <Settings size={16} /> },
      ],
    },
  ];

  // Apply custom tab order if saved
  const orderedSidebarGroups = (() => {
    if (customTabOrder.length === 0) return sidebarGroupsData;
    // Flatten all tabs, reorder by custom order, then re-group preserving original group structure
    const allTabs = sidebarGroupsData.flatMap(g => g.tabs);
    const reordered = [...allTabs].sort((a, b) => {
      const ai = customTabOrder.indexOf(a.id);
      const bi = customTabOrder.indexOf(b.id);
      if (ai === -1 && bi === -1) return 0;
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    });
    // Rebuild groups: keep original group keys/labels, but distribute reordered tabs
    const groupSizes = sidebarGroupsData.map(g => g.tabs.length);
    let idx = 0;
    return sidebarGroupsData.map((g, gi) => ({
      key: g.key,
      label: g.label,
      tabs: reordered.slice(idx, idx + groupSizes[gi]),
    }));
  })();

  const handleTabDrop = (targetId: string) => {
    if (!draggedTab || draggedTab === targetId) return;
    const allTabs = sidebarGroupsData.flatMap(g => g.tabs.map(t => t.id)) as any[];
    const fromIdx = allTabs.indexOf(draggedTab);
    const toIdx = allTabs.indexOf(targetId);
    if (fromIdx === -1 || toIdx === -1) return;
    const reordered = [...allTabs];
    reordered.splice(fromIdx, 1);
    reordered.splice(toIdx, 0, draggedTab as any);
    setCustomTabOrder(reordered);
    localStorage.setItem('huginn:tab-order', JSON.stringify(reordered));
    setDraggedTab(null);
    setDragOverTab(null);
  };
  useEffect(() => {
    if (activeTab === "knowledge") {
      loadKnowledge();
    }
    if (activeTab === "plugins") {
      loadMcp();
      discoverMcp();
    }
    if (activeTab === "threads") {
      loadThreads();
    }
    if (activeTab === "project") {
      loadProjectContext();
      loadCodebaseStatus();
    }
    if (activeTab === "memory") {
      loadMemory();
      loadMemoryStats();
    }
    if (activeTab === "workflows" && workflowTemplates.length === 0) {
      loadWorkflowTemplates();
    }
    if (activeTab === "provenance") {
      loadProvenance();
    }
    if (activeTab === "side") {
      loadSidePending();
    }
    if (activeTab === "solver" && solverModels.length === 0) {
      loadSolverModels();
    }
  }, [activeTab, memoryFilter.category, memoryFilter.tier]);

  // ── System tray: restart backend from tray menu ────────────────
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    let unlisten: (() => void) | null = null;
    (async () => {
      unlisten = await listen("tray-restart-backend", async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        try { await invoke("stop_backend"); } catch { /* not running */ }
        await startBackend();
      });
    })();
    return () => { unlisten?.(); };
  }, [startBackend]);

  // ── Keyboard shortcuts: Ctrl+K palette, Ctrl+F search, Ctrl+N new thread ──
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isMod = e.ctrlKey || e.metaKey;
      if (isMod && e.key === "k") {
        e.preventDefault();
        setToolPaletteOpen((prev) => !prev);
      }
      if (isMod && e.key === "f" && activeTab === "chat") {
        e.preventDefault();
        setChatSearchOpen((prev) => !prev);
      }
      if (isMod && e.key === "n" && activeTab === "chat") {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent("huginn:new-thread"));
      }
      if (isMod && e.key === "b") {
        e.preventDefault();
        setSidebarHidden((prev) => !prev);
      }
      if (isMod && e.key === ",") {
        e.preventDefault();
        setActiveTab("settings");
      }
      if (isMod && e.key === "l" && activeTab === "chat") {
        e.preventDefault();
        setMessages([]);
        toast.success(t('chat.cleared'));
      }
      if (e.key === "Escape" && toolPaletteOpen) {
        setToolPaletteOpen(false);
        setToolSearch("");
      }
      if (isMod && e.key === "/") {
        e.preventDefault();
        setShortcutHelpOpen(prev => !prev);
      }
      if (e.key === "Escape" && shortcutHelpOpen) {
        setShortcutHelpOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toolPaletteOpen, activeTab, setChatSearchOpen, setSidebarHidden, setActiveTab, setMessages, t, shortcutHelpOpen]);

  // ── Derived constants ────────────────────────────────────────
  const providerLabel = PROVIDERS.find((p) => p.id === config.provider)?.label || config.provider;
  const allTabs = sidebarGroupsData.flatMap((g) => g.tabs);
  const activeTabInfo = allTabs.find((t) => t.id === activeTab);

  const handleCoderRun = coder.run;
  const handleExecuteRun = execute.run;
  const handleDiagnoseRun = diagnose.run;
  const handleWorkflowRun = workflow.run;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-primary text-text-primary">
      {/* Skip-link for keyboard users */}
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[9999] focus:rounded-md focus:bg-accent focus:px-3 focus:py-1.5 focus:text-sm focus:text-white">
        Skip to content
      </a>
      {/* Sidebar — chat-first: 4 primary destinations + tool palette */}
      {sidebarHidden ? (
        <button
          onClick={() => setSidebarHidden(false)}
          className="z-50 flex h-full w-10 items-center justify-center border-r border-border bg-bg-secondary text-text-muted hover:text-text-primary transition-colors"
          title="Show sidebar"
          aria-label="Show sidebar"
        >
          <ChevronDown size={16} className="-rotate-90" />
        </button>
      ) : (
        <>
      <aside
        className="sidebar-shell flex flex-col border-r border-border bg-bg-secondary"
        style={{ width: `${sidebarWidth}px` }}
      >
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-border">
          <img src="/raven-logo-64.png" srcSet="/raven-logo-64.png 1x, /raven-logo-128.png 2x" alt="Huginn" className="h-8 w-8 rounded-md object-contain" />
          <div className="flex flex-1 flex-col">
            <div className="text-[15px] font-bold tracking-tight">Huginn</div>
            <div className="text-[12px] text-text-muted leading-none font-medium">{t('app.subtitle')}</div>
          </div>
          <button
            onClick={() => setSidebarHidden(true)}
            className="text-text-muted hover:text-text-primary transition-colors"
            title="Hide sidebar"
            aria-label="Hide sidebar"
          >
            <ChevronDown size={16} className="rotate-90" />
          </button>
        </div>

        {/* Compact icon tabs — horizontal bar */}
        <div className="flex items-center gap-1 border-b border-border px-2 py-1.5">
          {([
            { id: "chat", label: t('tab.chat'), icon: <MessageSquare size={16} /> },
            { id: "knowledge", label: t('tab.knowledge'), icon: <BookOpen size={16} /> },
            { id: "projects", label: "Projects", icon: <Briefcase size={16} /> },
            { id: "threads", label: t('tab.threads'), icon: <MessageCircle size={16} /> },
            { id: "result", label: "Result", icon: <Maximize2 size={16} /> },
            { id: "settings", label: t('tab.settings'), icon: <Settings size={16} /> },
          ] as const).map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              aria-current={activeTab === item.id ? "page" : undefined}
              title={item.label}
              className={`flex flex-1 items-center justify-center rounded-md px-1 py-1.5 transition-all duration-150 ${
                activeTab === item.id
                  ? "bg-bg-tertiary text-text-primary"
                  : "text-text-muted hover:bg-bg-tertiary hover:text-text-secondary"
              }`}
            >
              {item.icon}
            </button>
          ))}
        </div>

        {/* Thread list when chat is active */}
        {activeTab === "chat" && (
          <div className="flex flex-col border-b border-border">
            <button
              onClick={() => { createThread(); }}
              className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-text-secondary hover:bg-bg-tertiary transition-colors"
            >
              <Plus size={16} /> {t('threads.new') || 'New Chat'}
            </button>
            <div className="max-h-[calc(100vh-320px)] overflow-y-auto px-1 pb-2">
              {threads.map((th) => (
                <div
                  key={th.id}
                  onClick={() => switchThread(th.id)}
                  className={`group flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-sm transition-colors ${
                    activeThread === th.id
                      ? "bg-accent/15 text-text-primary"
                      : "text-text-secondary hover:bg-bg-tertiary"
                  }`}
                >
                  <MessageSquare size={13} className="shrink-0 opacity-50" />
                  <span className="flex-1 truncate">{th.label}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteThread(th.id); }}
                    className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-error transition-opacity"
                    title={t('common.delete') || 'Delete'}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
              {threads.length === 0 && (
                <div className="px-3 py-4 text-center text-xs text-text-muted">
                  {t('threads.emptyHint') || 'No conversations yet'}
                </div>
              )}
            </div>
          </div>
        )}

        <nav className="flex-1 overflow-y-auto px-2 py-2" aria-label="Main navigation">
          {/* More tools — opens command palette */}
          <button
            onClick={() => setToolPaletteOpen(true)}
            className="sidebar-nav-item flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[15px] font-bold text-text-secondary hover:bg-bg-tertiary hover:text-text-primary transition-all duration-150 border-t border-border/50 pt-3"
          >
            <Grid size={16} /> More Tools
          </button>

          {/* Quick access to active tool (if non-primary) */}
          {activeTab !== "chat" && activeTab !== "knowledge" && activeTab !== "projects" && activeTab !== "threads" && activeTab !== "settings" && activeTabInfo && (
            <div className="mt-2 flex items-center gap-2 rounded-md bg-bg-tertiary px-2.5 py-1.5">
              <span className="flex-shrink-0 text-accent">{activeTabInfo.icon}</span>
              <span className="truncate text-[13px] font-semibold text-text-primary">{activeTabInfo.label}</span>
              <button
                onClick={() => setActiveTab("chat")}
                className="ml-auto text-text-muted hover:text-text-primary"
                title="Back to chat"
              >
                <ChevronDown size={14} className="rotate-90" />
              </button>
            </div>
          )}
        </nav>

        {petState && <PetStatusWidget petState={petState} />}
        <div className="border-t border-border px-3 py-3">
          <div className="flex items-center gap-2 text-[13px] text-text-muted">
            <span className={`h-2 w-2 rounded-full ${
              isConnected ? "bg-success" : wsFailed ? "bg-error" : "bg-warning animate-pulse"
            }`} />
            <span className="truncate">
              {wsFailed ? "Backend stopped" : wsReconnecting ? t('chat.reconnecting') + '…' : status || (isConnected ? t('status.connected') : t('status.offline'))}
            </span>
          </div>
          <div className="mt-2 flex gap-1.5">
            <button
              onClick={() => setShowGuide(true)}
              className="sidebar-footer-btn flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1 text-[13px] text-text-muted hover:text-text-secondary"
              title="Help"
            >
              <HelpCircle size={13} /> Guide
            </button>
            <button
              onClick={toggleTheme}
              className="sidebar-footer-btn flex items-center justify-center rounded-md px-2 py-1 text-[13px] text-text-muted hover:text-text-secondary"
              title={theme === "dark" ? "Switch to light" : "Switch to dark"}
            >
              {theme === "dark" ? <Sun size={13} /> : <Moon size={13} />}
            </button>
            <button
              onClick={openPetWindow}
              className="sidebar-footer-btn flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1 text-[13px] text-text-muted hover:text-text-secondary"
              title={t('app.summonPet')}
            >
              <Bird size={13} /> {t('app.pet')}
            </button>
          </div>
        </div>
      </aside>
      {/* Sidebar resize handle */}
      <div
        onMouseDown={handleSidebarResizeStart}
        className="group relative w-1 shrink-0 cursor-col-resize transition-colors hover:bg-accent/30 active:bg-accent/50"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        title="Drag to resize"
      >
        <div className="absolute inset-y-0 left-0 w-full" />
      </div>
      </>
      )}

      {/* Main */}
      <main id="main-content" className="flex flex-1 flex-col min-w-0 bg-bg-primary" aria-busy={isConnected ? undefined : "true"}>
        {/* Header */}
        <header className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
          <div className="flex items-center gap-2.5">
            <span className="text-text-muted">
              {activeTabInfo?.icon}
            </span>
            <span className="text-sm font-semibold">
              {activeTabInfo?.label}
            </span>
            {activeTab !== "chat" && activeTab !== "knowledge" && activeTab !== "threads" && activeTab !== "settings" && (
              <button
                onClick={() => setActiveTab("chat")}
                className="flex items-center gap-1 rounded-md px-2 py-0.5 text-xs text-text-muted hover:bg-bg-tertiary hover:text-text-primary transition-colors"
                title="Back to chat"
              >
                <ChevronDown size={12} className="rotate-90" /> Chat
              </button>
            )}
            {activeTab === "chat" && (
              <>
                <span className="badge border border-border bg-bg-tertiary text-text-secondary">
                  {config.models.length > 0
                    ? `${config.models.filter((m) => m.enabled).length} ${t('app.models')}`
                    : `${providerLabel} / ${config.model || t('app.default')}`}
                </span>
                <label className="flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
                  <input
                    type="checkbox"
                    checked={config.team_mode_enabled}
                    onChange={(e) => {
                      const next = { ...config, team_mode_enabled: e.target.checked };
                      setConfig(next);
                      setConfigDirty(true);
                      saveConfig(next);
                    }}
                  />
                  Team mode
                </label>
              </>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setToolPaletteOpen(true)}
              className="flex items-center gap-2 rounded-lg border border-border bg-bg-tertiary px-2.5 py-1 text-xs text-text-muted hover:text-text-primary transition-colors"
              title="Open tool palette (Ctrl+K)"
            >
              <Grid size={14} />
              <kbd className="text-[10px] font-bold">⌘K</kbd>
            </button>
            <LanguageSwitcher />
            {activeTab === "chat" && (
              <button
                onClick={() => { setChatSearchOpen((p: boolean) => !p); if (chatSearchOpen) setChatSearchQuery(""); }}
                className={`rounded-lg p-1.5 transition-colors ${chatSearchOpen ? "bg-bg-tertiary text-accent" : "text-text-muted hover:text-text-secondary"}`}
                title={t('chat.search')}
                aria-label={t('chat.search')}
                aria-expanded={chatSearchOpen}
              >
                <Search size={16} />
              </button>
            )}
            {!isConnected && !status.includes("starting") && (
              <button
                onClick={startBackend}
                className="badge bg-error/10 text-error border border-error/20 hover:bg-error/20"
              >
                ▶ {t('app.startBackend')}
              </button>
            )}
            {status.includes("starting") && (
              <span className="badge bg-warning/10 text-warning border border-warning/20">
                {t('app.startingBackend')}
              </span>
            )}
          </div>
        </header>

        {/* Real-time metrics bar */}
        <MetricsBar />

        {/* Autoloop progress bar */}
        {autoloopPhase && (
          <AutoloopProgress currentPhase={autoloopPhase} progress={autoloopProgress} />
        )}

        {/* Context window usage indicator */}
        <ContextBar pct={contextPct} />

        {/* Content */}
        <div className="relative flex-1 overflow-hidden">
          {/* Restart waiting overlay */}
          {(wsReconnecting || wsFailed) && (
            <div className="absolute inset-0 z-40 flex items-center justify-center bg-bg-primary/80 backdrop-blur-sm">
              <div className="flex flex-col items-center gap-3 rounded-2xl border border-border bg-bg-secondary p-8 shadow-2xl">
                {wsFailed ? (
                  <>
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-error/10">
                      <span className="text-2xl">!</span>
                    </div>
                    <div className="text-sm font-semibold text-text-primary">Backend stopped</div>
                    <div className="text-xs text-text-muted">Reconnection attempts exhausted. Please restart the backend.</div>
                  </>
                ) : (
                  <>
                    <div className="h-10 w-10 animate-spin rounded-full border-2 border-border border-t-accent" />
                    <div className="text-sm font-semibold text-text-primary">Reconnecting…</div>
                    <div className="text-xs text-text-muted">Waiting for backend to come back online</div>
                  </>
                )}
              </div>
            </div>
          )}
          {/* ChatPanel — 永久挂载，用 hidden 控制可见性，切换 tab 不丢失状态 */}
          <div hidden={activeTab !== "chat"} className="h-full">
            <ChatPanel
              messages={messages}
              chatSearchOpen={chatSearchOpen}
              chatSearchQuery={chatSearchQuery}
              setChatSearchOpen={setChatSearchOpen}
              setChatSearchQuery={setChatSearchQuery}
              wsClientRef={wsClientRef}
              setMessages={setMessages}
              answerClarification={answerClarification}
              pendingClarifications={pendingClarifications}
              isConnected={isConnected}
              wsReconnecting={wsReconnecting}
              wsFailed={wsFailed}
              undoWindow={undoWindow}
              undoSend={undoSend}
              sendMessage={sendMessage}
              pendingPlan={pendingPlan}
              setPendingPlan={setPendingPlan}
              setMode={setMode}
              input={input}
              setInput={setInput}
              mode={mode}
              isStreaming={isStreaming}
              messagesEndRef={messagesEndRef}
              pendingApproval={pendingApproval}
              respondToApproval={respondToApproval}
              autoApprove={autoApprove}
              toggleAutoApprove={toggleAutoApprove}
              thinkingIntensity={thinkingIntensity}
              setThinkingIntensity={setThinkingIntensity}
              pendingMessages={pendingMessages}
              stopGeneration={stopGeneration}
              pauseGeneration={pauseGeneration}
              resumeGeneration={resumeGeneration}
              isPaused={isPaused}
              researchMode={researchMode}
              setResearchMode={setResearchMode}
              contextBudgetTokens={config.context_budget_tokens}
              onExpandResult={(content, toolName) => {
                setResultContent(content);
                setResultToolName(toolName);
                setActiveTab("result");
              }}
            />
          </div>

          {activeTab === "team" && (
            <TeamPanel
              config={config}
              setConfig={setConfig}
              setConfigDirty={setConfigDirty}
              saveConfig={saveConfig}
              isConnected={isConnected}
              teamObjective={teamObjective}
              setTeamObjective={setTeamObjective}
              teamRunning={teamRunning}
              teamError={teamError}
              teamPlan={teamPlan || []}
              teamResult={teamResult}
              teamFusionResult={teamFusionResult}
              handleTeamPlan={handleTeamPlan}
              handleTeamRun={handleTeamRun}
              handleTeamFusion={handleTeamFusion}
            />
          )}

          {activeTab === "coder" && (
            <CoderPanel
              isConnected={isConnected}
              coderTask={coderTask}
              setCoderTask={setCoderTask}
              coderAutoApprove={coderAutoApprove}
              setCoderAutoApprove={setCoderAutoApprove}
              coderMaxIters={coderMaxIters}
              setCoderMaxIters={setCoderMaxIters}
              coderRunning={coder.running}
              coderError={coder.error}
              coderResult={coder.result}
              handleCoderRun={handleCoderRun}
            />
          )}

          <div hidden={activeTab !== "files"}>
            <FilesPanel
              cwd={cwd}
              selectedFile={selectedFile ?? ""}
              editorContent={editorContent}
              editorDirty={editorDirty}
              editorMsg={editorMsg}
              setEditorContent={setEditorContent}
              setEditorDirty={setEditorDirty}
              loadDir={loadDir}
              saveFile={saveFile}
              renderTree={renderTree}
            />
          </div>

          {activeTab === "code" && (
            <CodeSearchPanel apiBase={getApiBase()} />
          )}

          {activeTab === "git" && (
            <GitPanel apiBase={getApiBase()} />
          )}

          {activeTab === "terminal" && (
            <TerminalPanel
              terminalOutput={terminalOutput}
              terminalInput={terminalInput}
              terminalEndRef={terminalEndRef}
              setTerminalOutput={setTerminalOutput}
              setTerminalInput={setTerminalInput}
            />
          )}

          {activeTab === "review" && (
            <ReviewPanel
              cwd={cwd}
              checkpoints={checkpoints}
              activeCp={activeCp}
              diffs={diffs}
              createCheckpoint={createCheckpoint}
              loadDiffs={loadDiffs}
              acceptCheckpoint={acceptCheckpoint}
              rejectCheckpoint={rejectCheckpoint}
            />
          )}

          <div hidden={activeTab !== "knowledge"}>
            <KnowledgePanel
              config={config}
              setConfig={setConfig}
              saveConfig={saveConfig}
              fileInputRef={fileInputRef}
              parseFileInputRef={parseFileInputRef}
              parseLoading={parseLoading}
              uploadPct={uploadPct}
              kbLoading={kbLoading}
              kbMsg={kbMsg}
              kbDocs={kbDocs}
              kbAvailable={kbAvailable}
              kbQuery={kbQuery}
              kbChunks={kbChunks}
              setKbQuery={setKbQuery}
              uploadKnowledge={uploadKnowledge}
              parseDocument={parseDocument}
              loadDocumentGraph={loadDocumentGraph}
              deleteKnowledge={deleteKnowledge}
              queryKnowledge={queryKnowledge}
              ingestUrl={ingestUrl}
              loadProvenanceDag={loadProvenanceDag}
            />
          </div>

          {activeTab === "projects" && (
            <ResearchProjectPanel onOpenInChat={() => setActiveTab("chat")} />
          )}

          {activeTab === "logs" && (
            <LogsPanel
              backendLogs={backendLogs}
              logFilter={logFilter}
              backendLogEndRef={backendLogEndRef}
              setLogFilter={setLogFilter}
              setBackendLogs={setBackendLogs}
            />
          )}

          {activeTab === "tools" && (
            <ToolsPanel tools={tools} isConnected={isConnected} />
          )}

          {activeTab === "skills" && (
            <SkillsPanel skills={skills} isConnected={isConnected} />
          )}

          {activeTab === "emotion" && (
            <ErrorBoundary name="Emotion Tracker">
              <Suspense fallback={<LoadingFallback />}>
                <EmotionTrackerPanel apiBase={getApiBase()} />
              </Suspense>
            </ErrorBoundary>
          )}

          {activeTab === "persona" && (
            <ErrorBoundary name="Persona Manager">
              <Suspense fallback={<LoadingFallback />}>
                <PersonaManager />
              </Suspense>
            </ErrorBoundary>
          )}

          <div hidden={activeTab !== "memory"}>
            <MemoryPanel
              memories={memories}
              memoriesLoading={memoriesLoading}
              memoryHasMore={memoryHasMore}
              loadMoreMemory={() => loadMemory(true)}
              memoryStats={memoryStats}
              memorySearch={memorySearch}
              memoryFilter={memoryFilter}
              memoryForm={memoryForm}
              memoryMsg={memoryMsg}
              memoryView={memoryView}
              setMemorySearch={setMemorySearch}
              setMemoryFilter={setMemoryFilter}
              setMemoryForm={setMemoryForm}
              setMemoryView={setMemoryView}
              loadMemory={loadMemory}
              loadMemoryStats={loadMemoryStats}
              searchMemory={searchMemory}
              createMemory={createMemory}
              deleteMemory={deleteMemory}
              updateMemory={updateMemory}
              promoteMemory={promoteMemory}
              pruneMemory={pruneMemory}
              syncMemoryMd={syncMemoryMd}
            />
          </div>

          {activeTab === "plugins" && (
            <PluginsPanel
              mcpServers={mcpServers}
              discoveredServers={discoveredServers as any}
              mcpMsg={mcpMsg}
              newMcp={newMcp}
              setNewMcp={setNewMcp}
              loadMcp={loadMcp}
              discoverMcp={discoverMcp}
              connectMcp={connectMcp}
              disconnectMcp={disconnectMcp}
            />
          )}

          {activeTab === "project" && (
            <ProjectPanel
              projectContext={projectContext}
              projectContextSource={projectContextSource}
              projectContextMsg={projectContextMsg}
              setProjectContext={setProjectContext}
              loadProjectContext={loadProjectContext}
              saveProjectContext={saveProjectContext}
              codebaseStatus={codebaseStatus}
              codebaseQuery={codebaseQuery}
              codebaseResults={codebaseResults}
              codebaseMsg={codebaseMsg}
              setCodebaseQuery={setCodebaseQuery}
              indexCodebase={indexCodebase}
              searchCodebase={searchCodebase}
            />
          )}

          {activeTab === "threads" && (
            <ThreadsPanel
              threads={threads}
              activeThread={activeThread}
              setThreads={setThreads as any}
              switchThread={switchThread}
              createThread={createThread}
              renameThread={renameThread}
              deleteThread={deleteThread}
            />
          )}

          {activeTab === "result" && (
            <ResultPanel
              resultContent={resultContent}
              resultToolName={resultToolName}
            />
          )}

          <div hidden={activeTab !== "settings"}>
            <SettingsPanel
              config={config}
              configDirty={configDirty}
              configSavedMsg={configSavedMsg}
              settingsTab={settingsTab}
              llmCredOptions={llmCredOptions}
              expandedModels={expandedModels}
              expandedAgents={expandedAgents}
              setConfig={setConfig}
              setConfigDirty={setConfigDirty}
              setConfigSavedMsg={setConfigSavedMsg}
              setSettingsTab={setSettingsTab}
              saveConfig={saveConfig}
              updateModel={updateModel}
              addModel={addModel}
              removeModel={removeModel}
              updateAgent={updateAgent}
              addAgent={addAgent}
              removeAgent={removeAgent}
              toggleModelExpanded={toggleModelExpanded}
              toggleAgentExpanded={toggleAgentExpanded}
              switchPersona={switchPersona}
              startBackend={startBackend}
              status={status}
              isConnected={isConnected}
              personaList={personaList}
              personaEmotion={personaEmotion}
            />
          </div>

          {activeTab === "benchmark" && (
            <BenchmarkPanel
              isConnected={isConnected}
              benchEvolve={benchEvolve}
              setBenchEvolve={setBenchEvolve}
              benchCategories={benchCategories}
              setBenchCategories={setBenchCategories}
              benchRunning={bench.running}
              benchError={bench.error}
              benchResult={bench.result}
              benchRun={bench.run}
            />
          )}

          {activeTab === "evolution" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Evolution</h2>
                  <p className="text-sm text-text-secondary">Run a self-evolution cycle over recent execution logs to learn rules and skills.</p>
                </div>
                <div className="card space-y-3">
                  <button onClick={evolve.run} disabled={evolve.running || !isConnected} className="btn-primary text-xs">
                    {evolve.running ? "Evolving…" : "▶ Run evolution cycle"}
                  </button>
                  {evolve.error && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{evolve.error}</div>}
                </div>
                {evolve.result && (
                  <div className="card space-y-3">
                    <h3 className="text-sm font-semibold">Report</h3>
                    <div className="text-xs text-text-secondary">
                      Failure rules: {evolve.result.failure_rules?.length ?? 0} · Success skills: {evolve.result.success_skills?.length ?? 0} · Prompt patches: {evolve.result.prompt_patches?.length ?? 0}
                    </div>
                    <div className="text-xs text-text-secondary">Total rules: {(evolve.result.total_rules_after ?? 0).toLocaleString()} · Total skills: {(evolve.result.total_skills_after ?? 0).toLocaleString()}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "execute" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Execute</h2>
                  <p className="text-sm text-text-secondary">Run raw workflow stages through the execution orchestrator.</p>
                </div>
                <div className="card space-y-3">
                  <textarea
                    value={executeStages}
                    onChange={(e) => setExecuteStages(e.target.value)}
                    placeholder={`[{"id":"stage1","tool":"diagnose_tool","action":"...","params":{}}]`}
                    rows={8}
                    className="input font-mono text-xs resize-none"
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <input type="text" value={executeWorkingDir} onChange={(e) => setExecuteWorkingDir(e.target.value)} placeholder="Working dir" className="input text-xs" />
                    <input type="text" value={executeName} onChange={(e) => setExecuteName(e.target.value)} placeholder="Workflow name" className="input text-xs" />
                  </div>
                  <button onClick={handleExecuteRun} disabled={execute.running || !isConnected} className="btn-primary text-xs">
                    {execute.running ? "Executing…" : "▶ Execute stages"}
                  </button>
                  {execute.error && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{execute.error}</div>}
                </div>
                {execute.result && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Result</h3>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(execute.result, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "workflows" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Workflows</h2>
                  <p className="text-sm text-text-secondary">Run a workflow template with KEY=VALUE arguments.</p>
                </div>
                <div className="card space-y-3">
                  <select value={workflowTemplate} onChange={(e) => setWorkflowTemplate(e.target.value)} className="input text-sm">
                    <option value="">Select a template</option>
                    {workflowTemplates.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                  <input
                    type="text"
                    value={workflowArgs}
                    onChange={(e) => setWorkflowArgs(e.target.value)}
                    placeholder="key1=value1 key2=value2 ..."
                    className="input text-sm"
                  />
                  <button onClick={handleWorkflowRun} disabled={workflow.running || !isConnected || !workflowTemplate} className="btn-primary text-xs">
                    {workflow.running ? "Running…" : "▶ Run workflow"}
                  </button>
                  {workflow.error && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{workflow.error}</div>}
                </div>
                {workflow.result && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Result</h3>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(workflow.result, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "explore" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Explore</h2>
                  <p className="text-sm text-text-secondary">Systematically search a design space.</p>
                </div>
                <div className="card space-y-3">
                  <input type="text" value={exploreObjective} onChange={(e) => setExploreObjective(e.target.value)} placeholder="Objective, e.g. find highest energy density cathode" className="input text-sm" />
                  <div className="grid grid-cols-2 gap-3">
                    <input type="number" min={1} value={exploreMaxIters} onChange={(e) => setExploreMaxIters(parseInt(e.target.value || "1", 10))} placeholder="Max iterations" className="input text-xs" />
                    <input type="number" min={1} value={exploreMaxBranches} onChange={(e) => setExploreMaxBranches(parseInt(e.target.value || "1", 10))} placeholder="Max branches" className="input text-xs" />
                  </div>
                  <button onClick={explore.run} disabled={explore.running || !isConnected || !exploreObjective.trim()} className="btn-primary text-xs">
                    {explore.running ? "Exploring…" : "▶ Explore"}
                  </button>
                  {explore.error && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{explore.error}</div>}
                </div>
                {explore.result && (
                  <div className="card space-y-3">
                    <h3 className="text-sm font-semibold">Result</h3>
                    <div className="text-xs text-text-secondary">Explored: {explore.result.n_branches_explored} · Pruned: {explore.result.n_branches_pruned} · Convergence: {explore.result.convergence_reason}</div>
                    {explore.result.best_branch && <div className="text-xs text-text-secondary">Best branch: {explore.result.best_branch.name}</div>}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "diagnose" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Diagnose</h2>
                  <p className="text-sm text-text-secondary">Diagnose computational chemistry / MD errors.</p>
                </div>
                <div className="card space-y-3">
                  <textarea value={diagnoseError} onChange={(e) => setDiagnoseError(e.target.value)} placeholder="Paste error message…" rows={4} className="input resize-none text-sm" />
                  <div className="grid grid-cols-3 gap-3">
                    <input type="text" value={diagnoseSoftware} onChange={(e) => setDiagnoseSoftware(e.target.value)} placeholder="Software" className="input text-xs" />
                    <input type="text" value={diagnoseCalcType} onChange={(e) => setDiagnoseCalcType(e.target.value)} placeholder="Calc type" className="input text-xs" />
                    <input type="text" value={diagnoseContext} onChange={(e) => setDiagnoseContext(e.target.value)} placeholder="Context" className="input text-xs" />
                  </div>
                  <button onClick={handleDiagnoseRun} disabled={diagnose.running || !isConnected || !diagnoseError.trim()} className="btn-primary text-xs">
                    {diagnose.running ? "Diagnosing…" : "▶ Diagnose"}
                  </button>
                  {diagnose.error && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{diagnose.error}</div>}
                </div>
                {diagnose.result && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Findings</h3>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{JSON.stringify(diagnose.result, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "hpc" && (
            <HPCPanel
              isConnected={isConnected}
              hpcHost={hpcHost}
              hpcUsername={hpcUsername}
              hpcScheduler={hpcScheduler}
              hpcKeyPath={hpcKeyPath}
              hpcCommand={hpcCommand}
              hpcJobName={hpcJobName}
              hpcWalltime={hpcWalltime}
              hpcNodes={hpcNodes}
              hpcNtasks={hpcNtasks}
              hpcQueue={hpcQueue}
              hpcJobId={hpcJobId}
              hpcRunning={hpcRunning}
              hpcResult={hpcResult}
              hpcError={hpcError}
              setHpcHost={setHpcHost}
              setHpcUsername={setHpcUsername}
              setHpcScheduler={setHpcScheduler}
              setHpcKeyPath={setHpcKeyPath}
              setHpcCommand={setHpcCommand}
              setHpcJobName={setHpcJobName}
              setHpcWalltime={setHpcWalltime}
              setHpcNodes={setHpcNodes}
              setHpcNtasks={setHpcNtasks}
              setHpcQueue={setHpcQueue}
              setHpcJobId={setHpcJobId}
              handleHpcTest={handleHpcTest}
              handleHpcSubmit={handleHpcSubmit}
              handleHpcStatus={handleHpcStatus}
            />
          )}

          {activeTab === "periodic" && (
            <div className="h-full overflow-y-auto p-4">
              <ErrorBoundary name="Periodic Table">
                <Suspense fallback={<LoadingFallback />}>
                  <PeriodicTable API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "notebook" && (
            <div className="h-full overflow-hidden p-4">
              <ErrorBoundary name="Notebook">
                <Suspense fallback={<LoadingFallback />}>
                  <Notebook API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "sandbox" && (
            <div className="h-full overflow-hidden p-4">
              <ErrorBoundary name="Sandbox">
                <Suspense fallback={<LoadingFallback />}>
                  <SandboxPanel API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "sweep" && (
            <div className="h-full overflow-y-auto p-4">
              <ErrorBoundary name="Sweep Dashboard">
                <Suspense fallback={<LoadingFallback />}>
                  <SweepDashboard API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "structure" && (
            <div className="h-full overflow-hidden p-4">
              <ErrorBoundary name="Structure Viewer">
                <Suspense fallback={<LoadingFallback />}>
                  <StructureViewer API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "provenance" && (
            <div className="flex h-full flex-col">
              <PanelHeader title={t('provenance.title')}>
                <button onClick={loadProvenance} className="btn-primary px-3 py-1 text-xs">{t('common.refresh')}</button>
              </PanelHeader>
              <div className="flex-1 overflow-y-auto p-6">
                {provenanceRecords.length === 0 ? (
                  <p className="text-sm text-text-muted">{t('provenance.empty')}</p>
                ) : (
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-border text-text-muted">
                        <th className="py-2 pr-4 font-semibold cursor-pointer select-none hover:text-text-primary" onClick={() => toggleProvSort('tool')}>
                          {t('provenance.tool')}{provSortCol === 'tool' && (provSortDir === 'asc' ? ' ▲' : ' ▼')}
                        </th>
                        <th className="py-2 pr-4 font-semibold cursor-pointer select-none hover:text-text-primary" onClick={() => toggleProvSort('file')}>
                          {t('provenance.file')}{provSortCol === 'file' && (provSortDir === 'asc' ? ' ▲' : ' ▼')}
                        </th>
                        <th className="py-2 pr-4 font-semibold cursor-pointer select-none hover:text-text-primary" onClick={() => toggleProvSort('format')}>
                          {t('provenance.format')}{provSortCol === 'format' && (provSortDir === 'asc' ? ' ▲' : ' ▼')}
                        </th>
                        <th className="py-2 pr-4 font-semibold">{t('provenance.keyProps')}</th>
                        <th className="py-2 pr-4 font-semibold cursor-pointer select-none hover:text-text-primary" onClick={() => toggleProvSort('time')}>
                          {t('provenance.time')}{provSortCol === 'time' && (provSortDir === 'asc' ? ' ▲' : ' ▼')}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedProvRecords.map((rec, i) => (
                        <Fragment key={i}>
                          <tr
                            onClick={() => setProvenanceExpanded(provenanceExpanded === i ? null : i)}
                            className="cursor-pointer border-b border-border/50 hover:bg-bg-tertiary"
                          >
                            <td className="py-2 pr-4 font-mono text-accent">{rec.tool || "—"}</td>
                            <td className="py-2 pr-4 text-text-secondary">{rec.file || rec.path || "—"}</td>
                            <td className="py-2 pr-4 text-text-secondary">{rec.format || "—"}</td>
                            <td className="py-2 pr-4 text-text-secondary">
                              {rec.key_properties
                                ? Object.entries(rec.key_properties).slice(0, 3).map(([k, v]) => `${k}: ${v}`).join(", ")
                                : "—"}
                            </td>
                            <td className="py-2 pr-4 text-text-muted">{rec.timestamp || rec.time ? formatTimeAgo(rec.timestamp || rec.time) : "—"}</td>
                          </tr>
                          {provenanceExpanded === i && (
                            <tr key={`${i}-detail`} className="bg-bg-tertiary">
                              <td colSpan={5} className="py-3 px-4">
                                <pre className="max-h-60 overflow-auto rounded-lg bg-bg-secondary p-3 text-xs">
                                  {JSON.stringify(rec, null, 2)}
                                </pre>
                              </td>
                            </tr>
                          )}
                        </Fragment>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {activeTab === "side" && (
            <div className="flex h-full flex-col">
              <PanelHeader title={t('side.title')}>
                <button onClick={loadSidePending} className="btn-secondary px-3 py-1 text-xs">{t('common.refresh')}</button>
                <button onClick={clearSide} className="btn-secondary px-3 py-1 text-xs">{t('common.clear')}</button>
              </PanelHeader>
              <div className="flex-1 overflow-y-auto p-6">
                <div className="mx-auto max-w-2xl space-y-4">
                  <div className="card space-y-2">
                    <textarea
                      value={sideInput}
                      onChange={(e) => setSideInput(e.target.value)}
                      placeholder={t('side.placeholder')}
                      rows={3}
                      className="input resize-none text-sm"
                    />
                    <button onClick={sendSideQuestion} disabled={!sideInput.trim()} className="btn-primary text-xs">
                      Ask
                    </button>
                  </div>
                  {sideMsg && <div className="text-xs text-error">{sideMsg}</div>}
                  <div className="space-y-2">
                    {sideQuestions.length === 0 ? (
                      <p className="text-sm text-text-muted">{t('side.empty')}</p>
                    ) : sideQuestions.map((q) => (
                      <div key={q.id || q.question} className="card space-y-2">
                        <p className="text-sm font-medium">{q.question}</p>
                        {q.answer && <p className="text-xs text-text-secondary">{q.answer}</p>}
                        {sideAnswerId === (q.id || q.local_id) ? (
                          <div className="flex gap-2">
                            <input
                              type="text"
                              value={sideAnswer}
                              onChange={(e) => setSideAnswer(e.target.value)}
                              placeholder="Type answer…"
                              className="input flex-1 text-xs"
                              onKeyDown={(e) => { if (e.key === "Enter") answerSideQuestion(q.id || q.local_id); }}
                            />
                            <button onClick={() => answerSideQuestion(q.id || q.local_id)} disabled={!sideAnswer.trim()} className="btn-primary text-xs">{t('common.send')}</button>
                            <button onClick={() => setSideAnswerId(null)} className="btn-secondary text-xs">{t('common.cancel')}</button>
                          </div>
                        ) : (
                          <button onClick={() => setSideAnswerId(q.id || q.local_id)} className="btn-secondary text-xs">{t('common.answer')}</button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "solver" && (
            <div className="h-full overflow-y-auto p-6">
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="card">
                  <h2 className="mb-2 text-base font-semibold">Unified Solver</h2>
                  <p className="text-sm text-text-secondary">Derive, solve, and plot physical models.</p>
                </div>
                <div className="card space-y-3">
                  <div className="flex gap-3">
                    <select value={solverModel} onChange={(e) => setSolverModel(e.target.value)} className="input flex-1 text-sm">
                      <option value="">Select model</option>
                      {solverModels.map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                    <button onClick={loadSolverModels} className="btn-secondary text-xs">Refresh</button>
                  </div>
                  <textarea
                    value={solverInput}
                    onChange={(e) => setSolverInput(e.target.value)}
                    placeholder="Model input / parameters (JSON or free text)…"
                    rows={4}
                    className="input resize-none font-mono text-xs"
                  />
                  <div className="flex gap-2">
                    <button onClick={solverDerive} disabled={solverRunning || !solverModel || !solverInput.trim()} className="btn-primary text-xs">
                      {solverRunning ? "…" : "Derive"}
                    </button>
                    <button onClick={solverSolve} disabled={solverRunning || !solverDerived} className="btn-primary text-xs">
                      {solverRunning ? "…" : "Solve"}
                    </button>
                    <button onClick={solverPlot} disabled={solverRunning || !solverSolution} className="btn-primary text-xs">
                      {solverRunning ? "…" : "Plot"}
                    </button>
                  </div>
                  {solverError && <div className="rounded-lg border border-error/20 bg-error/10 px-3 py-2 text-xs text-error">{solverError}</div>}
                </div>
                {solverDerived && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Derived</h3>
                    <pre className="max-h-60 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{solverDerived}</pre>
                  </div>
                )}
                {solverSolution && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Solution</h3>
                    <pre className="max-h-60 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">{solverSolution}</pre>
                  </div>
                )}
                {solverPlotUrl && (
                  <div className="card">
                    <h3 className="text-sm font-semibold mb-2">Plot</h3>
                    {solverPlotUrl.startsWith("data:") || solverPlotUrl.startsWith("http") ? (
                      <img src={solverPlotUrl} alt="Solver plot" className="w-full rounded-lg" />
                    ) : (
                      <pre className="text-xs">{solverPlotUrl}</pre>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Tool palette — command palette style, all tools searchable */}
      {toolPaletteOpen && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-text-primary/40 p-4 pt-[15vh] backdrop-blur-sm"
          onClick={() => { setToolPaletteOpen(false); setToolSearch(""); }}
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
        >
          <div
            ref={toolPaletteRef}
            className="w-full max-w-2xl overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3 border-b border-border px-4 py-3">
              <Search size={18} className="text-text-muted" />
              <input
                type="text"
                value={toolSearch}
                onChange={(e) => setToolSearch(e.target.value)}
                placeholder="Search tools..."
                className="flex-1 bg-transparent text-sm font-medium text-text-primary placeholder:text-text-muted focus:outline-none"
                autoFocus
              />
              <kbd className="rounded border border-border bg-bg-tertiary px-1.5 py-0.5 text-[10px] text-text-muted">ESC</kbd>
              <button
                onClick={() => { setToolPaletteOpen(false); setToolSearch(""); }}
                className="text-text-muted hover:text-text-primary"
              >
                <ChevronDown size={16} />
              </button>
            </div>
            <div className="max-h-[55vh] overflow-y-auto p-3">
              {orderedSidebarGroups.map((group) => {
                const filtered = group.tabs.filter((tab) =>
                  tab.label.toLowerCase().includes(toolSearch.toLowerCase())
                );
                if (filtered.length === 0) return null;
                return (
                  <div key={group.key} className="mb-3">
                    <div className="mb-1.5 px-1 text-[11px] font-bold uppercase tracking-widest text-text-muted">
                      {group.label}
                    </div>
                    <div className="grid grid-cols-4 gap-1.5">
                      {filtered.map((tab) => (
                        <button
                          key={tab.id}
                          draggable
                          onDragStart={() => setDraggedTab(tab.id)}
                          onDragOver={(e) => { e.preventDefault(); setDragOverTab(tab.id); }}
                          onDragLeave={() => setDragOverTab(null)}
                          onDrop={() => handleTabDrop(tab.id)}
                          onDragEnd={() => { setDraggedTab(null); setDragOverTab(null); }}
                          onClick={() => {
                            setActiveTab(tab.id);
                            setToolPaletteOpen(false);
                            setToolSearch("");
                          }}
                          className={`flex flex-col items-center gap-1.5 rounded-lg border p-2.5 text-center transition-all ${
                            dragOverTab === tab.id && draggedTab !== tab.id
                              ? 'border-accent bg-accent/20 ring-2 ring-accent/30'
                              : activeTab === tab.id
                              ? "border-accent bg-accent/10"
                              : "border-border bg-bg-tertiary hover:border-accent/50 hover:bg-accent/5"
                          } ${draggedTab === tab.id ? 'opacity-40' : ''}`}
                        >
                          <span className="text-text-secondary">{tab.icon}</span>
                          <span className="text-[11px] font-medium leading-tight text-text-primary">
                            {toolSearch
                              ? (() => {
                                  const idx = tab.label.toLowerCase().indexOf(toolSearch.toLowerCase());
                                  if (idx === -1) return tab.label;
                                  return (
                                    <>
                                      {tab.label.slice(0, idx)}
                                      <mark className="rounded-sm bg-accent/30 text-accent">{tab.label.slice(idx, idx + toolSearch.length)}</mark>
                                      {tab.label.slice(idx + toolSearch.length)}
                                    </>
                                  );
                                })()
                              : tab.label}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
              {toolSearch && !orderedSidebarGroups.some((g) => g.tabs.some((t) => t.label.toLowerCase().includes(toolSearch.toLowerCase()))) && (
                <div className="py-8 text-center text-sm text-text-muted">No tools match "{toolSearch}"</div>
              )}
            </div>
          </div>
        </div>
      )}

      {showGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-text-primary/40 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Guide" onClick={() => setShowGuide(false)}>
          <div ref={guideModalRef} className="w-full max-w-lg rounded-2xl border border-border bg-bg-secondary p-6 shadow-2xl">
            <h2 className="mb-1 text-xl font-bold">{t('guide.title')}</h2>
            <p className="mb-5 text-sm italic text-text-secondary">
              {t('guide.subtitle')}
            </p>
            <p className="mb-5 text-sm text-text-secondary">
              {t('guide.intro')}
            </p>
            <ol className="mb-6 space-y-3 text-sm text-text-primary">
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  1
                </span>
                <span dangerouslySetInnerHTML={{ __html: t('guide.step1') }} />
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  2
                </span>
                <span dangerouslySetInnerHTML={{ __html: t('guide.step2') }} />
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  3
                </span>
                <span dangerouslySetInnerHTML={{ __html: t('guide.step3') }} />
              </li>
              <li className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-bold text-white">
                  4
                </span>
                <span dangerouslySetInnerHTML={{ __html: t('guide.step4') }} />
              </li>
            </ol>
            <div className="flex justify-end">
              <button onClick={closeGuide} className="btn-primary px-5 py-2">
                {t('guide.gotIt')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Keyboard shortcuts help */}
      {shortcutHelpOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-text-primary/40 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="Keyboard shortcuts"
          onClick={() => setShortcutHelpOpen(false)}
        >
          <div className="w-full max-w-md rounded-2xl border border-border bg-bg-secondary p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="mb-4 text-lg font-bold">Keyboard Shortcuts</h2>
            <div className="space-y-2">
              {[
                { key: "Ctrl+K", desc: "Open tool palette" },
                { key: "Ctrl+F", desc: "Search in chat" },
                { key: "Ctrl+N", desc: "New conversation thread" },
                { key: "Ctrl+B", desc: "Toggle sidebar" },
                { key: "Ctrl+,", desc: "Open settings" },
                { key: "Ctrl+L", desc: "Clear chat messages" },
                { key: "Ctrl+/", desc: "Toggle this help" },
                { key: "Esc", desc: "Close modal / cancel" },
                { key: "Enter", desc: "Send message" },
                { key: "Shift+Enter", desc: "New line in message" },
                { key: "↑ (empty)", desc: "Recall last message" },
              ].map((s) => (
                <div key={s.key} className="flex items-center justify-between gap-4 rounded-lg px-2 py-1 hover:bg-bg-tertiary">
                  <span className="text-sm text-text-secondary">{s.desc}</span>
                  <kbd className="rounded border border-border bg-bg-tertiary px-2 py-0.5 text-xs font-mono text-text-primary">{s.key}</kbd>
                </div>
              ))}
            </div>
            <div className="mt-4 flex justify-end">
              <button onClick={() => setShortcutHelpOpen(false)} className="btn-primary px-4 py-2 text-sm">Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
