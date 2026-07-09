import { useState, useEffect, lazy, Suspense, Fragment } from "react";
import { useTranslation } from "react-i18next";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
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
import { PROVIDERS } from "./lib/constants";
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
import { FilesPanel } from "./components/panels/FilesPanel";
import { ThreadsPanel } from "./components/panels/ThreadsPanel";
import { ReviewPanel } from "./components/panels/ReviewPanel";
import { CoderPanel } from "./components/panels/CoderPanel";
import { BenchmarkPanel } from "./components/panels/BenchmarkPanel";
import { HPCPanel } from "./components/panels/HPCPanel";
import { LogsPanel } from "./components/panels/LogsPanel";
import { TerminalPanel } from "./components/panels/TerminalPanel";
import { PanelHeader } from "./components/settings-shared";
import type { DiffEntry, Checkpoint, ToolInfo, SkillInfo } from "./types/domain";
import {
  MessageSquare, Wrench, Zap, FolderTree, Terminal, Settings,
  Users, Code2, FlaskConical, Brain, BookOpen, GitBranch,
  MessageCircle, Puzzle, FileText, Bird, Briefcase, HelpCircle,
  Dna, Play, Compass, Stethoscope, Monitor, ChevronDown, Sparkles,
  Search,
  Atom, Notebook as NotebookIcon, TerminalSquare, BarChart3, Box, Activity,
  History, Calculator, UserCircle,
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
  if (IS_PET_MODE) {
    return <Pet />;
  }

  const { t } = useTranslation();

  // ── Sidebar state ────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<
    | "chat" | "tools" | "memory" | "skills" | "settings" | "files"
    | "terminal" | "review" | "knowledge" | "logs" | "plugins"
    | "threads" | "project" | "team" | "coder" | "benchmark"
    | "evolution" | "execute" | "workflows" | "explore" | "diagnose"
    | "hpc" | "periodic" | "notebook" | "sandbox" | "sweep"
    | "structure" | "emotion" | "provenance" | "side" | "solver"
    | "persona"
  >("chat");
  const [sidebarGroups, setSidebarGroups] = useState<Record<string, boolean>>({
    core: true,
    research: false,
    workspace: false,
    system: false,
  });
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const toggleSidebarGroup = (group: string) =>
    setSidebarGroups((prev) => ({ ...prev, [group]: !prev[group] }));

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
    teamObjective, teamPlan, teamRunning, teamResult, teamError,
    setTeamObjective,
    handleTeamPlan, handleTeamRun,
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
    memories, memoryStats, memorySearch, memoryFilter, memoryForm, memoryMsg, memoryView,
    setMemorySearch, setMemoryFilter, setMemoryForm, setMemoryView,
    loadMemory, loadMemoryStats, searchMemory, createMemory, deleteMemory,
    promoteMemory, pruneMemory, syncMemoryMd,
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
    kbDocs, kbAvailable, kbMsg, kbQuery, kbChunks, parseLoading,
    fileInputRef, parseFileInputRef,
    setKbQuery,
    loadKnowledge, uploadKnowledge, parseDocument, loadDocumentGraph,
    deleteKnowledge, queryKnowledge,
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
    messages, input, mode, pendingPlan, planLoading,
    chatSearchOpen, chatSearchQuery,
    isStreaming,
    messagesEndRef,
    isConnected, status,
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
    thinkingIntensity, setThinkingIntensity,
    pendingMessages,
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
      key: "research",
      label: t('nav.research'),
      tabs: [
        { id: "knowledge" as const, label: t('tab.knowledge'), icon: <BookOpen size={16} /> },
        { id: "periodic" as const, label: t('tab.periodic'), icon: <Atom size={16} /> },
        { id: "project" as const, label: t('tab.project'), icon: <Briefcase size={16} /> },
        { id: "notebook" as const, label: t('tab.notebook'), icon: <NotebookIcon size={16} />, indented: true },
        { id: "benchmark" as const, label: t('tab.benchmark'), icon: <FlaskConical size={16} />, indented: true },
        { id: "evolution" as const, label: t('tab.evolution'), icon: <Dna size={16} />, indented: true },
        { id: "execute" as const, label: t('tab.execute'), icon: <Play size={16} />, indented: true },
        { id: "workflows" as const, label: t('tab.workflows'), icon: <Zap size={16} />, indented: true },
        { id: "sweep" as const, label: t('tab.sweep'), icon: <BarChart3 size={16} />, indented: true },
        { id: "explore" as const, label: t('tab.explore'), icon: <Compass size={16} />, indented: true },
        { id: "diagnose" as const, label: t('tab.diagnose'), icon: <Stethoscope size={16} />, indented: true },
        { id: "structure" as const, label: t('tab.structure'), icon: <Box size={16} />, indented: true },
        { id: "hpc" as const, label: t('tab.hpc'), icon: <Monitor size={16} />, indented: true },
        { id: "solver" as const, label: t('tab.solver'), icon: <Calculator size={16} />, indented: true },
      ],
    },
    {
      key: "workspace",
      label: t('nav.workspace'),
      tabs: [
        { id: "files" as const, label: t('tab.files'), icon: <FolderTree size={16} /> },
        { id: "terminal" as const, label: t('tab.terminal'), icon: <Terminal size={16} /> },
        { id: "sandbox" as const, label: t('tab.sandbox'), icon: <TerminalSquare size={16} /> },
        { id: "review" as const, label: t('tab.review'), icon: <GitBranch size={16} /> },
        { id: "tools" as const, label: t('tab.tools'), icon: <Wrench size={16} /> },
        { id: "skills" as const, label: t('tab.skills'), icon: <Sparkles size={16} /> },
      ],
    },
    {
      key: "system",
      label: t('nav.system'),
      tabs: [
        { id: "memory" as const, label: t('tab.memory'), icon: <Brain size={16} /> },
        { id: "persona" as const, label: t('tab.persona'), icon: <UserCircle size={16} /> },
        { id: "emotion" as const, label: t('tab.emotion'), icon: <Activity size={16} /> },
        { id: "provenance" as const, label: t('tab.provenance'), icon: <History size={16} /> },
        { id: "plugins" as const, label: t('tab.plugins'), icon: <Puzzle size={16} /> },
        { id: "threads" as const, label: t('tab.threads'), icon: <MessageCircle size={16} /> },
        { id: "logs" as const, label: t('tab.logs'), icon: <FileText size={16} /> },
        { id: "side" as const, label: t('tab.side'), icon: <HelpCircle size={16} /> },
        { id: "settings" as const, label: t('tab.settings'), icon: <Settings size={16} /> },
      ],
    },
  ];

  useEffect(() => {
    const group = sidebarGroupsData.find((g) => g.tabs.some((t) => t.id === activeTab));
    if (group) {
      setSidebarGroups((prev) => {
        if (prev[group.key]) return prev;
        return { ...prev, [group.key]: true };
      });
    }
  }, [activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── useEffect: tab data loading ──────────────────────────────
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

  // ── Derived constants ────────────────────────────────────────
  const providerLabel = PROVIDERS.find((p) => p.id === config.provider)?.label || config.provider;
  const allTabs = sidebarGroupsData.flatMap((g) => g.tabs);
  const activeTabInfo = allTabs.find((t) => t.id === activeTab);

  const handleCoderRun = coder.run;
  const handleExecuteRun = execute.run;
  const handleDiagnoseRun = diagnose.run;
  const handleWorkflowRun = workflow.run;

  const AUTOLOOP_PHASES = ["perceive", "hypothesize", "plan", "execute", "validate", "learn", "report"];

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-primary text-text-primary">
      {/* Sidebar — chat-first: collapsible to maximize chat area */}
      {sidebarHidden && (
        <button
          onClick={() => setSidebarHidden(false)}
          className="z-50 flex h-full w-10 items-center justify-center border-r border-border bg-bg-secondary text-text-muted hover:text-text-primary transition-colors"
          title="Show sidebar"
        >
          <ChevronDown size={16} className="-rotate-90" />
        </button>
      )}
      {!sidebarHidden && (
      <aside className="sidebar-shell flex w-60 flex-col border-r border-border bg-bg-secondary">
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-border">
          <img src="/raven-logo.png" alt="Huginn" className="h-8 w-8 rounded-md object-contain" />
          <div className="flex flex-1 flex-col">
            <div className="text-[15px] font-bold tracking-tight">Huginn</div>
            <div className="text-[12px] text-text-muted leading-none font-medium">{t('app.subtitle')}</div>
          </div>
          <button
            onClick={() => setSidebarHidden(true)}
            className="text-text-muted hover:text-text-primary transition-colors"
            title="Hide sidebar"
          >
            <ChevronDown size={16} className="rotate-90" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2 py-2" aria-label="Main navigation">
          {sidebarGroupsData.map((group, gi) => (
            <div key={group.key} className={gi > 0 ? "mt-2" : ""}>
              <button
                onClick={() => toggleSidebarGroup(group.key)}
                aria-expanded={sidebarGroups[group.key]}
                className="sidebar-group-header flex w-full items-center gap-1.5 px-2 py-1 text-[13px] font-bold uppercase tracking-widest text-text-muted hover:text-text-secondary transition-colors"
              >
                <ChevronDown
                  size={12}
                  className={`transition-transform duration-200 ${
                    sidebarGroups[group.key] ? "rotate-0" : "-rotate-90"
                  }`}
                />
                {group.label}
              </button>

              <div
                role="tablist"
                aria-label={group.label}
                className="sidebar-group-content overflow-hidden transition-all duration-200 ease-in-out"
                style={{
                  maxHeight: sidebarGroups[group.key] ? `${group.tabs.length * 36 + 4}px` : "0px",
                  opacity: sidebarGroups[group.key] ? 1 : 0,
                }}
              >
                {group.tabs.map((tab) => (
                  <button
                    key={tab.id}
                    role="tab"
                    aria-selected={activeTab === tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`sidebar-nav-item flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[15px] font-bold transition-all duration-150 ${
                      tab.indented ? "pl-5" : ""
                    } ${
                      activeTab === tab.id
                        ? "sidebar-nav-active"
                        : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
                    }`}
                  >
                    <span className="flex-shrink-0">{tab.icon}</span>
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {petState && <PetStatusWidget petState={petState} />}
        <div className="border-t border-border px-3 py-3">
          <div className="flex items-center gap-2 text-[13px] text-text-muted">
            <span className={`h-2 w-2 rounded-full ${isConnected ? "bg-success" : "bg-error"}`} />
            <span className="truncate">{status || (isConnected ? t('status.connected') : t('status.offline'))}</span>
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
              onClick={openPetWindow}
              className="sidebar-footer-btn flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1 text-[13px] text-text-muted hover:text-text-secondary"
              title={t('app.summonPet')}
            >
              <Bird size={13} /> {t('app.pet')}
            </button>
          </div>
        </div>
      </aside>
      )}

      {/* Main */}
      <main className="flex flex-1 flex-col min-w-0 bg-bg-primary">
        {/* Header */}
        <header className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-6">
          <div className="flex items-center gap-2.5">
            <span className="text-text-muted">
              {activeTabInfo?.icon}
            </span>
            <span className="text-sm font-semibold">
              {activeTabInfo?.label}
            </span>
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
          <div className="flex items-center gap-3 border-b border-border bg-bg-secondary px-6 py-2">
            <span className="text-xs font-semibold text-text-secondary">{t('app.autoloop')}</span>
            <div className="flex items-center gap-1.5">
              {AUTOLOOP_PHASES.map((phase) => (
                <div
                  key={phase}
                  className={`h-2.5 w-2.5 rounded-full transition-colors ${
                    phase === autoloopPhase
                      ? "bg-accent ring-2 ring-accent/30"
                      : AUTOLOOP_PHASES.indexOf(phase) < AUTOLOOP_PHASES.indexOf(autoloopPhase)
                      ? "bg-accent/40"
                      : "bg-bg-tertiary"
                  }`}
                  title={phase}
                />
              ))}
            </div>
            <span className="text-xs font-medium text-accent">{autoloopPhase}</span>
            <span className="text-xs text-text-muted">{autoloopProgress}%</span>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === "chat" && (
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
              sendMessage={sendMessage}
              pendingPlan={pendingPlan}
              setPendingPlan={setPendingPlan}
              planLoading={planLoading}
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
              researchMode={researchMode}
              setResearchMode={setResearchMode}
              autoloopPhase={autoloopPhase}
              autoloopProgress={autoloopProgress}
            />
          )}

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
              handleTeamPlan={handleTeamPlan}
              handleTeamRun={handleTeamRun}
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

          {activeTab === "files" && (
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

          {activeTab === "knowledge" && (
            <KnowledgePanel
              config={config}
              setConfig={setConfig}
              saveConfig={saveConfig}
              fileInputRef={fileInputRef}
              parseFileInputRef={parseFileInputRef}
              parseLoading={parseLoading}
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
            />
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
            <ErrorBoundary>
              <Suspense fallback={<LoadingFallback />}>
                <EmotionTrackerPanel apiBase={getApiBase()} />
              </Suspense>
            </ErrorBoundary>
          )}

          {activeTab === "persona" && (
            <ErrorBoundary>
              <Suspense fallback={<LoadingFallback />}>
                <PersonaManager />
              </Suspense>
            </ErrorBoundary>
          )}

          {activeTab === "memory" && (
            <MemoryPanel
              memories={memories}
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
              promoteMemory={promoteMemory}
              pruneMemory={pruneMemory}
              syncMemoryMd={syncMemoryMd}
            />
          )}

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

          {activeTab === "settings" && (
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
              personaList={personaList}
              personaEmotion={personaEmotion}
            />
          )}

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
                      Failure rules: {evolve.result.failure_rules?.length} · Success skills: {evolve.result.success_skills?.length} · Prompt patches: {evolve.result.prompt_patches?.length}
                    </div>
                    <div className="text-xs text-text-secondary">Total rules: {evolve.result.total_rules_after} · Total skills: {evolve.result.total_skills_after}</div>
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
              <ErrorBoundary>
                <Suspense fallback={<LoadingFallback />}>
                  <PeriodicTable API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "notebook" && (
            <div className="h-full overflow-hidden p-4">
              <ErrorBoundary>
                <Suspense fallback={<LoadingFallback />}>
                  <Notebook API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "sandbox" && (
            <div className="h-full overflow-hidden p-4">
              <ErrorBoundary>
                <Suspense fallback={<LoadingFallback />}>
                  <SandboxPanel API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "sweep" && (
            <div className="h-full overflow-y-auto p-4">
              <ErrorBoundary>
                <Suspense fallback={<LoadingFallback />}>
                  <SweepDashboard API_BASE={getApiBase()} />
                </Suspense>
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "structure" && (
            <div className="h-full overflow-hidden p-4">
              <ErrorBoundary>
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
                        <th className="py-2 pr-4 font-semibold">{t('provenance.tool')}</th>
                        <th className="py-2 pr-4 font-semibold">{t('provenance.file')}</th>
                        <th className="py-2 pr-4 font-semibold">{t('provenance.format')}</th>
                        <th className="py-2 pr-4 font-semibold">{t('provenance.keyProps')}</th>
                        <th className="py-2 pr-4 font-semibold">{t('provenance.time')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {provenanceRecords.map((rec, i) => (
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
                            <td className="py-2 pr-4 text-text-muted">{rec.timestamp || rec.time || "—"}</td>
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

      {showGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-text-primary/40 p-4 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl border border-border bg-bg-secondary p-6 shadow-2xl">
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
    </div>
  );
}
