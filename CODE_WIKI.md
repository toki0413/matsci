# Huginn — Code Wiki

> 一份面向开发者的结构化代码文档，覆盖项目整体架构、模块职责、关键类与函数、依赖关系及运行方式。
> 仓库根目录：`c:\Users\wanzh\Desktop\matsci-agent`

---

## 目录

1. [项目概览](#1-项目概览)
2. [整体架构](#2-整体架构)
3. [仓库目录结构](#3-仓库目录结构)
4. [核心模块职责](#4-核心模块职责)
5. [关键类与函数说明](#5-关键类与函数说明)
6. [依赖关系](#6-依赖关系)
7. [项目运行方式](#7-项目运行方式)
8. [数据流与调用拓扑](#8-数据流与调用拓扑)
9. [设计原则与约定](#9-设计原则与约定)

---

## 1. 项目概览

**Huginn** 是一个面向**计算材料科学**的智能体（Agent）系统，由 LLM 驱动，自动完成 DFT 计算、分子动力学、符号回归、自主探索等任务，并通过 **Lean 4** 对张量代数、有限元、数值线性代数、DFT 理论、热力学、概率等数学结构进行**形式化验证**。

- **语言构成**：Python（核心 Agent，约 13 万行）、Rust（CLI 前端 / Tauri 外壳 / PyO3 加速扩展 / Sidecar 进程管家）、Lean 4（形式化数学库）、TypeScript + React（桌面端 UI）。
- **License**：MIT
- **定位**：科研伴侣而非裁判，强调第一性原理、可解释性与数学正确性。

### 核心能力速览

| 能力域 | 说明 |
|--------|------|
| 多 Provider LLM | OpenAI / Anthropic / DeepSeek / Google / OpenRouter / NVIDIA / Ollama / vLLM / 本地 OpenAI 兼容端点 |
| 仿真工具链 | VASP / Quantum ESPRESSO / CP2K / Gaussian / ORCA（DFT/QC）；LAMMPS / GROMACS / OpenMM（MD）；Abaqus / COMSOL / OpenFOAM / FEniCS / Elmer（FEA/CFD） |
| 数学形式化 | SymPy 表达式自动翻译为 Lean 4，`lake build` 验证，6 阶段已形式化 |
| 桌面应用 | Tauri v2 + React 18，21 个功能面板，3D 结构查看器，WebSocket 流式聊天 |
| 自主科研 | Autoloop 七阶段闭环（perceive→hypothesize→plan→execute→validate→learn→report） |
| 记忆与知识 | 三层记忆 + ChromaDB 向量检索 + NetworkX 知识图谱 + RAG 反馈闭环 |
| 多智能体 | Orchestrator / Swarm / ModelTeam / SubagentDispatch，配套循环检测与工具预算 |

---

## 2. 整体架构

### 2.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户接入层                                │
│   Rust CLI (cli/)   │  Tauri 桌面应用 (desktop/)  │  HTTP/WS API  │
└──────────┬──────────┴──────────────┬──────────────┴──────┬───────┘
           │ 子进程委派 / HTTP / WS  │ Tauri Command      │
           └─────────────┬───────────┴────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python Agent 核心 (agent/huginn/)             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              HuginnAgent (Mixin 组合 + LangGraph)         │  │
│  │  ContextMixin │ StreamingMixin │ SessionMixin │           │  │
│  │  CallbackMixin │ ReflectionMixin                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐  │
│  │  Memory    │ │  Skills    │ │  Agents  │ │  Autoloop    │  │
│  │  (3-tier)  │ │ (声明式)   │ │ (编排)   │ │  (科研闭环)  │  │
│  └────────────┘ └────────────┘ └──────────┘ └──────────────┘  │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐  │
│  │  Tools     │ │   RAG      │ │   KG     │ │  Workflows   │  │
│  │  (130+)    │ │ (加密检索) │ │ (图谱)   │ │  (模板引擎)  │  │
│  └────────────┘ └────────────┘ └──────────┘ └──────────────┘  │
└────────┬───────────────────────────────────────────┬───────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────┐              ┌───────────────────────────┐
│  Lean 4 形式化库     │              │  基础设施层                 │
│  (agent/lean/)      │              │  Crypto / HPC / MCP / DB   │
│  12 模块 + SymPy桥   │              │  SQLite+FTS5 / ChromaDB    │
└─────────────────────┘              └───────────────────────────┘
```

### 2.2 五条核心设计决策（来自 `.omm/overall-architecture/context.md`）

1. **Mixin 组合而非继承** — `HuginnAgent` 由 5 个 Mixin 组合扩展，`core.py` 仅负责装配，避免 god-class。
2. **ContextBuilder 抽离** — 上下文构建（记忆召回、KG/KB 查询、情绪追踪）独立可测。
3. **双引擎架构** — `AutoloopEngine`（自主科研全闭环）与 `DeliAutoResearch`（学术写作半闭环）并行，`ResearchWorkflow` 桥接 WebSocket。
4. **AgentConfig 统一配置** — 8 个子配置 dataclass，`from_env()` / `from_config()` 加载。
5. **ServerContext 单例容器** — FastAPI 长生命周期对象集中管理，避免散落的全局变量。

### 2.3 工具三层过滤

工具对 LLM 的可见性经过三层过滤：
1. **模式过滤** — chat 模式隐藏重型仿真工具
2. **阶段过滤** — 当前研究阶段的 `tool_filter` 限制可用工具集
3. **查询相关过滤** — 工具数 > 25 时启用 query-aware 检索，只保留 top-15

---

## 3. 仓库目录结构

```
matsci-agent/
├── agent/                      # Python 核心（主仓库）
│   ├── huginn/                 # 核心 Python 包
│   │   ├── agent/              # HuginnAgent + Mixins (context/streaming/session/callbacks/reflection)
│   │   ├── agents/             # 多智能体编排 (factory/orchestrator/swarm/team/subagent)
│   │   ├── autoloop/           # 自主科研闭环 (engine/budget/conjecture/phase_gate/red_team)
│   │   ├── cli/                # Click CLI 入口 + commands/
│   │   ├── kg/                 # 知识图谱 (NetworkX)
│   │   ├── lean/               # Lean 4 Python 接口 (interface/pipeline/sympy_to_lean)
│   │   ├── memory/             # 三层记忆 (manager/session/longterm/decay)
│   │   ├── models/             # LLM 路由 (router/registry)
│   │   ├── rag/                # 检索增强 (vector_store/encrypted_rag/feedback)
│   │   ├── routes/             # FastAPI 路由 (~50 个文件)
│   │   ├── skills/             # 声明式技能 (base/presets/evolution)
│   │   ├── tools/              # 130+ 工具 (base/registry/adapter + sim/sci/fem/neb 子包)
│   │   ├── workflows/          # 计算工作流引擎 (engine/templates/checkpoint)
│   │   ├── agent_config.py     # AgentConfig 8 子配置
│   │   ├── config.py           # HuginnConfig 配置管理
│   │   ├── server.py / server_core.py / server_context.py  # FastAPI 应用
│   │   ├── llm.py / llm_retry.py  # LLM 工厂与重试
│   │   ├── mcp_client.py       # MCP 客户端
│   │   └── types.py            # 核心类型 (ToolResult/PermissionMode/...)
│   ├── lean/                   # Lean 4 形式化库源码
│   │   ├── HuginnLean/         # 12 个 Lean 模块
│   │   └── project/            # lake 项目 (lakefile.toml)
│   ├── tests/                  # pytest 测试套件 (192+ 测试)
│   └── docs/                   # 架构文档
├── cli/                        # Rust CLI 前端 (huginn 二进制)
├── desktop/                    # Tauri v2 + React 桌面应用
│   ├── src-tauri/              # Rust 外壳 (后端进程管理/系统托盘/终端)
│   └── src/                    # React 前端 (21 面板 + 3D 查看器)
├── sidecar/                    # Rust Sidecar (独立进程管家 + 日志广播)
├── pyext/                      # PyO3 性能扩展 (huginn_ext 模块)
├── servers/                    # MCP 服务器
│   ├── mat-db-mcp/             # 材料数据库查询 (Materials Project / NIST)
│   └── math-anything-mcp/      # 数学语义提取与比较
├── skills/                     # 共享技能定义
├── Cargo.toml                  # Rust workspace (pyext/cli/desktop/src-tauri/sidecar)
└── docker-compose.yml          # 单机编排
```

---

## 4. 核心模块职责

### 4.1 Python Agent 核心 (`agent/huginn/`)

#### 4.1.1 HuginnAgent — `agent/core.py`

Agent 的主类，通过 Mixin 组合而非继承扩展。所有上下文构建、流式输出、会话管理、回调、反思逻辑分布在 5 个 Mixin 中，`core.py` 仅负责装配与图构建。

```python
class HuginnAgent(ContextMixin, CallbackMixin, SessionMixin, ReflectionMixin, StreamingMixin):
    """Material Science specialized Agent."""
```

**职责**：状态管理、记忆集成、技能执行、探索引擎集成、工具调用循环（基于 LangGraph / deepagents）。

#### 4.1.2 工具系统 — `huginn/tools/`

| 文件 | 职责 |
|------|------|
| `base.py` | `HuginnTool` 抽象基类，定义 schema/权限/校验/执行/结果映射契约 |
| `registry.py` | `ToolRegistry` 类级单例注册表，管理 130+ 工具 |
| `adapter.py` | `ToolAdapter` 把 HuginnTool 包成 LangChain `StructuredTool`，含 6 道闸门 pre-check |
| `profile.py` | `ToolProfile` / `CostTier` 声明式调度元数据 |
| `defaults.py` | `ToolMetadata` fail-closed 默认值（未声明只读一律需确认） |
| `__init__.py` | `register_all_tools()` / `register_core_tools()` / `register_optional_tools()` |

**工具分组**（130+ 工具按域分类）：

- **DFT / 量子化学**：`vasp_tool` / `qe_tool` / `cp2k_tool` / `gaussian_tool` / `orca_tool`
- **分子动力学**：`lammps_tool` / `gromacs_tool` / `openmm_tool` / `packing_tool`
- **FEA / CFD**：`abaqus_tool` / `comsol_tool` / `openfoam_tool` / `fenics_tool` / `elmer_tool` / `fem_tool`
- **化学 / 自由能**：`rdkit_tool` / `vina_tool` / `fep_tool` / `msm_tool` / `thermo_tool`
- **数学 / 符号**：`bourbaki_tool` / `lean_tool` / `numerical_tool` / `autodiff_tool` / `gp_tool` / `uq_tool` / `tda_tool` / `symbolic_math_tool`
- **结构 / 表征**：`structure_tool` / `symmetry_tool` / `xrd_sim_tool` / `descriptor_tool` / `ml_potential_tool`
- **代码 / 开发**：`code_tool` / `bash_tool` / `file_read_tool` / `file_edit_tool` / `git_tool` / `github_tool` / `grep_tool` / `glob_tool`
- **可视化 / 报告**：`visualize_tool` / `report_tool` / `image_analysis_tool` / `image_design_tool`
- **Agent / Meta**：`orchestrate_tool` / `subagent_tool` / `skill_tool` / `memory_tool` / `phase_tool` / `workflow_tool` / `clarification_tool`

> 注：顶层 `*_tool.py` 多为单行 re-export shim，真实实现在 `tools/sim/`、`tools/sci/`、`tools/fem/`、`tools/neb/`、`tools/symbolic_math/` 等子包中，避免单文件膨胀。

#### 4.1.3 记忆系统 — `huginn/memory/`

三层记忆架构（仿 Claude Code memdir）：

| 层级 | 类 | 后端 | TTL |
|------|----|----|-----|
| Session | `SessionContext` | 内存 | 单轮会话 |
| Long-term | `LongTermMemory` | SQLite + FTS5 + ChromaDB | short=6h / mid=7d / long=永久 |
| Manager | `MemoryManager` | 编排层 | — |

**关键特性**：
- 自动消息压缩（>100 条触发 summarization）
- 重要性评分（0-1）与衰减晋升（`MemoryDecayPolicy`）
- LLM Wiki Lint 抓矛盾/孤儿/陈旧条目
- `MemoryMaintainer` 后台线程每小时自动剪枝去重

#### 4.1.4 Skills 声明式技能 — `huginn/skills/`

| 文件 | 职责 |
|------|------|
| `base.py` | `SkillDefinition` / `SkillStep` / `DeclarativeSkillExecutor`（支持 condition guard / loop / safe_eval 验证） |
| `presets.py` | 40+ 预设技能（standard_dft / aimd / phonon / elastic / 7 种 Lean 验证 / UQ / GP / ...） |
| `registry.py` | `SkillRegistry` 注册表 |
| `composite.py` | 6 个组合技能（band_structure / mechanical / md_pipeline / ...） |
| `evolution.py` | `SkillEvolutionLayer` — Beta 分布 + ANCCR 时间加权记录工具参数信念，UCB 探索推荐 |

#### 4.1.5 Autoloop 自主科研闭环 — `huginn/autoloop/`

七阶段闭环：`perceive → hypothesize → plan → execute → validate → learn → report`

| 文件 | 职责 |
|------|------|
| `engine.py` | `AutoloopEngine`（193KB）— 集成 Memory / KG / Workflow / Coder / PhaseGate / GoalScheduler |
| `budget.py` | `ProgressiveBudget` — 三档渐进预算（open→medium→light）限制后期迭代 |
| `conjecture.py` | `ConjectureGenerator` — Moonshine 三步法跨域猜想（extract→transfer→generate） |
| `goal_store.py` | `GoalStore` — 目标持久化，known/unknown/blind_spot 三类未知跟踪 |
| `phase_gate.py` | `PhaseGateHook` + `DempsterShaferCombiner` + `MathEvidenceChecker` 阶段门 |
| `plan_store.py` | `PlanStore` — 计划持久化（draft→confirmed→executing→completed） |
| `red_team.py` | `RedTeamReviewer` — 对抗审查（hidden_assumption / confounder / methodology_gap） |

#### 4.1.6 多智能体编排 — `huginn/agents/`

| 文件 | 职责 |
|------|------|
| `factory.py` | `AgentFactory` — config 驱动 profile 创建，注入 PRT 0/1 异常检测钩子 |
| `orchestrator.py` | `Orchestrator` — plan→execute→synthesize 三段式编排 |
| `swarm.py` | `HuginnSwarm` — 单模型多角色（PLANNER/SCIENTIST/CODER/CRITIC/EXECUTOR） |
| `team.py` | `ModelTeam` — 真·多模型 Fusion（OpenRouter Fusion 模式 fan-out/in） |
| `subagent.py` | `SubagentDispatch` — 隔离派发子 agent（explore/coder/analyst） |
| `speculator.py` | `IntentSpeculator` — 意图投机，预热安全工具缓存 |
| `loop_detector.py` | `LoopDetector` + `ThoughtLoopDetector` — 工具调用与思维死循环检测 |
| `tool_budget.py` | `ToolCallBudget` — 工具调用预算（max_calls=15 / max_per_tool=5） |

#### 4.1.7 服务器与路由 — `huginn/server*.py` + `routes/`

- `server.py` — FastAPI 应用入口（薄壳），挂中间件栈：RequestID → Maintenance → ErrorNormalize → Timeout(180s) → SizeLimit(10MB) → Prometheus → RateLimit(120/min) → CORS
- `lifespan.py` — 应用生命周期：注册工具、连 MCP、初始化 KB、启动后台监控
- `server_context.py` — `ServerContext` dataclass 单例容器
- `server_core.py` — 惰性工厂与线程 TTL 管理（24h 空闲清理）

`routes/` 下约 50 个路由模块按功能分组：

| 分组 | 模块 |
|------|------|
| 鉴权 / 用户 | `auth.py` / `users.py` / `credentials.py` |
| Agent 执行 | `agents.py` / `execution.py` / `coder.py` / `planner.py` / `autoloop.py` / `unified.py` / `workflows.py` / `skills.py` / `team.py` |
| 工具 / MCP | `tools.py` / `mcp.py` |
| 记忆 / 会话 | `memory.py` / `threads.py` / `checkpoints.py` / `interaction.py` / `side.py` |
| 知识 / RAG | `knowledge.py` / `codebase.py` / `data_dict.py` / `document.py` |
| HPC / 远程 | `hpc.py` / `tunnels.py` / `transfer.py` / `terminal.py` / `kernel.py` |
| 实时 / WS / SSE | `ws.py` / `event_stream.py` / `events.py` / `viewer3d.py` / `live_script.py` |
| Benchmark / 评测 | `bench.py` / `eval.py` / `parameters.py` / `advisor.py` |
| Admin / 诊断 | `admin.py` / `diagnostics.py` / `health.py` / `system.py` / `config.py` / `metrics.py` |

#### 4.1.8 LLM 多 Provider 路由 — `huginn/models/` + `llm.py` + `llm_retry.py`

- `models/registry.py` — `ModelRegistry` + `create_langchain_model()`：24 种 provider、API key 轮换、模型能力表 `ModelCaps`、推理强度（thinking）控制
- `models/router.py` — `ModelRouter`：按任务类型（default/agent/coding/science/reasoning/verification/...）路由到最便宜/最强模型
- `llm.py` — `get_model()` 便捷封装
- `llm_retry.py` — `with_retry()` / `call_with_fallback()` / `persistent_retry()`：429 限流 / 529 过载 / 上下文溢出 / 5xx 网络错误的分级重试与模型降级

#### 4.1.9 MCP 客户端 — `huginn/mcp_client.py`

`MCPClientManager` 管理多个 MCP server 连接（stdio + SSE），动态发现工具并路由调用，含健康监控、session expired 重连、memoized 连接缓存。

### 4.2 Lean 4 形式化库 (`agent/lean/`)

**目的**：把材料科学的数学结构用 Lean 4 形式化，SymPy 推导结果自动翻译为 Lean 定理并用 `lake build` 验证。

| 模块 | 内容 |
|------|------|
| `TensorAlgebra.lean` | 指标记号、缩并、度量变换 |
| `FiniteElement.lean` | 弱形式：线弹性、热传导、单元装配 |
| `NumericalLinearAlgebra.lean` | LU / Cholesky / Jacobi / CG / 误差矩阵 |
| `DFT.lean` | 自由电子气、紧束缚、LDA 交换关联 |
| `Thermodynamics.lean` | 状态方程、自由能、Clausius-Clapeyron、配分函数 |
| `Probability.lean` | 正态分布、GP 核、MC 采样、贝叶斯更新 |
| `ContinuumMechanics.lean` | 连续介质力学 |
| `Elasticity.lean` | 弹性理论 |
| `BornStability.lean` | Born 稳定性准则 |

**Python 桥接**（`huginn/lean/`）：
- `sympy_to_lean.py` — `SymPyToLean` SymPy AST → Lean 4 字符串翻译器
- `interface.py` — `LeanInterface` 封装 `lake build` / `lake env lean --run`
- `auto_pipeline.py` — `AutoLeanPipeline` 15+ 个 `verify_*` 方法 + `LeanCodeFixer` 规则修复 + reflection loop

### 4.3 Rust 原生侧

#### 4.3.1 CLI 前端 (`cli/`)

**目的**：`huginn` 二进制命令行入口。Rust 自身不实现业务逻辑，仅解析参数 + 加载配置，把命令委派给 Python 后端（`python -m huginn.cli <subcommand>`）。仅 `Version` / `Tools` / `Configure` 三个命令在 Rust 侧直接执行。

**关键文件**：
- `cli/src/main.rs` — `struct Cli`（clap derive）+ `enum Commands`（13 个子命令）
- `cli/src/config.rs` — `HuginnConfig`（21 字段，TOML/JSON 序列化）
- `cli/src/python.rs` — `find_python()` 解释器发现 + `run_python_cli()` 子进程委派

**子命令**：`Chat` / `Explore` / `Coder` / `Serve` / `Tools` / `Version` / `Configure` / `Bench` / `Evolve` / `Execute` / `Workflow` / `Diagnose` / `Hpc` / `EncryptConfig`

#### 4.3.2 Tauri 桌面应用 (`desktop/`)

**Rust 外壳**（`src-tauri/`）：
- `main.rs` — `AppState`（backend child / terminal / port）+ Tauri Commands（`start_backend` / `stop_backend` / `read_file` / `write_file` / `write_terminal` 等）+ 系统托盘（Show/Hide/Restart/Quit）+ 日志桥接
- `tauri.conf.json` — `productName: "Huginn"`，`identifier: com.huginn-agent.app`，CSP 强制 127.0.0.1

**React 前端**（`src/`）：
- 技术栈：React 18.3 + TypeScript 5.5 + Vite 5.4 + Tailwind CSS 3.4 + i18next（中英双语）+ react-virtuoso + recharts + three.js（3D 结构）+ react-markdown
- 21 个面板：`ChatPanel` / `CoderPanel` / `HPCPanel` / `KnowledgePanel` / `MemoryPanel` / `ToolsPanel` / `TeamPanel` / `TerminalPanel` / `ThreadsPanel` / `SettingsPanel` / ...
- 独立组件：`StructureViewer`（Three.js）/ `PeriodicTable` / `PersonaManager` / `EmotionTracker` / `DiffViewer` / `AutoloopProgress`
- 通信三通道：Tauri Commands（同步控制）+ HTTP REST（请求响应）+ WebSocket（流式聊天，双速渲染 `reasoning_delta` / `text_delta`）

#### 4.3.3 Sidecar 进程管家 (`sidecar/`)

独立运行的 Rust 进程管理器，启动/停止 Python 后端、聚合 stdout/stderr、通过 HTTP API 和 WebSocket 广播日志事件。可独立于 Tauri 运行。

- HTTP 路由：`GET /health` / `POST /start` / `POST /stop` / `GET /status` / `GET /ws`
- 默认端口 8001（sidecar）/ 8000（后端）
- 进程清理：Unix `SIGINT→SIGTERM→SIGKILL`，Windows `terminate()→kill()`

#### 4.3.4 PyO3 性能扩展 (`pyext/`)

`huginn_ext` Python 模块，加速热点路径：

| 函数 | 用途 |
|------|------|
| `parse_outcar(path)` | VASP OUTCAR 单遍流式解析 |
| `compute_msd(positions, timesteps)` | 均方位移 |
| `compute_rdf(positions, box_dims, bins)` | 径向分布函数（最小镜像约定） |
| `lammps.parse_lammps_dump(path, ...)` | LAMMPS dump 解析（可选内联 MSD/RDF） |
| `files.tail_lines(path, n)` | 反向分块读取最后 N 行 |
| `sandbox.run_sandboxed(command, ...)` | 沙箱子进程执行（白名单 + 禁止字符） |
| `vectors.top_k(query, matrix, k)` | 余弦相似度 top-k（rayon 并行） |

### 4.4 MCP 服务器 (`servers/`)

| 服务器 | 工具数 | 用途 |
|--------|--------|------|
| `mat-db-mcp` | 5 | Materials Project / NIST 势函数库查询（有 `MP_API_KEY` 走真实 API，否则 mock） |
| `math-anything-mcp` | 5 | 数学语义层：`extract_math` / `math_diff` / `dimensional_analysis` / `track_precision` / `normalize_expression` |

两者均采用 stdio 传输，作为独立进程被 `mcp_client` 接入。

---

## 5. 关键类与函数说明

### 5.1 核心类型 — `huginn/types.py`

```python
class PermissionMode(Enum):      # AUTO / ASK / DENY / PLAN
class BudgetDecision(Enum):      # ALLOW / WARN / DENY

@dataclass
class ToolResult:                # 工具执行结果，CLI-Anything JSON 契约
    data: Any
    success: bool = True
    error: str | None = None
    new_messages: list[dict]
    side_effects: list[str]
    metadata: dict[str, Any]

@dataclass
class PermissionResult:          # mode + reason
@dataclass
class ValidationResult:          # result + errors
@dataclass
class ToolContext:               # 工具执行上下文
```

### 5.2 HuginnTool 基类 — `huginn/tools/base.py`

```python
class HuginnTool(ABC, Generic[InputT, OutputT]):
    name: str = ""
    description: str = ""
    category: str = "misc"          # core/search/meta/sim/sci/design/cv/materials/misc
    destructive: bool = False
    read_only: bool = False
    input_schema: type[InputT] | None = None    # Pydantic v2
    output_schema: type[OutputT] | None = None
    profile: ToolProfile | None = None           # 调度元数据
    active: bool = True                           # False 时对 LLM 不可见但可直调
    is_background_task: bool = False

    def call(self, args, context) -> ToolResult   # 模板方法，自动抓 provenance
    def _execute(self, args, context)             # 子类 override
    def check_permissions(self, args) -> PermissionResult
    def validate_input(self, args) -> ValidationResult
    def format_result(self, result) -> dict
```

### 5.3 ToolRegistry — `huginn/tools/registry.py`

类级单例注册表，classmethod 管理：
- `register(tool)` / `unregister(name)` / `get(name)` / `list_tools()`
- `get_all_schemas()` — OpenAI function-calling 格式，跳过 `active=False`，fail-closed 标注 `requires_confirmation`
- `get_schemas_for_provider(provider)` — 适配 openai/anthropic/google 格式
- `get_assembled_schemas(permission_rules, mcp_tools)` — 单一装配点（合并内置+MCP → deny 过滤 → 排序去重）

### 5.4 ToolAdapter — `huginn/tools/adapter.py`

把 HuginnTool 包成 LangChain `StructuredTool`，`adapt()` 主流程：

1. `_check_permission()` — 边界态拦截 → 危险 pattern → AUTO/DENY 决策 → `approval_callback`
2. `_run_pre_checks()` — 6 道闸门：permission → cache → router → budget → loop_detector → circuit_breaker（熔断自动降级链）
3. `_needs_confirmation()` + `_ask_confirmation()` — 破坏性/高成本走 `ClarificationManager.ask`（120s 超时，fail-closed）
4. `validate_input()` → 分级超时 → `tool.call()`
5. `_check_constraints(result)` — `constraint_scope` 域约束校验
6. `_serialize(result)` — 隐私脱敏 + 压缩 + 超 20K token 卸盘 + 自动出图给 VLM
7. `_run_post_checks()` — provenance 注册 / SkillEvolution 贝叶斯反馈 / audit / 事件总线

### 5.5 HuginnAgent — `huginn/agent/core.py`

通过 5 个 Mixin 组合：
- `ContextMixin` — 上下文构建（委托 `ContextBuilder`）
- `StreamingMixin` — 流式输出
- `SessionMixin` — 会话管理
- `CallbackMixin` — 回调钩子
- `ReflectionMixin` — 反思

### 5.6 AgentConfig — `huginn/agent_config.py`

8 个子配置 dataclass：
- `AgentCoreConfig` — profile_id / thread_id / enable_exploration / agent_factory
- `AgentModelConfig` — model / model_router / system_prompt / prompt_cache_control
- `AgentToolConfig` — tools / tool_filter / max_tool_calls / compression_max_tokens
- `AgentMemoryConfig` — memory_manager / checkpointer / decay 配置
- `AgentSecurityConfig` / `AgentTelemetryConfig` / `AgentContextBudgetConfig`
- `AgentKnowledgeGraphConfig` / `AgentPersonalizationConfig`

### 5.7 ModelRegistry / ModelRouter — `huginn/models/`

- `ModelRegistry` — alias → `ModelConfig` 映射 + LRU 缓存（max=32），`get(alias)` / `resolve(ref)` / `from_config(config)`
- `create_langchain_model(provider, model_name, ...)` — 24 种 provider 工厂
- `ModelRouter` — 按任务类型路由（`select(task, prefer_cheap=False)`），Moonshine 三槽：`verification` / `archival` / 默认
- `pick_api_key(provider)` — 进程内 round-robin key 轮换

### 5.8 MemoryManager — `huginn/memory/manager.py`

编排 SessionContext + LongTermMemory：
- `add_message()` / `add_tool_call()`（自动晋升 `*_tool` 成功结果）
- `remember()` / `recall()`（含 `material_filter`）/ `recall_for_prompt()`
- `promote_tool_result()` / `promote_session_summary()`
- `_run_distillation()` — 调 `KnowledgeDistiller` 自动入 KB
- `maintenance()` — 衰减 + 剪枝 + 去重
- `sync_memory_md()` — 长记忆写 `MEMORY.md`

### 5.9 AutoloopEngine — `huginn/autoloop/engine.py`

七阶段闭环主引擎，集成：
- MemoryManager / ProjectKnowledgeGraph / HypothesisGraph / WorkflowEngine / CoderRunner
- ExplorationOrchestrator / PhaseGateHook（含 RedTeamReviewer + MathEvidenceChecker）
- GoalScheduler / ProgressTracker
- JEPA 式预测（plan 预期 vs validate 实际算 surprise，连续低 surprise 提前终止）
- 连续验证失败计数 + refine 循环计数（max 8）防死循环

### 5.10 WorkflowEngine — `huginn/workflows/engine.py`

```python
class WorkflowEngine:
    def execute(stages: list[ComputationalStage], context, checkpoint_path, budget_policy) -> WorkflowResult
    def resume(checkpoint_path) -> WorkflowResult
    def _diagnose_and_fix(stage, error)  # 调 diagnose_tool + rag_tool 按软件规则修参数
    def _validate(stage, result)         # convergence / energy_sign / force_threshold / custom
```

14 个预设模板（`templates.py`）：`standard_dft_workflow` / `aimd_workflow` / `defect_workflow` / `surface_workflow` / `ml_potential_workflow` / 7 种 `*_verify_workflow`（SymPy→Lean）/ `reviewer_workflow` / `plasma_simulation_workflow` / `reaction_pathway_workflow`。

### 5.11 MCPClientManager — `huginn/mcp_client.py`

```python
class MCPClientManager:
    async def connect(config: MCPServerConfig)         # stdio 或 SSE
    async def disconnect(name)
    async def call_tool(name, arguments) -> dict
    async def call_tool_with_retry(name, args, max_errors=3)  # session expired 自动重连
    def get_server_status() -> dict
    async def connect_batch(configs, concurrency=4)    # 并发连接
```

---

## 6. 依赖关系

### 6.1 Python 核心依赖（`agent/pyproject.toml`）

| 依赖 | 用途 |
|------|------|
| `pydantic>=2.0` | 所有工具 schema 与配置 dataclass |
| `langchain>=0.3.0` + `langchain-core` + `langchain-openai` | LLM 抽象层 |
| `langgraph>=0.2.0` | Agent 推理图 |
| `deepagents>=0.5.0` | Agent 图构建（优先于 langgraph prebuilt） |
| `click>=8.0` | CLI 框架 |
| `rich>=13.0` | 终端美化 |
| `fastapi>=0.100` + `uvicorn>=0.27` + `sse-starlette` | HTTP/SSE 服务 |
| `websockets>=12.0` | WebSocket 实时通信 |
| `networkx>=3.0` | 知识图谱后端 |
| `numpy>=1.24` + `scipy>=1.10` | 数值计算 |
| `cryptography>=42.0` | 加密（Fernet / AES-128-CBC + HMAC） |
| `mcp>=1.28` | MCP 协议客户端 |
| `tenacity>=8.0` | 重试装饰器 |
| `aiohttp>=3.9` + `httpx>=0.25` | 异步 HTTP |

### 6.2 可选依赖（`[project.optional-dependencies]`）

- `all` — `pymatgen` / `ase` / `dscribe` / `paramiko` / `chromadb` / `sentence-transformers` / `pymupdf` / `easyocr` / `mace-torch` / `fairchem-core` / `py4vasp` / `chgnet` / `matplotlib`
- `rag` — `chromadb` / `sentence-transformers` / `pymupdf` / `easyocr`
- `dev` — `pytest` / `pytest-asyncio` / `pytest-benchmark` / `hypothesis` / `black` / `ruff` / `mypy` / `pre-commit`

### 6.3 Rust Workspace（根 `Cargo.toml`）

```toml
[workspace]
members = ["pyext", "cli", "desktop/src-tauri", "sidecar"]
```

| Crate | 关键依赖 |
|-------|---------|
| `cli` | `clap 4.5` / `serde` + `toml` / `dotenvy` / `dialoguer` / `process-wrap` |
| `desktop/src-tauri` | `tauri 2` / `tauri-plugin-shell/notification/dialog` / `reqwest 0.12` |
| `sidecar` | `tokio 1` / `axum 0.7` (ws) / `reqwest 0.12` / `process-wrap` |
| `pyext` | `pyo3 0.23` / `numpy 0.23` / `rayon 1.10` / `wait-timeout 0.2` / `which 7.0` |

### 6.4 桌面前端依赖（`desktop/package.json`）

- React 18.3 + TypeScript 5.5 + Vite 5.4
- Tailwind CSS 3.4 + `lucide-react`
- `i18next` + `react-i18next`（中英双语）
- `react-virtuoso`（虚拟列表）/ `recharts`（图表）
- `three` + `@react-three/fiber` + `@react-three/drei`（3D）
- `react-markdown` + `rehype-highlight` + `remark-gfm`
- `@tauri-apps/api 2.0` + plugin-dialog/notification/shell
- `@playwright/test` + `@axe-core/playwright`（e2e）

### 6.5 模块间依赖拓扑

```
HuginnAgent ──► ToolAdapter ──► ToolRegistry ──► 130+ HuginnTool
     │                                              │
     ├──► MemoryManager ──► SessionContext         │
     │                  └─► LongTermMemory ──► ChromaDB
     ├──► ContextBuilder ──► KG / KB / RAG ────────┘
     ├──► ModelRouter ──► ModelRegistry ──► LangChain Model
     ├──► SkillExecutor ──► SkillRegistry
     └──► AutoloopEngine ──► WorkflowEngine ──► ToolRegistry
                       ├─► ConjectureGenerator ──► KG
                       ├─► PhaseGateHook ──► RedTeamReviewer + MathEvidenceChecker
                       └─► GoalStore / PlanStore
```

---

## 7. 项目运行方式

### 7.1 安装

#### Python 后端（必需）

```bash
cd matsci-agent/agent
pip install -e .
pip install toml  # 可选：配置文件支持
```

#### Rust CLI 前端（推荐）

```bash
cd matsci-agent/cli
cargo build --release
# 二进制位于 cli/target/release/huginn (或 .exe)
```

> Windows + GNU 工具链需确保 MinGW 的 `dlltool.exe` 在 PATH：
> `$env:PATH = "C:\mingw64\mingw64\bin;$env:PATH"`

#### Rust 性能加速扩展（可选）

```bash
cd matsci-agent/pyext
maturin build --release
pip install target/wheels/huginn_ext-*.whl
```

加速 LAMMPS/VASP 解析与 MSD/RDF 计算；未安装时自动回退纯 Python。

#### 桌面应用

```bash
cd matsci-agent/desktop
npm install
npm run dev      # 开发模式（Vite dev server + Tauri）
npm run build    # 生产构建
```

### 7.2 配置 LLM Provider

**方式 A — 环境变量**（`.env`）：

```bash
HUGINN_PROVIDER=anthropic
HUGINN_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...
```

**方式 B — 配置文件**（`huginn.toml`）：

```toml
provider = "openai"
model = "gpt-4o"
api_key = "sk-..."
base_url = ""
workspace = "."
auto_approve = false
enable_exploration = true
max_parallel_branches = 5
```

JSON 格式（`huginn.json`）同样接受。

**方式 C — CLI 标志**：

```bash
huginn chat --provider openai --model gpt-4o
huginn chat --provider ollama --ollama-url http://localhost:11434
huginn chat --provider vllm --base-url http://localhost:8000/v1 --model llama-3-8b
```

> 本地端点（vLLM / LM Studio / TGI）当 `--base-url` 指向 `localhost` / `127.*` / `::1` / `0.0.0.0` 时**不需要真实 API key**，自动发送 dummy key。

#### Provider 参考表

| Provider | 需要 API Key | 默认模型 |
|----------|-------------|---------|
| `openai` | 是（除非本地） | `gpt-4o` |
| `anthropic` | 是 | `claude-3-5-sonnet-20241022` |
| `deepseek` | 是 | `deepseek-chat` |
| `google-genai` | 是 | `gemini-2.5-pro` |
| `openrouter` | 是 | `anthropic/claude-sonnet-4` |
| `nvidia` | 是 | `meta/llama-3.1-405b-instruct` |
| `ollama` | 否 | `qwen2.5:14b` |
| `vllm` | 否 | 必填 |
| `local` | 否 | 必填（任意 OpenAI 兼容本地服务器） |

### 7.3 启动方式

#### 交互式聊天

```bash
huginn chat
# 或直接 Python：
python -m huginn.cli chat
```

#### 启动 HTTP/WS 后端服务

```bash
huginn serve --port 8000
# 或：
python -m huginn.server --port 8000
# 端口写入 ~/.huginn/backend_port 供前端发现
```

#### 桌面应用

启动 Tauri 桌面应用后，Rust 外壳自动：
1. 探测 `127.0.0.1:8000` 是否已被占用，是则复用
2. 否则分配新端口，依次尝试 `huginn-sidecar` sidecar → `huginn serve` sidecar → `python -m huginn.server` 兜底

#### Docker 单机编排

```bash
docker compose up -d                       # 只起 agent（端口 8999）
docker compose --profile ollama up -d      # 连本地 ollama 一起起
```

环境变量走 `.env` 或当前 shell，不写死在 compose 文件。

#### 自主编码模式

```bash
huginn coder "Refactor the CLI argument parsing into a separate module"
huginn coder "Update all docstrings" --auto-approve   # 谨慎使用
```

#### Autoloop 自主科研

```bash
huginn autoloop "Find stable 2D materials with bandgap > 1.5 eV"
```

### 7.4 可选集成

#### Abaqus MCP

```bash
# 默认位置：~/.abaqus-mcp/mcp_server.py
export ABAQUS_MCP_SERVER_PATH=/path/to/mcp_server.py
# 或在 huginn.toml: abaqus_mcp_server = "/path/to/mcp_server.py"
```

#### COMSOL / OpenFOAM / QE / CP2K

```bash
export COMSOL_EXECUTABLE=/path/to/comsol
export OPENFOAM_DIR=/path/to/openfoam/version
export QE_EXECUTABLE=/path/to/pw.x
export CP2K_EXECUTABLE=/path/to/cp2k.popt
```

未安装时工具导出生成的输入文件供手动执行。

### 7.5 测试

```bash
cd agent
pytest tests/ -x -q                      # Python 测试套件（192+ 测试）
pytest tests/ -m "not integration" -q    # 跳过重型集成测试

cd lean/HuginnLean
lake build HuginnLean                    # Lean 形式化库构建

cd ../../cli
./target/release/huginn --version        # Rust CLI 冒烟测试
./target/release/huginn tools            # 列出所有工具
```

### 7.6 关键环境变量

| 变量 | 默认 | 用途 |
|------|------|------|
| `HUGINN_PROVIDER` | 自动检测 | LLM provider |
| `HUGINN_MODEL` | provider 相关 | 模型名 |
| `HUGINN_API_KEY` | — | 通用 API key（最高优先级） |
| `HUGINN_BASE_URL` | — | OpenAI 兼容端点 |
| `HUGINN_AUTO_APPROVE` | `false` | 自动批准所有工具调用 |
| `HUGINN_ENABLE_EXPLORATION` | `true` | 启用多分支探索 |
| `HUGINN_MAX_BRANCHES` | `5` | 最大并行分支 |
| `HUGINN_LAZY_CLI` | — | `=1` 启用 CLI 懒加载模式 |
| `HUGINN_RATE_LIMIT_PER_MINUTE` | `120` | API 限流 |
| `HUGINN_SESSION_TTL_HOURS` | `24` | 空闲线程 TTL |
| `HUGINN_LLM_REQUEST_TIMEOUT` | `120` | LLM 请求超时（秒） |
| `HUGINN_MAX_TOOL_CALLS` | `15` | 单轮工具调用上限 |
| `HUGINN_MAX_TOOL_CALLS_PER_TOOL` | `5` | 单工具调用上限 |
| `HUGINN_CHECKPOINTER_PATH` | — | 检查点 SQLite 路径 |
| `HUGINN_MEMORY_DECAY_ENABLED` | `false` | 启用记忆衰减 |
| `HUGINN_ENV` | — | `=production` 关闭 `/docs` |
| `HUGINN_CORS_ORIGINS` | localhost:3000/1420/8000 | CORS 白名单 |

### 7.7 CLI 命令一览

| 命令 | 用途 |
|------|------|
| `huginn chat` | 交互式聊天 |
| `huginn coder "<task>"` | 自主编码（Codex 风格） |
| `huginn explore "<objective>"` | 设计空间系统搜索 |
| `huginn autoloop "<goal>"` | 自主科研闭环 |
| `huginn serve` | 启动 HTTP/WS 后端 |
| `huginn tools` | 列出所有工具 |
| `huginn configure` | 交互式首次配置向导 |
| `huginn bench` | 基准测试套件 |
| `huginn evolve` | 从执行日志运行自进化 |
| `huginn execute` | 运行 workflow 阶段 |
| `huginn workflow` | 运行 workflow 模板 |
| `huginn diagnose` | 诊断计算化学错误 |
| `huginn hpc test/submit/status` | HPC 集群操作 |
| `huginn encrypt-config` | 加密配置文件 |
| `huginn kg build` | 构建知识图谱 |
| `huginn swarm` / `huginn team` | 多智能体编排 |
| `huginn unified` | 统一科学计算框架 |
| `huginn persona` | 人格管理 |
| `huginn replay` | 回放 agent 决策轨迹 |
| `huginn sessions` | 会话管理 |
| `huginn plot` / `huginn visualize` | 可视化 |
| `huginn telemetry` | 遥测 |
| `huginn remote` | 远程操作 |
| `huginn memory-maintenance` | 记忆维护 |
| `huginn seed-knowledge` | 知识库种子 |
| `huginn skill-import` | 跨平台技能导入 |
| `huginn background` | 后台任务管理 |
| `huginn model-list` | 模型列表 |
| `huginn export` | 数据导出 |

### 7.8 全局 CLI 标志

```
-w, --workspace <dir>       工作区目录
-c, --config <path>         配置文件路径
-m, --model <name>          模型名
-p, --provider <name>       provider
-u, --base-url <url>        OpenAI 兼容端点
    --ollama-url <url>      Ollama base URL（默认 http://localhost:11434）
    --thinking low|medium|high   推理强度
    --dry-run               只显示将执行什么，不实际运行
    --plan                  Plan 模式（只读，所有写工具强制 ASK）
    --yolo                  跳过所有确认（等价 auto_approve_all）
-P, --prompt "<text>"       Headless 模式：执行一次 prompt 后退出
    --resume <thread_id>    恢复指定会话
    --allowed-tools <csv>   工具白名单
    --disallowed-tools <csv> 工具黑名单
```

---

## 8. 数据流与调用拓扑

### 8.1 单轮对话数据流

```
用户查询
   ↓
HuginnAgent.chat()
   ↓
ContextBuilder 构建上下文（记忆召回 + KG/KB 查询 + 情绪追踪）
   ↓
LLM 推理 → 工具选择
   ↓
ToolAdapter.adapt() — 6 道闸门 pre-check
   ↓
HuginnTool.call() — 工具执行（本地 / MCP / HPC）
   ↓
结果序列化（脱敏 + 压缩 + 卸盘 + 自动出图）
   ↓
结果存储（SessionContext + LongTermMemory 自动晋升）
   ↓
响应生成（流式）
```

### 8.2 Autoloop 闭环数据流

```
目标输入
   ↓
perceive      — 感知（读 KG / 记忆 / 文献）
   ↓
hypothesize   — 假设（ConjectureGenerator Moonshine 三步法）
   ↓
plan          — 计划（PlanStore 持久化，JEPA 预期结果）
   ↓
execute       — 执行（WorkflowEngine + ToolRegistry）
   ↓
validate      — 验证（MathEvidenceChecker + Dempster-Shafer 合成）
   ↓               失败 → refine（max 8 次防死循环）
learn         — 学习（写 KG / 记忆 / SkillEvolution 贝叶斯更新）
   ↓
report        — 报告
   ↓
阶段转移经 PhaseGateHook + RedTeamReviewer 三重把关
```

### 8.3 跨进程通信拓扑

```
┌──────────┐  子进程委派   ┌──────────────┐
│ Rust CLI │ ───────────► │ Python Agent │
└──────────┘              └──────┬───────┘
                                 │
┌──────────┐  HTTP + WS         │
│ 桌面应用 │ ◄─────────────────►│
│ (Tauri)  │  Tauri Command     │
└────┬─────┘ ───────────────────┘
     │
     │ spawn
     ▼
┌──────────┐  HTTP + WS         ┌──────────────┐
│ Sidecar  │ ◄─────────────────►│ Python Agent │
└──────────┘                     └──────┬───────┘
                                        │ stdio
                                        ▼
                                 ┌──────────────┐
                                 │ MCP Servers  │
                                 │ (mat-db /    │
                                 │  math-anything)│
                                 └──────────────┘
```

所有跨进程网络通信限制在 `127.0.0.1`，端口动态分配。

### 8.4 SymPy → Lean 验证链

```
LLM 调 symbolic_math_tool 推导表达式
   ↓
SymPyToLean.translate() 翻译为 Lean 4 字符串
   ↓
AutoLeanPipeline.verify_* 生成完整 Lean 代码
   ↓
LeanInterface.build() 调 lake build
   ↓
失败 → LeanCodeFixer 规则修复（最多 2 次重试）
   ↓
成功 → 数学正确性从 LLM 自评升级为机器可检验
```

---

## 9. 设计原则与约定

### 9.1 核心设计原则

1. **优雅降级** — 每个组件都有 mock/fallback 模式，无完整基础设施也可开发
2. **默认安全** — 静态加密、内存只存 key、per-item salt；权限 fail-closed（未声明只读一律需确认）
3. **模块化** — 每个组件可独立使用
4. **类型安全** — Pydantic v2 覆盖所有输入/输出
5. **可测试性** — 192+ 单元测试覆盖所有主要组件

### 9.2 关键约定

- **Fail-closed 默认**：`ToolMetadata` 默认 `requires_confirmation=True`，新工具必须显式声明 `read_only=True` 才进自动批准
- **声明式 profile**：`ToolProfile` 取代手工维护的四份 dict（HEAVY_TOOLS / LIGHT_TOOLS / PHASE_TOOLS / CONSTRAINT_SCOPES）
- **6 道闸门 pre-check**：permission → cache → router → budget → loop_detector → circuit_breaker
- **active=False 双轨**：对 LLM 不可见但 `ToolRegistry.get()` 仍可直调，配合 `tool_search_tool` 做 progressive tool discovery
- **shim 模式**：顶层 `*_tool.py` 多为单行 re-export，真实实现在子包，避免单文件膨胀
- **`_init_kwargs_map`**：声明式 config 注入，避免 `register_all_tools()` 里按类名写 if 分支
- **provenance 零成本**：contextvar collector 默认 `None`，独立工具调用零开销；engine 绑定后自动抓快照

### 9.3 开发规范

- **代码风格**：`black`（line-length=88）+ `ruff`（E/F/W/I/N/UP/B/C4/SIM 规则集）+ `mypy`
- **文档风格**：Google convention pydocstyle
- **提交前**：`pre-commit` 钩子
- **添加新工具**：继承 `HuginnTool` → 定义 `name`/`description`/`input_schema` → 实现 `_execute()` → 在 `__init__.py` 注册 → 加测试
- **添加新 Lean 模块**：创建 `.lean` 文件 → 加 import 到 `HuginnLean.lean` → 加 `verify_*` 到 `auto_pipeline.py` → 注册 workflow template → 加 skill preset

### 9.4 关键文件速查

| 关注点 | 文件路径 |
|--------|---------|
| Agent 主类 | `agent/huginn/agent/core.py` |
| 工具基类 | `agent/huginn/tools/base.py` |
| 工具注册 | `agent/huginn/tools/__init__.py` |
| 工具适配 | `agent/huginn/tools/adapter.py` |
| 配置管理 | `agent/huginn/config.py` + `agent_config.py` |
| LLM 工厂 | `agent/huginn/llm.py` + `models/registry.py` |
| LLM 重试 | `agent/huginn/llm_retry.py` |
| FastAPI 应用 | `agent/huginn/server.py` + `lifespan.py` + `server_context.py` |
| 记忆管理 | `agent/huginn/memory/manager.py` |
| Autoloop 引擎 | `agent/huginn/autoloop/engine.py` |
| 工作流引擎 | `agent/huginn/workflows/engine.py` |
| Lean 接口 | `agent/huginn/lean/interface.py` + `auto_pipeline.py` |
| CLI 入口 | `agent/huginn/cli/main.py` + `commands/` |
| Rust CLI | `cli/src/main.rs` |
| Tauri 外壳 | `desktop/src-tauri/src/main.rs` |
| React 入口 | `desktop/src/main.tsx` + `App.tsx` |
| MCP 客户端 | `agent/huginn/mcp_client.py` |

---

## 附录：架构决策记录速查

| 决策 | 理由 |
|------|------|
| Mixin 组合而非继承 | 避免 god-class，保持单一入口 |
| ContextBuilder 抽离 | 上下文构建独立可测可复用 |
| 双引擎架构 | Autoloop（全闭环）与 Deli（半闭环）职责分离 |
| ServerContext 单例容器 | 避免模块级全局变量散落 |
| 工具三层过滤 | 避免 130+ schema 一次性塞满 LLM 上下文 |
| SymPy → Lean 桥 | 数学正确性从 LLM 自评升级为机器可检验 |
| Rust 子进程委派 CLI | Rust 负责启动器，Python 负责业务，复用现有 Python 生态 |
| Tauri 三通道通信 | 同步控制 / 请求响应 / 流式各走最佳通道 |
| PyO3 加速热点 | LAMMPS/VASP 解析是性能瓶颈，Rust 加速 5-10x |
| fail-closed 权限 | 安全优先，新工具默认需确认 |
