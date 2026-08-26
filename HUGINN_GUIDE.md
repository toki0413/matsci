# Huginn 综合介绍与使用手册

> 适用版本：v1.3.x（包名 `huginn-agent`）。本文档向使用者与开发者全面介绍 Huginn 是什么、设计哲学、能做什么、架构分层，以及如何安装、配置、运行、编程扩展与排查问题。
>
> **GitHub 仓库**：`https://github.com/toki0413/matsci`
>
> 依据：根 [README](README.md)、`agent/README.md`、`agent/docs/{architecture.md,DESIGN_RATIONALE.md,tech-spec.md,INDEX.md}`、`cli/README.md`、`desktop/README.md`、`huginn.toml.example`。

---

## 目录

- 第一部分 · 综合介绍
  - 1. 一句话定位
  - 2. 核心动机与三根支柱
  - 3. 设计哲学（数学=工具箱+审计层；朗兰兹=隐喻）
  - 4. 核心特性地图
  - 5. 两条正交控制轴
  - 5.1 运行模式（Mode 与 Phase）
  - 5.2 权限模式（PermissionMode + RiskLevel）
  - 6. 系统架构与数据流
  - 7. 七条设计原则
  - 8. 分层能力详解（Agent / Capability / Metacog 数学审计 / Runtime）
  - 9. 演化路线（H0–H5）
- 第二部分 · 使用手册
  - 11. 安装
  - 12. 配置 LLM Provider 与多模型路由
    - 12.1 三种配置方式 · 12.2 云端 Provider · 12.3 本地 API · 12.4 多模型路由池
  - 13. 启动与聊天
  - 14. CLI 子命令速查
  - 14.1 HTTP/WS API 端点
  - 15. 桌面应用
  - 16. 技能（Skills）与预设
  - 17. 记忆系统、知识蒸馏与视觉/时空工作流
    - 17.1 三层记忆 · 17.2 知识蒸馏 · 17.3 外部思维（External Thinking） · 17.4 三库隔离 · 17.5 视觉→知识库→记忆闭环 · 17.6 时空可组合 · 17.7 接入 Embedding 模型 · 17.8 文档文字提取与 OCR 优先级
  - 18. Coder / 多 Agent / 自主科研（autoloop）
  - 19. 安全
  - 20. 监控、部署与运维
  - 21. 编程扩展（新增工具 / 技能 / MCP）
  - 22. 常见问题排查

---

# 第一部分 · 综合介绍

## 1. 一句话定位

Huginn 是一个 **LLM 驱动的通用科研智能 Agent 系统**：以 **LLM 推理循环为地基**，用**数学做真算工具 + 结构审计层**，对外以三根支柱承载**自主探索、系统自演化、知识可审计**。

它从计算材料科学起家，现已**泛化为覆盖多学科科研自动化**的通用 Agent：自动执行 DFT、分子动力学、CFD/FEA 仿真、符号回归、因果分析、文献检索、自主设计空间探索与多智能体协作。材料/仿真只是能力集里的一个切片，系统重心在通用科研自动化（研究项目编排、多智能体 team、知识蒸馏、因果/结构分析、MCP 工具生态、远程/HPC 调度）。

面向数学敏感的任务，Huginn 还提供 **Lean 4 形式化检查**这一可选能力：`agent/lean/HuginnLean/` 是一套 Lean 4 源码工程，为张量代数、有限元弱形式、数值线性代数、DFT 理论、热力学与概率建立了程序化定义，可把 SymPy 表达式翻译成 Lean 做编译/类型核验。它需要**外部 `lake` 工具链**（通过 elan 安装），由 `LeanInterface` 调用 `lake build` 编译 `.lean`。它是 agent 在数学验证类任务里可选调用的工具，日常普通科研流程不会自动触发（详见 §17.2 的触发条件）。

## 2. 核心动机与三根支柱

材料科学研究目标通常是一条**分叉的探索路径**：有的方向可行、有的撞墙、有的推翻了子假设。传统 workflow（定死的 DAG）很难表达"自主分叉—试错—学习—再试"。

所以 Huginn 围绕一个 **认知回路（Cognitive Loop）** 组织，让 agent：

1. **有主动性**：在 objective 下自主提出假设、执行、验证、迭代。
2. **可演化**：不仅让 LLM 跑任务，系统本身行为（prompt 模板 / workflow / phase / 工具白名单）也可以被改进。
3. **对错误诚实**：能区分"我知道 / 我推测 / 我没把握"，把不确定性如实标注出来。

这三点对应三根支柱：**Agent 循环**、**Harness 可演化**、**Knowledge 可审计**。

## 3. 设计哲学

Huginn 在 LLM 推理之上叠加了一层数学工具与结构审计（同调不变量、sheaf 上同调、因果模型、Lean 检查等），它们的角色是**补强** LLM 判断：把"结构上是否自洽"的信号喂回推理循环，用数学给结构边界、用 LLM 给语义判断，二者互补。

存储层面，memory（时序）、knowledge（向量）、kg（图拓扑）三种记忆各自用原生结构保存，只在明确边界（如组装 prompt 时）做跨库翻译，且翻译保持结构对应——避免把所有信息降维成单一向量。

> 系统的目标是把科研变成一条可回溯、可验证、可演化的探索路径，而不是迷信某一种理论框架。

## 4. 核心特性地图

| 维度 | 能力 |
|---|---|
| 通用科研编排 | 研究项目编排、多智能体 team、知识蒸馏闭环、自进化（evolve）、自主科研闭环（autoloop）、知识图谱 |
| 形式化数学 | 6 阶段 Lean 4 源码工程：张量代数 → FEM 弱形式 → 数值线性代数 → DFT → 热力学 → 概率；SymPy 表达式翻译为 Lean，经外部 `lake` 做编译与类型检查 |
| 多 Provider LLM | OpenAI、Anthropic、DeepSeek、Google GenAI、OpenRouter、NVIDIA、Ollama、vLLM、LM Studio、任意 OpenAI 兼容本地端点 |
| 运行模式 | 5 种 Mode（chat/research/extreme/code/fusion，顶层行为）+ 7 种 Phase（文献→假设→规划→执行→验证→报告，每 phase 独立工具预算） |
| 权限模式 | 4 种 PermissionMode（auto/ask/deny/plan）+ 5 档风险（none/low/medium/high/critical）+ 工具默认矩阵 + 危险命令白名单 + 沙箱硬底线路径 |
| 工具生态 | 约 178 个内置工具 + ToolRegistry；分核心（约 35 轻量）与可选（重依赖仿真）两类 |
| MCP 集成 | 连接 Materials Project、NIST 数据库与数学分析工具；内置 3 个 MCP 服务器（mat-db / math-anything / vision-pixel） |
| 智能检索 | ChromaDB 嵌入 + 关键词兜底 + 加密存储（RAG） |
| 记忆系统 | 三层记忆（session / 长期 SQLite+FTS5 / 自动提升）+ 知识蒸馏闭环 |
| 技能 | 声明式材料/科研工作流（12+ 预设 SkillDefinition + .md） |
| 元认知审计 | 同调、sheaf H¹、Hodge、范畴 functor、持续同调等接入主循环做 advisory 结构审计 |
| 物理世界访问 | `PhysicalWorkspace`：时间可逆 + 空间可组合 + 感知确认的实验协议编排 |
| 安全 | AES-128 静态加密、容器沙箱、命令白名单、fail-closed 工具元数据、统一错误信封、密钥分层 |
| 多形态入口 | Rust CLI、Tauri 桌面应用、Python CLI、FastAPI 服务、脚本 |

## 5. 两条正交控制轴（核心心智模型）

理解 Huginn 的关键：**极简模式** 与 **思考强度** 是两个**独立**维度，互不干扰。

| 维度 | 环境变量 | 控制什么 | 取值 |
|---|---|---|---|
| **极简模式 ModelTier** | `HUGINN_MODEL_TIER` | 认知编排开销（phase 机 / plan 门控 / 认知纪律 / compaction / 外部思考） | `full` / `balanced` / `minimal` |
| **思考强度 ThinkingIntensity** | `HUGINN_THINKING` | 模型推理深度（provider reasoning budget） | `low` / `medium` / `high` / `max` |

| 档位 | phase 机 | plan 门控 | 认知纪律 | compaction | 外部思考 | 适用 |
|---|---|---|---|---|---|---|
| `full` | ✅ | ✅ | `always` | `heavy` | ✅ | 本地弱模型，保留全部认知编排 |
| `balanced` | ✅ | ✅ | `event` | `medium` | ✅ | 中等模型，纪律降级为事件驱动 |
| `minimal` | ✗ | ✗ | `event` | `light` | ✗ | 顶尖大模型，跳过 phase/plan 门控 |

> 安全层（命令校验 / 物理 sanity check / 资源预算）在**所有档位始终保留**。越 `minimal` 越信任模型，跳过 phase/plan 门控。

```bash
# 本地弱模型：保留全部认知编排
HUGINN_MODEL_TIER=full huginn-agent chat

# 顶尖大模型：跳过编排，推理预算拉满
HUGINN_MODEL_TIER=minimal HUGINN_THINKING=max huginn-agent chat
```

## 5.1 运行模式（Mode 与 Phase）

除了"模型强弱"这条轴，Huginn 还通过 **Mode**（顶层行为）与 **Phase**（研究流程阶段）区分工作形态——同一套后端，按目标切成不同节奏。

**Mode —— 顶层行为**（影响 agent 的提示指令与整体姿态）：

| Mode | 行为说明 |
|---|---|
| `chat` | 对话助手：直接作答，不做重仿真 |
| `research` | 系统化研究：论断引文献、量化证据 |
| `extreme` | 极端模式：长程任务、放开全部能力上限 |
| `code` | Code-act 模式：靠写代码 + 执行解决问题 |
| `fusion` | 融合模式：跨仿真/实验/文献整合证据 |

**Phase —— 研究流程阶段**（每个 phase 有独立工具调用预算 `max_calls`）：

| phase | 提示头 | 预算 |
|---|---|---|
| literature | Literature Review | 50 |
| hypothesis | Hypothesis Formation | 30 |
| planning | Experiment Planning | 30 |
| execution | Execution | 300 |
| validation | Validation & Analysis | 100 |
| reporting | Reporting | 20 |
| open | （开放式） | 500 |

> mode 管"整体怎么行为"，phase 管"当前科研走到哪一步、允许多少工具调用"。二者与 ModelTier（编排开销）、Thinking（推理深度）相互独立、可任意组合。整套规范可自动再生成（`docs/modes-contract.md`）。

**工具调用预算（已随版本扩容）**：对长程科研/评测任务，工具调用上限结合 mode 设档——
- 普通跑分默认 `max_tool_calls=400`、`max_tool_calls_per_tool=50`（早期为 150，约 +2.7 倍）；
- `--extreme` 极限模式 `max_tool_calls=600`、`per_tool=100`、`context_budget_tokens=200K`；
- 递归深度 `recursion_limit`：research/extreme mode → 500，extreme dispatch → 400，普通 chat/plan → 250；
- extreme 模式同时放宽 autoloop 阈值（max_consecutive_failures/max_refines=50、max_pivots=20、stagnation_limit=15），并把 wall-clock 预算放宽到 1 天级（`HUGINN_RCB_TIMEOUT=86400`）。

（来自 `cli/rcb_runner.py` 的 C3 预算扩容记录与 `agent/core.py._effective_recursion_limit`。）

## 5.2 权限模式（PermissionMode + RiskLevel）

Huginn 的权限是**分层判定**，不是一刀切"要不要问"。

**PermissionMode —— 二元决策模式：**

| 模式 | 语义 |
|---|---|
| `auto` | 只读/安全工具直接放行 |
| `ask` | 潜在昂贵/破坏工具需确认 |
| `deny` | 显式拦截，不可执行 |
| `plan` | 只读模式，所有写工具强制 ASK |

**RiskLevel —— 五档风险（与 PermissionMode 互补）：**

| 等级 | 语义 |
|---|---|
| `none` | 纯只读/查询，直接放行 |
| `low` | 本地只读/可逆变更，默认放行 |
| `medium` | 外部 IO/网络/非破坏状态变更，需确认 |
| `high` | 破坏性/危险，必须确认 |
| `critical` | 不可逆/系统级/极高成本，强制拦截 |

**细粒度判定维度**（`PermissionConfig` 多阶段叠加，命中即记入 `matched_rules` 可观测）：

```
危险命令 → 路径规则(tool×glob×mode 矩阵) → 工具基础规则 → 成本分级 → 信任自适应
```

- **工具默认矩阵**：重仿真工具默认 `ask`（vasp/lammps/comsol/qe/cp2k/abaqus/openfoam…），读工具默认 `auto`（read/grep/glob/search/visualize…），`file_delete_tool`、`system_shell_tool` 默认 `deny`。
- **危险命令白名单**（27 条）：`rm -rf /`、`mkfs`、`dd >/dev/sda`、`shutdown`、`sudo`、`git push --force`、`git reset --hard` 等强制拦截。
- **沙箱硬底线路径**（只能收紧不能放宽）：`INSTRUCTIONS.md`、`score.py`、`evaluation/*.py`、`rubric.json`、`.huginn/checkpoints*` 一律 `deny`。
- 安全硬底线（危险命令/沙箱/成本预算）即使 `auto_approve_all` 也保留——**任何档位都不会放开这几个底线**。

运行时配置：`HUGINN_PERM_*` 环境变量 / 前端设置面板注入；`path_rules` 支持 `(tool, glob, mode)` 三维矩阵。规范见 `docs/permission-contract.md`。

## 6. 系统架构与数据流

### 单网关（Single Gateway）

`huginn.server` 是**唯一业务网关**（FastAPI + WebSocket/SSE，路由挂 `/v1` 并保留根路径兼容）。所有业务逻辑只能通过其后端 HTTP/WS API 被消费。CLI、桌面应用、脚本一律作为 **API 客户端**，不直接 `import huginn.*` 业务模块（由 `tests/test_arch_single_gateway.py` 在 CI 强制）。

```
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│ Rust CLI    │   │  Desktop App │   │ MCP / 脚本/外部 │
│ (API 客户端)│   │  (Tauri 壳)  │   │ (API 客户端)    │
└──────┬──────┘   └──────┬───────┘   └────────┬────────┘
       │  HTTP/WS        │  HTTP/WS          │  HTTP/WS
       └─────────────────┼───────────────────┘
                         ▼
              ┌──────────────────────┐
              │   huginn.server      │  ← 唯一业务网关 (FastAPI + WS/SSE)
              │  routes/**  /v1/*    │     统一鉴权 / 审计 / 错误信封
              └──────────┬───────────┘
                         ▼
        ┌────────────────────────────────────────────┐
        │          Agent 层 / 能力层                  │
        │  agent/ agents/ tools/ skills/ memory/     │
        │  evolution/ knowledge/ kg/ causal/         │
        │  autoloop/ metacog/ runtime/ security/     │
        │  + 可选 Lean 4 检查 (HuginnLean)             │
        └────────────────────────────────────────────┘
```

### 分层结构

简化成四层（各层解决什么问题）：

```
┌──────────────────────────────────────────────────────────────┐
│ Entry: CLI (~40 cmd) · API (FastAPI /v1, WS/SSE)             │
├──────────────────────────────────────────────────────────────┤
│ ① Agent / Autoloop 层 —— 决定"下一步做什么"                   │
│    agent/ core loop、session、streaming、reflection           │
│    autoloop/ CognitiveLoop(observe/decide/execute/reflect)    │
│    agents/  多 agent：orchestrator、subagent、swarm、team      │
├──────────────────────────────────────────────────────────────┤
│ ② Capability 层 —— 决定"怎么做"                               │
│    tools/(~178 + Registry) · skills/(presets + .md)           │
│    memory/ 3-tier · evolution/ · knowledge/ · causal/          │
├──────────────────────────────────────────────────────────────┤
│ ③ Metacog / 数学审计层 —— 决定"判断得可不可信"                 │
│    metacog/ 同调、sheaf H¹、Hodge、范畴 functor、mental imagery│
├──────────────────────────────────────────────────────────────┤
│ ④ Runtime / 安全 / 状态 —— 决定"能不能在真实环境里跑"          │
│    runtime/ · security/ · events/ · persistence/ · hpc/       │
└──────────────────────────────────────────────────────────────┘
```

**为什么这样分：**
- ①②分开是"策略"("下一步做什么")与"能力"("怎么做")是不同东西，混在一起会导致"换套能力就要改推理逻辑"。
- ③独立成层，让**结构审计**（数学不变量）与**语义推理**（LLM）互不污染：审计结果作为 advisory 注入 prompt（`engine_observe.py`），不直接改推理内核。
- ④兜住真实环境约束（安全、持久化、远程执行），让①②③可以"天真"地研究而不必担心炸掉生产。

### 主数据流

```
User Query → CLI / WS / HTTP
  → HuginnAgent loop (agent/core.py)
    → prompt_builder + memory injection
      → LLM reasoning (streaming 捕获 reasoning_content → reasoning_trace)
        → tool_call_router → tool execution (local / MCP / HPC)
          → ToolResult → session + long-term memory
            → 成功工具调用 → 蒸馏知识升级 confirmed → auto_ingest_to_kb → 知识库
      → response (streaming / SSE / WS)
```

自主探索（autoloop）：

```
Objective → 分支生成 → 评估 → Pareto 剪枝 → 失败诊断/回溯 → 返回最优方案
```

## 7. 七条设计原则

1. **优雅降级**：每个组件带 mock/fallback 路径。没有 chromadb 也能用离线 KB；GUDHI 没装也能退 networkx 风格 β₀/β₁。
2. **安全默认（fail-closed）**：工具元数据默认只读/非破坏/需确认；密钥只进内存、逐 item salt。
3. **模块化 + 类型安全**：Pydantic 统一 I/O；组件可独立使用。
4. **可测防回归**：session 级共享 app + autouse 护栏保证 `ToolRegistry` 测试间逐位一致；覆盖率门禁 60。
5. **文件系统即记忆**：harness 变体 / patch / 归档存文件（`.huginn/...`），不塞 context window（权限外记忆）。
6. **保持控制流稳定、换回退成本**：CognitiveLoop 的 4 个钩子（observe/decide/execute/reflect）签名稳定，演化发生在钩子**内部**。升级一次最坏回退成本是删一个目录，而非一次大重构。
7. **数学做真算审计、做隐喻、不做装饰**：每个数学模块要么真算出结构不变量喂给判断，要么明确标注 research/实验层；不允许"借个高大上名字但什么都没算"的东西混进主循环。

## 8. 分层能力详解

### 8.1 Agent 循环（`agent/` + `autoloop/`）

- `agent/core.py`：主循环 `start()/step()/process_action()`，状态流转与回调。
- `session.py` / `streaming.py`：会话上下文 + 流式输出；AIMessage 的 `reasoning_content`（如 DeepSeek COT）写入 `reasoning_trace` 供下游蒸馏。
- `autoloop/engine.py`：`AutoloopEngine` 实现 7 个 phase——**perceive → hypothesize → plan → execute → validate → learn → report**；每次迭代跑 `_metacog_topology_audit` 对假设图做结构审计。
- **7 phase 的必要性**：把"产生想法"与"验证想法"分开，才能在「想法—结果」间建立因果日志，否则 agent 无法判断"我上次改进到底起没起作用"。

### 8.2 多 Agent（`agents/`）

- `orchestrator.py`：多 agent 编排 `run_agent()` / `handle_subagent_output()`。
- `subagent.py`：子 agent 执行与输出；`swarm.py` / `team.py`：协作方式。
- `tool_call_router.py` / `tool_dedupe.py` / `loop_detector.py`：工具调用路由、去重、循环检测。
- **Orchestrator 与 SubagentDispatch 各司其职**：两者是不同的入口（顶层多 agent 工作流 vs 主 agent 上下文卸载工具）、不同的生命周期，共享同一 `factory.create()` 底层；防失控守卫保护不同的风险面（DAG 环检测+max_concurrent 计划合法性 vs max_depth 递归 + max_tool_calls 上下文预算）。

### 8.3 Capability 层（`tools/` / `skills/` / `memory/` / `causal/`）

工具按类别（约 178 个，见第 21 节完整分类）：

| 类别 | 代表工具 |
|---|---|
| Coder / 文件 | `bash_tool`、`code_tool`、`file_read/write/edit_tool`、`multi_edit_tool`、`glob_tool`、`grep_tool`、`git_tool`、`github_tool`、`diff_tool`、`eval_tool`、`validate_tool`、`diagnose_tool` |
| Sci / DFT | `vasp_tool`、`qe_tool`、`cp2k_tool`、`gaussian_tool`、`orca_tool`、`structure_tool`、`symmetry_tool`、`xrd_sim_tool` |
| Simulation | `lammps_tool`、`gromacs_tool`、`openmm_tool`、`openfoam_tool`、`comsol_tool`、`abaqus_tool`、`fenics_tool`、`elmer_tool`、`packing_tool`、`fep_tool`、`enhanced_sampling_tool` |
| Symbolic / 数学 | `symbolic_regression_tool`、`symbolic_math_tool`、`discrete_smt/group/oeis/additive_tool`、`numerical_tool`、`unit_tool`、`autodiff_tool`、`lean_tool`、`bourbaki_tool`、`tensor_algebra` |
| 数据 / 检索 | `database_tool`、`report_tool`、`extract_tool`、`tool_search_tool`、`agentic_search_tool`、`web_search_tool`、`literature_tool`、`materials_database_tool`、`experimental_data_tool` |
| 记忆 / 元 | `remember_tool`、`recall_tool`、`recall_context_tool`、`self_observe_tool`、`todo_tool`、`notebook_tool`、`scheduler_tool`、`plan_store_tool`、`prospective_tool` |
| 多 Agent | `subagent_tool`、`orchestrate_tool`、`review_committee_tool`、`skills_tool`、`workflow_tool` |
| 视觉 / ML | `visualize_tool`、`vision_describe_tool`、`image_analysis_tool`、`image_design_tool`、`model3d_tool`、`ml_potential_tool`、`active_learning_tool`、`interpretable_ml_tool`、`gnn_tool`、`vae_tool`、`transformer_tool` |

> 权威工具类清单见 `huginn/tools/__init__.py::_CORE_MODULES`（核心）与 `_OPTIONAL_MODULES`（可选重依赖仿真）。

### 8.4 Metacog 数学审计层（`metacog/`）

这是本项目区别于"知识库 = 文档检索"之处，也是最有特色的一层：

- **已接进主循环**：单纯同调/持续同调、sheaf H¹、Hodge 签名、范畴 functor、cognitive map、mental imagery（如 `simplicial_homology.py`、`sheaf_cohomology.py`、`hypothesis_loop.py`）。
- **advisory 定位**：数学不变量是**结构上**的诚实信号（"多源证据全局不一致 H¹≠0"、"旋转模式无对应流形 β_rotation=0"）；材料科学判断还需要**语义上**的诚实（某物相是否合理、某文献是否可信），这部分交给 LLM。二者互补：**数学给结构边界，LLM 给语义判断**。
- 数学结果经 `engine_observe.py:1026` **回灌 prompt**，作为"这个假设结构上自洽吗"的提示。部分更强力的拓扑/范畴模块（如 `experimental/persistence_landscape.py`）以 research 层形态存在，是未来扩展的起点。

### 8.5 Knowledge / CLAIM 层（`kg/`）

"像 debug 软件一样 debug 人类已有的科学认知"：

- 把文献结论提升为知识图谱里的 **CLAIM 一等节点**（`kg/entities.py`、`kg/graph.py`）。
- 用**超图**表达"结论 ← n 元前提"的 AND/OR 依赖（`kg/hypergraph.py`）。
- 用 **sheaf H¹** 检测多源证据全局矛盾；`ClaimAuditor` 编排"注册→冲突检测→挑战传播→自指环审计"。
- 当新文献进来时，系统回答的内容包含"它支持 / 挑战 / 改变了哪些结论可信度 / 改了谁的适用边界 / 若挑战源头则哪些下游要复查"。
- 存疑结论打 `contested` 标签，在 RAG 检索和 context 构建时注入提醒，让模型把有争议知识当作待核验项、谨慎使用。这套机制承担"审计 + 提示"的职责，让 agent 对证据冲突与结论受挑战更诚实。

### 8.6 Runtime / 安全 / 状态（`runtime/` `security/` `events/` `hpc/`）

- `runtime/`：任务生命周期、调度、trace 上下文、前置 grill、sanity gate。
- `security/`：鉴权、中间件、权限模式、物理 sanity 预检、可逆副作用。
- `events/`：事件总线 + 审计日志（append-only + hash-chained）。
- `persistence/`：检查点 / 状态持久化；`hpc/`：远程执行（SSH）。
- **可逆副作用**：`security/revertible.py`（RevertibleContext LIFO 逆栈 + composite 扭结算子 + transaction 回滚 + track_op 数据驱动逆 + journal 崩溃重放 + compensate 出站补偿 + track_world_action 物理世界逆）。`workflows/engine.py` 用 `composite()` 编排多 stage 为原子单元。`snapshot/file_snapshot.py` 做文件系统 undo。

## 9. 演化路线（H0–H5）

Huginn 的"自进化"按 H0→H5 递进落地：

| 阶段 | 内容 |
|---|---|
| H0 | stable_principles 接入 prompt（最小修复"PM 层在 autoloop 不通"） |
| H1 | prompt template patch（agent 能改自己的 prompt 模板 block，自改进最小闭环） |
| H2 | workflow 演化搜索（同一 objective 生成 N 个 workflow 变体，物理奖励 bandit 选优） |
| H3 | 联合优化（prompt block + workflow 参数联合 bandit） |
| H4 | phase 行为体可演化（phase 方法体 / subagent spec 抽成 PhaseSpec，可改行为但不失控） |
| H5 | unified LLM client / tool dispatch（LLM 调用、工具白名单真正统一） |

**为什么递进**：每一步是独立可验证闭环（selfcheck 过了、能跑 3 轮 autoloop 不崩才继续），前一步是后一步地基。失败代价停留在那一层。新功能同样按"0 骨架 → 1 单点 → 5 全链路"节奏。

---

# 第二部分 · 使用手册

## 11. 安装

**前提**：Python 3.11+（推荐 3.11）。

```bash
cd agent
# 推荐: uv
uv venv --python 3.11
uv pip install -e ".[all]"
# 或 pip
pip install -e ".[all]"
```

可选依赖组：
- `[all]` — 完整工具依赖（pymatgen/ase/matplotlib/chromadb/sentence-transformers/pymupdf/easyocr 等）。
- `[ocr]` — 中文/混合版式 OCR（paddleocr）。**不进 `[all]`**（依赖链庞大），仅在桌面打包时额外安装，或按需 `pip install ".[ocr]"`。

> 桌面分发版（Windows）：直接运行 `Huginn_*_x64-setup.exe`，自动附带 Python 运行时 + sidecar，离线可用。

可选 Rust 扩展（LAMMPS/VASP 解析、MSD/RDF 加速）：见 `agent/DEPLOYMENT.md` 与 `pyext/`。

后端依赖解析：`uv pip compile` 生成 `requirements.lock`，CI 门禁拦截漂移，保证可复现。

## 12. 配置 LLM Provider 与多模型路由

### 12.1 三种配置方式

优先级：**CLI 参数 > 配置文件 > 环境变量**。

```bash
# 方式 A — 环境变量
export HUGINN_PROVIDER=openai
export HUGINN_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

# 方式 B — 配置文件（交互向导）
huginn-agent configure            # 生成 huginn.toml
huginn-agent chat --config huginn.toml

# 方式 C — 命令行参数（临时切换，不落盘）
huginn-agent chat --provider ollama --ollama-url http://localhost:11434
huginn-agent chat --provider vllm --base-url http://localhost:8000/v1
```

### 12.2 云端 Provider（需要真实 API Key）

Huginn 按 provider 区分云端与本地：**云端必须配置有效 key，本地自动豁免**。云端 provider 清单（来自 `models/registry.py`）：

| provider | 环境变量 | 默认 base_url | 默认模型 | 原生多模态 |
|---|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | https://api.openai.com/v1 | gpt-4o | ✅ |
| `anthropic` | `ANTHROPIC_API_KEY` | https://api.anthropic.com | claude-3-5-sonnet-20241022 | ✅ |
| `deepseek` | `DEEPSEEK_API_KEY` | https://api.deepseek.com | deepseek-v4-flash（另供 deepseek-v4-pro / deepseek-v4-flash-vision-exp 视觉版） | —（仅 vision-exp ✅） |
| `google-genai` | `GOOGLE_API_KEY` | https://generativelanguage.googleapis.com | gemini-2.5-pro | ✅ |
| `openrouter` | `OPENROUTER_API_KEY` | https://openrouter.ai/api/v1 | anthropic/claude-sonnet-4 | ✅ |
| `nvidia` | `NVIDIA_API_KEY` | （NVIDIA AI Endpoints） | meta/llama-3.1-405b-instruct | — |

国内 OpenAI 兼容云端：

| provider | 环境变量 | 默认 base_url | 默认模型 | 原生多模态 |
|---|---|---|---|---|
| `siliconflow` | `SILICONFLOW_API_KEY` | https://api.siliconflow.cn/v1 | deepseek-ai/DeepSeek-V3 | — |
| `moonshot` | `MOONSHOT_API_KEY` | https://api.moonshot.cn/v1 | kimi-k2.6 | ✅ |
| `zhipu` | `ZHIPU_API_KEY` | https://open.bigmodel.cn/api/paas/v4/ | glm-4-flash | —（视觉走独立 GLM-4.xV/5V） |
| `baichuan` | `BAICHUAN_API_KEY` | https://api.baichuan-ai.com/v1 | Baichuan4 | — |
| `dashscope` | `DASHSCOPE_API_KEY` | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen3.5-plus | ✅ |
| `qianfan` | `QIANFAN_API_KEY` | https://qianfan.baidubce.com/v2 | ernie-4.0-turbo-8k | — |
| `doubao` | `DOUBAO_API_KEY` | https://ark.cn-beijing.volces.com/api/v3 | doubao-pro-32k | — |
| `hunyuan` | `HUNYUAN_API_KEY` | https://api.hunyuan.tencentcloudapi.com/v1 | hunyuan-turbo | — |
| `minimax` | `MINIMAX_API_KEY` | https://api.minimaxi.com/v1 | MiniMax-M2.7 | — |

> 「原生多模态」✅ 表示该模型 API 能直接接收图像（走 `BOTH` 路径），— 表示纯文本模型。**— 不代表拿图像没办法**：文本模型经 `vision/router.py` 的 `CV_TOOLS` 链路（视觉编码器 + 图像分析工具）照样能"看"图像，详细见 §17.5。✅/— 与 `models/registry.py` 的 `MODEL_CAPABILITIES` 一致。

任意一个都可省略 `base_url`（用上表默认值），**只需设对应的环境变量（或配置文件里写 key）即可**。

> **关于默认端点（建议以官方最新文档为准）**：上表的 base_url 与 Huginn 代码内置默认值（`models/registry.py`）逐项一致，是开箱即用的默认配置。注意个别厂商会调整官方端点——例如 SiliconFlow 已于 2025–2026 启用新官方端点 `api.siliconflow.com`，但旧的 `api.siliconflow.cn` 仍声明兼容可用；Moonshot/Kimi 分国内 `api.moonshot.cn` 与国际 `api.moonshot.ai` 两套。如果连接报 401/找不到主机或想用官方最新端点，在配置里显式覆盖 `base_url` 即可（见 12.1 CLI/配置文件方式）。

### 12.3 本地 API（vLLM / LM Studio / Ollama / llama.cpp / SGLang）

本地端点**不校验真实 key**：只要 `base_url` 指向 `localhost` / `127.*` / `0.0.0.0` / `::1`（`_is_local_url` 判定），客户端就自动填 `not-needed` 占位 key，免去手动设假 key。判据还额外认 `:11434`（Ollama 默认端口）。

| provider | 默认 base_url | 默认模型 | 说明 |
|---|---|---|---|
| `ollama` | http://localhost:11434 | qwen3.8 | `is_local_provider` 恒为 True，天然免 key |
| `vllm` | http://localhost:8000/v1 | default | 支持 speculative decoding（`HUGINN_SPECULATIVE_ENABLED`） |
| `lm-studio` | http://localhost:1234/v1 | local-model | |
| `llama-cpp` | http://localhost:8080/v1 | local-model | |
| `sglang` | http://localhost:30000/v1 | default | |
| `openai-compatible` | **必须显式给 base_url** | 无 | 任意 OpenAI 兼容服务，本地给 base_url 即免 key |

**`openai-compatible` 通用型**最灵活：任何 OpenAI 兼容的本地/自建服务都能用它，只需配显式 `base_url`：

```bash
# 连接一个本地任意 OpenAI 兼容服务
huginn-agent chat --provider openai-compatible --base-url http://127.0.0.1:9999/v1 --model my-model
```

### 12.4 多模型路由池

`huginn.toml` 中通过 `[[models]]` 定义别名，`[[agents]]` 按任务路由到不同模型。可混用云端与本地模型——`strong` 用云端大模型、`local` 用本地小模型：

```toml
provider = "openai"                # 默认（被 [[models]] 覆盖前的 legacy 单配置）
model = "gpt-4o"

[[models]]
alias = "strong"
provider = "anthropic"
model = "claude-sonnet-4-6"
# api_key = "..."                  # 省略则用 ANTHROPIC_API_KEY 环境变量

[[models]]
alias = "cheap"
provider = "openai"
model = "gpt-4o-mini"

[[models]]
alias = "local"
provider = "ollama"
model = "qwen3.8"
base_url = "http://localhost:11434"

# Agent profiles: 把任务/工具路由到某个 model alias
[[agents]]
id = "lead"
name = "Lead Scientist"
model_alias = "strong"
persona = "default"
tools = []

[[agents]]
id = "coder"
name = "Code Assistant"
model_alias = "cheap"
persona = "default"
tools = ["python_tool", "file_tool"]
```

### 12.5 常用配置项

```toml
# 行为
workspace = "."
auto_approve = false
enable_exploration = true
max_parallel_branches = 5

# 上下文 / 压缩
max_tool_output_tokens = 25000
context_budget_tokens = 0
tool_compression_max_tokens = 8000

# Prompt 缓存（provider 相关，禁用则关掉）
prompt_cache_control = true

# 持久化会话状态
checkpointer_path = "~/.huginn/checkpoints.sqlite"

# 两条控制轴
# model_tier  = "balanced"
# thinking   = "medium"

# 遥测
telemetry_enabled = true

# 记忆维护
memory_decay_enabled = false
memory_decay_interval_turns = 0
memory_decay_prune_threshold = 0.15

# 隐私
privacy_redact_secrets = true
privacy_block_on_secrets = false

# Pet
pet_name = "Muninn"
pet_personality = "cheerful"
```

> 完整环境变量契约（265 个 `HUGINN_*`）见 `agent/docs/env-contract.md`；功能开关（44 个 FeatureFlags）见 `agent/docs/feature-flags-contract.md`。

## 13. 启动与聊天

```bash
python -m huginn.server            # 启动 API 网关 (http://localhost:8000)
huginn-agent chat                  # 作为 API 客户端连接
```

直接运行（纯客户端模式）：

```bash
huginn-agent chat "计算 Si 的带隙"
huginn-agent coder "给 code_tool.py 加 docstring"
```

## 14. CLI 子命令速查

`huginn-agent` 提供约 40 个子命令（`huginn-agent --help` 查看全部）：

| 子命令 | 用途 |
|---|---|
| `chat [提示词]` | 交互式聊天（SSE 流式） |
| `coder [任务]` | 自主编码（Codex 风格，读/写/编辑/shell/git/执行） |
| `serve` | 启动 HTTP/WS 后端（`--port`/`--host`） |
| `explore <目标>` | 设计空间系统搜索（自主多目标优化） |
| `autoloop` | 自主科研闭环（迭代假设→验证，7 phase） |
| `swarm` / `team` | 多智能体协作 |
| `workflow <模板>` | 运行 workflow 模板 |
| `execute` | 执行 workflow 阶段 |
| `diagnose <错误>` | 诊断计算化学/MD 错误 |
| `hpc test/submit/status` | HPC 集群操作（SSH） |
| `bench` | 基准测试套件（`--evolve` 触发自进化） |
| `evolve` | 从执行日志运行自进化 |
| `kg` | 知识图谱操作 |
| `sessions` | 会话管理 |
| `scheduler` | 调度任务 |
| `replay` | 回放记录 |
| `refactor` | 代码重构 |
| `skills` / `tools` / `model-list` | 列表：技能 / 工具 / 模型 |
| `memory-maintenance` | 记忆维护 |
| `configure` | 交互式配置向导（写 huginn.toml） |
| `encrypt-config` | 加密配置文件 |
| `version` / `help` | 版本 / 帮助 |

全局参数：`--workspace`、`--config`、`--model`、`--provider`、`--dry-run`、`--base-url`、`--ollama-url`。

> Rust CLI（`cli/`）与 Python CLI 子命令一致；Rust CLI 是"壳"，后端在跑时作 HTTP/WS 客户端，未跑时经 `python -m huginn.cli` 子进程兜底。

### 14.1 HTTP / WS API 端点

`huginn.server` 是唯一业务网关，所有端点同时挂在**规范路径 `/v1/*`** 与**根路径 `/*`**（根路径走兼容，带 `Deprecation` 头）。按端点族一览（`routes/__init__.py` 的 `ALL_ROUTERS`，~50 组路由）：

| 端点族 | 职责 | 关键端点 |
|---|---|---|
| 鉴权 | 登录/换 token/JWT | `POST /auth/login`（API key 换 JWT）、`POST /auth/token`、`GET /auth/me`（角色/能力）、`POST /auth/refresh`、`POST /auth/logout` |
| 健康 | 存活/就绪 | `GET /health/live`（无条件）、`GET /health/ready`（检查 SQLite/LLM/MCP 依赖）、`/health/rust`、`/health/guidance` |
| 会话线程 | conversation 生命周期 | `GET/POST /threads`、`GET /threads/{id}`、`GET /threads/{id}/messages` |
| 实时事件 | 订阅全局 EventBus | `GET /events/stream`（SSE）、`GET /ws/agent`（WebSocket 实时 agent chat） |
| 配置/凭据 | 运行时配置、密钥 | `GET /config`（api key）、`POST /config`（admin key）、`/credentials` |
| 工具/技能 | 注册工具与技能 | `/tools`、`/skills`、`/catalog`、`/tool_search` |
| Agent 执行 | 跑 agent | `/execute`、`/explore`、`/coder`、`/agents`、`/autoloop`、`/workflows`、`/execution`、`/diagnose`、`/plan` |
| 研究项目 | 项目与知识 | `/project`、`/research_project`、`/planner`、`/codebase`、`/knowledge`、`/kg`（图谱 stats/graph/query/search/mermaid） |
| 记忆/知识 | 检索、溯源 | `/memory`、`/search`、`/provenance`（文件产出谱系/全文搜索）、`/data_dict` |
| 文档/视觉 | 理解与感知 | `/document`（PDF/DocGraph/信息包）、`/visual`（I-JEPA 图像编码/检索） |
| 用户/管理 | 多用户、运维 | `/users`（list/create/patch/rotate-key/deactivate/delete）、`/admin`（maintenance mode）、`/system`、`/metrics`（Prometheus） |
| HPC/远程 | 远程执行 | `/hpc`、`/tunnels`（SSH 隧道）、`/terminal`（远程终端）、`/transfer`（文件传输） |
| 导出/分享 | 打包迁移 | `/export_share`、`/transfer` |
| 3D 查看 | 实时分子视图 | `/viewer3d`（load/trajectory/elements）+ `WS /ws/viewer3d` |
| 其他 | 内核/交互/侧栏/机器人 | `/kernel`（ipykernel 会话）、`/interaction`（SSE/中途干预/主动提问）、`/side`（侧边并行 Q&A）、`/bot`（OneBot v11 QQ/WeChat）、`/checkpoints`、`/params`、`/diagnostics` |

**鉴权分层**：`/auth/*` public（自带凭据流程）；普通 API key（`require_api_key`）保护大部分业务路由；`/users/*`、`/admin/*`、`POST /config` 走**管理员 key**（`require_admin_key`）。返回统一错误信封 `huginn_error_response`（含 `request_id`）。

> API 版本化：新调用一律走 `/v1/...`。中间件对根路径请求加 `Deprecation: true` 头 + `Link: </v1/...>; rel="successor-version"`，提示客户端迁移。

## 15. 桌面应用

桌面是 Tauri v2 + React 的图形界面（Windows 发行版直接安装 exe）。

- **Chat**：与 Agent 实时流式对话（WebSocket）。
- **Tools**：浏览并调用已注册工具，自定义 JSON 参数。
- **Skills**：浏览并执行声明式技能（如 `standard_dft`、`elastic_constants`、`phonon_calculation`）。
- **Memory**：会话概览与长期记忆。

**端口鲁棒性机制**（本项目重点设计）：
- 后端端口采用**动态分配 + 存活校验**：`get_backend_port` 读取端口文件前先做 TCP 存活探测，stale 端口会回退到本进程分配的计划端口。
- pet 窗口以 **30s 轻量轮询**：每次（重）连前重新从 `backend_port` 文件拉取端口，检测到端口变化就断开重建 SSE/WS 连接，覆盖"连接看似健康但后端已换端口"的场景。

前端连接：WS `ws://localhost:8000/ws/agent`（chat）、HTTP `http://localhost:8000`（tools/skills/health），可用 `VITE_WS_URL` 覆盖。

## 16. 技能（Skills）与预设

声明式工作流（`agent/huginn/skills/` + `presets.py`），12+ 预设：

| 预设 | 用途 |
|---|---|
| `standard_dft` | 标准 DFT + static 计算 |
| `aimd` | 从头分子动力学（ab initio MD） |
| `defect_calculation` | 点缺陷形成能 |
| `surface_calculation` | 表面能与 slab 模型 |
| `lammps_melt_quench` | Melt-quench 玻璃生成 |
| `ml_potential_training` | 训练 ML 原子间势 |
| `band_gap_analysis` | 不同泛函带隙 |
| `elastic_constants` | 弹性常数计算 |
| `phonon_calculation` | 声子 DOS 与色散 |
| `convergence_diagnosis` | 自动收敛排查 |
| `high_throughput_screening` | 批量性质筛选 |
| `symbolic_regression_discovery` | 数值中发现解析关系 |

另有 `.md` 声明式技能（如 `band_structure.md`、`structure_relaxation.md`、`wavefunction_analysis.md`）。

## 17. 记忆系统、知识蒸馏与视觉/时空工作流

### 17.1 三层记忆（`agent/huginn/memory/`）

- **Session 记忆**：当前对话上下文；模型 `reasoning_content`（COT）经 `add_reasoning` 持久化到 `reasoning_trace`，作为下游蒸馏的原材料资产。
- **长期记忆**：SQLite + FTS5 全文检索 + 重要性打分。
- **自动提升**：`MemoryManager` 协调 session 与 long-term，把重要 session 数据自动晋升到长期存储。

### 17.2 知识蒸馏（`evolution/knowledge_distiller.py`）

不同于模型微调（改权重），Huginn 的蒸馏是把执行经验沉淀为**可检索的 RAG 知识**，闭环："执行日志 → 模式提取 → 知识条目 → RAG 摄入"。

**六类蒸馏产物**（`DistilledKnowledge.source_type`）：

| 类型 | 触发条件 | 内容示例 |
|---|---|---|
| `error_lesson` | `distill_error_lessons(failure_logs)` | 从失败里提炼"用什么软件算 X 撞了什么错、怎么绕" |
| `success_pattern` | `distill_success_patterns(success_logs)`（同组 ≥2 条） | 归纳成功计算的公共参数组合 |
| `tool_tip` | `distill_tool_tips(tool_logs)`（同工具 ≥3 条且成败并存） | 对比同工具成败，提炼使用技巧 |
| `domain_fact` | `distill_domain_facts(conversations)` | 从成功对话里提炼领域事实 |
| `feynman_note` | 主动请求 | 用通俗语言重组知识，暴露理解缺口（Feynman 学习法） |
| `visual_lesson` | 视觉工作流（见 17.5） | 从视觉原语中蒸馏视觉经验 |

**验证与元蒸馏（HiSME）**：
- 每条知识带 `verification_status ∈ {unverified, confirmed, rejected}`。被检索并成功使用后，`MemoryManager._verify_distilled_for_tool` 调 `verify_knowledge` 将其提升为 `confirmed`（usage_count+1，confidence+0.1）。
- **元蒸馏**：`evaluate_meta_knowledge_rules` 读回统计，施加维护决策——confirmed 且复用 ≥3 次、confidence ≥0.8 的知识 → `promote`（升级为稳定原则候选）；调了 ≥2 次但 confidence <0.4 → `flag_low_value`。等于系统在学**"怎么维护知识"**。
- **去重**：md5 防完全相同；Jaccard 词集重叠 ≥0.65 判语义重复（`_is_semantically_duplicate`），避免措辞不同内容相同的教训重复入库。
- **KB 回写闭环（F6）**：每次 `_save` 把新增条目 `add_text` 写回主 KB（带 `source_type`/`confidence`/`distilled=1` 元数据），让 agent 检索得到结构化知识——蒸馏结果不再只躺 JSON 文件。

### 17.3 外部思维（External Thinking，`deep_think` 外部草稿纸）

厂商普遍隐藏原生思维链（chain-of-thought）后，模型"愿意"暴露的 `reasoning_content` 并不总是有。External Thinking 的思路来自 oh-my-pi 的 `externalThinking`：**给模型一个普通工具，让它在动手前把分析写进工具参数**——工具参数经 API 返回，开发者能直接读取保存。Huginn 把它做成了可一键开启的正式能力，是 17.1 里 `reasoning_trace` 的另一条注入通道。

**核心工具 `deep_think`**（`tools/deep_think_tool.py`）：`read_only=True`、无副作用，输入 `analysis: str`。开启后系统提示要求模型在**回答问题、改代码、调其他工具之前**，先调用 `deep_think` 把逐步分析写进去。工具执行时经 `memory_manager.add_reasoning` 写入 `session.reasoning_trace`——与原生 `reasoning_content` 捕获**共用同一条蒸馏通道**，且不把分析内容回显给 LLM（避免重复占用上下文）。

**补充通道策略**：Huginn 对接的 provider 很杂（OpenAI-compatible / Anthropic / Ollama / 国产模型），统一强制关原生推理（`forceReasoningOff`）不现实，所以 deep_think 是**补充**而非替代——`deep_think` 拿显式草稿、`reasoning_content` 拿原生推理，两路都汇入 `reasoning_trace`，蒸馏闭环统一消费。

**开关**（`external_thinking` feature flag，默认关），三层开启：

| 方式 | 写法 |
|---|---|
| 配置文件 | `huginn.toml` 的 `[feature_flags]` 里 `external_thinking = true` |
| 环境变量 | `HUGINN_FEATURE_EXTERNAL_THINKING=true` |
| 运行时 | `FeatureFlags.enable("external_thinking")` |

开启时由 `_thinking_plugin`（"thinking" 段，priority 100）注入指令，关闭时不注入、默认行为不变。fail-open：flag 层异常返回空串，prompt 构建永不崩。与 `model_tier` 联动：`full` / `balanced` 档默认开，`minimal` 档默认关（可按需开）。

**结构化推理协议（深化）**：`memory/reasoning.py` 把被动捕获（自由文本 → 扁平字符串）升级为**结构化推理**——每条推理是一条 `ReasoningRecord`：

- **字段**：`claim`（核心论断）、`phase`（think → plan → pre_action → reflect 阶段化编排）、`evidence`（依据）、`estimate`（量化预估，带单位/范围）、`uncertainty`（边界条件）、`plan`（下一步）。
- **自校验闭环**：执行后回填 `outcome ∈ {confirmed, refuted, partial}`，`last_pending()` 从最近的 pre_action/plan 记录开始回填——预估在执行后被验证，形成"预测 → 对照 → 回映"的地基。
- **可蒸馏信号**：`is_distillable` = confirmed 且带 claim + estimate。**一条被验证过的 pre_action 量化预估，是留给蒸馏闭环的最强信号**——它直接成为 17.2 里可复用知识的候选。

结构化通道是**新增侧信道**，不破坏扁平 `reasoning_trace`（原生 reasoning_content + 旧 deep_think 仍写那里），下游蒸馏/反思消费方可平滑迁移。

### 17.4 三库存储隔离

memory（时序）/ knowledge（向量）/ kg（图拓扑）各自用原生结构存储，只在自然边界跨库翻译、保结构（见第 3 节）。

### 17.5 视觉 → 知识库 → 记忆闭环（`visual_hook.py`）

视觉不是旁观功能，而是一条注入记忆与推理的感知通道。核心是 **`extract_visual_primitives`**：把工具输出的数值/坐标抽成带结构化标签的原语，让 LLM 能精确引用图像细节，而不是"看图说一嘴"。

**原语类型**：
- `<point>[x,y]</point>(label)` / `<point3d>[x,y,z]</point3d>(label)`：2D/3D 坐标锚点，label 可带物类名（如 `Fe`、`atom0`）。3D 坐标兼容 **SE(3) 旋转**（旋转后坐标可为负也能解析），为"把分子旋转、在坐标系里重新定位"这类空间操作留了工程入口。
- 度量对提取（`_extract_metric_pairs`）：从 stdout 正则抓 `key: value`，用于曲线/图对比。
- 比较原语（`extract_comparative_primitives`）：baseline vs candidate 的成对比较。
- 数据置信度（`_estimate_data_confidence`）：附带产出可信度估计。

**闭环链路**（`SPEC_visual_kb_loop.md`，G4–G10 已落地）：

```
extract_visual_primitives
  ├→ RAG KB (add_text, content_type=visual_primitives)   G4
  │     → PMK build_pmk_state: 立场检查的 attempted 含"视觉"段  G6
  ├→ hippocampus record（视觉记忆）→ visual_inspect 时注入 prior  G5
  └→ knowledge_distiller.distill_visual_lessons → 视觉经验回写 KB  G8
```

配套一个 **hippocampus（海马体）记忆**：`visual_inspect` 做视觉检查时，会把 `visual_memory_prior` 注入给 LLM——"你之前见过的同款结构/谱长这样"。SE(3) 旋转产物也会 record，跨 session 召回。

视觉路径还包括 **mental_imagery**（`metacog/mental_imagery.py`）：`mental_imagery_loop(spec)` 做"空间结构想象 sketch → verify 校验"复合，已接入 autoloop 的 `engine_observe` 观察阶段（`engine_observe.py`），生成时空草图注入 prompt。默认按开关（`HUGINN_USE_HIPPOCAMPUS=1`）或非阻塞。

**文本模型也有视觉能力（CV_TOOLS 降级链路）**：`vision/router.py` 的 `VisionRouter` 对所有模型统一处理图像。原生多模态模型（如 GPT-4o/Claude/Gemini，`vision` 能力为真）走 `BOTH` 路径——LLM 直接看原图 + CV 预分析并行注入定量提示。**纯文本模型**（`vision` 能力为假，如本地 qwen2.5、deepseek 文本版）也**可以处理图像**，走 `CV_TOOLS` 路径：Huginn 用数字视觉管道把图"翻译"成文本给文本 LLM，链路包括——

1. **CV 预分析**（`_cv_pre_analyze`，~50ms 无 LLM 成本）：直方图统计、边缘密度、自动猜图像类型（SEM/TEM/XRD/EDS）
2. **visual→symbols 结构化提取**（`symbol_encoder`）：单次提取图表数据，同时产出文本版与结构化 JSON（agent 可精确引用字段 + 看 self_check 判断可信度）
3. **显微图定量**：SEM/TEM/EDS 调对应分析 action，把 metric 压成一行摘要
4. **视觉记忆**：视觉编码器把当前图编码后去图像索引搜相似图，给文本 LLM"这像你上次看过的 SEM 图"的前置
5. **显式路由决策**：native vision 不可用时，把"① 委托视觉 agent / ② 用视觉记忆 / ③ 调 image_analysis_tool"三个选项显式给 LLM，避免它瞎猜

所以"某模型 vision=false"只表示它**没有原生多模态输入**，不代表 Huginn 拿图像没办法——纯文本模型照样能"看图"，只是经由上述数字链路理解图像。这也是 Material 科研场景里关键能力：即便只有本地文本模型，也能分析 SEM/TEM/XRD 图谱。

### 17.6 时空可组合工作流（`security/revertible.py` + `workflows/engine.py`）

科研 agent 的每个副作用都要能**撤销**，才能真放心让它自主尝试。Huginn 把"时空可组合性"做成了可逆副作用栈：

- **`RevertibleContext`**：LIFO 逆栈。每个操作 `track/dispose` 一个逆操作；`revert_all()` 逆序回滚全部副作用。
- **事务**：`ctx.transaction()` 把一段多操作包成原子单元，异常时整体回滚。
- **数据驱动逆**：`track_op` 记录操作 + 应用逆函数；到 `_apply_inverse` 自动执行。
- **崩溃重放**：journal 落盘，`recover_from(journal_path)` 在进程崩溃后按日志重放、补全未完成的回滚。
- **出站补偿**：`compensate(kind, payload)` + `register_compensator`——把"调用外部 API"这类无法本地撤销的操作，用注册好的反向调用补偿。
- **物理世界逆**：`track_world_action(state_before, action, inverse)` 把物理实验动作也纳入撤销模型。
- **文件系统 undo**：`snapshot/file_snapshot.py` 的 revert/unrevert 独立于内存栈，提供整目录快照级撤销。

`workflows/engine.py` 用 `composite()` 把多 stage 编排成一个可逆的原子单元——**stage 之间任一失败，整体回到执行前状态**。这是"让 agent 敢大胆试错"的能力基础。

### 17.7 接入 Embedding 模型（知识库的向量引擎）

知识库（RAG KB）靠 embedding 模型把文档和查询映射到同一向量空间再做相似度检索，Embedding 模型是 `knowledge/store.py` 的底座，独立于第 12 章的 LLM Provider（Embedding 走 `sentence-transformers`，不消耗 LLM API）。

**默认模型**（`store.py` 的 `EMBED_MODEL`）：

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2（384 维，多语言）
```

选它是因为**多语言 + 轻量**：中英材料术语都能编，384 维向量内存占用小，桌面侧car 冷启动/内存压力都可控。曾尝试更强的 `BAAI/bge-m3`（1024 维，中英更准），但 ~2.2GB 权重在桌面环境下载/解码不稳、内存压力偏大，故退回到 MiniLM。**若你的语料偏单一语言且想要更强语义，可自行覆盖**。

**桌面版零操作（自动）**：python-runtime + `sentence-transformers` 都随安装包内置；首次用到知识库时自动下载权重（~470MB，见下方离线说明），无需手动装依赖或设环境变量。下面"三步"只针对想覆盖默认模型/离线分发的开发者。

**开发/覆盖三步**：

1. **安装依赖**：`sentence-transformers>=2.5`（KB 本身还需 `chromadb>=0.4`）。两处依赖声明在 `agent/pyproject.toml` 的 `[all]` / `rag` extra。

2. **一次性下载权重**（首次使用触发，~470MB）：默认从 HuggingFace 拉取，国内网络建议先设镜像再运行：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
   python -m sentence_transformers.models    # 任意触发生成即可预取
   ```
   离线环境需在联网机器下载权重到 HF 缓存目录（`~/.cache/huggingface/hub`），随安装包一并分发（打包时放入 `python-runtime`，设 `HF_HUB_OFFLINE=1`）。

3. **覆盖默认模型（可选）**：
   ```bash
   export HUGINN_EMBED_MODEL=BAAI/bge-m3      # 换成其他模型
   python agent/rebuild_kb.py --yes ~        # 重编码重灌存量向量
   ```

**⚠️ 换模型必须重建 collection**：不同模型的向量维度/语义空间不同，否则查询向量与存量向量"维度一样但模型不同、语义错位"，检索命中反而变差。用 `agent/rebuild_kb.py [workspace]` 迁移（先备份 → 读出源文本 → 删 collection → 用新模型重灌 → 校验条目数）。桌面 App 的默认工作区在主目录 `~/.huginn_kb`，运行前先退出桌面 App 释放文件占用。

**降级链路**（`store.py` 的 `_EmbeddingModel.encode`）：正常走 `sentence-transformers`（unit 向量，`normalize_embeddings=True` 与重灌存量对齐）；ST 不可用/挂起时退 `chroma` 自带 ONNX；再异常给确定性哈希向量兜底，保证 KB 流程不阻塞。**切换模型后务必重建 collection**，否则查询落在旧语义空间，混合检索（向量 + BM25 RRF）的排序会失真。

### 17.8 文档文字提取与 OCR 优先级（`knowledge/ocr_loader.py`）

扫描件 / 无文字层 PDF 入库前要先做文字提取，Huginn 的 OCR 链路是 **LLM-as-OCR 首选、无 LLM 才回退传统 OCR**：

**图像（单张）**：LLM-as-OCR → PaddleOCR（中文/混排强）→ EasyOCR → Tesseract。
**PDF**：逐页渲染成图后按上图链逐页提取；逐页全空时 Nougat 再整体兜底一次（保留数学/公式结构）。

- **LLM-as-OCR**（`_llm_ocr_image`，DeepSeek-OCR 思路）：解码器就是多模态 LLM，不走独立 OCR 模型。`server_core` 启动时若 agent 模型支持 vision 会注入 `_vision_decode` callback，`set_llm_vision_callback` 注入；对扫描件公式/表格/中文混排比传统 OCR 强。
- **无 vision / 无 LLM 时**：自动回退传统 OCR 链，功能不缺失；`engine=llm` 强制只走 LLM，`HUGINN_OCR_ENGINE` 可显式指定某引擎。
- **开关门控**：`llm_vision_available()` 判断是否已注入 callback；`smart_ingest` 在无 vision 时跳过整页视觉压缩，避免白渲染/白存图。

## 18. Coder / 多 Agent / 自主科研（autoloop）

### Coder 模式

```bash
# 一次性任务
huginn-agent coder "Add a docstring to huginn/tools/code_tool.py"

# 交互式
huginn-agent coder

# 自动批准破坏性动作（慎用）
huginn-agent coder "Refactor the CLI" --auto-approve
```

Coder 工具：`file_read_tool` / `file_write_tool` / `file_edit_tool` / `bash_tool` / `git_tool` / `code_tool`。

### 多 Agent

多 Agent 有三套协作模型（`agents/`）：

- **Orchestrator（编排）**：lead agent 把目标分解为结构化 `TaskPlan`/`SubTask`，多个 sub-agent **并行执行**，synthesizer 汇总为 `OrchestratorResult`。plan 可持久化到 plan store，重启可续。
- **SubagentDispatch（卸载）**：把长运行/上下文沉重的任务**卸载到隔离的 sub-agent session**，避免污染主对话上下文，结果摘要回传。声明式 `SubagentSpec` 控制工具预算与递归深度，防失控。
- **Team（多模型团队）**：按角色编排，角色包括 `PLANNER / SCIENTIST / CODER / EXECUTOR / CRITIC / VISION / SYNTHESIZER`，每个角色声明对推理/工具/视觉的能力要求（`ROLE_REQUIREMENTS`），用不同模型承担不同角色。
- `swarm` / `team` CLI 入口，`loop_detector` 防循环。

对应的工具：`orchestrate_tool`、`subagent_tool`、`review_committee_tool`。

**因果能力**（`causal/`，多 Agent 科研的下层支撑）：基于 `VisualSCM` 做 Pearl do-calculus 的 **L2 干预预测**（`predict_intervention` 按 SCM 拓扑序采样、`do(X=v)` 替换常数节点、与基线对照），以及从观测拟合 SCM、反事实渲染（`predict_intervention` / `FitSCMFromObservationsTool` / `CounterfactualRenderTool`）。接入了 autoloop / conjecture / red team 流程——问的不是"相关性"，而是"如果我改某个量，结局会变多少"。

### 自主科研（autoloop）

`huginn-agent autoloop <目标>`：以 objective 为驱动，经由 7 phase（perceive→hypothesize→plan→execute→validate→learn→report）自主迭代，Pareto 剪枝 + 失败诊断/回溯，最后返回最优方案。

## 19. 安全

- 统一错误信封（`huginn_error_response` + `request_id`）；所有 API 端点由 `require_api_key` 保护。
- AES-128-CBC + HMAC-SHA256 静态加密，逐条 salt；解密密钥仅存内存、从不落盘。
- Fail-closed 工具元数据（`is_read_only` / `is_destructive` / `requires_confirmation`）。
- 破坏性工具默认在容器沙箱内执行；命令白名单 + 超时 + 输出上限。
- 日志与配置转储默认屏蔽密钥；审计日志 append-only 且 hash-chained。
- **密钥分层**：用户服务凭据（可前端 `/credentials` 配置）与 operator 级密钥（仅 env / 密钥管理器注入）隔离。
- 物理预检：`PRE_TOOL_USE` 钩子检测物理不合理配置并警告，不永久阻断（可用 `force_proceed=True` 强制继续）。
- 沙箱多层防御：AST 预扫描 → 命令过滤 → 声明式策略引擎 → subprocess 软沙箱 → Docker 容器沙箱 → CLI 容器执行器 → 受限 Python 执行器。

**生产环境必须设置强随机密钥**——这里的 key 是 Huginn 自己后端（`huginn.server`）的访问门禁 `HUGINN_API_KEY`/`HUGINN_ADMIN_API_KEY`，用于保护你的服务本身（所有 HTTP/WS 端点由 `require_api_key` 拦截），与前面的"LLM 服务商凭据"是两回事。空 key 意味着服务无鉴权、任何能连到端口的人都能调用；弱 key 容易被暴力猜中。用强随机串生成：

```bash
openssl rand -hex 32
export HUGINN_API_KEY=<上面生成的串>
export HUGINN_ADMIN_API_KEY=<另一个强随机串>
```

威胁模型与事件响应见 `docs/threat_model.md`。

## 20. 监控、部署与运维

- 部署：`agent/DEPLOYMENT.md`；监控：`agent/MONITORING.md`；CI 见 `.github/workflows/{ci,desktop,release}.yml`。
- MCP 服务器（`servers/`）：mat-db / math-anything / vision-pixel，在仓库根 `.mcp.json` 配置，启动时经 `mcp_adapter.py` 内部注册，无需手动 `python servers/...`。
- 桌面圆桌流水线：desktop.yml（pre-release）与 release.yml（`v*` tag → stable）。发布产物自动附带 Python 运行时 + OCR 依赖。
- 监控告警示例见 `monitoring/alerts.yml`。

## 21. 编程扩展

### 新增一个工具

1. 在 `agent/huginn/tools/` 新建继承 `HuginnTool` 的类；
2. 定义 `name` / `description` / `input_schema`（Pydantic）；
3. 实现 `_execute()`，`call()` 返回 `ToolResult`；
4. 在 `agent/huginn/tools/__init__.py` 的 `_CORE_MODULES` / `_OPTIONAL_MODULES` 登记；
5. 在 `agent/tests/` 加测试。

### 新增一个技能

1. 在 `agent/huginn/skills/presets.py` 定义 `SkillDefinition`；
2. 添加引用工具名与参数的步骤；
3. 自动经 `agent.list_skills()` 可用。

### 新增 MCP 工具

遵循"Everything is a Plugin"：工具通过 ToolRegistry/ModelRegistry 可插拔按名注册，MCP 工具经适配层合并进最终工具池。

### 质量守则

```bash
cd agent
ruff check huginn tests
black --check huginn tests
mypy huginn
pytest tests/ -q
```

> 测试注意：用 `app_client` fixture（勿在模块顶层裸建 `TestClient`）；I/O 密集优先 `async`。改动架构时同步更新 `docs/tech-spec.md` 与 `docs/architecture.md`，并在 `docs/INDEX.md` 登记/更新文档。契约文档（env/flags/tools/routes/errors/permissions/events）由代码自动再生：`python -m huginn.cli.config_audit --<domain> --out docs/<domain>-contract.md`。

## 22. 常见问题排查

| 症状 | 处理 |
|---|---|
| 桌面状态栏显示 Backend offline | 先启动后端 `huginn-agent serve` 或 `python -m huginn.server` |
| pet 窗口断联 / 连接被拒 | 清理 `%USERPROFILE%\.huginn\backend_port` 与残留 Python 进程；新版本有端口存活校验 + 30s 轮询自动恢复 |
| Windows 构建报 `dlltool.exe not found` | 把 MinGW bin 目录加入 PATH |
| Cargo 构建报错 | 在 `src-tauri` 下 `cargo clean` 重试 |
| 本地工具 ImportError（打包版） | 确认打包时用 `pip install ".[all,ocr]"`；可选依赖默认懒加载，缺了才报错 |
| `import numpy` 失败 | 多为中断安装残留的二进制残渣，重装依赖环境即可 |

---

*本文档基于 Huginn v1.3.x 源码、架构与契约文档整理。完整文档导航见 `agent/docs/INDEX.md`（active/staging/report 状态标注）。*