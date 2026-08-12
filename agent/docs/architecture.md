# Huginn Architecture

> 本文件为架构导览，反映当前真实代码结构。最新且完整的技术事实以
> [docs/tech-spec.md](tech-spec.md)（现状事实记录）为准；本文件聚焦
> 模块职责与数据流，不再复述每个端点/工具清单。

## Overview

Huginn 是一个模块化的 LLM 驱动材料科学 agent 系统。覆盖 DFT（VASP/QE/CP2K/
Gaussian/ORCA）、分子动力学（LAMMPS/GROMACS/OpenMM）、CFD/FEA（OpenFOAM/
COMSOL/ABAQUS/FEniCS）、符号回归、RAG 文档检索、因果建模、加密数据管理与
自主探索工作流。包名 `huginn-agent`，v0.2.0，MIT。

## System Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Entry: CLI (huginn.cli:main) │ API (huginn.server)          │
│  CLI: chat/coder/explore/autoloop/serve/...  ~40 commands     │
│  API: FastAPI + WS/SSE, all routers under /v1 (+ root compat) │
├──────────────────────────────────────────────────────────────┤
│  Agent 层:                                     │
│   agent/     core loop, session, streaming(COT), reflection   │
│   agents/    orchestrator, subagent, swarm, team, speculator, │
│              loop_detector, tool_call_router, tool_dedupe     │
├──────────────────────────────────────────────────────────────┤
│  Capability 层:                                               │
│   tools/     ~178 tools + ToolRegistry + safety metadata      │
│   skills/    声明式工作流(presets + .md)                      │
│   memory/    session / long-term / manager (3-tier)           │
│   evolution/ knowledge distiller + evolution manager          │
│   knowledge/ KB + auto-ingest；kg/ 知识图谱                   │
│   causal/    SCM 生成/干预/反事实；autoloop/ 自主探索闭环      │
├──────────────────────────────────────────────────────────────┤
│  Runtime / Meta:                                              │
│   metacog/   元认知、自省、决策；runtime/ 任务生命周期         │
│   events/    事件总线、审计日志；persistence/ 检查点/状态      │
│   security/  auth、middleware；hpc/ 远程执行；api/ 契约层      │
└──────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### 1. Agent Loop (`huginn/agent/`)

- `core.py`：agent 主循环，`start()/step()/process_action()`，状态流转与回调。
- `session.py`：会话与任务上下文，`get_session()`。
- `streaming.py`：流式输出。AIMessage 上的 `reasoning_content`（如 DeepSeek
  COT）经 `memory.add_reasoning(...)` 写入 `session.reasoning_trace`，供下游
  蒸馏消费。
- `prompt_builder.py` / `context.py` / `reflection.py`：提示词组装、上下文注入、
  反思。

### 2. Multi-Agent (`huginn/agents/`)

- `orchestrator.py`：多 agent 编排，`run_agent()` / `handle_subagent_output()`。
- `subagent.py`：`SubAgent` 封装子 agent 执行与输出。
- `swarm.py` / `team.py`：swarm / team 协作方式。
- `tool_call_router.py` / `tool_dedupe.py` / `loop_detector.py`：工具调用路由、
  去重、循环检测。

### 3. Tools (`huginn/tools/`)

- `registry.py`：`ToolRegistry` 提供 `register/get/list_tools/unregister/clear/
  snapshot/restore`；全局注册表在 `ToolRegistry._tools`。
- `base.py`：`HuginnTool` 基类（`name/description/input_schema`，`call ->
  ToolResult`，子类实现 `_execute`）。
- `defaults.py`：`ToolMetadata`（`is_read_only/is_destructive/requires_confirmation`，
  默认 fail-closed）。
- `__init__.py`：`_CORE_MODULES`（~35 轻量）+ `_OPTIONAL_MODULES`（重依赖仿真）
  的 `(module, ClassName)` 列，`register_core_tools/register_optional_tools/
  register_all_tools` 注册并重建 phase/router/constraint 调度表。
- `assembly.py`：合并/过滤/排序内置与 MCP 工具，构建最终工具池。

### 4. Memory (`huginn/memory/`)

- `session.py`：会话级记忆，含 `add_reasoning`（COT 资产化）。
- `longterm.py`：长期记忆，SQLite + FTS5 全文检索 + 重要性打分。
- `manager.py`：`MemoryManager` 协调；`_verify_distilled_for_tool` 解析
  `source="distiller:{id}"` 并调用 `KnowledgeDistiller.verify_knowledge` 把
  成功使用过的蒸馏知识升级为 `confirmed`。

### 5. Evolution (`huginn/evolution/`)

- `knowledge_distiller.py`：从执行日志蒸馏知识；`verification_status ∈
  {unverified, confirmed, rejected}`；`verify_knowledge(knowledge_id, status)`
  升级状态；`auto_ingest_to_kb` 仅吸收 `confirmed` 条目。
- `manager.py` / `engine.py`：进化管理与闭环。

### 6. Knowledge & KG (`huginn/knowledge/`, `huginn/kg/`)

- `knowledge/auto_ingest.py`、`domain_pipeline.py`、`chunker.py`：知识入库管道。
- `kg/`：知识图谱（`builder/extractor/query`）。

### 7. API Server (`huginn/server.py` + `huginn/routes/`)

- FastAPI 应用，`include_v1_routes(app, keep_root_compat=True)`：所有路由挂 `/v1`
  前缀并保留根路径兼容。全应用 `Depends(require_api_key)`，统一错误信封
  `huginn_error_response`（含 `request_id`）。
- 端点族：health、threads、execute/explore/diagnose/plan、workflows、team/v2、
  auth、credentials、mcp、transfer、projects、export/import、events/stream(SSE)、
  ws/agent(WS)、data、bench/evolve/autoloop、live、advisor、users 等。
- 中间件链：限流 → metrics → 请求体大小/超时 → request-id → 错误归一化 →
  维护模式。

### 8. Runtime / Meta (`huginn/runtime/`, `huginn/metacog/`)

- `runtime/`：任务生命周期、调度、trace 上下文、前置 grill、sanity gate。
- `metacog/`：自我模型、自省、盲区映射、决策仲裁、拓扑视角等元认知能力。

## Data Flow

```
User Query
  → CLI / WS / HTTP
    → HuginnAgent loop (agent/core.py)
      → prompt_builder + memory injection
        → LLM reasoning (streaming.py 捕获 reasoning_content → reasoning_trace)
          → tool_call_router → tool execution (local / MCP / HPC)
            → ToolResult → session + long-term memory
              → 成功工具调用 → _verify_distilled_for_tool → 蒸馏知识 confirmed
                → auto_ingest_to_kb → 知识库
      → response (streaming / SSE / WS)
```

对于自主探索（autoloop）：
```
Objective → 分支生成 → 评估 → Pareto 剪枝 → 失败诊断/回溯 → 返回最优方案
```

## Design Principles

1. **Graceful degradation**：每个组件有 mock/fallback 模式，便于无完整基础设施时开发。
2. **Security by default**：fail-closed 工具元数据、内存仅存密钥、per-item salt。
3. **Modularity**：组件可独立使用。
4. **Type safety**：Pydantic 模型统一输入/输出。
5. **Testability & anti-regression**：session 级共享 app + module 级 TestClient
   上下文管理器；autouse 护栏强制 `ToolRegistry` 测试间逐位一致；CI 覆盖
   3.11/3.12/3.13，覆盖率门禁 60。

## Configuration

- `pyproject.toml`：依赖与元数据；`huginn.toml` / `huginn/config.py`：运行时配置。
- 环境变量：`HUGINN_ENV`、`HUGINN_HIDE_DOCS`、`HUGINN_RATE_LIMIT_PER_MINUTE`、
  `HUGINN_MAX_BODY_SIZE_MB`、`HUGINN_REQUEST_TIMEOUT_SEC`、`HUGINN_DEV_MODE`、
  `HUGINN_API_KEY` 等。
- 桌面端：Tauri v2 + React 18（WIP）。

## Development Guidelines

- 新增功能在 `tests/` 加测试；测试用 `app_client` fixture（勿在模块顶层裸建
  `TestClient`）。
- 工具返回统一用 `ToolResult`；新增工具在 `huginn/tools/__init__.py` 登记。
- I/O 密集操作优先 `async`。
- 改动架构时同步更新 `docs/tech-spec.md` 与本文档。