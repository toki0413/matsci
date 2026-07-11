# Huginn 整体架构

Huginn 是一个面向计算材料科学的智能体系统，核心围绕 `HuginnAgent` 构建，通过 LangGraph/deepagents 驱动 ReAct 循环，结合 130+ 专业工具实现自主科研。

## 层次划分

系统自上而下分为五层：

1. **入口层** — CLI (`cli/`) 与 FastAPI Server (`server.py`, `server_core.py`)，提供交互式终端和 WebSocket/REST API 两种接入方式。`ServerContext` 持有所有长生命周期对象（配置、注册表、Agent 工厂、记忆、知识库）。

2. **Agent 核心层** — `HuginnAgent` (`agent/core.py`) 由五个 Mixin 组合而成：
   - `ContextMixin` — 系统提示组装、工具过滤（模式+阶段+查询相关性三层过滤）
   - `StreamingMixin` — 异步聊天流、阶段管理、上下文压缩
   - `SessionMixin` — 会话状态、跨会话连续性
   - `CallbackMixin` — LLM 回调
   - `ReflectionMixin` — 任务反思

3. **上下文构建层** — `ContextBuilder` (`context_builder.py`) 在每轮 LLM 调用前组装动态上下文：长期记忆召回、知识图谱查询、领域知识库检索、情绪追踪、对话历史、计划进度。

4. **执行引擎层** — 两条并行管线：
   - **Autoloop Engine** (`autoloop/engine.py`) — 七阶段自主循环：Perceive → Hypothesize → Plan → Execute → Validate → Learn → Report，由 GoalScheduler、PhaseGate、Budget 驱动。
   - **Deli AutoResearch** (`academic/deli_research.py`) — 多智能体学术研究管线：文献检索 → Gap 分析 → 大纲 → 起草 → 引用验证 → 同行评审 → 修订，每个阶段设有 integrity gate。
   - `ResearchWorkflow` (`research_workflow.py`) 将 Deli 管线桥接到 WebSocket 事件流。

5. **基础设施层** — 提供持久化与检索能力：
   - **Memory** (`memory/`) — `MemoryManager` 统一协调会话记忆与长期记忆，支持自动提升与语义搜索
   - **Knowledge Base** (`knowledge/`) — ChromaDB 向量检索，含 36 个材料科学种子知识文件
   - **Knowledge Graph** (`kg/`) — 项目级知识图谱，实体抽取 + 关系查询
   - **Tools** (`tools/`) — `ToolRegistry` 集中注册，`HuginnTool` 基类提供权限检查、输入验证、provenance 追踪
   - **Hooks** (`hooks/`) — 工具调用前后、会话生命周期、上下文压缩等事件钩子
   - **Personas** (`personas.py`) — 角色系统，内置 default/dft_expert/md_expert/reviewer/tutor/research 等角色
   - **Prompts** (`prompts.py`) — 系统提示词定义

## 横切关注点

- **Agents** (`agents/`) — 多智能体编排：工厂模式、Orchestrator、Swarm、Subagent、Team、CircuitBreaker
- **Routes** (`routes/`) — 50+ FastAPI 路由模块，覆盖 WebSocket、工具管理、记忆、知识、执行等
- **Models** (`models/`) — 模型路由与注册，支持多 provider（Anthropic/OpenAI/DeepSeek/Google/Ollama）
- **Skills** (`skills/`) — 声明式技能系统，可组合的工作流预设
- **Phases** (`phases.py`) — 研究阶段状态机：LITERATURE → HYPOTHESIS → PLANNING → EXECUTION → VALIDATION → REPORTING
