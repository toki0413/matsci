# 整体架构 — 关键设计决策与约束

## 核心设计决策

### 1. Mixin 组合而非继承
`HuginnAgent` 通过多个 Mixin 组合而非深层继承树来扩展功能。`ContextMixin`、`StreamingMixin`、`SessionMixin`、`CallbackMixin`、`ReflectionMixin` 各自独立，`core.py` 仅负责装配。这避免了 god-class 问题，同时保持单一入口。

### 2. ContextBuilder 抽离
上下文构建逻辑从 Agent 中抽离到独立的 `ContextBuilder` 类，使得记忆召回、KG/KB 查询、情绪追踪等可以独立测试和复用。Agent 所有上下文构建方法都委托给它。

### 3. 双引擎架构
系统有两条并行的执行管线：
- **Autoloop Engine** — 面向自主科研的全闭环（perceive→hypothesize→plan→execute→validate→learn→report）
- **Deli AutoResearch** — 面向学术写作的半闭环（文献→gap→大纲→起草→引用验证→评审→修订）

`ResearchWorkflow` 作为桥接层，将 Deli 管线的进度事件转换为 WebSocket 事件流。

### 4. AgentConfig 统一配置
所有配置通过 `AgentConfig` dataclass 结构化管理，分为 model/tools/memory/security/telemetry/context_budget/knowledge_graph/personalization/core 八个子配置。`from_env()` 从环境变量加载，`from_config()` 从外部配置对象加载。

### 5. ServerContext 单例容器
FastAPI 服务端的所有长生命周期对象集中在 `ServerContext` 中，避免模块级全局变量散落。包括 config、tool_registry、skill_registry、permission_config、audit_logger、memory_manager、kb、agent_factory、orchestrator 等。

### 6. 工具三层过滤
工具可见性经过三层过滤：
1. **模式过滤** — chat 模式隐藏重型仿真工具（vasp_tool, lammps_tool 等）
2. **阶段过滤** — 当前研究阶段的 tool_filter 限制可用工具集
3. **查询相关过滤** — 工具数超过 25 个时启用 query-aware 检索，只保留 top-15 相关工具

## 约束

- **LLM 依赖** — 核心推理依赖外部 LLM（Anthropic/OpenAI/DeepSeek/Google/Ollama），需通过 `ModelRouter` 统一管理
- **LangGraph/deepagents** — Agent 图构建依赖 `deepagents`（优先）或 `langgraph.prebuilt.create_react_agent`（降级）
- **工具注册时机** — 核心工具在 `ServerContext` 创建时同步注册，可选工具在 lifespan 中后台注册
- **会话 TTL** — 空闲线程 24 小时后自动清理，每 50 次 `get_or_create_thread()` 触发一次扫描
- **provenance 追踪** — 通过 contextvars 在 Autoloop 运行期间自动捕获每次工具调用的快照
- **权限模型** — fail-closed 默认：未显式声明只读的工具一律需要确认
