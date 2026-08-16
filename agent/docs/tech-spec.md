# Huginn 技术规格（现状事实记录）

> 本文件由 `praxis/es onboard` 生成，只记录代码中真实存在的事实，不含解释、计划或差距分析。目标：为已存在的代码库提供一份可维护的"living documentation"。以 `docs/architecture.md` 为历史参照，本文件反映当前目录结构的真实状态。

## stack

- 语言/运行时：Python `>=3.11,<4.0`（CI 覆盖 3.11/3.12/3.13）。
- Web 服务：FastAPI + uvicorn、sse-starlette、websockets、python-multipart、httpx、requests。
- Agent 框架：langchain `>=1.3`、langchain-core、langchain-openai、langgraph `>=1.2`、langgraph-checkpoint-sqlite（生产必需，见 `pyproject.toml` 注释）、deepagents。
- 前端/桌面：Tauri v2 + React 18（WIP），包名 `huginn-agent` v1.3.0，MIT。
- 数值/符号/数据：numpy、scipy、sympy、z3-solver、networkx、Pillow。
- 存储：SQLite + FTS5、ChromaDB（可选 extra `rag`/`all`）、toml（读写 TOML 配置）。
- 加密：cryptography（AES-128-CBC + HMAC-SHA256，PBKDF2）。
- 可选科学栈（extra）：pymatgen、ase、MACE/fairchem/NEP（互斥分组的 `ml-mace`/`ml-fairchem`/`ml-nep`，因 e3nn 版本冲突拆分）、spglib、matplotlib 等。
- 工具链：black（line-length 88）、ruff、mypy（渐进式）、pytest + xdist + cov + hypothesis、pre-commit。

## entry

- 命令行入口：console script `huginn-agent = huginn.cli:main`（`huginn/cli/main.py` 的 click group）。子命令注册于 `huginn/cli/commands/__init__.py` 的 `register_commands`，包括：`chat`、`coder`、`refactor`、`explore`、`autoloop`、`serve`、`tools`、`version`、`configure`、`bench`、`evolve`、`execute`、`workflow`、`diagnose`、`model-list`、`memory-maintenance`、`telemetry`、`seed-knowledge`、`encrypt-config`、`export`、`build-kg`、`api-keys`、`hpc`、`remote`、`scheduler`、`autoresearch`、`plot`、`unified`、`persona`、`swarm`/`team`、`visualize`、`sessions`、`skill-import`、`bg` 等。
- API 服务：`python -m huginn.server` → uvicorn 监听 `127.0.0.1:8000`，`ws_ping_interval=300`；端口写入运行时目录 `backend_port` 文件供前端发现。
- 包入口：`huginn/__init__.py` 导出 `__version__`。

## contract

- 工具注册表契约（`huginn/tools/registry.py`）：`ToolRegistry.register/get/list_tools/unregister/clear/snapshot/restore`；装饰器 `register_tool`。全局注册表维护在 `ToolRegistry._tools`。
- 工具基类契约（`huginn/tools/base.py`）：`HuginnTool` 暴露 `name/description/input_schema/output_schema`；标准入口 `call() -> ToolResult`（内含 provenance 捕获），子类实现 `_execute()`；可选覆写 `check_permissions/validate_input/estimate_cost`。
- 工具安全元数据（`huginn/tools/defaults.py`）：`ToolMetadata` 含 `is_read_only/is_destructive/requires_confirmation`，默认 fail-closed（不显式声明即为不安全）。
- 工具注册分层（`huginn/tools/__init__.py`）：`register_core_tools()`（约 35 个轻量工具，同步）与 `register_optional_tools()`（重依赖仿真/科学工具，建议后台）→ `register_all_tools()`。配置 `_CORE_MODULES` / `_OPTIONAL_MODULES` 元组列表。注册后重建 phase/router/constraint 调度表（`_rebuild_dispatch_tables`）。
- HTTP API 面（`huginn/server.py` + `huginn/routes/*`）：`include_v1_routes(app, keep_root_compat=True)` 将所有路由挂到 `/v1` 前缀，并保留根路径兼容（带弃用提醒）。全应用统一 `Depends(require_api_key)`、统一错误信封 `huginn_error_response`（含 `request_id`）。端点族（真实路径）：
  - 健康：`/health`、`/health/live`、`/health/ready`、`/ready`、`/health/guidance`、`/health/rust`
  - 会话线程：`/threads` CRUD、`/threads/{id}/messages|state|archive|unarchive|fork|branches|switch-branch`
  - 执行：`/execute`、`/explore`、`/diagnose`、`/plan`、`/workflows`、`/workflows/execute`
  - 多 agent：`/team/v2/members|plan|run|plans/*|fusion`、`/team/profiles|plan|run`
  - 认证：`/auth/login|token|me|refresh|logout`
  - 凭据：`/credentials` CRUD、`/credentials/{cid}/test|link-model/{alias}`、`/credentials/import-from-config`、`/credentials/{service}`
  - MCP：`/mcp/servers`、`/mcp/servers/discover|connect|{name}/disconnect|reconnect`、`/mcp/tools/{tool}/call`、`/mcp/connect/batch`、`/mcp/prompts`、`/mcp/resources/subscribe`、`/mcp/status`
  - 传输：`/transfer/upload|download|browse|sync`（require_admin_key）
  - 研究项目：`/projects` CRUD、`/projects/{pid}/threads|knowledge|run-research|research-status`
  - 导出导入：`/export/status|all|memory|knowledge`、`/import/all`
  - 事件/WS：`/events/stream`（SSE）、`/ws/agent`（WebSocket 实时对话）
  - 其他：`/data/dictionary*`、`/data/validate`、`/bench/run`、`/evolve/run`、`/autoloop/start|status`、`/live/execute`、`/live/capabilities`、`/system/components`、`/advisor/models|recommend|compare`、`/users/*`、`/projects/*`、`/maintenance`、`/firewall/status`、`/sandbox/execute`、`/analyze/symmetry|spectral|dynamics|tda|sindy`、`/viz/dos|phase|persistence|sindy`、`/project-context`
- COT 捕获契约（`huginn/agent/streaming.py`）：AIMessage 上的 `reasoning_content`（如 DeepSeek）必须经 `memory.add_reasoning(...)` 写入 `session.reasoning_trace`，供下游蒸馏消费。
- 共享状态后端契约（`huginn/server_core.py` + `huginn/persistence/state_store.py`）：`_threads`/`_checkpoints` 为 `MutableMapping`，默认进程内内存 dict（`HUGINN_STATE_BACKEND` 未设置）；设置 `HUGINN_STATE_BACKEND=sqlite` 时切换为 `SqliteStore`（SQLite WAL + busy_timeout，持久化到 `<runtime_home>/state.sqlite`）。`SqliteStore` 实现 MutableMapping 并缓存解码值，保持 `_threads[tid]["label"]=x` 等原地修改语义与 dict 等价；检查点值 `(Path, dict)` 经 `encode_checkpoint`/`decode_checkpoint` 编解码。
- 知识验证闭环（`huginn/evolution/knowledge_distiller.py` + `huginn/memory/manager.py`）：蒸馏知识 `verification_status ∈ {unverified, confirmed, rejected}`；`verify_knowledge(knowledge_id, status)` 将成功使用过的知识升级为 confirmed；`auto_ingest_to_kb` 仅吸收 `confirmed` 条目。`MemoryManager._verify_distilled_for_tool` 解析 `source="distiller:{id}"` 并调用 `verify_knowledge`。

## convention

- 测试目录：`tests/`，pytest `asyncio_mode=auto`，`--cov=huginn`，覆盖率门禁 60（CI 显式 `--cov-fail-under=60`）。`tests/conftest.py` 顶部 `collect_ignore_glob` 忽略 7 个脚本式测试文件。
- 测试夹具约定（`tests/conftest.py`）：
  - session 级 `shared_huginn_app`：单例 FastAPI app，构建约 2–3GB，每个 xdist worker 只建一次。
  - module 级 `app_client`：`with TestClient(...)` 上下文管理器，模块结束时关闭 transport 并跑 lifespan 关闭，杜绝 anyio portal 线程泄漏与内存累积。
  - autouse `_clear_config_cache_between_tests`：清 config 缓存与路径覆盖，隔离认证状态。
  - session autouse `_canonical_tool_registry`：会话启动时填充一次规范注册表，作恢复护栏基线。
  - autouse `_restore_tool_registry`：每测试前后对比 `ToolRegistry.snapshot()`，不一致即红名报错并回滚，拦截全局注册表状态泄漏。
  - hook：CI runner 偶发 `sqlite3 disk I/O / database is locked` 判定为环境问题并 skip。
- 工具编码约定：每个工具一个类文件置于 `huginn/tools/`，继承 `HuginnTool`，在 `huginn/tools/__init__.py` 的模块列表登记；`ToolResult` 作为所有工具统一返回结构。
- 风格：ruff select `E,F,W,I,N,UP,B,C4,SIM`（ignore `E501,B008,SIM108,N803,N806,B905`）；black 88 列；mypy 渐进式（`disallow_untyped_defs=false`，`warn_redundant_casts`/`warn_unused_ignores` 开启）。
- 新增工具流程（README）：继承 `HuginnTool` → 定义 `name/description/input_schema`（Pydantic）→ 实现 `_execute`/`call` → 登记 → 测试。

## invariant

- 全局 `ToolRegistry` 在任意测试之后必须与运行前逐位一致（autouse 护栏强制）。
- `TestClient` 必须通过上下文管理器使用（`test_testclient_hygiene.py` 强制扫描所有测试文件）。
- 每个 API 端点都受 `require_api_key` 保护；错误响应统一走 `huginn_error_response` 信封（含 `request_id`）。
- 加密密钥仅存内存、绝不落盘；静态数据 AES-128-CBC + per-item salt。
- 模型暴露的 `reasoning_content` 必须落入 `session.reasoning_trace`，蒸馏/进化才能读到。
- 成功使用的蒸馏知识必须可升级为 `confirmed` 才能进入知识库（`auto_ingest_to_kb` 门槛）。

## constraint

- Python 版本硬约束 `>=3.11,<4.0`。
- `mcp` 锁定 `>=1.28.1,<2.0`（2.0 为重大 API 重构，暂不升级）。
- `chromadb` 存在 PYSEC-2026-311（预认证代码注入，仅影响 HTTP API + `trust_remote_code=true`）；huginn 用本地嵌入式 `PersistentClient`，不暴露 HTTP，不受影响，无 fix 版本。
- ML 势能依赖互斥：`ml-mace`（e3nn==0.4.4）与 `ml-fairchem`（e3nn>=0.5）必须分装，避免 uv 解析失败。
- 默认限流 120 req/min/IP，认证端点 10 req/min/IP；`HUGINN_RATE_LIMIT_PER_MINUTE=0` 关闭。
- 请求体上限默认 10MB（`HUGINN_MAX_BODY_SIZE_MB`），请求超时默认 180s（`HUGINN_REQUEST_TIMEOUT_SEC`）。
- 生产环境隐藏交互式文档（`HUGINN_HIDE_DOCS` 或 `HUGINN_ENV=production`）。
- WebSocket ping 超时 300s（适配 DeepSeek reasoner 长推理）。
- 进程内 `~/.huginn` 写入在测试中重定向到 per-xdist-worker 的 `.test_cache/<worker>` 目录，规避 SQLite 锁冲突。

## 已实现机制（原 design spec 落地）

> 本节收录各已实现 design spec 中仍有效的事实性内容（核心设计决策、数据结构、模块/函数路径、行为契约、环境开关）。来源为现已删除的原 spec 文件（文件名仅作历史标注）。

### 异步委派（async_dispatch_spec）

- 子任务 DAG 调度（`huginn/agents/task_dag.py`）：`TaskDAG` 提供 `topological_order`（Kahn 拓扑序）、`antichain_width`（最大反链，决定并行层宽度）、`critical_path`（关键路径，wall-clock 下限）、`parallel_layers`（同层 antichain 内并行）。`build_dag_with_provenance`/`infer_dependencies_from_provenance` 从 tool 输入 provenance 推断子任务依赖，供 `dispatch_parallel` 使用。
- 素数预算分解（`huginn/agents/budget_decomp.py`）：`budget_configurations`（用 sympy.factorint 分解总预算 N=depth×parallel×per）、`recommend`、`config_cost`。深度取小素数（2,3）、并行取中素数（5,7）、单 agent 取大素数（53,97）。
- 周期检测（`huginn/runtime/cycle_detect.py`）：`detect_cycle`（Floyd 环检测，O(1) 空间）、`is_stuck`（判定 action 序列是否陷入周期）。
- 历史轨迹匹配（`huginn/knowledge/trajectory_pattern.py`）：`trajectory_match`（工具名序列转 line graph + VF2 子图同构前缀匹配）、`extract_and_store_pattern`（写入去重）、`update_pattern_confidence(kb, pattern_id, success)`（±ε Bayesian 反馈）。
- 接入点：`huginn/autoloop/engine_reflect.py` 的 stuck 检测（`HUGINN_EXTREME_DISPATCH=1` 时跑 `is_stuck`+`trajectory_match`，否则直接不判）；`huginn/tools/subagent_tool.py` 的 `dispatch_parallel` DAG-aware 路径需极限模式开启。
- 环境开关：`HUGINN_EXTREME_DISPATCH`（极限模式，默认 `0` 关；`1` 开启 DAG 委派 + 周期检测 + 轨迹召回 + pattern confidence 闭环）。长程任务（`max_iterations>=20`）自动启用，短程任务需显式设置。

### 分层 memory（layered_memory_spec）

- 四层信息论分层：WM（working，进程内不持久）/ EM（episodic，SQLite+FTS5+可选 embedding）/ SM（semantic，KB+KG）/ PM（procedural，stable_principles+confidence）。
- WM sliding window（`huginn/memory/session.py`）：`_compact_if_needed` 超预算时 summarize 推 EM。策略由 `HUGINN_WM_SUMMARIZE` 切换：`rule`（默认）/`ngram`/`llm`/`hybrid`；触发周期 `HUGINN_WM_SUMMARIZE_EVERY_N`（默认 5）。
- typed memory 默认开启（`huginn/memory/typing.py`）：`remember_typed`/`recall_typed`，类型系统 `{episode, fact, principle, persona_history, calculation, distilled}`；`HUGINN_USE_MEMORY_TYPING` 默认 `1`（设为 `0` 可关）。旧行（memory_type NULL）recall 时 lazy migrate。
- PM Bayesian confidence：`update_pattern_confidence` 按复用成败 ±ε，低于阈值（`HUGINN_PM_C_MIN`，默认 0.2）删除。
- EM 检索：`LongTermMemory.retrieve`（`huginn/memory/longterm.py`）FTS5 + embedding（`semantic=True` 路径）双路。

### memory/context/dispatch 接通（memory_dispatch_integration_spec）

- `huginn/memory/manager.py` 的 `recall_for_prompt` 在原有 FTS5+category 路径上叠加 typed memory 结果（按 content hash 去重，typed 优先），`recall_typed`/`remember_typed` 为 typed 结构化列读写入口。
- typed 类型优先级：`failed_direction` > `iteration_result` > `cross_domain_transfer` > `persona_history` > `stable_principle`（失败方向最优先避免重蹈覆辙）。
- persona 选择走 EM 显式召回：`_pick_hypothesis_persona` 不再 grep fallback，直接查 typed `persona_history` 取 r_phys 平均。

### LSP 符号级编辑 + Hashline 锚定编辑（lsp_hashline_spec）

- `huginn/tools/lsp_tool.py`：`LspTool`（jedi-first，纯 Python 静态分析，外部 langserver 为 v2 升级路径）。`LspToolInput.action ∈ {rename, references, hover, diagnostics, code_action, definition}`，`provider ∈ {auto, jedi, external}`。除 `rename` 外均只读；`rename` 落盘前过 `_resolve_path` 边界 + 权限检查。jedi 缺失时优雅降级为明确错误，不影响文本编辑工具。
- Hashline 锚定编辑（`huginn/tools/file_edit_tool.py`、`multi_edit_tool.py`）：输入新增 `expected_hash`（sha256[:16]，取自上次 read/edit 的 `snapshot_hash`）与 `hash_policy ∈ {strict, warn, off}`（默认 `strict`）。写入前校验，`expected_hash` 与磁盘不符：`strict` 拒绝编辑（防并发覆盖），`warn` 记审计后继续，`off`/`expected_hash=None` 跳过校验（向后兼容）。`multi_edit_tool` 每个文件单独校验，任一 strict 不匹配整批拒绝（保持原子性）。`_content_hash` 用 sha256[:16] 稳定 utf-8 编码。

### 事件溯源 · 沙箱 · 增量前端（reinforcement_event_sourcing_sandbox_incremental_ui）

- `huginn/events/session_log.py`：`SessionEventLog` 追加式 JSONL 事件日志（`seq/id/parent_id/ts/payload`），接口 `append`/`read_after(seq)`/`branch(target_seq)`（移动叶指针，历史不删）/`branch_with_summary`。`seq` 单调递增作增量同步游标。
- `huginn/events/projection.py`：`ProjectionDefinition`（`init/apply/view/stateVersion`，纯函数 fold）+ `ProjectionEngine`（watermark cell + 弱引用缓存；`stateVersion` 变更整体重建）。事件层取树/叶指针，读模型层取投影，两者解耦。
- 前端增量：`desktop/src/hooks/useIncrementalMessages.ts` 块级消息模型（`blocks: {kind,text,frozen,rev}`），`text_delta` 只更新最后块，frozen 块跳过重渲染。

### 增强工具模块（enhanced_modules）

- 安全数学求值（`huginn/security/math_eval.py`）：`safe_math_eval` 取代裸 `eval()`，AST-walking 白名单，仅许算术 + 白名单 numpy 函数，禁属性访问/import/lambda。
- 数值/科学工具（均继承 `HuginnTool`，`huginn/tools/` 下）：
  - `numerical_tool.py`：`NumericalTool`（scipy/numpy 求解器统一接口，root/ode/constrained_minimize/svd/matrix_exp，表达式经 `safe_math_eval`）。
  - `sci/unit_tool.py`：`UnitTool`（pint 单位换算/量纲推断，pint 缺失回退 registry）。
  - `sci/symmetry_tool.py`：`SymmetryTool`（pymatgen/spglib 晶体对称分析，subgroups/wyckoff_split/magnetic）。
  - `sci/descriptor_tool.py`：`DescriptorTool`（材料描述符，composition/matminer/mbtr/acsf/coulomb_matrix/ibp）。
  - `sci/autodiff_tool.py`：`AutoDiffTool`（JAX 自动微分，finite-difference 回退，gradient/jacobian/optimize）。
  - `sci/uq_tool.py`：`UQTool`（不确定性量化，monte_carlo/pce/morris）。
  - `gp_tool.py`：`GPTool`（高斯过程回归 + 贝叶斯优化，Matérn 核、UCB/PI 采集、natural_gradient/fisher_information/kl_divergence）。
  - `sci/tda_tool.py`：`TDATool`（持久同调，ripser/gudhi，scipy 回退）。
  - `sci/evidence_fusion_tool.py`：`EvidenceFusionTool`（Dempster-Shafer 多源证据融合，combine/pignistic/weighted_combine）。
  - `sci/high_throughput_tool.py`：`HighThroughputTool`（对任意已注册工具参数扫描，grid/lhs）。
- 上下文/token 工具（`huginn/utils/`）：`compact_messages`（O(n) 压缩）、`summarize_compact_messages`（摘要上限 2000 token，超限自动再压缩）；`count_tokens`/`count_message_tokens`（`huginn/utils/tokens.py`，模型感知编码，cl100k_base 默认 / o200k_base 用于 gpt-4o、o1、o3）。
- 遥测内存追踪（`huginn/telemetry.py`）：span 记录 `memory_start_mb`/`memory_end_mb`/`memory_peak_mb`，`memory_snapshot()` 返回 RSS 快照，`summary()` 含内存统计。
- 会话分支树：`agent.fork_conversation()`/`switch_branch(node_id)`/`conversation_branches()` 支持多假设探索；ToolMessage metadata 保留以正确重建工具调用链。

### Ising 召回 + CRDT 合并（ising_crdt_p1_spec）

- 伊辛能量函数式 memory recall（`huginn/memory/longterm.py`）：`_ising_rerank(query, candidates, top_k, beta)` 把 FTS5 独立 top_k 排序升级为能量最低 K-子集（外场 H=sim(query,mᵢ)，耦合 T=sim(mᵢ,mⱼ)，贪心加入 ΔE<0 接受）。`semantic=False`/`top_k=1`/embedding 未初始化时跳过或退化原排序。开关 `HUGINN_ISING_RERANK`（默认 `1`）。
- CRDT 分支合并（`huginn/utils/conversation_tree.py`）：`merge_branch_into_active`/`_crdt_branch_merge` 用 CRDT 语义（G-Set union + LWW-Register）合并分支 metadata 回 active leaf，满足交换/结合/幂等。开关 `HUGINN_CRDT_BRANCH_MERGE`（默认 `1`）。
- CRDT 并行结果合并（`huginn/tools/subagent_tool.py`）：`_dispatch_parallel` 末尾 `_crdt_merge(all_results)`，findings/evidence/limitations 用 G-Set union（按 content hash 去重），单值字段用 LWW（ts 新者胜），语义冲突仍走 LLM 仲裁（`_resolve_support_finding`）。开关 `HUGINN_CRDT_MERGE`（默认 `1`）。

### HiLS 分层稀疏 attention（hils_active_inference_p2_spec）

- `huginn/memory/longterm.py` 的 `_hils_attention(query, candidates, top_k, ...)`：把 `_ising_rerank` 的离散贪心升级为连续分层稀疏 attention（地标层 K 个 landmark + 精细层 top-h 下 memory），接口 `(query, candidates, top_k) -> ranked` 不变，仅在 `semantic=True` 路径替代 rerank。N<K 时退化全 attention；embedding 未初始化回退 Ising 贪心。地标缓存 lazy init + 按 `HUGINN_HILS_LANDMARK_REFRESH`（默认 1000 次 retrieve）周期性重算。