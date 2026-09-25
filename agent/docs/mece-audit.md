# MECE 契约审计 (奖励面 + 授权面 + 工作流面 + 模式面 + 词汇面 + 工具面 + 钩子面 + 事件面 + SSE 消费面 + WS 消费面 + HTTP API 消费面)

自动生成: `python -m huginn.cli.contract_audit --out docs/mece-audit.md`.
以 MECE 两原则审计 agent 的**奖励面 / 授权面 / 工作流面 / 模式面 / 词汇面 / 工具面 / 钩子面 / 事件面 / SSE 消费面 / WS 消费面 / HTTP API 消费面**: **collectively exhaustive** 抓「宣称维度零调用者 / 面之间的缺口」; **mutually exclusive** 抓「同轴惩罚叠加」「跨模块同名重复实现」「词表互不一致」「同名工具名多类声明」「事件常量撞值」「SSE 帧名挂错通道」「WS 帧名挂错端点」「HTTP 同 method+path 多模块注册」. 纯静态扫描, 只提示候选, 不判死.

## 奖励面: 宣称项 vs 调用者 (collectively exhaustive)

来源: `claim_reward.py::__all__` 的宣称面. `状态 dead` = 宣称但零调用者; `internal-only` = 仅模块内被组合复用 (非死, 但无独立接线).

| 奖励项 | 生产调用 | 测试引用 | 模块内引用 | 状态 | 备注 |
|---|---|---|---|---|---|
| `numeric_accuracy_reward` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `grounding_source_reward` | 1 | 1 | 1 | `wired` |  |
| `grounded_accuracy_reward` | 1 | 0 | 0 | `wired` |  |
| `strict_scope_reward` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `efficiency_discount` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `idle_turn_penalty` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `anti_hacking_reward` | 1 | 0 | 0 | `wired` |  |
| `reconcile_r_phys` | 0 | 0 | 0 | `dead` | 零调用者 —— 宣称但未接线; 同名跨模块, 归属已按模块限定隔离 |

### 惩罚轴重叠 (mutually exclusive)

| 惩罚轴 | 惩罚项 (触发信号) | 是否重叠 |
|---|---|---|
| 授权越界 | `strict_scope_reward`(authorized_ratio) | 否 |
| 轮次 | `efficiency_discount`(solved_at_turn), `idle_turn_penalty`(extra_turns) | ⚠️ 是 |

### 跨模块同名 (mutually exclusive)

| 名称 | 其它模块也定义 |
|---|---|
| `reconcile_r_phys` | `huginn/security/world_state.py` |

## 授权面: 口径源 vs 消费点

来源: `scope_authority.py::__all__` (S1 合规口径 / S2 意图口径). 两口径声明为**独立开关**, 消费点在 `autoloop/engine_reflect.py::_apply_strict_scope`.

| 口径源 | 生产消费点 | 状态 | 说明 |
|---|---|---|---|
| `compute_authorized_ratio` | 1 | `wired` | S1 合规口径 |
| `compute_intent_ratio` | 1 | `wired` | S2 意图口径 |

### 独立开关

| 开关 | 已注册 | 默认 | 消费点 |
|---|---|---|---|
| `anti_hacking_reward` | True | `False` | huginn/autoloop/engine_reflect.py:2322 |
| `intent_scope_reward` | True | `False` | huginn/autoloop/engine_reflect.py:2323 |

## 工作流面: 执行 mode 分发 vs planner 提示面

来源: `phase_spec.dispatch_table` (登记面) / `engine_act._execute` (硬编码分支) / `plan_check` 的 planner 提示 (教给 LLM 的 MODE 候选). 三者应 collectively exhaustive 对齐; 登记面支持而提示未列的 mode, 只能靠非 prompt 路径触达.

| 面 | 集合 |
|---|---|
| dispatch_table | `coder`, `dynamic_workflow`, `explore`, `skill`, `visual_inspect`, `workflow` |
| engine_act 分支 | `coder`, `dynamic_workflow`, `explore`, `skill`, `visual_inspect`, `workflow` |
| planner 提示 #1 | `coder`, `explore`, `skill`, `visual_inspect`, `workflow` |
| planner 提示 #2 | `coder`, `explore`, `skill`, `visual_inspect`, `workflow` |

- dispatch_table == 硬编码分支: ✅ 一致
- planner 多处提示互相一致: ✅
- dispatch 支持但 planner 未教 (非 prompt 路径触达): `dynamic_workflow`

## 模式面: agent 顶层模式词表一致性

来源: prompt 段 (`_MODE_INSTRUCTIONS`) / session 恢复白名单 / `critique._VALID_MODES` / `core._LONG_HORIZON_MODES` / `set_mode()` 实参. 任何一处都不是全集即 MECE 违例 (mutex: 词表互不一致; exhaustive: 缺口).

| 来源 | 集合 |
|---|---|
| prompt_builder._MODE_INSTRUCTIONS | `chat`, `code`, `extreme`, `fusion`, `research` |
| session 恢复白名单 | `chat`, `extreme`, `plan`, `research` |
| critique._VALID_MODES | `chat`, `plan`, `research` |
| core._LONG_HORIZON_MODES | `extreme`, `research` |
| task_state 注释 | `chat`, `plan`, `research` |
| set_mode() 运行时实参 | `chat`, `plan`, `research` |

- 词表并集: `chat`, `code`, `extreme`, `fusion`, `plan`, `research`
- 有 prompt 段却无 `set_mode()` 生产者 (死提示词候选): `code`, `extreme`, `fusion`
- 被 `set_mode()` 却无 prompt 段 (无提示词的模式): `plan`
- 各词表互相一致: ⚠️ 否

## 词汇面: 值域词表雷达 (系统枚举 + 自动聚类)

系统枚举**值域词表** (闭集枚举: `Literal`/Enum/`frozenset`/全大写元组) 共 265 站点, 按值域 Jaccard 重叠自动聚成 240 簇 (其中闭集簇 156). 三类结构性违例: 同名跨模块定义 / 未登记撞名 / 映射非单射.

### 同名跨模块定义 (mutually exclusive)

| 名称 | 值域一致 | 定义点 (规模) |
|---|---|---|
| `KINDS` | ⚠️ 异 | `huginn/catalog/models.py:20`(7), `huginn/research/cspace.py:32`(3), `huginn/evolution/semantic_distiller.py:39`(4) |
| `Severity` | ⚠️ 异 | `huginn/execution/physics_auditor.py:24`(3), `huginn/metacog/failure_modes.py:27`(3) |
| `_ALLOWED_IMPORTS` | ⚠️ 异 | `huginn/security/script_runner.py:74`(22), `huginn/security/code_act_sandbox.py:32`(21) |
| `_DEFAULT_MCP_ALLOWED_COMMANDS` | ✅ 同 | `huginn/mcp_client.py:67`(5), `huginn/config.py:316`(5) |
| `_EXPENSIVE_TOOLS` | ✅ 同 | `huginn/research_budget.py:24`(8), `huginn/hooks/research_safety_hook.py:18`(8) |
| `_KINDS` | ⚠️ 异 | `huginn/share.py:21`(5), `huginn/workflows/registry.py:25`(4) |
| `_NEGATIVE_WORDS` | ⚠️ 异 | `huginn/persona_emotion.py:147`(22), `huginn/tools/design/gap_analysis_tool.py:30`(31) |
| `_READ_ACTIONS` | ⚠️ 异 | `huginn/tools/github_tool.py:55`(4), `huginn/tools/git_tool.py:85`(3) |
| `class MemoryType` | ⚠️ 异 | `huginn/memory/types.py:12`(5), `huginn/memory/typing.py:28`(10) |
| `class RiskLevel` | ✅ 同 | `huginn/core_types.py:29`(5), `huginn/ontology/actions.py:41`(5) |

### 未登记撞名 (同一词横跨两个命名空间)

| 词 | 出现于 (命名空间代表站点) |
|---|---|
| `api_key` | `huginn/mcp_client.py::_SENSITIVE_CONFIG_KEYS`, `huginn/rag/vector_store.py::SENSITIVE_META_FIELDS`, `huginn/tools/config_domain_tool.py::DENY_SET_FIELDS` |
| `ask` | `huginn/core_types.py::class BudgetDecision`, `huginn/core_types.py::class PermissionMode` |
| `band` | `huginn/tools/sim/vasp_tool.py::_COMPUTE_ACTIONS`, `huginn/utils/smart_prefetch.py::_PIPELINE_STAGES` |
| `bash_tool` | `huginn/modes/pi.py::PRIMITIVES`, `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/routes/ws_helpers.py::_DANGEROUS_TOOLS`, `huginn/tools/adapter.py::_CODING_TOOLS` |
| `body` | `huginn/events/audit_log.py::_BODY_KEYS`, `huginn/harness/joint_optimizer.py::_CORE_BLOCKS` |
| `code_tool` | `huginn/modes/pi.py::PRIMITIVES`, `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/tools/adapter.py::_CODING_TOOLS` |
| `coding` | `huginn/models/router.py::TaskT`, `huginn/pet/__init__.py::class PetMood` |
| `cp2k_tool` | `huginn/execution/compute_router.py::_DFT_MD_TOOLS`, `huginn/research_budget.py::_EXPENSIVE_TOOLS`, `huginn/tools/design/design_plan_tool.py::GATED_TOOLS` |
| `deny` | `huginn/core_types.py::class BudgetDecision`, `huginn/core_types.py::class PermissionMode` |
| `dos` | `huginn/tools/sim/vasp_tool.py::_COMPUTE_ACTIONS`, `huginn/utils/smart_prefetch.py::_PIPELINE_STAGES` |
| `elastic_constants` | `huginn/hooks/science_hooks.py::_ELASTIC_KEYS`, `huginn/tools/hypothesis_generator_tool.py::_WORKFLOW_TEMPLATES` |
| `execute` | `huginn/autoloop/cognitive_loop.py::VALID_ACTIONS`, `huginn/metacog/critique.py::_VALID_PHASES`, `huginn/session_state.py::class SessionPhase` |
| `explore` | `huginn/autoloop/budget.py::PlanMode`, `huginn/branch_policy.py::DevStage`, `huginn/session_state.py::class SessionPhase` |
| `figure` | `huginn/hooks/clarify_questions_hook.py::_OUTPUT_FORMATS_EN`, `huginn/perception/doc_types.py::class ElementType` |
| `file_edit_tool` | `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/routes/ws_helpers.py::_DANGEROUS_TOOLS`, `huginn/runtime/task_tool_router.py::CORE_TOOL_NAMES`, `huginn/tools/adapter.py::_HASHLINE_TOOLS` |
| `file_path` | `huginn/agents/task_dag.py::_PROV_PATH_FIELDS`, `huginn/core_types.py::class HandleType` |
| `file_read_tool` | `huginn/modes/pi.py::PRIMITIVES`, `huginn/runtime/task_tool_router.py::CORE_TOOL_NAMES`, `huginn/tools/adapter.py::_HASHLINE_TOOLS` |
| `file_write_tool` | `huginn/modes/pi.py::PRIMITIVES`, `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/runtime/task_tool_router.py::CORE_TOOL_NAMES`, `huginn/tools/adapter.py::_HASHLINE_TOOLS` |
| `fit` | `huginn/hooks/clarify_questions_hook.py::_COMPUTE_VERBS_EN`, `huginn/tools/bash_tool.py::_PY_LONG_RUN_KEYWORDS` |
| `formula` | `huginn/core_types.py::class HandleType`, `huginn/perception/doc_types.py::class ElementType` |
| `gaussian_tool` | `huginn/execution/compute_router.py::_QC_TOOLS`, `huginn/research_budget.py::_EXPENSIVE_TOOLS` |
| `goal` | `huginn/routes/deep_research.py::_STR_KEYS`, `huginn/runtime/context_router.py::CONTEXT_SEGMENTS` |
| `hypothesis` | `huginn/autoloop/cognitive_persist.py::_RUN_CONTEXT_KEYS`, `huginn/harness/joint_optimizer.py::_CORE_BLOCKS`, `huginn/phases.py::class ResearchPhase` |
| `hypothesize` | `huginn/autoloop/cognitive_loop.py::VALID_ACTIONS`, `huginn/metacog/critique.py::_VALID_PHASES` |
| `inconclusive` | `huginn/autoloop/cognitive_persist.py::_RUN_CONTEXT_KEYS`, `huginn/perception/doc_types.py::class EdgeType` |
| `json` | `huginn/hooks/clarify_questions_hook.py::_OUTPUT_FORMATS_EN`, `huginn/lean/conjecture_library.py::_PROOF_ALLOWED_MODULES` |
| `knowledge` | `huginn/export_share.py::ALL_COMPONENTS`, `huginn/routes/knowledge.py::ALLOWED_SOURCES`, `huginn/routes/search.py::SEARCH_SOURCE_TYPES` |
| `lammps_tool` | `huginn/execution/compute_router.py::_DFT_MD_TOOLS`, `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/research_budget.py::_EXPENSIVE_TOOLS`, `huginn/tools/design/design_plan_tool.py::GATED_TOOLS` |
| `learn` | `huginn/autoloop/cognitive_loop.py::VALID_ACTIONS`, `huginn/metacog/critique.py::_VALID_PHASES`, `huginn/ontology/actions.py::class ActionCategory` |
| `local` | `huginn/cli/lazy_loader.py::CommandType`, `huginn/models/router.py::TaskT`, `huginn/routes/tunnels.py::TunnelType` |
| `math` | `huginn/cli/rcb_utils.py::_DOMAIN_KNOWN`, `huginn/lean/conjecture_library.py::_PROOF_ALLOWED_MODULES` |
| `memory` | `huginn/export_share.py::ALL_COMPONENTS`, `huginn/routes/search.py::SEARCH_SOURCE_TYPES`, `huginn/runtime/context_router.py::CONTEXT_SEGMENTS` |
| `methodology` | `huginn/metacog/failure_modes.py::Category`, `huginn/tools/literature/tool.py::_DEFAULT_LENSES` |
| `model` | `huginn/catalog/models.py::KINDS`, `huginn/routes/deep_research.py::_STR_KEYS` |
| `multi_edit_tool` | `huginn/modes/pi.py::PRIMITIVES`, `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/routes/ws_helpers.py::_DANGEROUS_TOOLS`, `huginn/tools/adapter.py::_HASHLINE_TOOLS` |
| `network` | `huginn/ontology/actions.py::class ActionCategory`, `huginn/plugins/permissions.py::class PluginPermission` |
| `notebook_edit_tool` | `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/tools/adapter.py::_CODING_TOOLS` |
| `numpy` | `huginn/lean/conjecture_library.py::_PROOF_ALLOWED_MODULES`, `huginn/provenance/_legacy.py::_TRACKED_PACKAGES` |
| `orca_tool` | `huginn/execution/compute_router.py::_QC_TOOLS`, `huginn/research_budget.py::_EXPENSIVE_TOOLS` |
| `password` | `huginn/mcp_client.py::_SENSITIVE_CONFIG_KEYS`, `huginn/rag/vector_store.py::SENSITIVE_META_FIELDS` |
| `pivot` | `huginn/autoloop/cognitive_loop.py::VALID_ACTIONS`, `huginn/autoloop/hypothesis_loop.py::EdgeType` |
| `plan` | `huginn/autoloop/cognitive_loop.py::VALID_ACTIONS`, `huginn/core_types.py::class PermissionMode`, `huginn/memory/reasoning.py::class ReasoningPhase`, `huginn/metacog/critique.py::_VALID_MODES`, `huginn/metacog/critique.py::_VALID_PHASES`, `huginn/runtime/context_router.py::CONTEXT_SEGMENTS`, `huginn/session_state.py::class SessionPhase` |
| `prompt` | `huginn/catalog/models.py::KINDS`, `huginn/cli/lazy_loader.py::CommandType` |
| `provenance` | `huginn/routes/knowledge.py::ALLOWED_SOURCES`, `huginn/routes/search.py::SEARCH_SOURCE_TYPES` |
| `qe_tool` | `huginn/execution/compute_router.py::_DFT_MD_TOOLS`, `huginn/research_budget.py::_EXPENSIVE_TOOLS`, `huginn/tools/design/design_plan_tool.py::GATED_TOOLS` |
| `reflect` | `huginn/memory/reasoning.py::class ReasoningPhase`, `huginn/session_state.py::class SessionPhase` |
| `relax` | `huginn/tools/sim/vasp_tool.py::_COMPUTE_ACTIONS`, `huginn/utils/smart_prefetch.py::_PIPELINE_STAGES` |
| `report` | `huginn/hooks/clarify_questions_hook.py::_OUTPUT_FORMATS_EN`, `huginn/metacog/critique.py::_VALID_PHASES`, `huginn/session_state.py::class SessionPhase` |
| `research` | `huginn/metacog/critique.py::_VALID_MODES`, `huginn/workflows/registry.py::_KINDS` |
| `result` | `huginn/cli/slash_commands.py::_BG_SUBCOMMANDS`, `huginn/workflows/registry.py::_STAGE_RUNTIME_FIELDS` |
| `scipy` | `huginn/lean/conjecture_library.py::_PROOF_ALLOWED_MODULES`, `huginn/provenance/_legacy.py::_TRACKED_PACKAGES` |
| `secret` | `huginn/mcp_client.py::_SENSITIVE_CONFIG_KEYS`, `huginn/rag/vector_store.py::SENSITIVE_META_FIELDS` |
| `skill` | `huginn/catalog/models.py::KINDS`, `huginn/share.py::_KINDS` |
| `state` | `huginn/research/cspace.py::KINDS`, `huginn/tools/browser_tool.py::class BrowserAction` |
| `stop` | `huginn/autoloop/cognitive_loop.py::VALID_ACTIONS`, `huginn/cli/slash_commands.py::_BG_SUBCOMMANDS` |
| `structure` | `huginn/autoloop/hypothesis_semantic.py::_DIMENSION_LABELS`, `huginn/evaluation/matworld_bench.py::CATEGORIES`, `huginn/memory/longterm.py::MATERIAL_CATEGORIES`, `huginn/utils/smart_prefetch.py::_PIPELINE_STAGES` |
| `structure_tool` | `huginn/agent/context.py::_ALWAYS_ON_TOOLS`, `huginn/hooks/anomaly_llm_hook.py::_FORCE_WATCH_TOOLS`, `huginn/tools/tool_cache.py::PREFETCH_SAFE_TOOLS` |
| `table` | `huginn/hooks/clarify_questions_hook.py::_OUTPUT_FORMATS_EN`, `huginn/perception/doc_types.py::class ElementType` |
| `text` | `huginn/events/audit_log.py::_BODY_KEYS`, `huginn/perception/doc_types.py::class ElementType` |
| `topological` | `huginn/metacog/imagination.py::_TRANSFORM_TYPES`, `huginn/metacog/topology_lens.py::Family` |
| `validate` | `huginn/autoloop/cognitive_loop.py::VALID_ACTIONS`, `huginn/metacog/critique.py::_VALID_PHASES` |
| `vasp_tool` | `huginn/execution/compute_router.py::_DFT_MD_TOOLS`, `huginn/permissions.py::WRITE_EXEC_TOOLS`, `huginn/research_budget.py::_EXPENSIVE_TOOLS`, `huginn/tools/design/design_plan_tool.py::GATED_TOOLS` |
| `verification` | `huginn/models/router.py::TaskT`, `huginn/research_log.py::class RecordType` |
| `web_search_tool` | `huginn/agent/context.py::_ALWAYS_ON_TOOLS`, `huginn/runtime/task_tool_router.py::CORE_TOOL_NAMES` |
| `workflow` | `huginn/autoloop/budget.py::PlanMode`, `huginn/catalog/models.py::KINDS`, `huginn/evolution/semantic_distiller.py::KINDS`, `huginn/share.py::_KINDS` |

### 簇内分歧 (同簇成员值域不等)

- 簇 7 (并集 12 词):
  - `huginn/mcp_client.py:107` _SENSITIVE_CONFIG_KEYS 缺 `raw`
  - `huginn/events/audit_log.py:233` _SECRET_KEYS 缺 `authorization`, `cookie`
- 簇 13 (并集 6 词):
  - `huginn/config.py:29` ThinkingIntensity 缺 `critical`, `none`
  - `huginn/core_types.py:29` class RiskLevel 缺 `max`
  - `huginn/ontology/actions.py:41` class RiskLevel 缺 `max`
- 簇 14 (并集 5 词):
  - `huginn/config.py:59` ContainerRuntimeLiteral
  - `huginn/security/container_executor.py:80` _VALID_RUNTIMES 缺 `none`
- 簇 18 (并集 4 词):
  - `huginn/core_types.py:46` class BudgetDecision 缺 `ask`
  - `huginn/security/policy_engine.py:46` _VALID_ACTIONS 缺 `warn`
- 簇 21 (并集 8 词):
  - `huginn/research_budget.py:24` _EXPENSIVE_TOOLS
  - `huginn/agent/context.py:20` _EXPENSIVE_TOOL_NAMES 缺 `gaussian_tool`, `gromacs_tool`, `orca_tool`, `qe_tool`
  - `huginn/hooks/research_safety_hook.py:18` _EXPENSIVE_TOOLS
- 簇 36 (并集 8 词):
  - `huginn/lean/conjecture_library.py:36` _PROOF_ALLOWED_MODULES 缺 `numpy`, `pandas`, `scipy`
  - `huginn/bench/task_synthesizer.py:28` _JUDGE_ALLOWED_MODULES
- 簇 73 (并集 10 词):
  - `huginn/memory/types.py:12` class MemoryType 缺 `cross_domain_transfer`, `failed_direction`, `iteration_result`, `persona_history`, `stable_principle`
  - `huginn/memory/typing.py:28` class MemoryType
- 簇 135 (并集 4 词):
  - `huginn/tools/visualize_gate.py:22` RERENDERABLE 缺 `duplicate`
  - `huginn/tools/visualize_gate.py:29` _SEVERITY

### 映射表 (非单射 + 有反向表 ⇒ 往返丢信息)

| 映射表 | 条目 | 单射 | 反向表 | 共像 |
|---|---|---|---|---|
| `AUTOLOOP_TO_PHASE` | 7 | ⚠️ 否 | `PHASE_TO_AUTOLOOP` | `ResearchPhase.VALIDATION`←["'validate'", "'learn'"] |
| `STATE_TO_ATTENTION` | 8 | ⚠️ 否 | `ATTENTION_TO_STATE` (无) | `AttentionMode.AXIOM_FOCUS`←['CognitiveState.S4_CONSTRUCT', 'CognitiveState.S5_UNIFY']; `AttentionMode.MODE_SWITCH`←['CognitiveState.S3_SWITCH', 'CognitiveState.S6_FEEDBACK', 'CognitiveState.S7_SELF_MODIFY']; `AttentionMode.SINGULARITY_CONDENSATION`←['CognitiveState.S0_BLANK', 'CognitiveState.S1_DISCOVER', 'CognitiveState.S2_VALIDATE'] |
| `STATE_TO_MODEL_TASK` | 4 | ⚠️ 否 | `MODEL_TASK_TO_STATE` (无) | `'reasoning'`←['CognitiveState.S4_CONSTRUCT', 'CognitiveState.S6_FEEDBACK']; `'verification'`←['CognitiveState.S2_VALIDATE', 'CognitiveState.S7_SELF_MODIFY'] |
| `STATE_TO_PHASE` | 8 | ⚠️ 否 | `PHASE_TO_STATE` (无) | `'learn'`←['CognitiveState.S5_UNIFY', 'CognitiveState.S7_SELF_MODIFY']; `'validate'`←['CognitiveState.S2_VALIDATE', 'CognitiveState.S6_FEEDBACK'] |
| `SUITE_TO_BENCHMARK` | 6 | ⚠️ 否 | `BENCHMARK_TO_SUITE` (无) | `'PaperBench'`←["'physics'", "'repro'"] |

## 工具面: 注册声明面 vs 允许面

注册声明面: `tools/__init__.py` 注册清单 157 条 → 全仓 HuginnTool 子类声明的工具名 157 个. 允许面: 全仓工具名白名单 26 张. **死项** = 白名单里无同名工具声明的条目 (永不命中), **别名** = 裸名↔`_tool` 对应项 (非死项). 两类表不做死项判定: 与工具名零重叠的**外部命名空间** (MCP 外部工具名等), 以及含空格条目的 **关键词表** (匹配用户 prompt 文本, 不是工具名).

| 允许表 | 位置 | 条目 | 命中注册名 | 命名空间 | 状态 | 外部消费 | 死项 | 别名 |
|---|---|---|---|---|---|---|---|---|
| `READ_ONLY_TOOLS` | `huginn/permissions.py:43` | 3 | 3 | registry | `wired` | 1 | 0 | 0 |
| `WRITE_EXEC_TOOLS` | `huginn/permissions.py:50` | 8 | 8 | registry | `wired` | 1 | 0 | 0 |
| `_CORE_TOOLS` | `huginn/phases.py:109` | 15 | 15 | registry | `internal-only` | 0 | 0 | 0 |
| `_EDIT_TOOLS` | `huginn/server_core.py:96` | 2 | 2 | registry | `wired` | 1 | 0 | 0 |
| `_EXPENSIVE_TOOLS` | `huginn/research_budget.py:24` | 8 | 8 | registry | `internal-only` | 0 | 0 | 0 |
| `_EXPENSIVE_TOOL_NAMES` | `huginn/agent/context.py:20` | 4 | 4 | registry | `internal-only` | 0 | 0 | 0 |
| `_ALWAYS_ON_TOOLS` | `huginn/agent/context.py:27` | 5 | 5 | registry | `internal-only` | 0 | 0 | 0 |
| `_KNOWN_TOOLS` | `huginn/provenance/pipeline.py:90` | 39 | 38 | registry | `internal-only` | 0 | 0 | 1 |
| `_DFT_MD_TOOLS` | `huginn/execution/compute_router.py:15` | 8 | 4 | registry | `internal-only` | 0 | 0 | 4 |
| `_QC_TOOLS` | `huginn/execution/compute_router.py:29` | 4 | 2 | registry | `internal-only` | 0 | 0 | 2 |
| `_SNAPSHOT_TOOLS` | `huginn/snapshot/integration.py:26` | 26 | 26 | registry | `internal-only` | 0 | 0 | 0 |
| `_DANGEROUS_TOOLS` | `huginn/routes/ws_helpers.py:185` | 4 | 4 | registry | `internal-only` | 0 | 0 | 0 |
| `_BLOCKED_TOOLS` | `huginn/security/code_act_sandbox.py:29` | 2 | 2 | registry | `wired` | 1 | 0 | 0 |
| `PRIMITIVES` | `huginn/modes/pi.py:22` | 8 | 8 | registry | `internal-only` | 0 | 0 | 0 |
| `CORE_BOOTSTRAP_TOOLS` | `huginn/harness/promotion.py:25` | 2 | 2 | registry | `internal-only` | 0 | 0 | 0 |
| `_EXPENSIVE_TOOLS` | `huginn/hooks/clarify_questions_hook.py:54` | 15 | 4 | keywords | `internal-only` | 0 | 0 | 0 |
| `_FORCE_WATCH_TOOLS` | `huginn/hooks/anomaly_llm_hook.py:27` | 3 | 3 | registry | `internal-only` | 0 | 0 | 0 |
| `_EXPENSIVE_TOOLS` | `huginn/hooks/research_safety_hook.py:18` | 8 | 8 | registry | `internal-only` | 0 | 0 | 0 |
| `_WATCHED_TOOLS` | `huginn/hooks/__init__.py:352` | 3 | 3 | registry | `internal-only` | 0 | 0 | 0 |
| `_HASHLINE_TOOLS` | `huginn/tools/adapter.py:340` | 5 | 5 | registry | `internal-only` | 0 | 0 | 0 |
| `_CODING_TOOLS` | `huginn/tools/adapter.py:918` | 3 | 3 | registry | `internal-only` | 0 | 0 | 0 |
| `_REVIEWING_TOOLS` | `huginn/tools/adapter.py:925` | 5 | 5 | registry | `internal-only` | 0 | 0 | 0 |
| `PREFETCH_SAFE_TOOLS` | `huginn/tools/tool_cache.py:38` | 3 | 3 | registry | `wired` | 1 | 0 | 0 |
| `_HIGH_VALUE_MCP_TOOLS` | `huginn/tools/mcp_adapter.py:27` | 6 | 0 | external | `internal-only` | 0 | 0 | 0 |
| `CORE_TOOL_NAMES` | `huginn/runtime/task_tool_router.py:40` | 5 | 5 | registry | `wired` | 1 | 0 | 0 |
| `GATED_TOOLS` | `huginn/tools/design/design_plan_tool.py:45` | 12 | 12 | registry | `wired` | 1 | 0 | 0 |

- 被白名单覆盖的工具名: 75 / 157
- 外部命名空间 (整表与注册名零重叠, 不判死项): `_HIGH_VALUE_MCP_TOOLS`
- 关键词表 (含空格条目, 匹配 prompt 文本, 不判死项): `_EXPENSIVE_TOOLS`

### 死项 (白名单条目无同名注册工具 ⇒ 永不命中)

— 无。

### 注册声明缺口

- 注册清单引用的类均静态可解析。

## 钩子面: 事件声明面 vs 触发面 vs 注册面

声明面: `huginn/hooks/__init__.py` 的 10 个事件(`ALL_EVENTS` 权威清单). 触发面: `trigger()` / `_trigger_hook()` / `run_pre()`(≡`pre_tool_use`) / `run_post()`(≡`post_tool_use`). 注册面: `register()` / `register_hook()`. `trigger-only` = 会触发但零消费者 (对偶于奖励面「宣称项零调用者」); `dead` = 声明了却零触发零注册.

状态: `wired`=有触发点且有消费者; `trigger-only`=有触发点但零注册 (触发无人接); `register-only`=有注册但零生产触发; `dead`=声明零触发且零注册

| 事件常量 | 值 | 生产触发 | 生产注册 | 测试触发 | 测试注册 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|
| `PRE_TOOL_USE` | `pre_tool_use` | 2 | 5 | 1 | 0 | `wired` |  |
| `POST_TOOL_USE` | `post_tool_use` | 3 | 31 | 1 | 0 | `wired` |  |
| `SESSION_START` | `session_start` | 1 | 0 | 0 | 0 | `trigger-only` | 触发点存在但无注册消费者 —— 扩展点候选 |
| `SESSION_END` | `session_end` | 1 | 0 | 0 | 0 | `trigger-only` | 触发点存在但无注册消费者 —— 扩展点候选 |
| `STOP` | `stop` | 1 | 1 | 0 | 0 | `wired` |  |
| `SUBAGENT_STOP` | `subagent_stop` | 1 | 0 | 0 | 0 | `trigger-only` | 触发点存在但无注册消费者 —— 扩展点候选 |
| `PRE_COMPACT` | `pre_compact` | 1 | 0 | 0 | 0 | `trigger-only` | 触发点存在但无注册消费者 —— 扩展点候选 |
| `POST_COMPACT` | `post_compact` | 1 | 0 | 0 | 0 | `trigger-only` | 触发点存在但无注册消费者 —— 扩展点候选 |
| `USER_PROMPT_SUBMIT` | `user_prompt_submit` | 1 | 3 | 0 | 0 | `wired` |  |
| `POST_TOOL_USE_FAILURE` | `post_tool_use_failure` | 1 | 0 | 0 | 0 | `trigger-only` | 触发点自带 `if self._callbacks[POST_TOOL_USE_FAILURE]` 守卫 —— 零注册 ⇒ 该分支恒不执行, 是可证死的触发点 |

### 触发点 / 注册点明细

| 事件常量 | 生产触发点 | 生产注册点 |
|---|---|---|
| `PRE_TOOL_USE` | `huginn/agent/callbacks.py:61`, `huginn/events/unified_bus.py:212` | `huginn/snapshot/integration.py:152`, `huginn/agents/factory.py:339`, `huginn/hooks/browser_gate_hook.py:78`, `huginn/hooks/physical_precheck.py:329`, `huginn/hooks/research_safety_hook.py:51` |
| `POST_TOOL_USE` | `huginn/agent/callbacks.py:113`, `huginn/agent/callbacks.py:100`, `huginn/events/unified_bus.py:255` | `huginn/snapshot/integration.py:153`, `huginn/agents/factory.py:295`, `huginn/agents/factory.py:301`, `huginn/agents/factory.py:313`, `huginn/hooks/science_hooks.py:744`, `huginn/hooks/science_hooks.py:745`, `huginn/hooks/science_hooks.py:750`, `huginn/hooks/science_hooks.py:751`, `huginn/hooks/science_hooks.py:753`, `huginn/hooks/science_hooks.py:755`, `huginn/hooks/science_hooks.py:756`, `huginn/hooks/science_hooks.py:758`, `huginn/hooks/science_hooks.py:759`, `huginn/hooks/science_hooks.py:760`, `huginn/hooks/science_hooks.py:761`, `huginn/hooks/science_hooks.py:762`, `huginn/hooks/science_hooks.py:763`, `huginn/hooks/science_hooks.py:764`, `huginn/hooks/science_hooks.py:765`, `huginn/hooks/science_hooks.py:766`, `huginn/hooks/science_hooks.py:748`, `huginn/hooks/science_hooks.py:771`, `huginn/hooks/science_hooks.py:778`, `huginn/hooks/science_hooks.py:785`, `huginn/hooks/science_hooks.py:795`, `huginn/hooks/science_hooks.py:796`, `huginn/hooks/science_hooks.py:804`, `huginn/hooks/science_hooks.py:805`, `huginn/hooks/science_hooks.py:835`, `huginn/hooks/science_hooks.py:864`, `huginn/hooks/unit_check.py:265` |
| `SESSION_START` | `huginn/events/unified_bus.py:138` | — |
| `SESSION_END` | `huginn/events/unified_bus.py:169` | — |
| `STOP` | `huginn/events/unified_bus.py:199` | `huginn/hooks/science_hooks.py:843` |
| `SUBAGENT_STOP` | `huginn/agents/subagent.py:373` | — |
| `PRE_COMPACT` | `huginn/agent/streaming.py:793` | — |
| `POST_COMPACT` | `huginn/events/unified_bus.py:365` | — |
| `USER_PROMPT_SUBMIT` | `huginn/agent/streaming.py:1511` | `huginn/agents/factory.py:322`, `huginn/agents/factory.py:327`, `huginn/agents/factory.py:332` |
| `POST_TOOL_USE_FAILURE` | `huginn/hooks/__init__.py:280` | — |

### 互斥违例 (mutually exclusive)

- 事件常量值两两不同 —— 无撞值.
- 触发/注册均用事件常量, 无绕过常量的字面量.

### 声明缺口

- `ALL_EVENTS` 与事件常量定义面双向一致.

诚实边界: 未知名字面量 (`register("vasp", …)` 这类别的注册表) 无法静态区分, 故不计入互斥违例; 但实现层 `trigger` 用 `_callbacks.get(event, [])` 静默吞掉未知名 —— 字面量拼错会变空触发而不报错, 这是触发点须用常量的理由.

## 事件面: 事件类型声明面 vs 发布面 vs 订阅面

声明面: `huginn/events/event_types.py` 的 21 个点分事件类型(`ALL_TYPES` 为自述非穷尽的辅助清单). 发布面: `AgentEvent(type=…)` / 内部 `_publish(_internal)` / `publish_generic_sync` / `publish_event` / `_emit_campaign`. 订阅面: `EventBus.subscribe(<类型>, cb)` (含 `ALL` 通配与 `for X in <集合>` 反解). `subscribed-only` = 订阅了却零发布 (订阅永不发生); `dead` = 声明了却零发布零订阅.

状态: `published`=有生产发布点; `subscribed-only`=有订阅但零生产发布 (订阅永不发生); `dead`=声明零发布零订阅

| 事件常量 | 值 | 生产发布 | 生产订阅 | 测试发布 | 测试订阅 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|
| `TOOL_CALL` | `tool.call` | 4 | 1 | 0 | 0 | `published` |  |
| `TOOL_RESULT` | `tool.result` | 2 | 1 | 0 | 0 | `published` |  |
| `TOOL_ERROR` | `tool.error` | 1 | 1 | 0 | 0 | `published` |  |
| `TOOL_BLOCKED` | `tool.blocked` | 1 | 1 | 0 | 0 | `published` |  |
| `COMPACT_START` | `compact.start` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `COMPACT_END` | `compact.end` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `CONTEXT_OVERFLOW` | `context.overflow` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `PIPELINE_SUGGEST` | `pipeline.suggest` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `PIPELINE_STAGE_CHANGE` | `pipeline.stage_change` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `CAMPAIGN_ITERATION` | `campaign.iteration` | 2 | 1 | 0 | 0 | `published` |  |
| `CAMPAIGN_REFINE` | `campaign.refine` | 1 | 1 | 0 | 0 | `published` |  |
| `CAMPAIGN_HYPOTHESIS` | `campaign.hypothesis` | 1 | 1 | 0 | 0 | `published` |  |
| `SNAPSHOT_TAKE` | `snapshot.take` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `SNAPSHOT_REVERT` | `snapshot.revert` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `QUALITY_CHECK` | `quality.check` | 1 | 1 | 0 | 0 | `published` |  |
| `HEAT_ENGINE_HEALTH` | `heat_engine.health` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `SESSION_START` | `session.start` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `SESSION_END` | `session.end` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `DECISION_POINT` | `decision.point` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `COST_NARRATIVE` | `cost.narrative` | 1 | 0 | 0 | 0 | `published` | 仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配) |
| `STEP_RETRY` | `agent.step.retrying` | 2 | 1 | 0 | 0 | `published` |  |

### 未声明类型 (发布/订阅了却无常量)

| 事件值 | 生产发布 | 生产订阅 | 发布点 | 订阅点 |
|---|---|---|---|---|
| `campaign.budget_exhausted` | 1 | 0 | `huginn/autoloop/cognitive_loop.py:2124` | — |
| `campaign.retry` | 1 | 1 | `huginn/autoloop/cognitive_loop.py:2483` | `huginn/events/audit_log.py:571` |
| `campaign.suspect` | 1 | 1 | `huginn/autoloop/cognitive_loop.py:2503` | `huginn/events/audit_log.py:571` |
| `cognitive.csm.transition` | 2 | 0 | `huginn/cognitive_engine.py:513`, `huginn/events/unified_bus.py:337` | — |
| `embedding.download.done` | 1 | 0 | `huginn/knowledge/store.py:159` | — |
| `embedding.download.error` | 3 | 0 | `huginn/knowledge/store.py:153`, `huginn/knowledge/store.py:163`, `huginn/knowledge/store.py:261` | — |
| `embedding.download.progress` | 1 | 0 | `huginn/knowledge/store.py:130` | — |
| `embedding.download.start` | 1 | 0 | `huginn/knowledge/store.py:135` | — |
| `event_bus.dropped` | 1 | 0 | `huginn/events/event_bus.py:154` | — |
| `llm.response` | 1 | 0 | `huginn/events/unified_bus.py:280` | — |
| `pet.mood` | 1 | 0 | `huginn/events/unified_bus.py:322` | — |
| `team.batch.start` | 1 | 0 | `huginn/agents/team.py:490` | — |
| `team.member.done` | 2 | 0 | `huginn/agents/team.py:452`, `huginn/agents/team.py:443` | — |
| `team.member.start` | 1 | 0 | `huginn/agents/team.py:403` | — |
| `team.member.tool` | 1 | 0 | `huginn/agents/team.py:430` | — |
| `team.run.done` | 1 | 0 | `huginn/agents/team.py:352` | — |
| `team.run.start` | 2 | 0 | `huginn/agents/team.py:313`, `huginn/agents/team.py:690` | — |

### 互斥违例 + 声明缺口 (mutually exclusive)

- 事件类型常量值两两不同 —— 无撞值.
- `ALL_TYPES` 与事件类型常量定义面双向一致.
- `ALL` 通配订阅 (收全量, 覆盖上表所有类型): `huginn/events/audit_log.py:501`

诚实边界: `EventBus.publish` **不校验**类型 (与钩子面 `register` 抛错相反), 任意点分字符串都能发, 故「未声明发布」只作候选提示; 订阅走变量/前缀匹配等间接形式时静态解析不到, 不计入.

## SSE 消费面: 后端帧名生产面 vs 前端 EventSource 监听面

后端有三条 SSE 通道, 帧名 (`event:` 行) 只在**发它的那条 EventSource** 上才可能被 `addEventListener(<帧名>)` 命中 —— 所以监听必须**按通道**核: `progress`(`/tasks/stream`, 帧名来自 `interaction/progress.py` 的字面 `event:` 行), `event_bus` (`/events/stream`, 帧名 = `AgentEvent.to_sse()` 写入的事件类型值), `pet` (`/events`, 只发无名帧, 无命名帧).

状态: `wired`=该通道确实发此帧名; `channel-mismatch`=帧名存在但走别的通道 (监听挂错 EventSource, 永不触发); `no-source`=后端任何 SSE 通道都不发此帧名; `external`=非 SSE 通道 (window/document 等 DOM 事件), 不计入

| 通道 | URL 片段 | 生产帧名 |
|---|---|---|
| `progress` | `progress` | `campaign`, `heartbeat`, `snapshot`, `update` |
| `event_bus` | `event_bus` | `agent.step.retrying`, `campaign.budget_exhausted`, `campaign.hypothesis`, `campaign.iteration`, `campaign.refine`, `campaign.retry`, `campaign.suspect`, `cognitive.csm.transition`, `compact.end`, `compact.start`, `context.overflow`, `cost.narrative`, `decision.point`, `embedding.download.done`, `embedding.download.error`, `embedding.download.progress`, `embedding.download.start`, `event_bus.dropped`, `heat_engine.health`, `llm.response`, `pet.mood`, `pipeline.stage_change`, `pipeline.suggest`, `quality.check`, `session.end`, `session.start`, `snapshot.revert`, `snapshot.take`, `team.batch.start`, `team.member.done`, `team.member.start`, `team.member.tool`, `team.run.done`, `team.run.start`, `tool.blocked`, `tool.call`, `tool.error`, `tool.result` |
| `pet` | `pet` | — (无名帧) |

### 前端帧监听 (按通道归属)

| 帧名 | 通道 | 状态 | 位置 | 备注 |
|---|---|---|---|---|
| `mousemove` | — | `external` | `desktop/src/App.tsx:189` | 非 SSE 通道 (DOM 事件), 不计入 |
| `mouseup` | — | `external` | `desktop/src/App.tsx:190` | 非 SSE 通道 (DOM 事件), 不计入 |
| `keydown` | — | `external` | `desktop/src/App.tsx:1104` | 非 SSE 通道 (DOM 事件), 不计入 |
| `resize` | — | `external` | `desktop/src/App.tsx:1113` | 非 SSE 通道 (DOM 事件), 不计入 |
| `storage` | — | `external` | `desktop/src/Pet.tsx:773` | 非 SSE 通道 (DOM 事件), 不计入 |
| `pointermove` | — | `external` | `desktop/src/Pet.tsx:1070` | 非 SSE 通道 (DOM 事件), 不计入 |
| `keydown` | — | `external` | `desktop/src/Pet.tsx:1071` | 非 SSE 通道 (DOM 事件), 不计入 |
| `keydown` | — | `external` | `desktop/src/components/Modal.tsx:30` | 非 SSE 通道 (DOM 事件), 不计入 |
| `mousedown` | — | `external` | `desktop/src/components/SandboxPanel.tsx:194` | 非 SSE 通道 (DOM 事件), 不计入 |
| `mousedown` | — | `external` | `desktop/src/components/SaveToMemoryButton.tsx:38` | 非 SSE 通道 (DOM 事件), 不计入 |
| `click` | — | `external` | `desktop/src/components/panels/ChatPanel.tsx:488` | 非 SSE 通道 (DOM 事件), 不计入 |
| `click` | — | `external` | `desktop/src/components/panels/ThreadsPanel.tsx:46` | 非 SSE 通道 (DOM 事件), 不计入 |
| `visibilitychange` | — | `external` | `desktop/src/hooks/useChatAndConnection.ts:1300` | 非 SSE 通道 (DOM 事件), 不计入 |
| `keydown` | — | `external` | `desktop/src/hooks/useFocusTrap.ts:70` | 非 SSE 通道 (DOM 事件), 不计入 |
| `change` | — | `external` | `desktop/src/hooks/useTheme.ts:45` | 非 SSE 通道 (DOM 事件), 不计入 |
| `snapshot` | `progress` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1670` |  |
| `update` | `progress` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1671` |  |
| `campaign` | `progress` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1672` |  |
| `embedding.download.start` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1745` |  |
| `embedding.download.progress` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1745` |  |
| `embedding.download.done` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1745` |  |
| `embedding.download.error` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1745` |  |
| `team.run.start` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1807` |  |
| `team.run.done` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1807` |  |
| `team.batch.start` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1807` |  |
| `team.member.start` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1807` |  |
| `team.member.tool` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1807` |  |
| `team.member.done` | `event_bus` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1807` |  |

### 生产帧名零前端监听 (候选)

- `progress` / `heartbeat`
- `event_bus` / `agent.step.retrying`
- `event_bus` / `campaign.budget_exhausted`
- `event_bus` / `campaign.hypothesis` (经 payload 字段消费)
- `event_bus` / `campaign.iteration` (经 payload 字段消费)
- `event_bus` / `campaign.refine` (经 payload 字段消费)
- `event_bus` / `campaign.retry` (经 payload 字段消费)
- `event_bus` / `campaign.suspect` (经 payload 字段消费)
- `event_bus` / `cognitive.csm.transition`
- `event_bus` / `compact.end`
- `event_bus` / `compact.start`
- `event_bus` / `context.overflow`
- `event_bus` / `cost.narrative`
- `event_bus` / `decision.point`
- `event_bus` / `event_bus.dropped`
- `event_bus` / `heat_engine.health` (经 payload 字段消费)
- `event_bus` / `llm.response`
- `event_bus` / `pet.mood`
- `event_bus` / `pipeline.stage_change`
- `event_bus` / `pipeline.suggest`
- `event_bus` / `quality.check`
- `event_bus` / `session.end`
- `event_bus` / `session.start`
- `event_bus` / `snapshot.revert`
- `event_bus` / `snapshot.take`
- `event_bus` / `tool.blocked`
- `event_bus` / `tool.call`
- `event_bus` / `tool.error`
- `event_bus` / `tool.result`

### 前端 payload 字段匹配的事件名

| 事件名 | 消费通道 | 生产面 | 位置 |
|---|---|---|---|
| `campaign.iteration` | — | `bus` | `desktop/src/components/IterationTimeline.tsx:57` |
| `campaign.hypothesis` | — | `bus` | `desktop/src/components/IterationTimeline.tsx:66` |
| `campaign.retry` | — | `bus` | `desktop/src/components/IterationTimeline.tsx:69` |
| `campaign.suspect` | — | `bus` | `desktop/src/components/IterationTimeline.tsx:73` |
| `campaign.refine` | — | `bus` | `desktop/src/components/IterationTimeline.tsx:77` |
| `heat_engine.health` | `progress` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1644` |
| `plan.exec_start` | `progress` | `campaign` | `desktop/src/hooks/useChatAndConnection.ts:1659` |
| `plan.exec_complete` | `progress` | `campaign` | `desktop/src/hooks/useChatAndConnection.ts:1661` |
| `embedding.download.start` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1722` |
| `embedding.download.progress` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1725` |
| `embedding.download.done` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1728` |
| `embedding.download.error` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1735` |
| `team.run.start` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1761` |
| `team.batch.start` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1764` |
| `team.member.start` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1773` |
| `team.member.tool` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1780` |
| `team.member.done` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1788` |
| `team.run.done` | `event_bus` | `bus` | `desktop/src/hooks/useChatAndConnection.ts:1797` |

生产面取值: `bus` = 总线生产发布 (双通道之一); `campaign` = `emit_campaign_event(event_type="…")` 静态可见的字面量; `declared` = 已声明常量但未观测到生产发布; `unknown` = 静态不可见 (如 `f"campaign.{name}"` 动态拼接).

诚实边界: 前端是 TS, 本工具只做**行级**匹配 (`new EventSource` / `addEventListener` / `case …:` / `=== …`), 不做 TS 语法分析 —— 经变量中转的帧名、`es.onmessage` 的无名帧、动态拼接的通道 URL 都解析不到; campaign payload 里 `f"campaign.{name}"` 这类动态名同样不可穷尽, 故 `unknown` 只提示不判死.

## WS 消费面: 后端 WS 帧名生产面 vs 前端 WebSocket 判别面

SSE 消费面只核单向 (后端发帧 → 前端 `addEventListener`), WebSocket 是**双向**的: 后端 `send_json({"type": …})` 发帧前端 `switch (data.type)` 收, 前端`send({type: …})` 发请求后端分发表收. 本面按**端点通道** `agent`(`/ws/agent`, 唯一按 `type` 判别的通道) / `terminal` / `viewer3d` / `hpc` 分列: 帧名只在**发它的那条 WS 上**才可能命中, 故挂在别处 = 永不触发.

状态: `wired`=后端该通道确实发此帧名, 前端有 type 判别; `dynamic`=已声明帧, 后端经变量透传转发 (静态生产面不可穷尽, 只提示); `no-source`=后端该通道不发此帧名, 且未声明 (前端 case 永不命中); `handled`=后端该通道分发表认此入站类型; `unhandled`=后端该通道分发面无此入站类型 (回 error 帧); `field-probing`=前端只按字段取值, 不按 type 判别 (不计入)

| 通道 | URL 片段 | type 判别 | 生产帧名 | 入站类型 |
|---|---|---|---|---|
| `agent` | `/ws/agent` | 是 | `approval_request`, `auto_approve_set`, `auto_checkpoint`, `citations`, `clarification_request`, `context_compacted`, `decision_resolved`, `done`, `error`, `exploration_result`, `governance`, `guide_ack`, `hook_warning`, `pet_update`, `ping`, `plan`, `plan_result`, `pong`, `reasoning_delta`, `sediment`, `side_question_pending`, `suggest_mode_set`, `task_progress`, `text_delta`, `tool_auto_approved`, `tool_call`, `tool_result` | `approval_response`, `clarification_response`, `decision_response`, `explore_start`, `guide`, `ping`, `plan_confirm`, `pong`, `set_auto_approve`, `set_suggest_mode`, `suggest_response`, `user_input` |
| `terminal` | `/ws/terminal` | 否 (field-probing) | `closed`, `error`, `output`, `ready` | `input`, `resize`, `signal` |
| `viewer3d` | `/ws/viewer3d` | 否 (field-probing) | `error`, `force_ack`, `frame`, `paused`, `pong`, `resumed`, `structure` | `force`, `hello`, `pause`, `ping`, `resume` |
| `hpc` | `/ws/hpc/jobs` | 否 (field-probing) | `done`, `error`, `info`, `output`, `status` | — |

### 生产帧名 × 前端判别 (agent 通道)

| 帧名 | 已声明 (WSMessage) | 前端有 type 判别 |
|---|---|---|
| `approval_request` | 是 | 是 |
| `auto_approve_set` | 是 | 是 |
| `auto_checkpoint` | 是 | 是 |
| `citations` | 是 | 是 |
| `clarification_request` | 是 | 是 |
| `context_compacted` | 是 | 是 |
| `decision_resolved` | 否 | 否 |
| `done` | 是 | 是 |
| `error` | 是 | 是 |
| `exploration_result` | 是 | 是 |
| `governance` | 是 | 是 |
| `guide_ack` | 是 | 是 |
| `hook_warning` | 是 | 是 |
| `pet_update` | 是 | 是 |
| `ping` | 是 | 是 |
| `plan` | 是 | 是 |
| `plan_result` | 是 | 是 |
| `pong` | 是 | 是 |
| `reasoning_delta` | 是 | 是 |
| `sediment` | 是 | 是 |
| `side_question_pending` | 是 | 是 |
| `suggest_mode_set` | 是 | 是 |
| `task_progress` | 是 | 是 |
| `text_delta` | 是 | 是 |
| `tool_auto_approved` | 是 | 是 |
| `tool_call` | 是 | 是 |
| `tool_result` | 是 | 是 |

### 前端 server→client 判别点

| 帧名 | 通道 | 状态 | 位置 | 备注 |
|---|---|---|---|---|
| `mode_banner` | `agent` | `dynamic` | `desktop/src/hooks/useChatAndConnection.ts:919` | 已声明帧, 后端经变量透传转发 (静态生产面不可穷尽) |
| `trust_update` | `agent` | `dynamic` | `desktop/src/hooks/useChatAndConnection.ts:930` | 已声明帧, 后端经变量透传转发 (静态生产面不可穷尽) |
| `budget_update` | `agent` | `dynamic` | `desktop/src/hooks/useChatAndConnection.ts:934` | 已声明帧, 后端经变量透传转发 (静态生产面不可穷尽) |
| `budget_escalation` | `agent` | `dynamic` | `desktop/src/hooks/useChatAndConnection.ts:938` | 已声明帧, 后端经变量透传转发 (静态生产面不可穷尽) |
| `suggest_code` | `agent` | `dynamic` | `desktop/src/hooks/useChatAndConnection.ts:942` | 已声明帧, 后端经变量透传转发 (静态生产面不可穷尽) |
| `risk_threshold` | `agent` | `dynamic` | `desktop/src/hooks/useChatAndConnection.ts:955` | 已声明帧, 后端经变量透传转发 (静态生产面不可穷尽) |
| `text_delta` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:671` |  |
| `reasoning_delta` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:681` |  |
| `done` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:685` |  |
| `error` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:685` |  |
| `error` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:691` |  |
| `tool_call` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:696` |  |
| `tool_result` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:708` |  |
| `task_progress` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:722` |  |
| `plan` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:731` |  |
| `citations` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:731` |  |
| `reasoning_delta` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:744` |  |
| `text_delta` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:749` |  |
| `done` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:755` |  |
| `error` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:775` |  |
| `tool_call` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:788` |  |
| `tool_result` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:803` |  |
| `auto_checkpoint` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:820` |  |
| `exploration_result` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:823` |  |
| `pong` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:836` |  |
| `guide_ack` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:838` |  |
| `context_compacted` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:842` |  |
| `plan` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:857` |  |
| `plan_result` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:881` |  |
| `clarification_request` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:893` |  |
| `suggest_mode_set` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:951` |  |
| `side_question_pending` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:959` |  |
| `citations` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:974` |  |
| `task_progress` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:991` |  |
| `sediment` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1150` |  |
| `approval_request` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1162` |  |
| `tool_auto_approved` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1173` |  |
| `auto_approve_set` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1177` |  |
| `hook_warning` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1180` |  |
| `ping` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1190` |  |
| `pet_update` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1195` |  |
| `governance` | `agent` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1204` |  |

### 前端 client→server 发送点

| 类型 | 通道 | 状态 | 位置 | 备注 |
|---|---|---|---|---|
| `approval_response` | `agent` | `handled` | `desktop/src/Pet.tsx:1253` |  |
| `plan_confirm` | `agent` | `handled` | `desktop/src/components/panels/ChatPanel.tsx:1581` |  |
| `plan_confirm` | `agent` | `handled` | `desktop/src/components/panels/ChatPanel.tsx:1599` |  |
| `input` | `terminal` | `handled` | `desktop/src/components/panels/TerminalPanel.tsx:57` |  |
| `pong` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1192` |  |
| `user_input` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1405` |  |
| `guide` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1437` |  |
| `user_input` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1477` |  |
| `clarification_response` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1542` |  |
| `approval_response` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1562` |  |
| `set_auto_approve` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1574` |  |
| `set_suggest_mode` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1585` |  |
| `suggest_response` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1596` |  |
| `decision_response` | `agent` | `handled` | `desktop/src/hooks/useChatAndConnection.ts:1611` |  |
| `ping` | `agent` | `handled` | `desktop/src/lib/ws-client.ts:258` |  |

### 生产帧名零前端判别 (候选)

- `agent` / `decision_resolved` — ✅ 已确认有意: 决策点裁决的 ack 帧; 前端乐观清空 pendingDecisionPoint, 不消费它

### 未进 WSMessage 判别联合的生产帧 (候选登记)

- `agent` / `decision_resolved` — ✅ 已确认有意: 同一 ack 帧, 前端不消费故未进 WSMessage 联合 (与 zero-consumer 同源)

### 已声明但既非生产帧也非入站请求 (候选)

- `clarification_response` — ✅ 已确认有意: WSMessage 联合把双向帧并在一起; 它是 client→server 请求帧
- `plan_confirm` — ✅ 已确认有意: WSMessage 联合把双向帧并在一起; 它是 client→server 请求帧

### 入站类型无前端发送者 (候选)

- `agent` / `explore_start` — ✅ 已确认有意: 探索编排入口; 面向非桌面客户端 (仓库内无发送者), 保留为公开 WS API
- `terminal` / `resize` — ✅ 已确认有意: 终端尺寸同步; 桌面用普通输入框 (无 xterm fit), 面向外部客户端
- `terminal` / `signal` — ✅ 已确认有意: 终端信号 (Ctrl-C 等); 桌面未启用, 面向外部客户端

诚实边界: 前端 TS 与后端 `send_json(变量)` 都只做**静态**扫描 —— 经变量透传的入站类型 (如 `_ws_send(dict(state))` 转发的 agent 循环类型化事件 `mode_banner` / `trust_update` / `budget_update` 等) 生产面**不可穷尽**, 故只提示不判死; 前端terminal/hpc 按字段 (`data`/`output`) 取值而非按 `type` 判别, 记作 field-probing; viewer3d 无桌面前端 (由外部客户端驱动), 其入站不判 phantom.

## HTTP API 消费面: 后端路由注册面 vs 前端 api.* 调用面

前八面核 agent 内部契约, SSE/WS 消费面核流式推送, 本面补齐**请求-响应**第三块传输拼图: 后端 `huginn/routes/*.py` 的 `@router.<method>("<path>")` 是**注册生产面**, 前端 `desktop/src` 的 `api.get/post/put/patch/del/getBlob/upload*/search(...)` 是**调用消费面**. 桌面只是 HTTP API 的**一个**消费者 (外部客户端 / CLI / 测试也调), 故**只把「前端 → 后端」方向当硬契约**: 前端调了后端没注册的路径 = 404 死链, 方法对不上 = 405. 匹配只看**实存端点** (挂进 `ALL_ROUTERS` 的模块), 未挂载模块的端点另在挂载面报, 不重复计.

状态: `wired`=前端调用的方法与路径后端已注册; `method-mismatch`=路径已注册但无此方法 (405, 调用必失败); `no-source`=后端无此路径 (404 死链); `external`=绝对 URL / 非后端路径, 不计入

端点: 后端注册 **365** 个 (实存 **365** 个; 另有 4 个 WebSocket 端点归 WS 消费面); 桌面调用命中 **146** 个; 前端调用点 **171** 处.

### 前端调用点 (按状态)

| 方法 | 路径 | 状态 | 位置 | 备注 |
|---|---|---|---|---|
| `GET` | `/provenance/recent?n=50` | `wired` | `desktop/src/App.tsx:342` |  |
| `POST` | `/checkpoints` | `wired` | `desktop/src/App.tsx:352` |  |
| `GET` | `/checkpoints/${cpId}/diff` | `wired` | `desktop/src/App.tsx:363` |  |
| `POST` | `/checkpoints/${cpId}/accept` | `wired` | `desktop/src/App.tsx:373` |  |
| `POST` | `/checkpoints/${cpId}/reject` | `wired` | `desktop/src/App.tsx:386` |  |
| `GET` | `/side/pending` | `wired` | `desktop/src/App.tsx:525` |  |
| `POST` | `/side` | `wired` | `desktop/src/App.tsx:535` |  |
| `POST` | `/side` | `wired` | `desktop/src/App.tsx:546` |  |
| `DELETE` | `/side` | `wired` | `desktop/src/App.tsx:557` |  |
| `GET` | `/unified/models` | `wired` | `desktop/src/App.tsx:576` |  |
| `POST` | `/unified/derive` | `wired` | `desktop/src/App.tsx:588` |  |
| `POST` | `/unified/solve` | `wired` | `desktop/src/App.tsx:597` |  |
| `POST` | `/unified/plot` | `wired` | `desktop/src/App.tsx:606` |  |
| `GET` | `/workflows` | `wired` | `desktop/src/App.tsx:737` |  |
| `POST` | `/pet/feed` | `wired` | `desktop/src/Pet.tsx:1354` |  |
| `POST` | `/pet/pet` | `wired` | `desktop/src/Pet.tsx:1365` |  |
| `GET` | `/v1/fs/search` | `wired` | `desktop/src/components/CommandPalette.tsx:50` |  |
| `GET` | `/credentials?kind=ssh` | `wired` | `desktop/src/components/CredentialsPanel.tsx:101` |  |
| `GET` | `/credentials?kind=llm` | `wired` | `desktop/src/components/CredentialsPanel.tsx:102` |  |
| `GET` | `/credentials` | `wired` | `desktop/src/components/CredentialsPanel.tsx:103` |  |
| `GET` | `/config/providers` | `wired` | `desktop/src/components/CredentialsPanel.tsx:119` |  |
| `PUT` | `/credentials/${editing.id}` | `wired` | `desktop/src/components/CredentialsPanel.tsx:178` |  |
| `POST` | `/credentials` | `wired` | `desktop/src/components/CredentialsPanel.tsx:179` |  |
| `PUT` | `/credentials/${editing.id}` | `wired` | `desktop/src/components/CredentialsPanel.tsx:202` |  |
| `POST` | `/credentials` | `wired` | `desktop/src/components/CredentialsPanel.tsx:203` |  |
| `POST` | `/config/models/test` | `wired` | `desktop/src/components/CredentialsPanel.tsx:219` |  |
| `DELETE` | `/credentials/${id}` | `wired` | `desktop/src/components/CredentialsPanel.tsx:240` |  |
| `POST` | `/credentials/${id}/set-default` | `wired` | `desktop/src/components/CredentialsPanel.tsx:247` |  |
| `POST` | `/credentials/${id}/test` | `wired` | `desktop/src/components/CredentialsPanel.tsx:256` |  |
| `POST` | `/credentials/import-from-config` | `wired` | `desktop/src/components/CredentialsPanel.tsx:269` |  |
| `POST` | `/credentials/${apiKeyForm.service}` | `wired` | `desktop/src/components/CredentialsPanel.tsx:569` |  |
| `GET` | `/credentials/${s.service}/test` | `wired` | `desktop/src/components/CredentialsPanel.tsx:614` |  |
| `DELETE` | `/credentials/${s.service}` | `wired` | `desktop/src/components/CredentialsPanel.tsx:636` |  |
| `GET` | `/personas/${personaName}/emotion` | `wired` | `desktop/src/components/EmotionTracker.tsx:133` |  |
| `GET` | `/metrics` | `wired` | `desktop/src/components/MetricsBar.tsx:78` |  |
| `GET` | `/health` | `wired` | `desktop/src/components/MetricsBar.tsx:79` |  |
| `GET` | `/memory?category=notebook&limit=200` | `wired` | `desktop/src/components/Notebook.tsx:103` |  |
| `POST` | `/memory/search` | `wired` | `desktop/src/components/Notebook.tsx:128` |  |
| `POST` | `/memory` | `wired` | `desktop/src/components/Notebook.tsx:180` |  |
| `DELETE` | `/memory/${id}` | `wired` | `desktop/src/components/Notebook.tsx:198` |  |
| `POST` | `/tools/materials_database_tool` | `wired` | `desktop/src/components/PeriodicTable.tsx:188` |  |
| `GET` | `/personas` | `wired` | `desktop/src/components/PersonaManager.tsx:82` |  |
| `GET` | `/personas/${encodeURIComponent(name)}` | `wired` | `desktop/src/components/PersonaManager.tsx:99` |  |
| `POST` | `/personas/${encodeURIComponent(name)}/switch` | `wired` | `desktop/src/components/PersonaManager.tsx:125` |  |
| `PATCH` | `/personas/${encodeURIComponent(name)}/default` | `wired` | `desktop/src/components/PersonaManager.tsx:135` |  |
| `DELETE` | `/personas/${encodeURIComponent(name)}` | `wired` | `desktop/src/components/PersonaManager.tsx:150` |  |
| `POST` | `/personas` | `wired` | `desktop/src/components/PersonaManager.tsx:168` |  |
| `GET` | `/hpc/jobs` | `wired` | `desktop/src/components/RemoteJobsPanel.tsx:47` |  |
| `POST` | `/hpc/jobs/${localId}/refresh` | `wired` | `desktop/src/components/RemoteJobsPanel.tsx:65` |  |
| `POST` | `/hpc/jobs/${localId}/cancel` | `wired` | `desktop/src/components/RemoteJobsPanel.tsx:77` |  |
| `GET` | `/tasks` | `wired` | `desktop/src/components/RuntimeStatusPanel.tsx:95` |  |
| `GET` | `/inbox` | `wired` | `desktop/src/components/RuntimeStatusPanel.tsx:98` |  |
| `GET` | `/autoloop/resumable` | `wired` | `desktop/src/components/RuntimeStatusPanel.tsx:101` |  |
| `GET` | `/tool-economy` | `wired` | `desktop/src/components/RuntimeStatusPanel.tsx:102` |  |
| `POST` | `/autoloop/resume` | `wired` | `desktop/src/components/RuntimeStatusPanel.tsx:133` |  |
| `POST` | `/inbox/${encodeURIComponent(item.id)}/resolve` | `wired` | `desktop/src/components/RuntimeStatusPanel.tsx:150` |  |
| `POST` | `/sandbox/execute` | `wired` | `desktop/src/components/SandboxPanel.tsx:222` |  |
| `POST` | `/memory` | `wired` | `desktop/src/components/SaveToMemoryButton.tsx:46` |  |
| `GET` | `/metrics` | `wired` | `desktop/src/components/StatusBar.tsx:59` |  |
| `GET` | `/health` | `wired` | `desktop/src/components/StatusBar.tsx:60` |  |
| `GET` | `/fs/branch` | `wired` | `desktop/src/components/StatusBar.tsx:86` |  |
| `POST` | `/tools/structure_tool` | `wired` | `desktop/src/components/StructureViewer.tsx:235` |  |
| `POST` | `/viewer3d/load` | `wired` | `desktop/src/components/StructureViewer.tsx:246` |  |
| `GET` | `/workflows` | `wired` | `desktop/src/components/SweepDashboard.tsx:176` |  |
| `POST` | `/workflows/execute` | `wired` | `desktop/src/components/SweepDashboard.tsx:270` |  |
| `POST` | `/v1/fs/upload` | `wired` | `desktop/src/components/panels/ChatPanel.tsx:574` |  |
| `GET` | `/events/recent?n=400` | `wired` | `desktop/src/components/panels/EventAuditPanel.tsx:55` |  |
| `GET` | `/credentials/defaults` | `wired` | `desktop/src/components/panels/FilesPanel.tsx:48` |  |
| `POST` | `/transfer/web/upload` | `wired` | `desktop/src/components/panels/FilesPanel.tsx:65` |  |
| `GET` | `/transfer/browse?${params}` | `wired` | `desktop/src/components/panels/FilesPanel.tsx:88` |  |
| `POST` | `/transfer/sync` | `wired` | `desktop/src/components/panels/FilesPanel.tsx:102` |  |
| `GET` | `/transfer/web/download?${params}` | `wired` | `desktop/src/components/panels/FilesPanel.tsx:133` |  |
| `GET` | `/projects` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:60` |  |
| `GET` | `/threads` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:75` |  |
| `GET` | `/knowledge` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:76` |  |
| `POST` | `/projects` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:82` |  |
| `DELETE` | `/projects/${pid}` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:95` |  |
| `PATCH` | `/projects/${selected.id}` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:105` |  |
| `PATCH` | `/projects/${selected.id}` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:119` |  |
| `POST` | `/projects/${selected.id}/threads` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:130` |  |
| `DELETE` | `/projects/${selected.id}/threads/${tid}` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:142` |  |
| `POST` | `/projects/${selected.id}/knowledge` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:153` |  |
| `DELETE` | `/projects/${selected.id}/knowledge/${docId}` | `wired` | `desktop/src/components/panels/ResearchProjectPanel.tsx:165` |  |
| `GET` | `/config/local-models?${params.toString()}` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:49` |  |
| `GET` | `/models/caps` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:407` |  |
| `POST` | `/credentials` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:571` |  |
| `POST` | `/pet/reset` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:1102` |  |
| `POST` | `/config/encrypt` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:1157` |  |
| `GET` | `/export/status` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:1210` |  |
| `POST` | `/export/all` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:1240` |  |
| `POST` | `/export/memory` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:1257` |  |
| `POST` | `/export/knowledge` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:1274` |  |
| `POST` | `/import/all` | `wired` | `desktop/src/components/panels/SettingsPanel.tsx:1303` |  |
| `POST` | `/skills/execute` | `wired` | `desktop/src/components/panels/SkillsPanel.tsx:147` |  |
| `GET` | `/v1/todos` | `wired` | `desktop/src/components/panels/TodoPanel.tsx:27` |  |
| `PUT` | `/v1/todos` | `wired` | `desktop/src/components/panels/TodoPanel.tsx:48` |  |
| `POST` | `/tools/${name}` | `wired` | `desktop/src/components/panels/ToolsPanel.tsx:151` |  |
| `GET` | `/threads/${threadId}/messages` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:340` |  |
| `GET` | `/threads/${threadId}/state` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:357` |  |
| `GET` | `/threads?include_archived=${includeArchived ? "true" : "false"}` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:377` |  |
| `POST` | `/threads` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:388` |  |
| `PATCH` | `/threads/${id}` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:407` |  |
| `DELETE` | `/threads/${id}` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:416` |  |
| `POST` | `/threads/${id}/fork` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:431` |  |
| `POST` | `/threads/${id}/archive` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:450` |  |
| `POST` | `/threads/${id}/unarchive` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:463` |  |
| `GET` | `/threads/${tid}/messages` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1259` |  |
| `GET` | `/tools` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1319` |  |
| `GET` | `/skills` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1324` |  |
| `GET` | `/personas` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1332` |  |
| `GET` | `/personas/${config.persona}/emotion` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1354` |  |
| `POST` | `/agents/default/interrupt` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1493` |  |
| `POST` | `/agents/default/interrupt` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1518` |  |
| `POST` | `/agents/default/interrupt` | `wired` | `desktop/src/hooks/useChatAndConnection.ts:1530` |  |
| `POST` | `/config` | `wired` | `desktop/src/hooks/useConfig.ts:60` |  |
| `GET` | `/config/active-model` | `wired` | `desktop/src/hooks/useConfig.ts:179` |  |
| `POST` | `/config/active-model` | `wired` | `desktop/src/hooks/useConfig.ts:195` |  |
| `GET` | `/config/model-tier` | `wired` | `desktop/src/hooks/useConfig.ts:212` |  |
| `POST` | `/config/model-tier` | `wired` | `desktop/src/hooks/useConfig.ts:225` |  |
| `POST` | `/personas/${personaName}/switch` | `wired` | `desktop/src/hooks/useConfig.ts:245` |  |
| `GET` | `/credentials?kind=llm` | `wired` | `desktop/src/hooks/useConfig.ts:261` |  |
| `POST` | `/hpc/test` | `wired` | `desktop/src/hooks/useHPC.ts:31` |  |
| `POST` | `/hpc/submit` | `wired` | `desktop/src/hooks/useHPC.ts:53` |  |
| `POST` | `/hpc/status` | `wired` | `desktop/src/hooks/useHPC.ts:86` |  |
| `GET` | `/threads/${threadId}/events${qs ? ` | `wired` | `desktop/src/hooks/useIncrementalMessages.ts:87` |  |
| `GET` | `/knowledge` | `wired` | `desktop/src/hooks/useKnowledge.ts:42` |  |
| `POST` | `/knowledge/upload` | `wired` | `desktop/src/hooks/useKnowledge.ts:57` |  |
| `POST` | `/knowledge/query` | `wired` | `desktop/src/hooks/useKnowledge.ts:91` |  |
| `POST` | `/document/parse` | `wired` | `desktop/src/hooks/useKnowledge.ts:115` |  |
| `GET` | `/document/${docId}/graph` | `wired` | `desktop/src/hooks/useKnowledge.ts:141` |  |
| `GET` | `/knowledge/${doc.doc_id}/chunks` | `wired` | `desktop/src/hooks/useKnowledge.ts:163` |  |
| `GET` | `/knowledge/${docId}/images` | `wired` | `desktop/src/hooks/useKnowledge.ts:191` |  |
| `POST` | `/knowledge/report` | `wired` | `desktop/src/hooks/useKnowledge.ts:206` |  |
| `DELETE` | `/knowledge/${docId}` | `wired` | `desktop/src/hooks/useKnowledge.ts:253` |  |
| `POST` | `/knowledge/query` | `wired` | `desktop/src/hooks/useKnowledge.ts:264` |  |
| `POST` | `/knowledge/ingest-url` | `wired` | `desktop/src/hooks/useKnowledge.ts:279` |  |
| `GET` | `/provenance/dag?n=50` | `wired` | `desktop/src/hooks/useKnowledge.ts:296` |  |
| `GET` | `/memory?${params.toString()}` | `wired` | `desktop/src/hooks/useMemory.ts:38` |  |
| `GET` | `/memory/stats` | `wired` | `desktop/src/hooks/useMemory.ts:54` |  |
| `POST` | `/memory/search` | `wired` | `desktop/src/hooks/useMemory.ts:68` |  |
| `POST` | `/memory` | `wired` | `desktop/src/hooks/useMemory.ts:87` |  |
| `DELETE` | `/memory/${id}` | `wired` | `desktop/src/hooks/useMemory.ts:115` |  |
| `PATCH` | `/memory/${id}` | `wired` | `desktop/src/hooks/useMemory.ts:127` |  |
| `POST` | `/memory/promote/${id}` | `wired` | `desktop/src/hooks/useMemory.ts:143` |  |
| `POST` | `/memory/prune` | `wired` | `desktop/src/hooks/useMemory.ts:159` |  |
| `POST` | `/memory/sync-md` | `wired` | `desktop/src/hooks/useMemory.ts:172` |  |
| `GET` | `/memory/layers` | `wired` | `desktop/src/hooks/useMemory.ts:188` |  |
| `GET` | `/mcp/servers` | `wired` | `desktop/src/hooks/usePlugins.ts:24` |  |
| `GET` | `/mcp/servers/discover` | `wired` | `desktop/src/hooks/usePlugins.ts:33` |  |
| `POST` | `/mcp/servers/connect` | `wired` | `desktop/src/hooks/usePlugins.ts:43` |  |
| `POST` | `/mcp/servers/${name}/disconnect` | `wired` | `desktop/src/hooks/usePlugins.ts:61` |  |
| `POST` | `/mcp/servers/${name}/reconnect` | `wired` | `desktop/src/hooks/usePlugins.ts:78` |  |
| `POST` | `/mcp/tools/${serverName}/call` | `wired` | `desktop/src/hooks/usePlugins.ts:91` |  |
| `GET` | `/project-context` | `wired` | `desktop/src/hooks/useProject.ts:22` |  |
| `POST` | `/project-context` | `wired` | `desktop/src/hooks/useProject.ts:34` |  |
| `GET` | `/codebase` | `wired` | `desktop/src/hooks/useProject.ts:50` |  |
| `POST` | `/codebase/index` | `wired` | `desktop/src/hooks/useProject.ts:60` |  |
| `POST` | `/codebase/search` | `wired` | `desktop/src/hooks/useProject.ts:78` |  |
| `POST` | `/team/v2/plan` | `wired` | `desktop/src/hooks/useTeam.ts:24` |  |
| `POST` | `/team/plan` | `wired` | `desktop/src/hooks/useTeam.ts:31` |  |
| `POST` | `/team/v2/run` | `wired` | `desktop/src/hooks/useTeam.ts:57` |  |
| `POST` | `/team/run` | `wired` | `desktop/src/hooks/useTeam.ts:64` |  |
| `POST` | `/team/v2/fusion` | `wired` | `desktop/src/hooks/useTeam.ts:90` |  |
| `GET` | `/v1/fs/list` | `wired` | `desktop/src/hooks/useWorkspace.tsx:42` |  |
| `GET` | `/v1/fs/read` | `wired` | `desktop/src/hooks/useWorkspace.tsx:72` |  |
| `PUT` | `/v1/fs/write` | `wired` | `desktop/src/hooks/useWorkspace.tsx:125` |  |
| `POST` | `/v1/fs/mkdir` | `wired` | `desktop/src/hooks/useWorkspace.tsx:137` |  |
| `PUT` | `/v1/fs/rename` | `wired` | `desktop/src/hooks/useWorkspace.tsx:147` |  |
| `DELETE` | `/v1/fs/delete` | `wired` | `desktop/src/hooks/useWorkspace.tsx:163` |  |
| `POST` | `/v1/fs/open` | `wired` | `desktop/src/hooks/useWorkspace.tsx:177` |  |
| `GET` | `/v1/fs/cwd` | `wired` | `desktop/src/hooks/useWorkspace.tsx:187` |  |

### 前端调用无源 (404 死链) / 方法不符 (405)

- 无 —— 每个前端调用都命中后端已注册的方法+路径.

### 同一 method+path 被多个**已挂载**模块注册 (路由遮蔽)

- 无 —— 每个 method+path 唯一注册.

### 路由挂载面 (ALL_ROUTERS ↔ 各模块 APIRouter)

- 已挂载: 58 个 router 变量
- 未挂载: 无 —— 每个定义路由的模块都被 `ALL_ROUTERS` 挂上.

### 桌面零调用的路由模块 (候选, 只算已挂载模块)

以下 23 个模块的端点**全部**无桌面调用 —— HTTP API 面向外部客户端 / CLI / 测试, 零调用是**结构性常态**, 非缺陷; 列此仅供「哪些面桌面根本没接」参考:

| 模块 | 端点数 |
|---|---|
| `huginn/routes/admin.py` | 2 |
| `huginn/routes/advisor.py` | 3 |
| `huginn/routes/auth.py` | 5 |
| `huginn/routes/bench.py` | 2 |
| `huginn/routes/bot.py` | 13 |
| `huginn/routes/catalog.py` | 3 |
| `huginn/routes/coder.py` | 1 |
| `huginn/routes/data_dict.py` | 3 |
| `huginn/routes/deep_research.py` | 2 |
| `huginn/routes/diagnostics.py` | 4 |
| `huginn/routes/eval.py` | 4 |
| `huginn/routes/events.py` | 1 |
| `huginn/routes/execution.py` | 3 |
| `huginn/routes/kernel.py` | 5 |
| `huginn/routes/kg.py` | 5 |
| `huginn/routes/live_script.py` | 2 |
| `huginn/routes/parameters.py` | 5 |
| `huginn/routes/planner.py` | 6 |
| `huginn/routes/search.py` | 1 |
| `huginn/routes/system.py` | 1 |
| `huginn/routes/tunnels.py` | 6 |
| `huginn/routes/users.py` | 8 |
| `huginn/routes/visual.py` | 4 |

诚实边界: 前端只扫 `lib/api.ts` 的 `api.*` 包装 (裸 `fetch(...)` 与 EventSource 在别面); 路径里的 `${…}` 只保留静态前缀, 动态拼接的段不可穷尽; `getBlob(path, { method: … })` 的方法覆盖按调用实参里的 `method:` 字面量近似判定; **请求体形状 / 必填 query 参数不核** —— 前端发 multipart 而后端要 JSON body、漏传必填 query 参数这类「路径对、负载错」静态不可辨, 不在本面 (只报 404/405 这类路径+方法级硬违例).

## 发现汇总

- 奖励项零调用者: reconcile_r_phys
- 同轴惩罚候选 (轮次): efficiency_discount, idle_turn_penalty
- 跨模块同名: reconcile_r_phys @ huginn/security/world_state.py
- 工作流 mode 未在 planner 提示暴露: dynamic_workflow
- 模式有 prompt 段却无 set_mode 生产者: code
- 模式有 prompt 段却无 set_mode 生产者: extreme
- 模式有 prompt 段却无 set_mode 生产者: fusion
- 模式被 set_mode 却无 prompt 段: plan
- 模式: 各来源词表互相不一致
- 词汇: 同名跨模块定义值域不一致: KINDS @ huginn/catalog/models.py:20, huginn/research/cspace.py:32, huginn/evolution/semantic_distiller.py:39
- 词汇: 同名跨模块定义值域不一致: Severity @ huginn/execution/physics_auditor.py:24, huginn/metacog/failure_modes.py:27
- 词汇: 同名跨模块定义值域不一致: _ALLOWED_IMPORTS @ huginn/security/script_runner.py:74, huginn/security/code_act_sandbox.py:32
- 词汇: 同名跨模块定义值域不一致: _KINDS @ huginn/share.py:21, huginn/workflows/registry.py:25
- 词汇: 同名跨模块定义值域不一致: _NEGATIVE_WORDS @ huginn/persona_emotion.py:147, huginn/tools/design/gap_analysis_tool.py:30
- 词汇: 同名跨模块定义值域不一致: _READ_ACTIONS @ huginn/tools/github_tool.py:55, huginn/tools/git_tool.py:85
- 词汇: 词表漂移 (簇 7, 并集 12 词): huginn/mcp_client.py::_SENSITIVE_CONFIG_KEYS, huginn/events/audit_log.py::_SECRET_KEYS
- 词汇: 词表漂移 (簇 13, 并集 6 词): huginn/config.py::ThinkingIntensity, huginn/core_types.py::class RiskLevel, huginn/ontology/actions.py::class RiskLevel
- 词汇: 词表漂移 (簇 14, 并集 5 词): huginn/config.py::ContainerRuntimeLiteral, huginn/security/container_executor.py::_VALID_RUNTIMES
- 词汇: 词表漂移 (簇 18, 并集 4 词): huginn/core_types.py::class BudgetDecision, huginn/security/policy_engine.py::_VALID_ACTIONS
- 词汇: 词表漂移 (簇 21, 并集 8 词): huginn/research_budget.py::_EXPENSIVE_TOOLS, huginn/agent/context.py::_EXPENSIVE_TOOL_NAMES, huginn/hooks/research_safety_hook.py::_EXPENSIVE_TOOLS
- 词汇: 词表漂移 (簇 36, 并集 8 词): huginn/lean/conjecture_library.py::_PROOF_ALLOWED_MODULES, huginn/bench/task_synthesizer.py::_JUDGE_ALLOWED_MODULES
- 词汇: 词表漂移 (簇 73, 并集 10 词): huginn/memory/types.py::class MemoryType, huginn/memory/typing.py::class MemoryType
- 词汇: 词表漂移 (簇 135, 并集 4 词): huginn/tools/visualize_gate.py::RERENDERABLE, huginn/tools/visualize_gate.py::_SEVERITY
- 词汇: 映射往返丢信息: AUTOLOOP_TO_PHASE 共像 [ResearchPhase.VALIDATION←["'validate'", "'learn'"]] 且有反向表 PHASE_TO_AUTOLOOP
- 钩子: 事件有生产触发点但零生产注册 (触发无人接): SESSION_START
- 钩子: 事件有生产触发点但零生产注册 (触发无人接): SESSION_END
- 钩子: 事件有生产触发点但零生产注册 (触发无人接): SUBAGENT_STOP
- 钩子: 事件有生产触发点但零生产注册 (触发无人接): PRE_COMPACT
- 钩子: 事件有生产触发点但零生产注册 (触发无人接): POST_COMPACT
- 钩子: 事件有生产触发点但零生产注册 (触发无人接): POST_TOOL_USE_FAILURE
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): campaign.budget_exhausted
- 事件: 发布+订阅了未声明类型 (设计允许非穷尽, 候选登记): campaign.retry
- 事件: 发布+订阅了未声明类型 (设计允许非穷尽, 候选登记): campaign.suspect
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): cognitive.csm.transition
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): embedding.download.done
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): embedding.download.error
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): embedding.download.progress
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): embedding.download.start
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): event_bus.dropped
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): llm.response
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): pet.mood
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): team.batch.start
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): team.member.done
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): team.member.start
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): team.member.tool
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): team.run.done
- 事件: 发布了未声明类型 (设计允许非穷尽, 候选登记): team.run.start
