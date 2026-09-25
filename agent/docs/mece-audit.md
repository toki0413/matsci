# MECE 契约审计 (奖励面 + 授权面 + 工作流面 + 模式面 + 词汇面 + 工具面 + 钩子面 + 事件面)

自动生成: `python -m huginn.cli.contract_audit --out docs/mece-audit.md`.
以 MECE 两原则审计 agent 的**奖励面 / 授权面 / 工作流面 / 模式面 / 词汇面 / 工具面 / 钩子面 / 事件面**: **collectively exhaustive** 抓「宣称维度零调用者 / 面之间的缺口」; **mutually exclusive** 抓「同轴惩罚叠加」「跨模块同名重复实现」「词表互不一致」「同名工具名多类声明」「事件常量撞值」. 纯静态扫描, 只提示候选, 不判死.

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
