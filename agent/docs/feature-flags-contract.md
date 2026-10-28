# 功能开关契约 (FeatureFlags)

自动生成: `python -m huginn.cli.config_audit --flags --out docs/feature-flags-contract.md`.
统一登记 agent 的可关增强功能. 默认值来自 `FeatureFlags._DEFAULTS`; `legacy_env` 是迁移前的旧裸读变量名 (仍兼容, 见 `_ENV_ALIASES`); `read_points` 是 `is_enabled(...)` 的静态扫描消费点. 优先级: 硬编码默认 < 配置文件 < HUGINN_FEATURE_<NAME> 环境变量 < 运行时 API.

| 开关 | 默认 | 描述 | 旧 env 别名 | 消费点 |
|---|---|---|---|---|
| `anti_hacking_reward` | False | Anti-Hacking ① strict-scope 越界清零, 折进 _learn 的 r_phys (默认关) | — | autoloop/engine_reflect.py:3643 |
| `asd_ste100` | True | ASD-STE100 受控语言纪律: 向系统提示注入 agent 面向文本规则 (默认开; HUGINN_STE_MODE 选档) | — | plugins/asd_ste100/main.py:149 |
| `bandit_mdp` | True | bandit MDP 决策 (HUGINN_BANDIT_MDP) | HUGINN_BANDIT_MDP | agent/bandit_controller.py:122 |
| `belief_darwin` | True | 信念 Darwin 演化 (HUGINN_BELIEF_DARWIN) | HUGINN_BELIEF_DARWIN | autoloop/cognitive_loop.py:1432, autoloop/cognitive_loop.py:1529 |
| `belief_mode_switch` | True | 信念模式切换 (HUGINN_BELIEF_MODE_SWITCH) | HUGINN_BELIEF_MODE_SWITCH | task_reflector.py:390, task_reflector.py:40 |
| `belief_update` | True | 信念更新管线 (HUGINN_BELIEF_UPDATE) | HUGINN_BELIEF_UPDATE | tools/subagent_tool.py:42 |
| `clarification` | True | agent 主动向用户提问 | — | tools/clarification_tool.py:149 |
| `comms_contract` | True | 通信契约审计: 事件总线/工作流/agent 间消息的确定性契约检查 (默认开, 只报告不拦截) | — | events/event_bus.py:214 |
| `compute_policy` | False | M2 计算路由目标维度策略 + 预算 (默认关) | — | execution/orchestrator.py:225 |
| `context_router` | False | P3 信息路径多样性稀疏化 (context_builder, 默认关) | HUGINN_CONTEXT_ROUTER | context_builder.py:1284 |
| `crdt_branch_merge` | True | CRDT 分支合并 (HUGINN_CRDT_BRANCH_MERGE) | HUGINN_CRDT_BRANCH_MERGE | utils/conversation_tree.py:37, utils/conversation_tree.py:381 |
| `crdt_merge` | True | CRDT 合并 (HUGINN_CRDT_MERGE) | HUGINN_CRDT_MERGE | tools/subagent_tool.py:33 |
| `curiosity_hint` | True | 好奇心提示 (HUGINN_CURIOSITY_HINT) | HUGINN_CURIOSITY_HINT | autoloop/engine_observe.py:314, cli/rcb_step2.py:1405 |
| `external_thinking` | False | 外部草稿纸: 注入 deep_think 指令, 让模型动手前先写分析 (默认关) | — | agent/prompt_builder.py:379 |
| `extreme_dispatch` | False | 极端模式分发 (HUGINN_EXTREME_DISPATCH) | HUGINN_EXTREME_DISPATCH | agent/core.py:819, autoloop/engine_perceive.py:370, autoloop/engine_reflect.py:2807 +1 处 |
| `frontier_claim` | False | 假设图单写者领用前沿: execute 优先领已有可执行方向而非凭空生成 (ARTEX 借鉴, 默认关) | HUGINN_FRONTIER_CLAIM | autoloop/cognitive_loop.py:3684 |
| `frontier_empty_stop` | False | 可执行前沿空即停: 无'未执行且前置证据满足'的方向 → 主终止, 预算降为安全上限 (ARTEX 借鉴, 默认关) | HUGINN_FRONTIER_EMPTY_STOP | autoloop/cognitive_loop.py:3668 |
| `frontier_single_writer_strict` | False | 假设图单写者纪律: 非 hypothesize/branch 阶段追加方向时拒绝 (默认关, 只观测计数) | HUGINN_FRONTIER_SINGLE_WRITER_STRICT | autoloop/hypothesis_loop.py:74 |
| `fts_auto_rebuild` | True | 全文检索自动重建 (HUGINN_FTS_AUTO_REBUILD) | HUGINN_FTS_AUTO_REBUILD | — |
| `harness_adoption_gate` | False | 严格 gate 模式: RED 不自动采纳 (实验性, 默认 advisory 只评分不拦) | — | — |
| `harness_joint_optimizer` | False | 联合优化 phase/block/params (实验性, 默认关) | — | — |
| `harness_meta_improver` | False |  | — | — |
| `harness_ood_holdout` | False | H6 OOD 留出验证, 防背题补丁 (实验性, 默认关) | — | — |
| `harness_phase_evolve` | False | 阶段规范演化 (实验性, 默认关) | — | — |
| `harness_prompt_patch` | False | 提示补丁, 跨域提示增强 (实验性, 默认关) | — | — |
| `harness_significance_gate` | False | H5 结果显著性门, 统计检验 (实验性, 默认关) | — | — |
| `harness_workflow_evolution` | False | H2 variant bandit 演化回路 (实验性, 默认关) | — | — |
| `hils_attention` | True | HILS 注意力 (HUGINN_HILS_ATTENTION) | HUGINN_HILS_ATTENTION | — |
| `hypothesis_llm_semantic` | False | 假设维度/方法族/失败类型 LLM 语义判定 (P1#1, 默认关, 优雅降级) | — | — |
| `intent_scope_reward` | False | Anti-Hacking ② 意图口径越界清零: 改动偏离 plan 声明的目标集即清零 (默认关) | — | autoloop/engine_reflect.py:3644 |
| `ising_frontier` | True | Ising 前沿 (HUGINN_ISING_FRONTIER) | HUGINN_ISING_FRONTIER | autoloop/hypothesis_loop.py:2490, autoloop/hypothesis_loop.py:58 |
| `ising_rerank` | True | Ising 重排 (HUGINN_ISING_RERANK) | HUGINN_ISING_RERANK | — |
| `jev_enabled` | False | JEV (System One) 外部判断总开关 (实验性, 默认关; 受隐私外发闸约束) | — | — |
| `jev_guardrail` | False | 工具调用放行初筛 deny/ask/allow (实验性, 默认关, advisory) | — | — |
| `jev_tool_router` | False | 未知域工具子集用 JEV 并行 Noul 宽松补充 (实验性, 默认关, advisory) | — | — |
| `json_logs` | True | JSON 结构化日志 (HUGINN_JSON_LOGS) | HUGINN_JSON_LOGS | utils/json_logging.py:122 |
| `loop_detector` | True | 对话循环检测 | — | agent/streaming.py:2479 |
| `memory_typing` | True | 记忆类型标注 (HUGINN_USE_MEMORY_TYPING) | HUGINN_USE_MEMORY_TYPING | memory/typing.py:52 |
| `persistent_terminal` | False | 持久化终端 (HUGINN_PERSISTENT_TERMINAL) | HUGINN_PERSISTENT_TERMINAL | tools/persistent_terminal.py:319, tools/persistent_terminal.py:618 |
| `personalization` | True | 学习用户通信风格 | — | personalization/user_style.py:228, personalization/user_style.py:426 |
| `pi_mode` | False |  | — | — |
| `privacy_block_on_secrets` | False | 检测到密钥时阻断 (HUGINN_PRIVACY_BLOCK_ON_SECRETS) | HUGINN_PRIVACY_BLOCK_ON_SECRETS | agent_config.py:182 |
| `privacy_local_only` | False | 隐私级别: local_only (完全本地) | — | privacy_guard.py:81, runtime/jev/_enabled.py:45 |
| `privacy_off` | True | 隐私级别: off (不脱敏, 默认. 仅由 set_level 维护互斥, 外部设置无效) | — | — |
| `privacy_redact` | False | 隐私级别: redact (脱敏后发云端) | — | privacy_guard.py:83, runtime/jev/_enabled.py:47 |
| `privacy_redact_secrets` | True | 检测到密钥时脱敏 (HUGINN_PRIVACY_REDACT_SECRETS) | HUGINN_PRIVACY_REDACT_SECRETS | agent_config.py:181, tools/adapter.py:1253 |
| `prompt_cache_control` | True | prompt-cache control 注入 (HUGINN_PROMPT_CACHE_CONTROL) | HUGINN_PROMPT_CACHE_CONTROL | agent_config.py:108 |
| `provenance` | True | 计算 provenance 快照 | — | provenance/_legacy.py:240, tools/base.py:376 |
| `sandbox_net_isolation` | True | P1 沙箱内核级网络隔离 (Landlock ABI>=4, 默认开; 不支持则降级) | — | security/sandbox.py:60 |
| `speculator` | True | 投机执行 (意图预测+工具预热) | — | agents/speculator.py:463 |
| `system_health_auto_fix` | False | 监控异常后自动熔断工具 (默认关) | — | diagnostics/system_health.py:552 |
| `system_health_monitor` | True | 系统资源监控 (CPU/内存/磁盘) | — | diagnostics/system_health.py:554, routes/config.py:995, tools/__init__.py:533 |
| `task_tool_router` | True | task keyword → tool category 动态路由 (默认开, 无命中给 core 子集) | HUGINN_TASK_TOOL_ROUTER | agent/core.py:919, agent/streaming.py:1860 |
| `telemetry` | True | 遥测采集 (HUGINN_TELEMETRY_ENABLED) | HUGINN_TELEMETRY_ENABLED | agent_config.py:197 |
| `three_cabin` | False | 三舱模型 (HUGINN_USE_THREE_CABIN) | HUGINN_USE_THREE_CABIN | autoloop/cognitive_loop.py:3335 |
| `tool_call_router` | True | 重型工具 sanity check 路由 | — | agents/tool_call_router.py:124 |
| `trace_inspect` | False | 模型侧过程级 trace 检索 mode: 按关键词/tool/intent recall 过往执行 (VISTA inspect 借鉴, 默认关) | HUGINN_TRACE_INSPECT | autoloop/plan_check.py:560, autoloop/plan_check.py:586 |
| `use_atomworld` | False | AtomWorld 环境 (HUGINN_USE_ATOMWORLD) | HUGINN_USE_ATOMWORLD | agent/code_act_loop.py:360, agent/code_act_loop.py:402 |
| `use_cognitive_map` | False | 认知地图 (HUGINN_USE_COGNITIVE_MAP) | HUGINN_USE_COGNITIVE_MAP | agent/code_act_loop.py:372, agent/code_act_loop.py:424, runtime/engine_state.py:251 |
| `visual_frame_memory` | False | 无损观测记忆: 每次 execute 的视觉帧原样落盘 + frame_inspect 区域回看/read_pixels (VISTA 借鉴, 默认关) | HUGINN_VISUAL_FRAME_MEMORY | autoloop/plan_check.py:562, autoloop/plan_check.py:593 |
| `world_model` | True | 世界模型 (HUGINN_WORLD_MODEL) | HUGINN_WORLD_MODEL | autoloop/engine_observe.py:357 |

共 61 个功能开关。
