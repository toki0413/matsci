# 功能开关契约 (FeatureFlags)

自动生成: `python -m huginn.cli.config_audit --flags --out docs/feature-flags-contract.md`.
统一登记 agent 的可关增强功能. 默认值来自 `FeatureFlags._DEFAULTS`; `legacy_env` 是迁移前的旧裸读变量名 (仍兼容, 见 `_ENV_ALIASES`); `read_points` 是 `is_enabled(...)` 的静态扫描消费点. 优先级: 硬编码默认 < 配置文件 < HUGINN_FEATURE_<NAME> 环境变量 < 运行时 API.

| 开关 | 默认 | 描述 | 旧 env 别名 | 消费点 |
|---|---|---|---|---|
| `bandit_mdp` | True | bandit MDP 决策 (HUGINN_BANDIT_MDP) | HUGINN_BANDIT_MDP | agent/bandit_controller.py:117 |
| `belief_darwin` | True | 信念 Darwin 演化 (HUGINN_BELIEF_DARWIN) | — | — |
| `belief_mode_switch` | True | 信念模式切换 (HUGINN_BELIEF_MODE_SWITCH) | — | — |
| `belief_update` | True | 信念更新管线 (HUGINN_BELIEF_UPDATE) | — | — |
| `clarification` | True | agent 主动向用户提问 | — | tools/clarification_tool.py:206 |
| `context_router` | False | P3 信息路径多样性稀疏化 (context_builder, 默认关) | — | — |
| `crdt_branch_merge` | True | CRDT 分支合并 (HUGINN_CRDT_BRANCH_MERGE) | HUGINN_CRDT_BRANCH_MERGE | utils/conversation_tree.py:37 |
| `crdt_merge` | True | CRDT 合并 (HUGINN_CRDT_MERGE) | — | — |
| `curiosity_hint` | False | 好奇心提示 (HUGINN_CURIOSITY_HINT) | HUGINN_CURIOSITY_HINT | — |
| `external_thinking` | False | 外部草稿纸: 注入 deep_think 指令, 让模型动手前先写分析 (默认关) | — | agent/prompt_builder.py:250 |
| `extreme_dispatch` | False | 极端模式分发 (HUGINN_EXTREME_DISPATCH) | — | — |
| `fts_auto_rebuild` | True | 全文检索自动重建 (HUGINN_FTS_AUTO_REBUILD) | — | — |
| `harness_adoption_gate` | False | 严格 gate 模式: RED 不自动采纳 (实验性, 默认 advisory 只评分不拦) | — | — |
| `harness_joint_optimizer` | False | 联合优化 phase/block/params (实验性, 默认关) | — | — |
| `harness_ood_holdout` | False | H6 OOD 留出验证, 防背题补丁 (实验性, 默认关) | — | — |
| `harness_phase_evolve` | False | 阶段规范演化 (实验性, 默认关) | — | — |
| `harness_prompt_patch` | False | 提示补丁, 跨域提示增强 (实验性, 默认关) | — | — |
| `harness_significance_gate` | False | H5 结果显著性门, 统计检验 (实验性, 默认关) | — | — |
| `harness_workflow_evolution` | False | H2 variant bandit 演化回路 (实验性, 默认关) | — | — |
| `hils_attention` | True | HILS 注意力 (HUGINN_HILS_ATTENTION) | — | — |
| `hypothesis_llm_semantic` | False | 假设维度/方法族/失败类型 LLM 语义判定 (P1#1, 默认关, 优雅降级) | — | — |
| `ising_frontier` | True | Ising 前沿 (HUGINN_ISING_FRONTIER) | — | — |
| `ising_rerank` | True | Ising 重排 (HUGINN_ISING_RERANK) | — | — |
| `json_logs` | True | JSON 结构化日志 (HUGINN_JSON_LOGS) | HUGINN_JSON_LOGS | utils/json_logging.py:122 |
| `loop_detector` | True | 对话循环检测 | — | agent/streaming.py:1523 |
| `memory_typing` | True | 记忆类型标注 (HUGINN_USE_MEMORY_TYPING) | HUGINN_USE_MEMORY_TYPING | memory/typing.py:52 |
| `persistent_terminal` | False | 持久化终端 (HUGINN_PERSISTENT_TERMINAL) | HUGINN_PERSISTENT_TERMINAL | — |
| `personalization` | True | 学习用户通信风格 | — | personalization/user_style.py:152, personalization/user_style.py:349 |
| `privacy_block_on_secrets` | False | 检测到密钥时阻断 (HUGINN_PRIVACY_BLOCK_ON_SECRETS) | HUGINN_PRIVACY_BLOCK_ON_SECRETS | — |
| `privacy_local_only` | False | 隐私级别: local_only (完全本地) | — | privacy_guard.py:77 |
| `privacy_off` | True | 隐私级别: off (不脱敏, 默认. 仅由 set_level 维护互斥, 外部设置无效) | — | — |
| `privacy_redact` | False | 隐私级别: redact (脱敏后发云端) | — | privacy_guard.py:79 |
| `privacy_redact_secrets` | True | 检测到密钥时脱敏 (HUGINN_PRIVACY_REDACT_SECRETS) | HUGINN_PRIVACY_REDACT_SECRETS | — |
| `prompt_cache_control` | True | prompt-cache control 注入 (HUGINN_PROMPT_CACHE_CONTROL) | HUGINN_PROMPT_CACHE_CONTROL | — |
| `provenance` | True | 计算 provenance 快照 | — | provenance/_legacy.py:259, tools/base.py:325 |
| `speculator` | True | 投机执行 (意图预测+工具预热) | — | agents/speculator.py:444 |
| `system_health_auto_fix` | False | 监控异常后自动熔断工具 (默认关) | — | diagnostics/system_health.py:519 |
| `system_health_monitor` | True | 系统资源监控 (CPU/内存/磁盘) | — | diagnostics/system_health.py:521, routes/config.py:784, tools/__init__.py:477 |
| `task_tool_router` | False | task keyword → tool category 动态路由 (默认关) | — | — |
| `telemetry` | True | 遥测采集 (HUGINN_TELEMETRY_ENABLED) | HUGINN_TELEMETRY_ENABLED | agent_config.py:158 |
| `three_cabin` | False | 三舱模型 (HUGINN_USE_THREE_CABIN) | HUGINN_USE_THREE_CABIN | — |
| `tool_call_router` | True | 重型工具 sanity check 路由 | — | agents/tool_call_router.py:123 |
| `use_atomworld` | False | AtomWorld 环境 (HUGINN_USE_ATOMWORLD) | HUGINN_USE_ATOMWORLD | — |
| `use_cognitive_map` | False | 认知地图 (HUGINN_USE_COGNITIVE_MAP) | HUGINN_USE_COGNITIVE_MAP | — |
| `world_model` | False | 世界模型 (HUGINN_WORLD_MODEL) | HUGINN_WORLD_MODEL | — |

共 45 个功能开关。
