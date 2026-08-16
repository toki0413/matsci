# 环境变量契约 (HUGINN_*)

自动生成: `python -m huginn.cli.config_audit --out docs/env-contract.md`.
本表登记 huginn/ 代码库中所有 `HUGINN_*` 环境变量的引用, 是配置面的审计基线. **推断状态仅提示, 不代替人工判定**: `code-set` 表示代码某处设置了它; `external` 表示未在代码内设置 (可能由用户 shell/.env 注入,需人工确认是否为有效配置).

| 变量 | 默认值 | 读取点 | 设置点 | 推断状态 |
|---|---|---|---|---|
| `HUGINN_ACTION_HIST_MAX` | 1000 | autoloop/cognitive_loop.py:66 | — | external |
| `HUGINN_ADMIN_API_KEY` | '' | security/auth.py:115 | — | external |
| `HUGINN_AGENTS` | '' | config.py:876 | routes/config.py:316 | code-set |
| `HUGINN_AINVOKE_TIMEOUT` | str(_thinking_scale_timeout( | agent/streaming.py:1803 | — | external |
| `HUGINN_ALERT_WEBHOOK_URL` | '' | diagnostics/system_health.py:391 | — | external |
| `HUGINN_ALIGNMENT_SURPRISE_TRIGGER` | 0 | autoloop/engine_reflect.py:1827 | — | external |
| `HUGINN_ALLOW_LOCAL_BASH` | '' | config.py:769, security/execution.py:22 | routes/agents.py:84, tools/__init__.py:189, cli/rcb_runner.py:61 | code-set |
| `HUGINN_ALLOW_UNRESTRICTED_READ` | '' | routes/fs.py:47, tools/file_read_tool.py:64, tools/sci/xrd_sim_tool.py:141 | — | external |
| `HUGINN_API_KEY` | '' | bench/llm_judge.py:169, cli/availability.py:49, config.py:650, +3 处 | bench/llm_judge.py:299, bench/runner.py:818, bench/llm_judge.py:288, bench/runner.py:803 | code-set |
| `HUGINN_APPROVAL_MODE` | '' | agent/code_act_loop.py:704, agent/core.py:330 | — | external |
| `HUGINN_ATOMWORLD_DATA_DIR` | ./atomworld_data | bench/atomworld_bench.py:23 | — | external |
| `HUGINN_AUDIT_SIGNING_KEY` | '' | security/audit.py:626 | — | external |
| `HUGINN_AUTOLOOP_HUMAN_PAUSE` | 0 | autoloop/cognitive_loop.py:2670, autoloop/engine_reflect.py:2890 | — | external |
| `HUGINN_AUTOLOOP_STREAMING` | 1 | autoloop/engine.py:67 | — | external |
| `HUGINN_AUTO_APPROVE` | '' | config.py:750, server_core.py:612, tools/adapter.py:562, +2 处 | — | external |
| `HUGINN_AUTO_WAKE` | 1 | autoloop/engine.py:605 | — | external |
| `HUGINN_BANDIT_Q_PATH` | '' | agent/bandit_controller.py:136 | cli/rcb_runner.py:70 | code-set |
| `HUGINN_BASE_URL` | '' | config.py:680, config.py:703 | — | external |
| `HUGINN_BELIEF_DARWIN` | 1 | autoloop/cognitive_loop.py:725, autoloop/cognitive_loop.py:834 | — | external |
| `HUGINN_BELIEF_ENTROPY_FACT_CHECK` | 0 | utils/belief_entropy.py:455 | — | external |
| `HUGINN_BELIEF_ENTROPY_HIGH` | 0.7 | utils/belief_entropy.py:461 | — | external |
| `HUGINN_BELIEF_ENTROPY_LOW` | 0.3 | utils/belief_entropy.py:458 | — | external |
| `HUGINN_BELIEF_MODE_SWITCH` | 1 | task_reflector.py:39, task_reflector.py:296 | task_reflector.py:297, task_reflector.py:329, task_reflector.py:339, task_reflector.py:337 | code-set |
| `HUGINN_BELIEF_UPDATE` | 1 | tools/subagent_tool.py:42 | tools/subagent_tool.py:880, tools/subagent_tool.py:889 | code-set |
| `HUGINN_BENCHMARK_MODE_PROMPT` | '' | agent/core.py:872 | cli/rcb_runner.py:76 | code-set |
| `HUGINN_BLIND_RECONSTRUCTION` | 0 | autoloop/engine_reflect.py:384 | — | external |
| `HUGINN_BOURBAKI_PATH` | '' | tools/__init__.py:190 | — | external |
| `HUGINN_BRANCH_INCUBATOR_DEPTH` | 1 | autoloop/hypothesis_loop.py:1936 | — | external |
| `HUGINN_BUDGET_MODE` | cumulative | scheduling/scheduler.py:115 | — | external |
| `HUGINN_CACHE_DIR` | str(get_runtime_home( | autoloop/goal_store.py:124, autoloop/plan_store.py:186, cli/rcb_runner.py:64, +19 处 | agents/task_dag.py:312, autoloop/bandit.py:424, autoloop/variant_gen.py:299, +9 处, autoloop/variant_gen.py:310 | code-set |
| `HUGINN_CHECKPOINTER_PATH` | '' | agent_config.py:111, bench/orchestrator.py:172, checkpointer.py:34, +4 处 | bench/orchestrator.py:451, bench/orchestrator.py:475, bench/orchestrator.py:490, bench/orchestrator.py:501 | code-set |
| `HUGINN_CI` | '' | tools/sci/gp_tool.py:675 | — | external |
| `HUGINN_CODEACT_MEM_CAP` | 2147483648 | agent/code_act_loop.py:793 | — | external |
| `HUGINN_CODER_DONE_MARKER` | [DONE] | config.py:1281 | — | external |
| `HUGINN_CODER_MAX_ITER` | 50 | config.py:1280 | — | external |
| `HUGINN_COEVOLUTION` | 0 | cli/rcb_runner.py:1239 | — | external |
| `HUGINN_COGNITIVE_LLM_DECIDER` | 1 | autoloop/engine.py:388 | cli/rcb_runner.py:101 | code-set |
| `HUGINN_COMPACT_KIND` | remote | agent/reflection.py:60 | — | external |
| `HUGINN_COMPACT_STRATEGY` | trim,summarize | agent/streaming.py:724 | — | external |
| `HUGINN_COMPLETION_CHECK_INTERVAL` | 10 | cli/rcb_step2.py:422 | — | external |
| `HUGINN_CONFIG_FILE` | huginn.toml | config.py:1332, routes/config.py:55, routes/config.py:745, +2 处 | — | external |
| `HUGINN_CONSOLIDATE_MODEL` | '' | memory/longterm.py:1726 | — | external |
| `HUGINN_CONSOLIDATE_PROVIDER` | deepseek | memory/longterm.py:1725 | — | external |
| `HUGINN_CONTAINER_IMAGE` | '' | config.py:723, security/execution.py:28 | — | external |
| `HUGINN_CONTAINER_RUNTIME` | none | config.py:722, security/execution.py:27 | — | external |
| `HUGINN_CONTEXT_BUDGET_TOKENS` | 0 | agent_config.py:177, config.py:797 | — | external |
| `HUGINN_CONTEXT_MAX_MESSAGES` | 60 | agent/core.py:370 | — | external |
| `HUGINN_CONTEXT_ROUTER` | 0 | context_builder.py:1177 | cli/rcb_runner.py:363 | code-set |
| `HUGINN_CORE_API_KEY` | '' | tools/literature/search_sources.py:594 | — | external |
| `HUGINN_CORE_SUPPORT_PROTOCOL` | 1 | tools/bash_tool.py:180, tools/code_tool.py:126 | — | external |
| `HUGINN_CORS_ORIGINS` | '' | lifespan.py:837 | — | external |
| `HUGINN_COST_BUDGET` | 50.0 | autoloop/budget.py:107 | — | external |
| `HUGINN_CRDT_BRANCH_MERGE` | '' | utils/conversation_tree.py:345 | utils/conversation_tree.py:346, utils/conversation_tree.py:385, utils/conversation_tree.py:396, +1 处, utils/conversation_tree.py:418 | code-set |
| `HUGINN_CRDT_MERGE` | 1 | tools/subagent_tool.py:33 | tools/subagent_tool.py:779, tools/subagent_tool.py:781 | code-set |
| `HUGINN_CREDENTIAL_DB` | '' | security/credential_store.py:516 | — | external |
| `HUGINN_CREDENTIAL_KEY_FILE` | '' | security/credential_store.py:98 | — | external |
| `HUGINN_CSM_SUBSET_MODE` | '' | agent/reflection.py:368 | cli/causal_runner.py:162, cli/rcb_runner.py:68 | code-set |
| `HUGINN_CURIOSITY_HINT` | 0 | autoloop/engine_observe.py:323, cli/rcb_step2.py:869 | — | external |
| `HUGINN_DARWIN_LLM_EVAL` | 0 | metacog/llm_likelihood.py:46, metacog/llm_likelihood.py:271, metacog/step_evaluator.py:557, +1 处 | metacog/llm_likelihood.py:274, metacog/llm_likelihood.py:277, metacog/step_evaluator.py:1045, +3 处, metacog/step_evaluator.py:1107 | code-set |
| `HUGINN_DARWIN_LLM_INTERVAL` | 5 | metacog/step_evaluator.py:573, metacog/step_evaluator.py:1042 | metacog/step_evaluator.py:1046, metacog/step_evaluator.py:1077, metacog/step_evaluator.py:1109, metacog/step_evaluator.py:1111 | code-set |
| `HUGINN_DARWIN_STAGNATION_LIMIT` | 5 | autoloop/cognitive_loop.py:775 | cli/rcb_runner.py:346 | code-set |
| `HUGINN_DEEPSEEK_OCR_PATH` | '' | tools/vision_describe_tool.py:67 | — | external |
| `HUGINN_DEV_MODE` | '' | middleware/error_normalize.py:145, middleware/ws_governance.py:123, routes/agents.py:83, +1 处 | — | external |
| `HUGINN_DISABLE_WEB_SEARCH` | '' | tools/agentic_search_tool.py:287, tools/literature/_http.py:40, tools/web_search_tool.py:56 | tools/web_search_tool.py:650, tools/web_search_tool.py:655 | code-set |
| `HUGINN_DOCKER_SANDBOX` | '' | security/sandbox.py:594 | — | external |
| `HUGINN_DOC_ENGINE` | auto | perception/pdf_parser.py:506 | — | external |
| `HUGINN_EMBED_MODEL` | all-MiniLM-L6-v2 | knowledge/store.py:33 | — | external |
| `HUGINN_EM_RECALL_TOP_K` | 5 | config.py:828 | — | external |
| `HUGINN_ENABLE_EXPLORATION` | true | config.py:751 | — | external |
| `HUGINN_ENABLE_SCIHUB` | '' | tools/literature/pdf_fetch.py:238 | — | external |
| `HUGINN_ENCRYPTION_ENABLED` | '' | config.py:771 | — | external |
| `HUGINN_ENCRYPTION_KEY` | '' | security/credential_store.py:614 | — | external |
| `HUGINN_ENCRYPTION_KEY_FILE` | '' | config.py:780 | — | external |
| `HUGINN_ENCRYPTION_PASSWORD` | '' | config.py:779, config.py:1142, config.py:1178, +1 处 | — | external |
| `HUGINN_ENCRYPT_CONFIG` | '' | config.py:773 | — | external |
| `HUGINN_ENCRYPT_RAG_DOCS` | true | config.py:781 | — | external |
| `HUGINN_ENCRYPT_RAG_META` | true | config.py:785 | — | external |
| `HUGINN_ENFORCE_WRITE_CAPABILITY` | 1 | security/auth.py:328 | — | external |
| `HUGINN_ENV` | '' | security/auth.py:47, server.py:84 | — | external |
| `HUGINN_EPISODIC_REPLAY` | 0 | autoloop/engine_observe.py:417 | — | external |
| `HUGINN_EPISODIC_SHARD_INTERVAL` | _DEFAULT_INTERVAL | memory/episodic_shard.py:81 | memory/episodic_shard.py:259, memory/episodic_shard.py:313, memory/episodic_shard.py:327 | code-set |
| `HUGINN_EXECUTION_BACKEND` | local | config.py:721 | — | external |
| `HUGINN_EXTREME_DISPATCH` | 0 | agent/core.py:489, agent/core.py:553, autoloop/engine_perceive.py:307, +3 处 | cli/rcb_runner.py:324, agent/core.py:492, agent/core.py:502, memory/session.py:387, agent/core.py:500, memory/session.py:377, memory/session.py:467 | code-set |
| `HUGINN_EZPROXY_DOMAINS` | '' | tools/literature/crawl_web.py:204 | — | external |
| `HUGINN_EZPROXY_PREFIX` | '' | tools/literature/crawl_web.py:197 | — | external |
| `HUGINN_FAILURE_INVERSION` | 0 | autoloop/engine_reflect.py:2132 | — | external |
| `HUGINN_FEATURE_LOOP_DETECTOR` | '' | — | cli/rcb_runner.py:339, cli/rcb_runner.py:108 | code-set |
| `HUGINN_FILE_READ_MAX_OUTPUT_TOKENS` | str(DEFAULT_MAX_OUTPUT_TOKENS | tools/file_read_tool.py:93 | — | external |
| `HUGINN_FILE_READ_MAX_SIZE_BYTES` | str(DEFAULT_MAX_SIZE_BYTES | tools/file_read_tool.py:88 | — | external |
| `HUGINN_FP_DB` | _DEFAULT_DB | rag/adaptive_parser.py:35 | — | external |
| `HUGINN_FTS_AUTO_REBUILD` | 1 | memory/longterm.py:361 | — | external |
| `HUGINN_GOVERNANCE_DEFAULT_DECISION` | deny | governance.py:41 | — | external |
| `HUGINN_HARD_CHECKPOINT_PHASES` | '' | autoloop/phase_gate.py:54 | — | external |
| `HUGINN_HEALTH_MONITOR` | 0 | tools/adapter.py:148 | cli/rcb_runner.py:103 | code-set |
| `HUGINN_HIDE_DOCS` | '' | security/auth.py:48, server.py:85 | — | external |
| `HUGINN_HILS_ATTENTION` | 1 | memory/longterm.py:729 | memory/longterm.py:2232, memory/longterm.py:2234 | code-set |
| `HUGINN_HINT_COORDINATOR` | 1 | cli/rcb_step2.py:195 | — | external |
| `HUGINN_HPC_HOST` | '' | cli/availability.py:53 | — | external |
| `HUGINN_HTTP_QPS` | 5 | tools/literature/_http.py:85 | — | external |
| `HUGINN_HUMAN_CHECKPOINT_PHASES` | '' | autoloop/phase_gate.py:31 | — | external |
| `HUGINN_INHERIT_STABLE_PRINCIPLES` | 1 | memory/longterm.py:1903 | — | external |
| `HUGINN_ISING_FRONTIER` | 1 | autoloop/hypothesis_loop.py:49, autoloop/hypothesis_loop.py:1732 | autoloop/hypothesis_loop.py:1733, autoloop/hypothesis_loop.py:1779, autoloop/hypothesis_loop.py:1788, autoloop/hypothesis_loop.py:1786 | code-set |
| `HUGINN_ISING_RERANK` | 1 | memory/longterm.py:620 | memory/longterm.py:2177, memory/longterm.py:2179 | code-set |
| `HUGINN_ITER_HIST_MAX` | 50 | autoloop/cognitive_loop.py:68 | — | external |
| `HUGINN_JWT_SECRET` | '' | security/auth.py:80 | — | external |
| `HUGINN_KB_CHUNKS_PATH` | '' | evolution/knowledge_distiller.py:497 | — | external |
| `HUGINN_KEEP_ROOT_N` | 2 | agent/streaming.py:854, agent/streaming.py:1509, agent/streaming.py:1552 | cli/rcb_runner.py:89 | code-set |
| `HUGINN_KG_DEPTH` | 1 | config.py:766 | — | external |
| `HUGINN_KG_ENABLED` | '' | config.py:765 | — | external |
| `HUGINN_KG_TOP_K` | 10 | config.py:767 | — | external |
| `HUGINN_LAZY_CLI` | '' | cli/main.py:137 | — | external |
| `HUGINN_LITERATURE_TIMEOUT` | 20 | tools/literature/_http.py:29 | — | external |
| `HUGINN_LLM_LIKELIHOOD` | 0 | metacog/llm_likelihood.py:42, metacog/llm_likelihood.py:270 | metacog/llm_likelihood.py:273, metacog/llm_likelihood.py:278, metacog/llm_likelihood.py:281 | code-set |
| `HUGINN_LLM_LIKELIHOOD_INTERVAL` | 5 | metacog/llm_likelihood.py:52 | metacog/llm_likelihood.py:285, metacog/llm_likelihood.py:287 | code-set |
| `HUGINN_LLM_REQUEST_TIMEOUT` | 120 | models/registry.py:158 | — | external |
| `HUGINN_LOCAL_ONLY` | 0 | config.py:768, models/registry.py:1173 | — | external |
| `HUGINN_LOG_LEVEL` | INFO | utils/json_logging.py:126 | — | external |
| `HUGINN_LOOP_LIGHT_ON_TRACK` | 1 | autoloop/cognitive_loop.py:2390 | — | external |
| `HUGINN_LOOP_ROLLBACK` | 1 | autoloop/cognitive_loop.py:1776 | — | external |
| `HUGINN_MAINTENANCE` | '' | middleware/maintenance.py:40 | — | external |
| `HUGINN_MAX_BODY_SIZE_MB` | _DEFAULT_MAX_BODY_MB | middleware/limits.py:87 | — | external |
| `HUGINN_MAX_BRANCHES` | 5 | config.py:755 | — | external |
| `HUGINN_MAX_CONCURRENT_SUBAGENTS` | 3 | config.py:708 | — | external |
| `HUGINN_MAX_CONSECUTIVE_FAILURES` | 20 | autoloop/engine.py:342 | cli/rcb_runner.py:343 | code-set |
| `HUGINN_MAX_CONTEXT_TOKENS` | '' | context_manager.py:115 | — | external |
| `HUGINN_MAX_PIVOTS` | 10 | autoloop/engine.py:375 | cli/rcb_runner.py:345 | code-set |
| `HUGINN_MAX_REFINES` | 20 | autoloop/engine.py:370 | cli/rcb_runner.py:344 | code-set |
| `HUGINN_MAX_TOKENS` | '' | config.py:852 | — | external |
| `HUGINN_MAX_TOOL_CALLS` | 15 | agent_config.py:87 | — | external |
| `HUGINN_MAX_TOOL_CALLS_PER_TOOL` | 5 | agent_config.py:89 | — | external |
| `HUGINN_MAX_TOOL_OUTPUT_TOKENS` | 25000 | agent_config.py:85, config.py:794, config.py:1264, +1 处 | — | external |
| `HUGINN_MCMC_ALIGNMENT` | 0 | cli/rcb_runner.py:1347, cli/rcb_step2.py:416 | — | external |
| `HUGINN_MCMC_ALIGNMENT_TEMPERATURE` | 1.0 | cli/rcb_runner.py:1352, cli/rcb_step2.py:418 | — | external |
| `HUGINN_MCMC_CHAINS` | 4 | cli/rcb_runner.py:1317 | — | external |
| `HUGINN_MCMC_CHECKPOINT_INTERVAL` | 10000 | cli/rcb_runner.py:1322, cli/rcb_step2.py:408 | — | external |
| `HUGINN_MCMC_CKPT_DIR` | '' | metacog/hypothesis_manifold.py:1306 | — | external |
| `HUGINN_MCMC_GLOBAL_PROPOSAL` | 0.3 | cli/rcb_mcmc.py:278 | — | external |
| `HUGINN_MCMC_HAPTIC` | 1 | cli/rcb_runner.py:1337, cli/rcb_step2.py:412 | — | external |
| `HUGINN_MCMC_HAPTIC_TEMPERATURE` | 1.0 | cli/rcb_runner.py:1342, cli/rcb_step2.py:415 | — | external |
| `HUGINN_MCMC_INTERVAL` | 5 | autoloop/engine_observe.py:718, cli/rcb_step2.py:407 | — | external |
| `HUGINN_MCMC_MODE` | '' | cli/rcb_runner.py:1307 | — | external |
| `HUGINN_MCMC_NO_ANNEAL` | 0 | cli/rcb_mcmc.py:276 | — | external |
| `HUGINN_MCMC_PARALLEL` | 1 | metacog/hypothesis_manifold.py:764 | — | external |
| `HUGINN_MCMC_SE3` | 0 | cli/rcb_runner.py:1327 | — | external |
| `HUGINN_MCMC_SE3_ANGLE_SIGMA` | 30.0 | cli/rcb_runner.py:1332 | — | external |
| `HUGINN_MCMC_SEED` | 42 | autoloop/engine.py:613, cli/rcb_mcmc.py:230, cli/rcb_step2.py:396, +1 处 | — | external |
| `HUGINN_MCMC_STEPS` | 7000000 | cli/rcb_runner.py:1312 | — | external |
| `HUGINN_MCMC_T_HIGH` | 10 | cli/rcb_mcmc.py:277 | — | external |
| `HUGINN_MEMORY_CLUSTER` | 0 | memory/longterm.py:1582 | — | external |
| `HUGINN_MEMORY_DECAY_ENABLED` | '' | agent_config.py:113, config.py:806 | — | external |
| `HUGINN_MEMORY_DECAY_INTERVAL_TURNS` | 0 | agent_config.py:116, config.py:811 | — | external |
| `HUGINN_MEMORY_DECAY_PRUNE_THRESHOLD` | 0.15 | agent_config.py:119, config.py:814 | — | external |
| `HUGINN_MODEL` | '' | config.py:678, config.py:701, config.py:1260 | — | external |
| `HUGINN_MODELS` | '' | config.py:863 | routes/config.py:314 | code-set |
| `HUGINN_MODEL_TIER` | full | plugins/model_tier.py:88 | — | external |
| `HUGINN_MP_API_KEY` | '' | tools/literature/search_sources.py:1428 | — | external |
| `HUGINN_MULTI_AGENT` | 0 | memory/longterm.py:1913 | — | external |
| `HUGINN_NORMALIZE_ERRORS` | '' | middleware/error_normalize.py:139 | — | external |
| `HUGINN_NO_RUST_SANDBOX` | '' | — | cli/rcb_runner.py:98 | code-set |
| `HUGINN_OAUTH_TOKEN` | '' | cli/availability.py:52 | — | external |
| `HUGINN_OCR_ENGINE` | auto | knowledge/ocr_loader.py:146, knowledge/ocr_loader.py:190, knowledge/ocr_loader.py:238 | — | external |
| `HUGINN_PERSISTENT_GOAL_MODE` | 0 | autoloop/cognitive_loop.py:801, autoloop/engine_reflect.py:906, cli/rcb_runner.py:373 | cli/rcb_runner.py:350 | code-set |
| `HUGINN_PERSISTENT_TERMINAL` | 0 | tools/persistent_terminal.py:280, tools/persistent_terminal.py:536 | tools/persistent_terminal.py:538, tools/persistent_terminal.py:547, tools/persistent_terminal.py:559, tools/persistent_terminal.py:557 | code-set |
| `HUGINN_PERSISTENT_TERMINAL_TIMEOUT` | '' | tools/persistent_terminal.py:30 | — | external |
| `HUGINN_PERSONA` | default | config.py:694, config.py:756 | cli/commands/persona.py:131, routes/agents.py:462 | code-set |
| `HUGINN_PERSONA_AUTO_ROUTE` | true | config.py:757 | — | external |
| `HUGINN_PERSONA_AUTO_ROUTE_THRESHOLD` | 0.3 | config.py:762 | — | external |
| `HUGINN_PER_HYP_BUDGET` | 0 | autoloop/cognitive_loop.py:915, autoloop/cognitive_loop.py:2160, autoloop/engine_reflect.py:436 | — | external |
| `HUGINN_PET_NAME` | 渡鸦 | config.py:821 | — | external |
| `HUGINN_PET_PERSONALITY` | cheerful | config.py:822 | — | external |
| `HUGINN_PLAN_AUTO_CONFIRM` | 0 | config.py:710 | — | external |
| `HUGINN_PMK_INJECT` | 0 | autoloop/engine_observe.py:473 | — | external |
| `HUGINN_PM_C_MIN` | 0.2 | config.py:829 | — | external |
| `HUGINN_PRIVACY_BLOCK_ON_SECRETS` | 0 | agent_config.py:142, config.py:789 | — | external |
| `HUGINN_PRIVACY_REDACT_SECRETS` | 1 | agent_config.py:139, config.py:775, tools/adapter.py:750 | — | external |
| `HUGINN_PRM_VERIFIER` | 0 | agents/factory.py:278 | — | external |
| `HUGINN_PROFILE_MODEL` | '' | tools/personalization_tool.py:158 | — | external |
| `HUGINN_PROFILE_PROVIDER` | deepseek | tools/personalization_tool.py:157 | — | external |
| `HUGINN_PROMPT_CACHE_CONTROL` | true | agent_config.py:64, config.py:799 | — | external |
| `HUGINN_PROVENANCE_ENABLED` | 1 | provenance/registry.py:370 | — | external |
| `HUGINN_PROVIDER` | default | config.py:647, config.py:1261 | — | external |
| `HUGINN_PRT_LEVEL1` | 0 | agents/factory.py:271 | — | external |
| `HUGINN_RAG_ENABLED` | '' | config.py:764 | — | external |
| `HUGINN_RATE_LIMIT_ENABLED` | 1 | security/rate_limiter.py:491 | cli/rcb_runner.py:58 | code-set |
| `HUGINN_RATE_LIMIT_PER_MINUTE` | 120 | server.py:110 | — | external |
| `HUGINN_RATE_LIMIT_TOKENS_PER_SECOND` | 5000 | security/rate_limiter.py:483 | — | external |
| `HUGINN_RATE_LIMIT_TOKENS_PER_TURN` | 100000 | security/rate_limiter.py:480 | cli/rcb_runner.py:59 | code-set |
| `HUGINN_RATE_LIMIT_TOTAL_COST_USD` | 10.0 | security/rate_limiter.py:486 | — | external |
| `HUGINN_RATE_LIMIT_WARNING_THRESHOLD` | 0.8 | security/rate_limiter.py:489 | — | external |
| `HUGINN_RCB_BLOCKED_TOOLS` | # 默认空集 — extreme 模式信任 agent, 全开放.
                " | cli/rcb_runner.py:651 | — | external |
| `HUGINN_RCB_CROSS_TASK` | 1 | cli/rcb_runner.py:520, cli/rcb_step2.py:435, cli/rcb_step2.py:1768 | — | external |
| `HUGINN_RCB_CROSS_TASK_DIR` | str(get_runtime_home( | cli/rcb_runner.py:522, cli/rcb_step2.py:437 | — | external |
| `HUGINN_RCB_DEADLINE` | '' | cli/rcb_step3.py:253 | — | external |
| `HUGINN_RCB_EXEC_ITERS` | 20 | cli/rcb_step2.py:244 | — | external |
| `HUGINN_RCB_FORK_ENABLED` | 1 | cli/rcb_step2.py:287 | — | external |
| `HUGINN_RCB_FORK_K_MAX` | 3 | cli/rcb_step2.py:286 | — | external |
| `HUGINN_RCB_MAX_COMPLETE_REJECTIONS` | 3 | cli/rcb_step2.py:505 | — | external |
| `HUGINN_RCB_TIMEOUT` | 7200 | cli/rcb_runner.py:376 | cli/rcb_runner.py:351 | code-set |
| `HUGINN_REMOTE_WORK_DIR` | ~/huginn_jobs | config.py:730 | — | external |
| `HUGINN_REQUEST_TIMEOUT_SEC` | _DEFAULT_TIMEOUT_SEC | middleware/limits.py:177 | — | external |
| `HUGINN_RESTRICTED_PYTHON` | 1 | cli/rcb_runner.py:250 | — | external |
| `HUGINN_ROOT_MARKERS` | _DEFAULT_ROOT_MARKERS | agent/streaming.py:203 | cli/rcb_runner.py:93 | code-set |
| `HUGINN_SANDBOX_BLOCKED_PATHS` | '' | permissions.py:234 | cli/rcb_runner.py:87 | code-set |
| `HUGINN_SANDBOX_RELAX` | '' | security/sandbox.py:152 | — | external |
| `HUGINN_SECRET_BACKEND` | env | security/secrets.py:439 | — | external |
| `HUGINN_SELF_GOAL_SYNTHESIS` | 0 | autoloop/engine_reflect.py:785 | — | external |
| `HUGINN_SELF_MODEL` | 0 | autoloop/engine_reflect.py:2300 | — | external |
| `HUGINN_SERVER_URL` | '' | cli/slash_commands.py:268 | — | external |
| `HUGINN_SESSION_TTL_HOURS` | 24 | server_core.py:112 | — | external |
| `HUGINN_SKILL_ABSTRACTION` | 0 | autoloop/engine_reflect.py:706 | — | external |
| `HUGINN_SKILL_CONTEXT` | 0 | autoloop/engine_observe.py:399 | — | external |
| `HUGINN_SKIP_CSM` | '' | agent/reflection.py:229 | — | external |
| `HUGINN_SKIP_LOOP_DETECTOR` | '' | — | cli/rcb_runner.py:107, cli/rcb_runner.py:340 | code-set |
| `HUGINN_SKIP_SMOKE` | 0 | cli/rcb_runner.py:1359 | — | external |
| `HUGINN_SOBKO_HIERARCHICAL_INDEX` | '' | rag/router_retriever.py:123 | — | external |
| `HUGINN_SOBKO_TROUBLESHOOTING` | '' | tools/diagnose_tool.py:74 | — | external |
| `HUGINN_SPECULATIVE_DRAFT_TOKENS` | 5 | models/registry.py:789 | — | external |
| `HUGINN_SPECULATIVE_ENABLED` | '' | models/registry.py:781 | — | external |
| `HUGINN_SPECULATIVE_MODEL` | '' | models/registry.py:786 | — | external |
| `HUGINN_SPECULATOR_HISTORY` | '' | agents/speculator.py:32 | — | external |
| `HUGINN_STANDING_RULES` | '' | tools/adapter.py:515, tools/adapter.py:550 | — | external |
| `HUGINN_STATE_BACKEND` | '' | server_core.py:79 | — | external |
| `HUGINN_STATE_REGISTRY_PATH` | '' | persistence/state_registry.py:74 | — | external |
| `HUGINN_STREAM_IDLE_TIMEOUT` | 60 | agent/streaming.py:64 | — | external |
| `HUGINN_SWARM_DISTRIBUTED` | '' | agents/swarm.py:577 | cli/rcb_runner.py:354, agents/swarm.py:677, agents/swarm.py:668 | code-set |
| `HUGINN_TASK_TOOL_ROUTER` | 0 | agent/core.py:644, agent/streaming.py:1081 | cli/rcb_runner.py:364 | code-set |
| `HUGINN_TEAM_MODE` | '' | config.py:706 | — | external |
| `HUGINN_TELEMETRY_ENABLED` | true | config.py:804 | — | external |
| `HUGINN_TELEMETRY_PATH` | '' | autoloop/phase_gate.py:156 | — | external |
| `HUGINN_THINKING` | high | agent/streaming.py:71, agent/streaming.py:84, cli/causal_runner.py:255, +2 处 | cli/causal_runner.py:159, cli/rcb_runner.py:335 | code-set |
| `HUGINN_THREAD_POOL_SIZE` | 64 | lifespan.py:578 | — | external |
| `HUGINN_TOKEN_BUDGET` | 10000000 | autoloop/budget.py:103 | — | external |
| `HUGINN_TOOLUNIVERSE_ENABLED` | 0 | lifespan.py:164 | — | external |
| `HUGINN_TOOL_COMPRESSION_MAX_TOKENS` | 8000 | agent_config.py:92, config.py:817, tools/adapter.py:442 | — | external |
| `HUGINN_TOOL_TIMEOUT` | '' | tools/timeouts.py:113 | — | external |
| `HUGINN_TORCH_DEVICE` | cpu | cli/rcb_runner.py:270, tools/sci/__init__.py:27 | — | external |
| `HUGINN_TRACE_SHARD_INTERVAL` | str(_DEFAULT_SHARD_INTERVAL | cli/rcb_step2.py:240, events/audit_log.py:54 | events/audit_log.py:652, events/audit_log.py:708, events/audit_log.py:703, events/audit_log.py:762 | code-set |
| `HUGINN_TRAJECTORY_PATTERN` | 0 | autoloop/cognitive_loop.py:1692 | — | external |
| `HUGINN_TRANSCRIPT_DIR` | '' | events/transcript.py:88 | — | external |
| `HUGINN_UNPAYWALL_EMAIL` | user@example.com | tools/literature/pdf_fetch.py:309, tools/literature/search_sources.py:347 | — | external |
| `HUGINN_USE_ATOMWORLD` | 0 | agent/code_act_loop.py:343, agent/code_act_loop.py:397, bench/atomworld_bench.py:53, +2 处 | bench/atomworld_bench.py:154, bench/atomworld_bench.py:162, bench/atomworld_bench.py:173, +1 处, security/code_act_sandbox.py:220, security/code_act_sandbox.py:234 | code-set |
| `HUGINN_USE_COGNITIVE_MAP` | 0 | agent/code_act_loop.py:354, agent/code_act_loop.py:416, bench/mini_rotation_baseline.py:522, +2 处 | cli/rcb_runner.py:325, security/code_act_sandbox.py:236 | code-set |
| `HUGINN_USE_COMPLETION_GATE` | 0 | autoloop/cognitive_loop.py:2815 | — | external |
| `HUGINN_USE_CROSS_DOMAIN` | 0 | autoloop/hypothesis_loop.py:2598 | — | external |
| `HUGINN_USE_DOCKER` | '' | security/sandbox.py:591 | — | external |
| `HUGINN_USE_EVOLUTION_MANAGER` | 1 | autoloop/engine_reflect.py:2146, autoloop/engine_reflect.py:2259, memory/manager.py:1376 | — | external |
| `HUGINN_USE_KNOWLEDGE_GRAPH` | 0 | cli/rcb_runner.py:547 | cli/rcb_runner.py:333 | code-set |
| `HUGINN_USE_MENTAL_IMAGERY` | 0 | cli/rcb_step2.py:738 | cli/rcb_runner.py:332 | code-set |
| `HUGINN_USE_MISI` | 0 | bench/misi_bench.py:77, bench/misi_bench.py:208 | bench/misi_bench.py:211, bench/misi_bench.py:219, bench/misi_bench.py:262 | code-set |
| `HUGINN_USE_RUST_SANDBOX` | '' | tools/bash_tool.py:202 | — | external |
| `HUGINN_USE_THREE_CABIN` | 0 | autoloop/cognitive_loop.py:2567 | — | external |
| `HUGINN_USE_UNIFIED_DECISION` | 0 | autoloop/cognitive_loop.py:2703 | — | external |
| `HUGINN_VALIDATE_FAIL_THRESHOLD` | 0.8 | autoloop/engine.py:365 | — | external |
| `HUGINN_VALIDATE_WINDOW` | 100 | autoloop/engine.py:363 | — | external |
| `HUGINN_VAULT_ADDR` | '' | security/secrets.py:166 | — | external |
| `HUGINN_VAULT_TOKEN` | '' | security/secrets.py:167 | — | external |
| `HUGINN_WEB_SEARCH_TIMEOUT` | 15 | tools/web_search_tool.py:45 | — | external |
| `HUGINN_WECOM_TOKEN` | '' | routes/bot.py:285, routes/bot.py:303 | — | external |
| `HUGINN_WETLAB_ENDPOINT` | '' | tools/wetlab_rpc_tool.py:426 | — | external |
| `HUGINN_WM_SUMMARIZE` | rule | config.py:825, memory/session.py:234 | memory/session.py:407, memory/session.py:424, memory/session.py:434, +1 处, memory/session.py:378, memory/session.py:468 | code-set |
| `HUGINN_WM_SUMMARIZE_EVERY_N` | 5 | config.py:830, memory/session.py:68 | — | external |
| `HUGINN_WM_TOKEN_BUDGET` | 8192 | config.py:827, memory/session.py:63 | — | external |
| `HUGINN_WORKSPACE` | . | cli/causal_runner.py:179, config.py:749, config.py:1342, +9 处 | — | external |
| `HUGINN_WORLD_MODEL` | 0 | autoloop/engine_observe.py:362 | — | external |
| `HUGINN_WS_MAX_CONNECTIONS` | 50 | middleware/ws_governance.py:59 | — | external |
| `HUGINN_WS_MAX_MSGS_PER_SEC` | 20 | middleware/ws_governance.py:60 | — | external |

共 266 个环境变量。
