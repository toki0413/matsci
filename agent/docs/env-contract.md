# 环境变量契约 (HUGINN_*)

自动生成: `python -m huginn.cli.config_audit --out docs/env-contract.md`.
本表登记 huginn/ 代码库中所有 `HUGINN_*` 环境变量的引用, 是配置面的审计基线. **推断状态仅提示, 不代替人工判定**: `code-set` 表示代码某处设置了它; `external` 表示未在代码内设置 (可能由用户 shell/.env 注入,需人工确认是否为有效配置).

| 变量 | 默认值 | 读取点 | 设置点 | 推断状态 |
|---|---|---|---|---|
| `HUGINN_ACTION_HIST_MAX` | 1000 | autoloop/cognitive_loop.py:78 | — | external |
| `HUGINN_ADMIN_API_KEY` | '' | security/auth.py:118 | — | external |
| `HUGINN_AGENTS` | '' | config.py:1033 | routes/config.py:351 | code-set |
| `HUGINN_AINVOKE_TIMEOUT` | str(_thinking_scale_timeout( | agent/streaming.py:213 | — | external |
| `HUGINN_ALERT_WEBHOOK_URL` | '' | diagnostics/system_health.py:422 | — | external |
| `HUGINN_ALIGNMENT_SURPRISE_TRIGGER` | 0 | autoloop/engine_reflect.py:4118 | — | external |
| `HUGINN_ALLOW_LOCAL_BASH` | '' | config.py:862, security/execution.py:22 | routes/agents.py:116, tools/__init__.py:201, cli/rcb_runner.py:61 | code-set |
| `HUGINN_ALLOW_POWERSHELL` | '' | security/sandbox.py:40 | — | external |
| `HUGINN_ALLOW_UNRESTRICTED_READ` | '' | routes/fs.py:60, tools/file_read_tool.py:100, tools/sci/xrd_sim_tool.py:395 | — | external |
| `HUGINN_API_KEY` | '' | bench/llm_judge.py:179, cli/availability.py:49, config.py:681, +3 处 | bench/llm_judge.py:363, bench/runner.py:957, bench/llm_judge.py:350, bench/runner.py:942 | code-set |
| `HUGINN_APPROVAL_MODE` | '' | agent/code_act_loop.py:997, agent/core.py:560 | — | external |
| `HUGINN_ATOMWORLD_DATA_DIR` | ./atomworld_data | bench/atomworld_bench.py:24 | — | external |
| `HUGINN_AUDIT_SIGNING_KEY` | '' | security/audit.py:683 | — | external |
| `HUGINN_AUTOLOOP_AUTO_APPROVE` | '' | permissions.py:117 | — | external |
| `HUGINN_AUTOLOOP_HUMAN_PAUSE` | 0 | autoloop/cognitive_loop.py:3342, autoloop/engine_reflect.py:5698 | — | external |
| `HUGINN_AUTOLOOP_NO_PLUGINS` | 0 | cli/commands/autoloop.py:37 | — | external |
| `HUGINN_AUTOLOOP_STREAMING` | 1 | autoloop/engine.py:113 | — | external |
| `HUGINN_AUTO_APPROVE` | False | agent_config.py:183, config.py:829, permissions.py:105, +4 处 | — | external |
| `HUGINN_AUTO_WAKE` | 1 | autoloop/engine.py:829 | — | external |
| `HUGINN_BANDIT_Q_PATH` | '' | agent/bandit_controller.py:141 | cli/rcb_runner.py:70 | code-set |
| `HUGINN_BASE_URL` | '' | config.py:723, config.py:971 | — | external |
| `HUGINN_BELIEF_ENTROPY_FACT_CHECK` | 0 | utils/belief_entropy.py:508 | — | external |
| `HUGINN_BELIEF_ENTROPY_HIGH` | 0.7 | utils/belief_entropy.py:516 | — | external |
| `HUGINN_BELIEF_ENTROPY_LOW` | 0.3 | utils/belief_entropy.py:513 | — | external |
| `HUGINN_BENCHMARK_MODE_PROMPT` | '' | agent/core.py:123 | cli/rcb_runner.py:76 | code-set |
| `HUGINN_BING_API_KEY` | '' | tools/web_search_tool.py:425, tools/web_search_tool.py:513 | — | external |
| `HUGINN_BING_ENDPOINT` | https://api.bing.microsoft.com/v7.0/search | tools/web_search_tool.py:517 | — | external |
| `HUGINN_BLIND_RECONSTRUCTION` | 0 | autoloop/engine_reflect.py:1560 | — | external |
| `HUGINN_BOURBAKI_PATH` | '' | tools/__init__.py:202 | — | external |
| `HUGINN_BRANCH_INCUBATOR_DEPTH` | 2 | autoloop/hypothesis_loop.py:133 | — | external |
| `HUGINN_BRANCH_SLICE_MIN_S` | '' | metacog/branch_incubator.py:157 | — | external |
| `HUGINN_BRANCH_VALUE_PRM` | 1 | autoloop/hypothesis_loop.py:110 | — | external |
| `HUGINN_BRAVE_API_KEY` | '' | tools/web_search_tool.py:427, tools/web_search_tool.py:553 | — | external |
| `HUGINN_BUDGET_APPROVAL` | off | budget_pause.py:168 | — | external |
| `HUGINN_BUDGET_DEADLINE_UNIFIED` | 1 | autoloop/engine_control.py:680 | — | external |
| `HUGINN_BUDGET_EXPIRE_SEMANTICS` | 1 | autoloop/cognitive_loop.py:4078 | — | external |
| `HUGINN_BUDGET_MIN_SLICE_S` | '' | agent/streaming.py:133 | — | external |
| `HUGINN_BUDGET_MODE` | cumulative | scheduling/scheduler.py:115 | — | external |
| `HUGINN_BUDGET_RESERVE_S` | '' | agent/streaming.py:132 | — | external |
| `HUGINN_CACHE_DIR` | str(get_runtime_home( | autoloop/goal_store.py:124, autoloop/plan_store.py:192, cli/rcb_runner.py:64, +21 处 | agents/task_dag.py:334, autoloop/bandit.py:482, autoloop/variant_gen.py:319, +11 处, autoloop/variant_gen.py:330, tools/evidence_reducer.py:147, tools/observation_pack.py:301 | code-set |
| `HUGINN_CHECKPOINTER_MAX_MESSAGES` | 120 | agent/streaming.py:1432 | — | external |
| `HUGINN_CHECKPOINTER_PATH` | '' | agent_config.py:154, bench/orchestrator.py:185, checkpointer.py:90, +4 处 | bench/orchestrator.py:539, bench/orchestrator.py:576, bench/orchestrator.py:594, bench/orchestrator.py:514 | code-set |
| `HUGINN_CI` | '' | tools/sci/gp_tool.py:804 | — | external |
| `HUGINN_CODEACT_MEM_CAP` | 2147483648 | agent/code_act_loop.py:1092 | — | external |
| `HUGINN_CODELAB_REPAIR_ATTEMPTS` | 3 | autoloop/engine_act.py:886 | — | external |
| `HUGINN_CODELAB_SCAFFOLD` | '' | autoloop/engine_act.py:625 | — | external |
| `HUGINN_CODELAB_SLICE` | '' | autoloop/engine_control.py:720 | — | external |
| `HUGINN_CODELAB_SLICE_MIN_S` | '' | autoloop/engine_control.py:746 | — | external |
| `HUGINN_CODELAB_TAIL_RESERVE_S` | '' | autoloop/engine_control.py:733 | — | external |
| `HUGINN_CODELAB_TIMEOUT_S` | 600 | autoloop/engine_act.py:621, autoloop/engine_control.py:732 | — | external |
| `HUGINN_CODER_DONE_MARKER` | [DONE] | config.py:1482 | — | external |
| `HUGINN_CODER_MAX_ITER` | 50 | config.py:1481 | — | external |
| `HUGINN_COEVOLUTION` | 0 | cli/rcb_runner.py:1666 | — | external |
| `HUGINN_COGNITIVE_LLM_DECIDER` | 1 | autoloop/engine.py:537 | cli/rcb_runner.py:103 | code-set |
| `HUGINN_COMPACT_KIND` | remote | agent/reflection.py:37 | — | external |
| `HUGINN_COMPACT_STRATEGY` | trim,summarize | agent/streaming.py:1355, agent/streaming.py:1463 | — | external |
| `HUGINN_COMPACT_THRESHOLD` | '' | models/registry.py:955 | — | external |
| `HUGINN_COMPLETION_CHECK_INTERVAL` | 10 | cli/rcb_step2.py:2844 | — | external |
| `HUGINN_CONFIG_FILE` | huginn.toml | config.py:1517, routes/config.py:56, routes/config.py:897, +3 处 | — | external |
| `HUGINN_CONSOLIDATE_MODEL` | '' | memory/longterm.py:1936 | — | external |
| `HUGINN_CONSOLIDATE_PROVIDER` | deepseek | memory/longterm.py:1935 | — | external |
| `HUGINN_CONTAINER_IMAGE` | '' | config.py:779, security/execution.py:28 | — | external |
| `HUGINN_CONTAINER_RUNTIME` | none | config.py:776, security/execution.py:27 | — | external |
| `HUGINN_CONTEXT_BUDGET_TOKENS` | 0 | agent_config.py:216, config.py:897 | — | external |
| `HUGINN_CONTEXT_MAX_MESSAGES` | 60 | agent/core.py:605 | — | external |
| `HUGINN_CORE_API_KEY` | '' | tools/literature/search_sources.py:881 | — | external |
| `HUGINN_CORE_SUPPORT_PROTOCOL` | 1 | tools/bash_tool.py:213, tools/code_tool.py:132 | — | external |
| `HUGINN_CORS_ORIGINS` | '' | lifespan.py:1030 | — | external |
| `HUGINN_CPU_HOUR_BUDGET` | '' | scheduling/scheduler.py:113 | — | external |
| `HUGINN_CREDENTIAL_DB` | '' | security/credential_store.py:526 | — | external |
| `HUGINN_CREDENTIAL_KEY_FILE` | '' | security/credential_store.py:98 | — | external |
| `HUGINN_CSM_SUBSET_MODE` | '' | agent/reflection.py:488 | cli/causal_runner.py:224, cli/rcb_runner.py:68 | code-set |
| `HUGINN_DARWIN_LLM_EVAL` | 0 | metacog/llm_likelihood.py:48, metacog/llm_likelihood.py:283, metacog/step_evaluator.py:620, +1 处 | metacog/llm_likelihood.py:286, metacog/llm_likelihood.py:289, metacog/step_evaluator.py:1291, +3 处, metacog/step_evaluator.py:1403 | code-set |
| `HUGINN_DARWIN_LLM_INTERVAL` | 5 | metacog/step_evaluator.py:662, metacog/step_evaluator.py:1288 | metacog/step_evaluator.py:1292, metacog/step_evaluator.py:1345, metacog/step_evaluator.py:1405, metacog/step_evaluator.py:1407 | code-set |
| `HUGINN_DARWIN_STAGNATION_LIMIT` | 5 | autoloop/cognitive_loop.py:1219, autoloop/cognitive_loop.py:1496 | cli/rcb_runner.py:354 | code-set |
| `HUGINN_DEEPSEEK_OCR_PATH` | '' | tools/vision_describe_tool.py:70 | — | external |
| `HUGINN_DEV_MODE` | '' | middleware/error_normalize.py:167, middleware/ws_governance.py:168, permissions.py:107, +3 处 | — | external |
| `HUGINN_DISABLE_WEB_SEARCH` | '' | tools/_offline.py:22 | tools/web_search_tool.py:926, tools/web_search_tool.py:931 | code-set |
| `HUGINN_DOCKER_SANDBOX` | '' | security/sandbox.py:882 | — | external |
| `HUGINN_DOC_ENGINE` | auto | perception/pdf_parser.py:555 | — | external |
| `HUGINN_EMBED_MODEL` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | knowledge/store.py:37 | — | external |
| `HUGINN_EM_RECALL_TOP_K` | 5 | config.py:949 | — | external |
| `HUGINN_ENABLE_AGENT_COLLAB` | 0 | cli/commands/autoloop.py:81 | — | external |
| `HUGINN_ENABLE_EXPLORATION` | true | config.py:830 | — | external |
| `HUGINN_ENABLE_SCIHUB` | '' | tools/literature/pdf_fetch.py:306 | — | external |
| `HUGINN_ENCRYPTION_ENABLED` | '' | config.py:864 | — | external |
| `HUGINN_ENCRYPTION_KEY` | '' | security/credential_store.py:624 | — | external |
| `HUGINN_ENCRYPTION_KEY_FILE` | '' | config.py:875 | — | external |
| `HUGINN_ENCRYPTION_PASSWORD` | '' | config.py:874, config.py:1336, config.py:1372, +1 处 | — | external |
| `HUGINN_ENCRYPT_CONFIG` | '' | config.py:868 | — | external |
| `HUGINN_ENCRYPT_RAG_DOCS` | true | config.py:876 | — | external |
| `HUGINN_ENCRYPT_RAG_META` | true | config.py:880 | — | external |
| `HUGINN_ENFORCE_WRITE_CAPABILITY` | 1 | security/auth.py:397 | — | external |
| `HUGINN_ENV` | '' | security/auth.py:51, server.py:83 | — | external |
| `HUGINN_EPISODIC_REPLAY` | 0 | autoloop/engine_observe.py:626 | — | external |
| `HUGINN_EPISODIC_SHARD_INTERVAL` | _DEFAULT_INTERVAL | memory/episodic_shard.py:81 | memory/episodic_shard.py:259, memory/episodic_shard.py:313, memory/episodic_shard.py:327 | code-set |
| `HUGINN_EXECUTION_BACKEND` | local | config.py:773 | — | external |
| `HUGINN_EXTREME_DISPATCH` | 0 | agent/core.py:734, config.py:942 | cli/rcb_runner.py:339, agent/core.py:735, agent/core.py:743, memory/session.py:418, agent/core.py:741, memory/session.py:398, memory/session.py:406 | code-set |
| `HUGINN_EZPROXY_DOMAINS` | '' | tools/literature/crawl_web.py:205 | — | external |
| `HUGINN_EZPROXY_PREFIX` | '' | tools/literature/crawl_web.py:198 | — | external |
| `HUGINN_FAILURE_INVERSION` | 0 | autoloop/engine_reflect.py:4608 | — | external |
| `HUGINN_FALLBACK_STREAM_IDLE` | '' | agent/streaming.py:178 | — | external |
| `HUGINN_FEATURE_LOOP_DETECTOR` | '' | — | cli/rcb_runner.py:109 | code-set |
| `HUGINN_FILE_READ_MAX_OUTPUT_TOKENS` | str(DEFAULT_MAX_OUTPUT_TOKENS | tools/file_read_tool.py:360 | — | external |
| `HUGINN_FILE_READ_MAX_SIZE_BYTES` | str(DEFAULT_MAX_SIZE_BYTES | tools/file_read_tool.py:355 | — | external |
| `HUGINN_FP_DB` | _DEFAULT_DB | rag/adaptive_parser.py:35 | — | external |
| `HUGINN_FS_READ_MAX_SIZE_BYTES` | str(_FS_READ_MAX_BYTES | routes/fs.py:209 | — | external |
| `HUGINN_FTS_AUTO_REBUILD` | 1 | memory/longterm.py:645 | — | external |
| `HUGINN_GOVERNANCE_DEFAULT_DECISION` | deny | governance.py:41 | — | external |
| `HUGINN_GPU_HOUR_BUDGET` | '' | scheduling/scheduler.py:114 | — | external |
| `HUGINN_HARD_CHECKPOINT_PHASES` | '' | autoloop/phase_gate.py:54 | — | external |
| `HUGINN_HARNESS_GATES` | 0 | evolution/manager.py:36 | — | external |
| `HUGINN_HEALTH_MONITOR` | 0 | tools/adapter.py:610 | cli/rcb_runner.py:105 | code-set |
| `HUGINN_HF_ENDPOINT` | '' | knowledge/store.py:45 | — | external |
| `HUGINN_HIDE_DOCS` | '' | security/auth.py:51, server.py:83 | — | external |
| `HUGINN_HILS_ATTENTION` | 1 | memory/longterm.py:983 | memory/longterm.py:2443, memory/longterm.py:2445 | code-set |
| `HUGINN_HINT_COORDINATOR` | 1 | cli/rcb_step2.py:2587 | — | external |
| `HUGINN_HPC_HOST` | '' | cli/availability.py:53 | — | external |
| `HUGINN_HTTP_QPS` | 5 | tools/literature/_http.py:106 | — | external |
| `HUGINN_HUMAN_CHECKPOINT_PHASES` | '' | autoloop/phase_gate.py:31 | — | external |
| `HUGINN_INHERIT_STABLE_PRINCIPLES` | 1 | memory/longterm.py:2127 | — | external |
| `HUGINN_ISING_RERANK` | 1 | memory/longterm.py:931 | memory/longterm.py:2388, memory/longterm.py:2390 | code-set |
| `HUGINN_ITER_HIST_MAX` | 50 | autoloop/cognitive_loop.py:80 | — | external |
| `HUGINN_JEPA_CORPUS` | '' | autoloop/engine_reflect.py:694 | — | external |
| `HUGINN_JEPA_EMBED_MODEL` | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | autoloop/engine_reflect.py:3372 | — | external |
| `HUGINN_JWT_SECRET` | '' | security/auth.py:81 | — | external |
| `HUGINN_KB_CHUNKS_PATH` | '' | evolution/knowledge_distiller.py:693 | — | external |
| `HUGINN_KEEP_ROOT_N` | 2 | agent/streaming.py:627, agent/streaming.py:2324 | cli/rcb_runner.py:89 | code-set |
| `HUGINN_KG_DEPTH` | 1 | config.py:837 | — | external |
| `HUGINN_KG_ENABLED` | '' | config.py:836 | — | external |
| `HUGINN_KG_TOP_K` | 10 | config.py:838 | — | external |
| `HUGINN_LAZY_CLI` | '' | cli/main.py:138 | — | external |
| `HUGINN_LITERATURE_TIMEOUT` | 20 | tools/literature/_http.py:31 | — | external |
| `HUGINN_LLM_LIKELIHOOD` | 0 | metacog/llm_likelihood.py:44, metacog/llm_likelihood.py:282 | metacog/llm_likelihood.py:285, metacog/llm_likelihood.py:290, metacog/llm_likelihood.py:293 | code-set |
| `HUGINN_LLM_LIKELIHOOD_INTERVAL` | 5 | metacog/llm_likelihood.py:54 | metacog/llm_likelihood.py:297, metacog/llm_likelihood.py:299 | code-set |
| `HUGINN_LLM_REQUEST_TIMEOUT` | 120 | models/registry.py:161 | — | external |
| `HUGINN_LOCAL_ONLY` | 0 | config.py:860, models/registry.py:2013 | — | external |
| `HUGINN_LOCAL_VISION_MODEL` | qwen2.5-vl | agent/streaming.py:1983, vision/local_decoder.py:24 | — | external |
| `HUGINN_LOCAL_VISION_TIMEOUT` | 20 | vision/local_decoder.py:30 | — | external |
| `HUGINN_LOCAL_VISION_TTL` | 2.0 | vision/local_decoder.py:28 | — | external |
| `HUGINN_LOG_LEVEL` | INFO | utils/json_logging.py:126 | — | external |
| `HUGINN_LOOP_LIGHT_ON_TRACK` | 1 | autoloop/cognitive_loop.py:3063 | — | external |
| `HUGINN_LOOP_ROLLBACK` | 1 | autoloop/cognitive_loop.py:2712 | — | external |
| `HUGINN_MAINTENANCE` | '' | middleware/maintenance.py:40 | — | external |
| `HUGINN_MAX_BODY_SIZE_MB` | _DEFAULT_MAX_BODY_MB | middleware/limits.py:86 | — | external |
| `HUGINN_MAX_BRANCHES` | 5 | config.py:834 | — | external |
| `HUGINN_MAX_CONCURRENT_SUBAGENTS` | 3 | config.py:749 | — | external |
| `HUGINN_MAX_CONSECUTIVE_FAILURES` | 20 | autoloop/engine.py:488 | cli/rcb_runner.py:351 | code-set |
| `HUGINN_MAX_CONTEXT_TOKENS` | '' | context_manager.py:119 | — | external |
| `HUGINN_MAX_FAILURES_BY_TYPE` | '' | autoloop/engine.py:74 | — | external |
| `HUGINN_MAX_HEAVY_CONCURRENT` | 2 | scheduling/scheduler.py:111 | — | external |
| `HUGINN_MAX_LIGHT_CONCURRENT` | 8 | scheduling/scheduler.py:112 | — | external |
| `HUGINN_MAX_PIVOTS` | 10 | autoloop/engine.py:519 | cli/rcb_runner.py:353 | code-set |
| `HUGINN_MAX_REFINES` | 20 | autoloop/engine.py:514 | cli/rcb_runner.py:352 | code-set |
| `HUGINN_MAX_TOKENS` | '' | config.py:1009 | — | external |
| `HUGINN_MAX_TOOL_CALLS` | 15 | agent_config.py:130 | — | external |
| `HUGINN_MAX_TOOL_CALLS_PER_TOOL` | 5 | agent_config.py:132 | — | external |
| `HUGINN_MAX_TOOL_OUTPUT_TOKENS` | 25000 | agent_config.py:128, config.py:894, config.py:1465, +1 处 | — | external |
| `HUGINN_MCMC_ALIGNMENT` | 0 | cli/rcb_runner.py:2002, cli/rcb_step2.py:2835 | — | external |
| `HUGINN_MCMC_ALIGNMENT_TEMPERATURE` | 1.0 | cli/rcb_runner.py:2008, cli/rcb_step2.py:2837 | — | external |
| `HUGINN_MCMC_ANCHOR` | 1 | autoloop/engine_observe.py:1049 | — | external |
| `HUGINN_MCMC_ANCHOR_LAMBDA` | 0.1 | autoloop/engine_observe.py:1048 | — | external |
| `HUGINN_MCMC_CHAINS` | 4 | cli/rcb_runner.py:1962 | — | external |
| `HUGINN_MCMC_CHECKPOINT_INTERVAL` | 10000 | cli/rcb_runner.py:1968, cli/rcb_step2.py:2825 | — | external |
| `HUGINN_MCMC_CKPT_DIR` | '' | metacog/hypothesis_manifold.py:1735 | — | external |
| `HUGINN_MCMC_GLOBAL_PROPOSAL` | 0.3 | cli/rcb_mcmc.py:366 | — | external |
| `HUGINN_MCMC_GRAMIAN` | 1 | autoloop/engine_observe.py:1043, cli/rcb_mcmc.py:333, cli/rcb_step2.py:2251 | — | external |
| `HUGINN_MCMC_GRAMIAN_K` | 1 | autoloop/engine_observe.py:1044, cli/rcb_mcmc.py:334, cli/rcb_step2.py:2252 | — | external |
| `HUGINN_MCMC_HAPTIC` | 1 | cli/rcb_runner.py:1990, cli/rcb_step2.py:2830 | — | external |
| `HUGINN_MCMC_HAPTIC_TEMPERATURE` | 1.0 | cli/rcb_runner.py:1996, cli/rcb_step2.py:2833 | — | external |
| `HUGINN_MCMC_INTERVAL` | 5 | autoloop/engine_observe.py:1001, cli/rcb_step2.py:2823 | — | external |
| `HUGINN_MCMC_MODE` | '' | cli/rcb_runner.py:1950 | — | external |
| `HUGINN_MCMC_NO_ANNEAL` | 0 | cli/rcb_mcmc.py:364 | — | external |
| `HUGINN_MCMC_PARALLEL` | 1 | metacog/hypothesis_manifold.py:1067 | — | external |
| `HUGINN_MCMC_SE3` | 0 | cli/rcb_runner.py:1978 | — | external |
| `HUGINN_MCMC_SE3_ANGLE_SIGMA` | 30.0 | cli/rcb_runner.py:1984 | — | external |
| `HUGINN_MCMC_SEED` | 42 | autoloop/engine.py:840, cli/rcb_mcmc.py:561, cli/rcb_step2.py:2812, +1 处 | — | external |
| `HUGINN_MCMC_STEPS` | 7000000 | cli/rcb_runner.py:1956 | — | external |
| `HUGINN_MCMC_T_HIGH` | 10 | cli/rcb_mcmc.py:365 | — | external |
| `HUGINN_MCP_ALLOWED_COMMANDS` | '' | config.py:819, mcp_client.py:76 | — | external |
| `HUGINN_MEMORY_CLUSTER` | 0 | memory/longterm.py:1853 | — | external |
| `HUGINN_MEMORY_DECAY_ENABLED` | '' | agent_config.py:156, config.py:916 | — | external |
| `HUGINN_MEMORY_DECAY_INTERVAL_TURNS` | 0 | agent_config.py:159, config.py:921 | — | external |
| `HUGINN_MEMORY_DECAY_PRUNE_THRESHOLD` | 0.15 | agent_config.py:162, config.py:924 | — | external |
| `HUGINN_MEMORY_DIR` | '' | memory/manager.py:1162 | — | external |
| `HUGINN_MODEL` | '' | config.py:721, config.py:969, config.py:1461 | — | external |
| `HUGINN_MODELS` | '' | config.py:1020 | routes/config.py:349 | code-set |
| `HUGINN_MODEL_TIER` | full | plugins/model_tier.py:88 | — | external |
| `HUGINN_MP_API_KEY` | '' | tools/literature/search_sources.py:1738 | — | external |
| `HUGINN_MULTI_AGENT` | 0 | memory/longterm.py:2137 | — | external |
| `HUGINN_NORMALIZE_ERRORS` | '' | middleware/error_normalize.py:161 | — | external |
| `HUGINN_NOVELTY_EVAL` | 0 | autoloop/cognitive_loop.py:3673 | — | external |
| `HUGINN_OAUTH_TOKEN` | '' | cli/availability.py:52 | — | external |
| `HUGINN_OCR_ENGINE` | auto | knowledge/ocr_loader.py:195, knowledge/ocr_loader.py:262, knowledge/ocr_loader.py:313 | — | external |
| `HUGINN_OTEL_BATCH_SIZE` | 32 | otel.py:292 | — | external |
| `HUGINN_OTEL_ENDPOINT` | '' | otel.py:274 | — | external |
| `HUGINN_OTEL_HEADERS` | '' | otel.py:281 | — | external |
| `HUGINN_OTEL_INTERVAL` | 5 | otel.py:293 | — | external |
| `HUGINN_OTEL_SERVICE_NAME` | huginn | otel.py:291 | — | external |
| `HUGINN_PASTE_OFFLOAD` | 1 | tools/paste_offload.py:33 | — | external |
| `HUGINN_PASTE_OFFLOAD_THRESHOLD` | '' | tools/paste_offload.py:43 | — | external |
| `HUGINN_PERSISTENT_GOAL_MODE` | 0 | autoloop/cognitive_loop.py:1181, autoloop/cognitive_loop.py:1283, autoloop/cognitive_loop.py:4069, +5 处 | cli/rcb_runner.py:357, cli/commands/autoloop.py:265 | code-set |
| `HUGINN_PERSISTENT_TERMINAL_TIMEOUT` | '' | tools/persistent_terminal.py:37 | — | external |
| `HUGINN_PERSONA` | default | config.py:737, config.py:844 | cli/commands/persona.py:131, routes/agents.py:507 | code-set |
| `HUGINN_PERSONA_AUTO_ROUTE` | true | config.py:845 | — | external |
| `HUGINN_PERSONA_AUTO_ROUTE_THRESHOLD` | 0.3 | config.py:850 | — | external |
| `HUGINN_PERSONA_AUTO_SYNC` | '' | config.py:852 | — | external |
| `HUGINN_PERSONA_SYNC_DIR` | '' | config.py:854 | — | external |
| `HUGINN_PER_HYP_BUDGET` | 0 | autoloop/cognitive_loop.py:1663, autoloop/cognitive_loop.py:3672, autoloop/engine_reflect.py:1580 | — | external |
| `HUGINN_PET_NAME` | 渡鸦 | config.py:931 | — | external |
| `HUGINN_PET_PERSONALITY` | cheerful | config.py:933 | — | external |
| `HUGINN_PLAN_AUTO_CONFIRM` | 0 | config.py:751 | — | external |
| `HUGINN_PMK_INJECT` | 0 | autoloop/engine_observe.py:685 | — | external |
| `HUGINN_PM_C_MIN` | 0.2 | config.py:950 | — | external |
| `HUGINN_PRIVACY_BLOCK_ON_SECRETS` | '' | config.py:884 | — | external |
| `HUGINN_PRIVACY_REDACT_SECRETS` | true | config.py:870 | — | external |
| `HUGINN_PRM_MODEL` | '' | runtime/step_verifier.py:320 | — | external |
| `HUGINN_PRM_PROVIDER` | '' | runtime/step_verifier.py:319 | — | external |
| `HUGINN_PRM_VERIFIER` | 0 | agents/factory.py:401 | — | external |
| `HUGINN_PROFILE_MODEL` | '' | tools/personalization_tool.py:158 | — | external |
| `HUGINN_PROFILE_PROVIDER` | deepseek | tools/personalization_tool.py:157 | — | external |
| `HUGINN_PROGRESSIVE_BUDGET_BY_REMAINING` | 1 | autoloop/engine_control.py:760 | — | external |
| `HUGINN_PROGRESS_INVARIANT` | '' | autoloop/cognitive_loop.py:4279 | — | external |
| `HUGINN_PROGRESS_INVARIANT_WINDOW` | '' | autoloop/cognitive_loop.py:4283 | — | external |
| `HUGINN_PROMPT_CACHE_CONTROL` | true | config.py:899 | — | external |
| `HUGINN_PROVENANCE_ENABLED` | 1 | provenance/registry.py:431 | — | external |
| `HUGINN_PROVIDER` | default | config.py:678, config.py:1462 | — | external |
| `HUGINN_PRT_LEVEL1` | 0 | agents/factory.py:393 | — | external |
| `HUGINN_RAG_ENABLED` | '' | config.py:835 | — | external |
| `HUGINN_RATE_LIMIT_ENABLED` | 1 | security/rate_limiter.py:517 | cli/rcb_runner.py:58 | code-set |
| `HUGINN_RATE_LIMIT_PER_MINUTE` | 0 | server.py:113 | — | external |
| `HUGINN_RATE_LIMIT_TOKENS_PER_SECOND` | 5000 | security/rate_limiter.py:509 | — | external |
| `HUGINN_RATE_LIMIT_TOKENS_PER_TURN` | 100000 | security/rate_limiter.py:506 | cli/rcb_runner.py:59 | code-set |
| `HUGINN_RATE_LIMIT_TOTAL_COST_USD` | 10.0 | security/rate_limiter.py:512 | — | external |
| `HUGINN_RATE_LIMIT_WARNING_THRESHOLD` | 0.8 | security/rate_limiter.py:515 | — | external |
| `HUGINN_RCB_BLOCKED_TOOLS` | # 默认空集 — extreme 模式信任 agent, 全开放.
            " | cli/rcb_runner.py:752 | — | external |
| `HUGINN_RCB_CROSS_TASK` | 1 | cli/rcb_runner.py:574, cli/rcb_step2.py:2121, cli/rcb_step2.py:2857 | — | external |
| `HUGINN_RCB_CROSS_TASK_DIR` | str(get_runtime_home( | cli/rcb_runner.py:578, cli/rcb_step2.py:2859 | — | external |
| `HUGINN_RCB_DEADLINE` | '' | cli/rcb_step3.py:469 | — | external |
| `HUGINN_RCB_EXEC_ITERS` | 20 | cli/rcb_step2.py:2642 | — | external |
| `HUGINN_RCB_FORK_ENABLED` | 1 | cli/rcb_step2.py:2690 | — | external |
| `HUGINN_RCB_FORK_K_MAX` | 3 | cli/rcb_step2.py:2689 | — | external |
| `HUGINN_RCB_MAX_COMPLETE_REJECTIONS` | 3 | cli/rcb_step2.py:2948 | — | external |
| `HUGINN_RCB_TIMEOUT` | 7200 | cli/rcb_runner.py:386 | cli/rcb_runner.py:358 | code-set |
| `HUGINN_RECALL_NOISE_FLOOR` | 0.35 | memory/manager.py:535 | — | external |
| `HUGINN_REMOTE_WORK_DIR` | ~/huginn_jobs | config.py:791 | — | external |
| `HUGINN_RENAME_DEBT_LIMIT` | 8 | autoloop/cognitive_loop.py:238, autoloop/replay_audit.py:52 | — | external |
| `HUGINN_REPEAT_EXEC_HARD_STREAK` | 2 | autoloop/engine_reflect.py:52 | — | external |
| `HUGINN_REQUEST_TIMEOUT_SEC` | _DEFAULT_TIMEOUT_SEC | middleware/limits.py:181 | — | external |
| `HUGINN_RESTRICTED_PYTHON` | 1 | cli/rcb_runner.py:254 | — | external |
| `HUGINN_ROOT_MARKERS` | _DEFAULT_ROOT_MARKERS | agent/streaming.py:406 | cli/rcb_runner.py:93 | code-set |
| `HUGINN_SANDBOX_BLOCKED_PATHS` | '' | permissions.py:373, validation/scope_authority.py:89 | cli/rcb_runner.py:87 | code-set |
| `HUGINN_SANDBOX_ISOLATE_NETWORK` | '' | security/sandbox.py:54 | — | external |
| `HUGINN_SANDBOX_MAX_CPU_SECONDS` | '' | security/sandbox.py:186 | — | external |
| `HUGINN_SANDBOX_MAX_FILE_BYTES` | '' | security/sandbox.py:189 | — | external |
| `HUGINN_SANDBOX_MAX_PROCESSES` | '' | security/sandbox.py:192 | — | external |
| `HUGINN_SANDBOX_RELAX` | '' | security/sandbox.py:486 | — | external |
| `HUGINN_SEARXNG_URL` | '' | tools/web_search_tool.py:429, tools/web_search_tool.py:593 | — | external |
| `HUGINN_SECRET_BACKEND` | env | security/secrets.py:439 | — | external |
| `HUGINN_SELF_GOAL_SYNTHESIS` | 0 | autoloop/engine_reflect.py:2156 | — | external |
| `HUGINN_SELF_MODEL` | 0 | autoloop/engine_reflect.py:4801 | — | external |
| `HUGINN_SERVER_URL` | '' | cli/slash_commands.py:268 | — | external |
| `HUGINN_SESSION_TTL_HOURS` | 24 | server_core.py:114 | — | external |
| `HUGINN_SKILL_ABSTRACTION` | 0 | autoloop/engine_reflect.py:2071 | — | external |
| `HUGINN_SKILL_CONTEXT` | 0 | autoloop/engine_observe.py:607 | — | external |
| `HUGINN_SKIP_CSM` | '' | agent/reflection.py:311 | — | external |
| `HUGINN_SKIP_SMOKE` | 0 | cli/rcb_runner.py:2039 | — | external |
| `HUGINN_SOBKO_HIERARCHICAL_INDEX` | '' | rag/router_retriever.py:124 | — | external |
| `HUGINN_SOBKO_TROUBLESHOOTING` | '' | tools/diagnose_tool.py:75 | — | external |
| `HUGINN_SPECULATIVE_DRAFT_TOKENS` | 5 | models/registry.py:1370 | — | external |
| `HUGINN_SPECULATIVE_ENABLED` | '' | models/registry.py:1363 | — | external |
| `HUGINN_SPECULATIVE_MODEL` | '' | models/registry.py:1367 | — | external |
| `HUGINN_SPECULATOR_HISTORY` | '' | agents/speculator.py:32 | — | external |
| `HUGINN_STALL_AS_ACTION` | 0 | autoloop/cognitive_loop.py:1213 | — | external |
| `HUGINN_STANDING_RULES` | '' | tools/adapter.py:906, tools/adapter.py:922 | — | external |
| `HUGINN_STARTUP_DEADCODE` | '' | diagnostics/startup_validator.py:239 | — | external |
| `HUGINN_STATE_BACKEND` | '' | server_core.py:79 | — | external |
| `HUGINN_STATE_REGISTRY_PATH` | '' | persistence/state_registry.py:74 | — | external |
| `HUGINN_STE_MODE` | agents | plugins/asd_ste100/main.py:131 | plugins/asd_ste100/main.py:140 | code-set |
| `HUGINN_STREAM_FALLBACK_RESUME` | 1 | agent/streaming.py:204 | — | external |
| `HUGINN_STREAM_IDLE_TIMEOUT` | 60 | agent/streaming.py:69 | — | external |
| `HUGINN_STRENGTH_SCHEDULE` | 1 | autoloop/signals.py:230 | — | external |
| `HUGINN_SURPRISE_HIST_CAP` | 200 | autoloop/engine_reflect.py:57 | — | external |
| `HUGINN_SWARM_DISTRIBUTED` | '' | agents/swarm.py:577 | cli/rcb_runner.py:361, agents/swarm.py:677, agents/swarm.py:668 | code-set |
| `HUGINN_TEAM_MODE` | '' | config.py:746 | — | external |
| `HUGINN_TELEMETRY_ENABLED` | true | config.py:904 | — | external |
| `HUGINN_TELEMETRY_PATH` | '' | autoloop/phase_gate.py:156 | — | external |
| `HUGINN_THINKING` | high | agent/streaming.py:142, agent/streaming.py:155, cli/causal_runner.py:364, +2 处 | cli/causal_runner.py:221, cli/rcb_runner.py:346 | code-set |
| `HUGINN_THREAD_POOL_SIZE` | 64 | lifespan.py:681 | — | external |
| `HUGINN_TOOLUNIVERSE_ENABLED` | 0 | lifespan.py:226 | — | external |
| `HUGINN_TOOL_COMPRESSION_MAX_TOKENS` | 8000 | agent_config.py:135, config.py:909, tools/adapter.py:845 | — | external |
| `HUGINN_TOOL_TIMEOUT` | '' | tools/timeouts.py:113 | — | external |
| `HUGINN_TORCH_DEVICE` | cpu | cli/rcb_runner.py:283, tools/sci/__init__.py:27 | — | external |
| `HUGINN_TRACE_SHARD_INTERVAL` | str(_DEFAULT_SHARD_INTERVAL | cli/rcb_step2.py:2636, events/audit_log.py:54 | events/audit_log.py:671, events/audit_log.py:745, events/audit_log.py:705, events/audit_log.py:763 | code-set |
| `HUGINN_TRAJECTORY_PATTERN` | 0 | autoloop/cognitive_loop.py:2416 | — | external |
| `HUGINN_TRANSCRIPT_DIR` | '' | events/transcript.py:94 | — | external |
| `HUGINN_UNPAYWALL_EMAIL` | user@example.com | tools/literature/pdf_fetch.py:381, tools/literature/search_sources.py:648 | — | external |
| `HUGINN_USE_ATOMWORLD` | 0 | bench/atomworld_bench.py:54, bench/atomworld_bench.py:156, security/code_act_sandbox.py:126 | bench/atomworld_bench.py:159, bench/atomworld_bench.py:167, bench/atomworld_bench.py:179, +1 处, security/code_act_sandbox.py:231, security/code_act_sandbox.py:245 | code-set |
| `HUGINN_USE_COGNITIVE_MAP` | 0 | bench/mini_rotation_baseline.py:645, security/code_act_sandbox.py:133 | cli/rcb_runner.py:340, security/code_act_sandbox.py:247 | code-set |
| `HUGINN_USE_COMPLETION_GATE` | 0 | autoloop/cognitive_loop.py:2832 | — | external |
| `HUGINN_USE_CROSS_DOMAIN` | 0 | autoloop/hypothesis_loop.py:3770 | — | external |
| `HUGINN_USE_DOCKER` | '' | security/sandbox.py:879 | — | external |
| `HUGINN_USE_EVOLUTION_MANAGER` | 1 | autoloop/engine_reflect.py:4584, autoloop/engine_reflect.py:4752, memory/manager.py:261 | — | external |
| `HUGINN_USE_KNOWLEDGE_GRAPH` | 0 | cli/rcb_runner.py:636 | cli/rcb_runner.py:344 | code-set |
| `HUGINN_USE_MENTAL_IMAGERY` | 0 | cli/rcb_step2.py:3065 | cli/rcb_runner.py:343 | code-set |
| `HUGINN_USE_MISI` | 0 | bench/misi_bench.py:78, bench/misi_bench.py:246 | bench/misi_bench.py:252, bench/misi_bench.py:261, bench/misi_bench.py:269 | code-set |
| `HUGINN_USE_RUST_SANDBOX` | '' | tools/bash_tool.py:237 | cli/rcb_runner.py:100 | code-set |
| `HUGINN_USE_UNIFIED_DECISION` | 0 | autoloop/cognitive_loop.py:2830 | — | external |
| `HUGINN_VALIDATE_FAIL_THRESHOLD` | 0.8 | autoloop/engine.py:506 | — | external |
| `HUGINN_VALIDATE_WINDOW` | 100 | autoloop/engine.py:504 | — | external |
| `HUGINN_VAULT_ADDR` | '' | security/secrets.py:166 | — | external |
| `HUGINN_VAULT_TOKEN` | '' | security/secrets.py:167 | — | external |
| `HUGINN_VISION_BASE_URL` | '' | vision/local_decoder.py:119 | — | external |
| `HUGINN_VISION_MODEL` | '' | vision/local_decoder.py:118 | — | external |
| `HUGINN_VISION_PROVIDER` | '' | vision/local_decoder.py:109 | — | external |
| `HUGINN_WEB_SEARCH_TIMEOUT` | 15 | tools/web_search_tool.py:51 | — | external |
| `HUGINN_WECOM_TOKEN` | '' | routes/bot.py:285, routes/bot.py:303 | — | external |
| `HUGINN_WETLAB_ENDPOINT` | '' | tools/wetlab_rpc_tool.py:497 | — | external |
| `HUGINN_WM_SUMMARIZE` | rule | config.py:944, memory/session.py:243 | memory/session.py:448, memory/session.py:470, memory/session.py:487, +1 处, memory/session.py:399, memory/session.py:407 | code-set |
| `HUGINN_WM_SUMMARIZE_EVERY_N` | 5 | config.py:952, memory/session.py:68 | — | external |
| `HUGINN_WM_TOKEN_BUDGET` | 8192 | config.py:948, memory/session.py:63 | — | external |
| `HUGINN_WORKSPACE` | . | cli/causal_runner.py:284, config.py:828, config.py:1524, +11 处 | autoloop/cognitive_loop.py:1892 | code-set |
| `HUGINN_WS_MAX_CONNECTIONS` | 50 | middleware/ws_governance.py:60 | — | external |
| `HUGINN_WS_MAX_MSGS_PER_SEC` | 20 | middleware/ws_governance.py:61 | — | external |

共 324 个环境变量。
