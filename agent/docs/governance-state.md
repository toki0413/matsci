# 低熵治理台账 (governance state)

自动生成: `python -m huginn.cli.governance_audit --out docs/governance-state.md`.
登记 `tests/test_arch_*.py` 门族 + 显式治理元门 (`GOVERNANCE_META_TESTS`)、`.github/workflows/*.yml` **全部工作流**的 python guard step (pytest 与 `python -m huginn.<module>`)、非 pytest 工程门 (`ENGINEER_GATE_SPECS`)、仓根 + `agent/` 两份 `.pre-commit-config.yaml` 本地门, 以及命名子系统的**能力锚点** (`CAPABILITY_SPECS`: 开关默认值/承载模块/不变量测试) 的对齐面: **孤儿门** (存在却未登记)、**悬空门** (指向不存在的测试/模块/step)、**能力锚点脱节** (开关默认翻转/承载模块或回归测试缺失) 在此显形。只提示不判死; 硬判定在 `tests/test_arch_gate_registry.py`.

## 工作流 (workflows) — 7

被扫描的**全部**工作流文件 (门可能因整个工作流被删/改名而集体消失; 数下限棘轮 `workflow_floor` 兜底).

`audit-regression.yml`, `ci.yml`, `coverage-matrix.yml`, `desktop.yml`, `release.yml`, `skill-audit.yml`, `stress-test.yml`

## 工作流触发器 (workflow triggers)

每个工作流的**期望触发集**与**实际触发集**比对 (源: `governance_audit.WORKFLOW_TRIGGER_SPECS`). 门 step 存在 ≠ 门会运行 —— 删 `on: schedule` 或收窄 `push`/`pull_request` 的 `paths` 会让门永不执行, 而 step 仍在文件里 (存在性判据判「在」= 假绿). **状态=漂移** 即触发器缺失/多出; 未登记触发规格的工作流 (文件在、规格缺) 与规格未落地的工作流 (规格在、文件缺) 单列于缺口区.

| 工作流 | 期望触发 | 实际触发 | 状态 |
|---|---|---|---|
| `audit-regression.yml` | `schedule`, `workflow_dispatch` | `schedule`, `workflow_dispatch` | 一致 |
| `ci.yml` | `pull_request`, `push` | `pull_request`, `push` | 一致 |
| `coverage-matrix.yml` | `pull_request`, `push`, `schedule`, `workflow_dispatch` | `pull_request`, `push`, `schedule`, `workflow_dispatch` | 一致 |
| `desktop.yml` | `pull_request`, `push`, `workflow_dispatch` | `pull_request`, `push`, `workflow_dispatch` | 一致 |
| `release.yml` | `push` | `push` | 一致 |
| `skill-audit.yml` | `pull_request`, `push`, `schedule`, `workflow_dispatch` | `pull_request`, `push`, `schedule`, `workflow_dispatch` | 一致 |
| `stress-test.yml` | `pull_request`, `schedule`, `workflow_dispatch` | `pull_request`, `schedule`, `workflow_dispatch` | 一致 |

## 门 (gates)

| 测试 | arch | 状态 | CI step |
|---|---|---|---|
| `tests/security/fuzz_api.py` | 否 | `fast-fail-guard` | Run API fuzz tests |
| `tests/security/pentest_api_security.py` | 否 | `fast-fail-guard` | Run API security PoC tests |
| `tests/security/pentest_archive_safety.py` | 否 | `fast-fail-guard` | Run archive security PoC tests |
| `tests/stress/test_http_stress.py` | 否 | `fast-fail-guard` | Run pytest stress tests (fallback) |
| `tests/test_agent_blackboard.py` | 否 | `fast-fail-guard` | Agent blackboard contract guard |
| `tests/test_arch_cleanliness.py` | 是 | `full-suite-only` | — |
| `tests/test_arch_config_migration.py` | 是 | `fast-fail-guard` | Config migration guard |
| `tests/test_arch_credential_gateway.py` | 是 | `fast-fail-guard` | Credential gateway guard |
| `tests/test_arch_endpoint_contract.py` | 是 | `fast-fail-guard` | Endpoint contract guard |
| `tests/test_arch_gate_registry.py` | 是 | `fast-fail-guard` | Governance gate registry guard |
| `tests/test_arch_matdb_dep.py` | 是 | `fast-fail-guard` | Mat-DB real MP dependency guard |
| `tests/test_arch_mcp_governance.py` | 是 | `fast-fail-guard` | MCP single-gateway guard |
| `tests/test_arch_mcp_json.py` | 是 | `fast-fail-guard` | MCP .mcp.json consumption guard |
| `tests/test_arch_mp_key_flow.py` | 是 | `fast-fail-guard` | MP API key flow guard |
| `tests/test_arch_no_hardcoded_paths.py` | 是 | `fast-fail-guard` | No-hardcoded-machine-paths guard |
| `tests/test_arch_no_import_cycles.py` | 是 | `full-suite-only` | — |
| `tests/test_arch_plugin_governance.py` | 是 | `fast-fail-guard` | Plugin admission guard |
| `tests/test_arch_rag_memory_seam.py` | 是 | `full-suite-only` | — |
| `tests/test_arch_seam_integrity.py` | 是 | `full-suite-only` | — |
| `tests/test_arch_single_gateway.py` | 是 | `fast-fail-guard` | Architecture single-gateway guard |
| `tests/test_arch_skill_pool_bridge.py` | 是 | `fast-fail-guard` | Skill pool bridge guard |
| `tests/test_arch_tools_direction.py` | 是 | `full-suite-only` | — |
| `tests/test_config_contracts.py` | 否 | `full-suite-only` | — |
| `tests/test_contract_audit.py` | 否 | `full-suite-only` | — |
| `tests/test_env_surface_governance.py` | 否 | `full-suite-only` | — |
| `tests/test_mechanics.py` | 否 | `fast-fail-guard` | Run 7 historical-fix regression tests |
| `tests/test_naturebench.py` | 否 | `fast-fail-guard` | NatureBench repro gate |
| `tests/test_ponytail_debt_registry.py` | 否 | `full-suite-only` | — |
| `tests/test_prompt_contract.py` | 否 | `full-suite-only` | — |
| `tests/test_protocol_contract.py` | 否 | `fast-fail-guard` | Protocol adapter contract guard |
| `tests/test_publish_mirror_drift.py` | 否 | `full-suite-only` | — |
| `tests/test_recall_facade.py` | 否 | `fast-fail-guard` | Recall facade contract guard |
| `tests/test_recovery_planner.py` | 否 | `fast-fail-guard` | World model recovery contract guard |
| `tests/test_testclient_hygiene.py` | 否 | `fast-fail-guard` | TestClient hygiene guard |

## CLI 门 (cli gates)

**全部工作流**里 `python -m huginn.<module>` 形态的治理门 (CLI 自查 / 棘轮, 非 pytest). 旧解析只认 pytest 会漏掉; 现单列。**存在=否** 即悬空 CLI 门 (指向已删/改名的模块).

| 模块 | 工作流 | CI step | 存在 |
|---|---|---|---|
| `huginn.cli.contract_audit` | `ci.yml` | MECE wiring-contract guard (ratchet) | 是 |
| `huginn.security.policy_engine` | `ci.yml` | Policy engine self-check | 是 |

## 本地门 (pre-commit local gates)

仓根与 `agent/` 两份 `.pre-commit-config.yaml` 的本地门 (如 G37 audit-regression). 既不在工作流也不在别处登记, 故单列。**存在=否** 即悬空本地门 (引用的测试已改名/删除 —— 此时 hook 的 `files:` 过滤器会静默不再匹配, 门悄悄死掉). **files 覆盖=否** 即 `entry` 引用的测试不在 `files:` 过滤范围内 —— 改它不会触发 hook, 门静默失效.

| 来源 | 本地门 (hook id) | 引用测试 | 存在 | files 覆盖 |
|---|---|---|---|---|
| `.pre-commit-config.yaml` | `audit-regression-7fixes` | `tests/test_context_and_jwt.py` | 是 | 是 |
| `.pre-commit-config.yaml` | `audit-regression-7fixes` | `tests/test_eos_fit.py` | 是 | 是 |
| `.pre-commit-config.yaml` | `audit-regression-7fixes` | `tests/test_fts5_consistency.py` | 是 | 是 |
| `.pre-commit-config.yaml` | `audit-regression-7fixes` | `tests/test_mechanics.py` | 是 | 是 |
| `.pre-commit-config.yaml` | `audit-regression-7fixes` | `tests/test_sim_audit_regression.py` | 是 | 是 |

## 工程门 (engineer gates)

非 pytest 的**阻断门** (ruff lint / pip-audit / npm audit / bandit SAST / cargo audit / gitleaks / lock 漂移 / 前端构建·单测·E2E / 技能库审计 / ultra-long benchmark / quickstart smoke). 由工作流 step 承载, 旧台账完全看不见 —— step 删/改名即静默缩小治理面. 现由 `ENGINEER_GATE_SPECS` 登记并判存在。**存在=否** 即悬空工程门 (step 名与工作流脱节).

| 门 (id) | 工作流 | 类别 | CI step | 存在 |
|---|---|---|---|---|
| `bandit_sast` | `ci.yml` | sast | SAST scan with bandit | 是 |
| `cargo_audit` | `ci.yml` | supply-chain | Cargo dependency vulnerability scan | 是 |
| `cargo_check` | `ci.yml` | build | Cargo check | 是 |
| `cargo_test` | `ci.yml` | test | Cargo test | 是 |
| `desktop_frontend_typecheck` | `desktop.yml` | build | Type-check frontend | 是 |
| `desktop_rust_build` | `desktop.yml` | build | Build Rust CLI + sidecar | 是 |
| `desktop_tauri_build` | `desktop.yml` | build | Build Tauri desktop app | 是 |
| `frontend_build` | `ci.yml` | build | Install and build | 是 |
| `frontend_e2e` | `ci.yml` | test | Run frontend E2E (chromium) | 是 |
| `frontend_unit` | `ci.yml` | test | Run vitest unit tests | 是 |
| `gitleaks` | `ci.yml` | secret | Secret scan with gitleaks | 是 |
| `lint_ruff` | `ci.yml` | lint | Run ruff check (gating) | 是 |
| `lock_drift` | `ci.yml` | supply-chain | Dependency lock drift guard | 是 |
| `npm_audit` | `ci.yml` | supply-chain | npm dependency vulnerability scan | 是 |
| `pip_audit` | `ci.yml` | supply-chain | Dependency vulnerability scan | 是 |
| `quickstart_smoke` | `audit-regression.yml` | smoke | Quickstart smoke test (G38) | 是 |
| `release_frontend_typecheck` | `release.yml` | build | Type-check frontend | 是 |
| `release_rust_build` | `release.yml` | build | Build Rust CLI + sidecar | 是 |
| `release_tauri_build` | `release.yml` | build | Build Tauri desktop app | 是 |
| `skill_audit` | `skill-audit.yml` | ratchet | Run skill library audit (gate) | 是 |
| `ultra_long_bench` | `stress-test.yml` | benchmark | Run ultra-long benchmark (gating) | 是 |

## 能力锚点 (capability anchors)

命名子系统 (RSI / PMK / 世界模型 …) 的生存锚点 (源: `CAPABILITY_SPECS`): **一枚开关 + 一组承载模块 + 一组不变量测试**. 门存在性判据看不见能力面 —— 开关还在但**默认翻转** (从开变关 = 能力悄悄关掉)、承载模块被删 (= 能力失去实现)、不变量测试被删(= 能力失去回归网) 都会让能力静默萎缩而门全绿. **默认一致=否** 即开关被删/默认翻转; 承载模块/不变量测试列**存在=否** 即锚点脱节. 数下限棘轮 `capability_floor` 兜底. `always_on` 型 = 工具 action 面/机制面能力 (无开关可翻转, 硬凑开关 = 噪声), 锚点退化为双判据: 只判承载模块/不变量测试存在, 依赖面另由 L0 探针 (runtime_probe) 守.

| 能力 | 开关 | 类型 | 期望默认 | 实际默认 | 默认一致 | 承载模块 | 不变量测试 |
|---|---|---|---|---|---|---|---|
| ASD-STE100 受控语言纪律 (`asd_ste100`) | `asd_ste100` | feature_flag | true | true | 是 | `huginn/plugins/asd_ste100/main.py` | `tests/test_asd_ste100_plugin.py` |
| 主动提问 (clarification) (`clarification`) | `clarification` | feature_flag | true | true | 是 | `huginn/tools/clarification_tool.py`, `huginn/interaction/clarification.py` | `tests/test_clarification_tool.py` |
| 计算路由策略 + 预算 (`compute_policy`) | `compute_policy` | feature_flag | false | false | 是 | `huginn/execution/compute_policy.py`, `huginn/execution/compute_router.py` | `tests/test_compute_router_audit.py` |
| ARTEX 假设前沿 (frontier) (`frontier`) | `frontier_claim` | feature_flag | false | false | 是 | `huginn/autoloop/control_mechanisms.py`, `huginn/autoloop/cognitive_loop.py` | `tests/test_control_mechanism_governance.py` |
| JEV 外部判断接入 (`jev`) | `jev_enabled` | feature_flag | false | false | 是 | `huginn/runtime/jev/_enabled.py`, `huginn/runtime/jev/client.py`, `huginn/runtime/jev/tool_router.py` | `tests/test_jev_wiring.py` |
| 分子结构文件输入 (MOL/SDF/PDB) (`mol_structure_input`) | `—` | 常开 (无开关) | — | — | 是 | `huginn/tools/sci/rdkit_tool.py` | `tests/test_rdkit_mol_file.py` |
| 个性化 (风格/口味) (`personalization`) | `personalization` | feature_flag | true | true | 是 | `huginn/personalization/user_style.py`, `huginn/personalization/taste_profile.py`, `huginn/tools/personalization_tool.py` | `tests/test_taste_profile.py` |
| Pi 极简内核模式 (`pi_mode`) | `pi_mode` | feature_flag | false | false | 是 | `huginn/modes/pi.py`, `huginn/tools/pi_mode_tool.py` | `tests/test_make_tool_pi_mode.py` |
| PMK 一致性 (persona/memory/knowledge) (`pmk`) | `HUGINN_PMK_INJECT` | env | 0 | 0 | 是 | `huginn/autoloop/cognitive_guard.py`, `huginn/autoloop/engine_observe.py`, `huginn/runtime/task_lifecycle.py` | `tests/test_temporal_pmk.py` |
| 多协议适配层 (共享 IR + 每协议成对) (`protocol_adapter`) | `—` | 常开 (无开关) | — | — | 是 | `huginn/protocols/__init__.py` | `tests/test_protocol_contract.py` |
| 计算 provenance 快照 (`provenance`) | `provenance` | feature_flag | true | true | 是 | `huginn/provenance/pipeline.py`, `huginn/provenance/registry.py`, `huginn/provenance/correction.py` | `tests/test_provenance.py`, `tests/test_argus_provenance.py` |
| 统一召回门面 (memory/context/observation 意图路由) (`recall_facade`) | `—` | 常开 (无开关) | — | — | 是 | `huginn/tools/recall_facade.py`, `huginn/tools/memory_tool.py`, `huginn/tools/recall_context_tool.py` | `tests/test_recall_facade.py` |
| 复现重跑闭环 (reproduce) (`reproduce_closed_loop`) | `—` | 常开 (无开关) | — | — | 是 | `huginn/autoloop/reproduce.py`, `huginn/cli/reproduce.py` | `tests/test_reproduce.py` |
| 递归自我改进 (RSI) (`rsi`) | `harness_meta_improver` | feature_flag | false | false | 是 | `huginn/harness/meta_improver.py`, `huginn/self_improvement/core.py`, `huginn/evolution/engine.py` | `tests/test_meta_improver.py`, `tests/test_evolution_engine.py` |
| NMR/质谱谱图解析 (JCAMP-DX/mzML) (`spectra_parse`) | `—` | 常开 (无开关) | — | — | 是 | `huginn/tools/sci/spectra_parsers.py`, `huginn/tools/characterization_tool.py` | `tests/test_spectra_parsers.py` |
| 投机执行 (speculator) (`speculator`) | `speculator` | feature_flag | true | true | 是 | `huginn/agents/speculator.py` | `tests/test_proactive_injection.py` |
| 系统资源监控 (`system_health`) | `system_health_monitor` | feature_flag | true | true | 是 | `huginn/diagnostics/system_health.py` | `tests/test_system_health.py` |
| 三舱反射模型 (`three_cabin`) | `three_cabin` | feature_flag | false | false | 是 | `huginn/metacog/three_cabin_reflector.py` | `tests/test_verify_4flags.py`, `tests/test_metacog_smoke.py` |
| VISTA 过程级 trace 检索 (`trace_inspect`) | `trace_inspect` | feature_flag | false | false | 是 | `huginn/autoloop/recall_modes.py`, `huginn/autoloop/plan_check.py` | `tests/test_vista_recall_modes.py` |
| VISTA 无损帧记忆 (`visual_frame_memory`) | `visual_frame_memory` | feature_flag | false | false | 是 | `huginn/autoloop/frame_store.py`, `huginn/autoloop/recall_modes.py` | `tests/test_vista_recall_modes.py` |
| 世界模型 (`world_model`) | `world_model` | feature_flag | true | true | 是 | `huginn/capabilities/world_model.py`, `huginn/security/world_model.py`, `huginn/research/law_model.py` | `tests/test_world_model.py`, `tests/test_research_planning_team.py` |
| 世界模型故障恢复层 (五层定位 + 硬门策略 + 代理来源标记) (`worldmodel_recovery`) | `—` | 常开 (无开关) | — | — | 是 | `huginn/security/recovery_planner.py`, `huginn/security/workspace.py`, `huginn/tools/experiment_protocol_tool.py` | `tests/test_recovery_planner.py` |

## prompt 段生效面 (prompt segments)

prompt 段的**声明面** (`prompt_segments._APPLIES_TO`: 段名 -> 必须生效的生产路径 {main,s7,autoloop}) 与**生效面** (各生产路径实际渲染点) 的登记 (源: `governance_audit.PROMPT_SPECS` 期望段名 + 按文件路径 ast 读源文件 `_APPLIES_TO`). 逐段的生效路径**不在此复制** (无镜像漂移); 「声明 == 渲染」由 `tests/test_prompt_contract.py` 硬判定. 此处只显形**清单漂移** (清单有、源里无 = 段被删; 源里有、清单无 = 新增未登记) 与逐段声明路径. 段数下限棘轮 `prompt_segment_floor` 兜底.

| 段 | 声明生效路径 |
|---|---|
| `persona` | `s7` |
| `core_principles` | `main`, `s7`, `autoloop` |
| `behavioral_discipline` | `main`, `s7`, `autoloop` |
| `response_format` | `main`, `s7`, `autoloop` |
| `first_principles` | `main`, `s7` |
| `mode` | `s7` |
| `phase` | `s7` |
| `metacog` | `s7` |
| `tools` | `main`, `s7` |
| `multimodal` | `main`, `s7` |
| `writing` | `main`, `s7` |
| `multi_agent_protocol` | `main`, `s7` |
| `thinking` | `main`, `s7` |
| `safety` | `s7` |

## 预算 (budgets)

治理棘轮的冻结预算/基线单一台账 (源: `governance_audit.BUDGET_SPECS`). **权威** = 值在台账声明、消费者门 `budget_value()` 只读; **引用** = 值仍在其自然归属, 台账只登记。实测/余量仅对 weightless 可测项给出; 其余由各自消费者门校验 (current ≤ frozen); 重型探针 (如镜像漂移实测) 需 `--probe`。**消费者门**列: `ci_step` 型是工作流里 pytest/CLI guard step (有存在性门); `workflow_step` 型是非 pytest 工程门 step; 带 `· 全量` 标记的是 `full_suite` 型 (仅全量套件跑, 不在工作流 step 里)。

| 预算 | 位置 | 模式 | 冻结值 | 实测 | 余量 | 消费者门 |
|---|---|---|---|---|---|---|
| MECE 发现基线 | `tests/golden/mece_findings_baseline.txt` | 引用 | 43 项 | — | — | MECE wiring-contract guard (ratchet) |
| 镜像漂移预算 | `tests/test_publish_mirror_drift.py` | 权威 | 0/0/0 | — | — | publish mirror drift (only-shrink) · 全量 |
| 环境裸读基线 | `huginn/env_schema.py` | 引用 | 463 | — | — | env surface governance · 全量 |
| 外部直连白名单 | `tests/test_arch_single_gateway.py` | 引用 | 19 项 | — | — | Architecture single-gateway guard |
| 冻结委托子命令 | `tests/test_arch_single_gateway.py` | 引用 | 11 项 | — | — | Architecture single-gateway guard |
| 插件准入清单 | `tests/test_arch_plugin_governance.py` | 引用 | 2 项 | — | — | Plugin admission guard |
| 硬编码路径白名单 | `tests/test_arch_no_hardcoded_paths.py` | 引用 | 7 项 | — | — | No-hardcoded-machine-paths guard |
| MCP 接线白名单 | `tests/test_arch_mcp_governance.py` | 引用 | 7 项 | — | — | MCP single-gateway guard |
| arch 门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 17 | 17 | 0 | Governance gate registry guard |
| CLI 门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 2 | 2 | 0 | Governance gate registry guard |
| 本地门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 5 | 5 | 0 | Governance gate registry guard |
| 工程门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 21 | 21 | 0 | Governance gate registry guard |
| 工作流数下限 | `tests/test_arch_gate_registry.py` | 权威 | 6 | 7 | 1 | Governance gate registry guard |
| 快速失败门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 19 | 23 | 4 | Governance gate registry guard |
| 字段审计 golden | `tests/golden/field_audit` | 引用 | 5 文件 | — | — | MECE 字段级面 · 全量 |
| 技能库审计基线 | `scripts/skill_audit_baseline.json` | 引用 | 56/38 | — | — | Run skill library audit (gate) |
| 工作流命名 step 总数下限 | `tests/test_arch_gate_registry.py` | 权威 | 122 | 131 | 9 | Governance gate registry guard |
| 能力锚点数下限 | `tests/test_arch_gate_registry.py` | 权威 | 19 | 22 | 3 | Governance gate registry guard |
| prompt 段数下限 | `tests/test_arch_gate_registry.py` | 权威 | 14 | 14 | 0 | Governance gate registry guard |
| gitleaks 密钥豁免数 | `.gitleaks.toml` | 上限 | 4 | 4 | 0 | Governance gate registry guard |
| pip-audit 漏洞豁免数 | `.github/workflows/ci.yml` | 上限 | 24 | 24 | 0 | Governance gate registry guard |
| dependabot 忽略项数 | `.github/dependabot.yml` | 上限 | 5 | 5 | 0 | Governance gate registry guard |
| 覆盖率下限 | `pyproject.toml` | 下限 | 60 | 60 | 0 | Governance gate registry guard |
| 可运行性探针清单下限 | `huginn/capabilities/smoke_specs.py` | 权威 | 7 | 7 | 0 | Governance gate registry guard |

## 循环闸位 (loop caps)

自主循环的轮数/步数**上限** (源: `governance_audit.BUDGET_SPECS` 中 `group=loop` 项). 这些值此前散在 `env_schema` 与各引擎/路由的硬编码默认里, 无单一上限可读。**模式=上限** (ceiling): 代码持有实测值 (weightless regex 探针从源码读出), 台账持有允许的最大值; 棘轮 = 实测 ≤ 上限 —— 提高上限 (扩大自主面) 必须显式改台账 `value` (可见决策)。**模式=权威** 一行为闸位**数下限** (只减不增)。

| 预算 | 位置 | 模式 | 冻结值 | 实测 | 余量 | 消费者门 |
|---|---|---|---|---|---|---|
| autoloop CLI 迭代上限 | `huginn/cli/commands/autoloop.py` | 上限 | 20 | 20 | 0 | Governance gate registry guard |
| autoloop 引擎迭代上限 | `huginn/autoloop/cognitive_loop.py` | 上限 | 50 | 50 | 0 | Governance gate registry guard |
| 长程 recursion_limit 上限 | `huginn/agent/core.py` | 上限 | 500 | 500 | 0 | Governance gate registry guard |
| 单轮 tool-call 上限 | `huginn/env_schema.py` | 上限 | 15 | 15 | 0 | Governance gate registry guard |
| 单轮单工具调用上限 | `huginn/env_schema.py` | 上限 | 5 | 5 | 0 | Governance gate registry guard |
| CodeAct 轮数上限 | `huginn/agent/code_act_loop.py` | 上限 | 15 | 15 | 0 | Governance gate registry guard |
| coder 路由迭代上限 | `huginn/env_schema.py` | 上限 | 50 | 50 | 0 | Governance gate registry guard |
| 探索编排迭代上限 | `huginn/exploration/orchestrator.py` | 上限 | 40 | 40 | 0 | Governance gate registry guard |
| deep_research 迭代上限 | `huginn/routes/deep_research.py` | 上限 | 12 | 12 | 0 | Governance gate registry guard |
| RCB Step-2 执行迭代上限 | `huginn/cli/rcb_step2.py` | 上限 | 20 | 20 | 0 | Governance gate registry guard |
| harness phase 迭代上限 | `huginn/harness/phase_spec.py` | 上限 | 5 | 5 | 0 | Governance gate registry guard |
| 技能 step 循环上限 | `huginn/skills/base.py` | 上限 | 20 | 20 | 0 | Governance gate registry guard |
| 技能优化迭代上限 | `huginn/skills/presets.py` | 上限 | 10 | 10 | 0 | Governance gate registry guard |
| CodeAct sandbox 轮数上限 | `huginn/security/code_act_sandbox.py` | 上限 | 15 | 15 | 0 | Governance gate registry guard |
| WS 每 turn 自动续跑上限 | `huginn/routes/ws_helpers.py` | 上限 | 3 | 3 | 0 | Governance gate registry guard |
| per-agent 步数上限 | `huginn/config.py` | 上限 | 10 | 10 | 0 | Governance gate registry guard |
| causal_runner tool-call 上限 | `huginn/cli/causal_runner.py` | 上限 | 40 | 40 | 0 | Governance gate registry guard |
| deli_research 迭代上限 | `huginn/academic/deli_research.py` | 上限 | 5 | 5 | 0 | Governance gate registry guard |
| 循环闸位数下限 | `tests/test_arch_gate_registry.py` | 权威 | 18 | 18 | 0 | Governance gate registry guard |

## 缺口 (gaps)

- 孤儿门 (0): 无
- 悬空门 (0): 无
- CLI 悬空门 (0): 无
- 本地悬空门 (0): 无
- 本地门 files/entry 不一致 (0): 无
- 悬空工程门 (0): 无
- 触发器漂移 (0): 无
- 未登记触发规格的工作流 (0): 无
- 触发规格未落地的工作流 (0): 无
- 能力锚点脱节 (0): 无
- prompt 段清单有、源里无 (0): 无
- prompt 段源里有、清单无 (0): 无
