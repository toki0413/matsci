# 低熵治理台账 (governance state)

自动生成: `python -m huginn.cli.governance_audit --out docs/governance-state.md`.
登记 `tests/test_arch_*.py` 门族、`.github/workflows/ci.yml` 的 python guard step (pytest 与 `python -m huginn.<module>`) 与 `.pre-commit-config.yaml` 本地门的对齐面: **孤儿门** (存在却未登记)、**悬空门** (指向不存在的测试/模块) 在此显形。只提示不判死; 硬判定在 `tests/test_arch_gate_registry.py`.

## 门 (gates)

| 测试 | arch | 状态 | CI step |
|---|---|---|---|
| `tests/security/fuzz_api.py` | 否 | `fast-fail-guard` | Run API fuzz tests |
| `tests/security/pentest_api_security.py` | 否 | `fast-fail-guard` | Run API security PoC tests |
| `tests/security/pentest_archive_safety.py` | 否 | `fast-fail-guard` | Run archive security PoC tests |
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
| `tests/test_naturebench.py` | 否 | `fast-fail-guard` | NatureBench repro gate |
| `tests/test_testclient_hygiene.py` | 否 | `fast-fail-guard` | TestClient hygiene guard |

## CLI 门 (cli gates)

`ci.yml` 里 `python -m huginn.<module>` 形态的治理门 (CLI 自查 / 棘轮, 非 pytest). 旧解析只认 pytest 会漏掉; 现单列。**存在=否** 即悬空 CLI 门 (指向已删/改名的模块).

| 模块 | CI step | 存在 |
|---|---|---|
| `huginn.cli.contract_audit` | MECE wiring-contract guard (ratchet) | 是 |
| `huginn.security.policy_engine` | Policy engine self-check | 是 |

## 本地门 (pre-commit local gates)

`.pre-commit-config.yaml` 的本地门 (如 G37 audit-regression). 既不在 `ci.yml` 也不在别处登记, 故单列。**存在=否** 即悬空本地门 (引用的测试已改名/删除 —— 此时 hook 的 `files:` 过滤器会静默不再匹配, 门悄悄死掉). **files 覆盖=否** 即 `entry` 引用的测试不在 `files:` 过滤范围内 —— 改它不会触发 hook, 门静默失效.

| 本地门 (hook id) | 引用测试 | 存在 | files 覆盖 |
|---|---|---|---|
| `audit-regression-7fixes` | `tests/test_context_and_jwt.py` | 是 | 是 |
| `audit-regression-7fixes` | `tests/test_eos_fit.py` | 是 | 是 |
| `audit-regression-7fixes` | `tests/test_fts5_consistency.py` | 是 | 是 |
| `audit-regression-7fixes` | `tests/test_mechanics.py` | 是 | 是 |
| `audit-regression-7fixes` | `tests/test_sim_audit_regression.py` | 是 | 是 |

## 预算 (budgets)

治理棘轮的冻结预算/基线单一台账 (源: `governance_audit.BUDGET_SPECS`). **权威** = 值在台账声明、消费者门 `budget_value()` 只读; **引用** = 值仍在其自然归属, 台账只登记。实测/余量仅对 weightless 可测项给出; 其余由各自消费者门校验 (current ≤ frozen); 重型探针 (如镜像漂移实测) 需 `--probe`。**消费者门**列: `ci_step` 型是 `ci.yml` 真实 step (有存在性门); 带 `· 全量` 标记的是 `full_suite` 型 (仅全量套件跑, 不在 `ci.yml`)。

| 预算 | 位置 | 模式 | 冻结值 | 实测 | 余量 | 消费者门 |
|---|---|---|---|---|---|---|
| MECE 发现基线 | `tests/golden/mece_findings_baseline.txt` | 引用 | 47 项 | — | — | MECE wiring-contract guard (ratchet) |
| 镜像漂移预算 | `tests/test_publish_mirror_drift.py` | 权威 | 0/0/0 | — | — | publish mirror drift (only-shrink) · 全量 |
| 环境裸读基线 | `huginn/env_schema.py` | 引用 | 467 | — | — | env surface governance · 全量 |
| 外部直连白名单 | `tests/test_arch_single_gateway.py` | 引用 | 19 项 | — | — | Architecture single-gateway guard |
| 冻结委托子命令 | `tests/test_arch_single_gateway.py` | 引用 | 11 项 | — | — | Architecture single-gateway guard |
| 插件准入清单 | `tests/test_arch_plugin_governance.py` | 引用 | 2 项 | — | — | Plugin admission guard |
| 硬编码路径白名单 | `tests/test_arch_no_hardcoded_paths.py` | 引用 | 7 项 | — | — | No-hardcoded-machine-paths guard |
| MCP 接线白名单 | `tests/test_arch_mcp_governance.py` | 引用 | 7 项 | — | — | MCP single-gateway guard |
| arch 门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 17 | 17 | 0 | Governance gate registry guard |
| CLI 门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 2 | 2 | 0 | Governance gate registry guard |
| 本地门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 5 | 5 | 0 | Governance gate registry guard |
| 字段审计 golden | `tests/golden/field_audit` | 引用 | 5 文件 | — | — | MECE 字段级面 · 全量 |

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
| 循环闸位数下限 | `tests/test_arch_gate_registry.py` | 权威 | 9 | 9 | 0 | Governance gate registry guard |

## 缺口 (gaps)

- 孤儿门 (0): 无
- 悬空门 (0): 无
- CLI 悬空门 (0): 无
- 本地悬空门 (0): 无
- 本地门 files/entry 不一致 (0): 无
