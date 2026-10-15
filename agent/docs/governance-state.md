# 低熵治理台账 (governance state)

自动生成: `python -m huginn.cli.governance_audit --out docs/governance-state.md`.
登记 `tests/test_arch_*.py` 门族与 `.github/workflows/ci.yml` 的 python guard step 的对齐面: **孤儿门** (存在却未登记) 与 **悬空门** (CI step 指向不存在的测试) 在此显形。只提示不判死; 硬判定在 `tests/test_arch_gate_registry.py`.

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

## 预算 (budgets)

治理棘轮的冻结预算/基线单一台账 (源: `governance_audit.BUDGET_SPECS`). **权威** = 值在台账声明、消费者门 `budget_value()` 只读; **引用** = 值仍在其自然归属, 台账只登记。实测/余量仅对 weightless 可测项给出; 其余由各自消费者门校验 (current ≤ frozen); 重型探针 (如镜像漂移实测) 需 `--probe`。

| 预算 | 位置 | 模式 | 冻结值 | 实测 | 余量 | 消费者门 |
|---|---|---|---|---|---|---|
| MECE 发现基线 | `tests/golden/mece_findings_baseline.txt` | 引用 | 47 项 | — | — | MECE wiring-contract guard (ratchet) |
| 镜像漂移预算 | `tests/test_publish_mirror_drift.py` | 权威 | 0/0/0 | — | — | publish mirror drift (only-shrink) |
| 环境裸读基线 | `huginn/env_schema.py` | 引用 | 467 | — | — | env surface governance |
| 外部直连白名单 | `tests/test_arch_single_gateway.py` | 引用 | 19 项 | — | — | Architecture single-gateway guard |
| 冻结委托子命令 | `tests/test_arch_single_gateway.py` | 引用 | 11 项 | — | — | Architecture single-gateway guard |
| 插件准入清单 | `tests/test_arch_plugin_governance.py` | 引用 | 2 项 | — | — | Plugin admission guard |
| 硬编码路径白名单 | `tests/test_arch_no_hardcoded_paths.py` | 引用 | 7 项 | — | — | No-hardcoded-machine-paths guard |
| MCP 接线白名单 | `tests/test_arch_mcp_governance.py` | 引用 | 7 项 | — | — | MCP single-gateway guard |
| arch 门数下限 | `tests/test_arch_gate_registry.py` | 权威 | 17 | 17 | 0 | Governance gate registry guard |
| 字段审计 golden | `tests/golden/field_audit` | 引用 | 5 文件 | — | — | MECE 字段级面 |

## 缺口 (gaps)

- 孤儿门 (0): 无
- 悬空门 (0): 无
