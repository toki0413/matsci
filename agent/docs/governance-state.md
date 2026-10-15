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

## 棘轮基线 (ratchet baselines)

| 基线 | 冻结值 |
|---|---|
| MECE 发现基线 (`tests/golden/mece_findings_baseline.txt`) | 47 项发现 |
| 镜像漂移预算 (`tests/test_publish_mirror_drift.py`) | missing/extra/differing = 0/0/0 |

## 缺口 (gaps)

- 孤儿门 (0): 无
- 悬空门 (0): 无
