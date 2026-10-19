# Huginn Threat Model

**Version**: 1.0  
**Scope**: Core agent runtime, tool execution layer, Lean verification bridge, HPC integration, and sandbox endpoint.

---

## 1. Attack Surface Summary

| Surface | Trust Boundary | Risk Level | Mitigation Status |
|---------|---------------|------------|-------------------|
| LLM output → Tool calls | **Critical** | High | Sandboxed execution, dry-run mode |
| LLM output → Lean code gen | **Critical** | High | Lean type-checker rejects malformed code |
| `/sandbox/execute` endpoint | **Critical** | High | AST pre-scan + restricted import policy |
| HPC SSH client | **High** | High | `shlex.quote`, job-name sanitization, path validation |
| Config files (API keys) | **Medium** | Medium | `env:` / `keyring:` prefix support, masked `to_dict()` |
| RAG document store | **Medium** | Medium | AES-128-CBC + HMAC at rest |
| Lean package supply chain | **Medium** | Medium | `lake-manifest.json` pinning, commit verification |
| MCP server subprocesses | **Medium** | Low | Standard OS process isolation |

---

## 2. STRIDE Analysis

### Spoofing (S)
- **Threat**: Attacker tricks the agent into using a fake LLM endpoint (DNS hijacking of `base_url`).
- **Mitigation**: Use HTTPS for all remote providers; local endpoints restricted to `127.*`/`localhost`.
- **Residual**: Certificate pinning not implemented.

### Tampering (T)
- **Threat**: Lean `.olean` cache files or `lake-packages/` are modified on disk to inject malicious tactics.
- **Mitigation**: Commit `lake-manifest.json`; verify upstream Git tags; review diffs on updates.
- **Residual**: No cryptographic checksum verification of `.olean` files.

- **Threat**: Audit log `huginn_audit.jsonl` is modified to hide malicious activity.
- **Mitigation**: Hash chain (`prev_hash`) makes tamper-evident; append-only file mode.
- **Residual**: Log file itself is not encrypted or signed.

### Repudiation (R)
- **Threat**: User denies submitting a destructive HPC job.
- **Mitigation**: Audit logger records every tool call, LLM invocation, and subprocess execution with SHA-256 content hashes.
- **Residual**: No digital signatures on audit events.

### Information Disclosure (I)
- **Threat**: API keys leaked via config files or process listings.
- **Mitigation**: `env:` and `keyring:` prefixes avoid plaintext storage; `to_dict(mask_key=True)` redacts keys in logs.
- **Residual**: Memory dumps could still expose keys while agent is running.

- **Threat**: VASP/LAMMPS output files contain sensitive material structures.
- **Mitigation**: EncryptedVectorStore for RAG documents; workspace permissions are user's responsibility.
- **Residual**: Raw simulation output is not encrypted by default.

### Denial of Service (DoS)
- **Threat**: LLM generates an infinite loop or fork bomb via tool call.
- **Mitigation**: Sandbox enforces `max_timeout` (default 1h, clamped) and `max_output_bytes` (50 MiB).
- **Residual**: HPC jobs submitted via SSH run outside the agent's timeout scope.

### Elevation of Privilege (E)
- **Threat**: Prompt injection causes the agent to execute arbitrary shell commands.
- **Mitigation**:
  - `shell=True` is **absolutely forbidden** in `SandboxExecutor`.
  - Only whitelisted executables may run.
  - Working directory can be restricted to allowed roots.
  - HPC commands use `shlex.join` to prevent injection.
- **Residual**: If the whitelist itself is too broad (e.g., `python` allows arbitrary Python code), the `/sandbox/execute` endpoint uses an additional AST restriction layer.

---

## 3. Trust Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│  Untrusted Zone                                             │
│  (LLM provider, user input, external MCP servers)           │
└──────────────────────────┬──────────────────────────────────┘
                           │ Network / Subprocess
┌──────────────────────────▼──────────────────────────────────┐
│  Agent Runtime (Semi-Trusted)                               │
│  - HuginnAgent                                              │
│  - ToolRegistry                                             │
│  - Skills framework                                         │
│  - Memory manager                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ Sandbox boundary
┌──────────────────────────▼──────────────────────────────────┐
│  Restricted Execution Zone (Sandboxed)                      │
│  - SandboxExecutor (whitelist, timeout, output limits)      │
│  - LeanInterface (lake build, lean verification)            │
│  - VASP / LAMMPS subprocesses                               │
└──────────────────────────┬──────────────────────────────────┘
                           │ SSH / Network
┌──────────────────────────▼──────────────────────────────────┐
│  External HPC Zone (Trusted infrastructure)                 │
│  - SLURM / PBS scheduler                                    │
│  - Remote compute nodes                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Incident Response Playbook

### 4.1 Suspicious Tool Call Detected
1. Enable `--dry-run` mode: `huginn-agent chat --dry-run`
2. Review `huginn_audit.jsonl` for the chain of events leading to the call.
3. Check the hash chain integrity: `audit.verify_chain()`
4. If the agent is running, disconnect from MCP servers and revoke API keys.

### 4.2 HPC Job Injection Suspected
1. Immediately `scancel` / `qdel` the suspicious job ID.
2. Review remote work directory for unexpected scripts.
3. Rotate SSH keys and review HPC access logs.

### 4.3 Lean Package Compromise
1. Delete `lake-packages/` and rebuild from committed `lake-manifest.json`.
2. Verify upstream Git commits against known-good hashes.
3. Audit all recent proof additions for suspicious tactics (e.g., `sorry`, `axiom`, custom tactics).

---

## 5. Hardening Roadmap

| Phase | Item | Priority |
|-------|------|----------|
| Phase 1 ✅ | Command sandbox, audit logs, safe eval, key resolution | Done |
| Phase 2 ✅ | HPC command injection fix, AST-based sandbox restriction, supply-chain docs | Done |
| Phase 3 | RBAC / multi-user isolation | Medium |
| Phase 3 | Secret manager integration (HashiCorp Vault, AWS Secrets Manager) | Medium |
| Phase 3 | Containerized tool execution (Docker / gVisor) | Low |
| Phase 3 | Encrypted audit logs with HSM-backed signing | Low |
| Phase 3 | Formal verification of sandbox policy in Lean | Research |

---

## 6. Red / Blue Coverage Matrix

> 概念框架借鉴 *Awesome Cyber Security University* 的分级红/蓝双轨路径:
> **红队 = 进攻探针**(主动尝试突破边界, 验证"攻击被阻止"),
> **蓝队 = 防守校验**(断言控制按预期生效, 验证"控制不静默失效").
> 二者必须成对 —— 只有单边覆盖的攻击面是治理盲区。
>
> 下表把 §1 的每个攻击面按两轴归位到**现有测试资产**(文件级), 并显式标出缺口。
> 这是诊断视图, 不重复行为验证; 归类依据各测试模块的定位。

| 攻击面 (§1) | 层级 | 红队探针 (进攻) | 蓝队校验 (防守) | 缺口 |
|---|---|---|---|---|
| LLM output → Tool calls | Critical | `test_prompt_security.py`, `test_math_prompt_injection.py` | `test_tool_base_permissions.py`, `test_permissions_granularity.py` | — |
| LLM output → Lean code gen | Critical | `tests/security/pentest_lean_supply_chain.py` (不健全证明标记) | `test_lean.py`, `test_lean_e2e.py`, `test_lean_pipeline.py` | — |
| `/sandbox/execute` endpoint | Critical | `tests/security/fuzz_api.py`, `tests/security/pentest_api_security.py`, `test_security_penetration.py` | `test_code_act_sandbox.py`, `test_container_executor.py`, `test_bash_rust_sandbox.py` | — |
| HPC SSH client | High | `tests/security/pentest_hpc_injection.py` | `test_hpc_privilege_patterns.py`, `test_hpc_host_keys.py`, `test_hpc_job_management.py`, `test_hpc_client.py` | — |
| Config files (API keys) | Medium | `tests/security/pentest_config_secrets.py` | `test_credential_store.py`, `test_credential_bridge.py`, `test_config_encryption.py`, `test_privacy.py` | — |
| RAG document store | Medium | `tests/security/pentest_api_security.py` (SSRF/ingest), `tests/security/fuzz_api.py` | `test_vector_store_crud.py`, `test_vector_store_search.py`, `test_crypto.py` | — |
| Lean package supply chain | Medium | `tests/security/pentest_lean_supply_chain.py` | `tests/security/pentest_lean_supply_chain.py` (真实项目完整性回归) | — |
| MCP server subprocesses | Medium | `tests/security/pentest_mcp_subprocess.py` | `test_mcp_command_validation.py`, `test_mcp_resilience.py`, `test_arch_mcp_governance.py` | — |

**读法**: 中间两列是文件级归位; "缺口"列即后续补测方向, 优先级按 §1 的风险等级。
本轮把 HPC / Config / MCP 三个红队缺口与 Lean 供应链的红/蓝双缺口一并补齐, 表内
8 个攻击面现全部双轨覆盖 (dual)。新增资产是 `tests/security/pentest_*` 探针族:

- `pentest_hpc_injection.py` — HPC 五个注入入口 (job-name / path / command 等);
- `pentest_config_secrets.py` — 配置密钥泄漏面 (明文落盘 / 日志 / 进程列表);
- `pentest_mcp_subprocess.py` — MCP 子进程命令白名单 + 配置脱敏;
- `pentest_lean_supply_chain.py` — Lean 不健全证明标记 + 未固定依赖 (红/蓝同源:
  同文件既做进攻探针, 又对真实项目做完整性回归)。

这张表由 `scripts/coverage_matrix_audit.py` 只读扫描, 断链引用与覆盖回退在 CI 里
直接变红 (见 `.github/workflows/coverage-matrix.yml`)。补测原则不变: 给缺口面加
**进攻探针**, 而非只堆防守断言。

### 6.1 评审层对偶 (Red / Blue Reviewer)

§6 主表按 §1 攻击面归位, 而"红/蓝双轨"同样落在 **autoloop 阶段门** 的评审层:

- 红队 `RedTeamReviewer` (`huginn/autoloop/red_team.py`) — 对抗性反驳 (方法学 / 混杂 /
  可证伪性);
- 蓝队 `BlueTeamReviewer` (`huginn/autoloop/blue_team.py`) — 断言被声明的控制真正
  在位且不静默失效 (来源信任 / 冗余控制 / 守卫完整性)。

两者独立注入 `PhaseGateHook` 的 `reviewer_fn` 与 `blue_reviewer_fn`, 任一拒绝即
`rejected`; `tests/test_red_team.py` 与 `tests/test_blue_team.py` 成对锁定两侧行为。
