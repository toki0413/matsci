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
1. Enable `--dry-run` mode: `huginn chat --dry-run`
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
