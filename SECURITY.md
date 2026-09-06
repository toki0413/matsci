# Security Policy

Huginn's security model, defaults, and known limitations are documented in
[`agent/SECURITY.md`](agent/SECURITY.md) and the root-level
[`docs/threat_model.md`](docs/threat_model.md). This file is the disclosure
entry point for the monorepo.

## Supported Versions

The core Python package (`huginn-agent`) is published on PyPI. Security fixes
land on the latest release and are backported to the prior version where
feasible.

| Version | Supported |
|---------|-----------|
| 1.3.x   | ✅ (latest) |
| 1.2.x   | ⚠️ 仅安全补丁 |
| < 1.2   | ❌ 不再支持 |

## Reporting a Vulnerability

**Do not open a public issue for a security bug.** Report privately by opening a
[draft security advisory](https://github.com/toki0413/matsci/security/advisories/new)
on GitHub.

Please include:

- Affected component(s) and version(s)
- A minimal reproduction / proof of concept
- Impact description
- Any suggested fix (optional)

We aim to acknowledge reports within **48 hours** and release a fix for critical
issues within **7 days**.

## Security Defaults (summary)

- API authentication required in production (`HUGINN_API_KEY`, `HUGINN_ADMIN_API_KEY`).
- Destructive tools run inside a container sandbox by default; command allowlists,
  timeouts, and output limits enforced.
- Secrets masked in logs and configuration dumps; operator-level keys isolated from
  user-service credentials.
- Audit logs are append-only and hash-chained.

## MCP Servers & Remote / HPC

- MCP servers (`servers/`) are standalone; review their `README.md` before exposing
  them beyond loopback.
- Remote / HPC execution follows the deployment and local↔remote boundary rules in
  `agent/DEPLOYMENT.md`.