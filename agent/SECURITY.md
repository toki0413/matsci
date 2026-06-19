# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in Huginn, please report it privately
to the maintainers by opening a draft security advisory on GitHub or by emailing
security@huginn-agent.local.

Please do **not** open public issues for security bugs.

We aim to acknowledge reports within 48 hours and release a fix within 7 days for
critical issues.

## Security Defaults

- API authentication is required in production (`HUGINN_API_KEY`).
- Destructive tools run inside a container sandbox by default.
- Secrets are masked in logs and configuration dumps.
- Audit logs are append-only and hash-chained.

## Known Limitations

- Local fallback for `bash_tool`/`code_tool` is disabled by default; enabling it
  reduces isolation guarantees.
- The in-memory telemetry backend does not persist across restarts; use the
  OpenTelemetry exporter for production.
