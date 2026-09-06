# Contributing to Huginn

Thank you for your interest in contributing to Huginn! This is a monorepo with
several components. The core Python package lives under `agent/` and holds the
authoritative contribution guide.

## Start here

- **Core package (Python, `huginn-agent`)** — see
  [`agent/CONTRIBUTING.md`](agent/CONTRIBUTING.md) for development setup,
  testing, and pull-request process. This is where most contributions land.
- **Documentation navigation** — [`agent/docs/INDEX.md`](agent/docs/INDEX.md)
  catalogs every document with its status; read it before editing docs.
- **MCP servers** (`servers/`) — standalone packaging; see
  [`servers/README.md`](servers/README.md).

## Repo layout

```
├── agent/     # Python core package (huginn-agent) + tests + docs
├── cli/       # Rust CLI frontend (HTTP/WS client)
├── desktop/   # Tauri v2 + React 18 desktop app
├── pyext/     # Rust performance extensions (huginn-ext)
├── servers/   # MCP servers (mat-db / math-anything / vision-pixel)
├── docs/      # root-level docs (ADR, threat model, quickstart)
└── huginn/    # shared skill definitions
```

## Reporting bugs / requesting features

Open an issue against `toki0413/matsci`. Include the `huginn-agent` version
(`huginn-agent version`) and a minimal reproducer where possible.

## License

Contributions are accepted under the MIT License; by submitting a PR you agree
to license your changes under the same terms. See `LICENSE`.