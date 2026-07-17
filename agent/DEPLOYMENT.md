# Huginn Deployment Guide

> Production deployment checklist for the Huginn materials-science agent.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Server](#running-the-server)
6. [Reverse Proxy (nginx / Caddy)](#reverse-proxy)
7. [Docker Deployment](#docker-deployment)
8. [Security Hardening](#security-hardening)
9. [Health Checks & Monitoring](#health-checks--monitoring)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python    | 3.11+   | Runtime |
| uv        | latest  | Fast package manager (recommended) |
| Node.js   | 18+     | Frontend build (optional) |
| Tauri CLI | latest  | Desktop app build (optional) |

**Optional heavy dependencies** (simulation tools):

| Tool | Install | Notes |
|------|---------|-------|
| VASP | Proprietary | Set `VASP_EXECUTABLE` env var |
| LAMMPS | `conda install -c conda-forge lammps` | Set `LAMMPS_EXECUTABLE` |
| Quantum ESPRESSO | `conda install -c conda-forge qe` | Set `QE_EXECUTABLE` |
| CP2K | `conda install -c conda-forge cp2k` | Set `CP2K_EXECUTABLE` |
| OpenFOAM | System package | Set `OPENFOAM_DIR` |
| COMSOL | Proprietary | Set `COMSOL_EXECUTABLE` |
| ABAQUS | Proprietary | Set `ABAQUS_EXECUTABLE` |
| Lean 4 | `elan` toolchain | Set `LEAN_PATH` or use `lake` in PATH |

> **Note**: All simulation tools fall back to "export mode" when the executable is not found, generating input files for manual execution.

---

## Environment Setup

### Required Environment Variables

```bash
# API Authentication (production MUST set these)
export HUGINN_API_KEY="sk-..."          # Required for all non-public endpoints
export HUGINN_ADMIN_API_KEY="admin-..."  # Required for config/admin endpoints

# LLM Provider
export HUGINN_PROVIDER="openai"           # or "anthropic", "google", "ollama"
export HUGINN_MODEL="gpt-4o"            # or "claude-3-opus", "gemini-pro"
export HUGINN_API_KEY="sk-..."            # Provider API key

# Optional: Container runtime for sandbox isolation
export HUGINN_CONTAINER_RUNTIME="docker"  # or "podman", "apptainer"
export HUGINN_CONTAINER_IMAGE="huginn-sandbox:latest"

# Optional: Allow local bash fallback (NOT for production)
# export HUGINN_ALLOW_LOCAL_BASH=1
```

### Simulation Tool Paths

```bash
export VASP_EXECUTABLE="/opt/vasp/vasp_std"
export LAMMPS_EXECUTABLE="/usr/bin/lmp"
export QE_EXECUTABLE="/usr/bin/pw.x"
export CP2K_EXECUTABLE="/usr/bin/cp2k"
export OPENFOAM_DIR="/opt/openfoam11"
export COMSOL_EXECUTABLE="/opt/comsol/bin/comsol"
export ABAQUS_EXECUTABLE="/usr/bin/abaqus"
```

---

## Installation

### Method 1: uv (Recommended)

```bash
git clone https://github.com/huginn-agent/huginn.git
cd huginn/agent

# Install Python + dependencies
uv venv --python 3.11
uv pip install -e ".[all]"
```

### Method 2: pip

```bash
pip install -e ".[all]"
```

### Method 3: Docker

```bash
docker build -t huginn:latest -f Dockerfile .
```

---

## Configuration

### Config File (`huginn_config.json`)

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "api_key": "env:OPENAI_API_KEY",
  "temperature": 0.7,
  "max_tokens": 4096,
  "workspace": "./workspace",
  "memory": {
    "backend": "sqlite",
    "path": "./memory.db"
  }
}
```

> API keys prefixed with `env:` are resolved from environment variables at runtime.

---

## Running the Server

### Development Mode

```bash
# No auth required (HUGINN_API_KEY not set)
python -m huginn.server
# → http://localhost:8000
```

### Production Mode

```bash
export HUGINN_API_KEY="secure-random-key-64-chars"
export HUGINN_ADMIN_API_KEY="admin-secure-random-key"
python -m huginn.server --host 0.0.0.0 --port 8000
```

### Using uvicorn directly

```bash
uvicorn huginn.server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Reverse Proxy

### nginx

```nginx
server {
    listen 443 ssl http2;
    server_name huginn.example.com;

    ssl_certificate /etc/letsencrypt/live/huginn.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/huginn.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-HUGINN-API-KEY $http_x_huginn_api_key;
        proxy_read_timeout 86400;
    }
}
```

### Caddy

```caddy
huginn.example.com {
    reverse_proxy localhost:8000
}
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for simulation tools
RUN apt-get update && apt-get install -y \
    git build-essential libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8000

CMD ["uvicorn", "huginn.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  huginn:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HUGINN_API_KEY=${HUGINN_API_KEY}
      - HUGINN_ADMIN_API_KEY=${HUGINN_ADMIN_API_KEY}
      - HUGINN_MODEL=${HUGINN_MODEL}
      - HUGINN_PROVIDER=${HUGINN_PROVIDER}
    volumes:
      - ./workspace:/app/workspace
      - ./memory.db:/app/memory.db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Security Hardening

### 1. API Key Rotation

```bash
# Generate secure keys
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 2. Container Sandboxing (Recommended)

```bash
export HUGINN_CONTAINER_RUNTIME="docker"
export HUGINN_CONTAINER_IMAGE="huginn-sandbox:latest"
# Do NOT set HUGINN_ALLOW_LOCAL_BASH
```

### 3. File System Restrictions

```bash
# Strict working directory policy
export HUGINN_STRICT_WORK_DIR=1
export HUGINN_ALLOWED_WORK_DIRS="/app/workspace,/tmp/huginn"
```

### 4. Secret Management

```bash
# Use Vault backend (requires hvac package)
export HUGINN_SECRET_BACKEND="vault"
export HUGINN_VAULT_ADDR="https://vault.example.com:8200"
export HUGINN_VAULT_TOKEN="s.xxx"
```

---

## Health Checks & Monitoring

### Built-in Endpoints

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /health` | Public | Basic health check |
| `GET /health/rust` | Public | Rust extension health |
| `GET /metrics` | Public | Prometheus-compatible metrics (G38 校准: `_PUBLIC_PATHS` 在 `security/auth.py:55` 有意公开, 让 Prometheus 无鉴权抓取; 生产环境用反代加 IP 白名单或 Basic Auth 收紧) |
| `GET /openapi.json` | Public | OpenAPI schema |
| `GET /docs` | Public | Swagger UI |

### Prometheus Metrics

```bash
# 默认公开 (Prometheus 无鉴权抓取)
curl http://localhost:8000/metrics

# 生产环境收紧 (反代层加 Basic Auth 或 IP 白名单)
curl -u prom:secret http://localhost:8000/metrics
```

Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `huginn_requests_total` | Counter | Total HTTP requests |
| `huginn_request_duration_seconds` | Histogram | Request latency |
| `huginn_tool_calls_total` | Counter | Tool invocations by name |
| `huginn_tool_duration_seconds` | Histogram | Tool execution time |
| `huginn_sandbox_executions_total` | Counter | Sandbox executions |
| `huginn_sandbox_failures_total` | Counter | Sandbox failures |
| `huginn_active_websockets` | Gauge | Active WebSocket connections |
| `huginn_memory_entries_total` | Gauge | Memory DB entries |
| `huginn_audit_events_total` | Counter | Audit log events |

### Alerting Rules (Prometheus)

```yaml
- alert: HuginnHighErrorRate
  expr: rate(huginn_requests_total{status=~"5.."}[5m]) > 0.1
  for: 5m
  labels:
    severity: warning

- alert: HuginnSandboxFailure
  expr: rate(huginn_sandbox_failures_total[5m]) > 0.5
  for: 2m
  labels:
    severity: critical

- alert: HuginnHighLatency
  expr: histogram_quantile(0.95, huginn_request_duration_seconds) > 10
  for: 5m
  labels:
    severity: warning
```

---

## Troubleshooting

### "No execution backend available"

Set `HUGINN_ALLOW_LOCAL_BASH=1` for dev, or configure container runtime for production.

### "Container runtime not found in PATH"

Install Docker/Podman and ensure it's in the system PATH. Or use local sandbox with `HUGINN_ALLOW_LOCAL_BASH=1`.

### "API key invalid"

Check `HUGINN_API_KEY` matches the `X-HUGINN-API-KEY` header exactly. Use `secrets_match` for constant-time comparison.

### "Simulation tool not found"

The tool falls back to input-export mode. Set the executable environment variable or install the software.

### Lean 4 / lake not found

Install via elan: `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh`

---

## See Also

- [README.md](README.md) — Feature overview and quick start
- [SECURITY.md](SECURITY.md) — Security policy and vulnerability reporting
- [architecture.md](architecture.md) — System architecture
- [openapi.json](openapi.json) — Full API schema (generated from FastAPI)
