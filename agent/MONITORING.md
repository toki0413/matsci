# Huginn Monitoring & Observability

> Metrics, alerts, and tracing configuration for production deployments.

---

## Metric Sources

### 1. Application Metrics (Prometheus)

Exposed at `GET /metrics` (requires admin API key).

```python
# Example scrape configuration
scrape_configs:
  - job_name: 'huginn'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    bearer_token: '<admin-api-key>'
```

### 2. Audit Logs (Structured JSONL)

Location: `./huginn_audit.jsonl` (configurable via `HUGINN_AUDIT_LOG_PATH`)

Each line is a JSON object with tamper-evident hash chain:

```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "event_type": "tool_call",
  "actor": "user",
  "action": "vasp_relax",
  "details": {"incar": "ENCUT=520..."},
  "input_hash": "a1b2c3...",
  "output_hash": "d4e5f6...",
  "prev_hash": "0x9f8e7d..."
}
```

Verify integrity:
```bash
python -c "from huginn.security import AuditLogger; \
  print(AuditLogger('huginn_audit.jsonl').verify_chain())"
```

### 3. Error Logs

Standard Python logging with structured JSON formatter:

```python
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    level=logging.INFO
)
```

---

## Key Metrics Reference

| Metric Name | Type | Labels | Description | Alert Threshold |
|-------------|------|--------|-------------|-----------------|
| `huginn_requests_total` | Counter | `method`, `path`, `status` | HTTP request count | > 1000/min |
| `huginn_request_duration_seconds` | Histogram | `path` | Request latency | p95 > 10s |
| `huginn_tool_calls_total` | Counter | `tool_name`, `status` | Tool invocations | N/A |
| `huginn_tool_duration_seconds` | Histogram | `tool_name` | Tool execution time | > 3600s |
| `huginn_sandbox_executions_total` | Counter | `executor_type` | Sandbox runs | N/A |
| `huginn_sandbox_failures_total` | Counter | `reason` | Sandbox failures | > 5/min |
| `huginn_active_websockets` | Gauge | — | WS connections | > 100 |
| `huginn_memory_entries_total` | Gauge | `tier` | Memory entries | N/A |
| `huginn_audit_events_total` | Counter | `event_type` | Audit events | N/A |
| `huginn_llm_tokens_total` | Counter | `model`, `provider` | LLM tokens consumed | > 1M/day |
| `huginn_llm_cost_usd_total` | Counter | `model` | Estimated LLM cost | > $100/day |

---

## Dashboards

### Grafana Dashboard (JSON Model)

```json
{
  "dashboard": {
    "title": "Huginn Agent Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(huginn_requests_total[5m])",
            "legendFormat": "{{method}} {{path}}"
          }
        ]
      },
      {
        "title": "Tool Call Duration",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, huginn_tool_duration_seconds)",
            "legendFormat": "{{tool_name}}"
          }
        ]
      },
      {
        "title": "Active WebSockets",
        "targets": [
          {
            "expr": "huginn_active_websockets",
            "legendFormat": "connections"
          }
        ]
      },
      {
        "title": "Sandbox Failures",
        "targets": [
          {
            "expr": "rate(huginn_sandbox_failures_total[5m])",
            "legendFormat": "{{reason}}"
          }
        ]
      }
    ]
  }
}
```

---

## Tracing (OpenTelemetry)

Optional distributed tracing integration:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)
trace.set_tracer_provider(tracer_provider)
```

Traced operations:
- HTTP request lifecycle
- Tool call execution (with input/output hashes)
- LLM invocation (with token count)
- Sandbox execution (with command and timeout)
- Workflow stage transitions
- Memory retrieval (with query and results)

---

## Log Aggregation

### Fluent Bit Configuration

```ini
[INPUT]
    Name tail
    Path /app/huginn_audit.jsonl
    Parser json
    Tag huginn.audit

[INPUT]
    Name tail
    Path /app/logs/huginn.log
    Parser python
    Tag huginn.app

[OUTPUT]
    Name loki
    Match huginn.*
    URL http://loki:3100
    Labels job=huginn
```

---

## Health Check Details

### `GET /health`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": "ok",
    "memory": "ok",
    "tools_registered": 42
  }
}
```

### `GET /health/rust`

```json
{
  "status": "healthy",
  "rust_extension_loaded": true
}
```

---

## Runbook: Common Incidents

### Incident: High Tool Failure Rate

1. Check `huginn_tool_calls_total{status="error"}` by tool name
2. Inspect tool logs in `huginn_audit.jsonl`
3. Verify simulation executables are in PATH
4. Check sandbox execution logs for policy violations

### Incident: Sandbox Timeout Storm

1. Check `huginn_sandbox_failures_total{reason="timeout"}`
2. Review `huginn_tool_duration_seconds` histogram for outliers
3. Verify HPC/cluster connectivity if using remote executor
4. Consider increasing `max_timeout` in SandboxConfig

### Incident: Memory Leak

1. Monitor `huginn_memory_entries_total` growth rate
2. Check if memory decay is enabled (`memory_decay_enabled=True`)
3. Review long-term memory DB size on disk
4. Restart agent with fresh memory if necessary

### Incident: API Key Leak

1. Rotate keys immediately: `HUGINN_API_KEY` and `HUGINN_ADMIN_API_KEY`
2. Review audit logs for unauthorized access patterns
3. Check if secrets were accidentally logged
4. Verify `redact_secrets` is active in all log paths

---

## Performance Baselines

From `tests/benchmark/test_performance.py`:

| Operation | Median Latency | OPS | Notes |
|-----------|----------------|-----|-------|
| API key comparison (HMAC) | ~800 ns | 1.2M | Constant-time |
| Tool registry lookup | ~1.7 μs | 550K | 42 tools |
| 50 serial tool calls | ~780 ms | 1.3 | 1 ms sleep per call |
| 50 parallel tool calls | ~15.3 ms | 65 | Concurrent asyncio |
| 100-case benchmark suite | ~9.1 ms | 107 | Mocked agent |
| Sandbox fast command | ~95 ms | 10.5 | Python --version |
| Audit chain verify (500 events) | ~10.9 ms | 91 | Hash chain verification |
| Audit log append (100 events) | ~430 ms | 2.3 | File I/O + fsync |
| Agent memory recall | ~2.1 ms | 457 | SQLite + FTS5 |
| Workflow engine init | ~5.9 μs | 100K | ToolRegistry binding |

---

## See Also

- [DEPLOYMENT.md](DEPLOYMENT.md) — Deployment procedures
- [SECURITY.md](SECURITY.md) — Security policies and key rotation
- [openapi.json](openapi.json) — Full API specification
