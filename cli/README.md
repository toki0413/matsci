# Huginn CLI (`cli/`)

Rust 前端 CLI。构建产物为二进制 **`huginn`**（与 Python 包的 console script
`huginn-agent` 是两回事，见下）。

## 与 Python CLI 的关系

| 产物 | 来源 | 二进制名 |
|---|---|---|
| Python CLI | `agent/pyproject.toml` `[project.scripts]` | `huginn-agent` |
| Rust CLI | 本目录 `cargo build --release` | `huginn` |

二者子命令一致；Rust CLI 是"壳"，负责解析全局参数并把子命令交给后端。

## 架构（ADR-0001 单网关）

遵循[单网关架构](../docs/architecture/decisions/0001-single-gateway.md)：

- **后端在跑** → `huginn` 作为 **HTTP/WS 客户端** 连接 `huginn.server`（`cli/src/http.rs`），
  走 `/chat`、`/explore`、`/coder`、`/bench/run`、`/evolve/run`、`/execute`、
  `/workflows/execute`、`/execution/diagnose`、`/hpc/*`、`/config/encrypt` 等端点。
- **后端未跑（离线兜底）** → 通过 `python -m huginn.cli` 子进程委托给 Python CLI
  （`cli/src/python.rs`）。该兜底路径被 `tests/test_arch_single_gateway.py` 冻结，
  只许缩、不许涨。

## 子命令

```
huginn chat             交互式聊天（SSE 流式）
huginn explore <目标>   设计空间系统搜索
huginn coder [任务]     自主编码（Codex 风格）
huginn serve            启动 HTTP/WS 后端（--port/--host）
huginn tools            列出工具（连后端拿元数据）
huginn configure        交互式首次配置向导（写 huginn.toml）
huginn bench            基准测试套件（--evolve 触发自进化）
huginn evolve           从执行日志运行自进化
huginn execute          执行 workflow 阶段
huginn workflow <模板>  运行 workflow 模板
huginn diagnose <错误>  诊断计算化学/MD 错误
huginn hpc test/submit/status   HPC 集群操作（SSH）
huginn encrypt-config   加密配置文件
```

全局参数：`--workspace`、`--config`、`--model`、`--provider`、`--dry-run`、
`--base-url`、`--ollama-url`。

## 构建

```bash
cd cli
cargo build --release
# 产物: target/release/huginn
```

## 模块

| 文件 | 职责 |
|---|---|
| `src/main.rs` | 参数解析、子命令分发、后端可用性判断与兜底委托 |
| `src/http.rs` | HTTP/WS 客户端（后端在跑时的通道） |
| `src/python.rs` | 子进程委托 `python -m huginn.cli`（离线兜底） |
| `src/config.rs` | 读取/写入 `huginn.toml`（配置向导用） |

依赖：clap、ureq、serde、toml、dotenvy、dialoguer、process-wrap。