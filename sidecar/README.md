# Huginn Sidecar (`sidecar/`)

Rust 进程管理器与事件总线，二进制名 **`huginn-sidecar`**。桌面应用（Tauri）通过它
统一管理 Python 后端的生命周期，并把后端 stdout/stderr 广播为 WebSocket 事件流。

## 职责

- **进程管理**：启动/停止 Python 后端（`python -m huginn.server`），
  用 JobObject（Windows）/ ProcessGroup（Unix）包裹，保证退出时回收子进程。
- **事件总线**：把后端 stdout / stderr / 状态事件经 `broadcast` 通道推给所有
  WebSocket 订阅者。
- **后端发现**：若后端已在外部运行则"收养"之，不重复启动。
- **运行时定位**：优先找打包的 Python 运行时（`resources/python-runtime/`），
  找不到则回退到 `PATH` 上的 `python`；从用户家目录启动，避免向只读安装目录写文件。

## HTTP/WS 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 存活文本 |
| `/health` | GET | 侧车 + 后端健康探针 |
| `/status` | GET | 后端是否运行/可达/健康 |
| `/start` | POST | 启动后端 |
| `/stop` | POST | 停止后端 |
| `/ws` | GET | WebSocket 事件流（stdout/stderr/status） |

## 端口

- 侧车默认监听 `127.0.0.1:8001`（`--port`）
- 后端默认 `8000`（`--backend-port`），由单一来源决定（见 ADR-0001 端口单源）

## 构建

```bash
cd sidecar
cargo build --release
# 产物: target/release/huginn-sidecar
```

## 模块

| 文件 | 职责 |
|---|---|
| `src/main.rs` | HTTP/WS 服务器 + 后端进程编排 + 事件广播 |

依赖：axum、tokio、clap、process-wrap、reqwest、serde、futures。