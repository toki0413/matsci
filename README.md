# Huginn

**Huginn**（包名 `huginn-agent`，v1.3.0，MIT）是一个面向**计算材料科学**的
LLM 驱动智能 Agent 系统，内置**形式化验证（Lean 4）**数学能力。它自动执行
DFT 计算、分子动力学、CFD/FEA、符号回归、文献检索与"自主探索"材料设计空间，
并用 Lean 4 对张量代数、有限元方法、数值线性代数、DFT 理论、热力学与概率同时
进行形式化证明。

---

## 核心特性 Highlights

- **6 阶段形式化数学**：张量代数 → FEM 弱形式 → 数值线性代数 → DFT → 热力学 →
  概率，全程 Lean 4 形式化证明；SymPy 表达式自动翻译为 Lean `Float` 定义并类型检查
- **多 Provider LLM**：OpenAI、Anthropic、DeepSeek、Google GenAI、OpenRouter、
  NVIDIA、Ollama、vLLM、LM Studio，以及任意 OpenAI 兼容本地端点
- **两条正交控制轴**（本项目的核心心智模型，见下）：
  - **极简模式（ModelTier）**：控制"认知编排开销"——`full / balanced / minimal`
  - **思考强度（ThinkingIntensity）**：控制"模型推理深度"——`low / medium / high / max`
- **单网关架构**：所有业务逻辑统一经 `huginn.server` 的 HTTP/WS API 消费，
  外部消费者（CLI / 桌面 / 脚本）一律作为 API 客户端（见 [ADR-0001](docs/architecture/decisions/0001-single-gateway.md)）
- **契约文档自动生成**：`config_audit` 工具扫描代码生成 env / flags / tools /
  路由 / 错误 / 权限等契约文档，防止文档与代码漂移
- **安全沙箱**：容器化隔离、命令白名单、超时与输出上限、fail-closed 工具元数据
- **可复现依赖**：`uv pip compile` 生成 `requirements.lock`，CI 门禁拦截漂移
- **密钥分层**：区分用户服务凭据与 operator 级密钥，前端可配置与仅 env 配置分离

---

## 两条正交控制轴

理解 Huginn 的关键：**极简模式**与**思考强度**是两个*独立*维度，互不干扰。

| 维度 | 控制什么 | 取值 | 机制 |
|---|---|---|---|
| **极简模式** <br>`HUGINN_MODEL_TIER` | 认知编排开销（phase 机 / plan 门控 / 认知纪律 / compaction / 外部思考） | `full` / `balanced` / `minimal` | 越强模型越可跳过编排步骤 |
| **思考强度** <br>`HUGINN_THINKING` | 模型推理深度（provider reasoning budget） | `low` / `medium` / `high` / `max` | 映射到各 provider 推理预算 |

| 极简模式档位 | phase 机 | plan 门控 | 认知纪律 | compaction | 外部思考 | 适用 |
|---|---|---|---|---|---|---|
| `full` | ✅ | ✅ | `always` | `heavy` | ✅ | 本地弱模型，保留全部认知编排 |
| `balanced` | ✅ | ✅ | `event` | `medium` | ✅ | 中等模型，纪律降级为事件驱动 |
| `minimal` | ✗ | ✗ | `event` | `light` | ✗ | 顶尖大模型，跳过 phase/plan 门控 |

> 安全层（命令校验 / 物理 sanity check / 资源预算）在所有档位**始终保留**。
> 完整契约见 [model-tier-contract.md](agent/docs/model-tier-contract.md)。

示例：

```bash
# 本地弱模型：保留全部认知编排
HUGINN_MODEL_TIER=full huginn-agent chat

# 顶尖大模型：跳过编排，同时把推理预算拉满
HUGINN_MODEL_TIER=minimal HUGINN_THINKING=max huginn-agent chat
```

---

## 架构概览

Huginn 遵循**单网关（Single Gateway）**架构：`huginn.server` 是唯一业务网关，
所有业务逻辑只有通过其后端 HTTP/WS API 才能被消费。CLI、桌面应用、脚本一律作为
API 客户端，不直接 `import huginn.*` 业务模块（由 `tests/test_arch_single_gateway.py`
在 CI 强制）。

```
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│ Rust CLI    │   │  Desktop App │   │ MCP / 脚本/外部 │
│ (API 客户端)│   │  (Tauri 壳)  │   │ (API 客户端)    │
└──────┬──────┘   └──────┬───────┘   └────────┬────────┘
       │  HTTP/WS        │  HTTP/WS           │  HTTP/WS
       └─────────────────┼────────────────────┘
                         ▼
              ┌──────────────────────┐
              │   huginn.server      │  ← 唯一业务网关 (FastAPI + WS/SSE)
              │  routes/**  /v1/*    │     统一鉴权 / 审计 / 错误信封
              └──────────┬───────────┘
                         ▼
        ┌────────────────────────────────────────┐
        │            Agent 层 / 能力层            │
        │  agent/ agents/ tools/ skills/ memory/  │
        │  evolution/ knowledge/ kg/ causal/      │
        │  autoloop/ metacog/ runtime/ security/  │
        │  + Lean 4 形式化验证 (HuginnLean)        │
        └────────────────────────────────────────┘
```

真实入口点：
- Python CLI（console script）：`huginn-agent` → `huginn.cli:main`
- API 服务器：`python -m huginn.server` → FastAPI + WebSocket
- Rust CLI 前端（`cli/`）：构建产物 `huginn`，作为 HTTP/WS 客户端连接后端
- 桌面应用（`desktop/`）：Tauri v2 + React 18

详见 [architecture.md](agent/docs/architecture.md) 与
[tech-spec.md](agent/docs/tech-spec.md)。

---

## 快速上手 Quick Start

### 1. 安装 Python 后端（必需）

```bash
cd agent
# 推荐: uv
uv venv --python 3.11
uv pip install -e ".[all]"
# 或 pip
pip install -e ".[all]"
```

可选的 Rust 扩展（LAMMPS/VASP 解析、MSD/RDF 加速）见
[pyext](pyext) 与 [DEPLOYMENT.md](agent/DEPLOYMENT.md)。

### 2. 配置 LLM Provider

```bash
# 方式 A — 环境变量
export HUGINN_PROVIDER=openai
export HUGINN_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

# 方式 B — 配置文件
huginn-agent configure            # 交互向导
huginn-agent chat --config huginn.toml

# 方式 C — 命令行参数
huginn-agent chat --provider ollama --ollama-url http://localhost:11434
```

本地端点（vLLM / LM Studio / Ollama）不需要真实 API key，`--base-url` 指向
`localhost` / `127.*` 时自动发送 dummy key。

### 3. 启动后端并聊天

```bash
python -m huginn.server            # 启动 API 网关 (http://localhost:8000)
huginn-agent chat                  # 作为 API 客户端连接
```

也可以直接运行（等价于子命令纯客户端模式）：

```bash
huginn-agent chat "计算 Si 的带隙"
huginn-agent coder "给 code_tool.py 加 docstring"
```

### 4. 运行测试

```bash
cd agent
pytest tests/ -x -q               # 410 个测试文件
```

---

## 配置

运行时配置为 TOML（`huginn.toml`）或 JSON，加载优先级：
**CLI 参数 > 配置文件 > 环境变量**。完整示例见
[huginn.toml.example](huginn.toml.example) 与
[env-contract.md](agent/docs/env-contract.md)（265 个 `HUGINN_*` 环境变量契约）。

```toml
provider = "openai"
model = "gpt-4o"
api_key = "sk-..."                # 或 HUGINN_API_KEY 环境变量
workspace = "env:HUGINN_WORKSPACE"
auto_approve = false

# 两条控制轴
# model_tier  = "balanced"          # full / balanced / minimal
# thinking   = "medium"             # low / medium / high / max
```

密钥管理：用户服务凭据可前端配置（`/credentials`）；operator 级密钥仅通过
环境变量 / 密钥管理器注入，二者隔离（见 [SECURITY.md](agent/SECURITY.md)）。

---

## 文档索引 Documentation

Huginn 的文档按类别登记在 **[agent/docs/INDEX.md](agent/docs/INDEX.md)**，
统一标注状态（active / staging / report），是文档的**导航入口**。核心文档：

| 主题 | 位置 |
|---|---|
| 项目总览 / 快速上手 | 本文件 |
| 文档导航索引 | [agent/docs/INDEX.md](agent/docs/INDEX.md) |
| 系统架构 | [agent/docs/architecture.md](agent/docs/architecture.md) |
| 技术规格（现状事实） | [agent/docs/tech-spec.md](agent/docs/tech-spec.md) |
| 架构决策记录 | [docs/architecture/decisions/](docs/architecture/decisions/) |
| 快速上手分步指南 | [docs/quickstart.md](docs/quickstart.md) |
| 威胁模型 | [docs/threat_model.md](docs/threat_model.md) |
| 部署指南 | [agent/DEPLOYMENT.md](agent/DEPLOYMENT.md) |
| 监控 / 可观测性 | [agent/MONITORING.md](agent/MONITORING.md) |
| 安全策略 | [agent/SECURITY.md](agent/SECURITY.md) |
| 贡献指南 | [agent/CONTRIBUTING.md](agent/CONTRIBUTING.md) |
| 路线图 | [agent/ROADMAP.md](agent/ROADMAP.md) |
| 契约文档（自动生成） | [agent/docs/env-contract.md](agent/docs/env-contract.md) 等 |

---

## 开发 Development

### 项目结构

```
huginn/
├── agent/            # Python 核心包 (huginn-agent)
│   ├── huginn/       #   业务模块 (agent/ tools/ memory/ routes/ security/ ...)
│   ├── tests/        #   410 个测试文件
│   ├── docs/         #   架构 + 契约文档 + INDEX.md 导航
│   ├── README.md     #   Agent 包 README
│   ├── DEPLOYMENT.md / MONITORING.md / SECURITY.md / CONTRIBUTING.md / ROADMAP.md
├── cli/              # Rust CLI 前端 (HTTP/WS 客户端)
├── desktop/          # Tauri v2 + React 18 桌面应用
├── pyext/            # Rust 性能扩展 (huginn-ext)
├── sidecar/          # Rust sidecar (进程管理)
├── servers/          # MCP 服务器 (mat-db / math-anything / vision-pixel)
├── docs/             # 根级文档 (架构决策 / 威胁模型 / 快速上手)
└── skills/           # 共享技能定义
```

### 新增一个工具

1. 在 `agent/huginn/tools/` 新建继承 `HuginnTool` 的类
2. 定义 `name` / `description` / `input_schema`（Pydantic）
3. 实现 `_execute()`，`call()` 返回 `ToolResult`
4. 在 `agent/huginn/tools/__init__.py` 的 `_CORE_MODULES` / `_OPTIONAL_MODULES` 登记
5. 在 `agent/tests/` 加测试

### 质量守则

```bash
cd agent
ruff check huginn tests
black --check huginn tests
mypy huginn
pytest tests/ -q
```

所有 CI 检查通过后才能合并。改动架构时记得同步 `docs/tech-spec.md` 与
`docs/architecture.md`，并在 [INDEX.md](agent/docs/INDEX.md) 登记 / 更新文档。

---

## 安全 Security

- 生产环境必须设置 `HUGINN_API_KEY` / `HUGINN_ADMIN_API_KEY`（强随机密钥）
- 破坏性工具默认在容器沙箱内执行；命令白名单 + 超时 + 输出上限
- 日志与配置转储默认屏蔽密钥；审计日志 append-only 且 hash-chained
- 密钥分层：用户服务凭据 vs operator 级密钥

详见 [SECURITY.md](agent/SECURITY.md) 与 [docs/threat_model.md](docs/threat_model.md)。

---

## License

MIT License