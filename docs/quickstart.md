# Huginn Quick Start

本指南带你本地运行 Huginn——从安装到完成第一个"经形式化验证的 FEM 弱形式推导"。

> 这是根级分步快速上手。完整文档导航见 [agent/docs/INDEX.md](../agent/docs/INDEX.md)，
> 部署细节见 [agent/DEPLOYMENT.md](../agent/DEPLOYMENT.md)。

---

## 1. 安装 Installation

```bash
# 进入 agent 包
cd agent

# 推荐: uv
uv venv --python 3.11
uv pip install -e ".[all]"

# 或 pip
pip install -e ".[all]"

# Lean 4 (形式化验证所需)
# 通过 elan 安装: https://github.com/leanprover/elan
lake --version
```

---

## 2. 配置 LLM Configure

Huginn 支持多 provider 与本地端点。完全离线方案：

### 方式 A: Ollama（推荐新手）

```bash
ollama pull qwen2.5:14b

huginn-agent configure
# Provider: ollama
# Model: qwen2.5:14b
# Ollama host: http://localhost:11434
```

### 方式 B: vLLM / LM Studio

```bash
huginn-agent chat --provider vllm \
  --base-url http://localhost:8000/v1 \
  --model llama-3.1-8b
```

本地端点**不需要真实 API key**，`--base-url` 指向 `localhost` / `127.*` 时自动发送 dummy key。

---

## 3. 验证形式化验证链路（无需 LLM）

验证"符号推导 → Lean 形式化"在你的机器上可用：

```bash
cd agent/lean/project
lake build Huginn
```

预期输出：`Huginn` 目标编译成功（生成 `.olean` 缓存）。

这演示了：
- **SymPy** 从强形式推导弱形式
- **Lean 4** 把符号表达式编译为已验证的 `Float` 定义
- 从微积分到类型检查证明的整条桥全程自动化

---

## 4. 交互聊天 Interactive Chat

```bash
huginn-agent chat
```

试试提问：

```
> 推导 1D 热传导的弱形式，并在 Lean 中验证
```

Agent 会：
1. 调用 `symbolic_math_tool`（`action=weak_form`，`target=heat_conduction`）
2. 得到双线性形式 `k*ux*vx` 与线性泛函 `f*v`
3. 自动路由到 `lean_tool`（`auto_verify_action=fem`）
4. 生成 Lean 代码，用 `lake build` 编译并回报成功

---

## 5. 架构一览 Architecture

```
User Input
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│  SymbolicMath   │────▶│    LeanTool     │
│  (SymPy)        │     │  (Lean 4 + Lake)│
└─────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
   weak_form terms         type-checked
   bilinear_form           Float definitions
   linear_functional
```

**关键洞见**：Agent 不"信任"LLM 的数学。每个符号结果都会翻译成 Lean 4 并必须通过
类型检查器才呈现给用户。

---

## 6. 疑难排查 Troubleshooting

| 问题 | 解决 |
|------|------|
| Windows 下 `UnicodeEncodeError` | 用 `PYTHONIOENCODING=utf-8` 运行 |
| `lake` 未找到 | 通过 `elan` 安装 Lean 4 |
| Lean 构建超时 | 首次构建较慢；保留 `build/` 缓存 |
| VASP/LAMMPS 未找到 | 工具自动降级为 mock/导出模式 |
| 本地模型 API key 报错 | 确保 `--base-url` 指向 `localhost` 或 `127.*` |
| 后端未启动 | 先 `python -m huginn.server`，CLI 作为客户端连接 |

---

## 7. 下一步 Next Steps

- **探索工作流**：`agent/huginn/workflows/templates.py` 含多个预设 pipeline
- **添加 Lean 模块**：见 `agent/lean/project/`
- **运行测试**：`cd agent && pytest tests/ -x -q`
- **阅读威胁模型**：`docs/threat_model.md`
- **阅读文档导航**：`agent/docs/INDEX.md`