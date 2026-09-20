# JEV MCP Server（`servers/jev-mcp/`）

把 TypeSafe 的 **System One 决策模型 JEV**（[typesafe.ai](https://typesafe.ai)，2026-09）
暴露成 **MCP 工具**，让任意 MCP host（含 Huginn 的 `mcp_client`）以普通工具消费，
不再写小作文、返回带校准概率的类型化决策。

## 暴露的工具

| MCP 工具 | 原语 | 入参 | 返回 |
|---|---|---|---|
| `jev_noul` | Noul | `state`(dict) + `questions`(`{名称: 命题}`) | `{名称: {noul, confidence}}` |
| `jev_choice` | Choice | `state` + `instruction` + `options[]` | `{choice, probabilities, confidence}` |
| `jev_score` | Score | `state` + `instruction`(+`levels[]`) | `{score, confidence}` |

所有问题对同一份 `state` **并行、互相隔离**求值；fail-open（无 key/超时 → 显式错误文本）。

## 运行

```bash
# stdio（默认，agent 本地子进程调用）
python servers/jev-mcp/server.py

# 或独立安装后
pip install -e servers/jev-mcp
jev-mcp
```

环境变量：`TYPESAFE_API_KEY`（必需，缺省时 fail-open）；可选 `TYPESAFE_BASE_URL`、
`TYPESAFE_MODEL`。

## Huginn 接入

在 `agent/huginn.toml`（或 `agent/.mcp.json`）注册一个 stdio server，Huginn 的
`mcp_client` 会自动把 `jev_*` 注册为普通工具：

```toml
[mcp_servers.jev]
command = "python"
args = ["/workspace/servers/jev-mcp/server.py"]
env = { TYPESAFE_API_KEY = "..." }   # 用你的 key
```

> 接线语义与 `agent/huginn/runtime/jev/client.py` 一致；三元原语的返回键
> （`noul`/`choice`/`score`/`confidence`）对齐，可复用
> `agent/huginn/security/gate.py::jev_adapter` 把置信归一化进统一门禁链。
> 红线不越：只做工具面/软判断，不进 `claim_grounding` 证据门或 harness 统计门。

## 测试

```bash
python -m pytest tests/test_jev_mcp_endpoint.py -q   # 在仓库根/agent 侧运行
```