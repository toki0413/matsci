# 托管部署落地方案 + 推广形态分析

> 目标：破「没人下载 agent」的死结。核心思路是从「用户本地下载安装运行」迁移为「用户浏览器访问、服务器托管、Huginn + 自部署 LLM(如 qwen3.8-27B) 做后端」的 SaaS 形态。本文同时给出规模化落地的技术清单与推广形态判断路径。

---

## 0. 当前形态基线（从代码核实）

项目已经具备服务化的全部基础件，不是从零搭框架：

| 能力 | 位置 | 现状 |
|---|---|---|
| FastAPI + WebSocket server | [server.py](file:///workspace/agent/huginn/server.py) | `FastAPI(...)` + `lifespan`，`uvicorn.run(app, host="127.0.0.1", port=<默认8000>, ws_ping_interval=300)` |
| 实时对话/流式 | `routes/ws.py`、autoloop streaming | autoloop streaming 默认 on |
| 鉴权 | [require_api_key](file:///workspace/agent/huginn/security/auth.py) + `routes/auth.py` | 全局 `Depends(require_api_key)`，另有 login/token |
| 限流 | `server.py` 头 | sliding window，生产默认 120 req/min，auth 更严 |
| CORS/中间件 | `server.py` + `middleware/*` | CORS、error_normalize、limits、maintenance、request_id |
| 本地模型池 | [models/registry.py](file:///workspace/agent/huginn/models/registry.py) | `ollama`(默认 qwen3.8)、`vllm`、`sglang`、`llama-cpp`、`lm-studio`、`openai-compatible`，`base_url` 可指向远程 |
| 安全沙箱 | `huginn/security/*` | sandbox/container/docker/landlock/RBAC/command_filter |

**现状缺口**：`host="127.0.0.1"` 只绑本机 → 需要改成 `0.0.0.0`（并配好反向代理/鉴权）；无静态前端托管、无 TLS、无多租户隔离、无一键部署。

---

## 1. 目标形态

```
 用户浏览器 ──HTTPS──► 反向代理(nginx/caddy) ──► Huginn Server(FastAPI+WS)
                                                    │
                                                    ├─（同一进程/容器内）qwen3.8-27B (vLLM/Ollama)
                                                    └─ 工具执行 + 沙箱 + 存储
```

三形态定位：
- **托管网页版（主攻获客）**：零安装、零配置、拿账号即用。
- **终端/CLI 版（维持）**：重度、自动化、隐私敏感用户。
- **本地版（保留）**：本地 vLLM 离线跑（已有 `local_only_mode`）。

---

## 2. 部署落地清单（分里程碑）

### M0 · 最小可用托管（先跑通获客闭环）
- 改绑定：`uvicorn.run(app, host="0.0.0.0")`（加 `HUGINN_HOST` env 覆盖，默认仍本机）。
- 反向代理：nginx/caddy 终结 TLS（Let's Encrypt），代理到服务器 `127.0.0.1:8000`。
- 鉴权：启用 `require_api_key`，通过 `/auth/login` 签发 token；前端把 API key/token 带上 `Authorization`/`X-HUGINN-API-KEY`。
- 静态前端：新增简单 HTML/JS 前端（或复用 ws 清理面）挂到 server/代理，让用户网页对话触发工作流。
- 模型：vLLM serve qwen3.8-27B，`HUGINN_BASE_URL=http://127.0.0.1:8000/v1`，registry 里配 `vllm` provider。
- 存储持久化：挂卷持久化 `.huginn`（memory/plans/state）。

### M1 · 稳健性与安全（多用户前必做）
- 多用户隔离：`request_id`→全程 trace；`credential_store` 按用户；`workspace/project` 按用户分目录。
- 资源上限：`TokenBudget`、tool 超时、container sandbox 强制、`rate_limiter` 按用户。
- 审计：`security/audit.py`、`cumulative_audit` 落到服务器侧日志。
- 后台任务编排：`scheduling`, `queue`, `persistence/remote_job` 接上（长时任务与用户会话解耦）。

### M2 · 规模化/产品化
- 多副本 + 共享存储（模型、工具只读层 + 用户态 wd 分离）。
- 观测：`metrics`、`telemetry`、`health`。
- 计费/配额：按 token/model tier 计费（已有 `cost_ledger`、`ModelConfig` tier）。
- 发布：`behavior_lifecycle`（整体换目录 + 健康门控回滚）作为服务更新范式。

---

## 3. 自由度的「旋钮化」建议

不增加绝对自由度，而是把现有阀门暴露成「自治档位」面向用户：

- 档位（低/中/高自治），映射到现有机制：
  - `dynamic_workflow` 开关
  - `hypothesis loop` 开启程度
  - `phase_gate` / `cognitive_checks` 的严苛度
  - `plan_check` 注入深度
- 托管版默认**低档收敛**（安全、可预期），明显降低用户理解成本，也利于服务器侧安全。
- 本地版可默认高档放开。

> 原则：同一个引擎，自由度是「用户可选参数」，不是改死。当前 `feature_flags` 已是单一读取路径，扩展成三档预设即可。

---

## 4. 推广形态分析

### 现状痛点
- **用户无感无体验**：要用户下载、装依赖、配 key，前置失败率高。
- **心智成本高**：功能强但使用门槛高，普通材料用户不愿深入。

### 主攻形态建议：**托管网页 SaaS（Plug & Play）**
理由：
1. 零安装 = 获客杠杆最大。「拿账号即用」把 ARPU 转化路径前移。
2. 模型放后端（自托管 qwen/27B），用户不碰 GPU/配置 → 门槛归零。
3. 天然支持订阅/用量计费，形成商业模式闭环。

### 配套形态
- 学术/企业试用：给出带 API 的托管版，接 B 端用量付费。
- 开源引流：保留本地版（尤其 `local_only_mode` 离线），作为信任种子与开发者生态入口。
- 内容/案例驱动：用托管版做公开 demo、基准结果（bench/evidence_manifest）做传播素材。

### 阶段判断
1. **先上 M0 托管最小闭环**，验证「用户愿不愿意用网页直接跑物理/材料工作流」。
2. 用埋点（telemetry/metrics）观察留存与任务完成率，再决定深入 M2。
3. 若托管版留存好 → 把重心放服务器侧规模化；若不好 → 说明漏斗在别处（示例/引导），调整而非加功能。

---

## 5. 开放式问题（需决策）
- 单用户私有云 vs 多租户公网：决定沙箱与隔离复杂度。首推单机多用户＋容器沙箱过渡。
- 模型预算：27B 自托管 vs 云端更强模型的 tier 混合。
- 前端：极简单页是否够（首个 demo），还是要完整 IDE 式面板。
- 成本归属：token 是平台承担（体验期）还是转给用户（付费）。