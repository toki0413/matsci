# 技术债登记：`ponytail:` 临时捷径清单

> 状态：**active**。本文件是代码库内所有 `ponytail:` 标记的**显式登记面**，
> 把"作者心里清楚的权宜之计"从隐式注释升级为可审计、可收敛的清单。

## 1. 为什么要有这份清单

代码里大量使用 `ponytail:` 前缀标注"够用就好的最小实现"（源自 ponytail
"懒人资深开发者"决策法：先问是否需要存在，再复用/一行方案，最后才最小实现）。
这套标记本身是好习惯——它承认了天花板。但问题是：**只有承认、没有收敛**。

结果（外部审计原话）："代码里到处是 `ponytail:` 自认的临时捷径……项目处于
'铺框架'而非'打磨可靠性'的阶段。"

本清单的作用：
1. 把所有捷径按**严重度**分级，S1/S2 级（安全/边界）不再混在 S3/S4 级里被淹没。
2. 每条 S1/S2 记明**天花板**与**升级路径**，并给出**可被测试锚定**的定位，
   防止"文档说改了、代码没改"或"代码改了、文档没删"的双向漂移。
3. 明确政策：新增捷径需显式登记；S1 级只减不增。

## 2. 严重度分级

| 级别 | 含义 | 处理要求 |
|---|---|---|
| **S1 安全** | 可被利用：凭据泄漏 / 沙箱逃逸 / 权限绕过 | 必须登记，收敛优先级最高，只减不增 |
| **S2 边界** | 隔离/权限/资源边界不完整，存在明确绕过面 | 必须登记，给出升级路径 |
| **S3 能力** | 功能可用但覆盖窄（工具/算法/格式子集） | 建议登记，按需升级 |
| **S4 工程** | 依赖/可读性/一致性权衡 | 可不登记，代码注释即可 |

## 3. 普查（可复现）

```bash
# agent 侧 Python 标记总数与文件数
grep -rEoh 'ponytail:' agent/huginn --include='*.py' | wc -l      # 1261
grep -rlE 'ponytail:'  agent/huginn --include='*.py' | wc -l      # 289
```

> 说明：1261 条里绝大多数是 S3/S4（"不引新依赖""本地缓存够用"这类）。
> 真正需要盯住的是下面 S1/S2 两张表——它们是能力可靠性与安全边界的杠杆点。

标记密度最高的前几个文件（提示"深水区"，改动时需格外小心）：
`cli/rcb_step2.py`(59)、`autoloop/cognitive_loop.py`(47)、`cli/rcb_runner.py`(35)、
`autoloop/hypothesis_loop.py`(31)、`autoloop/engine_reflect.py`(27)、
`metacog/hypothesis_manifold.py`(24)。

## 4. S1 安全级登记

| ID | 位置 | 天花板（风险） | 升级路径 | 状态 |
|---|---|---|---|---|
| DEBT-S1-01 | `desktop/src/lib/api-client.ts:26` | JWT 存 `localStorage`，页面内任意脚本（XSS）可读 | 迁移到 `tauri-plugin-stronghold`（OS keychain） | open |
| DEBT-S1-02 | `desktop/src/lib/api.ts:46` | 原始 API key 存 `localStorage`，同 XSS 暴露面 | 与 S1-01 一起迁 stronghold | open |
| DEBT-S1-03 | `desktop/src/lib/config-store.ts:86` | 配置 blob 含 `api_key`/`encryption_password`，明文落 `localStorage` | 敏感字段迁 stronghold，非敏感字段留 `localStorage` | open |
| DEBT-S1-04 | `agent/huginn/tools/sci/numerical_tool.py:738` | `eval(obj_line, {"__builtins__": {}})` 挡得住常见逃逸，但 numpy/cvxpy 属性链仍可触达 `os`/`subprocess` | 表达式改走 `huginn.security.safe_eval`（AST 白名单，禁属性访问）；或按不可信输入路由到 Docker 沙箱 | open |
| DEBT-S1-05 | `agent/huginn/security/code_act_sandbox.py:10` | in-process `exec` 共享解释器，属性遍历（`().__class__.__bases__[0].__subclasses__()`）可逃逸 | 硬隔离走 Docker 沙箱 + 独立 namespace，或 E2B | open |

## 5. S2 边界级登记

| ID | 位置 | 天花板（风险） | 升级路径 | 状态 |
|---|---|---|---|---|
| DEBT-S2-01 | `agent/huginn/security/sandbox.py:293` | Windows 用 `cmd /c` 适配 POSIX coreutils，参数语义与 POSIX 不完全一致 | 装 git-bash 让 coreutils 进 PATH，或切 WSL | open |
| DEBT-S2-02 | `agent/huginn/security/docker_sandbox.py:43` | 只给 light/standard/heavy 三档 profile，无连续配置 | profile 从 env 读，或按 tool `constraint_scope` 自动选 | open |
| DEBT-S2-03 | `agent/huginn/security/policy_engine.py:206` | `HUGINN_WORKSPACE` 未设时静默丢路径隔离（path scoping 失效） | 未设时改为"拒绝执行"或显式告警升级为错误 | open |

## 6. 本轮已收敛（外部审计批评项）

| ID | 原问题 | 收敛动作 | 位置 |
|---|---|---|---|
| FIX-01 | `auto_approve_all=True` 硬编码 | 改为可配置：`sidecar_auto_approve()` / `autoloop_auto_approve()`，dev 模式默认放行、生产默认不放行 | [permissions.py](../huginn/permissions.py)、[routes/agents.py](../huginn/routes/agents.py)、[routes/knowledge.py](../huginn/routes/knowledge.py)、[autoloop/engine.py](../huginn/autoloop/engine.py) |
| FIX-02 | 鉴权默认裸奔 `/diagnostics` | 从 `_PUBLIC_PATHS` 移除 `/diagnostics*`，强制鉴权 | [security/auth.py](../huginn/security/auth.py)，回归见 [test_security_fixes.py](../tests/test_security_fixes.py) |
| FIX-03 | 沙箱纵深靠外层容器 | `sandbox_net_isolation` 默认打开（Landlock 内核级网络隔离，不支持则优雅降级） | [feature_flags.py](../huginn/feature_flags.py)、[security/sandbox.py](../huginn/security/sandbox.py) |
| FIX-04 | 评测"有壳无闭环" | NatureBench 复现门禁接入 CI（10 题论文复现 + SOTA 判分闭环） | [tests/test_naturebench.py](../tests/test_naturebench.py)、[.github/workflows/ci.yml](../../.github/workflows/ci.yml) |

## 7. 政策

1. **新增捷径**：提交引入新的 `ponytail:` 且属 S1/S2 级时，必须在本清单登记
   （ID / 位置 / 天花板 / 升级路径）。
2. **S1 只减不增**：安全级条目数量不得增加；新增须在 PR 说明理由并给出收敛计划。
3. **双向防漂移**：`tests/test_ponytail_debt_registry.py` 锚定本清单第 4/5 节
   每条 open 条目对应的代码事实，防止"文档说改了、代码没改"。
4. **收敛即销账**：条目修复后从第 4/5 节移除、记入第 6 节，并删掉对应代码里的
   `ponytail:` 标记。