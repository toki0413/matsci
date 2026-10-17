# 能力可运行性面 (Capability Runtime Surface) — Design

> Status: APPROVED
> Source: brainstorming 会话 (superpowers)
> Date: 2026-10-16
> Scope: 单 spec; 落地由 writing-plans 承接

## 1. 背景与问题

治理层只保证**能力面的存在性与不退化**(锚点/开关/承载模块/不变量测试在位),不保证
**能力真的能跑**。当前存在一条确定的盲区链:

- 缺 Python 库时,工具若在模块顶层 `import`(如 `characterization_tool` 的 `numpy`
  顶层导入)会在注册期 `ImportError → skip`,**名字根本不进 LLM 工具表**;
- 若工具用 `try/except` 包住重依赖,则**注册成功但依赖缺失**,`is_available()` 默认
  仍为 `True`,LLM 看得见、**调用时才失败**;
- 只有 `is_available()` 被正确覆盖的工具才会被 `get_all_schemas()` 过滤掉,表现为
  "隐形缺失"。

后果:**LLM 对「缺了哪些能力、缺什么、怎么装」普遍不自知**;治理台账也看不见这条缝隙。
现有的两处机制都不闭环:

- `huginn/tools/__init__.py:probe_tool_dependencies` 已做**静态 import 扫描**,能算出
  `{module: [missing_deps]}`,但返回值被 **丢弃、只进日志**——面向 operator,不面向模型。
- `huginn/capabilities/introspection.py` 已做**能力面自省**(registered / discoverable /
  dangling),但**只审名字层面**,不审依赖层面,且不区分"缺依赖"与"被策略隐藏"。
- `huginn/cli/rcb/smoke.py` 有启动冒烟,但**只覆盖 RCB 的 4 项依赖、只 fail-fast 给
  operator**,不进 LLM 上下文、不进台账。

## 2. 目标与非目标

### 目标

1. **L0 可运行性盘点**:对每个工具与能力锚点,真实探测"依赖能否加载、工具能否实例化",
   产出结构化报告,区分 `runnable / missing_deps / not_instantiable / hidden`。
2. **L1 精选端到端 smoke**:对科学关键能力各跑一条最小真跑任务,验证"能出正确产物"。
3. **常驻汇总 + 按需详单**:汇总行进 system prompt,详单走诊断工具/CLI。
4. **运行时自愈**:缺依赖 → `pip install`(既有 ask 门禁) → 复验 → 回写状态。
5. **治理可观察**:台账登记探针清单与冻结基线,防清单静默缩水。

### 非目标 (YAGNI)

- **不阻断 CI**:环境差异大,可运行性探针结果不阻断流水线。
- **不自动装包**:保留 ask 门禁,维护供应链安全。
- **不新增注册表**:复用 `ToolRegistry` + `CAPABILITY_SPECS`。
- **不承诺科学正确**:L1 只验证"最小任务能跑通并产出可判定的产物",不验证领域正确性。
- **不推倒 introspection**:只**扩展**其语义。

## 3. 四层保证模型

把"确保端到端跑通"落成四层**可证伪**的保证,逐层堵不同的漏:

| 层 | 判什么 | 防什么 | 现状 |
|---|---|---|---|
| 存在性棘轮 (已有) | 锚点/开关/模块/测试在否 | 能力被悄悄删/关 | 已有 (`CAPABILITY_SPECS`) |
| **L0 可运行性探针** | 依赖能否加载、工具能否实例化 | 名字在、依赖缺的**静默降级** | **本 spec 新增** |
| **L1 精选端到端 smoke** | 最小任务能否出正确产物 | 能加载但**跑错** | **本 spec 新增** |
| **自愈 + 棘轮** | 缺依赖可 ask-install → 复验 → 回写 | 缺口从"永久红灯"变"可修复动作" | **本 spec 新增** |

## 4. 架构与边界

关键张力:`governance_audit.py` 刻意**保持 weightless**(不 import `huginn.*`、纯 stdlib、
CI 可跑);而"依赖在不在"必须**真实 import**。两者分离:

```
                 ┌─ weightless 台账 (CI 每次跑, 不阻断)
治理面 ──────────┤   governance_audit: 只登记「探针清单 + 数量冻结基线」
                 └─ 棘轮: 探针清单/锚点不得静默缩水

                 ┌─ L0 可运行性探针 (会话构建时跑; 快、无网)
运行面 ──────────┤   静态依赖扫描 + is_available() + 实例化
                 ├─ L1 精选端到端 smoke (按需 / nightly; 依赖齐备环境)
                 └─ 自愈: missing_deps → pip install(ask) → 复验 → 回写
```

一句话:**台账管「清单不许缩」,运行面管「依赖真的在、任务真的能跑」。**

## 5. 组件

| 单元 | 类型 | 职责 | 依赖 |
|---|---|---|---|
| `huginn/capabilities/runtime_probe.py` | 新增 | L0 探针 + 报告组装 + `summary_line()` | 复用 `probe_tool_dependencies` / `_collect_top_level_imports` |
| `huginn/capabilities/smoke_specs.py` | 新增 | L1 精选清单 `SMOKE_SPECS` + `run_smoke()` | 各科学工具 |
| `huginn/agent/context.py` | 改 | 常驻注入 `summary_line()`(扩展现有 VASP/LAMMPS note 逻辑) | runtime_probe |
| `huginn/tools/system_diagnostic_tool.py` | 改 | 新增 `capability_runtime` 子命令 → 按需详单 | runtime_probe |
| `huginn/cli/capability_runtime.py` | 新增 | `--json / --smoke / --check`,与 `config_audit` 同构 | runtime_probe |
| `huginn/cli/governance_audit.py` | 改 | 新增「能力可运行性探针」组 + 计数棘轮(weightless) | 无 |
| `tests/test_capability_runtime_probe.py` | 新增 | 探针单测(weightless) | runtime_probe |
| `tests/test_capability_runtime_report.py` | 新增 | 报告确定性单测 | runtime_probe |

**不新增注册表**;**不自建装包器**(复用 `bash_tool` 的 ask 通道)。

## 6. 关键接口

```python
# huginn/capabilities/runtime_probe.py

@dataclass
class ProbeResult:
    name: str                       # 工具名或能力锚点名
    kind: str                       # "tool" | "anchor"
    status: str                     # runnable | missing_deps | not_instantiable | hidden
    missing_deps: list[str] = field(default_factory=list)
    install_hint: str = ""          # 例: "pip install rdkit torch"

def probe_runtime(tool_lists=None, anchors=None) -> list[ProbeResult]: ...
    # 纯逻辑 + 可注入清单 (确定性, 可测); 任何异常收敛为 status, 绝不抛。

def build_report(l0: list[ProbeResult] | None = None,
                 l1: list[SmokeResult] | None = None) -> CapabilityRuntimeReport: ...

def summary_line(report: CapabilityRuntimeReport) -> str: ...
    # 一行, 供注入 system prompt。
```

```python
# huginn/capabilities/smoke_specs.py

@dataclass
class SmokeSpec:
    name: str
    capability: str                 # 如 "molecule.property"
    requires: tuple[str, ...] = ()  # 缺这些则 skip, 不 fail
    runner: Callable[[], Any]       # 真跑; 抛异常 = 失败
    expect: Callable[[Any], bool] | None = None

@dataclass
class SmokeResult:
    name: str
    capability: str
    status: str                     # passed | failed | skipped
    detail: str = ""                # skipped 时记缺失依赖; failed 时记异常摘要
    elapsed: float = 0.0

def run_smoke(specs=None) -> list[SmokeResult]: ...
```

### L1 起始清单 (可调)

| name | capability | requires | 最小真跑 |
|---|---|---|---|
| `lit_arxiv_reach` | 文献检索 | (网络) | 构造 arXiv 查询并校验返回结构 |
| `rdkit_mol_props` | 分子结构 | rdkit | `MolFromSmiles("c1ccccc1")` → `MolWt` |
| `xrd_peak_detect` | 谱图解析 | numpy | 合成峰 → 峰位数与位置 |
| `curve_fit` | 实验曲线 | numpy | 线性数据 → 拟合参数近似 |

前两条 + 后两条与 `rcb/smoke.py` 的口径一致,可复用其探测逻辑,避免第二套。

## 7. 数据流与自愈时序

1. **会话构建**:跑 L0(快、无网) → `summary_line()` 注入 system prompt。
   例:`能力降级 12 项: rdkit/torch/pymatgen 缺,可 pip install rdkit torch;3 项需外部许可证`。
2. **按需详单**:LLM 调 `system_diagnostic_tool` 的 `capability_runtime` 子命令,或
   `python -m huginn.cli.capability_runtime`,拿到 `ProbeResult` 全表。
3. **L1**:不在会话构建跑(慢) → `--smoke` 或 nightly 触发;**不阻断 CI**。
4. **自愈**:LLM 见 `missing_deps` → 调 `bash_tool` 执行 `pip install ...`
   (**ask 门禁 → 用户确认**) → 成功后令报告缓存失效 → 重算 L0 → 状态转 `runnable`。

## 8. 错误处理与诚实边界

- 探针**绝不抛**:任何 import/实例化异常收敛为 `status`(对齐 `Capability.run()` 的收敛风格)。
- **绝不用"能 import"冒充"能算对"**:L0 与 L1 状态**分开报告**,`runnable` 仅指"可加载"。
- 轻量/沙箱无重栈:best-effort;无 registry 时按清单静态判(对齐 introspection 的
  `except → 空` 模式)。
- L1 runner 缺依赖 → `skip` 不 `fail`(对齐既有 rcb/hypothesis 模式)。
- `summary_line()` 只做**陈述**,不向模型暗示"能力已可用"。

## 9. 测试策略

- `test_capability_runtime_probe.py`:注入假清单 → 断言四类 status 分类、`install_hint`
  生成、异常收敛不抛(**weightless**,不依赖重栈)。
- `test_capability_runtime_report.py`:`summary_line` 确定性、详单有序(字典序)。
- L1:至少一条 **weightless runner**(如纯 stdlib 能力)保证机制可测;重依赖 runner 标
  `requires`,缺则 skip。
- 确定性:报告输出固定列序 + 字典序,跨机逐字一致(对齐 `config_audit` 范式)。

## 10. 治理集成

- `governance_audit.py` 新增 `RUNTIME_PROBE_SPECS` 与计数棘轮 `runtime_probe_floor`
  (weightless:只登记清单与数量,不真跑 import)。
- 确定性再生成 `agent/docs/governance-state.md`,新增「能力可运行性探针」节。
- `tests/test_arch_gate_registry.py` 新增冻结棘轮:探针清单条目数不得缩。

## 11. 验收标准 (AC)

- **AC-1** `python -m huginn.cli.capability_runtime --json` 输出确定性的
  `ProbeResult` 列表;同一环境重复运行逐字一致。
- **AC-2** `probe_runtime` 对注入的假清单正确分类四类 status,且任何输入**不抛异常**。
- **AC-3** `summary_line()` 在存在降级能力时非空、含缺失依赖名与 `install_hint`;
  全部就绪时输出"全部可运行"。
- **AC-4** 会话构建注入的 summary 与实际 L0 结果一致(不杜撰、不缓陈旧值)。
- **AC-5** L1 `--smoke` 对 requires 齐备的能力产出 `passed=True`;缺依赖则 `skip`。
- **AC-6** 自愈路径:bash_tool `pip install` 仍走 ask 门禁(不被绕过);成功后重算 L0 状态转
  `runnable`。
- **AC-7** `governance-state.md` 新增探针节,且与 `render_markdown` 输出逐字相等。
- **AC-8** 元门禁棘轮:探针清单条目数下降 → 红。

## 12. 风险与权衡

- **报告陈旧**:缓存可能与真实环境漂移。缓解:自愈成功后强制失效;会话级构建一次。
- **L0 假阴性**:静态扫描可能漏掉运行时动态 import。缓解:辅以 `is_available()` 与实例化。
- **token 开销**:汇总行须保持一行;详单不进常驻。
- **与 introspection 语义重叠**:明确边界——introspection 管**名字面**,本模块管**依赖面**。
- **沙箱无网**:自愈在网络受限环境不可用 → 如实报告,不假装成功。

## 13. 后续 (超出本 spec)

- L1 清单扩展到更多科学能力(仿真/多模态)。
- 把 L0/L1 历史落成时序工件,支持"可运行性随时间"观察。
- 探讨在专用依赖齐备环境跑 L1 并纳入观察(现阶段不做,避免 CI 变重)。
