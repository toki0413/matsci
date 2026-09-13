# Huginn 产业级验收清单（INDUSTRIAL ACCEPTANCE CHECKLIST）

> 版本：2026-09-12（基于当日实盘验证，非读文档杜撰）
> 目的：把"agent 能不能产业级验收、能不能干科研"从主观争论变成**可复现、可判定**的检查单。
> 每个门禁都必须给出**跑了什么命令 + 看到了什么输出 + 通过/未通过**，不允许 "应该能过"。

---

## 0. 结论速览

| 维度 | 状态 | 证据位置 |
|---|---|---|
| 运行环境可拉起（import / 依赖） | 🟢 通过 | §1 |
| 测试套件可收集（0 import 错误） | 🟢 通过 | §2 |
| 核心逻辑 + 规范门禁 | 🟢 通过（81 pass, 0 fail） | §3 |
| 数学/物理计算链路真实可算 | 🟢 通过（50 pass, 5 skip） | §4 |
| 科研执行层产出可复现数值证据 | 🟢 通过 | §5 |
| 全量套件绿灯（9526 用例） | ⚪ 待跑（受时间/重型依赖限制） | §6 |

> ⚪ 不代表"失败"，代表"尚未给出证据"。验收关键是：每个 ⚪ 都要有人填上实跑数字才能宣称"全绿通过"。

---

## 1. 运行环境：import 必须零错误

**目的**：证明依赖装配可复现，不是"这台机器碰巧能跑"。

```bash
# 1.1 干净环境必备依赖清单（来自 pyproject.toml dependencies + dev 组）
python3 -m pip install \
  "pydantic>=2.13" "langchain" "langchain-core" "langchain-openai" "langgraph" \
  "langgraph-checkpoint-sqlite" "scipy" "sympy" "z3-solver" "aiohttp" \
  "rich" "click" "networkx" "python-dotenv" "pyyaml" "cryptography" "toml" \
  "mcp<2.0" "fastapi" "uvicorn" "sse-starlette" "python-multipart" "httpx" \
  "websockets" "requests" "tenacity" "Pillow" \
  "pytest" "pytest-asyncio" "pytest-cov"

# 1.2 证明包本身可导入
PYTHONPATH=. python3 -c "import huginn; print('OK', huginn.__file__)"
```

**判定**：`import huginn` 打印 `OK` 且无 traceback → 通过。

**必须注意**：
- 依赖清单**尚未与测试实际需要的重型库完全对齐**。当日实测：装完核心依赖后收集 ERROR 从 334 → 0，靠的是逐个补齐缺失包（曾缺 `pydantic`、`langchain_core`、`pytest-asyncio`）。
- 建议：用 `uv lock` / `requirements.lock` 生成**锁定文件**，让验收环境一键复现，而非靠人肉补装。

---

## 2. 测试套件可收集：0 import 错误

**目的**：证明 500+ 测试文件全部能被 pytest 导入，没有坏掉的模块。

```bash
cd agent
python3 -m pytest -o addopts="" --collect-only -q 2>&1 | tail -4
```

**当日实测输出**：
```
9526 tests collected in 12.91s
```

**判定**：
- 无 `ERROR collecting ...` → 通过
- 有任何 collection ERROR → **未通过**，逐条修到 0（多为缺依赖，本文档 §1）

> 注意：pyproject 的 `addopts = "--cov=..."` 需要 `pytest-cov`，且覆盖率门禁 `fail_under=60` 是纸面声明，CI 全量也用 `-p no:cov` 关闭（见 pyproject 注释）。收集请用 `-o addopts=""` 避免被 cov 参数干扰。

---

## 3. 核心逻辑 + 规范门禁

**目的**：证明 agent 主循环、设计规范、价值约束不被破坏。

```bash
timeout 300 python3 -m pytest -o addopts="" \
  tests/test_arch_cleanliness.py \
  tests/test_arch_single_gateway.py \
  tests/test_math_eval.py \
  tests/test_physics_auditor.py \
  tests/test_adapter_constraints.py \
  -q -p no:cacheprovider
```

**当日实测输出**：
```
81 passed, 2 warnings in 7.09s
```

**判定**：`0 failed` → 通过。

**此项覆盖的关键能力**：
- **架构整洁门禁**：自研接缝（autoloop/agent/agents/research/...）不得有静默 catch-all、不得有 NotImplementedError 占位。
  - 当日已修复：`engine_act.py` / `engine_reflect.py` / `plan_check.py` 共 11 处裸 `except Exception` 全补原因注释，红灯转绿。
- **单网关铁律**：唯一业务网关，禁止旁路。
- **安全数值求值**、**物理量纲审计**：科学计算结果不被篡改、量纲不被误用。

---

## 4. 数学/物理计算链路真实可算

**目的**：证明 agent 的"计算"不是 LLM 嘴硬，而是符号引擎 + 数值引擎真的算对了。

```bash
timeout 300 python3 -m pytest -o addopts="" \
  tests/test_bourbaki_sympy.py \
  tests/test_math_validation.py \
  tests/test_numerical_analysis.py \
  tests/test_math_eval.py \
  -q -p no:cacheprovider
```

**当日实测输出**：
```
50 passed, 5 skipped in 6.41s
```

**判定**：`0 failed` → 通过。覆盖：符号守恒校验、2×2 特征值数值分析、安全求值器等。

---

## 5. 科研执行层：产出可复现数值证据

**目的**：证明 agent 能对科研问题输出**带证据的数值判定**，并能诚实给出边界（inconclusive），而不是编数。

**验证方式**（自包含探针，不依赖 LLM）：
```python
import subprocess, tempfile, re
from pathlib import Path
def _run(snippet):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "run.py"; f.write_text(snippet)
        r = subprocess.run([__import__("sys").executable, str(f)],
                           capture_output=True, text=True, timeout=20)
        return "" if r.returncode else r.stdout.strip()
```

**当日实测（真实执行的可复现数值）**：
| 目标 | 执行代码 | 输出 | 判定 |
|---|---|---|---|
| 应力安全（F=500N, A=0.01, yield=250MPa） | `s=500/0.01; print(f'{s/1e6:.6g} {"PASS" if s<250e6 else "FAIL"}')` | `0.05 PASS` | 闭式可判定 |
| 维恩位移（T=3000K） | `print(f'{b/T*1e9:.6g} nm')` | `965.924 nm` | 有 exec 证据 |
| 折射（Snell 60°→n2=1.5） | `math.degrees(asin(n1/n2·sin60))` | `35.2644` | 有 exec 证据 |
| 托里拆利出口速度（h=2m） | `math.sqrt(2·9.81·2)` | `6.264 m/s` | 可复现 |
| 基础复现 | `print(round(500/0.01,6))` | `50000.0` | 可复现 |

**判定**：子进程能真实 exec 并取回 stdout 中的数值、且可复现 → 具备"科研能跑数"的底层能力。

**诚实边界（重要）**：执行层每次只算**给定参数下的单点数值**，判定闭式问题**可判定性**（PASS/FAIL）。但它**不自动验证"物理模型是否适用于该参数域"**——超出假设边界时（如 Arrhenius 参数下溢），需要 agent 在 VERDICT 里显式标 INCONCLUSIVE + RECORD 边界。这条靠 §3 的 `physics_auditor` / 规范门禁兜底，仍需人工复核科研结论的模型适用性。

---

## 6. 未跑项：全量 9526 用例（明确诚实声明）

- **为何没跑**：全量含 heavy 集成 / 网络 / 需要外部求解器（VASP、LAMMPS、FEniCS 等）的测试（`markers = integration/network`），单机长时间跑会超时，且其中部分依赖第三方可执行文件，环境不具备。
- **禁忌**：**任何人在未跑全量、未用 `pytest -m "not integration and not network"` 拿到绿之前，不得宣称"全部测试通过"**。这违反验证纪律。
- **可跑的有界快照**（建议 CI 用）：
  ```bash
  python3 -m pytest -o addopts="" -m "not integration and not network" -q
  ```

---

## 7. 交付判定模板（每次验收填这张表）

| 门禁 | 命令 | 实测结果 | 通过? |
|---|---|---|---|
| 环境可导 | §1 | import OK | 🟢/🔴 |
| 套件可收集 | §2 | N tests, 0 errors | 🟢/🔴 |
| 核心+规范 | §3 | X pass, 0 fail | 🟢/🔴 |
| 计算链路 | §4 | X pass, 0 fail | 🟢/🔴 |
| 执行层证据 | §5 | 数值可复现 + 边界诚实 | 🟢/🔴 |
| 全量绿灯 | §6 | X pass, Y fail | 🟢/🔴/⚪ |

**产业级可达的一个合理定义**：
1. §1–§5 全 🟢；
2. §6 无轻量/纯逻辑 fail；
3. §7 表格由**实跑**（而非"应该"）填满。
三项齐 → 可宣示"达到产业级验收基线"。任何缺失 → 如实标 ⚪，不得宣称完成。

---

## A. 科研实证案例：分形盆地假说的三探针检验（2026-09-13）

**目的**：不把"agent 是否能科研"停留在表态，而是对一个**真实可证伪的科学假说**走完整实证流程，并诚实给出"否"的结论（含记录测量陷阱）。

**假说（研究对象）**：Agent 的失败判定边界 / 想象末态语义分布在参数空间的边界呈**自相似分形**，即不确定性指数 β<1（其中 β 由 `P_cross(δ) ~ δ^β` 界定，Chern–Markus 标度）。

**为什么值得测**：若成立，意味着"初始状态微小扰动→结果大幅分叉"是 agent 的内禀特征，影响可复现性论证。**但必须在测量前先证值得、测量中不为凑 β<1 而改系统（动机倒置红线）**。

**方法论（三探针共用一个测量协议）**：
- 在 2D 参数网格上按间距 δ 采样两点，统计落入不同"末态吸引域"占比 `P_cross(δ)`；
- 拟合 `log P_cross ~ β·log δ`；
- **决定性判别**：分形 = 细/粗 δ 窗斜率**一致且稳定 <1**；有限直边（非分形）= 细斜率→1、粗斜率→0；纯量化伪影 = β 随分辨率收敛到 1；
- **双向对照**：光滑构造（竖直直线）必须标定出 β≈1，噪声构造（随机标签）必须标定出 β≈0，以证明估计器本身可靠。

### A.1 探针一：连续语义空间（JEPA 语义余弦）

```bash
cd /workspace && HF_ENDPOINT=https://hf-mirror.com \
  PYTHONPATH=/workspace/agent python3 scripts/research_a_jepa_surprise.py --embedding --beta
```
实测：`真实吸引域 β=+0.978 R²=0.998 (拟合点 8/8) → 平滑(β≈1)`；对照 `光滑 β=+1.126`、`噪声 β=-0.010`。
**判定**：边界平滑（β≈1），**不支持分形**。

### A.2 探针二：布尔操作分类器（CompletionAuditor）

```bash
cd /workspace && PYTHONPATH=/workspace/agent \
  python3 scripts/probe_failure_basins.py --beta2d
python3 scripts/probe_failure_basins.py --deepdive   # shallow: 分辨率扫描
```
实测：
- baseline / deep 基态：整网格指纹恒定 → `无边界可测`（布尔阈值太粗）；
- shallow 基态：总体 β≈0.81–0.86 稳定，但分解后 `细δβ均值=+1.066`、`粗δβ均值=+0.420`，细粗差 −0.646 且**不随分辨率改变** → **有限直边交叉网，非分形**（β≈0.85 只是整数阈值在非渐近 δ 窗口的混叠伪影）。
**判定**：非分形；布尔阈值"量化阶梯"会生成伪亚 1 信号，必须靠细/粗 δ 分解识破。

### A.3 探针三：想象末态盆地（动机 B 数值代理）

```bash
cd /workspace && python3 scripts/research_imagination_basins.py
```
实测（扫描拉伸折叠强度 λ）：
| λ | 总体β | 细δβ | 粗δβ | 判读 |
|----|------|------|------|------|
| 1.0 | 0.631 | 0.880 | 0.212 | 有限直边，非分形 |
| 1.5 | 0.037 | 0.020 | 0.020 | 混沌纠缠(≈噪声) |
| 2.0 | 0.028 | 0.041 | 0.004 | 混沌纠缠(≈噪声) |
对照：光滑 β≈1.13、噪声 β≈0.01。
**判定**：λ≥1.5 时 β 陡降至与噪声同量级。**关键陷阱**：β→0 在数值上"很像分形"，但它在统计上与随机打标**不可区分**——因此不构成可证伪分形证据，如实标为"混沌纠缠(≈噪声)"，**禁止**宣称"发现分形"。

### A.4 总判定

- 三条独立、方法同构的测量路（连续语义 / 布尔操作 / 想象末态）**均未给出可证伪的分形边界**。
- 机制层面结论：agent 的完成判定与 `imagine` 均为**单步/离散/规则**结构，**缺失分形盆地所需的动力学折叠 + 多个吸引子收敛**——借"混沌动力学的分形盆地"类比到 agent 是**模型错配**。
- **诚实收口**：假说被否决，不是测量做得不够，而是机制上不成立；且本流程演示了"为凑 β<1 而改系统 == 动机倒置"与"把 β→0 噪声当分形"两个必须拒绝的陷阱。

**证据产物（脚本，均可复现）**：
`scripts/research_a_jepa_surprise.py`（`--embedding --beta`）、`scripts/probe_failure_basins.py`（`--beta2d` / `--deepdive`）、`scripts/research_imagination_basins.py`。