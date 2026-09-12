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