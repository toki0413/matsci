# Spec: 离散数学工具层 — 互补人类连续化偏置

> **状态**: Draft (待用户审阅)
> **作者**: agent
> **日期**: 2026-07-20
> **关联**: 因果推断层三阶段已完成 (commit `df11376`); 本 spec 是"互补人类连续化偏置"的第一步落地

---

## 1. 动机

### 1.1 人类思维惯性的局限

人类处理离散问题的默认路径是 **lift 到连续**,再用连续框架的系统化工具 (微积分/谱理论/表示论) 处理。这不是偷懒,是**试错成本逼的**:连续框架一次建工具复用无数次,边际成本趋零;离散问题每个题都需要单独的巧妙洞察 (反射原理/Szemerédi/Erdős 概率法),人力算不过来。

**AI 的机会**:试错边际成本被算力摊薄到几乎为零,可以不走连续化捷径,直接在离散空间里暴力 + 模式识别。

### 1.2 当前 agent 的偏置

代码库现状 ([调研报告](#)):

| 能力 | 现状 | 评分 |
|---|---|---|
| LLM 跨域离散结构类比 | 强 (LLM 本身能力) | 7/10 |
| 连续框架工具 (GP/symbolic regression/TDA/autodiff) | 完整 | 9/10 |
| SAT/SMT 离散推理 | **没有** | 0/10 |
| 离散组合枚举 (生成函数/OEIS 反查) | **没有** | 0/10 |
| 有限群/有限域计算 | **没有** (SymmetryTool 只做晶体学连续群) | 0/10 |
| 加性组合实验 | **没有** | 0/10 |
| 反连续化反例搜索 | **没有主动机制** | 1/10 |

**问题不是"AI 没能力",是"我们装的工具本身就是连续化偏置的"**。当前 agent 更像"人类连续化框架的放大器",不是互补。

### 1.3 历史决策的显式回应

[phase_gate.py:381](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/phase_gate.py#L381) 有显式注释:
> `不新建 SMT 组件, 只路由已有 BourbakiTool / symbolic_math_tool 输出.`

本 spec **部分推翻**这个决策:在 **math tools 层** 新建离散工具 (不只是路由),但在 **autoloop/phase_gate 层** 仍保持"路由"原则 — phase_gate 不直接调 z3,而是消费离散工具的输出。这是分层一致性。

---

## 2. 设计原则 (ponytail)

1. **零新重依赖**:z3-solver 是单 wheel pip 装 (~50MB),允许。SageMath (1GB+) 不引入,用 sympy.combinatorics + sympy.ntheory 替代。
2. **sympy 优先**:有限群/数论用 sympy 已有能力,sympy 不够才上外部库。
3. **shim 模式**:顶层 `tools/discrete_<name>_tool.py` 是 shim,真实实现在 `tools/sci/discrete_<name>.py` 或 `tools/discrete/<name>.py`,与 [symmetry_tool.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/symmetry_tool.py) shim 模式对齐。
4. **声明式天花板**:每个工具用 `ponytail:` 注释声明天花板和升级路径 (参考 [symmetry_tool.py:270-272](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/sci/symmetry_tool.py#L270-L272) 风格)。
5. **selfcheck 必须**:每个工具文件底部 `if __name__ == "__main__":` 块,`assert <cond>, f"expected X, got {actual}"` 模式,最后 `print("[tool_name] self-check OK")`。`python -m huginn.tools.<module>` 可跑。无框架无 fixture。
6. **category 归 "sci"**:不新建 "discrete" category,与 sympy/jax/symmetry 工具对齐 (避免 ToolProfile / phase 表的多处改动)。
7. **不动 VisualSCM**:本 spec 只做工具层,不扩 SCM 的 value_type。离散 SCM 留待后续 spec。
8. **RedTeamReviewer 是 hook 落点**:不在 hypothesis_generator 里加钩子,在 [red_team.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/red_team.py) 的 `review()` 里加 `_discrete_counterexample_scan` 平级扫描器。

---

## 3. P0 — SAT/SMT 求解器 (z3)

### 3.1 目标

离散约束求解的"通用底座",对应连续优化里的 scipy.optimize。

### 3.2 文件结构

```
agent/huginn/tools/
├── discrete_smt_tool.py              # shim (3 行 re-export)
└── sci/
    └── discrete_smt.py               # 实现 (~400 行)
```

### 3.3 工具接口

```python
class DiscreteSMTInput(BaseModel):
    action: Literal[
        "solve_sat",          # 布尔可满足性
        "solve_smt",          # SMT 理论求解 (线性整数算术 / BV / 数论)
        "optimize",           # 离散优化 (min/max 目标)
        "all_solutions",      # 枚举所有解 (最多 N 个)
        "verify_implication", # 验证 A ⟹ B (反例搜索)
    ]
    variables: list[dict]     # 变量声明 [{name, type, domain}]
                              # type: "bool" | "int" | "bv" (bit-vector)
                              # domain: None (bool) | [lo, hi] (int) | width (bv)
    constraints: list[str]    # z3 Python API 表达式字符串
                              # e.g. ["x + y > 10", "x != y", "And(a, Or(b, c))"]
    objective: str | None     # optimize action 的目标 ("minimize x+y" / "maximize x")
    max_solutions: int = 10   # all_solutions 的上限
    timeout_ms: int = 5000    # z3 超时
```

### 3.4 action 行为

- **solve_sat**: 返 `{"sat": True/False, "model": {x: 1, y: 2}, "n_constraints": N}`
- **solve_smt**: 同上,支持 Int/BV/实数理论
- **optimize**: 返 `{"optimal": True/False, "opt_value": 42, "model": {...}}`
- **all_solutions**: 返 `{"solutions": [...], "n_found": K, "truncated": bool}`
- **verify_implication**: 输入 `constraints=A`, `objective=None`,额外参数 `implication=B`,返 `{"holds": bool, "counterexample": dict | None}`

### 3.5 安全约束

- `constraints` 字符串通过 z3 Python API 的 `eval()` 在受限 namespace 求值,**只允许** `z3` 模块的符号 (参考 [conjecture_library.py:34-36](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/lean/conjecture_library.py#L34-L36) 的白名单模式)
- 白名单: `And, Or, Not, Implies, If, Xor, Bool, Int, BitVec, Array, Function, Sum, Product, ForAll, Exists, Arith, ToInt, ToReal, UGE, ULE, UGT, ULT, LShR, RotateLeft, RotateRight, SignExt, ZeroExt`
- timeout 强制执行 (z3 `set_param("timeout", ms)`)
- 单次约束数 ≤ 500,变量数 ≤ 100 (防 DoS)

### 3.6 selfcheck (10 项)

```python
if __name__ == "__main__":
    # 1. SAT 简单: x And Not y → sat, model 有 x=True y=False
    # 2. UNSAT: x And Not x → unsat
    # 3. SMT 整数: x + y = 10, x > 3, y > 3 → sat
    # 4. SMT UNSAT: x + y = 10, x > 8, y > 8 → unsat
    # 5. 优化: minimize(x + y) s.t. x + y >= 10, x >= 0, y >= 0 → opt=10
    # 6. all_solutions: x != y, x,y in [0,3] → n_found=12
    # 7. verify_implication: (x > 5) ⟹ (x > 3) → holds=True
    # 8. verify_implication 反例: (x > 3) ⟹ (x > 5) → holds=False, ce={x:4}
    # 9. N-queens N=4 → sat, model 是合法布局
    # 10. timeout: 构造难解实例 (pigeonhole 10→9) → 5s 超时返 unknown
```

### 3.7 接入 RedTeamReviewer

新增 [red_team.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/red_team.py) 的 `_discrete_counterexample_scan(evidence)` 钩子,与 `_topology_scan` / `_literature_consensus_check` 平级:

- 触发条件:`evidence.hypothesis` 含可离散化的语句 (整数约束 / 有限集 / "存在"/"对所有" 量词)
- 调用 `DiscreteSMTTool.verify_implication` 找反例
- 找到反例 → `RedTeamFinding(category="hidden_assumption", severity="high", description=f"离散反例: {ce}")`
- 找不到 → 不发 finding (z3 unknown 不算通过,只是无证据)

---

## 4. P1 — 组合枚举 + OEIS 反查

### 4.1 目标

"非体系化洞察"的最大外部记忆库。给定序列前几项 → 反查 OEIS → 候选闭式/递推/生成函数。

### 4.2 文件结构

```
agent/huginn/tools/
├── discrete_oeis_tool.py             # shim
└── sci/
    └── discrete_oeis.py              # 实现 (~300 行)
```

### 4.3 工具接口

```python
class DiscreteOEISInput(BaseModel):
    action: Literal[
        "lookup",            # 序列反查
        "lookup_formula",    # 公式反查 (给定闭式找 OEIS 序列)
        "describe",          # 取 A 号的元数据 (公式/参考/作者)
        "related",           # 相关序列
    ]
    sequence: list[int] | None       # lookup: 前 N 项 (≥4 项)
    formula: str | None              # lookup_formula: sympy 表达式字符串
    a_number: str | None             # describe/related: "A000045"
    max_results: int = 5
    fetch_full: bool = False         # 是否抓 oeis.org 完整页面 (默认只用本地离线索引)
```

### 4.4 数据源策略

**Tier 1 (本地, 优先)**: 预打包 OEIS 离线索引 (~50MB, 含 A 号 + 前 10 项 + 关键词),随包分发。`fetch_full=False` 时只用本地。
**Tier 2 (在线, 兜底)**: `fetch_full=True` 时调 `https://oeis.org/search` (WebFetch 工具),解析返回 JSON。

**离线索引构建** (一次性, 不进 spec 主线):
- 从 `https://oeis.org/stripped.gz` 下载 (OEIS 官方前缀数据, ~20MB 压缩)
- 解析成 `{a_number: (first_terms, keywords, name)}` pickle
- 放 `agent/huginn/data/oeis_index.pkl`

### 4.5 action 行为

- **lookup**: 输入 `[1, 1, 2, 3, 5, 8, 13]` → 返 `[{"a_number": "A000045", "name": "Fibonacci", "formula": "F(n) = F(n-1) + F(n-2)", "terms_match": 7}]`
- **lookup_formula**: 输入 `n**2` → 返 `[{"a_number": "A000290", "name": "The squares"}]`
- **describe**: 输入 `"A000045"` → 返完整元数据 (formula / references / links / example / maple / mathematica code)
- **related**: 输入 `"A000045"` → 返 `[{"a_number": "A000030", "relation": "prefix"}]`

### 4.6 selfcheck (8 项)

```python
# 1. lookup Fibonacci 前 7 项 → A000045
# 2. lookup Catalan 前 5 项 → A000108
# 3. lookup 素数前 5 项 → A000040
# 4. lookup 不存在序列 → max_results 内无匹配
# 5. lookup_formula n**2 → A000290
# 6. describe A000045 → name 含 "Fibonacci"
# 7. related A000045 → 至少 1 条相关序列
# 8. 离线索引加载 → 索引存在且 > 100k 条目
```

---

## 5. P1 — 有限群 + 有限域计算

### 5.1 目标

离散对称性问题的根工具。与 [SymmetryTool](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/sci/symmetry_tool.py) 的边界:

| | SymmetryTool | DiscreteGroupTool |
|---|---|---|
| 对象 | 晶体学空间群 (无限群, 230 种) | 抽象有限群 (任意阶) |
| 表示 | 4x4 仿射矩阵 (连续) | 置换 / 矩阵 / 生成元 |
| 操作 | spglib (pymatgen) | sympy.combinatorics + galois |
| 验证 | 群公理 + Wyckoff | 群公理 + Sylow / 共轭类 / 子群格 |

### 5.2 文件结构

```
agent/huginn/tools/
├── discrete_group_tool.py            # shim
└── sci/
    └── discrete_group.py             # 实现 (~500 行)
```

### 5.3 工具接口

```python
class DiscreteGroupInput(BaseModel):
    action: Literal[
        "from_generators",    # 生成元 → 群
        "permutation_group",  # 置换列表 → 群
        "cyclic",             # C_n 循环群
        "dihedral",           # D_n 二面体群
        "symmetric",          # S_n 对称群
        "alternating",        # A_n 交错群
        "analyze",            # 群分析 (阶/中心/共轭类/Sylow/子群格)
        "verify_subgroup",    # 验证 H ≤ G
        "verify_homomorphism",# 验证 φ: G → H 同态
        "group_action_orbits",# 群作用轨道
        "finite_field",       # GF(p^k) 运算
    ]
    generators: list[str] | None       # from_generators: ["(1 2 3)", "(1 2)"]
    permutations: list[str] | None     # permutation_group: ["(1 2 3)", "(2 3)"]
    n: int | None                     # cyclic/dihedral/symmetric/alternating
    group_handle: str | None          # analyze/verify_subgroup 的群引用 (UUID)
    p: int | None                     # finite_field 特征
    k: int = 1                        # finite_field 扩张次数
    operation: str | None             # finite_field: "add" / "mul" / "invert" / "poly_factor"
    elements: list[int] | None        # finite_field 操作数
```

### 5.4 实现要点

- 群对象用 `sympy.combinatorics.PermutationGroup`,序列化为 handle (UUID → 缓存)
- 群分析:`order()` / `center()` / `conjugacy_classes()` / `sylow_subgroups()` / `subgroups()` (sympy 都有)
- 有限域用 `galois` 库 (轻量,纯 Python,~5MB) 或 `sympy.polys.galoistools` (慢但零新依赖)。**建议 sympy 优先,galois 备选**。
- 群作用轨道:`orbit()` / `orbit_transversal()` (sympy 有)
- 子群验证:H 是 G 的子集 + 群公理
- 同态验证:每个生成元映射到 G' 的同次幂 + 关系保持

### 5.5 ponytail 天花板声明

```python
# ponytail: sympy.combinatorics 对大群 (|G| > 10^6) 慢, 升级路径:
#   - Schreier-Sims 算法 (sympy 已有, 但接口不直接暴露)
#   - GAP / SageMath 接口 (重依赖, 不引)
# 天花板: 不做表示论 (特征标表 / 不可约表示), 升级路径: 接 sage
```

### 5.6 selfcheck (12 项)

```python
# 1. C_6 阶 = 6, 是 Abel
# 2. S_4 阶 = 24, 不是 Abel
# 3. D_4 阶 = 8, 中心 = {e, r^2}
# 4. A_4 阶 = 12, 共轭类数 = 4
# 5. from_generators ["(1 2 3)", "(1 2)"] → S_3
# 6. S_3 的 Sylow_3 子群数量 = 1
# 7. verify_subgroup: C_3 ≤ S_3 → True
# 8. verify_subgroup: D_4 ≤ S_3 → False (阶不整除)
# 9. verify_homomorphism: sign: S_3 → C_2 → True
# 10. group_action_orbits: S_3 作用于 {1,2,3} → 1 轨道
# 11. finite_field GF(5): 2+3 = 0, 2*3 = 1
# 12. finite_field GF(2^2) 元素数 = 4, x^2+x+1 不可约
```

### 5.7 与 SymmetryTool 的协作

- SymmetryTool 的 `verify_group` action 已用 sympy 4x4 矩阵验群公理
- DiscreteGroupTool 提供更强的有限群分析 (Sylow/共轭类/子群格),SymmetryTool 可路由过来
- 不修改 SymmetryTool,只在文档里加交叉引用

---

## 6. P2 — 加性组合实验台

### 6.1 目标

集合求和 / 差集 / progression 检测 / Gowers 范数。材料科学里 lattice 点阵密度/覆盖问题用得上,但优先级最低。

### 6.2 文件结构

```
agent/huginn/tools/
├── discrete_additive_tool.py         # shim
└── sci/
    └── discrete_additive.py          # 实现 (~350 行)
```

### 6.3 工具接口

```python
class DiscreteAdditiveInput(BaseModel):
    action: Literal[
        "sumset",            # A + B = {a+b : a∈A, b∈B}
        "difference_set",    # A - A
        "ap_detection",      # 算术 progression 检测 (van der Waerden 风格)
        "gowers_norm",       # Gowers U^k 范数
        "additive_energy",    # 加性能量 E(A,B,C,D)
        "schur_triple",      # Schur 三元组 x+y=z 搜索
        "ramsey_check",      # 小 Ramsey 数验证
    ]
    set_a: list[int]
    set_b: list[int] | None
    k: int | None                     # ap_detection: progression 长度; gowers_norm: U^k
    target: int | None                # ramsey_check: R(k) 下界验证
    modulo: int | None                # 有限域 Z/nZ 上做
```

### 6.4 selfcheck (8 项)

```python
# 1. sumset {1,2} + {1,2} = {2,3,4}
# 2. difference_set {1,2,4} - {1,2,4} 含 {0,±1,±2,±3}
# 3. ap_detection {1,2,3,4,5} k=3 → 至少 1 条 3-AP
# 4. ap_detection {1,2,4,5,10,11,13,14} (Behrend 构造) k=3 → 无 3-AP
# 5. gowers_norm U^2 of 常数函数 = 1
# 6. additive_energy {1,2,3} with itself = 3 (3 个 a+b=c+d 解)
# 7. schur_triple {1,2,3,4,5} → 含 (1,2,3) / (1,3,4) / (1,4,5) / (2,3,5)
# 8. ramsey_check R(3) >= 6 → True (K_6 必有单色三角形)
```

### 6.5 ponytail 天花板

```python
# ponytail: 加性组合的核心定理 (Szemerédi / Green-Tao) 没有有效算法,
#   只能做小规模实验. 升级路径: 接 MIP solver (Gurobi/CPLEX) 做大实例.
# 天花板: |A| ≤ 1000, k ≤ 5. 超出走 LLM 跨域类比.
```

---

## 7. 实施顺序与依赖

```
P0 (先做, ROI 最高):
  1. DiscreteSMTTool (z3)        ~400 行 + 10 selfcheck
  2. RedTeamReviewer 钩子接入     ~80 行 (red_team.py 改动)
  3. pyproject.toml 加 z3-solver  1 行

P1 (z3 落地后, 并行做):
  4. DiscreteOEISTool             ~300 行 + 8 selfcheck
     + 一次性 OEIS 离线索引构建脚本 (单独跑, 不进包)
  5. DiscreteGroupTool            ~500 行 + 12 selfcheck
     (sympy.combinatorics 已在依赖, 零新依赖)

P2 (优先级最低, 视用例驱动):
  6. DiscreteAdditiveTool         ~350 行 + 8 selfcheck
```

### 7.1 工具注册 (tools/__init__.py)

`_OPTIONAL_MODULES` 中按字母序插入,math tools 聚类处:

```python
("huginn.tools.discrete_smt_tool", "DiscreteSMTTool"),
("huginn.tools.discrete_oeis_tool", "DiscreteOEISTool"),
("huginn.tools.discrete_group_tool", "DiscreteGroupTool"),
("huginn.tools.discrete_additive_tool", "DiscreteAdditiveTool"),
```

### 7.2 依赖变更

- `pyproject.toml` dependencies 加 `z3-solver>=4.12` (P0)
- P1 不加新依赖 (sympy 已有)
- P2 不加新依赖

---

## 8. 验证标准

每个工具 selfcheck 全过 (assert 失败就阻塞 commit):

| 工具 | selfcheck 项 | 阻塞 commit |
|---|---|---|
| DiscreteSMTTool | 10 | 是 |
| DiscreteOEISTool | 8 | 是 |
| DiscreteGroupTool | 12 | 是 |
| DiscreteAdditiveTool | 8 | 是 |

RedTeamReviewer 钩子额外加 3 项 selfcheck:
- 离散假设触发反例搜索 → 找到反例 → severity="high"
- 连续假设不触发反例搜索 → 无 finding
- z3 unknown → 不发 finding (不算通过)

---

## 9. 不做的事 (Out of Scope)

1. **不动 VisualSCM** — 离散 SCM (value_type / 离散结构方程) 留待后续 spec
2. **不引 SageMath** — 1GB+ 重依赖,sympy + galois 够用
3. **不引 GAP** — 有限群更专业的后端,但重,sympy 够用
4. **不做 lean 形式化** — [conjecture_library.py:9](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/lean/conjecture_library.py#L9) 已显式拒绝
5. **不新建 "discrete" category** — 归 "sci" 与现有 math 工具对齐
6. **不实现表示论** (特征标表 / 不可约表示) — 升级路径,留给后续
7. **不做 OEIS 离线索引自动更新** — 一次性打包,后续手动重跑构建脚本

---

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| z3 eval 字符串注入 | 白名单 namespace + 禁用 builtins + 长度限制 |
| z3 求解超时 | 强制 timeout_ms,unknown 不算通过 |
| OEIS 在线抓取被限流 | 默认走离线索引,fetch_full 显式 opt-in |
| sympy 大群慢 | ponytail 声明天花板,|G| > 10^6 走 LLM 类比 |
| 离散工具用例不足 | P0/P1 先做,P2 视实际调用频率决定是否做 |
| "不新建 SMT 组件" 历史决策冲突 | spec §1.3 已显式回应:math tools 层新建,autoloop 层仍路由 |

---

## 11. 成功标准

1. `python -m huginn.tools.discrete_smt_tool` selfcheck 10/10 过
2. `python -m huginn.tools.discrete_oeis_tool` selfcheck 8/8 过
3. `python -m huginn.tools.discrete_group_tool` selfcheck 12/12 过
4. (P2) `python -m huginn.tools.discrete_additive_tool` selfcheck 8/8 过
5. RedTeamReviewer 钩子 selfcheck 3/3 过
6. 4 个工具注册成功,`ToolRegistry.list_tools()` 含 4 个新工具
7. 至少 1 个端到端用例:用户假设 "X 在 N={1,...,10} 上存在" → SMT 反例搜索 → 找到反例 → RedTeam 阻断

---

## 12. 后续 spec 占位

- **Spec: 离散 SCM** — VisualSCM 加 `value_type` / `range: tuple | list` / 离散结构方程
- **Spec: 形式化验证层** — 接 lean4 (当 sympy + z3 不够时)
- **Spec: 反连续化 agent** — 主动机制:用户给连续化假设,子 agent 专门找离散反例

---

## 附录 A: 与现有工具的边界

| 现有工具 | 离散工具边界 |
|---|---|
| [SymmetryTool](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/sci/symmetry_tool.py) | 晶体学空间群 (连续, 230 种) vs 抽象有限群 (任意阶) |
| [SymbolicMathTool](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/symbolic_math/tool.py) | 连续代数/微积分 vs 离散组合/数论 |
| [AutoDiffTool](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/sci/autodiff_tool.py) | 连续导数 vs 离散差分/有限群 |
| [BourbakiTool](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/bourbaki_tool.py) | 结构主义元层 vs 离散工具是具体求解器 |
| [ConjectureLibrary](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/lean/conjecture_library.py) | 命题进化环 vs 离散工具是验证 backend |
| [RedTeamReviewer](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/red_team.py) | 已有 topology/literature 钩子 vs 新增 discrete 钩子 |

## 附录 B: 历史决策溯源

- [phase_gate.py:381](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/phase_gate.py#L381) "不新建 SMT 组件" — 本 spec §1.3 部分推翻 (math tools 层新建, autoloop 层路由)
- [conjecture_library.py:9](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/lean/conjecture_library.py#L9) "用户约束: lean 太重" — 本 spec 遵守, 不引 lean
- [symmetry_tool.py:270-272](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/sci/symmetry_tool.py#L270-L272) "sympy 优先, lean 升级路径" — 本 spec 遵守同样原则
