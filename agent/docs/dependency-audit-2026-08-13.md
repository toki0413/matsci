# 沙箱依赖告警与缺失包清单 (2026-08-13)

> 来源: 服务启动日志 `/tmp/huginn_server.log` + `importlib.util.find_spec` 实测。
> 环境: `/root/.pyenv/versions/3.12.13` (Python 3.12)。

## 一、工具能力降级告警 (33 个 optional 工具)

`huginn/tools` 注册时 `probe_tool_dependencies()` 探测到顶层第三方依赖缺失,
工具仍注册但能力降级 (已有 `huginn/tools/__init__.py::probe_tool_dependencies`
做显式告警)。

| 缺失依赖 | 影响工具 | 所属 extra |
|---|---|---|
| pymatgen | vasp, symmetry, descriptor, xrd_sim, model3d, materials_database, thermo | `all` |
| ase | packing, descriptor, ml_potential, materials_database | `all` |
| matplotlib | packing, bench_infra/plot | `all` |
| rdkit | packing, vina, rdkit | `benchmark` |
| sklearn | plasma, dynamics_discovery, sklearn, gp, gnn, c2st | `all`(sklearn) |
| torch | transolver, symbolic_regression, interpretable_ml, vae, gnn, matrix, vision_describe | `all`/`ml-*` |
| torch_geometric | gnn | `all` |
| jax | autodiff | 三方 |
| openmm | openmm | 三方 |
| vina | vina | 三方 |
| cvxpy, pulp | numerical | `all` |
| spglib | symmetry | `all` |
| gudhi, ripser | tda | 三方 |
| chaospy | uq | 三方 |
| statsmodels | stat_tests, doe | 三方 |
| pandas | active_learning | `all` |
| dscribe, matminer | descriptor | `all` |
| fairchem, mace, chgnet, pynep | ml_potential | `ml-*` |
| easyocr, paddleocr, pytesseract | vision_describe | `rag`/三方 |
| trimesh | model3d | 三方 |
| thermo | thermo | 三方 |
| playwright, selenium | browser | 三方 |
| `model` | symbolic_regression | (内部 stub) |

## 二、核心服务告警 (非工具)

| 类别 | 详情 | 影响 |
|---|---|---|
| psutil 缺失 | system_health monitor disabled | 系统资源监控不可用 |
| chromadb 缺失 | KB + Codebase index 无法初始化 | 知识库/代码检索不可用 |
| MCP server 子进程 | mat-db / math-anything 报 `No module named mcp` | 外部 MCP 工具不可用 |

## 三、实测 MISSING 包 (36 个, 3.12.13 环境)

ase, chaospy, chgnet, chromadb, cvxpy, dscribe, easyocr, fairchem, gudhi,
jax, mace, matminer, matplotlib, mp_api, openmm, paddleocr, pandas, paramiko,
psutil, pulp, pymatgen, pymupdf, pynep, pypdf, pytesseract, rdkit, ripser,
sentence_transformers, sklearn, spglib, statsmodels, thermo, torch,
torch_geometric, trimesh, vina

## 四、关键澄清: MCP 报错根因

`mcp` 包在 3.12.13 环境**已装且可导入** (`import mcp` OK, site-packages/mcp)。
但 mat-db / math-anything 是两个**独立子进程**, 用系统 `/usr/bin/python3` 启动,
该系统 Python 未装 mcp → 子进程 import 失败。这是**子进程环境问题**, 非
huginn 服务缺依赖。

## 五、修复方案 (按优先级)

### P0 — 让可选能力可达 (装 `all` + 关键三方)
```bash
# 3.12 环境 (服务进程): 装全量可选组
/root/.pyenv/versions/3.12.13/bin/python -m pip install -e ".[all]"
# 补 CI 用到的额外重型依赖 (见 ci.yml test job)
/root/.pyenv/versions/3.12.13/bin/python -m pip install \
  chromadb scikit-learn scipy sympy matplotlib sentence-transformers pymupdf \
  pymatgen ase paramiko pypdf pint spglib pandas statsmodels
# ML 势能 (互斥, 二选一)
/root/.pyenv/versions/3.12.13/bin/python -m pip install -e ".[ml-mace]"   # e3nn==0.4.4
# 或
/root/.pyenv/versions/3.12.13/bin/python -m pip install -e ".[ml-fairchem]"  # e3nn>=0.5
```

### P1 — 修复 MCP 子进程 (装到系统 Python 或指定子进程解释器)
```bash
/usr/bin/python3 -m pip install mcp  # 或确保子进程用 3.12.13 解释器
```
> 更稳: 在 `.mcp.json` 的 server `command` 改为 3.12.13 的 python 路径, 或
> 给子进程装 mcp。flint-chart 用 npx (Node), 需 `npm` 环境。

### P2 — 可选/按需
- `mp-api` (db extra): 真实 Materials Project 查询才需要。
- `easyocr/paddleocr/pytesseract` (rag): OCR 场景。
- `browser` (playwright/selenium): Web 自动化。
- `pynep` (ml-nep): NEP 模型, 不在公开 PyPI, 需本地装。

## 六、附带发现 (疑似 bug, 待确认)
- `huginn.execution.orchestrator` 反复告警 `register_tool ignored: tool_registry is type, not a dict`
  (日志 99-245 行, 调 `/v1/execute` 时触发)。疑为 orchestrator 期望 dict 但拿到
  类对象。需核查 `huginn/execution/orchestrator.py`。