"""低熵治理台账 (governance_audit): 治理门登记 + 孤儿门检测 + 确定性再生成.

治理层**自身**的熵有四种典型形态, 本模块把它们变成可观察、可门禁的事实:

1. **孤儿门** —— `tests/test_arch_*.py` 存在, 却既未接入 `ci.yml` 快速失败门,
   也未显式声明「只靠全量套件兜底」(状态未知 = 熵).
2. **悬空门** —— `ci.yml` 的 guard step 指向一个已被改名/删除的测试文件
   (门永远绿灯, 因为它根本没跑).
3. **CLI 门漏登记** —— `ci.yml` 里 `python -m huginn.<module>` 形态的治理门
   (CLI 自查 / 棘轮, 如 contract_audit 棘轮), 旧解析只认 pytest 会漏掉; 现单列.
4. **基线漂移** —— 棘轮/预算冻结值散落在 `tests/golden/*`、`huginn/` 与各测试
   常量里, 没有单一台账可读; 现由 `BUDGET_SPECS` 收成一张表.

扫描 `tests/test_arch_*.py` 门族 + `.github/workflows/ci.yml` 的两类 python guard
step (pytest 与 `python -m huginn.<module>`), 并把散落的棘轮预算/基线收成单一
台账, 渲染成 `docs/governance-state.md`.

设计 (对齐 config_audit / contract_audit):
- **纯 stdlib**: 不 import huginn.*, 不硬依赖 yaml (有则用, 无则退化行扫描),
  保证门禁与生成器都 weightless.
- **确定性输出**: 固定列序 + 字典序, 跨机逐字一致.
- **只提示不判死**: 硬判定交给 `tests/test_arch_gate_registry.py`.

重生成: ``python -m huginn.cli.governance_audit --out docs/governance-state.md``.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# huginn/cli/governance_audit.py -> parents[2] == agent/ (tests/ 与 docs/ 的宿主).
AGENT_ROOT = Path(__file__).resolve().parents[2]

# 显式声明「只靠全量套件兜底、不做快速失败门」的 arch 测试 (可见决策).
# 新增 arch 测试若不接入 ci.yml 且不在此列, 会被判为孤儿门 (强制一次明确选择).
FULL_SUITE_ONLY = frozenset(
    {
        "tests/test_arch_cleanliness.py",
        "tests/test_arch_no_import_cycles.py",
        "tests/test_arch_rag_memory_seam.py",
        "tests/test_arch_seam_integrity.py",
        "tests/test_arch_tools_direction.py",
    }
)

_ARCH_GLOB = "test_arch_*.py"
_TEST_PATH_RE = re.compile(r"tests/[A-Za-z0-9_./-]+\.py")
# 非 pytest 形态的治理门: `python -m huginn.<module>` (CLI 自查 / 棘轮). 旧解析只认
# pytest, 会漏掉 contract_audit 棘轮、policy_engine 自查这类门.
_CLI_GUARD_RE = re.compile(r"python\s+-m\s+(huginn\.[A-Za-z0-9_.]+)")


def discover_arch_tests(agent_root: Path) -> list[str]:
    """仓内 `tests/test_arch_*.py` (相对 agent_root 的 posix 路径), 字典序."""
    tests_dir = agent_root / "tests"
    if not tests_dir.is_dir():
        return []
    return sorted(f"tests/{p.name}" for p in tests_dir.glob(_ARCH_GLOB))


def _parse_guard_steps(text: str) -> list[tuple[str, str]]:
    """从 ci.yml 文本抽出 (step_name, test_path). 优先 yaml, 退化行扫描."""
    try:
        import yaml  # 可选依赖: 有则精确, 无则退化

        data = yaml.safe_load(text) or {}
        out: list[tuple[str, str]] = []
        for job in (data.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                run = step.get("run") or ""
                if "pytest" in run:
                    m = _TEST_PATH_RE.search(run)
                    if m:
                        out.append((step.get("name") or "", m.group(0)))
        if out:
            return out
    except Exception:  # noqa: BLE001 — yaml 缺失/解析失败则退化为行扫描
        logger.debug("yaml 解析 ci.yml 失败, 退化为行扫描", exc_info=True)
    # 行扫描退化: 记住最近的 `- name:`, 命中含 pytest + tests/*.py 的 run 行.
    out = []
    name = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- name:"):
            name = s[len("- name:") :].strip()
        elif "pytest" in s and "tests/" in s:
            m = _TEST_PATH_RE.search(s)
            if m:
                out.append((name, m.group(0)))
    return out


def _read_ci_text(agent_root: Path) -> str:
    """读 `<root>/../.github/workflows/ci.yml`; 不存在则空串."""
    ci_path = agent_root.parent / ".github" / "workflows" / "ci.yml"
    return ci_path.read_text(encoding="utf-8") if ci_path.exists() else ""


def _cli_guard_in_line(line: str) -> str | None:
    """一行里若有 CLI 门则返回模块名, 否则 ``None``.

    注释行 (含 `# 重生成: python -m huginn...`) 与 pytest 行都不算门 —— yaml 主路径
    与逐行 fallback **共用此判据**, 使结果只取决于 ci.yml 内容, 而非运行环境是否装了
    PyYAML (否则同一份文件会给出不同门集, 破坏确定性再生成).
    """
    s = line.strip()
    if s.startswith("#") or "pytest" in s:
        return None
    m = _CLI_GUARD_RE.search(s)
    return m.group(1) if m else None


def _parse_cli_guards(text: str) -> list[tuple[str, str]]:
    """从 ci.yml 抽出 `python -m huginn.<module>` 形态的门 `(step_name, module)`.

    这是**非 pytest** 的治理门 (CLI 自查 / 棘轮) —— 旧解析只认 pytest, 会漏掉
    contract_audit 棘轮、policy_engine 自查这类门. 只收 ``huginn.*``, 排除
    pip / pytest / uv / bandit 等.
    """
    try:
        import yaml  # 可选依赖: 有则精确, 无则退化

        data = yaml.safe_load(text) or {}
        out: list[tuple[str, str]] = []
        for job in (data.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                name = step.get("name") or ""
                for line in (step.get("run") or "").splitlines():
                    module = _cli_guard_in_line(line)
                    if module:
                        out.append((name, module))
        if out:
            return out
    except Exception:  # noqa: BLE001 — yaml 缺失/解析失败则退化为行扫描
        logger.debug("yaml 解析 ci.yml 失败, 退化为行扫描", exc_info=True)
    out = []
    name = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            name = stripped[len("- name:") :].strip()
            continue
        module = _cli_guard_in_line(line)
        if module:
            out.append((name, module))
    return out


# ── 预算登记 (单一机器可读来源) ──
# 治理棘轮的冻结预算/基线此前散落在 tests/、huginn/、tests/golden/ 各处. 这里收成
# **一张声明表**: 每项指明位置、抽取方式与消费者门, 供 governance-state.md 单点可读.
# 抽取**只读、weightless**(regex + ast, 不 import 被测模块), 以免拖慢快速失败门.
#
# phase 2 (M1′, 未落): 把测试侧冻结清单逐项提升为**权威读取**(门 from 本表 import),
# 使预算编辑收敛到单一入口; env_schema.RAW_READ_BASELINE 因层级原因保留原处、此处引用.
BUDGET_SPECS: tuple[dict, ...] = (
    {
        "id": "mece_findings",
        "name": "MECE 发现基线",
        "path": "tests/golden/mece_findings_baseline.txt",
        "kind": "file-lines",
        "owner_gate": "MECE wiring-contract guard (ratchet)",
    },
    {
        "id": "mirror_drift",
        "name": "镜像漂移预算",
        "path": "tests/test_publish_mirror_drift.py",
        "kind": "const-triple",
        "keys": ("MAX_MISSING", "MAX_EXTRA", "MAX_DIFFERING"),
        "source": "inline",
        "value": (0, 0, 0),
        "owner_gate": "publish mirror drift (only-shrink)",
        "probe": "mirror_drift_live",
        "heavy": True,
    },
    {
        "id": "env_raw_read",
        "name": "环境裸读基线",
        "path": "huginn/env_schema.py",
        "kind": "const",
        "key": "RAW_READ_BASELINE",
        "owner_gate": "env surface governance",
        "probe": "env_raw_read_live",
        "heavy": True,
        "direction": "cap",
    },
    {
        "id": "gateway_imports",
        "name": "外部直连白名单",
        "path": "tests/test_arch_single_gateway.py",
        "kind": "collection-len",
        "var": "ALLOWED_EXTERNAL_IMPORTS",
        "owner_gate": "Architecture single-gateway guard",
    },
    {
        "id": "gateway_subcommands",
        "name": "冻结委托子命令",
        "path": "tests/test_arch_single_gateway.py",
        "kind": "collection-len",
        "var": "CLI_DELEGATED_SUBCOMMANDS",
        "owner_gate": "Architecture single-gateway guard",
    },
    {
        "id": "sanctioned_plugins",
        "name": "插件准入清单",
        "path": "tests/test_arch_plugin_governance.py",
        "kind": "collection-len",
        "var": "SANCTIONED_PLUGINS",
        "owner_gate": "Plugin admission guard",
    },
    {
        "id": "hardcoded_allow",
        "name": "硬编码路径白名单",
        "path": "tests/test_arch_no_hardcoded_paths.py",
        "kind": "collection-len",
        "var": "ALLOWED_HARDCODED_MATCHES",
        "owner_gate": "No-hardcoded-machine-paths guard",
    },
    {
        "id": "mcp_wiring_allow",
        "name": "MCP 接线白名单",
        "path": "tests/test_arch_mcp_governance.py",
        "kind": "collection-len",
        "var": "_ALLOWED_WIRING_FILES",
        "owner_gate": "MCP single-gateway guard",
    },
    {
        "id": "arch_gate_floor",
        "name": "arch 门数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "ARCH_GATE_FLOOR",
        "source": "inline",
        "value": 17,
        "owner_gate": "Governance gate registry guard",
        "probe": "arch_gate_count",
        "direction": "floor",
    },
    {
        "id": "cli_guard_floor",
        "name": "CLI 门数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "CLI_GUARD_FLOOR",
        "source": "inline",
        "value": 2,
        "owner_gate": "Governance gate registry guard",
        "probe": "cli_guard_count",
        "direction": "floor",
    },
    {
        "id": "field_audit_golden",
        "name": "字段审计 golden",
        "path": "tests/golden/field_audit",
        "kind": "dir-file-count",
        "owner_gate": "MECE 字段级面",
    },
)

BUDGET_BY_ID: dict[str, dict] = {s["id"]: s for s in BUDGET_SPECS}


def budget_value(budget_id: str) -> int | tuple[int, ...]:
    """取**权威**(inline)预算的值, 供消费者门 import, 使编辑收敛到单一入口.

    非 inline 项 (值在原位 / 列表内容即契约) 不提供读数并抛 KeyError —— 避免把值
    搬进台账而丢失其自然归属 (如 env_schema 的层级、插件清单的内容).
    """
    spec = BUDGET_BY_ID[budget_id]
    if spec.get("source") != "inline":
        raise KeyError(
            f"预算 `{budget_id}` 非 inline 权威项; 其值在原位 (BUDGET_SPECS.path={spec.get('path')})"
        )
    return spec["value"]


def _extract_int_const(text: str, key: str) -> int | None:
    """抽取形如 `KEY = 123` 的整数常量 (行首锚定)."""
    m = re.search(rf"^{re.escape(key)}\s*=\s*(\d+)", text, re.M)
    return int(m.group(1)) if m else None


def _extract_collection_len(module_text: str, var: str) -> int | None:
    """用 ast 读 `var = <dict/set/frozenset 字面量>` 的元素数 (不 import 模块)."""
    try:
        tree = ast.parse(module_text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        target_name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target_name, value = node.targets[0].id, node.value
        if target_name != var or value is None:
            continue
        container = value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"frozenset", "set", "list", "tuple", "dict"}
            and value.args
        ):
            container = value.args[0]
        try:
            return len(ast.literal_eval(container))
        except (ValueError, SyntaxError):
            return None
    return None


def _frozen_value(spec: dict, agent_root: Path) -> str:
    """取某预算项的冻结值 (weightless): inline 权威项读台账声明, 其余从原位抽取."""
    if spec.get("source") == "inline":
        value = spec["value"]
        if isinstance(value, tuple):
            return "/".join(str(v) for v in value)
        return str(value)
    path = agent_root / spec["path"]
    kind = spec["kind"]
    if kind == "dir-file-count":
        if not path.is_dir():
            return "?"
        return f"{sum(1 for p in path.iterdir() if p.is_file())} 文件"
    if not path.is_file():
        return "?"
    text = path.read_text(encoding="utf-8")
    if kind == "file-lines":
        n = sum(1 for ln in text.splitlines() if ln.strip())
        return f"{n} 项"
    if kind == "const":
        v = _extract_int_const(text, spec["key"])
        return str(v) if v is not None else "?"
    if kind == "const-triple":
        vals = [_extract_int_const(text, k) for k in spec["keys"]]
        return "/".join("?" if v is None else str(v) for v in vals)
    if kind == "collection-len":
        n = _extract_collection_len(text, spec["var"])
        return f"{n} 项" if n is not None else "?"
    return "?"


def _probe_current(
    spec: dict, agent_root: Path, heavy: bool = False
) -> int | tuple[int, ...] | None:
    """实测当前值.

    默认 (``heavy=False``) 只跑 **weightless** 探针, 保持快速失败门的轻量; ``heavy=True``
    才跑需遍历全仓的重型探针 (如镜像漂移). 其余项由消费者门自行校验, 此处不重复扫描.
    """
    probe = spec.get("probe")
    if probe == "arch_gate_count":
        return len(discover_arch_tests(agent_root))
    if probe == "cli_guard_count":
        return len(_parse_cli_guards(_read_ci_text(agent_root)))
    if probe == "mirror_drift_live" and heavy:
        return _probe_mirror_drift(agent_root)
    if probe == "env_raw_read_live" and heavy:
        return _probe_env_raw_read(agent_root)
    return None


def _load_module_by_path(name: str, path: Path):
    """按文件路径加载模块并登记进 ``sys.modules`` (兼容 @dataclass). 失败返回 None.

    ``sys.modules`` 登记是必需的: ``@dataclass`` 处理时会 ``sys.modules.get(cls.__module__)``,
    未登记则 ``None.__dict__`` → 崩 (曾让镜像探针静默失效).
    """
    try:
        import importlib.util
        import sys

        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001 — 加载失败不拖垮台账
        logger.debug("按路径加载 %s 失败", path, exc_info=True)
        return None


def _probe_mirror_drift(agent_root: Path) -> tuple[int, int, int] | None:
    """实测镜像漂移 (missing/extra/differing 合计).

    复用 ``scripts/mcpb_vendor`` 的既有口径 (非重复实现), 失败返回 None.
    重型: 遍历两套镜像 + 源树, 故只在 ``--probe`` 下跑.
    """
    mod = _load_module_by_path(
        "mcpb_vendor_probe", agent_root.parent / "scripts" / "mcpb_vendor.py"
    )
    if mod is None:
        return None
    try:
        missing = extra = differing = 0
        for bundle in mod.BUNDLES:
            rep = mod.drift_report(bundle)
            missing += len(rep.missing)
            extra += len(rep.extra)
            differing += len(rep.differing)
        return (missing, extra, differing)
    except Exception:  # noqa: BLE001 — 探针失败不影响台账
        logger.debug("mirror drift 探针失败", exc_info=True)
        return None


def _probe_env_raw_read(agent_root: Path) -> int | None:
    """实测裸 ``HUGINN_*`` 读点数 (不含经 env_access 的规范读点).

    复用 ``config_audit.build_inventory`` 的既有口径 (其模块级仅 stdlib, 可按文件路径轻量
    加载), 与 ``test_env_surface_governance`` 的棘轮**同源** —— 不重复实现, 无口径分叉.
    重型: 全仓 regex 扫 (约 1s), 故只在 ``--probe`` 下跑.
    """
    mod = _load_module_by_path(
        "config_audit_probe", agent_root / "huginn" / "cli" / "config_audit.py"
    )
    if mod is None:
        return None
    try:
        inv = mod.build_inventory(agent_root / "huginn")
        return sum(
            1 for item in inv.values() for r in item["reads"] if not r.get("canonical")
        )
    except Exception:  # noqa: BLE001 — 探针失败不影响台账
        logger.debug("env 裸读探针失败", exc_info=True)
        return None


def collect_budgets(agent_root: Path, heavy: bool = False) -> list[dict]:
    """治理棘轮预算的单一台账 (只读): 冻结值 + 可测项的实测值/余量.

    ``heavy=True`` 才跑重型探针 (遍历全仓); 默认只跑 weightless 探针, 保证门禁轻量.
    """
    out: list[dict] = []
    for spec in BUDGET_SPECS:
        frozen = _frozen_value(spec, agent_root)
        current = _probe_current(spec, agent_root, heavy=heavy)
        headroom: int | None = None
        if isinstance(current, int):
            m = re.match(r"^(\d+)", frozen)
            if m:
                frozen_n = int(m.group(1))
                # floor 型: 实测应 ≥ 冻结下限, 余量 = 实测 - 下限; cap 型: 余量 = 上限 - 实测.
                headroom = (
                    current - frozen_n
                    if spec.get("direction") == "floor"
                    else frozen_n - current
                )
        out.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "path": spec["path"],
                "mode": "权威" if spec.get("source") == "inline" else "引用",
                "frozen": frozen,
                "current": current,
                "headroom": headroom,
                "owner_gate": spec["owner_gate"],
            }
        )
    return out


def audit(agent_root: Path, ci_text: str | None = None, heavy: bool = False) -> dict:
    """扫描一棵仓树, 返回治理门登记快照.

    ``ci_text`` 显式给定时不读盘 (供合成树测试); 为 None 时读
    ``<root>/../.github/workflows/ci.yml`` (真实仓: agent_root=/repo/agent).
    ``heavy=True`` 才在预算台账里跑重型探针 (见 ``collect_budgets``).
    """
    if ci_text is None:
        ci_text = _read_ci_text(agent_root)

    guard_by_test: dict[str, str] = {}
    for name, test in _parse_guard_steps(ci_text):
        guard_by_test.setdefault(test, name)

    gates: list[dict] = []
    for test, name in guard_by_test.items():
        gates.append(
            {
                "test": test,
                "arch": test.startswith("tests/test_arch_"),
                "status": "fast-fail-guard",
                "ci_step": name,
                "exists": (agent_root / test).exists(),
            }
        )

    orphans: list[str] = []
    for test in discover_arch_tests(agent_root):
        if test in guard_by_test:
            continue
        if test in FULL_SUITE_ONLY:
            gates.append(
                {
                    "test": test,
                    "arch": True,
                    "status": "full-suite-only",
                    "ci_step": None,
                    "exists": True,
                }
            )
        else:
            orphans.append(test)

    gates.sort(key=lambda g: g["test"])
    dangling = [
        {"ci_step": g["ci_step"], "test": g["test"]}
        for g in gates
        if g["status"] == "fast-fail-guard" and not g["exists"]
    ]

    # 非 pytest 形态的治理门 (CLI 自查 / 棘轮): `python -m huginn.<module>`.
    # 按模块去重 (一个门多处调用只登记一次), module 映射到仓内 .py / 包目录判存在.
    cli_guards: list[dict] = []
    seen_cli: set[str] = set()
    for name, module in _parse_cli_guards(ci_text):
        if module in seen_cli:
            continue
        seen_cli.add(module)
        rel = module.replace(".", "/")
        exists = (agent_root / f"{rel}.py").exists() or (agent_root / rel).is_dir()
        cli_guards.append({"module": module, "ci_step": name, "exists": exists})
    cli_guards.sort(key=lambda g: g["module"])
    dangling_cli = [
        {"ci_step": g["ci_step"], "module": g["module"]}
        for g in cli_guards
        if not g["exists"]
    ]
    return {
        "gates": gates,
        "orphans": sorted(orphans),
        "dangling": dangling,
        "cli_guards": cli_guards,
        "dangling_cli": dangling_cli,
        "budgets": collect_budgets(agent_root, heavy=heavy),
    }


def build_gate_registry() -> dict:
    """真实仓快照 (agent_root = 本文件所在包根); 只跑 weightless 探针, 门禁轻量."""
    return audit(AGENT_ROOT)


def _render_current(current: int | tuple[int, ...] | list[int] | None) -> str:
    if current is None:
        return "—"
    if isinstance(current, (tuple, list)):
        return "/".join(str(v) for v in current)
    return str(current)


def render_markdown(reg: dict) -> str:
    """确定性渲染 (固定列序 + 门按测试路径字典序)."""
    lines: list[str] = []
    lines.append("# 低熵治理台账 (governance state)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.governance_audit --out docs/governance-state.md`."
    )
    lines.append(
        "登记 `tests/test_arch_*.py` 门族与 `.github/workflows/ci.yml` 的 python guard step "
        "的对齐面: **孤儿门** (存在却未登记) 与 **悬空门** (CI step 指向不存在的测试) "
        "在此显形。只提示不判死; 硬判定在 `tests/test_arch_gate_registry.py`."
    )
    lines.append("")
    lines.append("## 门 (gates)")
    lines.append("")
    lines.append("| 测试 | arch | 状态 | CI step |")
    lines.append("|---|---|---|---|")
    for g in reg["gates"]:
        step = g["ci_step"] if g["ci_step"] else "—"
        if g["status"] == "fast-fail-guard" and not g["exists"]:
            step = f"{step} (悬空)"
        lines.append(f"| `{g['test']}` | {'是' if g['arch'] else '否'} | `{g['status']}` | {step} |")
    lines.append("")
    lines.append("## CLI 门 (cli gates)")
    lines.append("")
    lines.append(
        "`ci.yml` 里 `python -m huginn.<module>` 形态的治理门 (CLI 自查 / 棘轮, 非 pytest). "
        "旧解析只认 pytest 会漏掉; 现单列。**存在=否** 即悬空 CLI 门 (指向已删/改名的模块)."
    )
    lines.append("")
    lines.append("| 模块 | CI step | 存在 |")
    lines.append("|---|---|---|")
    for c in reg["cli_guards"]:
        step = c["ci_step"] if c["ci_step"] else "—"
        mark = "是" if c["exists"] else "否 (悬空)"
        lines.append(f"| `{c['module']}` | {step} | {mark} |")
    lines.append("")
    lines.append("## 预算 (budgets)")
    lines.append("")
    lines.append(
        "治理棘轮的冻结预算/基线单一台账 (源: `governance_audit.BUDGET_SPECS`). "
        "**权威** = 值在台账声明、消费者门 `budget_value()` 只读; **引用** = 值仍在其自然归属, "
        "台账只登记。实测/余量仅对 weightless 可测项给出; 其余由各自消费者门校验 (current ≤ frozen); "
        "重型探针 (如镜像漂移实测) 需 `--probe`。"
    )
    lines.append("")
    lines.append("| 预算 | 位置 | 模式 | 冻结值 | 实测 | 余量 | 消费者门 |")
    lines.append("|---|---|---|---|---|---|---|")
    for b in reg["budgets"]:
        cur = _render_current(b["current"])
        head = "—" if b["headroom"] is None else str(b["headroom"])
        lines.append(
            f"| {b['name']} | `{b['path']}` | {b['mode']} | {b['frozen']} | {cur} | {head} | {b['owner_gate']} |"
        )
    lines.append("")
    lines.append("## 缺口 (gaps)")
    lines.append("")
    if reg["orphans"]:
        lines.append(f"- 孤儿门 ({len(reg['orphans'])}): " + ", ".join(f"`{t}`" for t in reg["orphans"]))
    else:
        lines.append("- 孤儿门 (0): 无")
    if reg["dangling"]:
        items = ", ".join(f"`{d['ci_step']}` → `{d['test']}`" for d in reg["dangling"])
        lines.append(f"- 悬空门 ({len(reg['dangling'])}): {items}")
    else:
        lines.append("- 悬空门 (0): 无")
    if reg["dangling_cli"]:
        items = ", ".join(f"`{d['ci_step']}` → `{d['module']}`" for d in reg["dangling_cli"])
        lines.append(f"- CLI 悬空门 ({len(reg['dangling_cli'])}): {items}")
    else:
        lines.append("- CLI 悬空门 (0): 无")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="低熵治理台账 (governance_audit)")
    parser.add_argument("--out", default="", help="写 markdown 到文件 (缺省打印到 stdout)")
    parser.add_argument("--json", action="store_true", help="输出 JSON 快照")
    parser.add_argument(
        "--probe", action="store_true", help="额外跑重型探针 (遍历全仓, 较慢; 不入快速失败门)"
    )
    args = parser.parse_args(argv)

    reg = audit(AGENT_ROOT, heavy=args.probe)
    if args.json:
        import json

        print(json.dumps(reg, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    md = render_markdown(reg)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
