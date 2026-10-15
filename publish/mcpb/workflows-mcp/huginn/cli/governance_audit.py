"""低熵治理台账 (governance_audit): 治理门登记 + 孤儿门检测 + 确定性再生成.

治理层**自身**的熵有三种典型形态, 本模块把它们变成可观察、可门禁的事实:

1. **孤儿门** —— `tests/test_arch_*.py` 存在, 却既未接入 `ci.yml` 快速失败门,
   也未显式声明「只靠全量套件兜底」(状态未知 = 熵).
2. **悬空门** —— `ci.yml` 的 guard step 指向一个已被改名/删除的测试文件
   (门永远绿灯, 因为它根本没跑).
3. **基线漂移** —— 棘轮/预算冻结值散落在 `tests/golden/*` 与各测试常量里,
   没有单一台账可读.

扫描 `tests/test_arch_*.py` 门族 + `.github/workflows/ci.yml` 的 python guard
step + 两处棘轮基线, 渲染成 `docs/governance-state.md`.

设计 (对齐 config_audit / contract_audit):
- **纯 stdlib**: 不 import huginn.*, 不硬依赖 yaml (有则用, 无则退化行扫描),
  保证门禁与生成器都 weightless.
- **确定性输出**: 固定列序 + 字典序, 跨机逐字一致.
- **只提示不判死**: 硬判定交给 `tests/test_arch_gate_registry.py`.

重生成: ``python -m huginn.cli.governance_audit --out docs/governance-state.md``.
"""

from __future__ import annotations

import argparse
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


def _collect_baselines(agent_root: Path) -> list[dict]:
    """棘轮/预算冻结值的单一台账 (只读)."""
    out: list[dict] = []
    mece = agent_root / "tests" / "golden" / "mece_findings_baseline.txt"
    if mece.exists():
        n = sum(1 for ln in mece.read_text(encoding="utf-8").splitlines() if ln.strip())
        out.append(
            {
                "name": "MECE 发现基线",
                "path": "tests/golden/mece_findings_baseline.txt",
                "frozen": f"{n} 项发现",
            }
        )
    mirror = agent_root / "tests" / "test_publish_mirror_drift.py"
    if mirror.exists():
        txt = mirror.read_text(encoding="utf-8")

        def _budget(key: str) -> str:
            m = re.search(rf"^{key}\s*=\s*(\d+)", txt, re.M)
            return m.group(1) if m else "?"

        out.append(
            {
                "name": "镜像漂移预算",
                "path": "tests/test_publish_mirror_drift.py",
                "frozen": "missing/extra/differing = "
                f"{_budget('MAX_MISSING')}/{_budget('MAX_EXTRA')}/{_budget('MAX_DIFFERING')}",
            }
        )
    return out


def audit(agent_root: Path, ci_text: str | None = None) -> dict:
    """扫描一棵仓树, 返回治理门登记快照.

    ``ci_text`` 显式给定时不读盘 (供合成树测试); 为 None 时读
    ``<root>/../.github/workflows/ci.yml`` (真实仓: agent_root=/repo/agent).
    """
    if ci_text is None:
        ci_path = agent_root.parent / ".github" / "workflows" / "ci.yml"
        ci_text = ci_path.read_text(encoding="utf-8") if ci_path.exists() else ""

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
    return {
        "gates": gates,
        "orphans": sorted(orphans),
        "dangling": dangling,
        "baselines": _collect_baselines(agent_root),
    }


def build_gate_registry() -> dict:
    """真实仓快照 (agent_root = 本文件所在包根)."""
    return audit(AGENT_ROOT)


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
    lines.append("## 棘轮基线 (ratchet baselines)")
    lines.append("")
    lines.append("| 基线 | 冻结值 |")
    lines.append("|---|---|")
    for b in reg["baselines"]:
        lines.append(f"| {b['name']} (`{b['path']}`) | {b['frozen']} |")
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
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="低熵治理台账 (governance_audit)")
    parser.add_argument("--out", default="", help="写 markdown 到文件 (缺省打印到 stdout)")
    parser.add_argument("--json", action="store_true", help="输出 JSON 快照")
    args = parser.parse_args(argv)

    reg = build_gate_registry()
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
