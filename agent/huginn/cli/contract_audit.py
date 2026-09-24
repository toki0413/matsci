"""MECE 契约审计 — 扫"宣称维度零调用者" + "惩罚项重复计数".

为什么需要: `config_audit` 登记"有哪些配置面 / 注册面", 但不检查"宣称了却没人
接". 历史上 anti_hacking 三件套曾是**定义了但零调用者**的死代码 (见
`validation/scope_authority.py` 自述), 而奖励系统里 `reconcile_r_phys` 在
`claim_reward` 与 `security/world_state` 各有一份实现 —— 这类"宣称 vs 接线"缺口
与"同一概念两份实现"正是 MECE 要抓的两类违例:

  - **collectively exhaustive 违例**: 宣称的维度零调用者 (declared but unwired).
  - **mutually exclusive 违例**: 同名跨模块重复实现 / 同一惩罚轴上叠两项.

本工具只做**静态扫描 + 少量运行时读取**并**提示候选**, 不判死: "同轴/同名"是
可疑信号, 是否真重复计数需人工判定 (例如 efficiency_discount 按"首次全对轮次"
打折, idle_turn_penalty 按"达成后多余轮次"扣分 —— 同属轮次轴但语义有别).

用法:
    python -m huginn.cli.contract_audit                  # 打印 reward + scope 审计
    python -m huginn.cli.contract_audit --reward         # 只看奖励面
    python -m huginn.cli.contract_audit --scope          # 只看授权面
    python -m huginn.cli.contract_audit --json           # 机器可读快照
    python -m huginn.cli.contract_audit --check          # 有发现则 exit 1 (供 CI 门禁)
    python -m huginn.cli.contract_audit --out docs/mece-audit.md
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 包根 (huginn/) 与仓根 (agent/, 含 tests/). 模块路径统一相对仓根, 便于把
# tests/ 的引用与生产引用分开统计.
_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parent
# 本工具自身: 扫描引用时跳过 (表里的键名是字符串常量, 但显式排除更稳).
_SELF = Path(__file__).resolve()

_REWARD_MODULE = "huginn/validation/claim_reward.py"
_SCOPE_MODULE = "huginn/validation/scope_authority.py"

# 惩罚项 → (惩罚轴, 触发信号). 同轴出现 ≥2 项即标为"重复计数候选".
# 这张表是人对语义的判断 (哪两项在惩罚同一件事), 工具只做机械的轴内计数.
_PENALTY_AXES: dict[str, tuple[str, str]] = {
    "strict_scope_reward": ("授权越界", "authorized_ratio"),
    "efficiency_discount": ("轮次", "solved_at_turn"),
    "idle_turn_penalty": ("轮次", "extra_turns"),
}

# 授权面口径的开关注册名 (来自 scope_authority 自述, 两口径独立开关).
_SCOPE_FLAGS = ("anti_hacking_reward", "intent_scope_reward")

_FLAG_IS_ENABLED = re.compile(r'is_enabled\(\s*["\']([a-z0-9_]+)["\']\s*\)')


def _iter_py(root: Path):
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or py.resolve() == _SELF:
            continue
        yield py


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None


def _ident_counter(tree: ast.Module) -> Counter:
    """统计一个模块里的符号引用 (Name / Attribute / import alias).

    `def foo` 本身是 FunctionDef(name='foo'), 不产生 Name 节点, 故不计入引用 ——
    即计数天然是"被引用次数", 不含定义处.
    """
    c: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            c[node.id] += 1
        elif isinstance(node, ast.Attribute):
            c[node.attr] += 1
        elif isinstance(node, ast.alias):
            c[node.name.split(".")[-1]] += 1
    return c


def _declared_all(path: Path) -> list[str]:
    """静态读模块的 `__all__ = [...]` (宣称的公开面), 不 import (避免重依赖)."""
    tree = _parse(path)
    if tree is None:
        return []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            return [
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    return []


def _top_level_defs(path: Path) -> set[str]:
    tree = _parse(path)
    if tree is None:
        return set()
    return {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _is_test(rel: str) -> bool:
    return (
        rel.startswith("tests/")
        or "/tests/" in rel
        or Path(rel).name.startswith("test_")
    )


def _dotted(rel: str) -> str:
    """仓根相对路径 → 点分模块名. `huginn/validation/claim_reward.py` → `huginn.validation.claim_reward`."""
    return rel[:-3].replace("/", ".")


def _module_bindings(tree: ast.Module, target: str) -> tuple[set[str], set[str]]:
    """模块级绑定: (直接 from-import 的符号名, 指向 target 模块的本地别名).

    覆盖两种真实写法:
      - `from huginn.validation.claim_reward import anti_hacking_reward` → direct
      - `from huginn.validation import claim_reward as cr` / `import ... as cr` → alias
    """
    direct: set[str] = set()
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                if mod == target:
                    # `from <target> import N` —— N 是 target 自己的符号.
                    direct.add(a.name)
                elif f"{mod}.{a.name}" == target:
                    # `from <pkg> import <target 末段> [as A]` —— 把子模块绑到别名.
                    aliases.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == target:
                    aliases.add(a.asname or a.name.split(".")[-1])
    return direct, aliases


def _scan_surface(
    rel: str, names: list[str], root: Path
) -> dict[str, dict[str, list[str]]]:
    """按**模块限定**扫描公开面符号的引用点, 归属到 prod / test / internal.

    只认 `from <module> import <name>` 与 `<alias>.<name>` (alias 已绑定到
    <module>), 不认裸同名 —— 否则 `claim_reward.reconcile_r_phys` 会被
    `security/world_state.py` 的同名函数的调用点"借"走, 误报为 wired (这正是
    跨模块同名要实现隔离的场景).

    动态加载兜底: 有的消费者用 `spec_from_file_location("...", .../"claim_reward.py")`
    把本模块按文件路径加载再以局部名调用 (`experience_archive.py` 即如此), 静态
    import 抓不到. 若某文件源码里出现本模块**文件名**字符串, 则把该文件内任意
    `<x>.<公开名>` 属性访问都记作生产引用.
    """
    target = _dotted(rel)
    base = Path(rel).name
    out: dict[str, dict[str, list[str]]] = {
        n: {"prod": [], "test": [], "internal": []} for n in names
    }
    name_set = set(names)
    for py in _iter_py(root):
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        frel = py.relative_to(root).as_posix()
        bucket = "internal" if frel == rel else ("test" if _is_test(frel) else "prod")
        direct, aliases = _module_bindings(tree, target)
        dyn = base in text  # 动态按文件路径加载本模块的兜底信号
        hits: set[str] = {n for n in name_set if n in direct}
        if aliases or dyn:
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in name_set
                    and (
                        dyn
                        or (
                            isinstance(node.value, ast.Name)
                            and node.value.id in aliases
                        )
                    )
                ):
                    hits.add(node.attr)
        if bucket == "internal":
            # 定义文件内: 裸引用即本模块自身的调用点 (无歧义).
            counter = _ident_counter(tree)
            for n in name_set:
                if counter.get(n, 0):
                    hits.add(n)
        for n in hits:
            out[n][bucket].append(frel)
    return out


def _status(name: str, refs: dict[str, list[str]]) -> str:
    r = refs.get(name, {})
    if r.get("prod"):
        return "wired"
    if r.get("internal"):
        return "internal-only"
    if r.get("test"):
        return "test-only"
    return "dead"


# ---------------------------------------------------------------------------
# 奖励面: 宣称项 vs 调用者 + 惩罚轴重叠 + 跨模块同名
# ---------------------------------------------------------------------------


def build_reward_contract(root: Path | None = None) -> dict:
    """构建奖励面 MECE 审计."""
    root = root or _REPO
    names = _declared_all(root / _REWARD_MODULE)
    refs = _scan_surface(_REWARD_MODULE, names, root)

    # 跨模块同名 (其它模块有同名顶层定义) → MECE mutually exclusive 候选.
    # 先算, 因为同名会污染归属, 需在备注里点破.
    dupes: list[dict] = []
    for name in names:
        others: list[str] = []
        for py in _iter_py(root):
            rel = py.relative_to(root).as_posix()
            if rel in (_REWARD_MODULE, _SCOPE_MODULE):
                continue
            if name in _top_level_defs(py):
                others.append(rel)
        if others:
            dupes.append({"name": name, "modules": sorted(set(others))})
    dupe_names = {d["name"] for d in dupes}

    terms: list[dict] = []
    for name in names:
        st = _status(name, refs)
        note = {
            "wired": "",
            "internal-only": "仅模块内被组合调用 (经 anti_hacking_reward 等)",
            "test-only": "只有测试引用, 生产未接",
            "dead": "零调用者 —— 宣称但未接线",
        }.get(st, "")
        if name in dupe_names:
            note = (note + "; " if note else "") + "同名跨模块, 归属已按模块限定隔离"
        terms.append(
            {
                "name": name,
                "prod": len(refs[name]["prod"]),
                "test": len(refs[name]["test"]),
                "internal": len(refs[name]["internal"]),
                "status": st,
                "note": note,
            }
        )

    # 惩罚轴重叠: 轴内 ≥2 项 → overlap.
    axis_members: dict[str, list[dict]] = defaultdict(list)
    for name, (axis, signal) in _PENALTY_AXES.items():
        if name in names:
            axis_members[axis].append({"name": name, "signal": signal})
    penalty_axes = [
        {
            "axis": ax,
            "terms": [m["name"] for m in ms],
            "overlap": len(ms) > 1,
            "signals": [m["signal"] for m in ms],
        }
        for ax, ms in sorted(axis_members.items())
    ]

    return {"terms": terms, "penalty_axes": penalty_axes, "cross_module_dupes": dupes}


_REWARD_STATUS_DOC = {
    "wired": "生产代码消费",
    "internal-only": "仅模块内组合调用",
    "test-only": "仅测试引用",
    "dead": "零调用者",
}


def render_reward_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 奖励面: 宣称项 vs 调用者 (collectively exhaustive)")
    lines.append("")
    lines.append(
        "来源: `claim_reward.py::__all__` 的宣称面. `状态 dead` = 宣称但零调用者; "
        "`internal-only` = 仅模块内被组合复用 (非死, 但无独立接线)."
    )
    lines.append("")
    lines.append("| 奖励项 | 生产调用 | 测试引用 | 模块内引用 | 状态 | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    for t in contract["terms"]:
        lines.append(
            f"| `{t['name']}` | {t['prod']} | {t['test']} | {t['internal']} "
            f"| `{t['status']}` | {t['note']} |"
        )
    lines.append("")
    lines.append("### 惩罚轴重叠 (mutually exclusive)")
    lines.append("")
    lines.append("| 惩罚轴 | 惩罚项 (触发信号) | 是否重叠 |")
    lines.append("|---|---|---|")
    for ax in contract["penalty_axes"]:
        pairs = ", ".join(f"`{n}`({s})" for n, s in zip(ax["terms"], ax["signals"]))
        flag = "⚠️ 是" if ax["overlap"] else "否"
        lines.append(f"| {ax['axis']} | {pairs} | {flag} |")
    lines.append("")
    lines.append("### 跨模块同名 (mutually exclusive)")
    lines.append("")
    if contract["cross_module_dupes"]:
        lines.append("| 名称 | 其它模块也定义 |")
        lines.append("|---|---|")
        for d in contract["cross_module_dupes"]:
            lines.append(
                f"| `{d['name']}` | {', '.join('`'+m+'`' for m in d['modules'])} |"
            )
    else:
        lines.append("— 无跨模块同名定义。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 授权面: 口径源 vs 消费点
# ---------------------------------------------------------------------------


def _flag_info(root: Path) -> dict[str, dict]:
    """读 FeatureFlags 里授权口径开关的默认值 + is_enabled 消费点 (静态)."""
    defaults: dict[str, object] = {}
    descs: dict[str, str] = {}
    try:
        from huginn.feature_flags import FeatureFlags

        defaults = dict(FeatureFlags._DEFAULTS)
        descs = dict(FeatureFlags._DESCRIPTIONS)
    except Exception:  # noqa: BLE001 - 依赖缺失时降级为仅静态
        pass

    reads: dict[str, list[str]] = defaultdict(list)
    for py in _iter_py(root):
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _FLAG_IS_ENABLED.finditer(text):
            reads[m.group(1)].append(
                f"{py.relative_to(root).as_posix()}:{text[: m.start()].count(chr(10)) + 1}"
            )

    out: dict[str, dict] = {}
    for flag in _SCOPE_FLAGS:
        out[flag] = {
            "flag": flag,
            "default": defaults.get(flag),
            "registered": flag in defaults,
            "description": descs.get(flag, ""),
            "read_points": sorted(reads.get(flag, [])),
        }
    return out


def build_scope_contract(root: Path | None = None) -> dict:
    """构建授权面 MECE 审计: 口径源 (S1/S2) 的消费点 + 独立开关."""
    root = root or _REPO
    names = _declared_all(root / _SCOPE_MODULE)
    refs = _scan_surface(_SCOPE_MODULE, names, root)

    sources: list[dict] = []
    for name in names:
        st = _status(name, refs)
        sources.append(
            {
                "name": name,
                "consumers": len(refs[name]["prod"]),
                "status": st,
                "note": "S1 合规口径" if "authorized" in name else "S2 意图口径",
            }
        )

    return {"sources": sources, "flags": _flag_info(root)}


def render_scope_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 授权面: 口径源 vs 消费点")
    lines.append("")
    lines.append(
        "来源: `scope_authority.py::__all__` (S1 合规口径 / S2 意图口径). 两口径声明"
        "为**独立开关**, 消费点在 `autoloop/engine_reflect.py::_apply_strict_scope`."
    )
    lines.append("")
    lines.append("| 口径源 | 生产消费点 | 状态 | 说明 |")
    lines.append("|---|---|---|---|")
    for s in contract["sources"]:
        lines.append(
            f"| `{s['name']}` | {s['consumers']} | `{s['status']}` | {s['note']} |"
        )
    lines.append("")
    lines.append("### 独立开关")
    lines.append("")
    lines.append("| 开关 | 已注册 | 默认 | 消费点 |")
    lines.append("|---|---|---|---|")
    for flag, info in contract["flags"].items():
        reads = ", ".join(info["read_points"][:3]) or "—"
        lines.append(
            f"| `{flag}` | {info['registered']} | `{info['default']}` | {reads} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 组合 + 门禁
# ---------------------------------------------------------------------------


def build_mece_snapshot(root: Path | None = None) -> dict:
    return {
        "reward": build_reward_contract(root),
        "scope": build_scope_contract(root),
    }


def find_issues(snap: dict) -> list[str]:
    """汇总 MECE 违例 (供 --check). 只报客观事实, 语义判定留人工."""
    issues: list[str] = []
    for t in snap["reward"]["terms"]:
        if t["status"] == "dead":
            issues.append(f"奖励项零调用者: {t['name']}")
    for ax in snap["reward"]["penalty_axes"]:
        if ax["overlap"]:
            issues.append(f"同轴惩罚候选 ({ax['axis']}): {', '.join(ax['terms'])}")
    for d in snap["reward"]["cross_module_dupes"]:
        issues.append(f"跨模块同名: {d['name']} @ {', '.join(d['modules'])}")
    for s in snap["scope"]["sources"]:
        if s["status"] == "dead":
            issues.append(f"授权口径零消费点: {s['name']}")
    return issues


def render_mece_markdown(snap: dict) -> str:
    lines: list[str] = []
    lines.append("# MECE 契约审计 (奖励面 + 授权面)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.contract_audit --out docs/mece-audit.md`."
    )
    lines.append(
        "以 MECE 两原则审计 agent 的**奖励面 / 授权面**: "
        "**collectively exhaustive** 抓「宣称维度零调用者」; "
        "**mutually exclusive** 抓「同轴惩罚叠加」与「跨模块同名重复实现」. "
        "纯静态扫描, 只提示候选, 不判死."
    )
    lines.append("")
    lines.append(render_reward_markdown(snap["reward"]))
    lines.append(render_scope_markdown(snap["scope"]))
    issues = find_issues(snap)
    lines.append("## 发现汇总")
    lines.append("")
    if issues:
        for i in issues:
            lines.append(f"- {i}")
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", action="store_true", help="只看奖励面")
    parser.add_argument("--scope", action="store_true", help="只看授权面")
    parser.add_argument("--json", action="store_true", help="输出 JSON 快照")
    parser.add_argument("--check", action="store_true", help="有 MECE 发现时 exit 1")
    parser.add_argument("--out", type=str, default="", help="写 markdown 到文件")
    args = parser.parse_args(argv)

    if args.reward and not args.scope:
        data_obj = build_reward_contract()
        md = render_reward_markdown(data_obj)
    elif args.scope and not args.reward:
        data_obj = build_scope_contract()
        md = render_scope_markdown(data_obj)
    else:
        snap = build_mece_snapshot()
        data_obj = snap
        md = render_mece_markdown(snap)

    if args.json:
        json.dump(data_obj, sys.stdout, ensure_ascii=False, indent=2)
        print()
    elif args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"wrote mece audit -> {out_path}")
    else:
        print(md)

    if args.check:
        snap = data_obj if "reward" in data_obj else build_mece_snapshot()
        issues = find_issues(snap)
        if issues:
            print(f"\nMECE 审计发现 {len(issues)} 项:", file=sys.stderr)
            for i in issues:
                print(f"  - {i}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
