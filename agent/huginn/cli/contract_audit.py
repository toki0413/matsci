"""MECE 契约审计 — 扫"宣称维度零调用者" + "惩罚项重复计数".

为什么需要: `config_audit` 登记"有哪些配置面 / 注册面", 但不检查"宣称了却没人
接". 历史上 anti_hacking 三件套曾是**定义了但零调用者**的死代码 (见
`validation/scope_authority.py` 自述), 而奖励系统里 `reconcile_r_phys` 在
`claim_reward` 与 `security/world_state` 各有一份实现 —— 这类"宣称 vs 接线"缺口
与"同一概念两份实现"正是 MECE 要抓的两类违例:

  - **collectively exhaustive 违例**: 宣称的维度零调用者 (declared but unwired).
  - **mutually exclusive 违例**: 同名跨模块重复实现 / 同一惩罚轴上叠两项.

审计七面: **奖励面 / 授权面 / 工作流面 / 模式面 / 词汇面 / 工具面 / 钩子面**. 后五面
是本工具从奖励系统外延到"agent 自身怎么跑"的同类审计:

  - **工作流面**: 执行 mode 分发面 (`phase_spec.dispatch_table` ↔ `engine_act`
    硬编码分支 ↔ planner 提示教的 MODE 候选) 三者是否穷尽一致.
  - **模式面**: agent 顶层模式词表 (prompt_builder / session / critique / core /
    `set_mode()` 实参) 是否互斥且穷尽.
  - **词汇面**: 前四面都是**按名字抽查** (查到 `explore` 才翻谁用 `explore`), 漏的是
    **结构性事实**. 本面改成**系统枚举**全仓**值域词表** (闭集枚举) 与 `X_TO_Y`
    **映射表**, 按值域 Jaccard 重叠**自动聚类**成命名空间, 报三类结构性违例:
    同名跨模块定义 / 未登记撞名 / 映射非单射 (尤其"非单射 + 声明了反向表" ⇒ 往返
    丢信息, 如 `AUTOLOOP_TO_PHASE` 把 `learn`·`validate` 共像到一个 phase).
  - **工具面**: **注册声明面** (`tools/__init__.py::_CORE_MODULES/_OPTIONAL_MODULES`
    唯一注册清单 ↔ 全仓 HuginnTool 子类声明的 `name`) 与散在仓内的**允许面**
    (`*_TOOLS` / `*_TOOL_NAMES` / `PRIMITIVES` 白名单) 是否对得上: 清单引用了静态
    解析不到的类、同一工具名被多类声明、白名单列了永不命中的死项 (无同名注册工具)
    —— 与奖励面"宣称项零调用者"同型.
  - **钩子面**: `HookManager` 的**事件契约** —— **声明面** (`hooks/__init__.py` 的
    事件常量 + `ALL_EVENTS` 权威清单) ↔ **触发面** (`trigger()` / `run_pre` /
    `run_post`) ↔ **注册面** (`register()` / `register_hook()`). 宣称的事件若零
    触发 = "声明了但永不发生"; 零注册 = "会触发但没人接" (对偶于奖励面"宣称项零调用者").

本工具只做**静态扫描 + 少量运行时读取**并**提示候选**, 不判死: "同轴/同名/词表
不一致"是可疑信号, 是否真缺陷需人工判定 (例如 efficiency_discount 按"首次全对
轮次"打折, idle_turn_penalty 按"达成后多余轮次"扣分 —— 同属轮次轴但语义有别;
fusion 模式经 set_mode('research') 复用 CSM S3 是**有意设计**, 非漏接).

用法:
    python -m huginn.cli.contract_audit                  # 打印七面审计
    python -m huginn.cli.contract_audit --reward         # 只看奖励面
    python -m huginn.cli.contract_audit --scope          # 只看授权面
    python -m huginn.cli.contract_audit --workflow       # 只看工作流面
    python -m huginn.cli.contract_audit --modes          # 只看模式面
    python -m huginn.cli.contract_audit --vocab          # 只看词汇面
    python -m huginn.cli.contract_audit --tools          # 只看工具面
    python -m huginn.cli.contract_audit --hooks          # 只看钩子面
    python -m huginn.cli.contract_audit --json           # 机器可读快照
    python -m huginn.cli.contract_audit --check          # 有发现则 exit 1 (供 CI 门禁)
    python -m huginn.cli.contract_audit --out docs/mece-audit.md
"""

from __future__ import annotations

import argparse
import ast
import functools
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


@functools.cache
def _parse(path: Path) -> ast.Module | None:
    """解析模块 AST. 结果缓存: 五面审计各自全仓扫一遍, 不缓存则同一文件重解析
    数十次 (奖励面单面即 45s, 五面合计 >60s, 挡住 `--check` 进 CI). 只读不改,
    同一进程内路径→AST 稳定."""
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
# 工作流面: 执行 mode 分发 vs planner 提示面
# ---------------------------------------------------------------------------

_EXEC_SPEC_REL = "huginn/harness/phase_spec.py"    # dispatch_table (登记面)
_ENGINE_ACT_REL = "huginn/autoloop/engine_act.py"  # hardcode elif (执行面)
_PLAN_CHECK_REL = "huginn/autoloop/plan_check.py"  # planner 提示 (教给 LLM 的面)

# dispatch_table 项: "<mode>": ["_execute_<fn>", <arg_mode>]
_DISPATCH_ENTRY_RE = re.compile(r'"([a-z_]+)"\s*:\s*\[\s*"(_execute_[a-z_]+)"')
# engine_act 硬编码分支: mode == "<mode>" (负后顾排除 arg_mode/_mode).
_ACT_BRANCH_RE = re.compile(r'(?<![a-z_])mode\s*==\s*"([a-z_]+)"')
# planner 提示里的 MODE 候选集: MODE: <coder|workflow|explore|skill>
_PLAN_MODE_LIST_RE = re.compile(r"MODE:\s*<([a-z_|]+)>")
# 模块自测块起点 (仓内惯例: 文末 `def _selfcheck()` 或 `if __name__ == "__main__":`).
# 自测里的 override fixture 是"示例配置"不是生产契约 (phase_spec `_selfcheck` 注册的
# `custom_mode` 覆盖项即此类, 会让 dispatch_table 审计误报), 扫描前一律剥掉.
_SELFTEST_RE = re.compile(
    r'^(?:'
    r'if\s+__name__\s*==\s*["\']__main__["\']\s*:'
    r'|def\s+_?self_?check\w*\s*\('
    r')',
    re.M,
)


def _read(root: Path, rel: str) -> str:
    """读源码, 剥掉 `if __name__ == "__main__":` / `_selfcheck()` 起的自测块.

    自测 fixture (如 phase_spec 注册的 `custom_mode` 覆盖项) 不是生产契约,
    不剥会让"登记面 vs 执行面"误报。仓内自测统一在文末, 故首个标记处截断安全。
    """
    try:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = _SELFTEST_RE.search(text)
    return text[: m.start()] if m else text


def build_workflow_contract(root: Path | None = None) -> dict:
    """执行 mode 分发面: dispatch_table / 硬编码分支 / planner 提示面是否穷尽一致.

    三者应 collectively exhaustive 对齐: 登记面 (dispatch_table) 支持哪些 mode,
    执行面 (engine_act elif) 就该实现哪些, 教给 LLM 的 planner 提示也该列出哪些.
    缺口 (dispatch 有而 prompt 未教) 意味着该 mode 只能靠非 prompt 路径触达.
    """
    root = root or _REPO
    dispatch = sorted(
        {m for m, _fn in _DISPATCH_ENTRY_RE.findall(_read(root, _EXEC_SPEC_REL))}
    )
    branches = sorted(set(_ACT_BRANCH_RE.findall(_read(root, _ENGINE_ACT_REL))))
    taught = [
        sorted(set(s.split("|")))
        for s in _PLAN_MODE_LIST_RE.findall(_read(root, _PLAN_CHECK_REL))
    ]
    reachable = sorted({m for s in taught for m in s})
    return {
        "dispatch": dispatch,
        "branches": branches,
        "taught": taught,
        "reachable_from_prompt": reachable,
        "missing_from_prompt": [m for m in dispatch if m not in reachable],
        "branch_matches_dispatch": dispatch == branches,
        "prompt_inconsistent": len({tuple(s) for s in taught}) > 1,
    }


def render_workflow_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 工作流面: 执行 mode 分发 vs planner 提示面")
    lines.append("")
    lines.append(
        "来源: `phase_spec.dispatch_table` (登记面) / `engine_act._execute` (硬编码分支) "
        "/ `plan_check` 的 planner 提示 (教给 LLM 的 MODE 候选). 三者应 collectively "
        "exhaustive 对齐; 登记面支持而提示未列的 mode, 只能靠非 prompt 路径触达."
    )
    lines.append("")
    lines.append("| 面 | 集合 |")
    lines.append("|---|---|")
    lines.append(
        "| dispatch_table | " + (", ".join(f"`{m}`" for m in contract["dispatch"]) or "—") + " |"
    )
    lines.append(
        "| engine_act 分支 | " + (", ".join(f"`{m}`" for m in contract["branches"]) or "—") + " |"
    )
    for i, s in enumerate(contract["taught"], 1):
        lines.append(f"| planner 提示 #{i} | " + (", ".join(f"`{m}`" for m in s) or "—") + " |")
    lines.append("")
    lines.append(
        "- dispatch_table == 硬编码分支: "
        + ("✅ 一致" if contract["branch_matches_dispatch"] else "⚠️ 不一致")
    )
    lines.append(
        "- planner 多处提示互相一致: "
        + ("✅" if not contract["prompt_inconsistent"] else "⚠️ 否")
    )
    lines.append(
        "- dispatch 支持但 planner 未教 (非 prompt 路径触达): "
        + (", ".join(f"`{m}`" for m in contract["missing_from_prompt"]) or "— 无")
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 模式面: agent 顶层模式词表一致性
# ---------------------------------------------------------------------------

# (人类可读名, 仓根相对路径, 提取模式词的块正则)
_MODE_SITE_RES: tuple[tuple[str, str, str], ...] = (
    ("session 恢复白名单", "huginn/agent/session.py", r"mode in \(([^)]*)\)"),
    (
        "critique._VALID_MODES",
        "huginn/metacog/critique.py",
        r"_VALID_MODES\s*=\s*frozenset\(\{([^}]*)\}\)",
    ),
    (
        "core._LONG_HORIZON_MODES",
        "huginn/agent/core.py",
        r"_LONG_HORIZON_MODES\s*=\s*\(([^)]*)\)",
    ),
)
_TASKMODE_COMMENT_RE = re.compile(r'mode:\s*str\s*=\s*"[a-z_]+"\s*#\s*([a-z_ /]+)')
_SET_MODE_RE = re.compile(r'set_mode\(\s*"([a-z_]+)"\s*\)')
_SET_MODE_RELS = ("huginn/cli/slash_commands.py", "huginn/routes/ws_helpers.py")
_PROMPT_MODE_KEY = "prompt_builder._MODE_INSTRUCTIONS"
_RUNTIME_MODE_KEY = "set_mode() 运行时实参"


def build_mode_contract(root: Path | None = None) -> dict:
    """agent 顶层模式词表一致性: 各来源 vocab 是否互斥且穷尽.

    "模式"概念散在多处 (prompt 段 / session 恢复白名单 / critique 校验集 /
    长程名单 / `set_mode()` 实参). 任何一处都不是全集 — 本审计把它们并排, 报出
    "有 prompt 段却无 set_mode 生产者" (死提示词) 与 "被 set_mode 却无 prompt 段"
    (无提示词的模式) 两类 exhaustive 缺口, 以及词表间 mutex 缺口.
    """
    root = root or _REPO
    vocab: dict[str, list[str]] = {}

    # prompt 面: 复用 config_audit 的 MODE_INSTRUCTIONS 解析器 (期望 huginn/ 根).
    try:
        from huginn.cli import config_audit as _cfga

        prompt_modes = sorted({n for n, _ in _cfga._parse_mode_instructions(_PKG)})
    except Exception:  # noqa: BLE001 — 解析器不可用则跳过 prompt 面
        prompt_modes = []
    if prompt_modes:
        vocab[_PROMPT_MODE_KEY] = prompt_modes

    for label, rel, pat in _MODE_SITE_RES:
        m = re.search(pat, _read(root, rel))
        if m:
            vocab[label] = sorted(set(re.findall(r'"([a-z_]+)"', m.group(1))))

    m = _TASKMODE_COMMENT_RE.search(_read(root, "huginn/memory/task_state.py"))
    if m:
        vocab["task_state 注释"] = sorted(set(re.findall(r"[a-z_]+", m.group(1))))

    runtime = sorted(
        {x for rel in _SET_MODE_RELS for x in _SET_MODE_RE.findall(_read(root, rel))}
    )
    if runtime:
        vocab[_RUNTIME_MODE_KEY] = runtime

    universe = sorted({x for ms in vocab.values() for x in ms})
    prompt = set(vocab.get(_PROMPT_MODE_KEY, []))
    rt = set(vocab.get(_RUNTIME_MODE_KEY, []))
    return {
        "vocabularies": vocab,
        "universe": universe,
        "prompt_only": sorted(prompt - rt),
        "runtime_only": sorted(rt - prompt),
        "consistent": len({frozenset(v) for v in vocab.values()}) == 1,
    }


def render_mode_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 模式面: agent 顶层模式词表一致性")
    lines.append("")
    lines.append(
        "来源: prompt 段 (`_MODE_INSTRUCTIONS`) / session 恢复白名单 / "
        "`critique._VALID_MODES` / `core._LONG_HORIZON_MODES` / `set_mode()` 实参. "
        "任何一处都不是全集即 MECE 违例 (mutex: 词表互不一致; exhaustive: 缺口)."
    )
    lines.append("")
    lines.append("| 来源 | 集合 |")
    lines.append("|---|---|")
    for src, ms in contract["vocabularies"].items():
        lines.append(f"| {src} | " + (", ".join(f"`{m}`" for m in ms) or "—") + " |")
    lines.append("")
    lines.append(
        "- 词表并集: " + (", ".join(f"`{m}`" for m in contract["universe"]) or "—")
    )
    lines.append(
        "- 有 prompt 段却无 `set_mode()` 生产者 (死提示词候选): "
        + (", ".join(f"`{m}`" for m in contract["prompt_only"]) or "— 无")
    )
    lines.append(
        "- 被 `set_mode()` 却无 prompt 段 (无提示词的模式): "
        + (", ".join(f"`{m}`" for m in contract["runtime_only"]) or "— 无")
    )
    lines.append(
        "- 各词表互相一致: " + ("✅" if contract["consistent"] else "⚠️ 否")
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 词汇面: 值域词表雷达 (系统枚举 + 自动聚类 + 撞名/分歧 + 映射单射性)
# ---------------------------------------------------------------------------
#
# 前四面 (奖励/授权/工作流/模式) 都是**按名字抽查**: 查到 `explore` 就翻谁用
# `explore`. 这种打法漏的是**结构性事实** —— 同一个概念被几套词表切分、两套命名
# 之间的映射不是单射、同一名字在两个模块各定义一份. 本面改成**系统枚举**:
#
#   1. AST 扫全仓的**值域词表** (闭集枚举) 与 `X_TO_Y` **映射表**;
#   2. 按值域 Jaccard 重叠**自动聚类**成"命名空间", 不靠人指定词表名;
#   3. 报三类结构性违例:
#      - 同名跨模块定义 (mutually exclusive: 一个概念两份定义);
#      - 未登记撞名 (mutually exclusive: 同一词横跨两个命名空间);
#      - 映射非单射 + 声明了反向表 (往返丢信息, 如 `learn`/`validate` 共像).
#
# 只收**闭集枚举**: `Literal` / Enum 子类 / `frozenset` / 全大写 tuple·set·list.
# **dict 字面量排除** —— 其键名多是 payload schema (`{"ts","success",...}`) 不是
# 词表, 收进来噪声压过信号 (模式面的 `_MODE_INSTRUCTIONS` 键词表已由模式面覆盖).

_VOCAB_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
_VOCAB_MAP_RE = re.compile(r"^[A-Z][A-Z0-9_]*_TO_[A-Z0-9_]+$")
_VOCAB_ALL_CAPS_RE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
_VOCAB_KWARG_RE = re.compile(r"(modes|phases|actions|kinds)$")

# 通用 dunder: 每个模块天然重复, 不算"同名定义".
_VOCAB_STOP_NAMES = frozenset({"__slots__", "__all__", "__match_args__"})
# 自然语言虚词: 只出现在 NLP 词表 (如 `_NEGATIVE_WORDS`) 里. 簇内命中任一即判该簇
# 为"自然语言表"而非闭集枚举, 退出撞名/分歧统计.
_VOCAB_STOP_WORDS = frozenset(
    {
        "and", "for", "from", "if", "any", "a", "to", "of", "in", "on", "the",
        "is", "it", "or", "not", "no", "yes", "all", "more", "less", "as", "at",
        "by", "do", "be", "an", "we", "you", "i", "this", "that", "with", "but",
        "so", "then", "when", "what", "how", "why", "which", "who", "there",
        "here", "out", "up",
    }
)
# **命名空间登记表**: 登记为允许跨命名空间共用的词 (通用生命周期/状态词).
# 这些词在多个状态枚举里合法复用, 不算"同名不同物"; 未登记的词才报撞名候选.
_VOCAB_SHARED_TOKENS = frozenset(
    {
        "none", "error", "success", "failed", "pending", "running", "completed",
        "status", "approved", "rejected", "denied", "confirmed", "verified",
        "refuted", "superseded", "blocked", "in_progress", "ok", "unknown",
        "ready", "done", "active", "inactive", "enabled", "disabled", "warn",
        "warning", "info", "critical", "high", "low", "medium", "max",
        "minimum", "maximum", "default", "other", "mixed", "balanced",
    }
)
_VOCAB_MAX_CLOSED = 12  # 闭集规模上限: 超过多为词表/包清单, 退化为噪声
_VOCAB_CLUSTER_THR = 0.5  # Jaccard 阈值: 两站点值域重叠过半即同簇


def _vocab_is_token(s: object) -> bool:
    return isinstance(s, str) and bool(_VOCAB_TOKEN_RE.match(s))


def _vocab_str_consts(node: ast.AST) -> list[str] | None:
    """从 set/tuple/list 字面量取全字符串值; 含非字符串元素则弃 (混合类型非词表)."""
    if not isinstance(node, ast.Set | ast.Tuple | ast.List):
        return None
    vals: list[str] = []
    for e in node.elts:
        if isinstance(e, ast.Constant) and _vocab_is_token(e.value):
            vals.append(e.value)
        else:
            return None
    return vals


def _vocab_declared(name: str, kind: str) -> bool:
    """是否为"声明的枚举" (相对局部临时集合).

    只对声明枚举做撞名统计: `names_in = [...]` 这类局部集合的名字是变量名不是
    概念词, 混进来全是噪声.
    """
    base = name.split(" (")[0]
    if kind in {"Literal", "frozenset", "Enum"}:
        return True
    return kind in {"Tuple", "Set", "List"} and bool(_VOCAB_ALL_CAPS_RE.match(base))


def _vocab_enum_values(cls: ast.ClassDef) -> list[str] | None:
    """从 Enum/StrEnum/IntEnum 子类取全字符串成员值."""
    bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    if not bases & {"Enum", "StrEnum", "IntEnum"}:
        return None
    vals = [
        s.value.value
        for s in cls.body
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        and isinstance(s.value, ast.Constant)
        and _vocab_is_token(s.value.value)
    ]
    return vals or None


def _vocab_assign_target(node: ast.AST) -> tuple[ast.Name | None, ast.AST | None]:
    """取 `X = v` / `X: T = v` 的 (目标名, 右值). 注解式赋值同样要收."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0], node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target, node.value
    return None, None


def _vocab_values(value: ast.AST) -> tuple[list[str], str] | None:
    """从赋值右值提值域词表: `frozenset/set/tuple/list({...})` 或 `Literal[...]`."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.args
        and value.func.id in {"frozenset", "set", "tuple", "list"}
    ):
        vals = _vocab_str_consts(value.args[0])
        return (vals, value.func.id) if vals else None
    if isinstance(value, ast.Subscript) and ast.unparse(value.value).endswith("Literal"):
        sl = value.slice
        elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        vals = [e.value for e in elts if isinstance(e, ast.Constant) and _vocab_is_token(e.value)]
        if vals and len(vals) == len(elts):
            return vals, "Literal"
    vals = _vocab_str_consts(value)
    return (vals, type(value).__name__) if vals else None


def _vocab_mapping(name: str, value: ast.AST) -> dict[str, str] | None:
    """`X_TO_Y = {..}` 字面量映射表 → {键源码, 值源码}. 推导式派生表不在此列."""
    if not _VOCAB_MAP_RE.match(name) or not isinstance(value, ast.Dict) or not value.keys:
        return None
    out: dict[str, str] = {}
    for k, v in zip(value.keys, value.values):
        try:
            out[ast.unparse(k)] = ast.unparse(v)
        except Exception:  # noqa: BLE001 - 无法源码化的键值对跳过整表
            return None
    return out or None


def scan_vocabularies(root: Path) -> tuple[list[dict], list[dict]]:
    """AST 扫全仓值域词表站点 + 映射表. 跳过 tests/ (fixture 词表非生产契约)."""
    sites: list[dict] = []
    mappings: list[dict] = []
    for py in _iter_py(root):
        rel = py.relative_to(root).as_posix()
        if _is_test(rel):
            continue
        tree = _parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                vals = _vocab_enum_values(node)
                if vals and len(vals) >= 3:
                    sites.append(
                        {"rel": rel, "name": f"class {node.name}", "kind": "Enum", "line": node.lineno, "values": vals}
                    )
                continue
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg and _VOCAB_KWARG_RE.search(kw.arg):
                        vals = _vocab_str_consts(kw.value)
                        if vals and len(vals) >= 3:
                            sites.append({"rel": rel, "name": kw.arg, "kind": "kwarg", "line": node.lineno, "values": vals})
                continue
            target, value = _vocab_assign_target(node)
            if target is None or value is None:
                continue
            m = _vocab_mapping(target.id, value)
            if m:
                mappings.append({"rel": rel, "name": target.id, "line": node.lineno, "entries": m})
            cand = _vocab_values(value)
            if cand and len(cand[0]) >= 3:
                vals, kind = cand
                sites.append({"rel": rel, "name": target.id, "kind": kind, "line": node.lineno, "values": vals})

    seen: set[tuple[str, str, int]] = set()
    uniq: list[dict] = []
    for s in sites:
        key = (s["rel"], s["name"], s["line"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    for s in uniq:
        s["declared"] = _vocab_declared(s["name"], s["kind"])
    return uniq, mappings


def _assigned_names(root: Path) -> set[str]:
    """全仓模块级赋值目标名 (含推导式派生的反向映射表), 供"反向表是否存在"判断."""
    names: set[str] = set()
    for py in _iter_py(root):
        tree = _parse(py)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _vocab_jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def cluster_vocabularies(sites: list[dict], thr: float = _VOCAB_CLUSTER_THR) -> list[list[int]]:
    """按值域 Jaccard 重叠 union-find 聚类: 重叠过半即视为同一"命名空间"."""
    n = len(sites)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        si = set(sites[i]["values"])
        for j in range(i + 1, n):
            if _vocab_jaccard(si, set(sites[j]["values"])) >= thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return list(comps.values())


def build_vocabulary_contract(root: Path | None = None) -> dict:
    """词汇面雷达: 枚举→聚类→三类结构性违例 + 映射单射性."""
    root = root or _REPO
    sites, mappings = scan_vocabularies(root)
    clusters = cluster_vocabularies(sites)

    # 闭集簇: 全成员都是声明枚举 + 规模不超上限 + 不含自然语言虚词.
    closed: list[list[int]] = []
    for comp in clusters:
        toks = {v for i in comp for v in sites[i]["values"]}
        if not all(sites[i]["declared"] for i in comp):
            continue
        if max(len(sites[i]["values"]) for i in comp) > _VOCAB_MAX_CLOSED:
            continue
        if toks & _VOCAB_STOP_WORDS:
            continue
        closed.append(comp)

    # 1. 同名跨模块定义 (一个概念两份定义).
    # 只收**声明过的**词表: 函数内同名局部元组 (如 `required = ("location", ...)`)
    # 与模块级 frozenset 撞名不是"一个概念两份定义", 收进来是噪声.
    by_name: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sites):
        by_name[s["name"]].append(i)
    dup_defs: list[dict] = []
    for name, idxs in sorted(by_name.items()):
        if name.split(" (")[0] in _VOCAB_STOP_NAMES:
            continue
        if len({sites[i]["rel"] for i in idxs}) < 2:
            continue
        if not all(sites[i]["declared"] for i in idxs):
            continue
        sets = [frozenset(sites[i]["values"]) for i in idxs]
        dup_defs.append(
            {
                "name": name,
                "same_values": len(set(sets)) == 1,
                "sites": [
                    {"rel": sites[i]["rel"], "line": sites[i]["line"], "size": len(set(sites[i]["values"]))}
                    for i in idxs
                ],
            }
        )

    # 2. 未登记撞名 (同一词横跨 >=2 个闭集簇).
    tok2cluster: dict[str, set[int]] = defaultdict(set)
    for ci, comp in enumerate(closed):
        for i in comp:
            for t in set(sites[i]["values"]):
                tok2cluster[t].add(ci)
    collisions: list[dict] = []
    for tok, cis in sorted(tok2cluster.items()):
        if len(cis) < 2 or tok in _VOCAB_SHARED_TOKENS:
            continue
        spans = sorted(
            {f"{sites[closed[c][0]]['rel']}::{sites[closed[c][0]]['name']}" for c in cis}
        )
        collisions.append({"token": tok, "spans": spans})

    # 3. 簇内分歧 (同簇成员值域不等: 子集/超集漂移).
    divergence: list[dict] = []
    for ci, comp in enumerate(closed):
        if len(comp) < 2:
            continue
        sets = {frozenset(sites[i]["values"]) for i in comp}
        if len(sets) == 1:
            continue
        union = set().union(*[set(sites[i]["values"]) for i in comp])
        divergence.append(
            {
                "cluster": ci,
                "union": sorted(union),
                "members": [
                    {
                        "rel": sites[i]["rel"],
                        "line": sites[i]["line"],
                        "name": sites[i]["name"],
                        "extra": sorted(set(sites[i]["values"]) - union),
                        "missing": sorted(union - set(sites[i]["values"])),
                    }
                    for i in comp
                ],
            }
        )

    # 4. 映射表: 单射性 + 是否声明了反向表 (非单射 + 有反向表 ⇒ 往返丢信息).
    assigned = _assigned_names(root)
    map_reports: list[dict] = []
    for m in mappings:
        vals = list(m["entries"].values())
        a, b = m["name"].split("_TO_")
        reverse = f"{b}_TO_{a}"
        coll: dict[str, list[str]] = defaultdict(list)
        for k, v in m["entries"].items():
            coll[v].append(k)
        map_reports.append(
            {
                "name": m["name"],
                "rel": m["rel"],
                "entries": len(m["entries"]),
                "injective": len(set(vals)) == len(vals),
                "collisions": {v: ks for v, ks in sorted(coll.items()) if len(ks) > 1},
                "reverse_name": reverse,
                "reverse_present": reverse in assigned,
            }
        )

    return {
        "site_count": len(sites),
        "cluster_count": len(clusters),
        "closed_cluster_count": len(closed),
        "duplicate_defs": dup_defs,
        "collisions": collisions,
        "divergence": divergence,
        "mappings": map_reports,
    }


def render_vocabulary_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 词汇面: 值域词表雷达 (系统枚举 + 自动聚类)")
    lines.append("")
    lines.append(
        f"系统枚举**值域词表** (闭集枚举: `Literal`/Enum/`frozenset`/全大写元组) "
        f"共 {contract['site_count']} 站点, 按值域 Jaccard 重叠自动聚成 "
        f"{contract['cluster_count']} 簇 (其中闭集簇 {contract['closed_cluster_count']}). "
        "三类结构性违例: 同名跨模块定义 / 未登记撞名 / 映射非单射."
    )
    lines.append("")
    lines.append("### 同名跨模块定义 (mutually exclusive)")
    lines.append("")
    if contract["duplicate_defs"]:
        lines.append("| 名称 | 值域一致 | 定义点 (规模) |")
        lines.append("|---|---|---|")
        for d in contract["duplicate_defs"]:
            flag = "✅ 同" if d["same_values"] else "⚠️ 异"
            sites = ", ".join(f"`{s['rel']}:{s['line']}`({s['size']})" for s in d["sites"])
            lines.append(f"| `{d['name']}` | {flag} | {sites} |")
    else:
        lines.append("— 无。")
    lines.append("")
    lines.append("### 未登记撞名 (同一词横跨两个命名空间)")
    lines.append("")
    if contract["collisions"]:
        lines.append("| 词 | 出现于 (命名空间代表站点) |")
        lines.append("|---|---|")
        for c in contract["collisions"]:
            lines.append(f"| `{c['token']}` | {', '.join('`'+s+'`' for s in c['spans'])} |")
    else:
        lines.append("— 无。")
    lines.append("")
    lines.append("### 簇内分歧 (同簇成员值域不等)")
    lines.append("")
    if contract["divergence"]:
        for d in contract["divergence"]:
            lines.append(f"- 簇 {d['cluster']} (并集 {len(d['union'])} 词):")
            for m in d["members"]:
                tail = ""
                if m["missing"]:
                    tail += f" 缺 {', '.join('`'+x+'`' for x in m['missing'])}"
                if m["extra"]:
                    tail += f" 多 {', '.join('`'+x+'`' for x in m['extra'])}"
                lines.append(f"  - `{m['rel']}:{m['line']}` {m['name']}{tail}")
    else:
        lines.append("— 无。")
    lines.append("")
    lines.append("### 映射表 (非单射 + 有反向表 ⇒ 往返丢信息)")
    lines.append("")
    if contract["mappings"]:
        lines.append("| 映射表 | 条目 | 单射 | 反向表 | 共像 |")
        lines.append("|---|---|---|---|---|")
        for m in contract["mappings"]:
            coll = "; ".join(f"`{v}`←{ks}" for v, ks in m["collisions"].items()) or "—"
            rev = f"`{m['reverse_name']}`" + ("" if m["reverse_present"] else " (无)")
            lines.append(
                f"| `{m['name']}` | {m['entries']} | {'是' if m['injective'] else '⚠️ 否'} "
                f"| {rev} | {coll} |"
            )
    else:
        lines.append("— 无。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具面: 注册声明面 vs 允许面
# ---------------------------------------------------------------------------
#
# 前五面审"奖励/授权/工作流/模式/词汇", 这一面审**工具**自身的契约:
#
#   - **注册声明面**: `tools/__init__.py::_CORE_MODULES/_OPTIONAL_MODULES` 是唯一
#     注册清单 (runtime 逐条 import 后 `ToolRegistry.register(cls(**kwargs))`). 清单
#     引用的类若静态解析不到 (打错模块/类名 ⇒ 该工具静默消失), 或两个类声明同一
#     工具名 (后者覆盖前者), 即 exhaustive/mutex 违例.
#   - **允许面**: 散在仓内的 `*_TOOLS` / `*_TOOL_NAMES` / `PRIMITIVES` 白名单
#     (权限放行 / 危险拦截 / 昂贵工具节流 / pi 原语 …). 每条都**宣称**它列的是工具
#     名; 某项若在任何注册工具里都不存在, 就是**永不命中的死项** —— 与奖励面
#     "宣称项零调用者"同型.
#
# 归属口径与奖励面一致: 白名单"是否被消费"按**模块限定**统计 (同名白名单
# `_EXPENSIVE_TOOLS` 在三个模块各定义一份, 裸名扫描会互相借引用).
#
# 噪声控制: 白名单按**命名空间**分类 —— 与注册名零重叠的 (如 MCP 外部工具名
# `_HIGH_VALUE_MCP_TOOLS`) 整表判为"外部命名空间", 不参与死项判定; 有重叠的才逐项
# 判. 死项里若存在裸名↔`_tool` 别名 (`vasp`↔`vasp_tool`, `grep_tool`↔`grep`) 记为
# **别名**而非死项 —— 只报纯死项.

_TOOL_SPEC_REL = "huginn/tools/__init__.py"
_TOOL_SPEC_LISTS = ("_CORE_MODULES", "_OPTIONAL_MODULES")
# 允许表名: 全大写 (可带下划线前缀), 以 *_TOOLS / *_TOOL_NAMES 收尾, 或裸 PRIMITIVES.
_ALLOWLIST_RE = re.compile(
    r"^(?:_?[A-Z][A-Z0-9_]*_(?:TOOLS|TOOL_NAMES)|[A-Z][A-Z0-9_]*TOOLS|PRIMITIVES)$"
)


def _tool_declared_name(cls: ast.ClassDef) -> str | None:
    """取工具类声明的 `name`. 两种真实风格:
    (a) 类属性 `name = "..."` / `name: str = "..."`; (b) `@property def name` 返回
    常量 (`WebSearchTool` 即后者, 只看类属性会把它漏成"注册清单里解析不到的类")."""
    for st in cls.body:
        if (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
            and st.targets[0].id == "name"
            and isinstance(st.value, ast.Constant)
            and isinstance(st.value.value, str)
        ):
            return st.value.value
        if (
            isinstance(st, ast.AnnAssign)
            and isinstance(st.target, ast.Name)
            and st.target.id == "name"
            and isinstance(st.value, ast.Constant)
            and isinstance(st.value.value, str)
        ):
            return st.value.value
    for st in cls.body:
        if isinstance(st, ast.FunctionDef) and st.name == "name":
            for sub in ast.walk(st):
                if (
                    isinstance(sub, ast.Return)
                    and isinstance(sub.value, ast.Constant)
                    and isinstance(sub.value.value, str)
                ):
                    return sub.value.value
    return None


def _tool_classes(root: Path) -> dict[str, str]:
    """扫全仓 HuginnTool 子类 → {类名: 工具名}. 跳过 tests/ (fixture 非生产契约)."""
    out: dict[str, str] = {}
    for py in _iter_py(root):
        rel = py.relative_to(root).as_posix()
        if _is_test(rel):
            continue
        tree = _parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [ast.unparse(b).split("[")[0].split(".")[-1] for b in node.bases]
            if not any(b.endswith("Tool") for b in bases):
                continue
            nm = _tool_declared_name(node)
            if nm:
                out[node.name] = nm
    return out


def _tool_specs(root: Path) -> list[dict]:
    """读注册清单 `_CORE_MODULES`/`_OPTIONAL_MODULES` 的 (module, class) 条目."""
    tree = _parse(root / _TOOL_SPEC_REL)
    if tree is None:
        return []
    out: list[dict] = []
    for node in tree.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id not in _TOOL_SPEC_LISTS
            or not isinstance(node.value, ast.List)
        ):
            continue
        for e in node.value.elts:
            if (
                isinstance(e, ast.Tuple)
                and len(e.elts) == 2
                and all(
                    isinstance(x, ast.Constant) and isinstance(x.value, str)
                    for x in e.elts
                )
            ):
                out.append(
                    {
                        "list": node.targets[0].id,
                        "module": e.elts[0].value,
                        "class": e.elts[1].value,
                    }
                )
    return out


def _tool_allowlist_values(node: ast.AST) -> list[str] | None:
    """从 set/tuple/list 字面量 (含 `frozenset({...})` 调用) 取全字符串值域.

    含非字符串元素则弃 (混合类型不是工具名表). 值做 `strip()` —— 仓内确有
    `"qe "` 这类带尾空格的条目, 归一后才好判是否为裸名别名.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
        node = node.args[0]
    if not isinstance(node, ast.Set | ast.Tuple | ast.List):
        return None
    vals: list[str] = []
    for e in node.elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            vals.append(e.value.strip())
        else:
            return None
    return vals or None


def _tool_allowlists(root: Path) -> list[dict]:
    """扫全仓工具名白名单 (`*_TOOLS` / `*_TOOL_NAMES` / `PRIMITIVES`)."""
    out: list[dict] = []
    for py in _iter_py(root):
        rel = py.relative_to(root).as_posix()
        if _is_test(rel):
            continue
        tree = _parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            target: ast.Name | None = None
            value: ast.AST | None = None
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target, node.value
            if target is None or not _ALLOWLIST_RE.match(target.id):
                continue
            vals = _tool_allowlist_values(value)
            if vals:
                out.append(
                    {"name": target.id, "rel": rel, "line": node.lineno, "values": vals}
                )
    return out


def _resolve_relative(rel: str, level: int, mod: str) -> str:
    """import 目标解析成绝对点分模块名. `level == 0` 即绝对 import, 原样返回;
    否则按文件所在包 + 点数回退 (`from ..pkg import X`, level = 点数)."""
    if level == 0:
        return mod
    parts = _dotted(rel).split(".")[:-1]
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    return ".".join(parts + ([mod] if mod else []))


def _tool_consumers(
    root: Path, sites: list[dict]
) -> tuple[dict[tuple[str, str], int], set[tuple[str, str]]]:
    """单遍扫描: 每个白名单的消费点. 返回 (外部生产消费计数, 仅定义文件内被引用集).

    只认 `from <module> import <NAME>` 与 `<mod>.<NAME>` (mod 已绑定到目标模块),
    与奖励面同一归属口径 —— 三处同名 `_EXPENSIVE_TOOLS` 不会互相借引用.
    定义文件内的引用单独记 (internal-only), 与"零引用死表"区分开.
    """
    counts: dict[tuple[str, str], int] = {(s["rel"], s["name"]): 0 for s in sites}
    internal: set[tuple[str, str]] = set()
    by_rel: dict[str, set[str]] = defaultdict(set)
    # 定义行: `A_TOOLS = {...}` 的赋值目标本身就是一个 Name 节点, 会被
    # `_ident_counter` 计入 —— 不排除定义行的话每条白名单都至少"自引用"一次,
    # "dead" (零引用) 永不成立. 这里按行号把定义处剔除.
    def_line = {(s["rel"], s["name"]): s["line"] for s in sites}
    for s in sites:
        by_rel[s["rel"]].add(s["name"])

    for py in _iter_py(root):
        rel = py.relative_to(root).as_posix()
        tree = _parse(py)
        if tree is None:
            continue
        for nm in by_rel.get(rel, ()):
            ln = def_line[(rel, nm)]
            # 引用可以是裸名 (`A_TOOLS`) 也可以是属性 (`self._ALWAYS_ON_TOOLS`) ——
            # 只看 ast.Name 会把类属性风格的白名单误判成零引用死表.
            refs = [
                n
                for n in ast.walk(tree)
                if (isinstance(n, ast.Name) and n.id == nm)
                or (isinstance(n, ast.Attribute) and n.attr == nm)
            ]
            if any(n.lineno != ln for n in refs):
                internal.add((rel, nm))
        if _is_test(rel):
            continue
        consumed: set[tuple[str, str]] = set()
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = _resolve_relative(rel, node.level, node.module or "")
                for a in node.names:
                    aliases[a.asname or a.name] = f"{mod}.{a.name}"
                    consumed.add((mod, a.name))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    aliases[a.asname or a.name.split(".")[-1]] = a.name
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                mod = aliases.get(node.value.id)
                if mod:
                    consumed.add((mod, node.attr))
        for s in sites:
            if s["rel"] != rel and (_dotted(s["rel"]), s["name"]) in consumed:
                counts[(s["rel"], s["name"])] += 1
    return counts, internal


def _tool_alias(entry: str, registered: set[str]) -> bool:
    """裸名↔`_tool` 别名 (`vasp`↔`vasp_tool`, `grep_tool`↔`grep`) 不算死项."""
    return f"{entry}_tool" in registered or (
        entry.endswith("_tool") and entry[:-5] in registered
    )


def build_tool_contract(root: Path | None = None) -> dict:
    """工具面: 注册声明面 ↔ 允许面是否对得上."""
    root = root or _REPO
    classes = _tool_classes(root)
    specs = _tool_specs(root)
    registered = set(classes.values())

    spec_classes = {s["class"] for s in specs}
    unresolved = sorted(
        (s for s in specs if s["class"] not in classes),
        key=lambda d: (d["class"], d["module"]),
    )

    name2classes: dict[str, list[str]] = defaultdict(list)
    for cls, nm in classes.items():
        name2classes[nm].append(cls)
    dup_names = [
        {"name": nm, "classes": sorted(cs)}
        for nm, cs in sorted(name2classes.items())
        if len(cs) > 1
    ]

    sites = _tool_allowlists(root)
    consumers, internal = _tool_consumers(root, sites)
    allowlists: list[dict] = []
    covered: set[str] = set()
    for s in sites:
        vals = s["values"]
        matched = [v for v in vals if v in registered]
        raw_dead = [v for v in vals if v not in registered]
        # 与注册名零重叠且无任何裸名↔`_tool` 别名对应 ⇒ 整表是外部命名空间
        # (MCP 外部工具名等), "无同名注册工具"是预期而非缺陷, 故不发死项/别名判定.
        # 只按"零重叠"判会把纯别名表 (如 `{"vasp","lammps"}` 全写成裸名) 误当外部,
        # 连真死项一起吞掉 —— 故要求"连别名对应都没有"才算外部.
        aliases = [v for v in raw_dead if _tool_alias(v, registered)]
        external = not matched and not aliases
        # 工具名从不含空格; 表里出现空格条目 (如 `"quantum espresso"`) ⇒ 这是
        # prompt 关键词表 (匹配用户输入文本), 不是工具名白名单, 不做死项判定.
        prose = any(" " in v for v in vals)
        no_judge = external or prose
        if no_judge:
            aliases = []
        covered.update(matched)
        n = consumers.get((s["rel"], s["name"]), 0)
        allowlists.append(
            {
                "name": s["name"],
                "rel": s["rel"],
                "line": s["line"],
                "size": len(vals),
                "matched": len(matched),
                "namespace": "keywords"
                if prose
                else ("external" if external else "registry"),
                "aliases": aliases,
                "phantoms": [] if no_judge else [v for v in raw_dead if v not in aliases],
                "consumers": n,
                "status": "wired"
                if n
                else ("internal-only" if (s["rel"], s["name"]) in internal else "dead"),
            }
        )

    return {
        "registry_size": len(registered),
        "spec_count": len(specs),
        "unresolved_specs": unresolved,
        "duplicate_names": dup_names,
        "unregistered_classes": sorted(cls for cls in classes if cls not in spec_classes),
        "allowlists": allowlists,
        "covered_registry_names": len(covered),
    }


def render_tool_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 工具面: 注册声明面 vs 允许面")
    lines.append("")
    lines.append(
        f"注册声明面: `tools/__init__.py` 注册清单 {contract['spec_count']} 条 → 全仓 "
        f"HuginnTool 子类声明的工具名 {contract['registry_size']} 个. 允许面: 全仓工具名"
        f"白名单 {len(contract['allowlists'])} 张. **死项** = 白名单里无同名工具声明的条目 "
        "(永不命中), **别名** = 裸名↔`_tool` 对应项 (非死项). 两类表不做死项判定: "
        "与工具名零重叠的**外部命名空间** (MCP 外部工具名等), 以及含空格条目的 "
        "**关键词表** (匹配用户 prompt 文本, 不是工具名)."
    )
    lines.append("")
    lines.append("| 允许表 | 位置 | 条目 | 命中注册名 | 命名空间 | 状态 | 外部消费 | 死项 | 别名 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for a in contract["allowlists"]:
        lines.append(
            f"| `{a['name']}` | `{a['rel']}:{a['line']}` | {a['size']} | {a['matched']} "
            f"| {a['namespace']} | `{a['status']}` | {a['consumers']} | {len(a['phantoms'])} "
            f"| {len(a['aliases'])} |"
        )
    lines.append("")
    lines.append(
        f"- 被白名单覆盖的工具名: {contract['covered_registry_names']} / "
        f"{contract['registry_size']}"
    )
    lines.append(
        "- 外部命名空间 (整表与注册名零重叠, 不判死项): "
        + (
            ", ".join(
                f"`{a['name']}`" for a in contract["allowlists"] if a["namespace"] == "external"
            )
            or "— 无"
        )
    )
    lines.append(
        "- 关键词表 (含空格条目, 匹配 prompt 文本, 不判死项): "
        + (
            ", ".join(
                f"`{a['name']}`" for a in contract["allowlists"] if a["namespace"] == "keywords"
            )
            or "— 无"
        )
    )
    lines.append("")

    lines.append("### 死项 (白名单条目无同名注册工具 ⇒ 永不命中)")
    lines.append("")
    dead = [a for a in contract["allowlists"] if a["phantoms"]]
    if dead:
        lines.append("| 允许表 | 死项 |")
        lines.append("|---|---|")
        for a in dead:
            lines.append(
                f"| `{a['name']}` @ `{a['rel']}` "
                f"| {', '.join('`'+p+'`' for p in a['phantoms'])} |"
            )
    else:
        lines.append("— 无。")
    lines.append("")

    lines.append("### 注册声明缺口")
    lines.append("")
    if contract["unresolved_specs"]:
        lines.append("| 注册清单 | 模块 | 类 (静态解析不到) |")
        lines.append("|---|---|---|")
        for s in contract["unresolved_specs"]:
            lines.append(f"| {s['list']} | `{s['module']}` | `{s['class']}` |")
    else:
        lines.append("- 注册清单引用的类均静态可解析。")
    if contract["duplicate_names"]:
        for d in contract["duplicate_names"]:
            lines.append(
                f"- ⚠️ 同名工具名由多类声明: `{d['name']}` ← "
                + ", ".join(f"`{c}`" for c in d["classes"])
            )
    if contract["unregistered_classes"]:
        lines.append(
            "- 声明了 `name` 却不在注册清单 (宣称未注册): "
            + ", ".join(f"`{c}`" for c in contract["unregistered_classes"])
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 钩子面: 事件声明面 vs 触发面 vs 注册面
# ---------------------------------------------------------------------------
#
# 前六面审"奖励/授权/工作流/模式/词汇/工具", 这一面审 HookManager 的**事件契约**:
#
#   - **声明面**: `hooks/__init__.py` 的事件常量 (`PRE_TOOL_USE = "pre_tool_use"` …)
#     + `ALL_EVENTS` 元组. `ALL_EVENTS` 是唯一权威清单 —— `register()` 据此校验
#     (未知名直接 `ValueError`), `__init__` 据此预建 `_callbacks` 键, 故它同时是
#     "允许面"与"穷尽面".
#   - **触发面**: 全仓 `trigger(EVENT, …)` / `_trigger_hook(EVENT, …)`, 以及
#     `run_pre()`(≡ `PRE_TOOL_USE`) / `run_post()`(≡ `POST_TOOL_USE`) —— 这两个方法名
#     全仓只 HookManager 定义, 故可直接按名映射到事件.
#   - **注册面**: 全仓 `register(EVENT, …)` / `register_hook(EVENT, …)`.
#
# MECE 两原则落到钩子:
#   - **collectively exhaustive**: 声明的事件必须**既有触发点又有消费者**. 任一为
#     零即缺口 —— 零触发 = "声明了但永不发生"; 零注册 = "会触发但没人接" (对偶于
#     奖励面"宣称项零调用者"). 均拿生产代码 (排除 tests/) 判定.
#   - **mutually exclusive**: 事件常量**值两两不同** —— 撞值会让 `register` 把钩子
#     挂到错误的既有事件上. 另记一条来自实现的**非对称**: `register` 对未知名抛错,
#     而 `trigger` 走 `self._callbacks.get(event, [])` 静默吞掉未知名 —— 拼错的事件
#     会变成空触发而不报错, 故触发点须用常量. 字面量里**只有值与已知事件相同的**
#     才可判定为钩子接线 (其它 `register("vasp", …)` 类调用属于别的注册表, 无法从
#     静态上区分, 故不报 —— 见下方 render 的诚实边界).
#
# 归属口径: 事件引用按**模块限定**解析 —— 只认 `from huginn.hooks import <CONST>`
# (含 `as` 别名) 与 `<hooksmod>.<CONST>` (hooksmod 已绑定到 huginn.hooks), 不认裸
# 同名 —— 否则 `events/event_types.py` 里同名的 `SESSION_START = "session.start"`
# (点分事件值) 会被误算成钩子事件引用.

_HOOKS_MODULE = "huginn/hooks/__init__.py"
# 无事件实参、但语义等价某事件的 HookManager 方法 → 事件常量名.
_HOOK_METHOD_EVENTS = {"run_pre": "PRE_TOOL_USE", "run_post": "POST_TOOL_USE"}
_HOOK_REGISTER_METHODS = frozenset({"register", "register_hook"})
_HOOK_TRIGGER_METHODS = frozenset({"trigger", "_trigger_hook"})
# 事件常量名 → 人工判读备注 (工具只做机械计数, 语义备注单独列, 同 _PENALTY_AXES 风格).
_HOOK_EVENT_NOTES = {
    "POST_TOOL_USE_FAILURE": (
        "触发点自带 `if self._callbacks[POST_TOOL_USE_FAILURE]` 守卫 —— 零注册 ⇒ "
        "该分支恒不执行, 是可证死的触发点"
    ),
}
# 事件常量形如 `UPPER_SNAKE = "lower_snake"`; 值须为小写蛇形串 (排除 ALL_EVENTS 元组).
_HOOK_EVENT_VALUE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _hook_declarations(root: Path) -> dict:
    """读声明面: 事件常量 (名→值)、`ALL_EVENTS` 成员、以及两侧缺口."""
    tree = _parse(root / _HOOKS_MODULE)
    if tree is None:
        return {"consts": {}, "members": [], "not_in_all": [], "unresolved_members": []}
    consts: dict[str, str] = {}
    members: list[str] = []
    for node in tree.body:
        name: str | None = None
        value: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        if name is None:
            continue
        if name == "ALL_EVENTS" and isinstance(value, ast.Tuple):
            members = [e.id for e in value.elts if isinstance(e, ast.Name)]
        elif (
            name.isupper()
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and _HOOK_EVENT_VALUE_RE.match(value.value)
        ):
            consts[name] = value.value
    not_in_all = sorted(c for c in consts if c not in members)
    unresolved_members = sorted(m for m in members if m not in consts)
    return {
        "consts": consts,
        "members": members,
        "not_in_all": not_in_all,
        "unresolved_members": unresolved_members,
    }


def _resolve_hook_event(
    arg: ast.AST, direct: dict[str, str], mod_aliases: set[str], decl: dict[str, str]
) -> str | None:
    """把触发/注册的首参解析成事件值. 只认模块限定的两种写法 (见上方归属口径)."""
    if isinstance(arg, ast.Name) and arg.id in direct:
        return decl.get(direct[arg.id])
    if (
        isinstance(arg, ast.Attribute)
        and isinstance(arg.value, ast.Name)
        and arg.value.id in mod_aliases
    ):
        return decl.get(arg.attr)
    return None


def _hook_wiring(root: Path) -> dict[str, dict[str, list[str]]]:
    """扫触发/注册点, 按事件值归集 prod/test 位置."""
    decl = _hook_declarations(root)["consts"]
    out: dict[str, dict[str, list[str]]] = {
        v: {"prod_trigger": [], "prod_register": [], "test_trigger": [], "test_register": []}
        for v in decl.values()
    }
    for py in _iter_py(root):
        rel = py.relative_to(root).as_posix()
        tree = _parse(py)
        if tree is None:
            continue
        # 模块内解析: 事件常量本地名 → 常量名; 以及绑到 huginn.hooks 的模块别名.
        # 定义文件自身 (`hooks/__init__.py`) 里常量是本地定义而非 import, 故预置
        # 自身常量表 —— 否则 `trigger(POST_TOOL_USE_FAILURE, …)` 这类自触发漏计.
        direct: dict[str, str] = (
            {name: name for name in decl} if rel == _HOOKS_MODULE else {}
        )
        mod_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = _resolve_relative(rel, node.level, node.module or "")
                for a in node.names:
                    if mod == "huginn.hooks":
                        direct[a.asname or a.name] = a.name
                    elif f"{mod}.{a.name}" == "huginn.hooks":
                        mod_aliases.add(a.asname or a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "huginn.hooks":
                        mod_aliases.add(a.asname or "hooks")

        bucket = "test" if _is_test(rel) else "prod"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            method = (
                fn.attr
                if isinstance(fn, ast.Attribute)
                else (fn.id if isinstance(fn, ast.Name) else None)
            )
            if method is None:
                continue
            if method in _HOOK_METHOD_EVENTS:
                val = decl.get(_HOOK_METHOD_EVENTS[method])
                kind = "trigger"
            elif method in _HOOK_REGISTER_METHODS:
                val = _resolve_hook_event(node.args[0], direct, mod_aliases, decl) if node.args else None
                kind = "register"
            elif method in _HOOK_TRIGGER_METHODS:
                val = _resolve_hook_event(node.args[0], direct, mod_aliases, decl) if node.args else None
                kind = "trigger"
            else:
                continue
            if val is None or val not in out:
                continue
            out[val][f"{bucket}_{kind}"].append(f"{rel}:{node.lineno}")
    return out


def _hook_literal_wiring(root: Path) -> list[dict]:
    """触发/注册首参用了**字符串字面量且值等于已知事件**的调用点.

    只报"字面量值 = 已知事件值" —— 这类可确证是钩子接线却绕过了常量. 未知名的
    字面量无法与其它注册表 (`register("vasp", …)`) 区分, 故不报.
    """
    values = set(_hook_declarations(root)["consts"].values())
    out: list[dict] = []
    methods = _HOOK_REGISTER_METHODS | _HOOK_TRIGGER_METHODS
    for py in _iter_py(root):
        rel = py.relative_to(root).as_posix()
        tree = _parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            method = (
                fn.attr
                if isinstance(fn, ast.Attribute)
                else (fn.id if isinstance(fn, ast.Name) else None)
            )
            if method not in methods:
                continue
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.value in values:
                out.append({"rel": rel, "line": node.lineno, "method": method, "value": a0.value})
    return out


def build_hook_contract(root: Path | None = None) -> dict:
    """钩子面: 事件声明面 ↔ 触发面 ↔ 注册面是否穷尽一致."""
    root = root or _REPO
    decl = _hook_declarations(root)
    wiring = _hook_wiring(root)

    events: list[dict] = []
    for const in decl["members"]:
        value = decl["consts"].get(const, "")
        w = wiring.get(value, {})
        prod_t, prod_r = len(w.get("prod_trigger", [])), len(w.get("prod_register", []))
        test_t, test_r = len(w.get("test_trigger", [])), len(w.get("test_register", []))
        if prod_t and prod_r:
            status = "wired"
        elif prod_t:
            status = "trigger-only"
        elif prod_r:
            status = "register-only"
        else:
            status = "dead"
        events.append(
            {
                "const": const,
                "value": value,
                "prod_trigger": prod_t,
                "prod_register": prod_r,
                "test_trigger": test_t,
                "test_register": test_r,
                "status": status,
                "note": _HOOK_EVENT_NOTES.get(const)
                or ("触发点存在但无注册消费者 —— 扩展点候选" if status == "trigger-only" else ""),
                "trigger_sites": w.get("prod_trigger", []),
                "register_sites": w.get("prod_register", []),
            }
        )

    collisions: list[dict] = []
    by_value: dict[str, list[str]] = defaultdict(list)
    for const, value in decl["consts"].items():
        by_value[value].append(const)
    for value, consts in sorted(by_value.items()):
        if len(consts) > 1:
            collisions.append({"value": value, "consts": sorted(consts)})

    return {
        "module": _HOOKS_MODULE,
        "events": events,
        "collisions": collisions,
        "not_in_all_events": decl["not_in_all"],
        "unresolved_members": decl["unresolved_members"],
        "literal_wiring": _hook_literal_wiring(root),
    }


_HOOK_STATUS_DOC = {
    "wired": "有触发点且有消费者",
    "trigger-only": "有触发点但零注册 (触发无人接)",
    "register-only": "有注册但零生产触发",
    "dead": "声明零触发且零注册",
}


def render_hook_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 钩子面: 事件声明面 vs 触发面 vs 注册面")
    lines.append("")
    lines.append(
        f"声明面: `{contract['module']}` 的 {len(contract['events'])} 个事件"
        "(`ALL_EVENTS` 权威清单). 触发面: `trigger()` / `_trigger_hook()` / "
        "`run_pre()`(≡`pre_tool_use`) / `run_post()`(≡`post_tool_use`). 注册面: "
        "`register()` / `register_hook()`. `trigger-only` = 会触发但零消费者 "
        "(对偶于奖励面「宣称项零调用者」); `dead` = 声明了却零触发零注册."
    )
    lines.append("")
    lines.append("状态: " + "; ".join(f"`{k}`={v}" for k, v in _HOOK_STATUS_DOC.items()))
    lines.append("")
    lines.append("| 事件常量 | 值 | 生产触发 | 生产注册 | 测试触发 | 测试注册 | 状态 | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in contract["events"]:
        lines.append(
            f"| `{e['const']}` | `{e['value']}` | {e['prod_trigger']} | {e['prod_register']} "
            f"| {e['test_trigger']} | {e['test_register']} | `{e['status']}` | {e['note']} |"
        )
    lines.append("")

    lines.append("### 触发点 / 注册点明细")
    lines.append("")
    lines.append("| 事件常量 | 生产触发点 | 生产注册点 |")
    lines.append("|---|---|---|")
    for e in contract["events"]:
        trig = ", ".join(f"`{s}`" for s in e["trigger_sites"]) or "—"
        reg = ", ".join(f"`{s}`" for s in e["register_sites"]) or "—"
        lines.append(f"| `{e['const']}` | {trig} | {reg} |")
    lines.append("")

    lines.append("### 互斥违例 (mutually exclusive)")
    lines.append("")
    if contract["collisions"]:
        for c in contract["collisions"]:
            lines.append(
                f"- ⚠️ 事件常量撞值 `{c['value']}`: "
                + ", ".join(f"`{n}`" for n in c["consts"])
            )
    else:
        lines.append("- 事件常量值两两不同 —— 无撞值.")
    if contract["literal_wiring"]:
        for w in contract["literal_wiring"]:
            lines.append(
                f"- ⚠️ 触发/注册用字面量而非常量: `{w['value']}` @ "
                f"`{w['rel']}:{w['line']}` ({w['method']})"
            )
    else:
        lines.append("- 触发/注册均用事件常量, 无绕过常量的字面量.")
    lines.append("")

    lines.append("### 声明缺口")
    lines.append("")
    if contract["not_in_all_events"]:
        lines.append(
            "- ⚠️ 常量声明了却不在 `ALL_EVENTS` (register 无法校验): "
            + ", ".join(f"`{c}`" for c in contract["not_in_all_events"])
        )
    if contract["unresolved_members"]:
        lines.append(
            "- ⚠️ `ALL_EVENTS` 成员无对应常量定义: "
            + ", ".join(f"`{c}`" for c in contract["unresolved_members"])
        )
    if not contract["not_in_all_events"] and not contract["unresolved_members"]:
        lines.append("- `ALL_EVENTS` 与事件常量定义面双向一致.")
    lines.append("")
    lines.append(
        "诚实边界: 未知名字面量 (`register(\"vasp\", …)` 这类别的注册表) 无法静态"
        "区分, 故不计入互斥违例; 但实现层 `trigger` 用 `_callbacks.get(event, [])` "
        "静默吞掉未知名 —— 字面量拼错会变空触发而不报错, 这是触发点须用常量的理由."
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
        "workflow": build_workflow_contract(root),
        "modes": build_mode_contract(root),
        "vocabulary": build_vocabulary_contract(root),
        "tools": build_tool_contract(root),
        "hooks": build_hook_contract(root),
    }


def find_issues(snap: dict) -> list[str]:
    """汇总 MECE 违例 (供 --check). 只报客观事实, 语义判定留人工."""
    issues: list[str] = []
    rw = snap["reward"]
    for t in rw["terms"]:
        if t["status"] == "dead":
            issues.append(f"奖励项零调用者: {t['name']}")
    for ax in rw["penalty_axes"]:
        if ax["overlap"]:
            issues.append(f"同轴惩罚候选 ({ax['axis']}): {', '.join(ax['terms'])}")
    for d in rw["cross_module_dupes"]:
        issues.append(f"跨模块同名: {d['name']} @ {', '.join(d['modules'])}")
    for s in snap["scope"]["sources"]:
        if s["status"] == "dead":
            issues.append(f"授权口径零消费点: {s['name']}")
    wf = snap["workflow"]
    if not wf["branch_matches_dispatch"]:
        issues.append("工作流: dispatch_table 与 engine_act 硬编码分支不一致")
    if wf["prompt_inconsistent"]:
        issues.append("工作流: planner 多处 MODE 提示互相不一致")
    for m in wf["missing_from_prompt"]:
        issues.append(f"工作流 mode 未在 planner 提示暴露: {m}")
    md = snap["modes"]
    for m in md["prompt_only"]:
        issues.append(f"模式有 prompt 段却无 set_mode 生产者: {m}")
    for m in md["runtime_only"]:
        issues.append(f"模式被 set_mode 却无 prompt 段: {m}")
    if not md["consistent"]:
        issues.append("模式: 各来源词表互相不一致")
    vc = snap["vocabulary"]
    # 簇内已按"词表漂移"报过的名字不再按"同名定义"重复报一次.
    drift_names = {m["name"] for d in vc["divergence"] for m in d["members"]}
    for d in vc["duplicate_defs"]:
        if d["same_values"] or d["name"] in drift_names:
            continue
        sites = ", ".join(f"{s['rel']}:{s['line']}" for s in d["sites"])
        issues.append(f"词汇: 同名跨模块定义值域不一致: {d['name']} @ {sites}")
    for d in vc["divergence"]:
        members = ", ".join(f"{m['rel']}::{m['name']}" for m in d["members"])
        issues.append(f"词汇: 词表漂移 (簇 {d['cluster']}, 并集 {len(d['union'])} 词): {members}")
    for m in vc["mappings"]:
        if m["injective"] or not m["reverse_present"]:
            continue
        coll = ", ".join(f"{v}←{ks}" for v, ks in m["collisions"].items())
        issues.append(f"词汇: 映射往返丢信息: {m['name']} 共像 [{coll}] 且有反向表 {m['reverse_name']}")
    tl = snap["tools"]
    for s in tl["unresolved_specs"]:
        issues.append(f"工具: 注册清单引用的类静态解析不到: {s['class']} @ {s['module']}")
    for d in tl["duplicate_names"]:
        issues.append(
            f"工具: 同名工具名由多类声明: {d['name']} ← {', '.join(d['classes'])}"
        )
    for a in tl["allowlists"]:
        if not a["phantoms"]:
            continue
        preview = ", ".join(a["phantoms"][:6])
        if len(a["phantoms"]) > 6:
            preview += " …"
        issues.append(
            f"工具: 允许表死项 (无同名注册工具, 永不命中): {a['name']} @ {a['rel']} → {preview}"
        )
    hk = snap["hooks"]
    _hook_issue = {
        "trigger-only": "钩子: 事件有生产触发点但零生产注册 (触发无人接): ",
        "register-only": "钩子: 事件有注册但零生产触发: ",
        "dead": "钩子: 事件声明零触发零注册: ",
    }
    for e in hk["events"]:
        prefix = _hook_issue.get(e["status"])
        if prefix:
            issues.append(prefix + e["const"])
    for c in hk["collisions"]:
        issues.append(
            f"钩子: 事件常量撞值 {c['value']}: {', '.join(c['consts'])}"
        )
    for name in hk["not_in_all_events"]:
        issues.append(f"钩子: 事件常量未登记进 ALL_EVENTS: {name}")
    for name in hk["unresolved_members"]:
        issues.append(f"钩子: ALL_EVENTS 成员无常量定义: {name}")
    for w in hk["literal_wiring"]:
        issues.append(
            f"钩子: 触发/注册用字面量而非常量: {w['value']} @ {w['rel']}:{w['line']}"
        )
    return issues


def render_mece_markdown(snap: dict) -> str:
    lines: list[str] = []
    lines.append("# MECE 契约审计 (奖励面 + 授权面 + 工作流面 + 模式面 + 词汇面 + 工具面 + 钩子面)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.contract_audit --out docs/mece-audit.md`."
    )
    lines.append(
        "以 MECE 两原则审计 agent 的**奖励面 / 授权面 / 工作流面 / 模式面 / 词汇面 / "
        "工具面 / 钩子面**: **collectively exhaustive** 抓「宣称维度零调用者 / 面之间的缺口」; "
        "**mutually exclusive** 抓「同轴惩罚叠加」「跨模块同名重复实现」「词表互不一致」"
        "「同名工具名多类声明」「事件常量撞值」. 纯静态扫描, 只提示候选, 不判死."
    )
    lines.append("")
    lines.append(render_reward_markdown(snap["reward"]))
    lines.append(render_scope_markdown(snap["scope"]))
    lines.append(render_workflow_markdown(snap["workflow"]))
    lines.append(render_mode_markdown(snap["modes"]))
    lines.append(render_vocabulary_markdown(snap["vocabulary"]))
    lines.append(render_tool_markdown(snap["tools"]))
    lines.append(render_hook_markdown(snap["hooks"]))
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
    parser.add_argument("--workflow", action="store_true", help="只看工作流面")
    parser.add_argument("--modes", action="store_true", help="只看模式面")
    parser.add_argument("--vocab", action="store_true", help="只看词汇面")
    parser.add_argument("--tools", action="store_true", help="只看工具面")
    parser.add_argument("--hooks", action="store_true", help="只看钩子面")
    parser.add_argument("--json", action="store_true", help="输出 JSON 快照")
    parser.add_argument("--check", action="store_true", help="有 MECE 发现时 exit 1")
    parser.add_argument("--out", type=str, default="", help="写 markdown 到文件")
    args = parser.parse_args(argv)

    surfaces = {
        "reward": (args.reward, build_reward_contract, render_reward_markdown),
        "scope": (args.scope, build_scope_contract, render_scope_markdown),
        "workflow": (args.workflow, build_workflow_contract, render_workflow_markdown),
        "modes": (args.modes, build_mode_contract, render_mode_markdown),
        "vocabulary": (args.vocab, build_vocabulary_contract, render_vocabulary_markdown),
        "tools": (args.tools, build_tool_contract, render_tool_markdown),
        "hooks": (args.hooks, build_hook_contract, render_hook_markdown),
    }
    selected = [k for k, (on, _b, _r) in surfaces.items() if on]
    if len(selected) == 1:
        _on, builder, renderer = surfaces[selected[0]]
        data_obj = builder()
        md = renderer(data_obj)
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
