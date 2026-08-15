"""环境变量契约审计命令 — 扫描 huginn/ 下所有 HUGINN_* 引用.

为什么需要: huginn 有 200+ 个 HUGINN_* 环境变量, 散落各处用
``os.environ.get(...)`` 裸读, 无统一 schema, 无法审计哪些配置存在 /
默认值 / 是否在代码里被设置. 本命令提供机器可读的 inventory + 人读的
契约文档, 让"配置面"可解释、可维护.

用法:
    python -m huginn.cli.config_audit                     # 打印 markdown 契约
    python -m huginn.cli.config_audit --json              # 打印 JSON inventory
    python -m huginn.cli.config_audit --out docs/env-contract.md   # 写契约文档

说明:
    - 只做静态扫描, 不判死. "是否死"需结合运行时契约人工判断.
    - 状态字段: writes>0 → "code-set"(代码里被设置); ==0 → "external"(未在
      代码设, 可能是用户 shell/.env 注入, 需人工确认).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 顶层包根 (huginn/)
_ROOT = Path(__file__).resolve().parents[1]
# 本审计工具自身路径, 扫描注册面时跳过, 避免 docstring 里的 @filter.on_xxx /
# register_* 示例字样被误判成真实注册点.
_SELF = Path(__file__).resolve()

# os.environ 操作的正则. 捕获: 操作方式, 变量名, 默认值/设定值.
_ENV_GET = re.compile(
    r'os\.environ\.get\(\s*["\'](HUGINN_[A-Z0-9_]+)["\']\s*(?:,\s*(["\']?[^"\')]*["\']?))?'
)
_ENV_SETDEFAULT = re.compile(
    r'os\.environ\.setdefault\(\s*["\'](HUGINN_[A-Z0-9_]+)["\']\s*,\s*(["\']?[^"\')]*["\']?)'
)
_ENV_SETITEM = re.compile(
    r'os\.environ\[["\'](HUGINN_[A-Z0-9_]+)["\']\]\s*='
)
_ENV_POP = re.compile(
    r'os\.environ\.pop\(\s*["\'](HUGINN_[A-Z0-9_]+)["\']'
)


def _clean(value: str) -> str:
    """去掉默认值两端的引号, 保留原义."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _scan_file(path: Path, ops: dict):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for offset, (pattern, kind) in enumerate(
        (
            (_ENV_GET, "read"),
            (_ENV_SETDEFAULT, "setdefault"),
            (_ENV_SETITEM, "set"),
            (_ENV_POP, "pop"),
        )
    ):
        for m in pattern.finditer(text):
            name = m.group(1)
            default = ""
            if kind == "setdefault":
                default = _clean(m.group(2))
            elif kind == "read" and m.group(2) is not None:
                default = _clean(m.group(2))
            lineno = text[: m.start()].count("\n") + 1
            ops[name][kind].append(
                {
                    "file": str(path.relative_to(_ROOT)),
                    "line": lineno,
                    "default": default,
                }
            )


def build_inventory(root: Path | None = None) -> dict[str, dict]:
    """扫描 huginn/ 下所有 .py, 构建 env inventory."""
    root = root or _ROOT
    ops: dict[str, dict] = defaultdict(
        lambda: {"read": [], "setdefault": [], "set": [], "pop": []}
    )
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        _scan_file(py, ops)

    inventory: dict[str, dict] = {}
    for name in sorted(ops):
        o = ops[name]
        # read 默认值取"最常见"的 (多数调用点的一致默认, 便于人读).
        # 并列时按值字典序稳定取大, 避免 set() 哈希顺序导致渲染不确定.
        defaults = [r["default"] for r in o["read"] if r["default"]]
        most_common_default = (
            max(defaults, key=lambda d: (defaults.count(d), d)) if defaults else ""
        )
        writes = len(o["setdefault"]) + len(o["set"]) + len(o["pop"])
        inventory[name] = {
            "name": name,
            "reads": sorted(o["read"], key=lambda r: (r["file"], r["line"])),
            "setdefaults": sorted(
                o["setdefault"], key=lambda r: (r["file"], r["line"])
            ),
            "sets": sorted(o["set"], key=lambda r: (r["file"], r["line"])),
            "pops": sorted(o["pop"], key=lambda r: (r["file"], r["line"])),
            "default": most_common_default,
            "status": "external" if writes == 0 else "code-set",
        }
    return inventory


def _fmt_locs(entries: list[dict], limit: int = 3) -> str:
    if not entries:
        return "—"
    parts = [f"{e['file']}:{e['line']}" for e in entries[:limit]]
    if len(entries) > limit:
        parts.append(f"+{len(entries) - limit} 处")
    return ", ".join(parts)


def render_markdown(inventory: dict[str, dict]) -> str:
    """渲染人读的契约文档 (markdown)."""
    lines: list[str] = []
    lines.append("# 环境变量契约 (HUGINN_*)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --out docs/env-contract.md`."
    )
    lines.append(
        "本表登记 huginn/ 代码库中所有 `HUGINN_*` 环境变量的引用, 是配置面的"
        "审计基线. **推断状态仅提示, 不代替人工判定**: `code-set` 表示代码某处"
        "设置了它; `external` 表示未在代码内设置 (可能由用户 shell/.env 注入,"
        "需人工确认是否为有效配置)."
    )
    lines.append("")
    lines.append("| 变量 | 默认值 | 读取点 | 设置点 | 推断状态 |")
    lines.append("|---|---|---|---|---|")
    for name in sorted(inventory):
        item = inventory[name]
        reads = _fmt_locs(item["reads"])
        writes = ", ".join(
            p
            for p in (
                _fmt_locs(item["setdefaults"]),
                _fmt_locs(item["sets"]),
                _fmt_locs(item["pops"]),
            )
            if p != "—"
        )
        if not writes:
            writes = "—"
        default = item["default"] or "''"
        lines.append(
            f"| `{name}` | {default} | {reads} | {writes} | {item['status']} |"
        )
    lines.append("")
    lines.append(f"共 {len(inventory)} 个环境变量。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FeatureFlags 契约 (feature-flags-contract.md)
# ---------------------------------------------------------------------------
# 可选退出/可选进入的功能开关统一登记处. 与 env 契约互补: env 契约讲"配置面",
# flags 契约讲"功能开关面". 每个 flag 由 FeatureFlags._DEFAULTS 定义默认值,
# _DESCRIPTIONS 给描述, _ENV_ALIASES 兼容旧裸读变量名, 消费点由 is_enabled 静态扫描.

_FLAG_IS_ENABLED = re.compile(r'is_enabled\(\s*["\']([a-z0-9_]+)["\']\s*\)')


# ---------------------------------------------------------------------------
# 插件契约 (plugins-contract.md)
# ---------------------------------------------------------------------------
# "Everything is a Plugin" 的登记面. 覆盖两种形态:
#   - 形态 B (策略注册表): register_*_policy / register_prompt_segment 静态注册
#   - 形态 A (事件钩子):  @filter.on_xxx 装饰器
# 静态扫描只登记"在哪注册 / 注册名 / 显式 priority", 不判生命周期. 用于回答
# "项目里有哪些可插拔点 / 各自的注册位置 / 优先级" 的可解释性与可维护性基线.

_REG_PROMPT = re.compile(
    r'register_prompt_segment\(\s*["\']([a-z0-9_]+)["\']\s*,?\s*[^)]*?(?:priority\s*=\s*(\d+))?'
)
_REG_COMPACTION = re.compile(
    r'register_compaction_policy\(\s*["\']([a-z0-9_]+)["\']\s*,?\s*[^)]*?(?:priority\s*=\s*(\d+))?'
)
_REG_MEMORY = re.compile(
    r'register_memory_maintenance_policy\(\s*["\']([a-z0-9_]+)["\']\s*,?\s*[^)]*?(?:priority\s*=\s*(\d+))?'
)
_EVENT_DECO = re.compile(r'@filter\.(on_[a-z_]+)')

# 内置 prompt 段规范优先级 (来自 plugins/prompt_segments._PRIORITY), 当注册点未
# 显式传 priority 时用于标注"内置默认".
# 由 _load_prompt_priorities() 动态读取, 避免在此硬编码漂移.
_PROMPT_PRIORITIES: dict[str, int] = {}


def _load_prompt_priorities() -> dict[str, int]:
    """读取 prompt_segments._PRIORITY 作为内置段优先级基准."""
    if _PROMPT_PRIORITIES:
        return _PROMPT_PRIORITIES
    try:
        from huginn.plugins.prompt_segments import _PRIORITY
        _PROMPT_PRIORITIES.update(_PRIORITY)
    except Exception:
        pass
    return _PROMPT_PRIORITIES


def _scan_registrations(root: Path, pattern: re.Pattern) -> list[dict]:
    """扫描所有 register_xxx("name", ...) 调用, 归类 name/priority/位置."""
    rows: list[dict] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or py.resolve() == _SELF:
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in pattern.finditer(text):
            ptr = m.group(2)
            rows.append(
                {
                    "name": m.group(1),
                    "priority": int(ptr) if ptr else None,
                    "loc": f"{py.relative_to(root)}:{text[: m.start()].count(chr(10)) + 1}",
                }
            )
    return rows


def build_plugins_contract(root: Path | None = None) -> dict[str, list[dict]]:
    """构建插件契约: 两形态注册面 inventory."""
    root = root or _ROOT
    prompt_prio = _load_prompt_priorities()

    prompt = _scan_registrations(root, _REG_PROMPT)
    for r in prompt:
        if r["priority"] is None:
            r["priority"] = prompt_prio.get(r["name"])

    compaction = _scan_registrations(root, _REG_COMPACTION)
    memory = _scan_registrations(root, _REG_MEMORY)

    # 事件钩子: 每个 @filter.on_xxx 装饰器, 记录其跟随的 handler 函数名 (下一行).
    hooks: list[dict] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or py.resolve() == _SELF:
            continue
        try:
            lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            for m in _EVENT_DECO.finditer(line):
                handler = ""
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    hm = re.match(r'(async\s+def|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)', nxt)
                    if hm:
                        handler = hm.group(2)
                hooks.append(
                    {
                        "name": m.group(1),
                        "handler": handler,
                        "loc": f"{py.relative_to(root)}:{i + 1}",
                    }
                )

    return {
        "prompt_segments": sorted(prompt, key=lambda r: (r["priority"] if r["priority"] is not None else 10**9, r["name"])),
        "compaction_policies": sorted(compaction, key=lambda r: (r["priority"] if r["priority"] is not None else 0, r["name"])),
        "memory_policies": sorted(memory, key=lambda r: (r["priority"] if r["priority"] is not None else 0, r["name"])),
        "event_hooks": sorted(hooks, key=lambda r: (r["name"], r["loc"])),
        "defaults": _policy_defaults(),
    }


def _policy_defaults() -> dict:
    """运行时读取 compaction / 记忆整理策略的**内置默认** (轻量导入).

    静态扫描抓不到注册调用 (第三方策略在运行时注册), 但内置默认值是模块级
    常量, 运行时读取后契约诚实反映"无第三方接入时的生效策略".
    """
    out: dict = {}
    try:
        from huginn.plugins.compaction_policy import (
            _DEFAULT_NEVER_TRIM_BLOCK_TYPES,
            _DEFAULT_PROTECTED_ROLES,
        )
        from huginn.plugins.memory_maintenance_policy import _DEFAULT as _mem_default

        out["compaction"] = {
            "protected_roles": sorted(_DEFAULT_PROTECTED_ROLES),
            "never_trim_block_types": sorted(_DEFAULT_NEVER_TRIM_BLOCK_TYPES),
        }
        out["memory"] = {
            "decay_per_day": _mem_default.decay_per_day,
            "prune_threshold": _mem_default.prune_threshold,
            "deduplicate": _mem_default.deduplicate,
        }
    except Exception as e:  # noqa: BLE001 - 依赖缺失时降级
        out["error"] = str(e)
    return out


def _fmt_section(title: str, rows: list[dict], headers: tuple[str, ...], keys: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"| {' | '.join(headers)} |")
    lines.append(f"|{'---|' * len(headers)}")
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            cells.append("—" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    if not rows:
        lines.append("| — |")
    lines.append("")
    return lines


def render_plugins_markdown(contract: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    lines.append("# 插件契约 (Everything is a Plugin)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --plugins --out docs/plugins-contract.md`."
    )
    lines.append(
        "登记项目内所有可插拔注册面, 分两种形态: **形态 B** (策略注册表, 同步选策略, "
        "`register_*_policy` / `register_prompt_segment`), **形态 A** (事件钩子, "
        "`@filter.on_xxx`, 异步分发 + 可阻断)。静态扫描只报注册位置与显式 priority, "
        "不判生命周期; `priority —` 表示注册点未显式传参 (用内置默认)。"
    )
    lines.append("")
    lines += _fmt_section(
        "Prompt 段 (形态 B)",
        contract["prompt_segments"],
        ("段名", "priority", "注册位置"),
        ("name", "priority", "loc"),
    )
    lines += _fmt_section(
        "Compaction 策略 (形态 B)",
        contract["compaction_policies"],
        ("策略名", "priority", "注册位置"),
        ("name", "priority", "loc"),
    )
    lines += _fmt_section(
        "记忆整理策略 (形态 B)",
        contract["memory_policies"],
        ("策略名", "priority", "注册位置"),
        ("name", "priority", "loc"),
    )
    lines += _fmt_section(
        "事件钩子 (形态 A)",
        contract["event_hooks"],
        ("事件", "handler", "注册位置"),
        ("name", "handler", "loc"),
    )
    # 内置默认策略 (无第三方接入时的生效值). 字段名短:
    d = contract.get("defaults", {})
    if isinstance(d, dict) and "error" not in d:
        comp = d.get("compaction", {})
        mem = d.get("memory", {})
        lines.append("### 内置默认策略值 (无第三方接入时生效)")
        lines.append("")
        lines.append(
            f"- **Compaction**: protected_roles="
            f"`{', '.join(comp.get('protected_roles', []))}`; "
            f"never_trim_block_types=`{', '.join(comp.get('never_trim_block_types', []))}`"
        )
        lines.append(
            f"- **记忆整理**: decay_per_day=`{mem.get('decay_per_day')}`, "
            f"prune_threshold=`{mem.get('prune_threshold')}`, deduplicate=`{mem.get('deduplicate')}`"
        )
        lines.append("")
    elif isinstance(d, dict) and "error" in d:
        lines.append("> 注: 内置默认策略值读取失败 (依赖缺失): " + str(d["error"]))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具契约 (tools-contract.md)
# ---------------------------------------------------------------------------
# 运行时枚举核心工具注册表的注册面. 与插件契约互补: 插件契约讲"可插拔扩展点",
# 工具契约讲"agent 实际可调的服务". 用 register_core_tools(None) 枚举 (安全、快,
# 无重型依赖), 字段来自 HuginnTool 声明 (name/description/category/destructive/
# read_only/cost_tier). 是 agent 能力面的可解释性基线.

def build_tools_contract() -> list[dict]:
    """实例化核心工具并向 ToolRegistry 登记, 快照其声明元数据.

    契约描述的是"agent 核心能力基线", 只应反映核心工具集 (register_core_tools),
    不应受进程内已有注册 (如测试 fixture 预注册的全部工具) 影响. 因此先在
    隔离的临时注册表上登记, 结束后恢复现场, 保证输出与运行环境无关、可再生成.
    """
    from huginn.tools import register_core_tools
    from huginn.tools.registry import ToolRegistry

    ambient = ToolRegistry.snapshot()
    try:
        ToolRegistry.clear()
        register_core_tools(None)
        rows: list[dict] = []
        for name in sorted(ToolRegistry._tools):
            t = ToolRegistry._tools[name]
            rows.append(
                {
                    "name": name,
                    "category": t.category,
                    "destructive": t.destructive,
                    "read_only": t.read_only,
                    "cost_tier": t.cost_tier,
                    "description": (t.description or "").strip().replace("\n", " ")[:80],
                }
            )
        return rows
    finally:
        ToolRegistry.restore(ambient)


def render_tools_markdown(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# 工具契约 (ToolRegistry)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --tools --out docs/tools-contract.md`."
    )
    lines.append(
        "运行时枚举核心工具注册表 (register_core_tools). 字段来自 HuginnTool 声明: "
        "`category` (core/search/meta/sim/sci/design/cv/materials/misc), `destructive` / "
        "`read_only` 是权限系统判定依据, `cost_tier` 来自 ToolProfile. "
        "未含启动时后台注册的可选工具 (见 lifespan 的 register_optional_tools)."
    )
    lines.append("")
    lines.append("| 工具 | 分类 | 危险 | 只读 | cost | 描述 |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['category']} | {r['destructive']} | "
            f"{r['read_only']} | {r['cost_tier']} | {r['description']} |"
        )
    lines.append("")
    lines.append(f"共 {len(rows)} 个核心工具。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 事件契约 (events-contract.md)
# ---------------------------------------------------------------------------
# 插件事件面的契约. 两层:
#   - EventType 枚举 (plugins/api/event.py): 插件可监听的事件全集
#   - UnifiedBus.publish_* 方法: 语义化的事件发射契约 (统一入口, 扇出到 4 套系统)
# 派发点 = 代码里引用 EventType.<MEMBER> 的位置, 提示每个事件在哪被发出.

_EVENT_REF = re.compile(r'EventType\.([A-Z][A-Z0-9_]+)')
_EVENT_PUBLISH = re.compile(r'async\s+def\s+(publish_[a-z_]+)\(')


def _event_groups() -> dict[str, str]:
    """从 event.py 源码注释提取 EventType 成员 → 分组 的映射."""
    groups: dict[str, str] = {}
    try:
        text = (Path(__file__).resolve().parents[1] / "api" / "event.py").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return groups
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("# "):
            continue
        if stripped.startswith("# "):
            current = stripped[2:].strip()
        m = re.match(r'^\s*([A-Z][A-Z0-9_]+)\s*=\s*auto\(\)', stripped)
        if m:
            groups[m.group(1)] = current
    return groups


def build_events_contract(root: Path | None = None) -> dict:
    """构建事件契约: EventType 成员 + 派发点 + UnifiedBus publish 接口."""
    root = root or _ROOT
    from huginn.api.event import EventType

    groups = _event_groups()
    refs: dict[str, list[str]] = defaultdict(list)
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or py.resolve() == _SELF:
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _EVENT_REF.finditer(text):
            refs[m.group(1)].append(f"{py.relative_to(root)}:{text[: m.start()].count(chr(10)) + 1}")

    members: list[dict] = []
    for m in EventType:
        name = m.name
        locs = sorted(set(refs.get(name, [])))
        members.append(
            {
                "name": name,
                "group": groups.get(name, ""),
                "dispatch": ", ".join(locs[:3]) + (f" +{len(locs)-3} 处" if len(locs) > 3 else ""),
            }
        )

    # UnifiedBus publish 方法
    ub_path = root / "events" / "unified_bus.py"
    publishes: list[str] = []
    if ub_path.exists():
        try:
            publishes = _EVENT_PUBLISH.findall(
                ub_path.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            pass

    return {"members": members, "publishes": sorted(set(publishes))}


def render_events_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("# 事件契约 (插件事件面)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --events --out docs/events-contract.md`."
    )
    lines.append(
        "**EventType** 是插件可监听的事件全集 (plugins/api/event.py); `dispatch` 是代码里"
        "引用该事件的位置, 提示它在哪里被发出 (静态扫描, 取前 3 处)。**UnifiedBus 发射接口**"
        "是语义化统一入口, 每次 publish 扇出到 HookManager / 内部 EventBus / 插件 EventBus / "
        "PetBus 各子系统。"
    )
    lines.append("")
    lines += _fmt_section(
        "EventType 成员",
        contract["members"],
        ("事件", "分组", "派发点"),
        ("name", "group", "dispatch"),
    )
    pub_lines = ["### UnifiedBus 发射接口 (语义化)", ""]
    for p in contract["publishes"]:
        pub_lines.append(f"- `{p}()`")
    if not contract["publishes"]:
        pub_lines.append("- —")
    pub_lines.append("")
    lines += pub_lines
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 路由契约 (routes-contract.md)
# ---------------------------------------------------------------------------
# 模型路由面的契约: ModelRouter 的 task→tag 偏好映射 + 可选 task 全集 +
# HUGINN_MODEL_* 的 from_env 装配约定. 回答"某个 task 会优先落到哪些模型".

def build_routes_contract() -> dict:
    """读取 ModelRouter 的 task→tag 映射与 task 全集."""
    from huginn.models.router import ModelRouter, TaskT

    tasks = list(getattr(TaskT, "__args__", ()))
    return {
        "task_tags": dict(ModelRouter._TASK_TAGS),
        "tasks": tasks,
    }


def render_routes_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("# 模型路由契约 (ModelRouter)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --routes --out docs/routes-contract.md`."
    )
    lines.append(
        "登记 ModelRouter 的 task → 偏好 tag 映射 (`_TASK_TAGS`)。`select(task)` 按列表"
        "顺序找第一个有匹配模型的 tag; 无匹配回落默认。任务也可经 `HUGINN_MODEL_<TASK>` "
        "环境变量装配 (见 models/router.py::from_env)。"
    )
    lines.append("")
    lines.append("| task | 偏好 tag (按序) |")
    lines.append("|---|---|")
    for task, tags in contract["task_tags"].items():
        lines.append(f"| `{task}` | {', '.join(tags)} |")
    lines.append("")
    lines.append(f"可选 task 全集 ({len(contract['tasks'])}): " + ", ".join(f"`{t}`" for t in contract["tasks"]))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 错误语义契约 (errors-contract.md)
# ---------------------------------------------------------------------------
# ToolResult.error_kind 的分类契约. 回答"工具失败如何分类 / 谁在哪里做映射".
# 枚举来自 core_types.py::ErrorKind; 分类点 = 代码里产出 ErrorKind.<MEMBER> 的位置.

_ERROR_KIND_REF = re.compile(r'ErrorKind\.([A-Z][A-Z0-9_]+)')
_ERROR_KIND_DOC = re.compile(r'^\s*([A-Z][A-Z0-9_]+)\s*=\s*"[A-Za-z_]+"\s*#\s*(.+)$')


def build_errors_contract(root: Path | None = None) -> dict:
    """构建错误语义契约: ErrorKind 枚举 + 各分类点."""
    root = root or _ROOT
    # 枚举语义: 从 core_types.py 的 ErrorKind 类内注释行提取 (name → 说明), 避免硬编码漂移.
    docs: dict[str, str] = {}
    doc_fn = root / "core_types.py"
    if doc_fn.exists():
        try:
            clines = doc_fn.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            clines = []
        in_class = False
        for ln in clines:
            if "class ErrorKind" in ln:
                in_class = True
                continue
            if in_class and ln.strip() and not ln.startswith((" ", "\t")) and "=" not in ln:
                break
            if not in_class:
                continue
            m = _ERROR_KIND_DOC.match(ln)
            if m:
                docs[m.group(1)] = m.group(2).strip()

    # 分类点: ErrorKind.<MEMBER> 引用位置
    refs: dict[str, list[str]] = defaultdict(list)
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or py.resolve() == _SELF:
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _ERROR_KIND_REF.finditer(text):
            refs[m.group(1)].append(f"{py.relative_to(root)}:{text[: m.start()].count(chr(10)) + 1}")

    members: list[dict] = []
    for name, doc in docs.items():
        locs = sorted(set(refs.get(name, [])))
        members.append(
            {
                "name": name,
                "doc": doc,
                "points": ", ".join(locs[:3]) + (f" +{len(locs)-3} 处" if len(locs) > 3 else ""),
            }
        )
    return {"members": members}


def render_errors_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("# 错误语义契约 (ErrorKind)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --errors --out docs/errors-contract.md`."
    )
    lines.append(
        "登记 `ToolResult.error_kind` 的分类语义 (core_types.py::ErrorKind)。下游 "
        "(debugging / trace / auto-retry) 据此区分失败类别; 默认 `NONE` 保持既有调用方"
        "行为不变。`points` 是产出该分类的静态扫描位置。映射入口: "
        "`tools/bash_tool.py::_result_error_kind` (returncode==0→NONE, timed_out→TIMEOUT, "
        "blocked→DENIED, 其余→FATAL)。"
    )
    lines.append("")
    lines += _fmt_section(
        "ErrorKind 类别",
        contract["members"],
        ("类别", "语义", "产出点"),
        ("name", "doc", "points"),
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode/Phase 契约 (modes-contract.md)
# ---------------------------------------------------------------------------
# prompt 面的 mode 系统契约: 五种 mode (MODE_INSTRUCTIONS) + 七个 phase
# (PHASE_PROMPTS + PHASE_BUDGETS) + v6 G51 结构关系补充. 回答"agent 有哪些
# mode / 每个 phase 的预算与提示头 / 结构对齐补充落在哪".

_MODE_INSTR_OPEN = re.compile(r"_MODE_INSTRUCTIONS\s*=\s*\{")
_MODE_KEY = re.compile(r'^\s*"([a-z_]+)"\s*:\s*\(')
_G51_KEY = re.compile(r'^\s*"([a-z_]+)"\s*:\s*\(')


def _parse_mode_instructions(root: Path) -> list[tuple[str, str]]:
    """从 prompt_builder.py 源码解析 _MODE_INSTRUCTIONS 的 (mode, 首行说明)."""
    fn = root / "agent" / "prompt_builder.py"
    try:
        lines = fn.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows: list[tuple[str, str]] = []
    in_block = False
    cur: str | None = None
    for ln in lines:
        if _MODE_INSTR_OPEN.search(ln):
            in_block = True
            continue
        if not in_block:
            continue
        m = _MODE_KEY.match(ln)
        if m:
            cur = m.group(1)
            continue
        if cur and ln.strip().startswith('"'):
            rows.append((cur, ln.strip().strip('"').strip()))
            cur = None
            continue
        if ln.strip() == "}":
            break
    return rows


def build_modes_contract(root: Path | None = None) -> dict:
    """构建 mode/phase 契约."""
    root = root or _ROOT
    modes = [{"name": n, "summary": s} for n, s in _parse_mode_instructions(root)]

    # phases / budgets 来自 phases.py (纯 stdlib, 轻量导入安全).
    phases: list[dict] = []
    overlapping = False
    try:
        from huginn.phases import PHASE_BUDGETS, PHASE_PROMPTS, ResearchPhase

        for ph in ResearchPhase:
            budget = PHASE_BUDGETS.get(ph)
            prompt = PHASE_PROMPTS.get(ph, "")
            first_line = ""
            if prompt:
                for ln in prompt.splitlines():
                    if ln.strip():
                        first_line = ln.strip()
                        break
            phases.append(
                {
                    "name": ph.value,
                    "short": ph.name,
                    "budget": budget.max_calls if budget else None,
                    "head": first_line,
                }
            )
    except Exception:
        overlapping = True

    # G51 补充键 (从 prompt_builder 源码)
    g51: list[str] = []
    fn = root / "agent" / "prompt_builder.py"
    try:
        lines = fn.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    in_g51 = False
    for ln in lines:
        if "_PHASE_G51_NOTES" in ln and "=" in ln:
            in_g51 = True
            continue
        if in_g51:
            m = _G51_KEY.match(ln)
            if m:
                g51.append(m.group(1))
            elif ln.strip() == "}":
                break
    return {"modes": modes, "phases": phases, "g51": g51, "import_failed": overlapping}


def render_modes_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("# Mode/Phase 契约 (prompt 面)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --modes --out docs/modes-contract.md`."
    )
    lines.append(
        "登记 prompt 面的 mode 系统: **mode** (MODE_INSTRUCTIONS, agent 顶层行为) 与 "
        "**phase** (PHASE_PROMPTS + PHASE_BUDGETS, 研究流程阶段)。`budget` 是该 phase 的 "
        "工具调用预算 (max_calls); `head` 是 phase 提示头首行; `g51` 是 v6 结构关系语义"
        "对齐补充所覆盖的 phase。"
    )
    lines.append("")
    lines += _fmt_section(
        "Mode",
        contract["modes"],
        ("mode", "行为说明"),
        ("name", "summary"),
    )
    lines += _fmt_section(
        "Phase",
        contract["phases"],
        ("phase", "枚举名", "预算", "提示头首行"),
        ("name", "short", "budget", "head"),
    )
    lines.append("")
    lines.append("G51 结构关系补充覆盖 phase: " + (", ".join(f"`{g}`" for g in contract["g51"]) or "—"))
    lines.append("")
    if contract.get("import_failed"):
        lines.append("> 注: phases.py 导入失败, phase 表为空 (依赖缺失)。")
        lines.append("")
    return "\n".join(lines)


def build_flags_contract(root: Path | None = None) -> list[dict]:
    """构建 feature-flags 契约: 每个 flag 的默认值/描述/旧 env 别名/消费点."""
    root = root or _ROOT
    usage: dict[str, list[str]] = defaultdict(list)
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or str(py).endswith("feature_flags.py"):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _FLAG_IS_ENABLED.finditer(text):
            name = m.group(1)
            lineno = text[: m.start()].count("\n") + 1
            usage[name].append(f"{py.relative_to(root)}:{lineno}")

    from huginn.feature_flags import FeatureFlags

    defaults = FeatureFlags._DEFAULTS
    descs = FeatureFlags._DESCRIPTIONS
    legacy: dict[str, list[str]] = defaultdict(list)
    for env_name, flag in FeatureFlags._ENV_ALIASES.items():
        legacy[flag].append(env_name)

    rows: list[dict] = []
    for name in sorted(defaults):
        rows.append(
            {
                "name": name,
                "default": defaults[name],
                "description": descs.get(name, ""),
                "legacy_env": ", ".join(sorted(legacy.get(name, []))),
                "read_points": sorted(usage.get(name, [])),
            }
        )
    return rows


def render_flags_markdown(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# 功能开关契约 (FeatureFlags)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.config_audit --flags --out docs/feature-flags-contract.md`."
    )
    lines.append(
        "统一登记 agent 的可关增强功能. 默认值来自 `FeatureFlags._DEFAULTS`; "
        "`legacy_env` 是迁移前的旧裸读变量名 (仍兼容, 见 `_ENV_ALIASES`); "
        "`read_points` 是 `is_enabled(...)` 的静态扫描消费点. "
        "优先级: 硬编码默认 < 配置文件 < HUGINN_FEATURE_<NAME> 环境变量 < 运行时 API."
    )
    lines.append("")
    lines.append("| 开关 | 默认 | 描述 | 旧 env 别名 | 消费点 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        reads = ", ".join(r["read_points"][:3])
        if len(r["read_points"]) > 3:
            reads += f" +{len(r['read_points']) - 3} 处"
        if not reads:
            reads = "—"
        legacy = r["legacy_env"] or "—"
        lines.append(
            f"| `{r['name']}` | {r['default']} | {r['description']} | {legacy} | {reads} |"
        )
    lines.append("")
    lines.append(f"共 {len(rows)} 个功能开关。")
    lines.append("")
    return "\n".join(lines)


# 契约模式注册表: flag 名 → (builder, renderer, 默认输出文件名).
# 供 main() 分派, 也供契约漂移测试 (tests/test_config_contracts.py) 复用,
# 保证"文档与代码一致"的校验对象就是实际分派逻辑本身.
CONTRACT_MODES: dict[str, tuple[object, object, str]] = {
    "plugins": (build_plugins_contract, render_plugins_markdown, "plugins-contract.md"),
    "tools": (build_tools_contract, render_tools_markdown, "tools-contract.md"),
    "events": (build_events_contract, render_events_markdown, "events-contract.md"),
    "routes": (build_routes_contract, render_routes_markdown, "routes-contract.md"),
    "errors": (build_errors_contract, render_errors_markdown, "errors-contract.md"),
    "modes": (build_modes_contract, render_modes_markdown, "modes-contract.md"),
    "flags": (build_flags_contract, render_flags_markdown, "feature-flags-contract.md"),
}
# env 契约为默认模式 (无 --xxx 时), 单独登记供漂移测试统一遍历.
ENV_CONTRACT_NAME = "env-contract.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="输出 JSON inventory")
    parser.add_argument(
        "--flags",
        action="store_true",
        help="输出 FeatureFlags 契约 (功能开关面) 而非 env 契约",
    )
    parser.add_argument(
        "--plugins",
        action="store_true",
        help="输出插件契约 (Everything is a Plugin 注册面) 而非 env 契约",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="输出工具契约 (ToolRegistry 运行时枚举) 而非 env 契约",
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="输出事件契约 (EventType + UnifiedBus 发射面) 而非 env 契约",
    )
    parser.add_argument(
        "--routes",
        action="store_true",
        help="输出路由契约 (ModelRouter task→tag) 而非 env 契约",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="输出错误语义契约 (ErrorKind 分类面) 而非 env 契约",
    )
    parser.add_argument(
        "--modes",
        action="store_true",
        help="输出 Mode/Phase 契约 (prompt 面) 而非 env 契约",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="写 markdown 契约到指定文件 (否则打印到 stdout)",
    )
    args = parser.parse_args(argv)

    # 契约模式分派: 每个 --xxx 对应一个 (builder, renderer, 文件名).
    modes = CONTRACT_MODES
    for flag, (builder, renderer, default_name) in modes.items():
        if getattr(args, flag, False):
            data = builder()
            if args.json:
                json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
                print()
                return 0
            md = renderer(data)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(md, encoding="utf-8")
                print(f"wrote {flag} contract -> {out_path}")
            else:
                print(md)
            return 0

    inventory = build_inventory()
    if args.json:
        json.dump(inventory, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    md = render_markdown(inventory)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"wrote {len(inventory)} env contracts -> {out_path}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())