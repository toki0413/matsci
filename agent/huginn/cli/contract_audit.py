"""MECE 契约审计 — 扫"宣称维度零调用者" + "惩罚项重复计数".

为什么需要: `config_audit` 登记"有哪些配置面 / 注册面", 但不检查"宣称了却没人
接". 历史上 anti_hacking 三件套曾是**定义了但零调用者**的死代码 (见
`validation/scope_authority.py` 自述), 而奖励系统里 `reconcile_r_phys` 在
`claim_reward` 与 `security/world_state` 各有一份实现 —— 这类"宣称 vs 接线"缺口
与"同一概念两份实现"正是 MECE 要抓的两类违例:

  - **collectively exhaustive 违例**: 宣称的维度零调用者 (declared but unwired).
  - **mutually exclusive 违例**: 同名跨模块重复实现 / 同一惩罚轴上叠两项.

审计十四面: **奖励面 / 授权面 / 工作流面 / 模式面 / 词汇面 / 工具面 / 钩子面 / 事件面 /
SSE 消费面 / WS 消费面 / HTTP API 消费面 / 请求负载面 / 响应结构面 / WS 请求负载面**.
后十二面是本工具从奖励系统外延到"agent 自身怎么跑"的同类审计:

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
  - **事件面**: 内部 `EventBus` 的**点分事件类型契约** —— **声明面**
    (`events/event_types.py` 的常量 + `ALL_TYPES` 非穷尽清单) ↔ **发布面**
    (`AgentEvent(type=…)` / 内部 `_publish` / `publish_event` / `_emit_campaign`) ↔
    **订阅面** (`EventBus.subscribe`, 含 `ALL` 通配与 `for X in <集合>` 反解). 声明
    类型零发布 = "声明了却永不发生"; 发布/订阅了却未声明的类型 (如 `campaign.retry`)
    是跨模块孤立的字符串契约, 最该补常量.
  - **SSE 消费面**: 事件面查到"事件总线发布了"就为止, 但**发布了不等于被前端消费**.
    本面下潜一层核**通道归属**: **生产面** (三条 SSE 通道的帧名 —— `progress` 取
    `interaction/progress.py` 的字面 `event:` 行, `event_bus` 取总线生产发布的事件值,
    `pet` 只发无名帧) ↔ **消费面** (前端 `desktop/src` 的 `new EventSource(url)` +
    `addEventListener(<帧名>)`). 帧名只在**发它的那条 EventSource** 上才可能命中,
    故监听挂错通道 = "永不触发"; 生产了却零前端监听 = "宣称了却没人接".
  - **WS 消费面**: SSE 消费面只核单向 (后端发帧 → 前端 `addEventListener`), 但
    WebSocket 是**双向**的. 本面按**端点通道** (`agent` / `terminal` / `viewer3d` /
    `hpc`) 核两向: **server→client 生产面** (`send_json({"type": …})` 字面量 +
    三元赋值) ↔ **前端判别面** (`switch (data.type)` / `.type === …`, 仅 `agent`
    按 type 判别); **client→server 生产面** (前端 `send({type: …})`) ↔ **后端分发面**
    (`_MESSAGE_HANDLERS` registry + 分发比较). 前端 case 挂到不发该帧名的端点 =
    "永不触发"; 前端发了后端分发表不认的入站类型 = "回 error 帧".
  - **HTTP API 消费面**: SSE/WS 消费面核的是**流式推送**, 本面补齐**请求-响应**第三块
    传输拼图: **注册生产面** (`huginn/routes/*.py` 的 `@router.<method>(<path>)` 装饰器)
    ↔ **调用消费面** (前端 `api.get/post/put/patch/del/getBlob/upload*/search(...)`).
    桌面只是 HTTP API 的**一个**消费者 (外部客户端 / CLI / 测试也调), 故只把
    "前端 → 后端"当硬契约: 前端调了后端没注册的路径 = 404 死链, 方法对不上 = 405;
    反向"后端注册但桌面零调用"按模块聚合列候选 (结构性常态). 另核路由挂载面
    (`ALL_ROUTERS` ↔ 各模块 `APIRouter`) 与同 method+path 多模块注册 (路由遮蔽).
  - **请求负载面**: HTTP 消费面核「路径 + 方法」挂不挂得上 (404/405), 本面在**已命中
    端点**上核负载: 后端签名里的必填 query / 必填请求体 / Pydantic 模型必填字段 /
    `Form · File` 字段 ↔ 前端这次 `api.*` 调用发的实参. 只把「前端漏发后端必填」当
    违例 (422 死负载); 反向 (可选字段没发) 不是违例.
  - **响应结构面**: 请求负载面核「前端发的后端要不要求」, 本面反向核「后端返的前端读
    不读得到」: 前端 `api.*<T>` 泛型声明的响应字段 ↔ 后端处理函数 `return` 字面量 /
    `response_model` 实际返回字段. 只把「前端声明要读的字段后端从不返回」当违例
    (恒 undefined, 静默坏); 独有形状解析: 忽略嵌套 `def`/`lambda` 的 return、递归解析
    同模块 helper、识别 `if err: return err` 真值守卫排除 null 分支.
  - **WS 请求负载面**: WS 消费面只核「入站 `type` 认不认」, 本面再往里一层核 agent 通道
    入站消息的**负载字段**, 权威是 `WSMessage` Pydantic 模型. 两向硬违例: 后端 handler
    读 `msg.<X>` 而模型未声明 `X` (AttributeError 死帧) / 前端发该 type 时带了模型未
    声明字段 (被静默丢弃). 只核 agent 通道 (其余通道入站负载走原始 dict).
  - **SSE 事件负载面**: SSE 消费面只核「帧名认不认」, 本面再往里一层核帧 payload 的
    **顶层键**, 权威是后端发帧处 `json.dumps(<expr>)` 的 `<expr>` 形状 (`progress` 取
    `to_dict()` / campaign `evt`, `event_bus` 取 `AgentEvent.to_sse()` 信封). 硬方向:
    前端在该帧处理函数里读 `t.<X>` 而后端该帧 payload 从不发此顶层字段 (恒 undefined).
    只核顶层键 (`t.data.<X>` 的嵌套子形状由发布点决定, 不核).

本工具只做**静态扫描 + 少量运行时读取**并**提示候选**, 不判死: "同轴/同名/词表
不一致"是可疑信号, 是否真缺陷需人工判定 (例如 efficiency_discount 按"首次全对
轮次"打折, idle_turn_penalty 按"达成后多余轮次"扣分 —— 同属轮次轴但语义有别;
fusion 模式经 set_mode('research') 复用 CSM S3 是**有意设计**, 非漏接).

用法:
    python -m huginn.cli.contract_audit                  # 打印十五面审计
    python -m huginn.cli.contract_audit --reward         # 只看奖励面
    python -m huginn.cli.contract_audit --scope          # 只看授权面
    python -m huginn.cli.contract_audit --workflow       # 只看工作流面
    python -m huginn.cli.contract_audit --modes          # 只看模式面
    python -m huginn.cli.contract_audit --vocab          # 只看词汇面
    python -m huginn.cli.contract_audit --tools          # 只看工具面
    python -m huginn.cli.contract_audit --hooks          # 只看钩子面
    python -m huginn.cli.contract_audit --events         # 只看事件面
    python -m huginn.cli.contract_audit --sse            # 只看 SSE 消费面
    python -m huginn.cli.contract_audit --ws             # 只看 WS 消费面
    python -m huginn.cli.contract_audit --http           # 只看 HTTP API 消费面
    python -m huginn.cli.contract_audit --payload        # 只看请求负载面
    python -m huginn.cli.contract_audit --response       # 只看响应结构面
    python -m huginn.cli.contract_audit --ws-payload     # 只看 WS 请求负载面
    python -m huginn.cli.contract_audit --sse-payload    # 只看 SSE 事件负载面
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
from collections.abc import Iterator
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
# 事件面: 事件类型声明面 vs 发布面 vs 订阅面 (内部 EventBus)
# ---------------------------------------------------------------------------
#
# 钩子面审 HookManager 的**无点分**字符串事件 (register 强校验, 静默 trigger);
# 这一面审**内部 EventBus** 的**点分**事件类型 (`events/event_types.py` 的
# `TOOL_CALL = "tool.call"` …):
#
#   - **声明面**: `event_types.py` 的点分常量 + `ALL_TYPES` frozenset (自述
#     "非穷尽, 仅辅助排错", 不做校验) + `ALL = "*"` 通配.
#   - **发布面**: `AgentEvent(type=…)`、内部 `_publish` / `_publish_internal(_sync)` /
#     `publish_generic_sync` / `publish_event(_sync)` / `_emit_campaign`, 以及首参
#     字面量等于已声明值的本地 emit 包装 (`_emit("snapshot.take", …)`).
#   - **订阅面**: `EventBus.subscribe(<类型>, cb)`. 首参可为常量、等值字面量、`ALL`
#     通配 (audit log 全量订阅), 或 `for X in <常量集合>` 的循环变量 (反解集合元素).
#
# MECE 两原则落到事件:
#   - **collectively exhaustive**: 声明的事件类型须有**生产发布点** —— 零发布即
#     "声明了却永不发生" (对偶钩子面). 另记发布/订阅了却**未声明**的类型: `ALL_TYPES`
#     自述允许非穷尽, 故按"候选登记"报, 不判死; 其中既发布又订阅的 (如
#     `campaign.retry`/`campaign.suspect`) 是跨模块孤立的字符串契约, 最该补常量.
#   - **mutually exclusive**: 类型常量值两两不同 (撞值会让订阅者收到错类型);
#     `ALL_TYPES` 与常量定义面双向一致.
#
# 归属口径同钩子面: 只认 `from huginn.events.event_types import <CONST>` (含别名) 与
# `<mod>.<CONST>` (mod 绑定到 event_types), 不认裸同名. 定义文件自身预置本地常量表.
#
# 诚实边界: `EventBus.publish` **不校验**类型 (与钩子面 `register` 抛错相反), 任意
# 点分字符串都能发, 故"未声明发布"只能提示、无法判错; 订阅若走变量/前缀匹配等间接
# 形式, 静态解析不到, 不计入.

_EVENTS_MODULE = "huginn/events/event_types.py"
# 事件值形如 `group.name` (至少一段点分, 小写蛇形); 排除 `ALL = "*"`.
_EVENT_VALUE_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
# 已知发布入口 (按方法名); 另有"首参字面量 = 已声明值"的启发式兜本地 emit 包装.
_EVENT_PUBLISH_CALLS = frozenset({
    "AgentEvent",
    "_publish",
    "_publish_internal",
    "_publish_internal_sync",
    "publish_generic_sync",
    "publish_event",
    "publish_event_sync",
    "_emit_campaign",
})
_EVENT_SUBSCRIBE_CALLS = frozenset({"subscribe"})
_EVENT_ALL_CONST = "ALL"


def _event_declarations(root: Path) -> dict:
    """读声明面: 点分事件常量 (名→值)、`ALL_TYPES` 成员、以及两侧缺口."""
    tree = _parse(root / _EVENTS_MODULE)
    if tree is None:
        return {
            "consts": {},
            "members": [],
            "not_in_all_types": [],
            "unresolved_members": [],
        }
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
        if (
            name == "ALL_TYPES"
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and value.args
            and isinstance(value.args[0], ast.Set)
        ):
            members = [e.id for e in value.args[0].elts if isinstance(e, ast.Name)]
        elif (
            name.isupper()
            and name != _EVENT_ALL_CONST
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and _EVENT_VALUE_RE.match(value.value)
        ):
            consts[name] = value.value
    not_in_all_types = sorted(c for c in consts if c not in members)
    unresolved_members = sorted(m for m in members if m not in consts)
    return {
        "consts": consts,
        "members": members,
        "not_in_all_types": not_in_all_types,
        "unresolved_members": unresolved_members,
    }


def _event_bindings(
    tree: ast.Module, rel: str, decl: dict[str, str]
) -> tuple[dict[str, str], set[str]]:
    """模块内解析: 事件常量本地名 → 常量名; 及绑到 event_types 的模块别名."""
    direct: dict[str, str] = (
        {name: name for name in decl} if rel == _EVENTS_MODULE else {}
    )
    mod_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = _resolve_relative(rel, node.level, node.module or "")
            for a in node.names:
                if mod == "huginn.events.event_types":
                    direct[a.asname or a.name] = a.name
                elif f"{mod}.{a.name}" == "huginn.events.event_types":
                    mod_aliases.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "huginn.events.event_types":
                    mod_aliases.add(a.asname or "event_types")
    return direct, mod_aliases


def _event_container_elts(value: ast.AST) -> list[ast.AST] | None:
    """取元组/列表/集合字面量的元素; `frozenset({…})` 等包装也拆开."""
    if isinstance(value, ast.Tuple | ast.List | ast.Set):
        return list(value.elts)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"frozenset", "set", "list", "tuple"}
        and value.args
        and isinstance(value.args[0], ast.Tuple | ast.List | ast.Set)
    ):
        return list(value.args[0].elts)
    return None


def _resolve_event_type(
    arg: ast.AST, direct: dict[str, str], mod_aliases: set[str], decl: dict[str, str]
) -> tuple[str | None, str | None]:
    """把发布/订阅首参解析成 (常量名|None, 值|None). 字面量 → (None, 原始串)."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return None, arg.value
    if isinstance(arg, ast.Name) and arg.id in direct:
        name = direct[arg.id]
        return name, decl.get(name)
    if (
        isinstance(arg, ast.Attribute)
        and isinstance(arg.value, ast.Name)
        and arg.value.id in mod_aliases
    ):
        return arg.attr, decl.get(arg.attr)
    return None, None


def _event_collections(
    tree: ast.Module, direct: dict[str, str], mod_aliases: set[str], decl: dict[str, str]
) -> dict[str, set[str]]:
    """收集 `X = (<事件值>, …)` (含 `frozenset({…})`) —— 供 for 循环变量反解."""
    colls: dict[str, set[str]] = {}
    for node in ast.walk(tree):
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
        if name is None or value is None:
            continue
        elts = _event_container_elts(value)
        if elts is None:
            continue
        vals: set[str] = set()
        for e in elts:
            _n, v = _resolve_event_type(e, direct, mod_aliases, decl)
            if v is not None and _EVENT_VALUE_RE.match(v):
                vals.add(v)
        if vals:
            colls[name] = vals
    return colls


def _event_loop_vars(
    tree: ast.Module,
    colls: dict[str, set[str]],
    direct: dict[str, str],
    mod_aliases: set[str],
    decl: dict[str, str],
) -> dict[str, set[str]]:
    """`for X in <常量集合>` 的 X → 该集合的事件值集合 (反解动态订阅)."""
    loop_vars: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        it = node.iter
        vals: set[str] = set()
        if isinstance(it, ast.Name) and it.id in colls:
            vals = set(colls[it.id])
        else:
            elts = _event_container_elts(it)
            if elts is not None:
                for e in elts:
                    _n, v = _resolve_event_type(e, direct, mod_aliases, decl)
                    if v is not None and _EVENT_VALUE_RE.match(v):
                        vals.add(v)
        if vals:
            loop_vars.setdefault(node.target.id, set()).update(vals)
    return loop_vars


def _event_arg_values(
    arg: ast.AST,
    direct: dict[str, str],
    mod_aliases: set[str],
    decl: dict[str, str],
    loop_vars: dict[str, set[str]],
) -> tuple[set[str], str | None]:
    """解析发布/订阅首参 → (事件值集合, 通配标记). 未解析返回 (set(), None)."""
    if isinstance(arg, ast.Name) and arg.id in loop_vars and arg.id not in direct:
        return set(loop_vars[arg.id]), None
    if isinstance(arg, ast.Constant) and arg.value == "*":
        return set(), _EVENT_ALL_CONST
    name, value = _resolve_event_type(arg, direct, mod_aliases, decl)
    if name == _EVENT_ALL_CONST:
        return set(), _EVENT_ALL_CONST
    if value is not None and _EVENT_VALUE_RE.match(value):
        return {value}, None
    return set(), None


def _event_type_kwarg(call: ast.Call) -> ast.AST | None:
    """取 `AgentEvent(type=…)` 的 type 值, 退化为首位置参."""
    for kw in call.keywords:
        if kw.arg == "type":
            return kw.value
    return call.args[0] if call.args else None


def _event_wiring(root: Path) -> dict:
    """扫发布/订阅点, 按事件值归集 prod/test 位置."""
    decl = _event_declarations(root)["consts"]
    declared_values = set(decl.values())
    wiring: dict[str, dict[str, list[str]]] = {}
    wildcard: list[str] = []
    for py in _iter_py(root):
        rel = py.relative_to(root).as_posix()
        tree = _parse(py)
        if tree is None:
            continue
        direct, mod_aliases = _event_bindings(tree, rel, decl)
        colls = _event_collections(tree, direct, mod_aliases, decl)
        loop_vars = _event_loop_vars(tree, colls, direct, mod_aliases, decl)
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
            arg: ast.AST | None = None
            kind: str | None = None
            if method in _EVENT_SUBSCRIBE_CALLS:
                if not node.args:
                    continue
                kind, arg = "subscribe", node.args[0]
            elif method == "AgentEvent":
                kind, arg = "publish", _event_type_kwarg(node)
            elif method in _EVENT_PUBLISH_CALLS:
                kind, arg = "publish", (node.args[0] if node.args else None)
            elif (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value in declared_values
            ):
                # 启发式: 首参字面量 = 已声明值 ⇒ 本地 emit 包装 (`_emit("snapshot.take", …)`).
                kind, arg = "publish", node.args[0]
            if arg is None or kind is None:
                continue
            values, wild = _event_arg_values(arg, direct, mod_aliases, decl, loop_vars)
            if wild is not None:
                if kind == "subscribe":
                    wildcard.append(f"{rel}:{node.lineno}")
                continue
            for v in values:
                rec = wiring.setdefault(
                    v,
                    {
                        "prod_publish": [],
                        "prod_subscribe": [],
                        "test_publish": [],
                        "test_subscribe": [],
                    },
                )
                rec[f"{bucket}_{kind}"].append(f"{rel}:{node.lineno}")
    return {
        "wiring": wiring,
        "declared_values": declared_values,
        "wildcard_subscribers": sorted(wildcard),
    }


def build_event_contract(root: Path | None = None) -> dict:
    """事件面: 事件类型声明面 ↔ 发布面 ↔ 订阅面是否穷尽一致."""
    root = root or _REPO
    decl = _event_declarations(root)
    scan = _event_wiring(root)
    wiring = scan["wiring"]
    declared_values = scan["declared_values"]

    events: list[dict] = []
    for const in decl["members"]:
        value = decl["consts"].get(const, "")
        w = wiring.get(value, {})
        prod_p = len(w.get("prod_publish", []))
        prod_s = len(w.get("prod_subscribe", []))
        test_p = len(w.get("test_publish", []))
        test_s = len(w.get("test_subscribe", []))
        if prod_p:
            status = "published"
            note = "" if prod_s else "仅发布, 无 `.subscribe` 消费者 (外部 SSE 按字符串匹配)"
        elif prod_s:
            status = "subscribed-only"
            note = "有 `.subscribe` 却零生产发布 —— 订阅永不发生"
        else:
            status = "dead"
            note = "声明了却零生产发布零订阅"
        events.append(
            {
                "const": const,
                "value": value,
                "prod_publish": prod_p,
                "prod_subscribe": prod_s,
                "test_publish": test_p,
                "test_subscribe": test_s,
                "status": status,
                "note": note,
                "publish_sites": w.get("prod_publish", []),
                "subscribe_sites": w.get("prod_subscribe", []),
            }
        )

    collisions: list[dict] = []
    by_value: dict[str, list[str]] = defaultdict(list)
    for const, value in decl["consts"].items():
        by_value[value].append(const)
    for value, consts in sorted(by_value.items()):
        if len(consts) > 1:
            collisions.append({"value": value, "consts": sorted(consts)})

    undeclared: list[dict] = []
    for value in sorted(v for v in wiring if v not in declared_values):
        w = wiring[value]
        if not (w["prod_publish"] or w["prod_subscribe"]):
            continue
        undeclared.append(
            {
                "value": value,
                "prod_publish": len(w["prod_publish"]),
                "prod_subscribe": len(w["prod_subscribe"]),
                "test_publish": len(w["test_publish"]),
                "test_subscribe": len(w["test_subscribe"]),
                "publish_sites": w["prod_publish"],
                "subscribe_sites": w["prod_subscribe"],
            }
        )

    return {
        "module": _EVENTS_MODULE,
        "events": events,
        "collisions": collisions,
        "not_in_all_types": decl["not_in_all_types"],
        "unresolved_members": decl["unresolved_members"],
        "undeclared": undeclared,
        "wildcard_subscribers": scan["wildcard_subscribers"],
    }


_EVENT_STATUS_DOC = {
    "published": "有生产发布点",
    "subscribed-only": "有订阅但零生产发布 (订阅永不发生)",
    "dead": "声明零发布零订阅",
}


def render_event_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 事件面: 事件类型声明面 vs 发布面 vs 订阅面")
    lines.append("")
    lines.append(
        f"声明面: `{contract['module']}` 的 {len(contract['events'])} 个点分事件类型"
        "(`ALL_TYPES` 为自述非穷尽的辅助清单). 发布面: `AgentEvent(type=…)` / 内部 "
        "`_publish(_internal)` / `publish_generic_sync` / `publish_event` / `_emit_campaign`. "
        "订阅面: `EventBus.subscribe(<类型>, cb)` (含 `ALL` 通配与 `for X in <集合>` 反解). "
        "`subscribed-only` = 订阅了却零发布 (订阅永不发生); `dead` = 声明了却零发布零订阅."
    )
    lines.append("")
    lines.append("状态: " + "; ".join(f"`{k}`={v}" for k, v in _EVENT_STATUS_DOC.items()))
    lines.append("")
    lines.append("| 事件常量 | 值 | 生产发布 | 生产订阅 | 测试发布 | 测试订阅 | 状态 | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in contract["events"]:
        lines.append(
            f"| `{e['const']}` | `{e['value']}` | {e['prod_publish']} | {e['prod_subscribe']} "
            f"| {e['test_publish']} | {e['test_subscribe']} | `{e['status']}` | {e['note']} |"
        )
    lines.append("")

    lines.append("### 未声明类型 (发布/订阅了却无常量)")
    lines.append("")
    if contract["undeclared"]:
        lines.append("| 事件值 | 生产发布 | 生产订阅 | 发布点 | 订阅点 |")
        lines.append("|---|---|---|---|---|")
        for u in contract["undeclared"]:
            pub = ", ".join(f"`{s}`" for s in u["publish_sites"]) or "—"
            sub = ", ".join(f"`{s}`" for s in u["subscribe_sites"]) or "—"
            lines.append(
                f"| `{u['value']}` | {u['prod_publish']} | {u['prod_subscribe']} | {pub} | {sub} |"
            )
    else:
        lines.append("- 无 —— 发布/订阅的类型均已登记为常量.")
    lines.append("")

    lines.append("### 互斥违例 + 声明缺口 (mutually exclusive)")
    lines.append("")
    if contract["collisions"]:
        for c in contract["collisions"]:
            lines.append(
                f"- ⚠️ 事件类型常量撞值 `{c['value']}`: "
                + ", ".join(f"`{n}`" for n in c["consts"])
            )
    else:
        lines.append("- 事件类型常量值两两不同 —— 无撞值.")
    if contract["not_in_all_types"]:
        lines.append(
            "- ⚠️ 常量声明了却不在 `ALL_TYPES`: "
            + ", ".join(f"`{c}`" for c in contract["not_in_all_types"])
        )
    if contract["unresolved_members"]:
        lines.append(
            "- ⚠️ `ALL_TYPES` 成员无对应常量定义: "
            + ", ".join(f"`{c}`" for c in contract["unresolved_members"])
        )
    if not contract["not_in_all_types"] and not contract["unresolved_members"]:
        lines.append("- `ALL_TYPES` 与事件类型常量定义面双向一致.")
    wildcard = contract["wildcard_subscribers"]
    if wildcard:
        lines.append(
            "- `ALL` 通配订阅 (收全量, 覆盖上表所有类型): "
            + ", ".join(f"`{s}`" for s in wildcard)
        )
    lines.append("")
    lines.append(
        "诚实边界: `EventBus.publish` **不校验**类型 (与钩子面 `register` 抛错相反), "
        "任意点分字符串都能发, 故「未声明发布」只作候选提示; 订阅走变量/前缀匹配等间接"
        "形式时静态解析不到, 不计入."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SSE 消费面: 后端帧名 ↔ 前端 EventSource 监听 (按通道归属)
# ---------------------------------------------------------------------------

# 前端源码根: 默认取仓根的兄弟目录 `desktop/src` (agent/ 与 desktop/ 平级).
_FRONTEND_REL = "desktop/src"
# 已知 SSE 通道 (URL 片段 → 通道名). 通道 = 哪条 EventSource, 帧名只在发它的
# 那条 EventSource 上才可能被 `addEventListener` 命中.
_SSE_CHANNELS: dict[str, str] = {
    "/tasks/stream": "progress",
    "/events/stream": "event_bus",
    "/events": "pet",
}
# progress 通道的帧名 = 该模块字面 `f"event: <name>"`; event_bus 通道的帧名 =
# 总线生产发布的事件值 (`AgentEvent.to_sse` 写成 `event: <type>`); pet 通道只发
# 无名帧 (`data:` 无 `event:`), 故无命名帧.
_SSE_FRAME_MODULE = "huginn/interaction/progress.py"
_SSE_FRAME_RE = re.compile(r'event:\s*([a-zA-Z_][\w.]*)')
# campaign 通道 payload 事件名的一部分静态可见: `emit_campaign_event(event_type="…")`.
_SSE_CAMPAIGN_EMIT_RE = re.compile(r'event_type\s*=\s*["\']([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)["\']')

# 前端行级模式. 无 TS 解析器, 只做确定性行匹配 (边界见文档"诚实边界").
_FE_ES_ASSIGN = re.compile(r"(\w+)\s*=\s*new\s+EventSource\(\s*(.+)$")
_FE_LISTEN = re.compile(r"(\w+)\.addEventListener\(\s*[\"']([^\"']+)[\"']\s*,\s*(\w+)")
_FE_LISTEN_IDENT = re.compile(r"(\w+)\.addEventListener\(\s*(\w+)\s*,\s*(\w+)")
_FE_HANDLER = re.compile(r"const\s+(\w+)\s*=\s*\(\s*\w+\s*:\s*MessageEvent\s*\)\s*=>\s*\{")
_FE_ARRAY = re.compile(r"^\s*\[(.+)\]\s*$")
_FE_STR = re.compile(r"[\"']([^\"']+)[\"']")
_FE_PAYLOAD = re.compile(r"(?:case\s+|===?\s*)[\"']([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)[\"']")

_SSE_STATUS_DOC = {
    "wired": "该通道确实发此帧名",
    "channel-mismatch": "帧名存在但走别的通道 (监听挂错 EventSource, 永不触发)",
    "no-source": "后端任何 SSE 通道都不发此帧名",
    "external": "非 SSE 通道 (window/document 等 DOM 事件), 不计入",
}


def _display(frontend: Path, p: Path) -> str:
    """前端文件路径尽量相对仓根父目录显示 (desktop/src/…), 否则退化为相对前端根."""
    try:
        return p.relative_to(_REPO.parent).as_posix()
    except ValueError:
        return p.relative_to(frontend).as_posix()


def _scan_ts_source(rel: str, lines: list[str]) -> tuple[list[dict], list[dict]]:
    """行级扫一个 TS/TSX 源文件 → (帧监听点, payload 匹配点).

    归属链: 帧监听 → 所在 `EventSource` 变量的最近一次 `new EventSource(url)`;
    payload 字面量 → 所在 `(e: MessageEvent) => {}` 处理函数 → 该函数注册处的事件源.
    """
    chan_keys = sorted(_SSE_CHANNELS, key=len, reverse=True)
    assigns: list[tuple[int, str, str | None]] = []
    for i, ln in enumerate(lines):
        m = _FE_ES_ASSIGN.search(ln)
        if m:
            arg = m.group(2)
            ch = next((_SSE_CHANNELS[k] for k in chan_keys if k in arg), None)
            assigns.append((i, m.group(1), ch))

    def chan_of(var: str, idx: int) -> str | None:
        found = None
        for ai, av, ac in assigns:
            if av == var and ai <= idx:
                found = ac
        return found

    frames: list[dict] = []
    regs: list[tuple[int, str, str | None]] = []
    for i, ln in enumerate(lines):
        m = _FE_LISTEN.search(ln)
        if m:
            var, frame, handler = m.group(1), m.group(2), m.group(3)
            frames.append(
                {
                    "rel": rel,
                    "line": i + 1,
                    "var": var,
                    "frame": frame,
                    "channel": chan_of(var, i),
                }
            )
            regs.append((i, var, handler))
            continue
        m2 = _FE_LISTEN_IDENT.search(ln)
        if m2:
            var, handler = m2.group(1), m2.group(3)
            # `[ "a.b", "c.d" ].forEach((ev) => es.addEventListener(ev, h))`:
            # 帧名在上一行的数组字面量里, 通道取 forEach 那一行的事件源.
            for j in range(i - 1, max(-1, i - 6), -1):
                am = _FE_ARRAY.match(lines[j])
                if am:
                    for s in _FE_STR.findall(am.group(1)):
                        frames.append(
                            {
                                "rel": rel,
                                "line": j + 1,
                                "var": var,
                                "frame": s,
                                "channel": chan_of(var, i),
                            }
                        )
                    break
            regs.append((i, var, handler))

    handlers = [
        (i, m.group(1)) for i, ln in enumerate(lines) if (m := _FE_HANDLER.search(ln))
    ]

    def handler_at(idx: int) -> str | None:
        found = None
        for hi, hn in handlers:
            if hi <= idx:
                found = hn
        return found

    def chan_of_handler(name: str) -> str | None:
        for ri, rv, rh in regs:
            if rh == name:
                return chan_of(rv, ri)
        return None

    payloads: list[dict] = []
    for i, ln in enumerate(lines):
        for m in _FE_PAYLOAD.finditer(ln):
            h = handler_at(i)
            payloads.append(
                {
                    "rel": rel,
                    "line": i + 1,
                    "value": m.group(1),
                    "channel": chan_of_handler(h) if h else None,
                }
            )
    return frames, payloads


def _frontend_listen_sites(frontend: Path) -> dict:
    """扫前端 `.ts`/`.tsx`: 帧监听点与 payload 匹配点 (带通道归属)."""
    frames: list[dict] = []
    payloads: list[dict] = []
    if not frontend.is_dir():
        return {"frames": frames, "payloads": payloads}
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        f, pl = _scan_ts_source(_display(frontend, p), lines)
        frames.extend(f)
        payloads.extend(pl)
    return {"frames": frames, "payloads": payloads}


def _sse_frame_emitters(root: Path) -> dict[str, list[str]]:
    """各 SSE 通道的帧名生产面 (progress 取字面帧名, event_bus 取总线发布值)."""
    out: dict[str, list[str]] = {name: [] for name in _SSE_CHANNELS.values()}
    out["progress"] = sorted(set(_SSE_FRAME_RE.findall(_read(root, _SSE_FRAME_MODULE))))
    wiring = _event_wiring(root)["wiring"]
    out["event_bus"] = sorted(v for v, w in wiring.items() if w["prod_publish"])
    return out


def _campaign_payload_literals(root: Path) -> set[str]:
    """`event_type="…"` 字面量 → campaign 通道 payload 事件名 (静态可见的部分)."""
    vals: set[str] = set()
    for py in _iter_py(root):
        src = py.read_text(encoding="utf-8", errors="replace")
        vals.update(_SSE_CAMPAIGN_EMIT_RE.findall(src))
    return vals


def build_sse_contract(root: Path | None = None, frontend: Path | None = None) -> dict:
    """SSE 消费面: 后端帧名生产面 ↔ 前端 EventSource 监听消费面 (按通道归属)."""
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    emitters = _sse_frame_emitters(root)
    all_frames = {f for fs in emitters.values() for f in fs}
    declared = set(_event_declarations(root)["consts"].values())
    campaign = _campaign_payload_literals(root)
    scan = _frontend_listen_sites(frontend)

    listeners: list[dict] = []
    for site in scan["frames"]:
        ch, frame = site["channel"], site["frame"]
        if ch is None:
            status, note = "external", "非 SSE 通道 (DOM 事件), 不计入"
        elif frame in set(emitters.get(ch, [])):
            status, note = "wired", ""
        elif frame in all_frames:
            others = sorted(k for k, v in emitters.items() if k != ch and frame in v)
            status = "channel-mismatch"
            note = f"此通道不发该帧名, 实际由 {'/'.join(others)} 通道发出"
        else:
            status, note = "no-source", "后端任何 SSE 通道都不发此帧名"
        listeners.append({**site, "status": status, "note": note})

    consumed = {s["frame"] for s in scan["frames"] if s["channel"] is not None}
    payload_values = {p["value"] for p in scan["payloads"]}
    zero_consumer: list[dict] = []
    for ch in _SSE_CHANNELS.values():
        for f in emitters[ch]:
            if f in consumed:
                continue
            zero_consumer.append(
                {"channel": ch, "frame": f, "via_payload": f in payload_values}
            )

    payloads: list[dict] = []
    for p in scan["payloads"]:
        v = p["value"]
        if v in emitters["event_bus"]:
            source = "bus"
        elif v in campaign:
            source = "campaign"
        elif v in declared:
            source = "declared"
        else:
            source = "unknown"
        payloads.append({**p, "source": source})

    return {
        "frontend": str(frontend),
        "channels": {k: {"url": k, "frames": emitters[k]} for k in _SSE_CHANNELS.values()},
        "listeners": listeners,
        "zero_consumer": zero_consumer,
        "payloads": payloads,
        "campaign_literals": sorted(campaign),
    }


def render_sse_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## SSE 消费面: 后端帧名生产面 vs 前端 EventSource 监听面")
    lines.append("")
    lines.append(
        "后端有三条 SSE 通道, 帧名 (`event:` 行) 只在**发它的那条 EventSource** 上才可能"
        "被 `addEventListener(<帧名>)` 命中 —— 所以监听必须**按通道**核: `progress`"
        "(`/tasks/stream`, 帧名来自 `interaction/progress.py` 的字面 `event:` 行), "
        "`event_bus` (`/events/stream`, 帧名 = `AgentEvent.to_sse()` 写入的事件类型值), "
        "`pet` (`/events`, 只发无名帧, 无命名帧)."
    )
    lines.append("")
    lines.append("状态: " + "; ".join(f"`{k}`={v}" for k, v in _SSE_STATUS_DOC.items()))
    lines.append("")
    lines.append("| 通道 | URL 片段 | 生产帧名 |")
    lines.append("|---|---|---|")
    for k, v in contract["channels"].items():
        frames = ", ".join(f"`{f}`" for f in v["frames"]) or "— (无名帧)"
        lines.append(f"| `{k}` | `{v['url']}` | {frames} |")
    lines.append("")

    lines.append("### 前端帧监听 (按通道归属)")
    lines.append("")
    lines.append("| 帧名 | 通道 | 状态 | 位置 | 备注 |")
    lines.append("|---|---|---|---|---|")
    for s in sorted(contract["listeners"], key=lambda x: (x["status"], x["rel"], x["line"])):
        ch = f"`{s['channel']}`" if s["channel"] else "—"
        lines.append(
            f"| `{s['frame']}` | {ch} | `{s['status']}` | `{s['rel']}:{s['line']}` "
            f"| {s['note']} |"
        )
    lines.append("")

    lines.append("### 生产帧名零前端监听 (候选)")
    lines.append("")
    zero = contract["zero_consumer"]
    if zero:
        for z in zero:
            tail = " (经 payload 字段消费)" if z["via_payload"] else ""
            lines.append(f"- `{z['channel']}` / `{z['frame']}`{tail}")
    else:
        lines.append("- 无 —— 每个生产帧名都有前端监听.")
    lines.append("")

    lines.append("### 前端 payload 字段匹配的事件名")
    lines.append("")
    if contract["payloads"]:
        lines.append("| 事件名 | 消费通道 | 生产面 | 位置 |")
        lines.append("|---|---|---|---|")
        for p in contract["payloads"]:
            ch = f"`{p['channel']}`" if p["channel"] else "—"
            lines.append(
                f"| `{p['value']}` | {ch} | `{p['source']}` | `{p['rel']}:{p['line']}` |"
            )
    else:
        lines.append("- 无")
    lines.append("")
    lines.append(
        "生产面取值: `bus` = 总线生产发布 (双通道之一); `campaign` = "
        "`emit_campaign_event(event_type=\"…\")` 静态可见的字面量; `declared` = 已声明常量但未"
        "观测到生产发布; `unknown` = 静态不可见 (如 `f\"campaign.{name}\"` 动态拼接)."
    )
    lines.append("")
    lines.append(
        "诚实边界: 前端是 TS, 本工具只做**行级**匹配 (`new EventSource` / `addEventListener` / "
        "`case …:` / `=== …`), 不做 TS 语法分析 —— 经变量中转的帧名、`es.onmessage` 的无名帧、"
        "动态拼接的通道 URL 都解析不到; campaign payload 里 `f\"campaign.{name}\"` 这类动态名"
        "同样不可穷尽, 故 `unknown` 只提示不判死."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SSE 事件负载面: 后端帧 payload 顶层键 ↔ 前端 JSON.parse(e.data) 顶层读取
# ---------------------------------------------------------------------------

# event_bus 通道 payload = `AgentEvent.to_sse()` 的固定信封 (字面 dict, 封闭).
_SSE_PAYLOAD_EVENT_BUS_MODULE = "huginn/events/event_bus.py"
_SSE_PAYLOAD_TO_SSE = "to_sse"
# 前端: `const t = JSON.parse(e.data)` → 顶层 `t.<字段>` 读取; 内联事件处理函数.
_FE_JSON_PARSE = re.compile(r"(\w+)\s*=\s*JSON\.parse\(\s*(\w+)\.data\s*\)")
# 前端 `<var>.<字段>` 顶层读取 (可带 `?.`); 组1=变量, 组2=字段. SSE 与 WS 事件负载面共用.
_FE_DOT_FIELD_READ = re.compile(r"\b(\w+)\s*\??\.\s*([A-Za-z_$][\w$]*)")
_FE_LISTEN_INLINE_HANDLER = re.compile(
    r"(\w+)\.addEventListener\(\s*[\"']([^\"']+)[\"']\s*,\s*\(\s*\w+\s*:\s*MessageEvent\s*\)\s*=>\s*\{"
)

_SSE_PAYLOAD_KIND_DOC = {
    "read-undeclared": (
        "前端读 `t.<字段>` 而后端该帧 payload 从不发此顶层字段 (恒 undefined, 静默坏)"
    )
}

_SSE_PAYLOAD_TRIAGE_DOC = {
    "defect": "已确认缺陷 (待修)",
    "intentional": "已确认有意",
}

# 已确认分诊表. 键 (通道, 帧名, 字段) → (标签, 理由); 未登记即"待分诊",
# 回归测试会失败 (逼逐条人工判定). 空表 = 当前前端读的顶层字段都合契约.
_SSE_PAYLOAD_CONFIRMED: dict[tuple[str, str, str], tuple[str, str]] = {}


def _sse_payload_triage(channel: str, frame: str, field: str) -> tuple[str, str] | None:
    """SSE 负载违例分诊: 返回 (标签, 理由); 未登记则 None (待人工确认)."""
    return _SSE_PAYLOAD_CONFIRMED.get((channel, frame, field))


def _sse_payload_violation_mark(channel: str, frame: str, field: str) -> str:
    tri = _sse_payload_triage(channel, frame, field)
    if tri is None:
        return " — ⚠ 待分诊"
    doc = _SSE_PAYLOAD_TRIAGE_DOC[tri[0]]
    return f" — {'⛔' if tri[0] == 'defect' else '✅'} {doc}: {tri[1]}"


def _dict_literal_keys(node: ast.AST | None) -> set[str] | None:
    """`ast.Dict` 顶层字面键集; 含 `**` 展开或非常量键 → None (开放形状)."""
    if not isinstance(node, ast.Dict):
        return None
    keys: set[str] = set()
    for k in node.keys:
        if k is None or not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            return None
        keys.add(k.value)
    return keys


def _func_return_dict_keys(tree: ast.AST, name: str) -> set[str] | None:
    """同模块函数 `name` 顶层 `return {…}` 的字面键集 (无则 None)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    return _dict_literal_keys(stmt.value)
    return None


def _func_local_dict_keys(tree: ast.AST, func: str, var: str) -> set[str] | None:
    """函数 `func` 里 `var = {…}` 的字面键集 (无则 None)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Assign)
                    and isinstance(sub.targets[0], ast.Name)
                    and sub.targets[0].id == var
                ):
                    return _dict_literal_keys(sub.value)
    return None


def _sse_yield_frame_expr(value: ast.AST) -> tuple[str | None, ast.AST | None]:
    """SSE yield 的 f-string → (帧名, `json.dumps(<expr>)` 的 data 表达式)."""
    if not isinstance(value, ast.JoinedStr):
        return None, None
    prefix = ""
    data_expr: ast.AST | None = None
    for part in value.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            prefix += part.value
        elif isinstance(part, ast.FormattedValue):
            call = part.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "dumps"
                and call.args
            ):
                data_expr = call.args[0]
    m = re.search(r"event:\s*([A-Za-z_][\w.]*)", prefix)
    return (m.group(1) if m else None), data_expr


def _sse_for_bound_shape(
    func: ast.AST, name: str, task_keys: set[str] | None, queue_keys: set[str] | None
) -> set[str] | None:
    """`for <name> in <iter>` 的绑定形状: 任务清单 → 任务形状; 事件队列 → 队列形状."""
    for node in ast.walk(func):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            it = node.iter
            if (
                isinstance(it, ast.Call)
                and isinstance(it.func, ast.Attribute)
                and it.func.attr in {"list_all", "list_active"}
            ):
                return task_keys
            if isinstance(it, ast.Name | ast.Subscript):
                return queue_keys
            return None
    return None


def _sse_progress_shapes(root: Path) -> dict[str, dict]:
    """progress 通道帧名 → payload 形状 {keys, closed} (解析 `_SSE_FRAME_MODULE`).

    `snapshot` 取 `list_all()` → `to_dict()` 字面键; `update`/`campaign` 取
    `_events` 队列的混合形状 (`to_dict()` ∪ campaign `evt`, 由 `_kind` 分支择一,
    静态不细分故取并集 —— 安全下界); `heartbeat` 取自身字面 dict.
    """
    tree = _parse(root / _SSE_FRAME_MODULE)
    if tree is None:
        return {}
    task_keys = _func_return_dict_keys(tree, "to_dict")
    campaign_keys = _func_local_dict_keys(tree, "emit_campaign_event", "evt")
    # 队列写入形状: `self._events.append(<expr>)` —— to_dict() 或 campaign evt.
    queue_parts: list[set[str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_events"
            and node.args
        ):
            a = node.args[0]
            if (
                isinstance(a, ast.Call)
                and isinstance(a.func, ast.Attribute)
                and a.func.attr == "to_dict"
                and task_keys
            ):
                queue_parts.append(task_keys)
            elif isinstance(a, ast.Name) and a.id == "evt" and campaign_keys:
                queue_parts.append(campaign_keys)
    queue_keys = set().union(*queue_parts) if queue_parts else None

    shapes: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == "to_event_stream"
        ):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Yield):
                continue
            frame, expr = _sse_yield_frame_expr(sub.value)
            if frame is None:
                continue
            ks = _dict_literal_keys(expr)
            if ks is not None:
                shapes[frame] = {"keys": ks, "closed": True}
                continue
            keys = (
                _sse_for_bound_shape(node, expr.id, task_keys, queue_keys)
                if isinstance(expr, ast.Name)
                else None
            )
            if keys is not None:
                shapes[frame] = {"keys": keys, "closed": True}
            else:
                shapes[frame] = {"keys": set(), "closed": False}
    return shapes


def _sse_event_bus_shape(root: Path) -> set[str] | None:
    """event_bus 通道 payload 信封 = `AgentEvent.to_sse()` 的 `payload = {…}` 字面键."""
    tree = _parse(root / _SSE_PAYLOAD_EVENT_BUS_MODULE)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == _SSE_PAYLOAD_TO_SSE
        ):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Assign)
                    and isinstance(sub.targets[0], ast.Name)
                    and sub.targets[0].id == "payload"
                ):
                    return _dict_literal_keys(sub.value)
    return None


def _fe_ts_files(
    frontend: Path, skip_suffixes: tuple[str, ...] = ()
) -> Iterator[tuple[Path, str, str]]:
    """遍历前端 `.ts`/`.tsx` → `(path, rel, text)`; `rel` 以 `skip_suffixes` 结尾则跳过.

    统一前端扫描取材规则 (后缀过滤 / `rel` 归属 / 读失败容错), 供 SSE 事件负载面与
    WS 事件负载面共用.
    """
    if not frontend.is_dir():
        return
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        rel = _display(frontend, p)
        if skip_suffixes and rel.endswith(skip_suffixes):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        yield p, rel, text


def _fe_field_read_rows(
    *,
    rel: str,
    channel: str | None,
    frames: list[str],
    body: str,
    base_line: int,
    readable: frozenset[str],
    skip_fields: frozenset[str] = frozenset(),
) -> list[dict]:
    """`body` 内 `readable.<字段>` 顶层读取 → 行记录 (字段外层 × 帧内层, 保序).

    `base_line` 为 `body` 首行行号 (1 基), 行内偏移经 `body.count("\\n")` 折算.
    SSE 事件负载面与 WS 事件负载面共用.
    """
    rows: list[dict] = []
    for fm in _FE_DOT_FIELD_READ.finditer(body):
        if fm.group(1) not in readable:
            continue
        field = fm.group(2)
        if field in skip_fields:
            continue
        ln_no = base_line + body.count("\n", 0, fm.start())
        for fr in frames:
            rows.append(
                {
                    "rel": rel,
                    "line": ln_no,
                    "channel": channel,
                    "frame": fr,
                    "field": field,
                }
            )
    return rows


def _sse_payload_reads(frontend: Path) -> list[dict]:
    """扫前端 SSE 处理函数里的 `t.<字段>` 顶层读取 (t = `JSON.parse(e.data)` 结果)."""
    reads: list[dict] = []
    chan_keys = sorted(_SSE_CHANNELS, key=len, reverse=True)
    for _p, rel, text in _fe_ts_files(frontend):
        lines = text.splitlines()

        assigns: list[tuple[int, str, str | None]] = []
        for i, ln in enumerate(lines):
            m = _FE_ES_ASSIGN.search(ln)
            if m:
                arg = m.group(2)
                ch = next((_SSE_CHANNELS[k] for k in chan_keys if k in arg), None)
                assigns.append((i, m.group(1), ch))

        # 闭包按默认参数绑定当轮局部 (B023: 循环内定义且引用循环变量).
        def chan_of(
            var: str, idx: int, _assigns: list[tuple[int, str, str | None]] = assigns
        ) -> str | None:
            found = None
            for ai, av, ac in _assigns:
                if av == var and ai <= idx:
                    found = ac
            return found

        def emit(
            body: str,
            base_line: int,
            frames: list[str],
            ch: str | None,
            _rel: str = rel,
        ) -> None:
            pm = _FE_JSON_PARSE.search(body)
            if pm is None:
                return
            reads.extend(
                _fe_field_read_rows(
                    rel=_rel,
                    channel=ch,
                    frames=frames,
                    body=body,
                    base_line=base_line,
                    readable=frozenset({pm.group(1)}),
                )
            )

        # 具名处理函数: addEventListener(<帧名>, <handler>) → handler → 帧/通道
        reg: dict[str, list[tuple[str, str | None]]] = {}
        for i, ln in enumerate(lines):
            m = _FE_LISTEN.search(ln)
            if m:
                reg.setdefault(m.group(3), []).append((m.group(2), chan_of(m.group(1), i)))
                continue
            m2 = _FE_LISTEN_IDENT.search(ln)
            if m2:
                var, handler = m2.group(1), m2.group(3)
                # `[ "a.b", "c.d" ].forEach((ev) => es.addEventListener(ev, h))`:
                # 帧名在上一行的数组字面量里, 通道取 forEach 那一行的事件源.
                for j in range(i - 1, max(-1, i - 6), -1):
                    am = _FE_ARRAY.match(lines[j])
                    if am:
                        for s in _FE_STR.findall(am.group(1)):
                            reg.setdefault(handler, []).append((s, chan_of(var, i)))
                        break

        for m in _FE_HANDLER.finditer(text):
            handler = m.group(1)
            if handler not in reg:
                continue
            brace = m.end() - 1
            end = _ws_ts_brace_end(text, brace)
            if text[end - 1] != "}":
                continue
            frames: list[str] = []
            for fr, _c in reg[handler]:
                if fr not in frames:
                    frames.append(fr)
            emit(
                text[brace + 1 : end - 1],
                text.count("\n", 0, brace + 1) + 1,
                frames,
                reg[handler][0][1],
            )

        # 内联处理函数: es.addEventListener("帧名", (e: MessageEvent) => { … })
        for m in _FE_LISTEN_INLINE_HANDLER.finditer(text):
            var, frame = m.group(1), m.group(2)
            brace = m.end() - 1
            end = _ws_ts_brace_end(text, brace)
            if text[end - 1] != "}":
                continue
            emit(
                text[brace + 1 : end - 1],
                text.count("\n", 0, brace + 1) + 1,
                [frame],
                chan_of(var, text.count("\n", 0, m.start())),
            )
    return reads


def build_sse_payload_contract(
    root: Path | None = None, frontend: Path | None = None
) -> dict:
    """SSE 事件负载面: 后端帧 payload 顶层键 ↔ 前端 `JSON.parse(e.data)` 顶层读取.

    硬方向: 前端在该帧处理函数里读 `t.<字段>` 而后端该帧 payload 从不发此顶层字段
    (恒 undefined). 反向 (后端发前端没读) 不是违例, 只列候选. 只核**顶层**键;
    `t.data.<字段>` 的嵌套子形状由发布点决定, 静态不可穷尽, 跳过.
    """
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    emitters = _sse_frame_emitters(root)
    channels: dict[str, dict[str, dict]] = {}
    progress = _sse_progress_shapes(root)
    if progress:
        channels["progress"] = progress
    eb_keys = _sse_event_bus_shape(root)
    channels["event_bus"] = {
        f: {"keys": eb_keys if eb_keys is not None else set(), "closed": eb_keys is not None}
        for f in emitters.get("event_bus", [])
    }

    reads = _sse_payload_reads(frontend)
    violations: list[dict] = []
    checked = skip_frame = skip_shape = 0
    read_fields: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in reads:
        ch, frame, field = r["channel"], r["frame"], r["field"]
        frames = channels.get(ch, {})
        if frame not in frames:
            # 帧不属该通道 (该帧名本身已由 SSE 消费面报通道不匹配) → payload 面无权威.
            skip_frame += 1
            continue
        info = frames[frame]
        read_fields[(ch, frame)].add(field)
        if not info["closed"]:
            skip_shape += 1
            continue
        checked += 1
        if field in info["keys"]:
            continue
        violations.append(
            {
                "kind": "read-undeclared",
                "channel": ch,
                "frame": frame,
                "field": field,
                "detail": f"前端在 `{frame}` 处理函数里读 `t.{field}`",
                "rel": r["rel"],
                "line": r["line"],
            }
        )

    for v in violations:
        tri = _sse_payload_triage(v["channel"], v["frame"], v["field"])
        v["triage"] = tri[0] if tri else "untriaged"
        v["triage_reason"] = tri[1] if tri else ""

    zero_read: list[dict] = []
    for ch, frames in channels.items():
        closed_keys: set[str] = set()
        read_keys: set[str] = set()
        for frame, info in frames.items():
            if info["closed"]:
                closed_keys |= info["keys"]
            read_keys |= read_fields.get((ch, frame), set())
        for k in sorted(closed_keys - read_keys):
            zero_read.append({"channel": ch, "field": k})

    kinds = Counter(v["kind"] for v in violations)
    return {
        "frontend": str(frontend),
        "channels": {
            ch: {
                f: {"keys": sorted(i["keys"]), "closed": i["closed"]}
                for f, i in sorted(frames.items())
            }
            for ch, frames in channels.items()
        },
        "frame_reads": {
            ch: {frame: sorted(read_fields.get((ch, frame), set())) for frame in frames}
            for ch, frames in channels.items()
        },
        "read_count": len(reads),
        "coverage": {"checked": checked, "skip_frame": skip_frame, "skip_shape": skip_shape},
        "violations": violations,
        "untriaged": [v for v in violations if v["triage"] == "untriaged"],
        "kind_counts": dict(sorted(kinds.items())),
        "zero_read": zero_read,
    }


def render_sse_payload_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append(
        "## SSE 事件负载面: 后端帧 payload 顶层键 vs 前端 JSON.parse(e.data) 顶层读取"
    )
    lines.append("")
    lines.append(
        "SSE 消费面只核「帧名认不认」(监听挂没挂对 EventSource); 本面再往里一层, 核**帧 "
        "payload 的顶层键**。权威面是后端发帧处 `json.dumps(<expr>)` 的 `<expr>` 形状 —— "
        "`progress` 通道取 `interaction/progress.py` 的 `to_dict()` / campaign `evt` 字面, "
        "`event_bus` 通道取 `AgentEvent.to_sse()` 的固定信封。硬方向: **前端在该帧处理函数里读 "
        "`t.<字段>` 而后端该帧 payload 从不发此顶层字段** ⇒ 恒 `undefined` (静默坏). 反向"
        "(后端发了前端没读) 不是违例, 只列候选。只核**顶层**键: `t.data.<字段>` 的嵌套子形状由"
        "各事件发布点决定, 静态不可穷尽, 跳过."
    )
    lines.append("")
    lines.append(
        "违例类型: " + "; ".join(f"`{k}`={v}" for k, v in _SSE_PAYLOAD_KIND_DOC.items())
    )
    lines.append("")
    counts = (
        "  " + ", ".join(f"`{k}`×{n}" for k, n in contract["kind_counts"].items())
        if contract["kind_counts"]
        else ""
    )
    lines.append(
        f"前端 SSE payload 顶层读取点: **{contract['read_count']}** 处; 违例: "
        f"**{len(contract['violations'])}** 条.{counts}"
    )
    lines.append("")

    lines.append("### 各通道帧 payload 顶层键 (权威面)")
    lines.append("")
    lines.append("| 通道 | 帧名 | payload 顶层键 | 形状 | 前端读取字段 |")
    lines.append("|---|---|---|---|---|")
    for ch, frames in contract["channels"].items():
        fr = contract["frame_reads"].get(ch, {})
        for frame in sorted(frames):
            info = frames[frame]
            keys = ", ".join(f"`{k}`" for k in info["keys"]) or "—"
            shape = "封闭" if info["closed"] else "开放(读不出)"
            rd = ", ".join(f"`{f}`" for f in fr.get(frame, [])) or "—"
            lines.append(f"| `{ch}` | `{frame}` | {keys} | {shape} | {rd} |")
    lines.append("")

    lines.append("### 违例 (硬: 前端读的顶层字段后端从不发)")
    lines.append("")
    if contract["violations"]:
        lines.append(
            "硬违例 —— 前端读 `t.<字段>` 而后端该帧 payload 无此顶层键 (恒 undefined). "
            "逐条分诊, 未登记的落「待分诊」(回归测试会失败, 逼人工判定):"
        )
        lines.append("")
        for v in sorted(
            contract["violations"],
            key=lambda x: (x["channel"], x["frame"], x["field"], x["rel"], x["line"]),
        ):
            lines.append(
                f"- `[{v['kind']}]` `{v['channel']}/{v['frame']}.{v['field']}` "
                f"@ `{v['rel']}:{v['line']}` — {v['detail']}"
                + _sse_payload_violation_mark(v["channel"], v["frame"], v["field"])
            )
    else:
        lines.append("- 无 —— 前端读的每个顶层字段, 后端该帧都发.")
    lines.append("")

    lines.append("### payload 顶层键零前端读取 (候选, 反向不判违例)")
    lines.append("")
    if contract["zero_read"]:
        for z in contract["zero_read"]:
            lines.append(f"- `{z['channel']}` / `{z['field']}`")
    else:
        lines.append("- 无.")
    lines.append("")

    lines.append("### 静态核对覆盖面 (读不出形状即跳过, 不猜)")
    lines.append("")
    lines.append("| 维度 | 已核对 | 跳过 (帧不属该通道) | 跳过 (形状开放) |")
    lines.append("|---|---|---|---|")
    cov = contract["coverage"]
    lines.append(
        f"| 前端顶层字段读取 | {cov['checked']} | {cov['skip_frame']} | {cov['skip_shape']} |"
    )
    lines.append("")
    lines.append(
        "诚实边界: 只读**字面量**形状 —— 后端 payload 经变量间接构造 (`**` 展开 / 动态拼键) 或"
        "前端 `t` 经中转/解构读取时该处记开放并跳过, 故违例是**下界** (可能漏报); 只核帧 payload "
        "的**顶层**键 (嵌套 `t.data.<字段>` 的子形状由发布点决定, 不核); `t` 须为 "
        "`JSON.parse(e.data)` 直接赋值才归因; 帧名不属该通道时跳过 (该帧名本身已由 SSE 消费面报"
        "通道不匹配); 反向 (后端发前端没读) 不是违例. 只覆盖 `progress` / `event_bus` 两条命名帧"
        "通道 (`pet` 通道全为无名帧)."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WS 消费面: 后端 WS 帧名 ↔ 前端 WebSocket 客户端判别 (按端点通道)
# ---------------------------------------------------------------------------

# 已知 WS 通道 (URL 片段 → 通道名). 每个端点自带一套帧名词表, 帧名只在该
# 通道上才有意义.
_WS_CHANNELS: dict[str, str] = {
    "/ws/agent": "agent",
    "/ws/terminal": "terminal",
    "/ws/viewer3d": "viewer3d",
    "/ws/hpc/jobs": "hpc",
}
# 各通道后端生产模块 (server→client 帧的 `"type": "…"` 字面量来源).
_WS_BACKEND_MODULES: dict[str, tuple[str, ...]] = {
    "agent": ("huginn/routes/ws.py", "huginn/routes/ws_helpers.py"),
    "terminal": ("huginn/routes/terminal.py",),
    "viewer3d": ("huginn/routes/viewer3d.py",),
    "hpc": ("huginn/routes/hpc.py",),
}
# 前端只对 agent 通道按 `type` 显式判别 (`switch (data.type)` / `.type === …`);
# terminal/hpc 前端只按字段 (data/output) 取值, 不看 type —— 记作 field-probing,
# 不逐帧判"零判别" (诚实边界).
_WS_TYPED_CHANNELS = frozenset({"agent"})
# 前端文件 → 通道线索 (URL 片段或该通道专属 ws ref). URL 片段优先, ref 兜底
# (跨文件传 ref, 如 ChatPanel 用父组件传入的 wsClientRef).
_WS_FE_REF_HINTS: dict[str, tuple[str, ...]] = {
    "agent": ("wsClientRef", "petWsRef"),
}
# 共享传输层 (ReconnectingWebSocket) 的心跳 ping 打到 agent 端点.
_WS_TRANSPORT_REL = "lib/ws-client.ts"
# 声明面: 前端 `WSMessage` 判别联合 (server→client 帧的 TS 契约).
_WS_DECL_SUFFIX = "types/ws.ts"
# agent 通道 client→server 权威分发表 (handler registry).
_WS_REGISTRY_REL = "huginn/routes/ws.py"
_WS_REGISTRY_VAR = "_MESSAGE_HANDLERS"
# 本地发送包装函数 (名字节点形式的 send sink), 与 send_json/send_text 同列.
_WS_SEND_WRAPPERS = frozenset({"_ws_send"})

_WS_TERNARY_RE = re.compile(
    r'=\s*["\']([a-z][a-z0-9_]*)["\']\s+if\s+[^\n]+?\s+else\s+["\']([a-z][a-z0-9_]*)["\']'
)
# 前端 server→client 判别: `case "lit":` (须在 `switch (<wsVar>.type)` 块内) 与
# `<wsVar>.type === "lit"`. `wsVar` 限定为**标注了 WSMessage 的变量** —— 否则
# `switch (mood)` / `result.type === "thread"` / SSE 的 `data.type === "heartbeat"`
# 都会被误当 WS 帧名. 点号型 (embedding.download.* / team.*) 是 SSE payload, 不进.
_FE_WS_CASE_RE = re.compile(r'case\s+["\']([a-z][a-z0-9_]*)["\']\s*:')
_FE_WS_EQ_RE = re.compile(
    r'\b(\w+)\.type\s*!?={2,3}\s*["\']([a-z][a-z0-9_]*)["\']'
)
# `switch (<expr>) {` 头; 判别式须恰为 `<wsVar>.type`.
_FE_SWITCH_TYPE_HEAD_RE = re.compile(r'switch\s*\(([^)]*)\)\s*\{')
# WS 帧变量标注: `(data: WSMessage)`.
_FE_WS_MSG_VAR_RE = re.compile(r'\b(\w+)\s*:\s*WSMessage\b')
# 前端 client→server 生产: payload 式 JSON.stringify 与内联 `.send({…})`.
_FE_WS_SEND_STR_RE = re.compile(
    r'JSON\.stringify\(\s*\{\s*["\']?type["\']?\s*:\s*["\']([a-z][a-z0-9_]*)["\']', re.S
)
_FE_WS_SEND_INLINE_RE = re.compile(
    r'\.send\(\s*\{\s*["\']?type["\']?\s*:\s*["\']([a-z][a-z0-9_]*)["\']'
)
# 后端 client→server 消费: 分发比较 (`mtype == "…"` / `…get("type") == "…"`).
_WS_INBOUND_CMP_RE = re.compile(
    r'(?:\bmtype\b|\bmsg_type\b|get\(\s*["\']type["\']\s*\)'
    r'|\[\s*["\']type["\']\s*\])\s*==\s*["\']([a-z][a-z0-9_]*)["\']'
)
# 声明面: TS 判别联合的 `type: "lit"`.
_WS_DECL_TYPE_RE = re.compile(r'type\s*:\s*["\']([a-z][a-z0-9_]*)["\']')

_WS_STATUS_DOC = {
    "wired": "后端该通道确实发此帧名, 前端有 type 判别",
    "dynamic": "已声明帧, 后端经变量透传转发 (静态生产面不可穷尽, 只提示)",
    "no-source": "后端该通道不发此帧名, 且未声明 (前端 case 永不命中)",
    "handled": "后端该通道分发表认此入站类型",
    "unhandled": "后端该通道分发面无此入站类型 (回 error 帧)",
    "field-probing": "前端只按字段取值, 不按 type 判别 (不计入)",
}

# 已确认**有意**的 WS 候选 (非缺陷). 键 (kind, channel, frame) → 确认理由.
# kind: `zero-consumer` 生产帧零前端判别 / `undeclared` 生产帧未进 WSMessage 联合 /
# `declared-only` 已声明但非生产非判别 / `phantom-inbound` 入站类型无前端发送者.
# 审计仍**列出**候选, 但标注"已确认有意"; 未登记的候选落在"待确认" —— 新出现的
# 候选会失败回归测试 (逼人工分诊).
_WS_CONFIRMED_INTENTIONAL: dict[tuple[str, str, str], str] = {
    ("zero-consumer", "agent", "decision_resolved"): (
        "决策点裁决的 ack 帧; 前端乐观清空 pendingDecisionPoint, 不消费它"
    ),
    ("undeclared", "agent", "decision_resolved"): (
        "同一 ack 帧, 前端不消费故未进 WSMessage 联合 (与 zero-consumer 同源)"
    ),
    ("declared-only", "agent", "plan_confirm"): (
        "WSMessage 联合把双向帧并在一起; 它是 client→server 请求帧"
    ),
    ("declared-only", "agent", "clarification_response"): (
        "WSMessage 联合把双向帧并在一起; 它是 client→server 请求帧"
    ),
    ("phantom-inbound", "agent", "explore_start"): (
        "探索编排入口; 面向非桌面客户端 (仓库内无发送者), 保留为公开 WS API"
    ),
    ("phantom-inbound", "terminal", "resize"): (
        "终端尺寸同步; 桌面用普通输入框 (无 xterm fit), 面向外部客户端"
    ),
    ("phantom-inbound", "terminal", "signal"): (
        "终端信号 (Ctrl-C 等); 桌面未启用, 面向外部客户端"
    ),
}


def _ws_triage(kind: str, channel: str, frame: str) -> str | None:
    """候选分诊: 返回确认理由, 未登记则 None (待人工确认)."""
    return _WS_CONFIRMED_INTENTIONAL.get((kind, channel, frame))


def _ws_candidate_line(label: str, kind: str, channel: str, frame: str) -> str:
    reason = _ws_triage(kind, channel, frame)
    mark = f" — ✅ 已确认有意: {reason}" if reason else " — ⚠ 待确认"
    return f"- {label}{mark}"


def _ws_line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _ws_dict_type_literal(node: ast.Dict) -> str | None:
    """取字典字面量里 `"type": "…"` 的字符串常量值."""
    for k, v in zip(node.keys, node.values):
        if (
            isinstance(k, ast.Constant)
            and k.value == "type"
            and isinstance(v, ast.Constant)
            and isinstance(v.value, str)
        ):
            return v.value
    return None


def _ws_arg_type_literal(arg: ast.AST) -> str | None:
    """从 send 实参里取 `type` 字面量; 支持 `{…}` / `json.dumps({…})` / `dict({…})`.

    实参是变量 (`_ws_send(dict(state))` / `_ws_send(msg)`) 时返回 None —— 那类
    透传不静态可辨, 由调用方计为 dynamic.
    """
    node = arg
    if isinstance(node, ast.Call):
        f = node.func
        unwrap = (isinstance(f, ast.Attribute) and f.attr == "dumps") or (
            isinstance(f, ast.Name) and f.id == "dict"
        )
        if unwrap and node.args:
            node = node.args[0]
    if not isinstance(node, ast.Dict):
        return None
    return _ws_dict_type_literal(node)


def _ws_server_frames(root: Path) -> dict[str, dict]:
    """各 WS 通道的 server→client 帧名生产面 (AST 扫 send 实参 + 三元字面量).

    send sink: `websocket.send_json` / `send_text` 与本地包装 `_ws_send`. 参数是
    字典字面量 (或 `json.dumps({…})`) 才静态可辨; 变量透传 (如 `_ws_send(dict(
    state))` 把 agent 循环 yield 的类型化事件转发) 计为动态, 不在此穷尽.
    """
    out: dict[str, dict] = {}
    for ch, mods in _WS_BACKEND_MODULES.items():
        names: set[str] = set()
        dynamic = 0
        for rel in mods:
            tree = _parse(root / rel)
            if tree is None:
                continue
            for node in ast.walk(tree):
                # 帧名生产: 这些 WS 路由模块里的 `{"type": "…"}` 字典字面量都是
                # 该通道的 server→client 帧 —— 无论直接作 send 实参, 还是先赋给
                # 变量 (`tool_result_msg = {…}`) 再 `_ws_send(tool_result_msg)`.
                if isinstance(node, ast.Dict):
                    lit = _ws_dict_type_literal(node)
                    if lit is not None:
                        names.add(lit)
                    continue
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                is_send = (
                    isinstance(f, ast.Attribute) and f.attr in ("send_json", "send_text")
                ) or (isinstance(f, ast.Name) and f.id in _WS_SEND_WRAPPERS)
                if not is_send:
                    continue
                if not node.args or _ws_arg_type_literal(node.args[0]) is None:
                    dynamic += 1
            # `msg_type = "a" if … else "b"` 也静态可辨.
            for m in _WS_TERNARY_RE.finditer(_read(root, rel)):
                names.add(m.group(1))
                names.add(m.group(2))
        out[ch] = {"frames": sorted(names), "dynamic": dynamic}
    return out


def _ws_registry_keys(root: Path) -> list[str]:
    """`_MESSAGE_HANDLERS` 的键 = agent 通道 client→server 权威分发面.

    注册表是**带注解**的赋值 (`_MESSAGE_HANDLERS: dict[str, Any] = {…}`), 须按
    `ast.AnnAssign` 解析; 只查 `ast.Assign` 会读空并误报全部入站类型 unhandled.
    """
    tree = _parse(root / _WS_REGISTRY_REL)
    if tree is None:
        return []
    for node in ast.walk(tree):
        targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == _WS_REGISTRY_VAR for t in targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            return [
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
    return []


def _ws_inbound_types(root: Path) -> dict[str, list[str]]:
    """各通道后端消费的 client→server 类型 (registry + 分发比较)."""
    out: dict[str, list[str]] = {ch: [] for ch in _WS_CHANNELS.values()}
    out["agent"] = sorted(set(_ws_registry_keys(root)))
    for ch, mods in _WS_BACKEND_MODULES.items():
        found = set(out[ch])
        for rel in mods:
            found.update(_WS_INBOUND_CMP_RE.findall(_read(root, rel)))
        out[ch] = sorted(found)
    return out


def _ws_file_channel(rel: str, text: str) -> str | None:
    """前端文件归属通道: URL 片段优先, 专属 ref 兜底, 传输层归 agent."""
    hits = [ch for frag, ch in _WS_CHANNELS.items() if frag in text]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        ref = [ch for ch, hints in _WS_FE_REF_HINTS.items() if any(h in text for h in hints)]
        if len(ref) == 1:
            return ref[0]
        if rel.endswith(_WS_TRANSPORT_REL):
            return "agent"
    return None


def _skip_ts_string(text: str, i: int) -> int:
    """跳过从 `i` 起的 TS 字符串字面量, 返回结束后的下标 (含未闭合兜底)."""
    quote = text[i]
    i += 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        if c == "\n" and quote != "`":
            return i
        i += 1
    return i


def _fe_switch_type_cases(text: str, ws_vars: set[str]) -> list[tuple[str, int]]:
    """仅取 `switch (<wsVar>.type) { … }` 块内的 `case "lit":`.

    `case "lit":` 单独看是歧义的 —— Pet.tsx 的 `switch (mood)` 也用同样的
    标签 (`case "thinking":`), 但那是宠物心情而非 WS 帧名. 只有判别式恰为
    **标注了 WSMessage 的变量** 的 `.type` 才计入.
    """
    out: list[tuple[str, int]] = []
    for m in _FE_SWITCH_TYPE_HEAD_RE.finditer(text):
        var = re.fullmatch(r'\s*(\w+)\.type\s*', m.group(1))
        if var is None or var.group(1) not in ws_vars:
            continue
        start = m.end() - 1  # 指向 `{`
        depth = 0
        i = start
        n = len(text)
        while i < n:
            c = text[i]
            if c in "\"'`":
                i = _skip_ts_string(text, i)
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[start : i + 1]
        for cm in _FE_WS_CASE_RE.finditer(body):
            out.append((cm.group(1), _ws_line_of(text, start + cm.start())))
    return out


def _ws_scan_frontend(frontend: Path) -> dict:
    """扫前端 `.ts`/`.tsx`: server→client 判别点与 client→server 发送点 (带通道归属)."""
    consumers: list[dict] = []
    sends: list[dict] = []
    if not frontend.is_dir():
        return {"consumers": consumers, "sends": sends}
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        rel = _display(frontend, p)
        if rel.endswith(".spec.ts") or rel.endswith(_WS_DECL_SUFFIX):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ch = _ws_file_channel(rel, text)
        ws_vars = set(_FE_WS_MSG_VAR_RE.findall(text))
        for frame, line in _fe_switch_type_cases(text, ws_vars):
            consumers.append({"rel": rel, "line": line, "channel": ch, "frame": frame})
        for i, ln in enumerate(text.splitlines()):
            for m in _FE_WS_EQ_RE.finditer(ln):
                if m.group(1) not in ws_vars:
                    continue
                consumers.append(
                    {"rel": rel, "line": i + 1, "channel": ch, "frame": m.group(2)}
                )
        seen: set[tuple[str, int]] = set()
        for rx in (_FE_WS_SEND_STR_RE, _FE_WS_SEND_INLINE_RE):
            for m in rx.finditer(text):
                line = _ws_line_of(text, m.start())
                key = (m.group(1), line)
                if key in seen:
                    continue
                seen.add(key)
                sends.append({"rel": rel, "line": line, "channel": ch, "frame": m.group(1)})
    return {"consumers": consumers, "sends": sends}


def build_ws_contract(root: Path | None = None, frontend: Path | None = None) -> dict:
    """WS 消费面: 后端 WS 帧名生产面 ↔ 前端 WebSocket 客户端判别面 (按端点通道)."""
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    produced = _ws_server_frames(root)
    inbound = _ws_inbound_types(root)
    decl_path = frontend / _WS_DECL_SUFFIX
    declared: set[str] = set()
    if decl_path.is_file():
        declared = set(_WS_DECL_TYPE_RE.findall(decl_path.read_text(encoding="utf-8", errors="replace")))
    scan = _ws_scan_frontend(frontend)

    channel_out: dict[str, dict] = {}
    for frag, ch in _WS_CHANNELS.items():
        channel_out[ch] = {
            "url": frag,
            "typed": ch in _WS_TYPED_CHANNELS,
            "frames": produced[ch]["frames"],
            "dynamic_sends": produced[ch]["dynamic"],
            "inbound": inbound[ch],
        }

    agent_frames = produced["agent"]["frames"]
    consumed = {c["frame"] for c in scan["consumers"] if c["channel"] == "agent"}
    server_frames = [
        {"frame": f, "declared": f in declared, "consumed": f in consumed}
        for f in agent_frames
    ]
    zero_consumer = [f for f in agent_frames if f not in consumed]
    undeclared = [f for f in agent_frames if f not in declared]
    declared_only = sorted(declared - set(agent_frames) - consumed)

    consumers: list[dict] = []
    for c in scan["consumers"]:
        ch = c["channel"]
        pool = set(produced.get(ch, {}).get("frames", [])) if ch else set()
        if ch is None:
            status, note = "external", "非 WS 通道判别, 不计入"
        elif not channel_out[ch]["typed"]:
            status, note = "field-probing", "该通道前端按字段取值, 不按 type 判别"
        elif c["frame"] in pool:
            status, note = "wired", ""
        elif ch in _WS_TYPED_CHANNELS and c["frame"] in declared:
            status, note = "dynamic", "已声明帧, 后端经变量透传转发 (静态生产面不可穷尽)"
        else:
            status, note = "no-source", "后端该通道不发此帧名"
        consumers.append({**c, "status": status, "note": note})

    sends: list[dict] = []
    for s in scan["sends"]:
        ch = s["channel"]
        if ch is None:
            status, note = "external", "非 WS 通道发送, 不计入"
        else:
            ok = set(inbound.get(ch, []))
            if ch in _WS_TYPED_CHANNELS:
                ok |= set(_ws_registry_keys(root))
            if s["frame"] in ok:
                status, note = "handled", ""
            else:
                status, note = "unhandled", "后端该通道分发面无此入站类型 (回 error 帧)"
        sends.append({**s, "status": status, "note": note})

    fe_sends: dict[str, set[str]] = defaultdict(set)
    for s in scan["sends"]:
        if s["channel"]:
            fe_sends[s["channel"]].add(s["frame"])
    has_fe = {c["channel"] for c in scan["consumers"]} | set(fe_sends)
    phantom: list[dict] = []
    for ch, keys in inbound.items():
        if ch not in has_fe:
            continue  # 无前端消费者 (如 viewer3d 由外部客户端驱动), 不判
        for k in keys:
            if k not in fe_sends.get(ch, set()):
                phantom.append({"channel": ch, "frame": k})

    return {
        "frontend": str(frontend),
        "channels": channel_out,
        "declared": sorted(declared),
        "server_frames": server_frames,
        "consumers": consumers,
        "sends": sends,
        "zero_consumer": zero_consumer,
        "undeclared": undeclared,
        "declared_only": declared_only,
        "phantom_inbound": phantom,
    }


def render_ws_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## WS 消费面: 后端 WS 帧名生产面 vs 前端 WebSocket 判别面")
    lines.append("")
    lines.append(
        "SSE 消费面只核单向 (后端发帧 → 前端 `addEventListener`), WebSocket 是**双向**"
        "的: 后端 `send_json({\"type\": …})` 发帧前端 `switch (data.type)` 收, 前端"
        "`send({type: …})` 发请求后端分发表收. 本面按**端点通道** `agent`(`/ws/agent`, "
        "唯一按 `type` 判别的通道) / `terminal` / `viewer3d` / `hpc` 分列: 帧名只在"
        "**发它的那条 WS 上**才可能命中, 故挂在别处 = 永不触发."
    )
    lines.append("")
    lines.append("状态: " + "; ".join(f"`{k}`={v}" for k, v in _WS_STATUS_DOC.items()))
    lines.append("")
    lines.append("| 通道 | URL 片段 | type 判别 | 生产帧名 | 入站类型 |")
    lines.append("|---|---|---|---|---|")
    for ch, v in contract["channels"].items():
        typed = "是" if v["typed"] else "否 (field-probing)"
        frames = ", ".join(f"`{f}`" for f in v["frames"]) or "—"
        inbound = ", ".join(f"`{f}`" for f in v["inbound"]) or "—"
        lines.append(f"| `{ch}` | `{v['url']}` | {typed} | {frames} | {inbound} |")
    lines.append("")

    lines.append("### 生产帧名 × 前端判别 (agent 通道)")
    lines.append("")
    lines.append("| 帧名 | 已声明 (WSMessage) | 前端有 type 判别 |")
    lines.append("|---|---|---|")
    for f in contract["server_frames"]:
        lines.append(
            f"| `{f['frame']}` | {'是' if f['declared'] else '否'} "
            f"| {'是' if f['consumed'] else '否'} |"
        )
    lines.append("")

    lines.append("### 前端 server→client 判别点")
    lines.append("")
    lines.append("| 帧名 | 通道 | 状态 | 位置 | 备注 |")
    lines.append("|---|---|---|---|---|")
    for s in sorted(contract["consumers"], key=lambda x: (x["status"], x["rel"], x["line"])):
        ch = f"`{s['channel']}`" if s["channel"] else "—"
        lines.append(
            f"| `{s['frame']}` | {ch} | `{s['status']}` | `{s['rel']}:{s['line']}` "
            f"| {s['note']} |"
        )
    lines.append("")

    lines.append("### 前端 client→server 发送点")
    lines.append("")
    lines.append("| 类型 | 通道 | 状态 | 位置 | 备注 |")
    lines.append("|---|---|---|---|---|")
    for s in sorted(contract["sends"], key=lambda x: (x["status"], x["rel"], x["line"])):
        ch = f"`{s['channel']}`" if s["channel"] else "—"
        lines.append(
            f"| `{s['frame']}` | {ch} | `{s['status']}` | `{s['rel']}:{s['line']}` "
            f"| {s['note']} |"
        )
    lines.append("")

    lines.append("### 生产帧名零前端判别 (候选)")
    lines.append("")
    zero = contract["zero_consumer"]
    if zero:
        for f in zero:
            lines.append(_ws_candidate_line(f"`agent` / `{f}`", "zero-consumer", "agent", f))
    else:
        lines.append("- 无 —— 每个生产帧名都有前端 type 判别.")
    lines.append("")

    lines.append("### 未进 WSMessage 判别联合的生产帧 (候选登记)")
    lines.append("")
    und = contract["undeclared"]
    if und:
        for f in und:
            lines.append(_ws_candidate_line(f"`agent` / `{f}`", "undeclared", "agent", f))
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 已声明但既非生产帧也非入站请求 (候选)")
    lines.append("")
    if contract["declared_only"]:
        for f in contract["declared_only"]:
            lines.append(_ws_candidate_line(f"`{f}`", "declared-only", "agent", f))
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 入站类型无前端发送者 (候选)")
    lines.append("")
    if contract["phantom_inbound"]:
        for z in contract["phantom_inbound"]:
            lines.append(
                _ws_candidate_line(
                    f"`{z['channel']}` / `{z['frame']}`",
                    "phantom-inbound",
                    z["channel"],
                    z["frame"],
                )
            )
    else:
        lines.append("- 无")
    lines.append("")
    lines.append(
        "诚实边界: 前端 TS 与后端 `send_json(变量)` 都只做**静态**扫描 —— 经变量透传的"
        "入站类型 (如 `_ws_send(dict(state))` 转发的 agent 循环类型化事件 `mode_banner` / "
        "`trust_update` / `budget_update` 等) 生产面**不可穷尽**, 故只提示不判死; 前端"
        "terminal/hpc 按字段 (`data`/`output`) 取值而非按 `type` 判别, 记作 field-probing;"
        " viewer3d 无桌面前端 (由外部客户端驱动), 其入站不判 phantom."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP API 消费面: 后端路由注册面 ↔ 前端 api.* 调用面
# ---------------------------------------------------------------------------

# 后端路由定义目录 (每个模块一个 APIRouter) 与权威挂载点 (`ALL_ROUTERS`).
_HTTP_ROUTES_DIR = "huginn/routes"
_HTTP_REGISTRY_REL = "huginn/routes/__init__.py"
_HTTP_REGISTRY_VAR = "ALL_ROUTERS"
# 装饰器属性 → HTTP 方法 (`websocket` 端点归 WS 消费面, 不进本面).
_HTTP_METHOD_ATTRS: dict[str, str] = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "patch": "PATCH",
    "delete": "DELETE",
}
# 前端调用包装 (`lib/api.ts` 的 `api.*`).
_HTTP_CALL_RE = re.compile(
    r"\bapi\s*\.\s*(get|post|put|patch|del|getBlob|upload|uploadWithProgress"
    r"|uploadStream|search)\b"
)
# 包装动词 → HTTP 方法 (`upload*` 走 multipart POST, `getBlob` 走 GET).
_HTTP_VERB_METHOD: dict[str, str] = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "patch": "PATCH",
    "del": "DELETE",
    "getBlob": "GET",
    "upload": "POST",
    "uploadWithProgress": "POST",
    "uploadStream": "POST",
    "search": "GET",
}
# `getBlob(path, { method: "POST" })` 会在 options 里覆盖方法.
_HTTP_METHOD_OVERRIDE_RE = re.compile(r"method\s*:\s*[\"'](\w+)[\"']")
# `api.search(query, …)` 是固定端点的语法糖 (见 lib/api.ts).
_HTTP_SEARCH_PATH = "/search/global"

_HTTP_STATUS_DOC = {
    "wired": "前端调用的方法与路径后端已注册",
    "method-mismatch": "路径已注册但无此方法 (405, 调用必失败)",
    "no-source": "后端无此路径 (404 死链)",
    "external": "绝对 URL / 非后端路径, 不计入",
}

# 已确认**硬违例** (前端调用挂不上后端注册面) 的分诊. 键 (方法, 去 query 路径) →
# (标签, 理由); 标签 `defect` 已确认缺陷待修 / `intentional` 已确认有意.
# 硬违例不是候选 —— 未登记即"待分诊", 回归测试会失败 (逼逐条人工判定).
# 空表 = 当前每个前端调用都命中后端注册面; 新增硬违例必须先分诊再登记.
_HTTP_CONFIRMED_VIOLATIONS: dict[tuple[str, str], tuple[str, str]] = {}

_HTTP_TRIAGE_DOC = {
    "defect": "已确认缺陷 (待修)",
    "intentional": "已确认有意",
}


def _http_triage(method: str, path: str) -> tuple[str, str] | None:
    """硬违例分诊: 返回 (标签, 理由); 未登记则 None (待人工确认)."""
    return _HTTP_CONFIRMED_VIOLATIONS.get((method.upper(), path.split("?")[0]))


def _http_violation_mark(method: str, path: str) -> str:
    tri = _http_triage(method, path)
    if tri is None:
        return " — ⚠ 待分诊"
    return f" — {'⛔' if tri[0] == 'defect' else '✅'} {_HTTP_TRIAGE_DOC[tri[0]]}: {tri[1]}"


def _http_skip_generics(text: str, i: int) -> int:
    """跳过 TS 泛型实参 `<…>` (含箭头返回类型里的 `=>`), 返回 `>` 之后的下标."""
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _http_call_args_end(text: str, open_idx: int) -> int:
    """从 `(` 起找配对的 `)` (跳过字符串字面量); 未闭合则返回文末."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            i = _skip_ts_string(text, i)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n


def _http_router_prefixes(tree: ast.Module) -> dict[str, str]:
    """模块级 `<name> = APIRouter(prefix=…, …)` → {变量名: 路径前缀}."""
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        f = node.value.func
        if not (isinstance(f, ast.Name) and f.id == "APIRouter"):
            continue
        prefix = ""
        for kw in node.value.keywords:
            if (
                kw.arg == "prefix"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                prefix = kw.value.value
        for t in node.targets:
            if isinstance(t, ast.Name):
                out[t.id] = prefix
    return out


def _http_route_decorator(
    dec: ast.expr, prefixes: dict[str, str]
) -> tuple[list[str], str, str] | None:
    """路由装饰器 → (方法列表, 完整路径, router 变量名); 非路由装饰器返回 None."""
    if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
        return None
    f = dec.func
    if not isinstance(f.value, ast.Name) or f.value.id not in prefixes:
        return None
    if not dec.args:
        return None
    arg0 = dec.args[0]
    if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)):
        return None
    path = prefixes[f.value.id] + arg0.value
    if f.attr == "api_route":
        methods = [
            e.value
            for kw in dec.keywords
            if kw.arg == "methods" and isinstance(kw.value, ast.List | ast.Tuple)
            for e in kw.value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    elif f.attr == "websocket":
        methods = ["WEBSOCKET"]
    else:
        only = _HTTP_METHOD_ATTRS.get(f.attr)
        if only is None:
            return None
        methods = [only]
    return sorted({m.upper() for m in methods}), path, f.value.id


def _http_backend_routes(root: Path) -> list[dict]:
    """扫 `huginn/routes/*.py` 的路由装饰器 → 后端端点生产面.

    `@router.<method>("<path>")` 与 `@<var>.api_route("<path>", methods=[…])` 都
    登记; `@<var>.websocket(…)` 归 WS 消费面, 仍返回 (带 `WEBSOCKET` 方法) 以便
    核它是否被 `ALL_ROUTERS` 挂上. 前缀取自模块级 `APIRouter(prefix=…)`.
    """
    routes: list[dict] = []
    rdir = root / _HTTP_ROUTES_DIR
    if not rdir.is_dir():
        return routes
    for py in sorted(rdir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        tree = _parse(py)
        if tree is None:
            continue
        prefixes = _http_router_prefixes(tree)
        if not prefixes:
            continue
        rel = py.relative_to(root).as_posix()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for dec in node.decorator_list:
                parsed = _http_route_decorator(dec, prefixes)
                if parsed is None:
                    continue
                methods, path, var = parsed
                routes.append(
                    {
                        "methods": methods,
                        "path": path,
                        "rel": rel,
                        "handler": node.name,
                        "router": var,
                    }
                )
    return routes


def _http_registry(root: Path) -> dict:
    """`routes/__init__.py` 的挂载面: import 别名 → (模块, 变量名) ↔ `ALL_ROUTERS`."""
    tree = _parse(root / _HTTP_REGISTRY_REL)
    if tree is None:
        return {"mounted": [], "mounted_pairs": [], "dangling": []}
    imports: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        mod = getattr(node, "module", None)
        if isinstance(node, ast.ImportFrom) and mod and mod.startswith("huginn.routes."):
            stem = mod.rsplit(".", 1)[-1]
            for a in node.names:
                imports[a.asname or a.name] = (stem, a.name)
    aliases: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == _HTTP_REGISTRY_VAR for t in node.targets
        ):
            continue
        if isinstance(node.value, ast.List):
            aliases = [e.id for e in node.value.elts if isinstance(e, ast.Name)]
    pairs = [imports[a] for a in aliases if a in imports]
    return {
        "mounted": sorted(f"{m}.{n}" for m, n in pairs),
        "mounted_pairs": pairs,
        "dangling": sorted(a for a in aliases if a not in imports),
    }


def _http_scan_frontend(frontend: Path) -> list[dict]:
    """扫前端 `api.*` 调用点 → (动词, 方法, 路径字面量, 位置).

    只有 `api.get/post/…` 包装计入; 裸 `fetch(...)` 与 `EventSource` 不在本面.
    方法取包装默认值, 但调用实参里出现 `method: "…"` 时以它为准
    (`getBlob(path, { method: "POST" })` 的实际方法就是 POST).
    """
    calls: list[dict] = []
    if not frontend.is_dir():
        return calls
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        rel = _display(frontend, p)
        if rel.endswith(".spec.ts") or rel.endswith(".spec.tsx"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _HTTP_CALL_RE.finditer(text):
            verb = m.group(1)
            j = m.end()
            if j < len(text) and text[j] == "<":
                j = _http_skip_generics(text, j)
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if j >= len(text) or text[j] != "(":
                continue
            k = j + 1
            while k < len(text) and text[k] in " \t\n":
                k += 1
            if k >= len(text) or text[k] not in "\"'`":
                continue
            quote = text[k]
            end = _skip_ts_string(text, k)
            closed = end > k + 1 and text[end - 1] == quote
            raw = text[k + 1 : end - 1 if closed else end]
            args = text[j : _http_call_args_end(text, j) + 1]
            override = _HTTP_METHOD_OVERRIDE_RE.search(args)
            method = (
                override.group(1).upper()
                if override
                else _HTTP_VERB_METHOD.get(verb, verb.upper())
            )
            calls.append(
                {
                    "rel": rel,
                    "line": text.count("\n", 0, m.start()) + 1,
                    "verb": verb,
                    "method": method,
                    "path": _HTTP_SEARCH_PATH if verb == "search" else raw,
                }
            )
    return calls


def _http_fe_segments(path: str) -> tuple[tuple[str, str], ...] | None:
    """前端路径 → 段序列 (`p`=动态段, `s`=静态段); 非后端路径返回 None.

    query string 截掉; `/v1` 版本前缀剥离 (后端在 `/v1` 与根路径双挂载, 前端两者
    都用); 整段 `${x}` → 动态段; `events${qs}` 这类"静态前缀 + 动态后缀"取静态
    前缀 (后缀是 query 拼接, 不是路径段).
    """
    p = path.split("?")[0]
    if p.startswith("/v1/"):
        p = p[3:]
    elif p == "/v1":
        p = "/"
    if not p.startswith("/"):
        return None
    out: list[tuple[str, str]] = []
    for seg in p.split("/"):
        if not seg:
            continue
        k = seg.find("${")
        if k == 0 or seg.startswith(":"):
            out.append(("p", ""))
        elif k > 0:
            out.append(("s", seg[:k]))
        else:
            out.append(("s", seg))
    return tuple(out)


def _http_route_segments(path: str) -> tuple[tuple[str, str], ...]:
    """后端路径模板 → 段序列 (`{param}` → 动态段)."""
    return tuple(
        ("p", seg) if seg.startswith("{") else ("s", seg)
        for seg in path.split("?")[0].split("/")
        if seg
    )


def _http_path_match(
    fe: tuple[tuple[str, str], ...], be: tuple[tuple[str, str], ...]
) -> bool:
    """段数相同且逐段相容 (后端动态段或前端动态段都可通配)."""
    if len(fe) != len(be):
        return False
    for (fk, fv), (bk, bv) in zip(fe, be):
        if bk == "p" or fk == "p":
            continue
        if fv != bv:
            return False
    return True


def _http_pick_endpoint(
    fe: tuple[tuple[str, str], ...], method: str, live: list[dict]
) -> dict | None:
    """唯一最贴合的已挂载端点; 无法唯一判定则 None.

    动态段通配会让 `/personas/{name}` 同时命中 `/personas/templates`。按「强位置」
    (静态对静态 / 动态对动态) 计数取最高分; 出现并列即认定歧义, 返回 None (不猜,
    免把后端形状对错端点)。
    """
    best: dict | None = None
    best_score = -1
    tie = False
    for ep in live:
        if method not in ep["methods"] or not _http_path_match(fe, ep["segments"]):
            continue
        score = sum(1 for (fk, fv), (bk, _bv) in zip(fe, ep["segments"]) if fk == bk)
        if score > best_score:
            best, best_score, tie = ep, score, False
        elif score == best_score:
            tie = True
    return None if tie else best


def build_http_contract(root: Path | None = None, frontend: Path | None = None) -> dict:
    """HTTP API 消费面: 后端路由注册面 ↔ 前端 `api.*` 调用面.

    只把「前端 → 后端」方向当硬契约 (前端调了后端没注册的路径/方法 = 404/405);
    反向「后端注册但桌面零调用」结构性存在 (HTTP API 面向外部客户端/CLI/测试),
    只按模块聚合列出候选.
    """
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    all_routes = _http_backend_routes(root)
    http_routes = [r for r in all_routes if "WEBSOCKET" not in r["methods"]]
    registry = _http_registry(root)
    scan = _http_scan_frontend(frontend)

    mounted_pairs = set(registry["mounted_pairs"])
    endpoints: list[dict] = [
        {
            "methods": r["methods"],
            "path": r["path"],
            "rel": r["rel"],
            "handler": r["handler"],
            "router": r["router"],
            # 未挂进 `ALL_ROUTERS` 的模块, 其端点根本不在 app 上 —— 只有已挂载
            # 端点才可能被前端命中, 故调用匹配 / 遮蔽 / 零调用都只算「实存面」.
            "mounted": (Path(r["rel"]).stem, r["router"]) in mounted_pairs,
            "segments": _http_route_segments(r["path"]),
            "called": False,
        }
        for r in http_routes
    ]
    live = [ep for ep in endpoints if ep["mounted"]]

    calls: list[dict] = []
    for c in scan:
        segs = _http_fe_segments(c["path"])
        if segs is None:
            calls.append(
                {
                    **c,
                    "status": "external",
                    "note": "绝对 URL, 不计入",
                    "backend_methods": [],
                }
            )
            continue
        backend_methods: set[str] = set()
        for ep in live:
            if not _http_path_match(segs, ep["segments"]):
                continue
            backend_methods |= set(ep["methods"])
            if c["method"] in ep["methods"]:
                ep["called"] = True
        if not backend_methods:
            status, note = "no-source", "后端无此路径 (404 死链)"
        elif c["method"] in backend_methods:
            status, note = "wired", ""
        else:
            status = "method-mismatch"
            note = f"后端仅注册 {'/'.join(sorted(backend_methods))} (405)"
        calls.append(
            {**c, "status": status, "note": note, "backend_methods": sorted(backend_methods)}
        )

    by_module: dict[str, dict] = defaultdict(lambda: {"total": 0, "called": 0})
    for ep in live:
        slot = by_module[ep["rel"]]
        slot["total"] += 1
        if ep["called"]:
            slot["called"] += 1
    zero_modules = sorted(
        (
            {"rel": rel, **counts}
            for rel, counts in by_module.items()
            if counts["called"] == 0
        ),
        key=lambda x: x["rel"],
    )

    dup: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ep in live:
        for m in ep["methods"]:
            dup[(m, ep["path"])].add(ep["rel"])
    duplicates = sorted(
        (
            {"method": k[0], "path": k[1], "modules": sorted(v)}
            for k, v in dup.items()
            if len(v) > 1
        ),
        key=lambda d: (d["path"], d["method"]),
    )

    defined_pairs = {(Path(r["rel"]).stem, r["router"]) for r in all_routes}
    unmounted = sorted(
        f"{mod}.{var}" for mod, var in defined_pairs - set(registry["mounted_pairs"])
    )

    # 硬违例 (前端调用挂不上后端注册面) 逐条分诊: 未登记的即"待人工确认".
    hard = [c for c in calls if c["status"] in ("no-source", "method-mismatch")]
    for c in hard:
        tri = _http_triage(c["method"], c["path"])
        c["triage"] = tri[0] if tri else "untriaged"
        c["triage_reason"] = tri[1] if tri else ""

    return {
        "frontend": str(frontend),
        "endpoints": endpoints,
        "calls": calls,
        "dead_links": [c for c in calls if c["status"] == "no-source"],
        "method_mismatches": [c for c in calls if c["status"] == "method-mismatch"],
        "hard_violations": hard,
        "hard_untriaged": [c for c in hard if c["triage"] == "untriaged"],
        "duplicates": duplicates,
        "registration": {
            "mounted": registry["mounted"],
            "unmounted": unmounted,
            "dangling": registry["dangling"],
        },
        "zero_modules": zero_modules,
        "endpoint_count": len(endpoints),
        "live_endpoint_count": len(live),
        "called_endpoint_count": sum(1 for ep in live if ep["called"]),
        "ws_endpoint_count": len(all_routes) - len(http_routes),
    }


def render_http_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## HTTP API 消费面: 后端路由注册面 vs 前端 api.* 调用面")
    lines.append("")
    lines.append(
        "前八面核 agent 内部契约, SSE/WS 消费面核流式推送, 本面补齐**请求-响应**第三块"
        "传输拼图: 后端 `huginn/routes/*.py` 的 `@router.<method>(\"<path>\")` 是**注册"
        "生产面**, 前端 `desktop/src` 的 `api.get/post/put/patch/del/getBlob/upload*/"
        "search(...)` 是**调用消费面**. 桌面只是 HTTP API 的**一个**消费者 (外部客户端 / "
        "CLI / 测试也调), 故**只把「前端 → 后端」方向当硬契约**: 前端调了后端没注册的"
        "路径 = 404 死链, 方法对不上 = 405. 匹配只看**实存端点** (挂进 `ALL_ROUTERS` "
        "的模块), 未挂载模块的端点另在挂载面报, 不重复计."
    )
    lines.append("")
    lines.append("状态: " + "; ".join(f"`{k}`={v}" for k, v in _HTTP_STATUS_DOC.items()))
    lines.append("")
    lines.append(
        f"端点: 后端注册 **{contract['endpoint_count']}** 个 (实存 "
        f"**{contract['live_endpoint_count']}** 个; 另有 {contract['ws_endpoint_count']} 个 "
        f"WebSocket 端点归 WS 消费面); 桌面调用命中 **{contract['called_endpoint_count']}** "
        f"个; 前端调用点 **{len(contract['calls'])}** 处."
    )
    lines.append("")

    lines.append("### 前端调用点 (按状态)")
    lines.append("")
    lines.append("| 方法 | 路径 | 状态 | 位置 | 备注 |")
    lines.append("|---|---|---|---|---|")
    for c in sorted(contract["calls"], key=lambda x: (x["status"], x["rel"], x["line"])):
        lines.append(
            f"| `{c['method']}` | `{c['path']}` | `{c['status']}` "
            f"| `{c['rel']}:{c['line']}` | {c['note']} |"
        )
    lines.append("")

    lines.append("### 前端调用无源 (404 死链) / 方法不符 (405)")
    lines.append("")
    hard = contract["dead_links"] + contract["method_mismatches"]
    if hard:
        lines.append(
            "硬违例 —— 前端调用挂不上后端注册面. 逐条分诊: 未登记的落「待分诊」"
            "(回归测试会失败, 逼人工判定):"
        )
        lines.append("")
        for c in hard:
            lines.append(
                f"- `{c['method']} {c['path']}` @ `{c['rel']}:{c['line']}` — {c['note']}"
                + _http_violation_mark(c["method"], c["path"])
            )
    else:
        lines.append("- 无 —— 每个前端调用都命中后端已注册的方法+路径.")
    lines.append("")

    lines.append("### 同一 method+path 被多个**已挂载**模块注册 (路由遮蔽)")
    lines.append("")
    if contract["duplicates"]:
        for d in contract["duplicates"]:
            mods = ", ".join(f"`{m}`" for m in d["modules"])
            lines.append(f"- `{d['method']} {d['path']}` ← {mods}")
    else:
        lines.append("- 无 —— 每个 method+path 唯一注册.")
    lines.append("")

    reg = contract["registration"]
    lines.append("### 路由挂载面 (ALL_ROUTERS ↔ 各模块 APIRouter)")
    lines.append("")
    lines.append(f"- 已挂载: {len(reg['mounted'])} 个 router 变量")
    if reg["unmounted"]:
        for name in reg["unmounted"]:
            lines.append(f"- ⚠ 定义了 APIRouter 却未挂载 (整模块端点永不生效): `{name}`")
    else:
        lines.append("- 未挂载: 无 —— 每个定义路由的模块都被 `ALL_ROUTERS` 挂上.")
    for name in reg["dangling"]:
        lines.append(f"- ⚠ `ALL_ROUTERS` 引用了未 import 的别名: `{name}`")
    lines.append("")

    lines.append("### 桌面零调用的路由模块 (候选, 只算已挂载模块)")
    lines.append("")
    zm = contract["zero_modules"]
    if zm:
        lines.append(
            f"以下 {len(zm)} 个模块的端点**全部**无桌面调用 —— HTTP API 面向外部客户端 / "
            "CLI / 测试, 零调用是**结构性常态**, 非缺陷; 列此仅供「哪些面桌面根本没接」参考:"
        )
        lines.append("")
        lines.append("| 模块 | 端点数 |")
        lines.append("|---|---|")
        for m in zm:
            lines.append(f"| `{m['rel']}` | {m['total']} |")
    else:
        lines.append("- 无 —— 每个路由模块都至少有一个端点被桌面调用.")
    lines.append("")
    lines.append(
        "诚实边界: 前端只扫 `lib/api.ts` 的 `api.*` 包装 (裸 `fetch(...)` 与 EventSource "
        "在别面); 路径里的 `${…}` 只保留静态前缀, 动态拼接的段不可穷尽; "
        "`getBlob(path, { method: … })` 的方法覆盖按调用实参里的 `method:` 字面量近似判定;"
        " **请求体形状 / 必填 query 参数不核** —— 前端发 multipart 而后端要 JSON body、"
        "漏传必填 query 参数这类「路径对、负载错」静态不可辨, 不在本面 (只报 404/405 这类"
        "路径+方法级硬违例)."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 请求负载面: 前端调用实参形状 ↔ 后端模型必填字段 / query 参数
# ---------------------------------------------------------------------------
#
# HTTP 消费面只核「路径 + 方法」挂不挂得上 (404/405); 本面再往里一层, 核**同一个
# 命中端点的请求负载对不对**: 后端签名里的必填 query 参数 / 必填请求体 / Pydantic
# 模型必填字段 / Form·File 字段, 前端这次调用到底发了没有. 静态只能辨「字面量能读
# 出形状」的调用, 读到变量/模板即记 unknown 并跳过 (不猜).
#
# 硬契约方向同 HTTP 面: 只把「前端 → 后端」当硬违例 (漏必填 ⇒ 422). 反向「后端有
# 可选字段前端没发」不是违例.

# 后端特殊参数类型: FastAPI 直接注入, 既不是 query 也不是 body.
_PAYLOAD_SPECIAL_ANN = {
    "Request",
    "Response",
    "WebSocket",
    "BackgroundTasks",
    "HTTPConnection",
}
# Pydantic 模型基类 (含继承链, 用 fixpoint 解析).
_PAYLOAD_MODEL_BASES = {"BaseModel", "BaseSettings"}
# api.* 里走 multipart 的包装 (文件/表单上传).
_PAYLOAD_MULTIPART_VERBS = {"upload", "uploadWithProgress", "uploadStream"}
# FastAPI 里带 `...` 默认 = "必填" (ast.unparse(File(...)) == "File(...)").
_PAYLOAD_REQUIRED_DEFAULTS = {
    "File(...)",
    "Form(...)",
    "Query(...)",
    "Body(...)",
    "Header(...)",
    "Path(...)",
    "Cookie(...)",
}

_PAYLOAD_KIND_DOC = {
    "missing-query": "后端必填 query 参数前端未传 (422)",
    "missing-body": "后端必填请求体前端未发 (422)",
    "missing-body-field": "后端模型必填字段前端未含 (422)",
    "shape-mismatch": "前后端请求载体形状不符 (JSON ↔ multipart, 422)",
    "missing-form-field": "后端必填 Form/File 字段前端未含 (422)",
}

# 已确认硬违例分诊表. 键 (类型, 方法, 去 query 前端路径) → (标签, 理由); 未登记即
# "待分诊", 回归测试会失败 (逼逐条人工判定). 空表 = 当前每个命中端点的调用负载都合契约.
_PAYLOAD_CONFIRMED_VIOLATIONS: dict[tuple[str, str, str], tuple[str, str]] = {}

_PAYLOAD_TRIAGE_DOC = {
    "defect": "已确认缺陷 (待修)",
    "intentional": "已确认有意",
}


def _payload_triage(kind: str, method: str, path: str) -> tuple[str, str] | None:
    """负载违例分诊: 返回 (标签, 理由); 未登记则 None (待人工确认)."""
    return _PAYLOAD_CONFIRMED_VIOLATIONS.get(
        (kind, method.upper(), path.split("?")[0])
    )


def _payload_violation_mark(kind: str, method: str, path: str) -> str:
    tri = _payload_triage(kind, method, path)
    if tri is None:
        return " — ⚠ 待分诊"
    doc = _PAYLOAD_TRIAGE_DOC[tri[0]]
    return f" — {'⛔' if tri[0] == 'defect' else '✅'} {doc}: {tri[1]}"


def _payload_required_fields(cls: ast.ClassDef) -> set[str]:
    """Pydantic 模型的必填字段名 —— 无默认 / 默认是 `...` 即必填.

    `Field(...)`: 位置实参里有 `...` ⇒ 必填; 有位置实参但非 `...` ⇒ 有默认
    (`Field(None, …)`、`Field("auto", …)`); 无位置实参且无 `default` /
    `default_factory` 关键字 ⇒ 必填.
    """
    req: set[str] = set()
    for stmt in cls.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        v = stmt.value
        if v is None or (isinstance(v, ast.Constant) and v.value is Ellipsis):
            req.add(stmt.target.id)
        elif (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Name)
            and v.func.id == "Field"
        ):
            has_ellipsis = any(
                isinstance(a, ast.Constant) and a.value is Ellipsis for a in v.args
            )
            has_default_kw = any(
                k.arg in ("default", "default_factory") for k in v.keywords
            )
            if has_ellipsis or (not v.args and not has_default_kw):
                req.add(stmt.target.id)
    return req


def _payload_models(root: Path) -> dict[str, set[str]]:
    """全仓 Pydantic 模型名 → 必填字段集 (继承链 fixpoint 解析)."""
    classes: dict[str, ast.ClassDef] = {}
    for py in _iter_py(root):
        if "/tests/" in py.as_posix():
            continue
        tree = _parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, node)
    models: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for name, cls in classes.items():
            if name in models:
                continue
            bases = {ast.unparse(b).split("[")[0].strip() for b in cls.bases}
            if not (bases & _PAYLOAD_MODEL_BASES) and not (bases & set(models)):
                continue
            models[name] = _payload_required_fields(cls)
            changed = True
    return models


def _payload_base_ann(ann: str) -> str:
    """注解取基名: `dict[str, Any] | None` → `dict`, `LoadRequest` → `LoadRequest`."""
    return ann.split("[")[0].split("|")[0].strip()


def _payload_is_required(default: str | None) -> bool:
    """FastAPI 参数必填性: 无默认, 或默认是 `File(...)`/`Query(...)` 这类省略号标记."""
    if default is None:
        return True
    return default in _PAYLOAD_REQUIRED_DEFAULTS


def _payload_classify(param: dict, models: dict[str, set[str]]) -> str:
    """后端参数归类: query / path / body-model / body-dict / form / file / special."""
    ann = param["ann"].strip()
    dfl = (param["default"] or "").strip()
    if "Depends" in dfl:
        return "depends"
    base = _payload_base_ann(ann)
    if base in _PAYLOAD_SPECIAL_ANN:
        return "special"
    if "UploadFile" in ann or dfl.startswith("File("):
        return "file"
    if dfl.startswith("Form("):
        return "form"
    if base in ("dict", "Dict") or re.match(r"dict\s*\[", ann):
        return "body-dict"
    if base in models:
        return "body-model"
    if param["is_path"]:
        return "path"
    return "query"


def _payload_handler_params(node: ast.FunctionDef | ast.AsyncFunctionDef, path: str) -> list[dict]:
    """处理函数签名 → 参数表 (路径参数带 `is_path`, FastAPI 注入类型留给分类)."""
    path_names = {seg[1:-1] for seg in re.findall(r"\{[^}]+\}", path)}
    params: list[dict] = []
    pos = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(pos) - len(node.args.defaults)) + list(node.args.defaults)
    for a, d in zip(pos, defaults):
        if a.arg in ("self", "cls"):
            continue
        params.append(
            {
                "name": a.arg,
                "ann": ast.unparse(a.annotation) if a.annotation is not None else "",
                "default": ast.unparse(d) if d is not None else None,
                "is_path": a.arg in path_names,
            }
        )
    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
        params.append(
            {
                "name": a.arg,
                "ann": ast.unparse(a.annotation) if a.annotation is not None else "",
                "default": ast.unparse(d) if d is not None else None,
                "is_path": a.arg in path_names,
            }
        )
    return params


def _payload_handlers(root: Path) -> dict[tuple[str, str], list[dict]]:
    """(方法, 完整路径) → 处理函数参数表; 供负载核对查后端签名."""
    out: dict[tuple[str, str], list[dict]] = {}
    rdir = root / _HTTP_ROUTES_DIR
    if not rdir.is_dir():
        return out
    for py in sorted(rdir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        tree = _parse(py)
        if tree is None:
            continue
        prefixes = _http_router_prefixes(tree)
        if not prefixes:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for dec in node.decorator_list:
                parsed = _http_route_decorator(dec, prefixes)
                if parsed is None:
                    continue
                methods, path, _var = parsed
                if "WEBSOCKET" in methods:
                    continue
                params = _payload_handler_params(node, path)
                for m in methods:
                    out[(m, path)] = params
    return out


def _payload_split_args(inner: str) -> list[str]:
    """按顶层逗号切分调用实参 (跳过字符串字面量, 尊重括号嵌套)."""
    out: list[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(inner)
    while i < n:
        c = inner[i]
        if c in "\"'`":
            i = _skip_ts_string(inner, i)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(inner[start:i])
            start = i + 1
        i += 1
    tail = inner[start:]
    if tail.strip():
        out.append(tail)
    return out


def _payload_obj_keys(expr: str) -> tuple[set[str], bool]:
    """对象字面量 → (顶层键集, 是否含不可静态解析的部分); 非字面量即 unknown."""
    s = expr.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return set(), True
    keys: set[str] = set()
    for part in _payload_split_args(s[1:-1]):
        p = part.strip()
        if not p:
            continue
        if p.startswith("..."):
            return keys, True
        m = re.match(r"""([A-Za-z_$][\w$]*|['"][^'"]+['"])\s*:""", p)
        if m:
            keys.add(m.group(1).strip("'\""))
        elif re.fullmatch(r"[A-Za-z_$][\w$]*", p):
            keys.add(p)
        else:
            return keys, True
    return keys, False


def _payload_params_keys(expr: str) -> tuple[set[str], bool]:
    """`new URLSearchParams({...})` → 键集; 其他形式 (变量/模板) 记 unknown."""
    m = re.match(r"new\s+URLSearchParams\s*\((.*)\)\s*$", expr.strip(), re.S)
    if not m:
        return set(), True
    return _payload_obj_keys(m.group(1).strip())


def _payload_opt_params(opt: str) -> tuple[set[str], bool]:
    """请求选项对象字面量 → `params` 值的 query 名集.

    只解析顶层 `params: <URLSearchParams(...)>` 这一项; 选项不是对象字面量、或缺
    `params` / 值读不出形状, 都记 unknown (跳过该维度, 不猜).
    """
    s = opt.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return set(), True
    for part in _payload_split_args(s[1:-1]):
        m = re.match(r"\s*params\s*:\s*(.*)$", part, re.S)
        if m:
            return _payload_params_keys(m.group(1).strip())
    return set(), False


def _payload_query_names(raw: str) -> tuple[set[str], bool]:
    """路径字面量的 query → 参数名集. 值可含 `${}` (名仍可辨); 整段动态则 unknown."""
    if "?" not in raw:
        return set(), False
    names: set[str] = set()
    unknown = False
    for part in raw.split("?", 1)[1].split("&"):
        if not part:
            continue
        if "${" in part:
            head = part.split("${", 1)[0]
            if "=" in head:
                names.add(head.split("=", 1)[0].strip())
            else:
                unknown = True
        elif "=" in part:
            names.add(part.split("=", 1)[0].strip())
        else:
            unknown = True
    return {n for n in names if n}, unknown


def _payload_scan_frontend(frontend: Path) -> list[dict]:
    """扫前端 `api.*` 调用点 → 每个调用的负载形状 (query 名 / body 键 / multipart 字段).

    与 HTTP 面同一套定位逻辑, 额外读实参: 读得出字面量形状就记集合, 读到变量或模板
    就置 unknown (不猜, 后续核对跳过该维度).
    """
    calls: list[dict] = []
    if not frontend.is_dir():
        return calls
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        rel = _display(frontend, p)
        if rel.endswith(".spec.ts") or rel.endswith(".spec.tsx"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _HTTP_CALL_RE.finditer(text):
            verb = m.group(1)
            j = m.end()
            if j < len(text) and text[j] == "<":
                j = _http_skip_generics(text, j)
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if j >= len(text) or text[j] != "(":
                continue
            args = _payload_split_args(text[j + 1 : _http_call_args_end(text, j)])
            raw = args[0].strip() if args else ""
            if raw and raw[0] in "\"'`":
                raw = raw[1:-1]
            query, query_unknown = _payload_query_names(raw)
            body_keys: set[str] = set()
            body_unknown = False
            has_body = False
            fields: set[str] = set()
            fields_unknown = True
            if verb in ("post", "put", "patch"):
                if len(args) >= 2 and args[1].strip():
                    has_body = True
                    body_keys, body_unknown = _payload_obj_keys(args[1])
                opt_i = 2
            elif verb in ("get", "del", "getBlob"):
                opt_i = 1
            elif verb == "search":
                query = {"query", "limit", "sources"}
                opt_i = None
            elif verb in _PAYLOAD_MULTIPART_VERBS:
                opt_i = 1
                if verb == "uploadStream":
                    fields, fields_unknown = {"file"}, False
                elif verb == "uploadWithProgress" and len(args) >= 4:
                    fields, fields_unknown = _payload_obj_keys(args[3])
                    fields |= {"file"}  # 三个上传包装都固定 append("file", file)
            else:
                opt_i = 1
            if opt_i is not None and len(args) > opt_i:
                pk, pu = _payload_opt_params(args[opt_i])
                query |= pk
                query_unknown = query_unknown or pu
            method = _HTTP_VERB_METHOD.get(verb, verb.upper())
            override = _HTTP_METHOD_OVERRIDE_RE.search("".join(args))
            if override:
                method = override.group(1).upper()
            calls.append(
                {
                    "rel": rel,
                    "line": text.count("\n", 0, m.start()) + 1,
                    "verb": verb,
                    "method": method,
                    "path": raw,
                    "query": sorted(query),
                    "query_unknown": query_unknown,
                    "body_keys": sorted(body_keys),
                    "body_unknown": body_unknown,
                    "has_body": has_body,
                    "fields": sorted(fields),
                    "fields_unknown": fields_unknown,
                }
            )
    return calls


def _payload_live_endpoints(root: Path) -> list[dict]:
    """已挂进 `ALL_ROUTERS` 的 HTTP 端点 (未挂载模块的端点根本不在 app 上)."""
    mounted_pairs = set(_http_registry(root)["mounted_pairs"])
    out: list[dict] = []
    for r in _http_backend_routes(root):
        if "WEBSOCKET" in r["methods"]:
            continue
        if (Path(r["rel"]).stem, r["router"]) not in mounted_pairs:
            continue
        out.append(
            {
                "methods": r["methods"],
                "path": r["path"],
                "rel": r["rel"],
                "segments": _http_route_segments(r["path"]),
            }
        )
    return out


def _payload_violations_for(
    c: dict, end_path: str, params: list[dict], models: dict[str, set[str]], stats: dict
) -> list[dict]:
    """一次命中端点的调用 → 负载硬违例列表 (仅「前端漏发后端必填」方向)."""
    out: list[dict] = []

    def viol(kind: str, detail: str, model: str = "") -> None:
        out.append(
            {
                "kind": kind,
                "method": c["method"],
                "path": c["path"],
                "endpoint": end_path,
                "rel": c["rel"],
                "line": c["line"],
                "model": model,
                "detail": detail,
            }
        )

    kinds = [(p["name"], _payload_classify(p, models), p) for p in params]
    req_query = [
        n for n, k, p in kinds if k == "query" and _payload_is_required(p["default"])
    ]
    form_names = {n for n, k, _ in kinds if k in ("file", "form")}
    req_form = [
        n
        for n, k, p in kinds
        if k in ("file", "form") and _payload_is_required(p["default"])
    ]
    body_model = next((p for _n, k, p in kinds if k == "body-model"), None)
    body_dict = next((p for _n, k, p in kinds if k == "body-dict"), None)
    multipart = c["verb"] in _PAYLOAD_MULTIPART_VERBS
    json_body = c["method"] in ("POST", "PUT", "PATCH") and not multipart

    # (1) 必填 query 参数未传
    if req_query:
        if c["query_unknown"]:
            stats["query_skipped"] += 1
        else:
            stats["query_checked"] += 1
            miss = [n for n in req_query if n not in set(c["query"])]
            if miss:
                viol("missing-query", f"后端必填 query 未传: {', '.join(miss)}")
    # (2) 请求载体形状不符 (JSON ↔ multipart)
    if multipart and not form_names:
        viol("shape-mismatch", "前端走 multipart 上传, 后端无 Form/File 参数")
    if json_body and c["has_body"] and form_names and not body_model and not body_dict:
        viol("shape-mismatch", "前端发 JSON body, 后端只收 Form/File")
    # (3) 必填请求体整体缺失
    body_param = body_model or body_dict
    if (
        body_param is not None
        and json_body
        and _payload_is_required(body_param["default"])
        and not c["has_body"]
    ):
        viol(
            "missing-body",
            f"后端必填请求体 `{body_param['name']}` 未发",
            model=_payload_base_ann(body_model["ann"]) if body_model else "",
        )
    # (4) 模型必填字段未含
    if body_model is not None and c["has_body"]:
        if c["body_unknown"]:
            stats["body_skipped"] += 1
        else:
            stats["body_checked"] += 1
            mname = _payload_base_ann(body_model["ann"])
            miss = sorted(
                n for n in models.get(mname, set()) if n not in set(c["body_keys"])
            )
            if miss:
                viol(
                    "missing-body-field",
                    f"模型 `{mname}` 必填字段未含: {', '.join(miss)}",
                    model=mname,
                )
    # (5) multipart 必填 Form/File 字段未含
    if multipart and req_form and not c["fields_unknown"]:
        stats["form_checked"] += 1
        miss = [n for n in req_form if n not in set(c["fields"])]
        if miss:
            viol("missing-form-field", f"未含必填表单/文件字段: {', '.join(miss)}")
    return out


def build_payload_contract(root: Path | None = None, frontend: Path | None = None) -> dict:
    """请求负载面: 前端调用实参形状 ↔ 后端签名必填项.

    HTTP 面核路径/方法挂不挂得上; 本面在**已命中的端点**上核负载: 后端必填的 query
    参数 / 请求体 / Pydantic 模型必填字段 / Form·File 字段, 前端这次调用发了没有.
    只报「前端漏发后端必填」(422); 读到变量/模板即记 unknown 跳过该维度.
    """
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    models = _payload_models(root)
    handlers = _payload_handlers(root)
    live = _payload_live_endpoints(root)
    calls = _payload_scan_frontend(frontend)

    stats = {
        "wired": 0,
        "query_checked": 0,
        "query_skipped": 0,
        "body_checked": 0,
        "body_skipped": 0,
        "form_checked": 0,
    }
    violations: list[dict] = []
    for c in calls:
        segs = _http_fe_segments(c["path"])
        if segs is None:
            continue
        eps = [
            ep
            for ep in live
            if c["method"] in ep["methods"] and _http_path_match(segs, ep["segments"])
        ]
        if not eps:
            continue
        ep = eps[0]
        stats["wired"] += 1
        params = handlers.get((c["method"], ep["path"]), [])
        violations.extend(_payload_violations_for(c, ep["path"], params, models, stats))

    for v in violations:
        tri = _payload_triage(v["kind"], v["method"], v["path"])
        v["triage"] = tri[0] if tri else "untriaged"
        v["triage_reason"] = tri[1] if tri else ""

    kinds = Counter(v["kind"] for v in violations)
    return {
        "frontend": str(frontend),
        "call_count": len(calls),
        "wired_call_count": stats["wired"],
        "model_count": len(models),
        "violations": violations,
        "untriaged": [v for v in violations if v["triage"] == "untriaged"],
        "kind_counts": dict(sorted(kinds.items())),
        "coverage": {
            "query_checked": stats["query_checked"],
            "query_skipped": stats["query_skipped"],
            "body_checked": stats["body_checked"],
            "body_skipped": stats["body_skipped"],
            "form_checked": stats["form_checked"],
        },
    }


def render_payload_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 请求负载面: 前端调用实参形状 vs 后端签名必填项")
    lines.append("")
    lines.append(
        "HTTP 消费面核「路径 + 方法」挂不挂得上 (404/405); 本面再往里一层, 核**已命中"
        "端点**上的请求负载: 后端 `@router.<method>` 处理函数签名里的必填 query 参数 / "
        "必填请求体 / Pydantic 模型必填字段 / `Form · File` 字段, 前端这次 `api.*` 调用"
        "到底发了没有. 硬契约方向同 HTTP 面 —— 只把「前端漏发后端必填」当违例 (422 死负载);"
        "「后端有可选字段前端没发」不是违例."
    )
    lines.append("")
    lines.append(
        "违例类型: " + "; ".join(f"`{k}`={v}" for k, v in _PAYLOAD_KIND_DOC.items())
    )
    lines.append("")
    cov = contract["coverage"]
    lines.append(
        f"覆盖: 命中端点的调用 **{contract['wired_call_count']}** 处 (共 "
        f"{contract['call_count']} 个 `api.*` 调用点); 后端模型 **{contract['model_count']}** "
        f"个; 可静态核对 —— query {cov['query_checked']} / body {cov['body_checked']} / "
        f"multipart {cov['form_checked']} 处."
    )
    counts = (
        "  " + ", ".join(f"`{k}`×{n}" for k, n in contract["kind_counts"].items())
        if contract["kind_counts"]
        else ""
    )
    lines.append(f"违例: **{len(contract['violations'])}** 条.{counts}")
    lines.append("")

    lines.append("### 违例 (前端漏发后端必填)")
    lines.append("")
    if contract["violations"]:
        lines.append(
            "硬违例 —— 后端必填项前端没发, 请求必 422. 逐条分诊: 未登记的落「待分诊」"
            "(回归测试会失败, 逼人工判定):"
        )
        lines.append("")
        for v in sorted(
            contract["violations"],
            key=lambda x: (x["kind"], x["method"], x["path"], x["rel"], x["line"]),
        ):
            lines.append(
                f"- `[{v['kind']}]` `{v['method']} {v['path']}` → 后端 `{v['endpoint']}` "
                f"@ `{v['rel']}:{v['line']}` — {v['detail']}"
                + _payload_violation_mark(v["kind"], v["method"], v["path"])
            )
    else:
        lines.append("- 无 —— 每个命中端点的调用, 静态可辨的负载都满足后端必填项.")
    lines.append("")

    lines.append("### 静态核对覆盖面 (读不出形状即跳过, 不猜)")
    lines.append("")
    lines.append("| 维度 | 已核对 | 跳过 (变量/模板/未知形状) |")
    lines.append("|---|---|---|")
    lines.append(f"| 必填 query 参数 | {cov['query_checked']} | {cov['query_skipped']} |")
    lines.append(f"| 必填模型字段 | {cov['body_checked']} | {cov['body_skipped']} |")
    lines.append(f"| multipart 必填字段 | {cov['form_checked']} | — |")
    lines.append("")
    lines.append(
        "诚实边界: 只读**字面量**形状 —— body 传变量、`params` 传变量、路径 query 整段"
        "动态时该维度记 unknown 并跳过, 故违例是**下界** (可能漏报); 后端 `dict` 请求体"
        "无字段约束只核「发没发」; 模型必填性按 Pydantic 默认值 / `Field(...)` 静态判定, "
        "`model_config` 与 `Field` 的 `validate_*` 细粒度约束不核; 反向 (前端多发字段、"
        "后端可选字段缺失) 不是违例."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 响应结构面: 后端 return 形状 ↔ 前端声明消费的响应字段
# ---------------------------------------------------------------------------
# 请求负载面核「前端发的后端要不要求」; 本面反向核「后端返的前端读不读得到」。硬契约
# 方向: 前端**声明**要读的响应字段 (api.* 的 TS 泛型实参) 后端处理函数**从不返回**
# (return 字面量里没有, 且形状封闭) ⇒ 该字段恒 undefined (静默坏)。反向「后端返回了
# 前端没读的字段」不是违例 (响应本就可冗余).

# TS 响应类型里的字段声明: `name?: T` / `name: T` / `name(...)` / `'a-b': T`.
_RESP_FIELD_RE = re.compile(r"""([A-Za-z_$][\w$]*|['"][^'"]+['"])\s*\??\s*[:(]""")
# 非对象的 TS 响应类型 (标量/无形状) —— 与「对象字段」核对无关.
_RESP_NON_OBJECT = {
    "any",
    "void",
    "string",
    "number",
    "boolean",
    "unknown",
    "null",
    "undefined",
    "Record",
}
# 前端类型声明 (`interface Foo {…}` / `type Foo = {…}`).
_RESP_TYPE_DECL_RE = re.compile(
    r"(?:^|\n)\s*(?:export\s+)?(?:declare\s+)?(?:interface|type)\s+([A-Za-z_$][\w$]*)"
)

_RESP_KIND_DOC = {
    "missing-field": "前端声明的响应字段后端从不返回 (恒 undefined)",
}

# 已确认硬违例分诊表. 键 (类型, 方法, 去 query 前端路径) → (标签, 理由); 未登记即
# "待分诊", 回归测试会失败. 空表 = 当前每个命中端点声明的响应字段都在后端 return 里.
_RESP_CONFIRMED_VIOLATIONS: dict[tuple[str, str, str], tuple[str, str]] = {}

_RESP_TRIAGE_DOC = {
    "defect": "已确认缺陷 (待修)",
    "intentional": "已确认有意",
}


def _resp_triage(kind: str, method: str, path: str) -> tuple[str, str] | None:
    """响应结构违例分诊: 返回 (标签, 理由); 未登记则 None (待人工确认)."""
    return _RESP_CONFIRMED_VIOLATIONS.get((kind, method.upper(), path.split("?")[0]))


def _resp_violation_mark(kind: str, method: str, path: str) -> str:
    tri = _resp_triage(kind, method, path)
    if tri is None:
        return " — ⚠ 待分诊"
    return f" — {'⛔' if tri[0] == 'defect' else '✅'} {_RESP_TRIAGE_DOC[tri[0]]}: {tri[1]}"


def _resp_split_members(inner: str) -> list[str]:
    """按顶层 `,` / `;` 切分 TS 类型成员 (跳过字符串字面量, 尊重括号/泛型嵌套).

    TS 对象类型成员分隔符是 `;` / `,` / 换行, 与 JS 对象字面量的逗号不同, 故不能复用
    `_payload_split_args`. 嵌套 `{…}` / `(…)` / `Record<…>` 里的 `,` `;` 不切.
    `<` 只在非 `<=` 时计入, `>` 只在非 `=>` 时配对, 避免把箭头函数类型误当泛型.
    """
    out: list[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(inner)
    while i < n:
        c = inner[i]
        if c in "\"'`":
            i = _skip_ts_string(inner, i)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "<" and inner[i - 1 : i] != "=" and inner[i + 1 : i + 2] != "=":
            depth += 1
        elif c == ">" and inner[i - 1 : i] != "=":
            depth -= 1
        elif c in ",;" and depth == 0:
            out.append(inner[start:i])
            start = i + 1
        i += 1
    tail = inner[start:]
    if tail.strip():
        out.append(tail)
    return out


def _resp_type_keys(expr: str) -> tuple[set[str], bool]:
    """TS 响应类型表达式 → (顶层字段名集, 是否开放/读不出).

    `{ a?: T; b: U }` → 字段集; `X[]` / 标量 / 交叉含 `Record` / index signature /
    `&` 拼接 → 记开放 (有读不出的键, 跳过该维度). 只认字面量对象类型, 不猜.
    """
    s = expr.strip()
    if not s or s.endswith("[]") or s in _RESP_NON_OBJECT:
        return set(), True
    if not (s.startswith("{") and s.endswith("}")):
        return set(), True
    keys: set[str] = set()
    for part in _resp_split_members(s[1:-1]):
        p = part.strip()
        if not p:
            continue
        if p.startswith("["):  # index signature → 有额外键
            return keys, True
        m = _RESP_FIELD_RE.match(p)
        if m:
            keys.add(m.group(1).strip("'\""))
        else:
            return keys, True
    return keys, False


def _resp_brace_end(text: str, open_idx: int) -> int:
    """从 `{` 起找配对的 `}` (跳过字符串与模板字面量); 未闭合返回文末."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            i = _skip_ts_string(text, i)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n


def _resp_frontend_types(frontend: Path) -> dict[str, tuple[set[str], bool]]:
    """前端 `interface Foo {…}` / `type Foo = {…}` → {名: (字段集, 开放)}."""
    out: dict[str, tuple[set[str], bool]] = {}
    if not frontend.is_dir():
        return out
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _RESP_TYPE_DECL_RE.finditer(text):
            name = m.group(1)
            if name in out:
                continue
            head_end = text.find("{", m.end())
            if head_end < 0:
                continue
            # `interface Foo extends Bar {…}` / `type Foo = Bar & {…}`: 继承面读不出 ⇒ 开放.
            head = text[m.end() : head_end]
            if "extends" in head or "&" in head:
                out[name] = set(), True
                continue
            body_end = _resp_brace_end(text, head_end)
            out[name] = _resp_type_keys(text[head_end : body_end + 1])
    return out


def _resp_model_fields(cls: ast.ClassDef) -> set[str]:
    """Pydantic 模型的**全部**字段名 (必填 + 可选), 供 response_model 取响应键."""
    return {
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def _resp_local_models(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """同模块内定义的 Pydantic 模型 → {名: ClassDef} (跨模块引用读不出, 留空)."""
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            _payload_base_ann(ast.unparse(b)) in _PAYLOAD_MODEL_BASES for b in node.bases
        )
    }


def _resp_response_model(
    dec: ast.expr, local_models: dict[str, ast.ClassDef]
) -> tuple[set[str], bool] | None:
    """装饰器 `response_model=X` → 响应键; 无该 kwargs 返回 None, 引不到模型则空+开放."""
    if not isinstance(dec, ast.Call):
        return None
    for kw in dec.keywords:
        if kw.arg != "response_model":
            continue
        base = _payload_base_ann(ast.unparse(kw.value))
        cls = local_models.get(base)
        if cls is None:
            return set(), True
        return _resp_model_fields(cls), False
    return None


def _resp_own_walk(node: ast.AST):
    """遍历函数**自身**语法子树, 不下潜嵌套 def/lambda.

    嵌套 helper 的 `return` 属于该 helper, 不是端点的响应形状 —— `ast.walk` 会误收
    (如 `/transfer/web/upload` 的 `return (dest, expanded)`)。
    """
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for child in ast.iter_child_nodes(n):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue
            stack.append(child)


def _resp_return_guarded(node: ast.AST, ret: ast.Return, name: str) -> bool:
    """`return name` 是否被 `if name:` / `if name is not None:` 包裹.

    真值守卫排除该变量的 null 分支: `err = _check_thread_owner(...); if err: return err`
    里 `err` 可能为 None, 但这里只会走到「err 为真」的分支。
    """
    parents: dict[int, ast.AST] = {}
    for n in _resp_own_walk(node):
        for child in ast.iter_child_nodes(n):
            parents[id(child)] = n
    cur: ast.AST | None = ret
    while cur is not None and id(cur) in parents:
        par = parents[id(cur)]
        if isinstance(par, ast.If):
            test = ast.unparse(par.test).strip()
            if test == name or test == f"{name} is not None":
                return True
        cur = par
    return False


class _RespShapeResolver:
    """单文件内的响应形状解析 (保守下界).

    在「字面量 return 并集」之上再解析两类可静态推导的来源:
      - **同模块 helper**: `return f(...)` / `return await f(...)`, `f` 是本文件模块级
        函数 ⇒ 递归取其 return 并集 (带环保护, 环 ⇒ 开放)。跨模块一律不解析。
      - **局部变量**: `err = helper(...); if err: return err` ⇒ 真值守卫排除 null 分支;
        `result["k"] = …` (常量键) 记入键集, 但 `.update()` / 增广赋值 / 非常量键等
        读不出的改写 ⇒ 开放。

    读到确切键才判封闭; 任一 return 读不出即开放 (宁可不闭, 不可错闭)。
    """

    def __init__(self, tree: ast.Module, local_models: dict[str, ast.ClassDef]):
        self._funcs = {
            n.name: n
            for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        self._models = local_models
        self._memo: dict[str, tuple[set[str], bool, bool]] = {}
        self._busy: set[str] = set()

    def handler(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set[str], bool]:
        """端点函数 → (响应顶层键集, 是否封闭)."""
        for dec in node.decorator_list:
            rm = _resp_response_model(dec, self._models)
            if rm is not None:
                return rm
        keys: set[str] = set()
        open_ = False
        seen = False
        for n in _resp_own_walk(node):
            if not isinstance(n, ast.Return):
                continue
            seen = True
            k, o, mn = self._return(n, node)
            keys |= k
            open_ = open_ or o or mn  # 非守卫的 null 可能 ⇒ 开放 (字段可能缺失)
        return keys, (seen and not open_)

    def _return(
        self, ret: ast.Return, scope: ast.AST
    ) -> tuple[set[str], bool, bool]:
        """单个 return → (键集, 是否开放, 是否可能为 null)."""
        if (
            isinstance(ret.value, ast.Name)
            and _resp_return_guarded(scope, ret, ret.value.id)
        ):
            keys, open_, _none = self._var(ret.value.id, scope)
            return keys, open_, False
        return self._value(ret.value, scope)

    def _value(self, expr: ast.expr | None, scope: ast.AST) -> tuple[set[str], bool, bool]:
        if expr is None or (isinstance(expr, ast.Constant) and expr.value is None):
            return set(), False, True  # 裸 return / `return None` (响应 null) ⇒ 字段可能缺失
        if isinstance(expr, ast.Dict):
            keys: set[str] = set()
            open_ = False
            for k in expr.keys:
                if k is None:  # `**x` 展开 ⇒ 有额外键
                    open_ = True
                elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
                else:
                    open_ = True
            return keys, open_, False
        if isinstance(expr, ast.Await):
            return self._value(expr.value, scope)
        if isinstance(expr, ast.Call):
            fn = expr.func
            name = fn.id if isinstance(fn, ast.Name) else None
            if name and name in self._funcs:
                return self._func(name)
            cls = self._models.get(name) if name else None
            if cls is not None:
                return _resp_model_fields(cls), False, False
            return set(), True, False
        if isinstance(expr, ast.Name):
            return self._var(expr.id, scope)
        return set(), True, False

    def _var(self, name: str, scope: ast.AST) -> tuple[set[str], bool, bool]:
        keys: set[str] = set()
        open_ = False
        may_none = False
        found = False
        for n in _resp_own_walk(scope):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        found = True
                        k, o, mn = self._value(n.value, scope)
                        keys |= k
                        open_ = open_ or o
                        may_none = may_none or mn
                    elif (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == name
                    ):
                        if (
                            isinstance(t.slice, ast.Constant)
                            and isinstance(t.slice.value, str)
                        ):
                            keys.add(t.slice.value)
                        else:
                            open_ = True
                        found = True
            elif (
                isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == name
            ):
                found = True
                k, o, mn = self._value(n.value, scope)
                keys |= k
                open_ = open_ or o
                may_none = may_none or mn
            elif isinstance(n, ast.AugAssign) and (
                isinstance(n.target, ast.Subscript)
                and isinstance(n.target.value, ast.Name)
                and n.target.value.id == name
            ):
                found = True
                if isinstance(n.target.slice, ast.Constant) and isinstance(
                    n.target.slice.value, str
                ):
                    keys.add(n.target.slice.value)
                else:
                    open_ = True
            elif (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == name
            ):
                # `name.update(...)` 等读不出的改写 ⇒ 开放
                found = True
                open_ = True
        if not found:
            return set(), True, False
        return keys, (open_ or not keys), may_none

    def _func(self, name: str) -> tuple[set[str], bool, bool]:
        if name in self._busy:  # 环 ⇒ 保守开放
            return set(), True, False
        if name in self._memo:
            return self._memo[name]
        node = self._funcs[name]
        self._busy.add(name)
        keys: set[str] = set()
        open_ = False
        may_none = False
        seen = False
        for n in _resp_own_walk(node):
            if not isinstance(n, ast.Return):
                continue
            seen = True
            k, o, mn = self._return(n, node)
            keys |= k
            open_ = open_ or o
            may_none = may_none or mn
        self._busy.discard(name)
        res = (keys, (open_ or not seen or not keys), may_none)
        self._memo[name] = res
        return res


def _resp_backend_shapes(root: Path) -> dict[tuple[str, str], dict]:
    """已挂载端点的响应形状: (方法, 路径) → {keys, closed, rel, handler}."""
    out: dict[tuple[str, str], dict] = {}
    mounted_pairs = set(_http_registry(root)["mounted_pairs"])
    rdir = root / _HTTP_ROUTES_DIR
    if not rdir.is_dir():
        return out
    for py in sorted(rdir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        tree = _parse(py)
        if tree is None:
            continue
        prefixes = _http_router_prefixes(tree)
        if not prefixes:
            continue
        rel = py.relative_to(root).as_posix()
        local_models = _resp_local_models(tree)
        resolver = _RespShapeResolver(tree, local_models)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for dec in node.decorator_list:
                parsed = _http_route_decorator(dec, prefixes)
                if parsed is None:
                    continue
                methods, path, var = parsed
                if "WEBSOCKET" in methods:
                    continue
                if (Path(rel).stem, var) not in mounted_pairs:
                    continue
                keys, closed = resolver.handler(node)
                for m in methods:
                    out[(m, path)] = {
                        "keys": keys,
                        "closed": closed,
                        "rel": rel,
                        "handler": node.name,
                    }
    return out


def _resp_scan_frontend(
    frontend: Path, types: dict[str, tuple[set[str], bool]]
) -> list[dict]:
    """扫前端 `api.*` 调用点 → 声明的响应字段形状.

    泛型实参 `api.get<{…}>` 是前端**声明**的响应契约: 字面量对象类型取字段集, 具名类型
    经前端 `interface`/`type` 解析; `<any>` / 数组 / 交叉含 `Record` / 无泛型 ⇒ 记开放
    (跳过该维度, 不猜)。
    """
    calls: list[dict] = []
    if not frontend.is_dir():
        return calls
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        rel = _display(frontend, p)
        if rel.endswith(".spec.ts") or rel.endswith(".spec.tsx"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _HTTP_CALL_RE.finditer(text):
            verb = m.group(1)
            j = m.end()
            generic = ""
            if j < len(text) and text[j] == "<":
                k = _http_skip_generics(text, j)
                generic = text[j + 1 : k - 1].strip()
                j = k
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if j >= len(text) or text[j] != "(":
                continue
            args = _payload_split_args(text[j + 1 : _http_call_args_end(text, j)])
            raw = args[0].strip() if args else ""
            if raw and raw[0] in "\"'`":
                raw = raw[1:-1]
            declared, declared_open = _resp_declared(generic, types)
            method = _HTTP_VERB_METHOD.get(verb, verb.upper())
            override = _HTTP_METHOD_OVERRIDE_RE.search("".join(args))
            if override:
                method = override.group(1).upper()
            calls.append(
                {
                    "rel": rel,
                    "line": text.count("\n", 0, m.start()) + 1,
                    "method": method,
                    "path": raw,
                    "declared": sorted(declared),
                    "declared_open": declared_open,
                }
            )
    return calls


def _resp_declared(
    generic: str, types: dict[str, tuple[set[str], bool]]
) -> tuple[set[str], bool]:
    """泛型实参 → (声明字段集, 是否开放). 具名类型查前端声明表."""
    if not generic:
        return set(), True
    keys, open_ = _resp_type_keys(generic)
    if open_ and not keys and generic in types:
        return types[generic]
    return keys, open_


def build_response_contract(
    root: Path | None = None, frontend: Path | None = None
) -> dict:
    """响应结构面: 后端 return 形状 ↔ 前端声明的响应字段.

    只报「前端声明要读的字段, 后端从不返回」这一硬方向 (静默 undefined); 后端形状
    开放 (return 变量 / `**` / Response 对象 / 引不到的 response_model) 或前端声明开放
    (`<any>` / 数组 / `Record` 交叉 / 无泛型) 时该维度跳过, 不猜。路径歧义 (动态段同时
    命中多个端点且强位置并列) 一律跳过。
    """
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    shapes = _resp_backend_shapes(root)
    live = _payload_live_endpoints(root)
    types = _resp_frontend_types(frontend)
    calls = _resp_scan_frontend(frontend, types)

    stats = {"wired": 0, "checked": 0, "skip_shape": 0, "skip_decl": 0, "skip_ambiguous": 0}
    violations: list[dict] = []
    for c in calls:
        segs = _http_fe_segments(c["path"])
        if segs is None:
            continue
        ep = _http_pick_endpoint(segs, c["method"], live)
        if ep is None:
            if any(
                c["method"] in e["methods"] and _http_path_match(segs, e["segments"])
                for e in live
            ):
                stats["skip_ambiguous"] += 1
            continue
        shape = shapes.get((c["method"], ep["path"]))
        if shape is None:
            continue
        stats["wired"] += 1
        if not shape["closed"]:
            stats["skip_shape"] += 1
            continue
        if c["declared_open"]:
            stats["skip_decl"] += 1
            continue
        stats["checked"] += 1
        missing = sorted(set(c["declared"]) - shape["keys"])
        if missing:
            violations.append(
                {
                    "kind": "missing-field",
                    "method": c["method"],
                    "path": c["path"],
                    "endpoint": ep["path"],
                    "rel": c["rel"],
                    "line": c["line"],
                    "declared": c["declared"],
                    "produced": sorted(shape["keys"]),
                    "missing": missing,
                }
            )

    for v in violations:
        tri = _resp_triage(v["kind"], v["method"], v["path"])
        v["triage"] = tri[0] if tri else "untriaged"
        v["triage_reason"] = tri[1] if tri else ""

    kinds = Counter(v["kind"] for v in violations)
    return {
        "frontend": str(frontend),
        "call_count": len(calls),
        "wired_call_count": stats["wired"],
        "type_count": len(types),
        "violations": violations,
        "untriaged": [v for v in violations if v["triage"] == "untriaged"],
        "kind_counts": dict(sorted(kinds.items())),
        "coverage": {
            "checked": stats["checked"],
            "skip_shape": stats["skip_shape"],
            "skip_decl": stats["skip_decl"],
            "skip_ambiguous": stats["skip_ambiguous"],
        },
    }


def render_response_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## 响应结构面: 后端 return 形状 vs 前端声明的响应字段")
    lines.append("")
    lines.append(
        "请求负载面核「前端发的后端要不要求」; 本面反向核「后端返的前端读不读得到」。"
        "前端 `api.*<T>` 的泛型实参 `T` 是**声明的响应契约**: 对象类型的每个字段, 后端"
        "处理函数的 `return` 字面量 / `response_model` 里到底有没有。硬方向 —— 只把"
        "「前端声明要读的字段后端从不返回」(恒 undefined, 静默坏) 当违例; 「后端返回了"
        "前端没读的字段」不是违例 (响应本就可冗余)。"
    )
    lines.append("")
    lines.append(
        "违例类型: " + "; ".join(f"`{k}`={v}" for k, v in _RESP_KIND_DOC.items())
    )
    lines.append("")
    cov = contract["coverage"]
    lines.append(
        f"覆盖: 命中端点的调用 **{contract['wired_call_count']}** 处 (共 "
        f"{contract['call_count']} 个 `api.*` 调用点); 前端类型 **{contract['type_count']}** "
        f"个; 可静态核对 **{cov['checked']}** 处."
    )
    counts = (
        "  " + ", ".join(f"`{k}`×{n}" for k, n in contract["kind_counts"].items())
        if contract["kind_counts"]
        else ""
    )
    lines.append(f"违例: **{len(contract['violations'])}** 条.{counts}")
    lines.append("")

    lines.append("### 违例 (前端声明要读, 后端从不返回)")
    lines.append("")
    if contract["violations"]:
        lines.append(
            "硬违例 —— 前端声明/读取的字段后端 return 里没有, 运行时恒 undefined。逐条"
            "分诊: 未登记的落「待分诊」(回归测试会失败, 逼人工判定):"
        )
        lines.append("")
        for v in sorted(
            contract["violations"],
            key=lambda x: (x["method"], x["path"], x["rel"], x["line"]),
        ):
            lines.append(
                f"- `[{v['kind']}]` `{v['method']} {v['path']}` → 后端 `{v['endpoint']}` "
                f"@ `{v['rel']}:{v['line']}` — 缺字段 {', '.join(f'`{f}`' for f in v['missing'])}; "
                f"后端实际返回 {', '.join(f'`{k}`' for k in v['produced']) or '(无)'}"
                + _resp_violation_mark(v["kind"], v["method"], v["path"])
            )
    else:
        lines.append("- 无 —— 每个命中端点声明要读的字段, 后端 return 里都有.")
    lines.append("")

    lines.append("### 静态核对覆盖面 (读不出形状即跳过, 不猜)")
    lines.append("")
    lines.append("| 维度 | 已核对 | 跳过 (形状开放/读不出) |")
    lines.append("|---|---|---|")
    lines.append(f"| 后端响应形状封闭 | {cov['checked']} | {cov['skip_shape']} |")
    lines.append(f"| 前端声明可解析 | {cov['checked']} | {cov['skip_decl']} |")
    lines.append(f"| 路径唯一命中 | — | {cov['skip_ambiguous']} (歧义跳过) |")
    lines.append("")
    lines.append(
        "诚实边界: 只读**字面量**形状 —— 后端 `return` 传变量 / `**` 展开 / `Response`"
        "对象 / 引不到的 `response_model` 即记开放并跳过, 故违例是**下界** (可能漏报); "
        "前端泛型为 `<any>` / 数组 / `Record` 交叉 / 无泛型时该维度跳过; 类型解析只认同仓"
        "`interface`/`type` 字面量对象, `extends` / `&` 拼接一律记开放; 路径动态段同时命中"
        "多个端点且强位置并列时跳过 (不猜端点); 反向 (后端返回前端没读的字段) 不是违例."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WS 请求负载面: WSMessage 声明字段 ↔ 后端 handler 读取面 ↔ 前端发送面
# ---------------------------------------------------------------------------
# WS 消费面只核「入站 type 认不认」; 本面再往里一层核 agent 通道入站消息的**负载
# 字段**。权威是 `WSMessage` Pydantic 模型 (schemas.py 自述: "every field a handler
# consumes must be declared here to stay in sync"), 两个硬方向:
#   1. 后端 handler 读 `msg.<X>` 而 `WSMessage` 未声明 `X` ⇒ Pydantic BaseModel
#      AttributeError (该入站消息必崩, 回 error 帧).
#   2. 前端发该 type 时带了 `WSMessage` 未声明的字段 ⇒ 被 Pydantic 静默丢弃 (客户端
#      以为发了, 后端读到默认值, 功能静默失效).
# 反向 (模型声明了 handler 未读 / 前端未发) 不是违例 —— 模型面向全部 WS 客户端 (含
# 外部客户端), 冗余声明是诚实的. 只核 agent 通道: terminal/hpc/viewer3d 的入站负载
# 走原始 dict, 不受 `WSMessage` 约束.

_WS_PAYLOAD_SCHEMA_REL = "huginn/routes/schemas.py"
_WS_PAYLOAD_MODEL = "WSMessage"
# `_accept_message_field` 把前端发来的 `message` 归一成 `content`: 有意接纳的别名,
# 不算"未声明字段".
_WS_PAYLOAD_ALIASES = frozenset({"message"})
# 分发键: ws.py 从原始 dict 取 `type` (非 `msg.type`), 不参与"声明却零读取"判定.
_WS_PAYLOAD_ENVELOPE = frozenset({"type"})

_WS_PAYLOAD_KIND_DOC = {
    "handler-undeclared": "后端 handler 读 `msg.<字段>` 而 WSMessage 未声明 (AttributeError 死帧)",
    "fe-undeclared": "前端发送的字段 WSMessage 未声明 (被 Pydantic 静默丢弃)",
}

# 已确认分诊表. 键 (类型, 入站 type, 字段名) → (标签, 理由); 未登记即"待分诊",
# 回归测试会失败 (逼逐条人工判定). 空表 = 当前入站负载字段都合契约.
_WS_PAYLOAD_CONFIRMED: dict[tuple[str, str, str], tuple[str, str]] = {}

_WS_PAYLOAD_TRIAGE_DOC = {
    "defect": "已确认缺陷 (待修)",
    "intentional": "已确认有意",
}


def _ws_payload_triage(kind: str, mtype: str, field: str) -> tuple[str, str] | None:
    """WS 负载违例分诊: 返回 (标签, 理由); 未登记则 None (待人工确认)."""
    return _WS_PAYLOAD_CONFIRMED.get((kind, mtype, field))


def _ws_payload_violation_mark(kind: str, mtype: str, field: str) -> str:
    tri = _ws_payload_triage(kind, mtype, field)
    if tri is None:
        return " — ⚠ 待分诊"
    doc = _WS_PAYLOAD_TRIAGE_DOC[tri[0]]
    return f" — {'⛔' if tri[0] == 'defect' else '✅'} {doc}: {tri[1]}"


def _ws_payload_model_fields(root: Path) -> set[str]:
    """`WSMessage` 声明的字段集 (本面的权威面)."""
    tree = _parse(root / _WS_PAYLOAD_SCHEMA_REL)
    if tree is None:
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == _WS_PAYLOAD_MODEL:
            return {
                s.target.id
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            }
    return set()


def _ws_handler_registry(root: Path) -> dict[str, str]:
    """agent 通道 `_MESSAGE_HANDLERS`: 入站 type → handler 函数名."""
    tree = _parse(root / _WS_REGISTRY_REL)
    if tree is None:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign):
            names = [t for t in node.targets if isinstance(t, ast.Name)]
            target, value = (names[0] if names else None), node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == _WS_REGISTRY_VAR):
            continue
        if not isinstance(value, ast.Dict):
            continue
        out: dict[str, str] = {}
        for k, v in zip(value.keys, value.values):
            if (
                isinstance(k, ast.Constant)
                and isinstance(k.value, str)
                and isinstance(v, ast.Name)
            ):
                out[k.value] = v.id
        return out
    return {}


def _ws_payload_handlers(root: Path) -> dict[str, dict]:
    """入站 type → handler 读取的 `msg.<字段>` 面 (含定义位置, 供核对).

    handler 定义按名在全仓解析 (排除 tests/); 取 `msg` 形参上的属性读取。嵌套闭包
    里的 `msg.<X>` 也算 handler 的消费 (确属该入站路径), 故用 `ast.walk` 下潜。
    """
    reg = _ws_handler_registry(root)
    defs: dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = {}
    for py in _iter_py(root):
        if "/tests/" in py.as_posix():
            continue
        tree = _parse(py)
        if tree is None:
            continue
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name not in defs
            ):
                defs[node.name] = (node, rel)
    out: dict[str, dict] = {}
    for mtype, hname in reg.items():
        hit = defs.get(hname)
        if hit is None:
            out[mtype] = {
                "handler": hname,
                "fields": {},
                "resolved": False,
                "rel": "",
                "line": 0,
            }
            continue
        node, rel = hit
        params = [a.arg for a in node.args.args]
        msgp = "msg" if "msg" in params else (params[1] if len(params) > 1 else "msg")
        fields: dict[str, int] = {}
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id == msgp
            ):
                fields.setdefault(n.attr, n.lineno)
        out[mtype] = {
            "handler": hname,
            "fields": fields,
            "resolved": True,
            "rel": rel,
            "line": node.lineno,
        }
    return out


def _ws_ts_brace_end(text: str, i: int) -> int:
    """`text[i] == '{'` → 匹配闭括号后的下标 (尊重字符串/嵌套); 未闭合返回 n."""
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            i = _skip_ts_string(text, i)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _ws_top_split(inner: str) -> list[str]:
    """按顶层 `,` 切分对象字面量内容 (尊重字符串与 `{}`/`()`/`[]` 嵌套)."""
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(inner)
    while i < n:
        c = inner[i]
        if c in "\"'`":
            i = _skip_ts_string(inner, i)
            continue
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
        i += 1
    parts.append(inner[start:])
    return parts


_WS_OBJ_KEY_RE = re.compile(r'^["\']?([A-Za-z_$][\w$]*)["\']?\s*:')
_WS_OBJ_TYPE_RE = re.compile(r'^["\']?type["\']?\s*:\s*["\']([a-z][a-z0-9_]*)["\']')
_WS_OBJ_SHORTHAND_RE = re.compile(r'^([A-Za-z_$][\w$]*)$')


def _ws_obj_fields(inner: str) -> tuple[set[str], str | None, bool]:
    """对象字面量内容 → (顶层键集, `type` 字面量, 是否有展开/读不出的成员)."""
    keys: set[str] = set()
    typ: str | None = None
    unknown = False
    for part in _ws_top_split(inner):
        s = part.strip()
        if not s:
            continue
        if s.startswith("..."):
            unknown = True
            continue
        m = _WS_OBJ_KEY_RE.match(s)
        if m:
            key = m.group(1)
            keys.add(key)
            if key == "type":
                vm = _WS_OBJ_TYPE_RE.match(s)
                if vm:
                    typ = vm.group(1)
            continue
        sh = _WS_OBJ_SHORTHAND_RE.match(s)
        if sh:
            keys.add(sh.group(1))
        else:
            unknown = True
    return keys, typ, unknown


_WS_SEND_OBJ_RE = re.compile(r"(?:JSON\.stringify|\.send)\s*\(\s*\{")


def _ws_payload_scan_sends(frontend: Path) -> list[dict]:
    """扫前端 WS 发送对象字面量 → (通道, 入站 type, 键集, 是否有展开).

    只读字面量形状: `{ type: "user_input", content, thread_id }`; 对象含 `...` 展开
    或成员读不出时记 unknown (键集仍是下界). type 非字面量则记 None (不可归属).
    """
    sends: list[dict] = []
    if not frontend.is_dir():
        return sends
    for p in sorted(frontend.rglob("*")):
        if not p.is_file() or p.suffix not in (".ts", ".tsx"):
            continue
        rel = _display(frontend, p)
        if (
            rel.endswith(".spec.ts")
            or rel.endswith(".spec.tsx")
            or rel.endswith(_WS_DECL_SUFFIX)
        ):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ch = _ws_file_channel(rel, text)
        for m in _WS_SEND_OBJ_RE.finditer(text):
            brace = text.find("{", m.start())
            if brace < 0:
                continue
            end = _ws_ts_brace_end(text, brace)
            if end <= brace + 1 or text[end - 1] != "}":
                continue
            keys, typ, unknown = _ws_obj_fields(text[brace + 1 : end - 1])
            sends.append(
                {
                    "rel": rel,
                    "line": text.count("\n", 0, m.start()) + 1,
                    "channel": ch,
                    "type": typ,
                    "keys": sorted(keys),
                    "unknown": unknown,
                }
            )
    return sends


def build_ws_payload_contract(
    root: Path | None = None, frontend: Path | None = None
) -> dict:
    """WS 请求负载面: `WSMessage` 声明字段 ↔ 后端 handler 读取面 ↔ 前端发送面.

    硬方向两向: handler 读未声明字段 (AttributeError) / 前端发未声明字段 (静默丢弃).
    只核 agent 通道 (受 `WSMessage` 约束); 前端 type 非字面量或读不出键即跳过该处。
    """
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    model_fields = _ws_payload_model_fields(root)
    handlers = _ws_payload_handlers(root)
    inbound = set(_ws_inbound_types(root).get("agent", []))
    sends = _ws_payload_scan_sends(frontend)
    declared = model_fields | _WS_PAYLOAD_ALIASES

    violations: list[dict] = []
    # 方向 1: handler 读 `msg.<X>` 而模型未声明 X.
    resolved = 0
    for mtype in sorted(handlers):
        h = handlers[mtype]
        if not h["resolved"]:
            continue
        resolved += 1
        for f in sorted(h["fields"]):
            if f in declared:
                continue
            violations.append(
                {
                    "kind": "handler-undeclared",
                    "type": mtype,
                    "field": f,
                    "detail": f"handler `{h['handler']}` 读 `msg.{f}`",
                    "rel": h["rel"],
                    "line": h["fields"][f],
                }
            )

    # 方向 2: 前端发已知入站 type 时带模型未声明字段.
    agent_sends = [s for s in sends if s["channel"] == "agent"]
    sends_static = 0
    sends_unknown = 0
    sends_unattributed = 0
    sent_fields: set[str] = set()
    for s in agent_sends:
        if s["type"] not in inbound:
            sends_unattributed += 1
            continue
        sent_fields |= set(s["keys"])
        if s["unknown"]:
            sends_unknown += 1
        else:
            sends_static += 1
        for f in s["keys"]:
            if f in declared or f in _WS_PAYLOAD_ENVELOPE:
                continue
            violations.append(
                {
                    "kind": "fe-undeclared",
                    "type": s["type"],
                    "field": f,
                    "detail": f"前端发送 `{s['type']}` 带了 `{f}`",
                    "rel": s["rel"],
                    "line": s["line"],
                }
            )

    for v in violations:
        tri = _ws_payload_triage(v["kind"], v["type"], v["field"])
        v["triage"] = tri[0] if tri else "untriaged"
        v["triage_reason"] = tri[1] if tri else ""

    read_fields = {f for h in handlers.values() if h["resolved"] for f in h["fields"]}
    dead_fields = sorted(
        f
        for f in model_fields - _WS_PAYLOAD_ENVELOPE
        if f not in read_fields and f not in sent_fields
    )
    unresolved = sorted(
        m for m, h in handlers.items() if not h["resolved"]
    )
    kinds = Counter(v["kind"] for v in violations)
    return {
        "frontend": str(frontend),
        "model_fields": sorted(model_fields),
        "handler_count": len(handlers),
        "handlers_resolved": resolved,
        "handlers_unresolved": unresolved,
        "handler_fields": {m: sorted(h["fields"]) for m, h in handlers.items()},
        "send_count": len(sends),
        "agent_send_count": len(agent_sends),
        "sends_static": sends_static,
        "sends_unknown": sends_unknown,
        "sends_unattributed": sends_unattributed,
        "violations": violations,
        "untriaged": [v for v in violations if v["triage"] == "untriaged"],
        "kind_counts": dict(sorted(kinds.items())),
        "dead_fields": dead_fields,
    }


def render_ws_payload_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append("## WS 请求负载面: WSMessage 声明字段 vs 后端 handler 读取面 vs 前端发送面")
    lines.append("")
    lines.append(
        "WS 消费面只核「入站 `type` 认不认」（认了但字段发错照样坏）; 本面再往里一层, "
        "核 agent 通道入站消息的**负载字段**。权威是 `WSMessage` Pydantic 模型 "
        "(`schemas.py` 自述: handler 消费的每个字段都必须在此声明才不失同步)。两个硬"
        "方向: **后端 handler 读 `msg.<X>` 而 `WSMessage` 未声明 `X`** ⇒ Pydantic "
        "`BaseModel` 抛 `AttributeError`（该入站消息必崩, 回 error 帧）; **前端发该 "
        "`type` 时带了 `WSMessage` 未声明的字段** ⇒ 被 Pydantic 静默丢弃（客户端以为"
        "发了, 后端读默认值, 功能静默失效）。反向（模型声明了 handler 未读 / 前端未发）"
        "不是违例 —— 模型面向全部 WS 客户端（含外部客户端）."
    )
    lines.append("")
    lines.append(
        "违例类型: " + "; ".join(f"`{k}`={v}" for k, v in _WS_PAYLOAD_KIND_DOC.items())
    )
    lines.append("")
    mf = contract["model_fields"]
    lines.append(
        f"权威面: `{_WS_PAYLOAD_MODEL}` 声明 **{len(mf)}** 字段; agent 入站 type "
        f"**{contract['handler_count']}** 个 (解析到 handler 定义 {contract['handlers_resolved']} 个); "
        f"前端 agent 发送点 **{contract['agent_send_count']}** 处 (type 已知 {contract['sends_static'] + contract['sends_unknown']}, "
        f"其中含展开/读不出 {contract['sends_unknown']}; type 非字面量/不在分发面 {contract['sends_unattributed']})."
    )
    counts = (
        "  " + ", ".join(f"`{k}`×{n}" for k, n in contract["kind_counts"].items())
        if contract["kind_counts"]
        else ""
    )
    lines.append(f"违例: **{len(contract['violations'])}** 条.{counts}")
    lines.append("")

    lines.append("### 违例 (硬: 后端读未声明字段 / 前端发未声明字段)")
    lines.append("")
    if contract["violations"]:
        lines.append(
            "硬违例 —— handler 读模型未声明字段必 `AttributeError`; 前端发模型未声明字段"
            "必被静默丢弃. 逐条分诊: 未登记的落「待分诊」(回归测试会失败, 逼人工判定):"
        )
        lines.append("")
        for v in sorted(
            contract["violations"],
            key=lambda x: (x["kind"], x["type"], x["field"], x["rel"], x["line"]),
        ):
            lines.append(
                f"- `[{v['kind']}]` `{v['type']}.{v['field']}` @ `{v['rel']}:{v['line']}` "
                f"— {v['detail']}" + _ws_payload_violation_mark(v["kind"], v["type"], v["field"])
            )
    else:
        lines.append("- 无 —— handler 读的字段都在 `WSMessage` 里, 前端发的字段也都认得.")
    lines.append("")

    lines.append("### 声明却无人接 (WSMessage 字段零 handler 读取且零前端发送)")
    lines.append("")
    if contract["dead_fields"]:
        for f in contract["dead_fields"]:
            lines.append(f"- `{f}`")
    else:
        lines.append("- 无.")
    lines.append("")

    lines.append("### 各入站 type 的 handler 读取字段")
    lines.append("")
    lines.append("| 入站 type | handler | 读取字段 | 解析 |")
    lines.append("|---|---|---|---|")
    hf = contract["handler_fields"]
    for mtype in sorted(hf):
        fields = hf[mtype]
        ok = "✅" if fields or mtype not in contract["handlers_unresolved"] else "⚠"
        lines.append(
            f"| `{mtype}` | {', '.join(f'`{f}`' for f in fields) or '(无)'} | "
            f"{len(fields)} | {ok} |"
        )
    lines.append("")

    lines.append("### 静态核对覆盖面 (读不出形状即跳过, 不猜)")
    lines.append("")
    lines.append("| 维度 | 已核对 | 跳过 (读不出) |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 后端 handler 定义解析 | {contract['handlers_resolved']} | "
        f"{len(contract['handlers_unresolved'])} |"
    )
    lines.append(
        f"| 前端发送对象字段 | {contract['sends_static']} | {contract['sends_unknown']} |"
    )
    lines.append("")
    lines.append(
        "诚实边界: 只读**字面量**形状 —— handler 经 `getattr(msg, …)` / 变量间接读取, 或"
        "前端发送对象含 `...` 展开 / `type` 非字面量时该处记 unknown 并跳过, 故违例是"
        "**下界** (可能漏报); 只核 agent 通道 (terminal/hpc/viewer3d 的入站负载走原始 "
        "dict, 不受 `WSMessage` 约束); 别名 `message`(→`content`) 与分发键 `type` 不算"
        "未声明; 反向 (模型声明了但 handler 未读 / 前端未发) 不是违例."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WS 事件负载面: 后端 server→client 帧 payload 顶层键 ↔ 前端 type 分支字段读取
# ---------------------------------------------------------------------------

# 只核 agent 通道: 只有它在前端用 `WSMessage` 标注 (`(data: WSMessage) => …`),
# 其余通道 (terminal/hpc/viewer3d) 前端按裸字段取值、无类型标注, 帧 wise 不可靠归因.
_WS_EV_PAYLOAD_CHANNELS = frozenset({"agent"})
# 信封键: `type` 是判别键 (帧名, 已由 WS 消费面核); `_ws_send` 统一为每帧注入
# `thread_id` 供前端按线程路由 (见 ws_helpers.py), 故二者不属各帧 payload 形状.
_WS_EV_PAYLOAD_ENVELOPE = frozenset({"type", "thread_id"})

_WS_EV_PAYLOAD_KIND_DOC = {
    "read-undeclared": (
        "前端在某 type 分支读 `data.<字段>` 而后端该帧 payload 从不发此顶层键 (恒 undefined)"
    )
}

_WS_EV_PAYLOAD_TRIAGE_DOC = {
    "defect": "已确认缺陷 (待修)",
    "intentional": "已确认有意",
}

# 已确认分诊表. 键 (通道, 帧名, 字段) → (标签, 理由); 未登记即"待分诊",
# 回归测试会失败 (逼逐条人工判定). 空表 = 当前前端读的顶层字段都合契约.
_WS_EV_PAYLOAD_CONFIRMED: dict[tuple[str, str, str], tuple[str, str]] = {}


def _ws_ev_payload_triage(channel: str, frame: str, field: str) -> tuple[str, str] | None:
    """WS 事件负载违例分诊: 返回 (标签, 理由); 未登记则 None (待人工确认)."""
    return _WS_EV_PAYLOAD_CONFIRMED.get((channel, frame, field))


def _ws_ev_payload_violation_mark(channel: str, frame: str, field: str) -> str:
    tri = _ws_ev_payload_triage(channel, frame, field)
    if tri is None:
        return " — ⚠ 待分诊"
    doc = _WS_EV_PAYLOAD_TRIAGE_DOC[tri[0]]
    return f" — {'⛔' if tri[0] == 'defect' else '✅'} {doc}: {tri[1]}"


def _ws_ev_payload_shapes(root: Path) -> dict[str, dict[str, dict]]:
    """各通道 server→client 帧 payload 顶层键形状: {帧名: {keys, closed}}.

    权威是后端 WS 路由模块里 `{"type": "…", …}` 字典字面量的**其余顶层键**;
    同帧名在多个字面量出现时取并集 (安全下界). 帧名仅经变量透传 (无字面量)
    或字面量含 `**` 展开 / 非常量键 → 形状开放 (closed=False).
    """
    produced = _ws_server_frames(root)
    out: dict[str, dict[str, dict]] = {}
    for ch in _WS_EV_PAYLOAD_CHANNELS:
        present: set[str] = set()
        closed_ok: dict[str, bool] = {}
        keys: dict[str, set[str]] = defaultdict(set)
        for rel in _WS_BACKEND_MODULES.get(ch, ()):
            tree = _parse(root / rel)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                lit = _ws_dict_type_literal(node)
                if lit is None:
                    continue
                present.add(lit)
                top = _dict_literal_keys(node)
                if top is None:
                    closed_ok[lit] = False
                    continue
                closed_ok.setdefault(lit, True)
                keys[lit] |= top
        frames: dict[str, dict] = {}
        for f in produced.get(ch, {}).get("frames", []):
            if f in present and closed_ok.get(f, False):
                frames[f] = {"keys": keys[f] - _WS_EV_PAYLOAD_ENVELOPE, "closed": True}
            else:
                frames[f] = {"keys": set(), "closed": False}
        out[ch] = frames
    return out


_FE_AS_ANY_RE = re.compile(r"\(\s*(\w+)\s+as\s+any\s*\)")
_FE_TS_VAR_ALIAS_RE = re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(\w+)\s*;")


def _skip_balanced_paren(text: str, i: int) -> int:
    """`text[i] == '('` → 匹配闭括号后的下标 (尊重字符串/嵌套); 未闭合返回 n."""
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            i = _skip_ts_string(text, i)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _fe_ws_case_segments(text: str, ws_vars: set[str]) -> list[tuple[str, str, int]]:
    """`switch (<wsVar>.type) { case "lit": … }` 每段 → (帧名, body, body 起始行)."""
    out: list[tuple[str, str, int]] = []
    for m in _FE_SWITCH_TYPE_HEAD_RE.finditer(text):
        vm = re.fullmatch(r"\s*(\w+)\.type\s*", m.group(1))
        if vm is None or vm.group(1) not in ws_vars:
            continue
        start = m.end() - 1  # 指向 `{`
        end = _ws_ts_brace_end(text, start)
        inner = text[start + 1 : end - 1]
        marks = [
            (cm.group(1), cm.start(), cm.end()) for cm in _FE_WS_CASE_RE.finditer(inner)
        ]
        for idx, (frame, _cs, ce) in enumerate(marks):
            seg_end = marks[idx + 1][1] if idx + 1 < len(marks) else len(inner)
            out.append((frame, inner[ce:seg_end], _ws_line_of(text, start + 1 + ce)))
    return out


def _fe_ws_if_segments(text: str, ws_vars: set[str]) -> list[tuple[str, str, int]]:
    """`if (<wsVar>.type === "a" [|| …]) { … }` 块 → 每帧 (帧名, body, body 起始行).

    仅接受**纯 type 比较的析取式** (如 `data.type === "done" || data.type === "error"`);
    含其它子表达式的条件 (线程路由等) 不归属任何帧.
    """
    out: list[tuple[str, str, int]] = []
    for m in re.finditer(r"\bif\s*\(", text):
        i = m.end() - 1
        j = _skip_balanced_paren(text, i)
        cond = text[i + 1 : j - 1]
        frames: list[str] = []
        ok = bool(cond.strip())
        for part in cond.split("||"):
            pm = re.fullmatch(
                r'\s*(\w+)\.type\s*===?\s*["\']([a-z][a-z0-9_]*)["\']\s*', part
            )
            if pm is None or pm.group(1) not in ws_vars:
                ok = False
                break
            frames.append(pm.group(2))
        if not ok or not frames:
            continue
        k = j
        n = len(text)
        while k < n and text[k] in " \t\n":
            k += 1
        if k >= n or text[k] != "{":
            continue
        end = _ws_ts_brace_end(text, k)
        body = text[k + 1 : end - 1]
        line = _ws_line_of(text, k + 1)
        for fr in frames:
            out.append((fr, body, line))
    return out


def _ws_ev_payload_reads(frontend: Path) -> list[dict]:
    """扫前端 `.ts`/`.tsx`: WS `type` 分支内的 `data.<字段>` 顶层读取 (带通道归属)."""
    reads: list[dict] = []
    for _p, rel, text in _fe_ts_files(
        frontend, skip_suffixes=(".spec.ts", _WS_DECL_SUFFIX)
    ):
        ws_vars = set(_FE_WS_MSG_VAR_RE.findall(text))
        if not ws_vars:
            continue
        ch = _ws_file_channel(rel, text)
        # `const tp = data;` 之类的同值别名也计入可读变量 (task_progress 分支用此式).
        readable = set(ws_vars)
        for am in _FE_TS_VAR_ALIAS_RE.finditer(text):
            if am.group(2) in ws_vars:
                readable.add(am.group(1))
        segments = _fe_ws_case_segments(text, ws_vars) + _fe_ws_if_segments(text, ws_vars)
        for frame, body, base_line in segments:
            reads.extend(
                _fe_field_read_rows(
                    rel=rel,
                    channel=ch,
                    frames=[frame],
                    body=_FE_AS_ANY_RE.sub(r"\1", body),
                    base_line=base_line,
                    readable=frozenset(readable),
                    skip_fields=_WS_EV_PAYLOAD_ENVELOPE,
                )
            )
    return reads


def build_ws_ev_payload_contract(
    root: Path | None = None, frontend: Path | None = None
) -> dict:
    """WS 事件负载面: 后端 server→client 帧 payload 顶层键 ↔ 前端 type 分支字段读取.

    硬方向: 前端在某帧的 type 分支里读 `data.<字段>` 而后端该帧 payload 从不发此
    顶层键 (恒 undefined, 静默坏). 反向 (后端发了前端没读) 不是违例, 只列候选.
    只核**顶层**键; 嵌套 `data.<对象>.<字段>` 的子形状由发布点决定, 跳过. 只核
    agent 通道 (前端唯一用 `WSMessage` 标注的通道).
    """
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    shapes = _ws_ev_payload_shapes(root)
    reads = _ws_ev_payload_reads(frontend)
    violations: list[dict] = []
    checked = skip_frame = skip_shape = 0
    read_fields: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in reads:
        ch, frame, field = r["channel"], r["frame"], r["field"]
        frames = shapes.get(ch, {}) if ch else {}
        if frame not in frames:
            # 帧名不属该通道 (或通道未归属) → payload 面无权威 (帧名本身由消费面核).
            skip_frame += 1
            continue
        info = frames[frame]
        read_fields[(ch, frame)].add(field)
        if not info["closed"]:
            skip_shape += 1
            continue
        checked += 1
        if field in info["keys"]:
            continue
        violations.append(
            {
                "kind": "read-undeclared",
                "channel": ch,
                "frame": frame,
                "field": field,
                "detail": f"前端在 `{frame}` 分支里读 `data.{field}`",
                "rel": r["rel"],
                "line": r["line"],
            }
        )

    for v in violations:
        tri = _ws_ev_payload_triage(v["channel"], v["frame"], v["field"])
        v["triage"] = tri[0] if tri else "untriaged"
        v["triage_reason"] = tri[1] if tri else ""

    zero_read: list[dict] = []
    for ch, frames in shapes.items():
        closed_keys: set[str] = set()
        read_keys: set[str] = set()
        for frame, info in frames.items():
            if info["closed"]:
                closed_keys |= info["keys"]
            read_keys |= read_fields.get((ch, frame), set())
        for k in sorted(closed_keys - read_keys):
            zero_read.append({"channel": ch, "field": k})

    kinds = Counter(v["kind"] for v in violations)
    return {
        "frontend": str(frontend),
        "channels": {
            ch: {
                f: {"keys": sorted(i["keys"]), "closed": i["closed"]}
                for f, i in sorted(frames.items())
            }
            for ch, frames in shapes.items()
        },
        "frame_reads": {
            ch: {frame: sorted(read_fields.get((ch, frame), set())) for frame in frames}
            for ch, frames in shapes.items()
        },
        "read_count": len(reads),
        "coverage": {"checked": checked, "skip_frame": skip_frame, "skip_shape": skip_shape},
        "violations": violations,
        "untriaged": [v for v in violations if v["triage"] == "untriaged"],
        "kind_counts": dict(sorted(kinds.items())),
        "zero_read": zero_read,
    }


def render_ws_ev_payload_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append(
        "## WS 事件负载面: 后端 server→client 帧 payload 顶层键 vs 前端 type 分支字段读取"
    )
    lines.append("")
    lines.append(
        "WS 消费面只核「帧名认不认」, WS 请求负载面只核「入站字段」; 本面再核**出站帧 "
        "payload 的顶层键**。权威面是后端 WS 路由模块里 `{\"type\": \"…\", …}` 字典字面量的"
        "其余顶层键 (同帧名多字面量取并集)。硬方向: **前端在某帧的 type 分支里读 "
        "`data.<字段>` 而后端该帧 payload 从不发此顶层键** ⇒ 恒 `undefined` (静默坏)。反向"
        "(后端发了前端没读) 不是违例, 只列候选。只核**顶层**键: 嵌套 "
        "`data.<对象>.<字段>` 的子形状由发布点决定, 跳过。"
    )
    lines.append("")
    lines.append(
        "违例类型: " + "; ".join(f"`{k}`={v}" for k, v in _WS_EV_PAYLOAD_KIND_DOC.items())
    )
    lines.append("")
    counts = (
        "  " + ", ".join(f"`{k}`×{n}" for k, n in contract["kind_counts"].items())
        if contract["kind_counts"]
        else ""
    )
    lines.append(
        f"前端 WS type 分支顶层字段读取点: **{contract['read_count']}** 处; 违例: "
        f"**{len(contract['violations'])}** 条.{counts}"
    )
    lines.append("")

    lines.append("### 各帧 payload 顶层键 (权威面)")
    lines.append("")
    lines.append("| 通道 | 帧名 | payload 顶层键 | 形状 | 前端读取字段 |")
    lines.append("|---|---|---|---|---|")
    for ch, frames in contract["channels"].items():
        fr = contract["frame_reads"].get(ch, {})
        for frame in sorted(frames):
            info = frames[frame]
            keys = ", ".join(f"`{k}`" for k in info["keys"]) or "—"
            shape = "封闭" if info["closed"] else "开放(读不出)"
            rd = ", ".join(f"`{f}`" for f in fr.get(frame, [])) or "—"
            lines.append(f"| `{ch}` | `{frame}` | {keys} | {shape} | {rd} |")
    lines.append("")

    lines.append("### 违例 (硬: 前端读的顶层字段后端从不发)")
    lines.append("")
    if contract["violations"]:
        lines.append(
            "硬违例 —— 前端读 `data.<字段>` 而后端该帧 payload 无此顶层键 (恒 undefined). "
            "逐条分诊, 未登记的落「待分诊」(回归测试会失败, 逼人工判定):"
        )
        lines.append("")
        for v in sorted(
            contract["violations"],
            key=lambda x: (x["channel"], x["frame"], x["field"], x["rel"], x["line"]),
        ):
            lines.append(
                f"- `[{v['kind']}]` `{v['channel']}/{v['frame']}.{v['field']}` "
                f"@ `{v['rel']}:{v['line']}` — {v['detail']}"
                + _ws_ev_payload_violation_mark(v["channel"], v["frame"], v["field"])
            )
    else:
        lines.append("- 无 —— 前端读的每个顶层字段, 后端该帧都发.")
    lines.append("")

    lines.append("### payload 顶层键零前端读取 (候选, 反向不判违例)")
    lines.append("")
    if contract["zero_read"]:
        for z in contract["zero_read"]:
            lines.append(f"- `{z['channel']}` / `{z['field']}`")
    else:
        lines.append("- 无.")
    lines.append("")

    lines.append("### 静态核对覆盖面 (读不出形状即跳过, 不猜)")
    lines.append("")
    lines.append("| 维度 | 已核对 | 跳过 (帧不属该通道) | 跳过 (形状开放) |")
    lines.append("|---|---|---|---|")
    cov = contract["coverage"]
    lines.append(
        f"| 前端顶层字段读取 | {cov['checked']} | {cov['skip_frame']} | {cov['skip_shape']} |"
    )
    lines.append("")
    lines.append(
        "诚实边界: 只读**字面量**形状 —— 后端帧经变量透传转发 (`_ws_send(dict(state))`) 或"
        "字面量含 `**` 展开 / 动态拼键时该帧记开放并跳过, 故违例是**下界** (可能漏报); 只核"
        "**顶层**键 (嵌套 `data.<对象>.<字段>` 的子形状由发布点决定, 不核); 只核 agent "
        "通道 (前端唯一用 `WSMessage` 标注的通道, 其余通道按裸字段取值、无类型归因); 信封键 "
        "`type`(判别键) 与 `thread_id`(`_ws_send` 统一注入) 不算各帧 payload; 前端经**别名**"
        "中转后的字段读取 (如 `const x = data.obj; x.field`) 只归因到直接读取的变量; 反向 "
        "(后端发前端没读) 不是违例."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 第十七面: HTTP 请求字段面 — 端点请求体字段 声明/读取/发送 三面一致
# ---------------------------------------------------------------------------

# Pydantic BaseModel 的内建属性/方法 (非字段): `body.model_dump()` 类调用不算字段读取.
_HTTP_FIELD_PYDANTIC_ATTRS = frozenset(
    {
        "model_dump",
        "model_dump_json",
        "model_validate",
        "model_validate_json",
        "model_copy",
        "model_fields",
        "model_config",
        "model_extra",
        "model_fields_set",
        "model_construct",
        "model_post_init",
        "dict",
        "json",
        "copy",
        "parse_obj",
        "parse_raw",
        "schema",
        "schema_json",
        "construct",
        "update_forward_refs",
        "validate",
    }
)

_HTTP_FIELD_KIND_DOC = {
    "handler-undeclared": (
        "body-model 端点 handler 读 `body.<字段>` 而模型未声明 ⇒ AttributeError (死端点)"
    ),
    "fe-undeclared": (
        "body-model 端点前端发的 body 键模型未声明 ⇒ Pydantic 静默丢弃 (前端以为传了)"
    ),
    "dict-key-unsent": (
        'body-dict 端点 handler 下标读 `body["键"]` 而该端点前端调用从不发此键 ⇒ KeyError'
    ),
}

_HTTP_FIELD_TRIAGE_DOC = {
    "defect": "已确认缺陷 (待修)",
    "intentional": "已确认有意",
}

# 已确认分诊表. 键 (类型, 方法, 端点, 字段) → (标签, 理由); 未登记即"待分诊",
# 回归测试会失败 (逼逐条人工判定). 空表 = 当前请求体字段都合契约.
_HTTP_FIELD_CONFIRMED: dict[tuple[str, str, str, str], tuple[str, str]] = {}


def _http_field_triage(
    kind: str, method: str, endpoint: str, field: str
) -> tuple[str, str] | None:
    """请求字段违例分诊: 返回 (标签, 理由); 未登记则 None (待人工确认)."""
    return _HTTP_FIELD_CONFIRMED.get((kind, method.upper(), endpoint, field))


def _http_field_violation_mark(kind: str, method: str, endpoint: str, field: str) -> str:
    tri = _http_field_triage(kind, method, endpoint, field)
    if tri is None:
        return " — ⚠ 待分诊"
    doc = _HTTP_FIELD_TRIAGE_DOC[tri[0]]
    return f" — {'⛔' if tri[0] == 'defect' else '✅'} {doc}: {tri[1]}"


_HTTP_FIELD_EXTRA_ALLOW_RE = re.compile(r"extra\s*=\s*[\"']allow[\"']")


def _http_field_own_fields(cls: ast.ClassDef) -> set[str]:
    """类自身声明的注解字段 (不含继承)."""
    return {
        s.target.id
        for s in cls.body
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
    }


def _http_field_models(root: Path) -> tuple[dict[str, set[str]], set[str]]:
    """全仓 Pydantic 模型名 → **全部**声明字段 (含继承链 fixpoint); 及 extra=allow 模型.

    与请求负载面的 `_payload_models` (只取**必填**字段) 不同: 本面核「读/发未声明」,
    需要模型的完整字段面, 故单列。`extra=allow` 的模型收到未知键不会丢, 其发送面开放。
    """
    classes: dict[str, ast.ClassDef] = {}
    for py in _iter_py(root):
        if "/tests/" in py.as_posix():
            continue
        tree = _parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, node)
    models: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for name, cls in classes.items():
            if name in models:
                continue
            bases = {ast.unparse(b).split("[")[0].strip() for b in cls.bases}
            if not (bases & _PAYLOAD_MODEL_BASES) and not (bases & set(models)):
                continue
            fields = _http_field_own_fields(cls)
            for b in bases:
                if b in models:
                    fields |= models[b]
            models[name] = fields
            changed = True
    extra_allow = {
        name
        for name, cls in classes.items()
        if _HTTP_FIELD_EXTRA_ALLOW_RE.search(ast.unparse(cls))
    }
    return models, extra_allow


def _http_field_body_reads(node: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> tuple[set[str], set[str]]:
    """handler 自身子树里对 body 变量的读取: (属性读取集, 字面下标读取集).

    只收 `body.<f>` 与 `body["f"]` 两种直读; `body.get("f")` 归 `.get` (下标面不含,
    因其缺省返回 None 是**有意**的可选语义); 嵌套 def/lambda 内不计 (`_resp_own_walk`)。
    """
    attrs: set[str] = set()
    keys: set[str] = set()
    for n in _resp_own_walk(node):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == param
        ):
            attrs.add(n.attr)
        elif (
            isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == param
            and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)
        ):
            keys.add(n.slice.value)
    return attrs, keys


def _http_field_endpoints(root: Path, models: dict[str, set[str]]) -> list[dict]:
    """已挂载路由里请求体为 Pydantic 模型 / 裸 dict 的端点 + handler 对 body 的读取面."""
    out: list[dict] = []
    rdir = root / _HTTP_ROUTES_DIR
    if not rdir.is_dir():
        return out
    for py in sorted(rdir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        tree = _parse(py)
        if tree is None:
            continue
        prefixes = _http_router_prefixes(tree)
        if not prefixes:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for dec in node.decorator_list:
                parsed = _http_route_decorator(dec, prefixes)
                if parsed is None:
                    continue
                methods, path, _var = parsed
                if "WEBSOCKET" in methods:
                    continue
                for p in _payload_handler_params(node, path):
                    kind = _payload_classify(p, models)
                    if kind not in ("body-model", "body-dict"):
                        continue
                    attrs, keys = _http_field_body_reads(node, p["name"])
                    out.append(
                        {
                            "rel": _display(root, py),
                            "line": node.lineno,
                            "methods": methods,
                            "path": path,
                            "param": p["name"],
                            "kind": kind,
                            "model": _payload_base_ann(p["ann"]) if kind == "body-model" else "",
                            "attrs": sorted(attrs),
                            "keys": sorted(keys),
                        }
                    )
                    break
                break
    return out


def _http_field_sends(frontend: Path, live: list[dict]) -> dict[tuple[str, str], dict]:
    """端点 → 前端调用的 body 形状: {keys 并集, sites 逐调用点, unknown}.

    只归因**唯一命中**端点的调用: 动态段并列命中多个端点即跳过 (不猜); 调用实参数是
    变量 / 含 `...` 展开时该调用点记 unknown (键集只是下界, 不据此判违例).
    """
    out: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"keys": set(), "sites": [], "unknown": False}
    )
    for c in _payload_scan_frontend(frontend):
        if not c["has_body"]:
            continue
        segs = _http_fe_segments(c["path"])
        if segs is None:
            continue
        eps = [
            ep
            for ep in live
            if c["method"] in ep["methods"] and _http_path_match(segs, ep["segments"])
        ]
        if len(eps) != 1:
            continue
        rec = out[(c["method"], eps[0]["path"])]
        rec["keys"] |= set(c["body_keys"])
        rec["unknown"] = rec["unknown"] or c["body_unknown"]
        rec["sites"].append(
            {
                "rel": c["rel"],
                "line": c["line"],
                "keys": c["body_keys"],
                "unknown": c["body_unknown"],
            }
        )
    return out


def build_http_field_contract(
    root: Path | None = None, frontend: Path | None = None
) -> dict:
    """HTTP 请求字段面: 端点请求体字段 声明(模型) / 读取(handler) / 发送(前端) 三面一致.

    权威面三处: 请求体 Pydantic 模型的**声明字段**; handler 内 `body.<字段>` /
    `body["字段"]` 读取; 前端 `api.*` 调用实参的 body 键. 硬方向:
      - `handler-undeclared`: body-model 端点 handler 读模型未声明字段 ⇒ AttributeError;
      - `fe-undeclared`: body-model 端点前端发的键模型未声明 ⇒ Pydantic 静默丢弃;
      - `dict-key-unsent`: body-dict 端点 handler 下标读键而前端调用从不发 ⇒ KeyError.
    反向 (handler 读了前端从不发 / 模型声明却无人接管) 不是违例, 只列候选。请求负载面
    只核「前端漏发后端必填」(422), 本面核字段级声明一致性, 与之互补。
    """
    root = root or _REPO
    frontend = frontend if frontend is not None else root.parent / "desktop" / "src"
    models, extra_allow = _http_field_models(root)
    endpoints = _http_field_endpoints(root, models)
    live = _payload_live_endpoints(root)
    sends = _http_field_sends(frontend, live)

    violations: list[dict] = []
    cand_read_unsent: list[dict] = []
    cand_unread_unsent: list[dict] = []
    cov = {
        "handler_checked": 0,
        "handler_skipped": 0,
        "fe_checked": 0,
        "fe_skipped": 0,
        "dict_checked": 0,
        "dict_skipped": 0,
    }
    rows: list[dict] = []

    for e in endpoints:
        key = None
        for m in e["methods"]:
            if (m, e["path"]) in sends:
                key = (m, e["path"])
                break
        rec = sends.get(key) if key is not None else None
        method = key[0] if key is not None else (e["methods"][0] if e["methods"] else "")
        attrs = set(e["attrs"]) - _HTTP_FIELD_PYDANTIC_ATTRS
        row = {
            "rel": e["rel"],
            "line": e["line"],
            "method": method,
            "endpoint": e["path"],
            "kind": e["kind"],
            "model": e["model"],
            "declared": [],
            "read": [],
            "sent": [],
            "shape": "封闭",
        }
        if e["kind"] == "body-model":
            declared = models.get(e["model"])
            if declared is None:
                cov["handler_skipped"] += 1
                row["shape"] = "开放(模型解析不到)"
            else:
                open_model = e["model"] in extra_allow
                row["declared"] = sorted(declared)
                row["read"] = sorted(attrs & declared)
                if open_model:
                    row["shape"] = "开放(extra=allow)"
                # (1) handler 读模型未声明字段 (读面与 extra 配置无关, 照核).
                cov["handler_checked"] += 1
                for f in sorted(attrs - declared):
                    violations.append(
                        {
                            "kind": "handler-undeclared",
                            "method": method,
                            "endpoint": e["path"],
                            "field": f,
                            "model": e["model"],
                            "rel": e["rel"],
                            "line": e["line"],
                            "detail": f"handler 读 `{e['param']}.{f}` 而模型 "
                            f"`{e['model']}` 未声明",
                        }
                    )
                # (2) 前端发模型未声明键 + 反向候选.
                if rec is not None:
                    row["sent"] = sorted(rec["keys"])
                    if open_model:
                        cov["fe_skipped"] += len(rec["sites"])
                    else:
                        for site in rec["sites"]:
                            if site["unknown"]:
                                cov["fe_skipped"] += 1
                                continue
                            cov["fe_checked"] += 1
                            for f in sorted(set(site["keys"]) - declared):
                                violations.append(
                                    {
                                        "kind": "fe-undeclared",
                                        "method": method,
                                        "endpoint": e["path"],
                                        "field": f,
                                        "model": e["model"],
                                        "rel": site["rel"],
                                        "line": site["line"],
                                        "detail": f"前端发 body 键 `{f}` 而模型 "
                                        f"`{e['model']}` 未声明 (静默丢弃)",
                                    }
                                )
                    if not open_model and not rec["unknown"]:
                        sent = rec["keys"]
                        for f in sorted((attrs & declared) - sent):
                            cand_read_unsent.append(
                                {
                                    "method": method,
                                    "endpoint": e["path"],
                                    "field": f,
                                    "model": e["model"],
                                    "rel": e["rel"],
                                    "line": e["line"],
                                }
                            )
                        for f in sorted(declared - attrs - sent):
                            cand_unread_unsent.append(
                                {
                                    "method": method,
                                    "endpoint": e["path"],
                                    "field": f,
                                    "model": e["model"],
                                    "rel": e["rel"],
                                    "line": e["line"],
                                }
                            )
        else:  # body-dict: 无声明面, 只核「下标读键 vs 前端发送键」
            row["read"] = sorted(e["keys"])
            if rec is None or rec["unknown"]:
                cov["dict_skipped"] += 1
                row["shape"] = "开放(无前端调用)" if rec is None else "开放(调用含展开)"
            else:
                cov["dict_checked"] += 1
                sent = rec["keys"]
                row["sent"] = sorted(sent)
                for k in sorted(set(e["keys"]) - sent):
                    violations.append(
                        {
                            "kind": "dict-key-unsent",
                            "method": method,
                            "endpoint": e["path"],
                            "field": k,
                            "model": "",
                            "rel": e["rel"],
                            "line": e["line"],
                            "detail": f'handler 下标读 `{e["param"]}["{k}"]` 而该端点'
                            "前端调用从不发此键",
                        }
                    )
        rows.append(row)

    for v in violations:
        tri = _http_field_triage(v["kind"], v["method"], v["endpoint"], v["field"])
        v["triage"] = tri[0] if tri else "untriaged"
        v["triage_reason"] = tri[1] if tri else ""

    kinds = Counter(v["kind"] for v in violations)
    return {
        "frontend": str(frontend),
        "endpoint_count": len(endpoints),
        "model_count": len(models),
        "rows": rows,
        "violations": violations,
        "untriaged": [v for v in violations if v["triage"] == "untriaged"],
        "kind_counts": dict(sorted(kinds.items())),
        "coverage": cov,
        "candidates_read_unsent": cand_read_unsent,
        "candidates_unread_unsent": cand_unread_unsent,
    }


def render_http_field_markdown(contract: dict) -> str:
    lines: list[str] = []
    lines.append(
        "## HTTP 请求字段面: 请求体字段 声明(模型) / 读取(handler) / 发送(前端) 三面一致"
    )
    lines.append("")
    lines.append(
        "请求负载面只核「前端漏发后端必填」(422); 本面再往里一层核**字段级声明一致性**: "
        "请求体 Pydantic 模型的声明字段 = 权威面, handler 内 `body.<字段>` / `body[\"字段\"]` "
        "读取 = 读取面, 前端 `api.*` 调用实参的 body 键 = 发送面。硬方向: handler 读模型"
        "未声明字段 (AttributeError) / 前端发模型未声明键 (Pydantic 静默丢弃) / body-dict "
        "端点 handler 下标读键而前端调用从不发 (KeyError)。反向不判违例, 只列候选。"
    )
    lines.append("")
    lines.append(
        "违例类型: " + "; ".join(f"`{k}`={v}" for k, v in _HTTP_FIELD_KIND_DOC.items())
    )
    lines.append("")
    counts = (
        "  " + ", ".join(f"`{k}`×{n}" for k, n in contract["kind_counts"].items())
        if contract["kind_counts"]
        else ""
    )
    lines.append(
        f"请求体端点: **{contract['endpoint_count']}** 处; 违例: "
        f"**{len(contract['violations'])}** 条.{counts}"
    )
    lines.append("")

    model_rows = [r for r in contract["rows"] if r["kind"] == "body-model"]
    lines.append("### body-model 端点: 声明字段 vs handler 读取 vs 前端发送")
    lines.append("")
    lines.append("| 端点 | 方法 | 模型 | 声明字段 | handler 读取 | 前端发送键 | 形状 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(model_rows, key=lambda x: (x["endpoint"], x["method"])):
        decl = ", ".join(f"`{f}`" for f in r["declared"]) or "—"
        read = ", ".join(f"`{f}`" for f in r["read"]) or "—"
        sent = ", ".join(f"`{f}`" for f in r["sent"]) or "—"
        lines.append(
            f"| `{r['endpoint']}` | `{r['method']}` | `{r['model']}` | {decl} | "
            f"{read} | {sent} | {r['shape']} |"
        )
    lines.append("")

    dict_rows = [r for r in contract["rows"] if r["kind"] == "body-dict"]
    lines.append("### body-dict 端点: handler 下标读键 vs 前端发送键 (仅列可核对的)")
    lines.append("")
    if dict_rows:
        lines.append("| 端点 | 方法 | handler 下标读键 | 前端发送键 | 形状 |")
        lines.append("|---|---|---|---|---|")
        for r in sorted(dict_rows, key=lambda x: (x["endpoint"], x["method"])):
            read = ", ".join(f"`{f}`" for f in r["read"]) or "—"
            sent = ", ".join(f"`{f}`" for f in r["sent"]) or "—"
            lines.append(
                f"| `{r['endpoint']}` | `{r['method']}` | {read} | {sent} | {r['shape']} |"
            )
    else:
        lines.append("- 无.")
    lines.append("")

    lines.append("### 违例 (硬)")
    lines.append("")
    if contract["violations"]:
        lines.append(
            "硬违例 —— 逐条分诊, 未登记的落「待分诊」(回归测试会失败, 逼人工判定):"
        )
        lines.append("")
        for v in sorted(
            contract["violations"],
            key=lambda x: (x["kind"], x["method"], x["endpoint"], x["field"], x["rel"], x["line"]),
        ):
            lines.append(
                f"- `[{v['kind']}]` `{v['method']} {v['endpoint']}` → `{v['field']}` "
                f"@ `{v['rel']}:{v['line']}` — {v['detail']}"
                + _http_field_violation_mark(
                    v["kind"], v["method"], v["endpoint"], v["field"]
                )
            )
    else:
        lines.append("- 无 —— 请求体字段的声明/读取/发送三面一致.")
    lines.append("")

    lines.append("### 候选: handler 读了而前端调用从不发 (反向不判违例)")
    lines.append("")
    if contract["candidates_read_unsent"]:
        for c in sorted(
            contract["candidates_read_unsent"],
            key=lambda x: (x["method"], x["endpoint"], x["field"]),
        ):
            lines.append(
                f"- `{c['method']} {c['endpoint']}` → 读 `{c['field']}` (模型 `{c['model']}`) "
                f"@ `{c['rel']}:{c['line']}`"
            )
    else:
        lines.append("- 无.")
    lines.append("")

    lines.append("### 候选: 模型声明却既无 handler 读取也零前端发送 (宣称无人接)")
    lines.append("")
    if contract["candidates_unread_unsent"]:
        for c in sorted(
            contract["candidates_unread_unsent"],
            key=lambda x: (x["method"], x["endpoint"], x["field"]),
        ):
            lines.append(
                f"- `{c['method']} {c['endpoint']}` → `{c['field']}` (模型 `{c['model']}`) "
                f"@ `{c['rel']}:{c['line']}`"
            )
    else:
        lines.append("- 无.")
    lines.append("")

    lines.append("### 静态核对覆盖面 (读不出形状即跳过, 不猜)")
    lines.append("")
    lines.append("| 维度 | 已核对 | 跳过 |")
    lines.append("|---|---|---|")
    cov = contract["coverage"]
    lines.append(f"| body-model handler 读取 | {cov['handler_checked']} | {cov['handler_skipped']} |")
    lines.append(f"| body-model 前端发送 | {cov['fe_checked']} | {cov['fe_skipped']} |")
    lines.append(f"| body-dict 下标读键 | {cov['dict_checked']} | {cov['dict_skipped']} |")
    lines.append("")
    lines.append(
        "诚实边界: 请求体类型静态解析不到 (跨模块 / 别名) 的端点跳过 (下界, 可能漏报); "
        "`extra=allow` 的模型发送面开放、跳过; body-dict 端点须有 ≥1 个**唯一命中**且 body "
        "静态可辨的前端调用才核 (外部客户端 / 动态段并列命中 / 调用含展开均跳过); "
        '`body.get("k")` 的缺省 None 是**有意**可选语义, 不计入下标读; handler 经别名或 '
        "`**body` / 迭代读取 (如 `for k in body`) 无法逐字段归因, 跳过 (漏报); 反向"
        "(handler 读了前端从不发 / 模型声明无人接管) 不是违例, 只列候选."
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
        "events": build_event_contract(root),
        "sse": build_sse_contract(root),
        "ws": build_ws_contract(root),
        "http": build_http_contract(root),
        "payload": build_payload_contract(root),
        "response": build_response_contract(root),
        "ws_payload": build_ws_payload_contract(root),
        "sse_payload": build_sse_payload_contract(root),
        "ws_ev_payload": build_ws_ev_payload_contract(root),
        "http_field": build_http_field_contract(root),
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
    ev = snap["events"]
    _event_issue = {
        "subscribed-only": "事件: 类型有 .subscribe 却零生产发布 (订阅永不发生): ",
        "dead": "事件: 类型声明零生产发布零订阅: ",
    }
    for e in ev["events"]:
        prefix = _event_issue.get(e["status"])
        if prefix:
            issues.append(prefix + e["const"])
    for c in ev["collisions"]:
        issues.append(f"事件: 类型常量撞值 {c['value']}: {', '.join(c['consts'])}")
    for name in ev["not_in_all_types"]:
        issues.append(f"事件: 类型常量未登记进 ALL_TYPES: {name}")
    for name in ev["unresolved_members"]:
        issues.append(f"事件: ALL_TYPES 成员无常量定义: {name}")
    for u in ev["undeclared"]:
        used = []
        if u["prod_publish"]:
            used.append("发布")
        if u["prod_subscribe"]:
            used.append("订阅")
        issues.append(
            f"事件: {'+'.join(used)}了未声明类型 (设计允许非穷尽, 候选登记): {u['value']}"
        )
    sse = snap["sse"]
    for s in sse["listeners"]:
        if s["status"] == "channel-mismatch":
            issues.append(
                f"SSE: 监听挂错通道 (该 EventSource 不发此帧名, 永不触发): {s['frame']} "
                f"@ {s['rel']}:{s['line']} — {s['note']}"
            )
        elif s["status"] == "no-source":
            issues.append(
                f"SSE: 监听的后端无此帧名 (三通道都不发): {s['frame']} "
                f"@ {s['rel']}:{s['line']}"
            )
    ws = snap["ws"]
    for s in ws["consumers"]:
        if s["status"] == "no-source":
            issues.append(
                f"WS: 前端 type 判别的后端不发此帧名 (该通道 case 永不命中): {s['frame']} "
                f"@ {s['rel']}:{s['line']} — 通道 {s['channel']}"
            )
    for s in ws["sends"]:
        if s["status"] == "unhandled":
            issues.append(
                f"WS: 前端发送的入站类型后端分发面不认 (回 error 帧): {s['frame']} "
                f"@ {s['rel']}:{s['line']} — 通道 {s['channel']}"
            )
    http = snap["http"]
    _http_issue = {
        "no-source": "HTTP: 前端调用的路径后端未注册 (404 死链): ",
        "method-mismatch": "HTTP: 前端调用的方法与后端注册不符 (405): ",
    }
    for c in http["hard_violations"]:
        if c["triage"] == "untriaged":
            mark = " 待分诊"
        else:
            mark = " " + _HTTP_TRIAGE_DOC[c["triage"]]
        issues.append(
            f"{_http_issue[c['status']]}{c['method']} {c['path']} @ "
            f"{c['rel']}:{c['line']} — {c['note']};{mark}"
        )
    for name in http["registration"]["unmounted"]:
        issues.append(f"HTTP: 定义了 APIRouter 却未挂进 ALL_ROUTERS (端点永不生效): {name}")
    for name in http["registration"]["dangling"]:
        issues.append(f"HTTP: ALL_ROUTERS 引用了未 import 的别名: {name}")
    for d in http["duplicates"]:
        issues.append(
            f"HTTP: 同一 {d['method']} {d['path']} 被多模块注册 (路由遮蔽): "
            f"{', '.join(d['modules'])}"
        )
    payload = snap["payload"]
    for v in payload["violations"]:
        mark = " 待分诊" if v["triage"] == "untriaged" else " " + _PAYLOAD_TRIAGE_DOC[v["triage"]]
        issues.append(
            f"负载: 前端漏发后端必填 ({_PAYLOAD_KIND_DOC[v['kind']]}): "
            f"{v['method']} {v['path']} → 后端 {v['endpoint']} @ "
            f"{v['rel']}:{v['line']} — {v['detail']};{mark}"
        )
    resp = snap["response"]
    for v in resp["violations"]:
        mark = (
            " 待分诊"
            if v["triage"] == "untriaged"
            else " " + _RESP_TRIAGE_DOC[v["triage"]]
        )
        issues.append(
            f"响应: 前端声明要读的响应字段后端从不返回 "
            f"({_RESP_KIND_DOC[v['kind']]}): {v['method']} {v['path']} → 后端 "
            f"{v['endpoint']} @ {v['rel']}:{v['line']} — 缺 "
            f"{', '.join(v['missing'])};{mark}"
        )
    wsp = snap["ws_payload"]
    _wsp_issue = {
        "handler-undeclared": (
            "WS 负载: 后端 handler 读 `msg.<字段>` 而 WSMessage 未声明 "
            "(AttributeError 死帧): "
        ),
        "fe-undeclared": (
            "WS 负载: 前端发送的字段 WSMessage 未声明 (被 Pydantic 静默丢弃): "
        ),
    }
    for v in wsp["violations"]:
        mark = (
            " 待分诊"
            if v["triage"] == "untriaged"
            else " " + _WS_PAYLOAD_TRIAGE_DOC[v["triage"]]
        )
        issues.append(
            f"{_wsp_issue[v['kind']]}{v['type']}.{v['field']} @ "
            f"{v['rel']}:{v['line']} — {v['detail']};{mark}"
        )
    for f in wsp["dead_fields"]:
        issues.append(
            f"WS 负载: WSMessage 声明字段既无 handler 读取也零前端发送 (宣称却无人接): {f}"
        )
    sp = snap["sse_payload"]
    for v in sp["violations"]:
        mark = (
            " 待分诊"
            if v["triage"] == "untriaged"
            else " " + _SSE_PAYLOAD_TRIAGE_DOC[v["triage"]]
        )
        issues.append(
            f"SSE 负载: 前端读的帧 payload 顶层字段后端从不发 "
            f"({_SSE_PAYLOAD_KIND_DOC[v['kind']]}): {v['channel']}/{v['frame']} → "
            f"`t.{v['field']}` @ {v['rel']}:{v['line']};{mark}"
        )
    wep = snap["ws_ev_payload"]
    for v in wep["violations"]:
        mark = (
            " 待分诊"
            if v["triage"] == "untriaged"
            else " " + _WS_EV_PAYLOAD_TRIAGE_DOC[v["triage"]]
        )
        issues.append(
            f"WS 事件负载: 前端读的帧 payload 顶层字段后端从不发 "
            f"({_WS_EV_PAYLOAD_KIND_DOC[v['kind']]}): {v['channel']}/{v['frame']} → "
            f"`data.{v['field']}` @ {v['rel']}:{v['line']};{mark}"
        )
    hf = snap["http_field"]
    for v in hf["violations"]:
        mark = (
            " 待分诊"
            if v["triage"] == "untriaged"
            else " " + _HTTP_FIELD_TRIAGE_DOC[v["triage"]]
        )
        issues.append(
            f"HTTP 请求字段: {_HTTP_FIELD_KIND_DOC[v['kind']]}: "
            f"{v['method']} {v['endpoint']} → `{v['field']}` @ "
            f"{v['rel']}:{v['line']} — {v['detail']};{mark}"
        )
    return issues


def render_mece_markdown(snap: dict) -> str:
    lines: list[str] = []
    lines.append(
        "# MECE 契约审计 (奖励面 + 授权面 + 工作流面 + 模式面 + 词汇面 + 工具面 + 钩子面 + 事件面 + SSE 消费面 + WS 消费面 + HTTP API 消费面 + 请求负载面 + 响应结构面 + WS 请求负载面 + SSE 事件负载面 + WS 事件负载面 + HTTP 请求字段面)"
    )
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.contract_audit --out docs/mece-audit.md`."
    )
    lines.append(
        "以 MECE 两原则审计 agent 的**奖励面 / 授权面 / 工作流面 / 模式面 / "
        "词汇面 / 工具面 / 钩子面 / 事件面 / SSE 消费面 / WS 消费面 / HTTP API 消费面 / "
        "请求负载面 / 响应结构面 / WS 请求负载面 / SSE 事件负载面 / WS 事件负载面 / "
        "HTTP 请求字段面**: "
        "**collectively exhaustive** 抓「宣称维度零调用者 / "
        "面之间的缺口」; **mutually exclusive** 抓「同轴惩罚叠加」「跨模块同名重复实现」「词表互不一致」"
        "「同名工具名多类声明」「事件常量撞值」「SSE 帧名挂错通道」「WS 帧名挂错端点」"
        "「HTTP 同 method+path 多模块注册」「前端漏发后端必填请求负载」"
        "「前端声明要读的响应字段后端从不返回」「WS 入站字段模型未声明」"
        "「前端读的 SSE 帧 payload 顶层字段后端从不发」「前端读的 WS 帧 payload 顶层字段后端从不发」"
        "「HTTP handler 读请求体模型未声明字段 / 前端发未声明键 / body-dict 下标读键而前端从不发」. 纯静态扫描, "
        "只提示候选, 不判死."
    )
    lines.append("")
    lines.append(render_reward_markdown(snap["reward"]))
    lines.append(render_scope_markdown(snap["scope"]))
    lines.append(render_workflow_markdown(snap["workflow"]))
    lines.append(render_mode_markdown(snap["modes"]))
    lines.append(render_vocabulary_markdown(snap["vocabulary"]))
    lines.append(render_tool_markdown(snap["tools"]))
    lines.append(render_hook_markdown(snap["hooks"]))
    lines.append(render_event_markdown(snap["events"]))
    lines.append(render_sse_markdown(snap["sse"]))
    lines.append(render_ws_markdown(snap["ws"]))
    lines.append(render_http_markdown(snap["http"]))
    lines.append(render_payload_markdown(snap["payload"]))
    lines.append(render_response_markdown(snap["response"]))
    lines.append(render_ws_payload_markdown(snap["ws_payload"]))
    lines.append(render_sse_payload_markdown(snap["sse_payload"]))
    lines.append(render_ws_ev_payload_markdown(snap["ws_ev_payload"]))
    lines.append(render_http_field_markdown(snap["http_field"]))
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
    parser.add_argument("--events", action="store_true", help="只看事件面")
    parser.add_argument("--sse", action="store_true", help="只看 SSE 消费面")
    parser.add_argument("--ws", action="store_true", help="只看 WS 消费面")
    parser.add_argument("--http", action="store_true", help="只看 HTTP API 消费面")
    parser.add_argument("--payload", action="store_true", help="只看请求负载面")
    parser.add_argument("--response", action="store_true", help="只看响应结构面")
    parser.add_argument("--ws-payload", action="store_true", help="只看 WS 请求负载面")
    parser.add_argument("--sse-payload", action="store_true", help="只看 SSE 事件负载面")
    parser.add_argument("--ws-ev-payload", action="store_true", help="只看 WS 事件负载面")
    parser.add_argument("--http-field", action="store_true", help="只看 HTTP 请求字段面")
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
        "events": (args.events, build_event_contract, render_event_markdown),
        "sse": (args.sse, build_sse_contract, render_sse_markdown),
        "ws": (args.ws, build_ws_contract, render_ws_markdown),
        "http": (args.http, build_http_contract, render_http_markdown),
        "payload": (args.payload, build_payload_contract, render_payload_markdown),
        "response": (args.response, build_response_contract, render_response_markdown),
        "ws_payload": (
            args.ws_payload,
            build_ws_payload_contract,
            render_ws_payload_markdown,
        ),
        "sse_payload": (
            args.sse_payload,
            build_sse_payload_contract,
            render_sse_payload_markdown,
        ),
        "ws_ev_payload": (
            args.ws_ev_payload,
            build_ws_ev_payload_contract,
            render_ws_ev_payload_markdown,
        ),
        "http_field": (
            args.http_field,
            build_http_field_contract,
            render_http_field_markdown,
        ),
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
