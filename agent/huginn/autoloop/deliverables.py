"""确定性交付物清单 + 验收门.

清单只从 objective 文本解析, agent **不能改写** —— 这是独立于模型的可判定锚点:
执行对象 (objective + 声明交付物) 与验收标准都不再由模型自己生成.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Contract",
    "Deliverable",
    "LineSpec",
    "check_contract",
    "check_manifest",
    "fixed_protocol_hypothesis",
    "objective_block_for",
    "parse_contract",
    "parse_manifest",
    "render_contract_hint",
    "render_objective_block",
]

_ARTIFACT_RE = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:md|json|jsonl|csv|tsv|txt|png|jpg|jpeg|pdf|xlsx|parquet|npz|npy|py)\b"
)
_SECTION_MARKERS = (
    "【交付物】",
    "交付物：",
    "交付物:",
    "必须交付",
    "产出文件",
    "deliverables:",
    "deliverables：",
)
_SOURCE_PREFIXES = ("agent/", "src/", "huginn/", "tests/")
_MIN_BYTES = 32


@dataclass(frozen=True)
class Deliverable:
    path: str
    min_bytes: int = _MIN_BYTES


def _section_text(objective: str) -> str:
    """取「交付物」段文本; 无该段返回空串 (调用方退回全文扫描)."""
    low = objective.lower()
    for marker in _SECTION_MARKERS:
        idx = low.find(marker.lower())
        if idx < 0:
            continue
        tail = objective[idx + len(marker):]
        stop = len(tail)
        for sep in ("\n\n", "【", "\n#"):
            j = tail.find(sep)
            if 0 <= j < stop:
                stop = j
        return tail[:stop]
    return ""


def parse_manifest(objective: str) -> list[Deliverable]:
    """objective → 去重后的交付物清单 (无声明时返回 []，表示不适用).

    `.py` 只在**显式交付物段**里才算验收物: 散文里的 `["python3","run.py"]` 是执行
    示例, 若当交付物会永久判 missing (ws31 实测的同类误判)。
    """
    if not isinstance(objective, str) or not objective.strip():
        return []
    section = _section_text(objective)
    text = section or objective
    seen: set[str] = set()
    out: list[Deliverable] = []
    # 用 finditer 而不是 findall: 需要知道匹配起点, 才能判定它前面是不是 "/".
    # 绝对路径 "/workspace/.../design.md" 的匹配会**跳过前导斜杠**从 "workspace"
    # 开始, findall 拿不到那个 "/", startswith("/") 守卫永远不生效 → 绝对路径被
    # 当成 workspace 下的相对路径去验收, 必然判 missing (ws31 实测).
    for m in _ARTIFACT_RE.finditer(text):
        raw = m.group(0)
        if m.start() > 0 and text[m.start() - 1] == "/":
            continue  # 绝对路径: 引用材料不是交付物
        if not section and raw.endswith(".py"):
            continue  # 无交付物段时, 散文里的脚本名只是执行示例
        path = raw.strip().strip("'\"`，,。；;：:")
        if not path or path.startswith(("/", "..", "./")):
            continue
        if any(path.startswith(p) for p in _SOURCE_PREFIXES):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(Deliverable(path))
    return out


_PROTOCOL_FIX_RE = re.compile(
    r"不允许事后改口径|不得改口径|不得换检验|不允许换检验|不许改口径|口径固定"
    r"|固定假设|预先注册|pre-registered|preregistered",
    re.I,
)
_H_ITEM_RE = re.compile(r"^\s*(?:[-*]\s*)?H\d\s*[（(:：]", re.M)
_FIXED_H_ITEM_MAX = 220


def fixed_protocol_hypothesis(objective: str) -> str:
    """objective 把假设/口径钉死时返回锚定陈述, 否则空串.

    ws35 实测 (结构性缺陷 I): objective 已写死 H1–H5 与"不允许事后改口径换检验",
    引擎仍走 hypothesize/pivot 生成合成假设 (ΔEa=α/L、ΔEa(f) 递减), 再拿这条合成
    假设去 plan_check 对齐 —— iter 10 的计划被判"与假设不对齐", 而 objective 的
    H1–H5 与已落盘交付物被搁在一边. objective 自带固定假设清单时, 它**就是**执行
    对象, 不该再由模型另立假设.
    """
    if not isinstance(objective, str) or not objective.strip():
        return ""
    if not _PROTOCOL_FIX_RE.search(objective):
        return ""
    items = list(_H_ITEM_RE.finditer(objective))
    if len(items) < 3:
        return ""
    lines: list[str] = []
    for i, m in enumerate(items):
        end = items[i + 1].start() if i + 1 < len(items) else len(objective)
        chunk = " ".join(objective[m.start():end].split())
        lines.append(chunk[:_FIXED_H_ITEM_MAX])
    body = "\n".join(f"- {ln}" for ln in lines)
    return (
        "【objective 已固定的分析协议 — 不要另立假设, 也不要 pivot 换方向】\n"
        f"{body}\n"
        "执行口径: 严格按 objective 写死的方法要求与交付物键名落地; 逐轮修的是"
        "交付物本身 (文件缺失 / 键名不符 / 数值未算), 不是换假设。"
    )


def render_acceptance_gap(gate: dict[str, Any]) -> str:
    """交付物验收缺口 (确定性门) 的 prompt 摘要 — 独立字段, 不挤在 spec_hint 里.

    ws35 实测: 门的缺口只写进 ``_speculator_hint``, 而 decider prompt 里该字段
    只取前 300 字符 (``[:300]``), 门拒绝原因根本没到决策者眼前 → 模型以为已
    完成, 继续 skip. 这里把它抬成独立 block, 并给出可执行的修复目标.
    """
    if not isinstance(gate, dict) or gate.get("passed", True):
        return ""
    _cap = 8

    def _fmt(key: str) -> str:
        vals = gate.get(key) or []
        if not isinstance(vals, list):
            vals = [vals]
        if not vals:
            return f"{key}=[]"
        shown = ", ".join(str(v) for v in vals[:_cap])
        more = f" (+{len(vals) - _cap} more)" if len(vals) > _cap else ""
        return f"{key}=[{shown}]{more}"

    parts = [_fmt(k) for k in ("missing", "empty", "invalid")]
    contract = gate.get("contract")
    if isinstance(contract, dict) and not contract.get("passed", True):
        cparts = [
            f"{k}=[{', '.join(str(x) for x in (contract.get(k) or [])[:_cap])}]"
            for k in ("missing_keys", "short_lines", "bad_lines", "missing_fields")
            if contract.get(k)
        ]
        if cparts:
            parts.append("contract: " + "; ".join(cparts))
    return "; ".join(p for p in parts if p)[:600]


def acceptance_block_for(host: Any, header: str) -> str:
    """把当前验收缺口渲染成 prompt 块 —— 谁要动手, 缺口就必须到谁手里.

    ws35 实测: 缺口只进 decider prompt, 而"怎么改"由 plan 写、"改哪段"由 coder
    做; 两者都看不到缺口时, 模型会把旧计划再执行一遍 (iter 4–9 反复 execute,
    results.json 一直缺). host 优先用 ``_deliverables_ok``, 没有时按 manifest +
    契约自己算 (engine 侧只有这两个字段).
    """
    gate_fn = getattr(host, "_deliverables_ok", None)
    if callable(gate_fn):
        gate = gate_fn()
    else:
        manifest = getattr(host, "_deliverable_manifest", None) or []
        if not manifest:
            return ""
        gate = check_manifest(
            getattr(host, "workspace", None),
            manifest,
            getattr(host, "_deliverable_contract", None),
        )
    gap = render_acceptance_gap(gate)
    if not gap:
        return ""
    return f"\n{header}\n{gap}\n"


def _json_has_substance(payload: Any) -> bool:
    """结构化交付物必须含至少一个非空叶子值.

    `{}` / `[]` / `{"results": []}` 都能解析、也能凑够 32 字节, 但没有任何可核对
    的数值 —— 存在性判据会把它判绿, 即假绿. 这里按递归内容判「有/无实质」.
    """
    if isinstance(payload, dict):
        return any(_json_has_substance(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_json_has_substance(v) for v in payload)
    if isinstance(payload, str):
        return bool(payload.strip())
    return payload is not None


@dataclass(frozen=True)
class LineSpec:
    """JSONL 交付物的行契约: 至少 N 行, 每行至少含这些字段."""

    min_lines: int = 0
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Contract:
    """objective 声明的内容契约 (JSON 必需键名 / JSONL 行 schema).

    只做**确定性**抽取: 从句式 ("键名至少包含 …" / "每行… 至少含 …" /
    "少于 N 行…") 里取标识符. 抽不到就是空契约 (不适用) —— 不猜、不由模型补写.
    """

    json_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)
    line_specs: dict[str, LineSpec] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.json_keys or self.line_specs)


_KEY_MARKERS = (
    "键名至少包含",
    "键名必须包含",
    "键名包含",
    "至少包含以下键",
    "必须包含以下键",
    "required keys",
)
_FILE_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.jsonl?\b")
_IDENT_RE = re.compile(r"[a-z][a-z0-9_]{2,}")
_KEY_STOP = frozenset(
    {"json", "jsonl", "csv", "md", "tsv", "txt", "true", "false", "supported", "refuted", "keys", "key"}
)
_MAX_MIN_LINES = 10_000


def _sentence_tail(text: str, pos: int) -> str:
    """pos 之后到句末 (。或换行) 的片段."""
    tail = text[pos:]
    stops = [i for i in (tail.find("。"), tail.find("\n")) if i >= 0]
    return tail[: min(stops)] if stops else tail


def _nearest_file_before(text: str, pos: int, suffix: str) -> str | None:
    """pos 之前最近一次出现的、以 suffix 结尾的文件名 (取 basename)."""
    found = None
    for m in _FILE_RE.finditer(text[:pos]):
        raw = m.group(0)
        if raw.endswith(suffix):
            found = raw.rsplit("/", 1)[-1]
    return found


def _idents(tail: str, stop: frozenset[str]) -> list[str]:
    """逗号/顿号分隔的片段里各取首个标识符 (去重保序)."""
    out: list[str] = []
    for seg in re.split(r"[、，,;；]", tail):
        m = _IDENT_RE.search(seg)
        if not m:
            continue
        tok = m.group(0)
        if tok in stop or tok in out:
            continue
        out.append(tok)
    return out


def parse_contract(objective: str, manifest: list[Deliverable] | None = None) -> Contract:
    """objective → 内容契约 (无声明时返回空契约, 表示不适用)."""
    if not isinstance(objective, str) or not objective.strip():
        return Contract()
    text = objective
    declared = [d.path for d in (manifest or [])]

    json_keys: dict[str, tuple[str, ...]] = {}
    for marker in _KEY_MARKERS:
        idx = text.find(marker)
        while idx >= 0:
            path = _nearest_file_before(text, idx, ".json")
            if path is None:
                path = next((p for p in declared if p.endswith(".json")), None)
            toks = _idents(_sentence_tail(text, idx + len(marker)).lstrip("：:，, "), _KEY_STOP)
            if path and toks:
                merged = tuple(dict.fromkeys(json_keys.get(path, ()) + tuple(toks)))
                json_keys[path] = merged
            idx = text.find(marker, idx + len(marker))

    line_specs: dict[str, LineSpec] = {}
    for m in re.finditer(r"至少含[：:，,]?", text):
        path = _nearest_file_before(text, m.start(), ".jsonl")
        if path is None:
            continue
        fields = _idents(_sentence_tail(text, m.end()).lstrip("：:，, "), _KEY_STOP)
        if fields:
            prev = line_specs.get(path, LineSpec())
            line_specs[path] = LineSpec(
                min_lines=prev.min_lines, fields=tuple(dict.fromkeys(prev.fields + tuple(fields)))
            )

    for path in dict.fromkeys(declared + list(line_specs)):
        if not path.endswith(".jsonl"):
            continue
        base = path.rsplit("/", 1)[-1]
        if base not in text:
            continue
        for sent in re.split(r"[。\n]", text):
            if base not in sent:
                continue
            mm = re.search(r"(\d+)\s*行", sent)
            if not mm:
                continue
            n = int(mm.group(1))
            if 0 < n <= _MAX_MIN_LINES:
                prev = line_specs.get(path, LineSpec())
                line_specs[path] = LineSpec(min_lines=n, fields=prev.fields)
                break

    return Contract(json_keys=json_keys, line_specs=line_specs)


def check_contract(workspace: Any, contract: Contract) -> dict[str, Any]:
    """内容契约校验: 必需键 / 行数 / 每行字段 (确定性).

    只报**契约**层面的问题; 文件缺失 / 不可解析由 ``check_manifest`` 报, 不重复计.
    """
    root = Path(workspace)
    missing_keys: list[str] = []
    short_lines: list[str] = []
    bad_lines: list[str] = []
    missing_fields: list[str] = []

    for path, want in contract.json_keys.items():
        p = root / path
        if not p.is_file():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        flat: dict[str, Any] = dict(payload)
        # 容忍一层嵌套 (results 放在 "q1": {...} 下面) —— 与审计口径一致.
        for v in payload.values():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat.setdefault(k2, v2)
        for k in want:
            if k not in flat:
                missing_keys.append(f"{path}:{k}")

    for path, spec in contract.line_specs.items():
        p = root / path
        if not p.is_file():
            continue
        try:
            lines = [ln for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        except OSError:
            continue
        if spec.min_lines and len(lines) < spec.min_lines:
            short_lines.append(f"{path}:{len(lines)}<{spec.min_lines}")
        if not spec.fields:
            continue
        for i, ln in enumerate(lines, 1):
            try:
                obj = json.loads(ln)
            except ValueError:
                bad_lines.append(f"{path}:{i}")
                continue
            if not isinstance(obj, dict):
                bad_lines.append(f"{path}:{i}")
                continue
            miss = [f for f in spec.fields if f not in obj]
            if miss:
                missing_fields.append(f"{path}:{i}:{','.join(miss)}")

    return {
        "passed": not (missing_keys or short_lines or bad_lines or missing_fields),
        "missing_keys": missing_keys,
        "short_lines": short_lines,
        "bad_lines": bad_lines,
        "missing_fields": missing_fields,
    }


def _contract_lines(contract: Contract, manifest: list[Deliverable] | None = None) -> list[str]:
    """契约的可读条目 (提示与 prompt 块共用同一口径)."""
    parts: list[str] = []
    if manifest:
        parts.append(f"必须交付: {', '.join(d.path for d in manifest)}")
    for path, keys in contract.json_keys.items():
        parts.append(f"{path} 必须含键: {', '.join(keys)}")
    for path, spec in contract.line_specs.items():
        seg = f"{path} 至少 {spec.min_lines} 行" if spec.min_lines else path
        if spec.fields:
            seg += f"，每行必须含: {', '.join(spec.fields)}"
        parts.append(seg)
    return parts


def render_objective_block(objective: str, contract: Contract | None) -> str:
    """把 objective 原文 + 内容契约渲染成**不可截断的 prompt 块**.

    根因: 各 phase prompt 里 context 被截到 2000/1000 字符, hint 被截到尾部
    500 字符; 7KB 的 objective (含 H1–H5 的定义与阈值) 到不了模型, 模型就只能
    自造一套假设 (ws35 实测: 自创 H1–H5, 键名对但语义不同).
    """
    obj = (objective or "").strip()
    if not obj:
        return ""
    text = (
        "\n### TASK OBJECTIVE (verbatim; authoritative — do not restate, "
        "shorten, or redefine it)\n" + obj + "\n"
    )
    if contract:
        text += (
            "\n### DELIVERABLES CONTRACT (machine-checked by the engine; "
            "deviation fails acceptance)\n- "
            + "\n- ".join(_contract_lines(contract))
            + "\n用 objective 里写明的 H1..H5 定义与阈值逐条计算, 不得替换成自己的假设.\n"
        )
    return text


def objective_block_for(host: Any) -> str:
    """从 host (engine / 协作对象) 上取 objective 与契约, 渲染成 prompt 块.

    host 上没有 ``_objective`` 时返回空串 (rcb 等不声明契约的路径保持原行为).
    """
    objective = getattr(host, "_objective", "") or ""
    return render_objective_block(objective, getattr(host, "_deliverable_contract", None))


def render_contract_hint(manifest: list[Deliverable], contract: Contract | None) -> str:
    if contract is None or not contract:
        return ""
    parts = _contract_lines(contract, manifest)
    return (
        "[交付物契约 · 机器验收] " + "；".join(parts)
        + "。键名/字段名必须与 objective 字面一致，禁止在交付物里自造另一套键名或另一套假设；"
        "上述每条都由引擎逐项检查，不满足即拒绝收结。"
    )


def check_manifest(
    workspace: Any,
    manifest: list[Deliverable],
    contract: "Contract | None" = None,
) -> dict[str, Any]:
    """三关验收: 存在 / 非空 / 结构化文件可解析 (确定性, 不由 LLM 判定).

    「非空」既指零字节, 也指 JSON 虽可解析但没有实质内容 (空壳) —— 两者都判红.
    ``contract`` 非空时再校验**内容契约** (必需键名 / JSONL 行 schema): 文件在、
    非空、可解析只证明"有东西", 不证明"交付的是 objective 要的东西".
    """
    root = Path(workspace)
    missing: list[str] = []
    empty: list[str] = []
    invalid: list[str] = []
    for d in manifest:
        p = root / d.path
        if not p.is_file():
            missing.append(d.path)
            continue
        try:
            size = p.stat().st_size
        except OSError:
            missing.append(d.path)
            continue
        if size == 0:
            empty.append(d.path)
            continue
        if p.suffix == ".json":
            # 零字节 → 空; 有内容但解析失败 → 无效; 可解析但无实质 → 空壳(判空).
            try:
                payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                # 读失败或 JSONDecodeError(⊂ValueError) → 声明交付物不可验收.
                invalid.append(d.path)
                continue
            if not isinstance(payload, (dict, list)) or not _json_has_substance(payload):
                empty.append(d.path)
                continue
        if size < d.min_bytes:
            empty.append(d.path)
    contract_report = check_contract(root, contract) if contract is not None else None
    passed = not (missing or empty or invalid)
    if contract_report is not None and not contract_report["passed"]:
        passed = False
    out = {
        "passed": passed,
        "declared": [d.path for d in manifest],
        "missing": missing,
        "empty": empty,
        "invalid": invalid,
        "checked": len(manifest),
    }
    if contract_report is not None:
        out["contract"] = contract_report
    return out
