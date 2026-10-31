"""确定性交付物清单 + 验收门.

清单/键名/行 schema 只从 ``contract.json`` (schema ``huginn.task-contract/v1``)
读取, agent **不能改写**; objective 散文不再被正则解析 —— 同一份契约字节给出
同一判定, 与题面措辞无关.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.task_contract import RequiredKey, TaskContract, check_required_keys

logger = logging.getLogger(__name__)

__all__ = [
    "Contract",
    "Deliverable",
    "LineSpec",
    "StaticConstraints",
    "acceptance_block_for",
    "check_contract",
    "check_manifest",
    "check_reproducible",
    "check_static_constraints",
    "fixed_protocol_hypothesis",
    "objective_block_for",
    "render_acceptance_gap",
    "render_contract_hint",
    "render_objective_block",
    "views_from_task_contract",
]

_MIN_BYTES = 32
#: 契约指纹绑定的交付物 (spec: 交付物 results.json 必须写入 contract_sha256).
_BINDING_FILE = "results.json"


@dataclass(frozen=True)
class Deliverable:
    path: str
    min_bytes: int = _MIN_BYTES


def views_from_task_contract(
    task_contract: TaskContract,
) -> tuple[list[Deliverable], Contract, StaticConstraints]:
    """``contract.json`` → 交付物清单 + 内容契约 + 静态约束 (纯投影, 不跑正则).

    三个视图都由同一份已校验契约派生, 门与 prompt 提示共用, 不存在"提示一套、
    检查另一套"的漂移.
    """
    manifest = [Deliverable(p) for p in task_contract.required_files]

    json_keys: dict[str, tuple[str, ...]] = {}
    for spec in task_contract.required_keys:
        merged = tuple(dict.fromkeys(json_keys.get(spec.file, ()) + (spec.key,)))
        json_keys[spec.file] = merged

    line_specs: dict[str, LineSpec] = {}
    if task_contract.loop_log is not None:
        ll = task_contract.loop_log
        line_specs[ll.path] = LineSpec(min_lines=ll.lines, fields=ll.fields)

    hyp = task_contract.hypotheses
    contract = Contract(
        json_keys=json_keys,
        line_specs=line_specs,
        hypothesis_range=hyp.labels if hyp else (),
        prediction_only=hyp.prediction_only if hyp else frozenset(),
        key_specs=task_contract.required_keys,
        contract_sha256=task_contract.sha256,
    )

    static = task_contract.static
    constraints = StaticConstraints(
        forbidden_imports=static.forbidden_imports if static else (),
        required_literals=(str(static.seed),) if static and static.seed is not None else (),
        required_inputs=(),
    )
    return manifest, contract, constraints


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
            for k in (
                "missing_keys", "bad_types", "short_lines", "bad_lines", "missing_fields",
                "bad_hypotheses", "bad_binding",
            )
            if contract.get(k)
        ]
        if cparts:
            parts.append("contract: " + "; ".join(cparts))
    static = gate.get("static_violations")
    if static:
        parts.append(f"static=[{'; '.join(str(s) for s in static[:_cap])}]")
    repro = gate.get("repro")
    if isinstance(repro, dict) and not repro.get("passed", True):
        seg = (
            f"repro: 跑 {repro.get('script')} 未重新生成产物 rc={repro.get('returncode')} "
            f"missing={repro.get('missing') or []} invalid={repro.get('invalid') or []}"
        )
        stderr = str(repro.get("stderr") or "").strip()
        if stderr:
            seg += f" stderr={stderr[-240:]}"
        parts.append(seg)
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


# ---------------------------------------------------------------------------
# 静态约束: 契约写死的"禁用依赖 / 固定种子" (源码静态判定, 不靠模型自述)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticConstraints:
    """契约里的硬性技术条款 — 由脚本源码静态判定, 不靠模型自述."""

    forbidden_imports: tuple[str, ...] = ()
    required_literals: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(
            self.forbidden_imports or self.required_literals or self.required_inputs
        )


def check_static_constraints(
    workspace: Any, manifest: list[Deliverable], constraints: StaticConstraints
) -> list[str]:
    """按静态约束检查声明的脚本 (返回违规清单; 空 = 通过)."""
    if not constraints or not manifest:
        return []
    root = Path(workspace)
    out: list[str] = []
    for d in manifest:
        if not d.path.endswith(".py"):
            continue
        p = root / d.path
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        for mod in constraints.forbidden_imports:
            if re.search(
                rf"^\s*(?:import\s+{mod}\b|from\s+{mod}\s+import)", src, re.M
            ):
                out.append(f"{d.path} 导入了 objective 禁用的 {mod}")
        for lit in constraints.required_literals:
            if lit not in src:
                out.append(f"{d.path} 缺少 objective 指定的固定随机种子 {lit}")
        for base in constraints.required_inputs:
            # 词边界匹配: "cleaned_ea_dataset.csv" 里的 "ea_dataset.csv" 不算引用
            # (ws35 实测 agent 就是这么换成变体数据的).
            pat = rf"(?<![A-Za-z0-9_]){re.escape(base)}"
            if not re.search(pat, src):
                out.append(f"{d.path} 未引用 objective 指定的数据文件 {base}")
    return out


# ---------------------------------------------------------------------------
# 可复现性: 产物必须由声明的脚本重跑生成, 不接受"手写数值"
# ---------------------------------------------------------------------------

_REPRO_TIMEOUT_S = float(os.environ.get("HUGINN_REPRO_TIMEOUT_S", "240"))
_REPRO_STASH_DIR = ".gate_rerun"
#: 只有机器产物的"手写 vs 真算"可判: 叙述型产物 (.md/.txt) 是作者写出来的,
#: 永远不可能由分析脚本逐字重跑生成. ws35 实测: 一个数值全对、报告六节齐全的
#: 交付物因为 report.md 不可重跑被判 success=False —— 门把正确解判死.
_MACHINE_OUTPUT_SUFFIXES = (".json", ".jsonl", ".csv", ".tsv")


def _sig(path: Path) -> tuple[int, int] | None:
    st = None
    with contextlib.suppress(OSError):
        st = path.stat()
    return None if st is None else (st.st_mtime_ns, st.st_size)


def repro_signature(workspace: Any, manifest: list[Deliverable]) -> tuple:
    """脚本 + 产物的文件签名 (缓存键: 任何一方被改写都重新判)."""
    root = Path(workspace)
    return tuple((d.path, _sig(root / d.path)) for d in manifest)


def check_reproducible(
    workspace: Any,
    manifest: list[Deliverable],
    timeout: float | None = None,
) -> dict[str, Any]:
    """把产物挪开重跑声明的脚本, 要求它自己把产物重新生成.

    ws35 实测: coder 没有真实执行分析, 直接手写 results.json (16 个键全在、数值
    全编), 存在性 + 键名两道门全绿 —— 只有和真值对账才发现. 判据改成**可复现**:
    产物能不能由一个可重跑的脚本产生. 失败时把原产物放回, 不破坏 agent 的中间成果.
    """
    root = Path(workspace)
    scripts = [d.path for d in manifest if d.path.endswith(".py")]
    outputs = [d for d in manifest if not d.path.endswith(".py")]
    # 叙述型产物不参与"重跑生成"判据 (作者写的报告不可能被脚本逐字重跑出来);
    # 但存在性/非空仍由 check_manifest 负责, 不是免检.
    authored = [d for d in outputs if not d.path.endswith(_MACHINE_OUTPUT_SUFFIXES)]
    outputs = [d for d in outputs if d.path.endswith(_MACHINE_OUTPUT_SUFFIXES)]
    if not scripts or not outputs:
        return {
            "passed": True,
            "skipped": "no machine output to reproduce",
            "authored_exempt": [d.path for d in authored],
        }
    script = scripts[0]
    if not (root / script).is_file():
        return {"passed": True, "skipped": f"{script} missing (manifest gate owns this)"}
    timeout = _REPRO_TIMEOUT_S if timeout is None else timeout
    stash = root / _REPRO_STASH_DIR
    moved: list[tuple[Path, Path]] = []
    try:
        stash.mkdir(exist_ok=True)
        for d in outputs:
            p = root / d.path
            if p.is_file():
                dst = stash / d.path.replace("/", "__")
                p.replace(dst)
                moved.append((p, dst))
        proc = subprocess.run(
            [sys.executable, script],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc = proc.returncode
        stderr = (proc.stderr or "")[-800:]
    except (subprocess.TimeoutExpired, OSError) as exc:
        # 超时/起不了进程: 都判红 (逐字记原因), 原产物在下面回滚.
        rc, stderr = -9, str(exc)[-800:]
    after = check_manifest(root, outputs)
    ok = rc == 0 and after["passed"]
    if not ok:
        for p, dst in moved:
            with contextlib.suppress(OSError):
                dst.replace(p)
        return {
            "passed": False,
            "script": script,
            "returncode": rc,
            "missing": after["missing"],
            "empty": after["empty"],
            "invalid": after["invalid"],
            "stderr": stderr,
        }
    for _, dst in moved:
        with contextlib.suppress(OSError):
            dst.unlink()
    return {
        "passed": True,
        "script": script,
        "returncode": rc,
        "authored_exempt": [d.path for d in authored],
    }


def _read_text(p: Path) -> str | None:
    """读文本; 读失败 (OSError) → None, 调用方按"不可验收"处理."""
    out: str | None = None
    with contextlib.suppress(OSError):
        out = p.read_text(encoding="utf-8", errors="replace")
    return out


def _loads(text: str) -> Any:
    """解析 JSON; 解析失败 (ValueError ⊃ JSONDecodeError) → None."""
    out: Any = None
    with contextlib.suppress(ValueError):
        out = json.loads(text)
    return out


def _load_json(p: Path) -> Any:
    """读 + 解析一个 JSON 文件; 任一步失败 → None."""
    text = _read_text(p)
    return None if text is None else _loads(text)


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
    #: objective 声明 "hypotheses 对象（P1–P5 各自 supported/refuted）" 时的必需子键.
    #: 结构性的: 交付物写成 list 或自造键名会被确定性判红, 不再靠人眼对账.
    hypothesis_range: tuple[str, ...] = ()
    #: 没有"推翻条件"的假设项 (契约里只给了可证伪预测, 尚无实测可判):
    #: 这些键允许 prediction/pending 类取值, 其余键仍只认 supported/refuted.
    prediction_only: frozenset[str] = frozenset()
    #: 键的类型约束 (来自 contract.json 的 required_keys; 空 = 只查存在性).
    key_specs: tuple[RequiredKey, ...] = ()
    #: contract.json 原始字节的 sha256; 非空时 results.json 必须绑定同一指纹.
    contract_sha256: str = ""

    def __bool__(self) -> bool:
        return bool(self.json_keys or self.line_specs or self.hypothesis_range)


#: 纯预测项允许的状态写法 (尚无实测可判, 只能登记预测).
_PREDICTION_STATUSES = frozenset({
    "prediction", "prediction_only", "prediction-only", "predictive",
    "pending", "untested", "not_testable", "预测", "待检验", "待验证",
})


def check_contract(workspace: Any, contract: Contract) -> dict[str, Any]:
    """内容契约校验: 必需键 / 行数 / 每行字段 (确定性).

    只报**契约**层面的问题; 文件缺失 / 不可解析由 ``check_manifest`` 报, 不重复计.
    """
    root = Path(workspace)
    missing_keys: list[str] = []
    bad_types: list[str] = []
    short_lines: list[str] = []
    bad_lines: list[str] = []
    missing_fields: list[str] = []
    bad_hypotheses: list[str] = []
    bad_binding: list[str] = []

    for path, want in contract.json_keys.items():
        p = root / path
        if not p.is_file():
            continue
        payload = _load_json(p)
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
        specs = tuple(s for s in contract.key_specs if s.file == path)
        if specs:
            bad_types.extend(check_required_keys(flat, specs))

    for path, spec in contract.line_specs.items():
        p = root / path
        if not p.is_file():
            continue
        text = _read_text(p)
        if text is None:
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if spec.min_lines and len(lines) < spec.min_lines:
            short_lines.append(f"{path}:{len(lines)}<{spec.min_lines}")
        if not spec.fields:
            continue
        for i, ln in enumerate(lines, 1):
            obj = _loads(ln)
            if not isinstance(obj, dict):
                bad_lines.append(f"{path}:{i}")
                continue
            miss = [f for f in spec.fields if f not in obj]
            if miss:
                missing_fields.append(f"{path}:{i}:{','.join(miss)}")

    if contract.hypothesis_range:
        path = next(
            (p for p, ks in contract.json_keys.items() if "hypotheses" in ks),
            next((p for p in contract.json_keys if p.endswith(".json")), None),
        )
        payload = None
        if path:
            fp = root / path
            if fp.is_file():
                payload = _load_json(fp)
            hyp = payload.get("hypotheses") if isinstance(payload, dict) else None
            if payload is not None:
                if not isinstance(hyp, dict):
                    bad_hypotheses.append(
                        f"{path}:hypotheses 必须是对象 {'/'.join(contract.hypothesis_range)}"
                    )
                else:
                    for k in contract.hypothesis_range:
                        v = hyp.get(k)
                        if isinstance(v, dict):
                            v = v.get("status") or v.get("result") or v.get("supported")
                        s = str(v).strip().lower()
                        if s in {
                            "supported", "refuted", "true", "false", "被支持", "被推翻",
                        }:
                            continue
                        # 没有推翻条件的假设项 (纯预测): 允许显式标注为预测/待测,
                        # 但 supported/refuted 之外的自造状态仍判红.
                        if k in contract.prediction_only and s in _PREDICTION_STATUSES:
                            continue
                        if k in contract.prediction_only:
                            bad_hypotheses.append(
                                f"{path}:hypotheses.{k} 必须是 supported/refuted "
                                "或 prediction/pending (该项无推翻条件)"
                            )
                        else:
                            bad_hypotheses.append(
                                f"{path}:hypotheses.{k} 缺失或不是 supported/refuted"
                            )

    if contract.contract_sha256:
        binding = root / _BINDING_FILE
        if binding.is_file():
            payload = _load_json(binding)
            got = ""
            if isinstance(payload, dict):
                got = str(payload.get("contract_sha256") or "").strip()
            if got != contract.contract_sha256:
                bad_binding.append(
                    f"{_BINDING_FILE}:contract_sha256 缺失或不等于契约字节指纹 "
                    f"{contract.contract_sha256[:12]}"
                )

    return {
        "passed": not (
            missing_keys or bad_types or short_lines or bad_lines
            or missing_fields or bad_hypotheses or bad_binding
        ),
        "missing_keys": missing_keys,
        "bad_types": bad_types,
        "short_lines": short_lines,
        "bad_lines": bad_lines,
        "missing_fields": missing_fields,
        "bad_hypotheses": bad_hypotheses,
        "bad_binding": bad_binding,
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
    if contract.hypothesis_range:
        allowed = "supported 或 refuted"
        if contract.prediction_only:
            allowed += (
                f" (无推翻条件的 {'/'.join(sorted(contract.prediction_only))} 也可写"
                " prediction/pending)"
            )
        parts.append(
            f"hypotheses 必须是对象且含 {'/'.join(contract.hypothesis_range)}, "
            f"每个取值只能是 {allowed}"
        )
    if contract.contract_sha256:
        parts.append(
            f"{_BINDING_FILE} 必须写入 contract_sha256: {contract.contract_sha256}"
            " (contract.json 字节指纹, 不一致即判失败)"
        )
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
            payload = _load_json(p)
            if payload is None:
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
