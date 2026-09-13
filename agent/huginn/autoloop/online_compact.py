"""online_compact —— Huginn 版 SoL-Pi「Online Context Compact」（落地④号）。

现状: ``agent/streaming.py::_maybe_auto_compact`` 按上下文压力(>60%)**全局**触发
压缩, 不问哪些内容已完成可压、哪些是未验证需要留证据。SoL-Pi 的关键升级是把
压缩变成**候选驱动 + 门控**:

  - 只把「已完成 + 已验证 + 证据已密封」的计划步骤当作压缩候选;
  - 未验证/进行中的步骤**fail-closed 保留证据**, 绝不折叠;
  - 门控受两个经济/窗口检查约束: 窗口压力(不设全局硬顶而用实测占用) +
    缓存读写比(重复读取少→压缩收益低, 不应压)。

本模块是**纯函数决策层**, 对任意 plan-step 对象用 duck-typing(``.id/.verified/
.status/.sealed``)工作, 不绑死 autoloop 数据结构 → 可独立单测、可被 streaming /
内存压缩 / 未来压缩策略共同复用。不写文件、不调 LLM。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# 默认经济门: 上下文占用(窗口压力)超过该值才考虑压缩
_DEFAULT_WINDOW_GATE_PCT = 60
# 默认缓存读写比阈值: 重复读取/重放的相对收益低于该比则不值得压 (SoL-Pi
# cacheWriteReadRatio 12.5 的工程默认)
_DEFAULT_CACHE_WRITE_READ_RATIO = 12.5

# status/verified 判定时认可的已完结词表
_DONE_WORDS = frozenset(
    {"completed", "complete", "done", "verified", "closed", "sealed", "passed"}
)
# 明确"未完成/进行中"的信号, 优先级高于 _DONE_WORDS (避免把 in_progress 当完成)
_ACTIVE_WORDS = frozenset(
    {"in_progress", "active", "running", "pending", "todo", "planned", "unverified"}
)


@dataclass
class CompactCandidate:
    """一个可安全折叠进摘要的已完结步骤."""

    step_id: str
    reason: str  # 为什么可压 (verified + sealed + 经济门通过)


def step_is_verified(step: Any) -> bool:
    """duck-typing: 判一个 plan step 是否已验证完成.

    优先读 ``.verified``; 否则读 ``.status``/``.step_status``; 还可接受
    ``.evidence`` 存在即视为已留证据。显式 active 信号覆盖 done 词表。
    """
    verified = getattr(step, "verified", None)
    if verified is not None:
        if isinstance(verified, str):
            return verified.lower().strip() in _DONE_WORDS
        return bool(verified)

    st = getattr(step, "status", None) or getattr(step, "step_status", None) or ""
    s = str(st).strip().lower()
    if not s:
        return False
    if s in _ACTIVE_WORDS:
        return False
    return s in _DONE_WORDS


def step_is_sealed(step: Any) -> bool:
    """证据已密封(有可核实的归档/证据引用), 否则不该压."""
    sealed = getattr(step, "sealed", None)
    if sealed is not None:
        return bool(sealed)
    # 无 sealed 字段时, 有 evidence / evidence_ref 视为已留证据
    return hasattr(step, "evidence") or hasattr(step, "evidence_ref")


def _step_id(step: Any) -> str:
    return str(getattr(step, "id", None) or getattr(step, "step_id", "") or id(step))


def select_compact_candidates(
    steps: list[Any],
    *,
    window_pct: float = _DEFAULT_WINDOW_GATE_PCT,
    cache_write_read_ratio: float = _DEFAULT_CACHE_WRITE_READ_RATIO,
) -> list[CompactCandidate]:
    """返回**可安全压缩**的候选步骤(已验证 + 已密封)。

    仅纯筛选, 不含门控——门控由调用方用 gate_compaction 决定是否真的触发
    压缩动作。未验证/未密封的步骤一律不进候选(fail-closed 保留证据)。
    """
    out: list[CompactCandidate] = []
    for step in steps:
        if not step_is_verified(step):
            continue
        if not step_is_sealed(step):
            continue
        out.append(CompactCandidate(step_id=_step_id(step), reason="verified+sealed"))
    return out


def gate_compaction(
    steps: list[Any],
    *,
    window_pct: float = _DEFAULT_WINDOW_GATE_PCT,
    cache_write_read_ratio: float = _DEFAULT_CACHE_WRITE_READ_RATIO,
) -> dict[str, Any]:
    """完整的候选驱动 + 经济/密封门控决策.

    返回 dict:
      compact_candidates : 可压步骤 id 列表
      preserve_evidence  : 必须保留证据的步骤 id(未验证 或 未密封)
      should_compact     : 是否值得触发压缩动作
      reasons            : [{kind, message}] 门控说明

    should_compact 判定:
      1) 存在已验证+已密封候选(候选驱动);
      2) 且 (窗口压力 ≥ 阈值 或 缓存读写比 ≥ 阈值)(经济/窗口门)。
    """
    eligible = select_compact_candidates(
        steps,
        window_pct=window_pct,
        cache_write_read_ratio=cache_write_read_ratio,
    )
    reasons: list[dict[str, str]] = []
    preserve: list[str] = []
    for step in steps:
        if not step_is_verified(step):
            preserve.append(_step_id(step))
            reasons.append({"kind": "preserve", "message": f"{_step_id(step)} 未验证"})
        elif not step_is_sealed(step):
            preserve.append(_step_id(step))
            reasons.append({"kind": "preserve", "message": f"{_step_id(step)} 未密封"})

    pressure_ok = window_pct >= _DEFAULT_WINDOW_GATE_PCT
    economy_ok = cache_write_read_ratio >= _DEFAULT_CACHE_WRITE_READ_RATIO
    should_compact = bool(eligible) and (pressure_ok or economy_ok)
    if eligible:
        reasons.append(
            {"kind": "economy", "message": (
                f"eligible={len(eligible)} pressure={pressure_ok} "
                f"ratio={cache_write_read_ratio:g}>=t ratio_ok={economy_ok}"
            )}
        )
    return {
        "compact_candidates": [c.step_id for c in eligible],
        "preserve_evidence": preserve,
        "should_compact": should_compact,
        "reasons": reasons,
    }


def compact_preserved_summary_delta(
    steps: list[Any], *, prefix: str = "[Verified-completed steps folded]"
) -> str:
    """生成供摘要附加的折叠增量: 只把已验证+已密封步骤压成一行计数。

    未验证步骤不进增量 → 其证据仍由上下文/归档保留(fail-closed)。
    返回空串 = 无可折叠步骤(调用方不应据此调低 budget)。
    """
    eligible = select_compact_candidates(steps)
    if not eligible:
        return ""
    folded = ", ".join(c.step_id for c in eligible[:8])
    if len(eligible) > 8:
        folded += ", ..."
    return f"{prefix} ({len(eligible)}): [{folded}]"


def _fake_step(step_id: str, *, verified: Any = None, status: str = "", sealed: Any = None,
               has_evidence: bool = False) -> Any:
    class _S:
        pass

    s = _S()
    s.id = step_id
    s.verified = verified  # 传 None 则走 status 判定
    s.status = status
    s.sealed = sealed
    s.evidence = "E" if has_evidence else None
    return s


def _selfcheck() -> None:
    done_sealed = _fake_step("s1", status="completed", has_evidence=True)
    done_unsealed = _fake_step("s2", status="completed", sealed=False)
    active = _fake_step("s3", status="in_progress")
    v_explicit = _fake_step("s4", verified=True, sealed=True)
    unverified = _fake_step("s5", status="unverified", has_evidence=True)

    steps = [done_sealed, done_unsealed, active, v_explicit, unverified]
    eligible = select_compact_candidates(steps)
    ids = {c.step_id for c in eligible}
    assert ids == {"s1", "s4"}, f"expected candidates s1/s4, got {ids}"

    g = gate_compaction(steps, window_pct=70, cache_write_read_ratio=20)
    assert g["should_compact"] is True, g
    assert set(g["compact_candidates"]) == {"s1", "s4"}, g
    assert "s3" in g["preserve_evidence"] and "s5" in g["preserve_evidence"]

    # 经济门不过 → 不压
    g2 = gate_compaction(steps, window_pct=30, cache_write_read_ratio=1)
    assert g2["should_compact"] is False, g2

    # 无候选 → 不压, 增量空
    g3 = gate_compaction([active, unverified], window_pct=95, cache_write_read_ratio=100)
    assert g3["should_compact"] is False
    assert compact_preserved_summary_delta([active, unverified]) == ""

    delta = compact_preserved_summary_delta([done_sealed, active])
    assert "s1" in delta and "s3" not in delta, delta
    print("OK online_compact self-check passed (candidate-driven + economy/seal gating)")


if __name__ == "__main__":
    _selfcheck()