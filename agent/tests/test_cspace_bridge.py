"""deep_think → CSpace 桥 (spec §10 测试 1-4 与 6 + M2 访问器).

红线验证: deep_think 草稿 ≠ 在场 —— 草稿只是 pending 候选, 有证据证实(promote)
才成为可引用 falsifiable 在场。
"""
import json

from huginn.memory.manager import MemoryManager
from huginn.memory.reasoning import ReasoningPhase, ReasoningRecord
from huginn.research import cspace_bridge as cbridge
from huginn.research.cspace import CSpace


def _readout_ids(cs: CSpace) -> set[str]:
    return {x["id"] for x in cs.readout()["at_hand"]}


def test_enqueue_makes_candidate_not_at_hand():
    """测试 1: enqueue → candidate(falsifiable=False) 在 pending_no_source, 不在 at_hand."""
    cs = CSpace()
    b = cbridge.enqueue_deliberation(
        cs,
        ReasoningRecord(claim="带隙预计 1.5 eV", phase=ReasoningPhase.PRE_ACTION,
                        estimate="1.5 eV"),
    )
    assert b.falsifiable is False
    assert b.id.startswith("dt_")
    assert b.phase == "pre_action"
    # 草稿只是候选, 绝不在可引用 at_hand
    ro = cs.readout()
    assert b.id in ro["pending_no_source"]
    assert b.id not in _readout_ids(cs)


def test_promote_needs_trace_truth():
    """测试 2: 无证据 → 保持 candidate; 带 trace 真值 → confirmed 进 at_hand."""
    cs = CSpace()
    b = cbridge.enqueue_deliberation(
        cs,
        ReasoningRecord(claim="带隙预测为 1.5 eV", phase=ReasoningPhase.THINK),
    )
    # 无证据: 默认门禁不通过 → 保持 candidate, pending 计数 +1
    r1 = cbridge.promote_to_at_hand(cs, b.id)
    assert r1["promoted"] is False
    assert r1["state"] == "candidate"
    assert b.falsifiable is False
    assert cbridge.governance(cs)["pending"] == 1
    # 补 trace 真值 (同 grounding 门禁同源) → 证实
    cs.trace.append(json.dumps({"type": "cspace_broadcast", "claim": "gap 1.5 eV"},
                               ensure_ascii=False))
    r2 = cbridge.promote_to_at_hand(cs, b.id)
    assert r2["promoted"] is True
    assert r2["state"] == "confirmed"
    assert b.falsifiable is True
    # 点亮 (pin) 后进 at_hand —— confirmed 才可被 probe 点亮为可引用在场
    cs.pin(b.id)
    cs.probe("gap")
    assert b.id in _readout_ids(cs)
    assert cbridge.governance(cs)["confirmed"] == 1
    # confirmed 是可引用集合成员
    assert b.id in {x.id for x in cbridge.confirmed_at_hand(cs)}


def test_estimate_reconcile_promote_or_reject():
    """测试 3: pre_action 预言 + 真实执行 reconcile → 吻合 promoted; 偏差 rejected."""
    cs = CSpace()
    # 吻合: 预判扩散系数 1.0e-9, 真实执行 1.0e-9 → 容差内 → 强在场
    b = cbridge.enqueue_deliberation(
        cs,
        ReasoningRecord(claim="扩散系数预计 1e-9", phase=ReasoningPhase.PRE_ACTION,
                        estimate="1.0e-9"),
    )
    r = cbridge.promote_to_at_hand(cs, b.id,
                                   corroborate=cbridge.reconcile_estimate(1.0e-9, tol=0.05))
    assert r["promoted"] is True
    assert b.falsifiable is True
    assert b.payload.get("reconcile", {}).get("borne_out") is True
    # 偏差: 预判 5.0e-9, 真实 1.0e-9 → 不符 → rejected 进治理计数
    c = cbridge.enqueue_deliberation(
        cs,
        ReasoningRecord(claim="扩散系数预计 5e-9", phase=ReasoningPhase.PRE_ACTION,
                        estimate="5.0e-9"),
    )
    r2 = cbridge.promote_to_at_hand(cs, c.id,
                                    corroborate=cbridge.reconcile_estimate(1.0e-9, tol=0.05))
    assert r2["promoted"] is False
    assert r2["state"] == "rejected"
    assert c.falsifiable is False
    gov = cbridge.governance(cs)
    assert gov["confirmed"] == 1
    assert gov["rejected"] == 1
    # 被拒草稿绝不可引用
    assert c.id not in {x.id for x in cbridge.confirmed_at_hand(cs)}


def test_ingest_batch_dedup_empty_id_stable():
    """测试 4: ingest 批处理 —— 幂等去重、空草稿计空、id 稳定."""
    cs = CSpace()
    recs = [
        ReasoningRecord(claim="A 相稳定, 能量 -3.5 eV", phase=ReasoningPhase.THINK),
        ReasoningRecord(claim="B 相不稳定, 能量 +1.2 eV", phase=ReasoningPhase.THINK),
        "   ",  # 纯空白 → rejected_empty
        ReasoningRecord(claim="A 相稳定, 能量 -3.5 eV", phase=ReasoningPhase.THINK),
        "",  # 空 → rejected_empty
    ]
    beings = cbridge.ingest_reasoning_trace(cs, recs)
    assert len(beings) == 2
    # 幂等去重: A 重复想也只一条; 空白不入场
    assert len(cs.beings) == 2
    a = [x for x in cs.beings.values() if "A 相" in x.payload["claim"]][0]
    # 空草稿计入治理账本
    gov = cbridge.governance(cs)
    assert gov["rejected_empty"] == 2
    # 空草稿不进任何 confirmed
    assert cbridge.confirmed_at_hand(cs) == []
    # id 稳定幂等
    b = cbridge.enqueue_deliberation(
        cs,
        ReasoningRecord(claim="A 相稳定, 能量 -3.5 eV", phase=ReasoningPhase.THINK),
    )
    assert b.id == a.id
    assert len(cs.beings) == 2
    # 再入同批次 → 仍 2 条 (幂等去重; 空草稿每次仍计入 rejected_empty)
    assert len(cbridge.ingest_reasoning_trace(cs, recs)) == 0 or len(cs.beings) == 2
    assert len(cs.beings) == 2


def test_fail_open_memory_none():
    """测试 6: memory=None → 入桥 no-op, 不抛."""
    cs = CSpace()
    assert cbridge.ingest_from_memory(cs, None) == []
    assert cbridge.ingest_from_memory(None, None) == []
    assert len(cs.beings) == 0


def test_m2_iter_reasoning_records_and_ingest_from_memory():
    """M2: memory_manager.iter_reasoning_records() 只读读取 + ingest_from_memory 接线."""
    mm = MemoryManager()
    mm.add_reasoning_record(
        ReasoningRecord(claim="PBE 更准, gap 3.4 eV", phase=ReasoningPhase.THINK)
    )
    mm.add_reasoning_record(
        ReasoningRecord(claim="LDA 低估, gap 1.2 eV", phase=ReasoningPhase.THINK)
    )
    records = list(mm.iter_reasoning_records())
    assert len(records) == 2
    assert records[0].claim == "PBE 更准, gap 3.4 eV"
    cs = CSpace()
    beings = cbridge.ingest_from_memory(cs, mm)
    assert len(beings) == 2
    assert len(cs.beings) == 2
    # 访问器缺位 → 无桥行为 (不抛)
    class _NoIter:
        pass
    assert cbridge.ingest_from_memory(cs, _NoIter()) == []
    assert len(cs.beings) == 2
