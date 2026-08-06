"""build_hint 长轨迹场景自检 — assert-based, no framework.

覆盖: progress 停滞→switch, 仍在推进→continue, 末尾 item 无候选,
progress 100%→空, requery 建议, 候选列表截断到 3.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, r"c:\Users\wanzh\Desktop\matsci-agent\agent")


class _Item:
    def __init__(self, name, label):
        self.name = name
        self.label = label


tmp = Path(tempfile.mkdtemp()) / "bandit_q.json"

from huginn.agent.bandit_controller import EffortBandit

EffortBandit._instance = None

# 6 个 item — 足够测候选列表截断
items = [_Item(f"task_{i}", f"PAT_{i % 3}") for i in range(6)]

# ── 1. 长轨迹 progress 停滞 → switch + 候选列表 ──────────────────
b = EffortBandit(persist_path=tmp)
b.set_items(items)
# state key: item 0, time_bucket 0(<30s), calls_bucket 3(>30), progress_bucket 1(1-25%)
_sk = "0|0|3|1"
b._Q[_sk] = {"continue": 0.01, "switch": 0.5, "requery": 0.1}
b._N[_sk] = {"continue": 5, "switch": 5, "requery": 5}
b.switch_item(0)
b._runtime.tool_calls = 35  # 长轨迹: 超过 warmup
b._runtime.last_progress_pct = 10.0  # 停滞在 10%
hint = b.build_hint()
assert "claiming a different item" in hint, f"expected switch hint, got: {hint[:200]}"
assert "item 2" in hint, f"missing candidate item 2: {hint[:300]}"
assert "item 4" in hint, f"missing candidate item 4: {hint[:300]}"
assert "item 5" not in hint, f"should only list 3 candidates, got item 5: {hint[:400]}"
assert "not limited above" in hint, f"missing autonomy text"
print("PASS 1: 长轨迹停滞 → switch + 3 候选 (不含第 4 个)")

# ── 2. 长轨迹 progress 仍在推进 → continue → 空字符串 ────────────
b2 = EffortBandit(persist_path=tmp)
b2.set_items(items)
_sk2 = "0|0|3|2"  # progress bucket 2 (25-75%)
b2._Q[_sk2] = {"continue": 0.8, "switch": 0.1, "requery": 0.05}
b2._N[_sk2] = {"continue": 5, "switch": 5, "requery": 5}
b2.switch_item(0)
b2._runtime.tool_calls = 35
b2._runtime.last_progress_pct = 50.0  # 还在推进
hint2 = b2.build_hint()
assert hint2 == "", f"progress moving should return empty, got: {hint2[:200]}"
print("PASS 2: 长轨迹仍在推进 → continue → 空字符串")

# ── 3. 末尾 item, switch 但无后续候选 ────────────────────────────
b3 = EffortBandit(persist_path=tmp)
b3.set_items(items)
_sk3 = "5|0|3|1"  # item 5 (最后一个)
b3._Q[_sk3] = {"continue": 0.01, "switch": 0.5, "requery": 0.1}
b3._N[_sk3] = {"continue": 5, "switch": 5, "requery": 5}
b3.switch_item(5)
b3._runtime.tool_calls = 35
b3._runtime.last_progress_pct = 10.0
hint3 = b3.build_hint()
assert "claiming a different item" in hint3, f"expected switch hint: {hint3[:200]}"
assert "(no further items)" in hint3, f"last item should have no candidates: {hint3[:300]}"
print("PASS 3: 末尾 item → switch + (no further items)")

# ── 4. progress 100% → continue → 空字符串 ─────────────────────
b4 = EffortBandit(persist_path=tmp)
b4.set_items(items)
b4.switch_item(0)
b4._runtime.tool_calls = 35
b4._runtime.last_progress_pct = 100.0
hint4 = b4.build_hint()
assert hint4 == "", f"progress 100% should return empty, got: {hint4[:200]}"
print("PASS 4: progress 100% → continue → 空字符串")

# ── 5. requery 建议 ─────────────────────────────────────────────
b5 = EffortBandit(persist_path=tmp)
b5.set_items(items)
_sk5 = "1|0|2|1"  # item 1, calls bucket 2(15-30), progress <25%
b5._Q[_sk5] = {"continue": 0.05, "switch": 0.05, "requery": 0.6}
b5._N[_sk5] = {"continue": 5, "switch": 5, "requery": 5}
b5.switch_item(1)
b5._runtime.tool_calls = 20
b5._runtime.last_progress_pct = 10.0
hint5 = b5.build_hint()
assert "stuck" in hint5, f"expected requery hint: {hint5[:200]}"
assert "alternative approach" in hint5, f"missing requery text: {hint5[:300]}"
assert "ignore this" in hint5, f"missing autonomy text: {hint5[:300]}"
print("PASS 5: requery 建议 → stuck hint")

# ── 6. warmup 期 (< 10 calls) → continue → 空字符串 ─────────────
b6 = EffortBandit(persist_path=tmp)
b6.set_items(items)
b6.switch_item(0)
b6._runtime.tool_calls = 5  # warmup
b6._runtime.last_progress_pct = 10.0
hint6 = b6.build_hint()
assert hint6 == "", f"warmup should return empty, got: {hint6[:200]}"
print("PASS 6: warmup (< 10 calls) → continue → 空字符串")

# ── 7. 候选列表 item 名称和 label 正确 ──────────────────────────
b7 = EffortBandit(persist_path=tmp)
b7.set_items(items)
_sk7 = "2|0|3|1"
b7._Q[_sk7] = {"continue": 0.01, "switch": 0.5, "requery": 0.1}
b7._N[_sk7] = {"continue": 5, "switch": 5, "requery": 5}
b7.switch_item(2)
b7._runtime.tool_calls = 35
b7._runtime.last_progress_pct = 10.0
hint7 = b7.build_hint()
assert "task_3" in hint7, f"missing task_3 name: {hint7[:300]}"
assert "[PAT_0]" in hint7, f"missing PAT_0 label: {hint7[:300]}"
assert "task_5" in hint7, f"missing task_5 name: {hint7[:400]}"
assert "[PAT_2]" in hint7, f"missing PAT_2 label: {hint7[:400]}"
print("PASS 7: 候选列表名称+label 正确")

print("\n=== ALL build_hint SELF-CHECKS PASSED ===")
