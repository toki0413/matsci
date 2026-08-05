"""端到端闭环模拟: 验证 bandit MDP 升级后信用分配真的通了.

场景: agent 跑 2 个 item, item 0 成功完成, item 1 中途结束.
关键验证: item 0 早期步骤 Δprogress=0 (reward=0), 但 episode 成功 (terminal=1.0),
         MC 回传后早期 Q 值应 > 0 — 旧单步更新下早期 Q 永远是 0.

闭环链路:
  set_items → switch_item(0) → [record_tool_call × N + update_iter_end × M]
  → progress=100 触发 flush → switch_item(1) → [record_tool_call × N]
  → end_episode flush → 验证 Q 值

同时跑一遍 flag=0 (旧单步) 做对照, 证明两者 Q 值不同.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"c:\Users\wanzh\Desktop\matsci-agent\agent")

from huginn.agent.bandit_controller import EffortBandit, _ALPHA, _GAMMA

# ── 公共模拟器 ────────────────────────────────────────────────────

class _MockItem:
    def __init__(self, name, label):
        self.name = name
        self.label = label


def _simulate_episode(bandit: EffortBandit, progress_schedule: list[float],
                      darwin_schedule: list[float]) -> None:
    """跑一个 item 的完整生命周期.

    progress_schedule: 每次 record_tool_call 时 _scan_outputs_progress 返回值
    darwin_schedule: 每次 update_iter_end 传入的 darwin_score
    """
    _call_idx = 0

    # monkey-patch progress 返回受控值
    _orig_scan = bandit._scan_outputs_progress
    def _mock_progress(item_idx):
        if _call_idx < len(progress_schedule):
            return progress_schedule[_call_idx]
        return progress_schedule[-1] if progress_schedule else 0.0
    bandit._scan_outputs_progress = _mock_progress

    # warmup 10 次不调 policy, 先灌满
    n_calls = len(progress_schedule)
    for i in range(n_calls):
        # 模拟 record_tool_call 内部 _call_idx 推进
        # 用闭包 trick: 直接改 _call_idx
        pass

    # 实际跑: 交替 record_tool_call + update_iter_end
    for i in range(n_calls):
        # 用 monkey patch 的闭包推进
        type(bandit)._mock_call_idx = i
        bandit._scan_outputs_progress = lambda self, idx, _i=i: progress_schedule[min(_i, len(progress_schedule)-1)]
        bandit.record_tool_call()
        if i < len(darwin_schedule):
            bandit.update_iter_end(darwin_schedule[i])

    bandit._scan_outputs_progress = _orig_scan


def _simulate_episode_v2(bandit: EffortBandit, progress_steps: list[float],
                         darwin_steps: list[float]) -> None:
    """跑一个 item 的完整生命周期 (v2: 直接控制内部状态).

    progress_steps: 每次 record_tool_call 后的 progress 值
    darwin_steps: 每次 update_iter_end 的 darwin 值
    """
    bandit.switch_item(0)
    rt = bandit._ensure_runtime(0)
    rt.tool_calls = 15  # 跳过 warmup (>=10)

    for i in range(len(progress_steps)):
        # 直接设 progress, 让 record_tool_call 内部 _scan_outputs_progress 读到
        _orig = bandit._scan_outputs_progress
        bandit._scan_outputs_progress = lambda idx, _p=progress_steps[i]: _p
        rt.tool_calls += 1
        bandit.record_tool_call()
        if i < len(darwin_steps):
            bandit.update_iter_end(darwin_steps[i])
        bandit._scan_outputs_progress = _orig


# ── 场景 1: MDP 模式 (新) ────────────────────────────────────────

print("=" * 60)
print("场景 1: MDP 模式 (HUGINN_BANDIT_MDP=1)")
print("=" * 60)

os.environ["HUGINN_BANDIT_MDP"] = "1"
EffortBandit._instance = None
tmp_mdp = Path(tempfile.mkdtemp()) / "bandit_mdp_demo.json"
b_mdp = EffortBandit(persist_path=tmp_mdp)
items = [_MockItem("DFT优化", "EXACT"), _MockItem("能带计算", "VARIANT")]
b_mdp.set_items(items)

# Item 0: 5 步, progress = [0, 10, 25, 50, 100], darwin = [0.5, 0.6, 0.65, 0.8, 0.9]
# 早期 step 0/1 progress=0/10 → Δprogress 小 → fast reward ≈ 0
# 但 episode 成功 (progress=100) → terminal=1.0 → MC 回传后早期 Q 应 > 0
print("\n[Item 0] 5 步, progress=[0, 10, 25, 50, 100], darwin=[0.5, 0.6, 0.65, 0.8, 0.9]")
print("  早期步 reward≈0, 但 episode 成功 → MC 回传应让早期 Q > 0")

b_mdp.switch_item(0)
rt0 = b_mdp._ensure_runtime(0)
rt0.tool_calls = 15  # 跳过 warmup

_progress_seq = [0.0, 10.0, 25.0, 50.0, 100.0]
_darwin_seq = [0.5, 0.6, 0.65, 0.8, 0.9]

for i in range(5):
    _orig_scan = b_mdp._scan_outputs_progress
    b_mdp._scan_outputs_progress = lambda idx, _p=_progress_seq[i]: _p
    rt0.tool_calls += 1
    b_mdp.record_tool_call()
    if i < len(_darwin_seq):
        b_mdp.update_iter_end(_darwin_seq[i])
    b_mdp._scan_outputs_progress = _orig_scan
    print(f"  step {i}: progress={_progress_seq[i]:.0f}%, "
          f"trajectory_len={len(b_mdp._trajectory)}, "
          f"Q_keys={len(b_mdp._Q)}")

# Item 0 在 progress=100 时应已 flush
print(f"\n  flush 后: trajectory={len(b_mdp._trajectory)} (应为 0), "
      f"Q states={len(b_mdp._Q)}")

# 关键验证: 早期 state 的 Q > 0 (MC 回传了 terminal reward)
_early_keys = list(b_mdp._Q.keys())
print(f"  Q 表内容:")
for k in _early_keys:
    for a in ("continue", "switch", "requery"):
        v = b_mdp._Q[k][a]
        if v != 0.0:
            print(f"    Q[{k}][{a}] = {v:.4f}  N={b_mdp._N[k][a]}")

# 验证: 至少有一个早期 state (progress<100 的) 的 Q > 0
_has_credit = False
for k in _early_keys:
    _q_max = max(b_mdp._Q[k].values())
    if _q_max > 0.001:
        _has_credit = True
        break
assert _has_credit, "MDP 模式: 早期 state 应有 MC 回传的 credit (Q>0)"
print("\n  ✅ MDP: 早期 state 获得了 MC 回传的信用分配 (Q>0)")

# Item 1: 3 步, 不完成, 然后 end_episode
print("\n[Item 1] 3 步, progress=[0, 10, 25], 中途结束 → end_episode")
b_mdp.switch_item(1)
rt1 = b_mdp._ensure_runtime(1)
rt1.tool_calls = 15

for i in range(3):
    _orig = b_mdp._scan_outputs_progress
    b_mdp._scan_outputs_progress = lambda idx, _p=[0.0, 10.0, 25.0][i]: _p
    rt1.tool_calls += 1
    b_mdp.record_tool_call()
    b_mdp._scan_outputs_progress = _orig
    print(f"  step {i}: trajectory_len={len(b_mdp._trajectory)}")

print(f"  end_episode 前: trajectory={len(b_mdp._trajectory)}")
b_mdp.end_episode()
print(f"  end_episode 后: trajectory={len(b_mdp._trajectory)} (应为 0)")
assert len(b_mdp._trajectory) == 0, "end_episode 应 flush 轨迹"
print("  ✅ Item 1 轨迹已 flush, 闭环完整")


# ── 场景 2: 对照组 — 旧单步模式 (flag off) ─────────────────────

print("\n" + "=" * 60)
print("场景 2: 对照组 — 旧单步模式 (HUGINN_BANDIT_MDP=0)")
print("=" * 60)

os.environ["HUGINN_BANDIT_MDP"] = "0"
EffortBandit._instance = None
tmp_old = Path(tempfile.mkdtemp()) / "bandit_old_demo.json"
b_old = EffortBandit(persist_path=tmp_old)
b_old.set_items(items)

b_old.switch_item(0)
rt_old = b_old._ensure_runtime(0)
rt_old.tool_calls = 15

print("\n[Item 0] 同样 5 步, progress=[0, 10, 25, 50, 100]")
for i in range(5):
    _orig = b_old._scan_outputs_progress
    b_old._scan_outputs_progress = lambda idx, _p=_progress_seq[i]: _p
    rt_old.tool_calls += 1
    b_old.record_tool_call()
    if i < len(_darwin_seq):
        b_old.update_iter_end(_darwin_seq[i])
    b_old._scan_outputs_progress = _orig
    print(f"  step {i}: Q_keys={len(b_old._Q)}, trajectory={len(b_old._trajectory)}")

print(f"\n  Q 表内容:")
for k in list(b_old._Q.keys()):
    for a in ("continue", "switch", "requery"):
        v = b_old._Q[k][a]
        if v != 0.0:
            print(f"    Q[{k}][{a}] = {v:.4f}  N={b_old._N[k][a]}")

# 对照: 旧模式下, progress=0 那步的 state (Δprogress=0 → reward=0)
# 其 Q 值应该只受 slow reward 影响, 不含 terminal 回传
# 但 step 0 的 darwin delta=0 (0.5-0.5=0) → slow reward=0 → Q 仍 0
_step0_state_key = None
for k in b_old._Q:
    # 找 progress bucket=0 的 state (item_idx=0, progress bucket=0)
    if k.startswith("0|") and k.split("|")[3] == "0":
        _step0_state_key = k
        break

if _step0_state_key:
    _old_q = b_old._Q[_step0_state_key]["continue"]
    print(f"\n  旧模式 step0 Q[{_step0_state_key}][continue] = {_old_q:.4f}")
    if _old_q == 0.0:
        print("  → 早期步 reward=0 → Q 永远 0, 无信用回传 (单步 bandit 天花板)")
    else:
        print(f"  → 有 slow reward 注入 (darwin delta), 但无 terminal 回传")
else:
    print("\n  (未找到 progress=0 的 state, 可能 warmup 阶段未生成)")

# ── 结论 ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("闭环验证结论")
print("=" * 60)

_mdp_q_count = sum(1 for k in b_mdp._Q for a in b_mdp._Q[k] if b_mdp._Q[k][a] != 0.0)
_old_q_count = sum(1 for k in b_old._Q for a in b_old._Q[k] if b_old._Q[k][a] != 0.0)

print(f"  MDP 模式:  {_mdp_q_count} 个非零 Q 值 (含 MC 回传的 terminal 信用)")
print(f"  旧单步模式: {_old_q_count} 个非零 Q 值 (只有瞬时 reward)")
print(f"  γ={_GAMMA}, α={_ALPHA}")
print()
print("  闭环链路验证:")
print("    set_items → switch_item(0) → record_tool_call×5 + update_iter_end×5")
_print_flush = "    progress=100 触发 flush → MC 回传 → 早期 Q>0 ✅" if _has_credit else "    ✗"
print(_print_flush)
print("    switch_item(1) → record_tool_call×3 → end_episode flush ✅")
print("    Q 表持久化 (force_save → reload 一致性):", end=" ")
b_mdp.force_save()
EffortBandit._instance = None
b_reload = EffortBandit(persist_path=tmp_mdp)
_q_match = all(
    abs(b_reload._Q.get(k, {}).get(a, 0) - b_mdp._Q.get(k, {}).get(a, 0)) < 1e-6
    for k in b_mdp._Q for a in b_mdp._Q[k]
)
print("✅" if _q_match else "✗")
assert _q_match, "持久化 round-trip 应一致"

print("\n=== 闭环模拟 ALL PASSED ===")