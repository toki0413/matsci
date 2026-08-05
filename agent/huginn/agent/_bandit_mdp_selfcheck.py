"""bandit MDP 升级 self-check — assert-based demo, no framework.

验证: (1) MC discounted return 正确回传; (2) flag off 回退旧单步增量更新;
(3) switch_item / end_episode 触发 flush.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"c:\Users\wanzh\Desktop\matsci-agent\agent")

from huginn.agent.bandit_controller import (
    EffortBandit, _BanditState, _ALPHA, _GAMMA,
)

tmp = Path(tempfile.mkdtemp()) / "bandit_mdp_q.json"
EffortBandit._instance = None
b = EffortBandit(persist_path=tmp)
b._mdp_enabled = True

# 1. MC return 正确性: 2 步轨迹 r=[0,1], terminal=0, gamma=0.9
#    G1 = 1 + gamma*0 = 1, G0 = 0 + gamma*1 = 0.9
_s0 = _BanditState(0, 1, 1, 1)
_s1 = _BanditState(0, 1, 1, 2)
b._record_step(_s0, "continue", 0.0)
b._record_step(_s1, "continue", 1.0)
b._flush_trajectory(0.0)
assert b._trajectory == [], "flush 后轨迹应清空"
assert abs(b._Q[_s0.key()]["continue"] - (_ALPHA * 0.9)) < 1e-6, \
    f"G0=0.9 回传失败: {b._Q}"
assert abs(b._Q[_s1.key()]["continue"] - _ALPHA) < 1e-6, \
    f"G1=1.0 回传失败: {b._Q}"
print("PASS 1: MC return 回传 (G0=0.9, G1=1.0)")

# 2. 同一 (s,a) 多次注入 reward 累加
b._record_step(_s0, "continue", 0.2)
b._record_step(_s0, "continue", 0.3)  # 同 state+action 应累加成一步 0.5
assert len(b._trajectory) == 1, f"应合并为一步: {b._trajectory}"
assert abs(b._trajectory[0][2] - 0.5) < 1e-6, f"reward 未累加: {b._trajectory}"
b._flush_trajectory(0.0)
print("PASS 2: 同 (s,a) reward 合并")

# 3. flag off -> 旧单步增量更新 (无轨迹缓冲)
b._mdp_enabled = False
b._Q.clear()
b._N.clear()
b._trajectory = []
b.set_items([])  # 空 items 让 progress proxy 走 0
b.switch_item(0)
b._runtime.tool_calls = 10
b.record_tool_call()
assert b._trajectory == [], "flag off 不应缓冲轨迹"
assert b._Q, "flag off 应立即更新 Q (incremental)"
b._mdp_enabled = True
print("PASS 3: flag off 回退旧增量更新")

# 4. switch_item flush 旧轨迹
b._trajectory = []
b._record_step(_s0, "continue", 0.5)
b.switch_item(1)
assert b._trajectory == [], "switch_item 应 flush 旧轨迹"
assert b._Q[_s0.key()]["continue"] != 0.0, "switch flush 应更新 Q"
print("PASS 4: switch_item flush")

# 5. end_episode flush
b._trajectory = []
b._record_step(_s1, "continue", 0.5)
b.end_episode()
assert b._trajectory == [], "end_episode 应 flush 轨迹"
print("PASS 5: end_episode flush")

# 6. 空轨迹 flush 不崩
b._trajectory = []
b._flush_trajectory(0.0)
b.end_episode()
print("PASS 6: 空轨迹 flush 不崩")

print("\n=== ALL BANDIT MDP SELF-CHECKS PASSED ===")