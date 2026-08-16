"""细粒度权限判定测试 (M5).

覆盖 PermissionChecker 的多阶段判定:
  1. 危险命令 (CRITICAL, 即使 auto_approve 也拦)
  2. 路径规则 (工具×路径矩阵)
  3. 工具基础规则 → 五档风险推断
  4. 成本分级 (超预算升 ASK / HIGH)
  5. 信任自适应 (高信任放行 medium, 低信任强制 ASK)
以及 StandingRulesStore 的 (tool, target) 常驻授权.
"""

from huginn.core_types import PermissionMode, RiskLevel
from huginn.permissions import (
    PermissionChecker,
    PermissionConfig,
    StandingRulesStore,
    get_standing_rules_store,
    reset_standing_rules_store,
    reset_trust,
)


def _checker(**kw) -> PermissionChecker:
    return PermissionChecker(PermissionConfig(**kw))


# ── 阶段1: 危险命令 ─────────────────────────────────────────────
async def test_dangerous_command_always_asks_even_auto_approve():
    c = _checker(auto_approve_all=True)
    r = await c.check("bash_tool", args={"command": "git push --force origin main"})
    assert r.mode == PermissionMode.ASK
    assert r.risk_level == RiskLevel.CRITICAL
    assert any(x.startswith("dangerous:") for x in r.matched_rules)


async def test_file_delete_is_dangerous():
    c = _checker()
    r = await c.check("file_delete_tool", args={"file_path": "/tmp/a.txt"})
    assert r.mode == PermissionMode.ASK
    assert r.risk_level == RiskLevel.CRITICAL


# ── 阶段2: 路径规则 ─────────────────────────────────────────────
async def test_path_deny_is_critical():
    c = _checker(path_rules=[("secrets/*", PermissionMode.DENY)])
    r = await c.check("file_write_tool", args={"file_path": "secrets/token.json"})
    assert r.mode == PermissionMode.DENY
    assert r.risk_level == RiskLevel.CRITICAL
    assert "path:deny" in r.matched_rules


async def test_path_ask_is_medium():
    c = _checker(path_rules=[("data/*", PermissionMode.ASK)])
    r = await c.check("file_write_tool", args={"file_path": "data/run1.in"})
    assert r.mode == PermissionMode.ASK
    assert r.risk_level == RiskLevel.MEDIUM


async def test_tool_path_matrix_only_matches_matching_tool():
    c = _checker(path_rules=[("file_write_tool", "/tmp/*", PermissionMode.ASK)])
    # 命中工具 → ASK
    r = await c.check("file_write_tool", args={"file_path": "/tmp/x.txt"})
    assert r.mode == PermissionMode.ASK
    # 其它工具写同一路径 → 不命中矩阵, 走基础规则
    r2 = await c.check("file_edit_tool", args={"file_path": "/tmp/x.txt"})
    assert "path:ask" not in r2.matched_rules


# ── 阶段3: 五档风险推断 ─────────────────────────────────────────
async def test_auto_tool_is_none_risk():
    c = _checker()
    r = await c.check("structure_tool")
    assert r.mode == PermissionMode.AUTO
    assert r.risk_level == RiskLevel.NONE


async def test_ask_tool_is_medium_risk():
    c = _checker()
    r = await c.check("vasp_tool")
    assert r.mode == PermissionMode.ASK
    assert r.risk_level == RiskLevel.MEDIUM


async def test_deny_tool_is_critical_risk():
    c = _checker()
    r = await c.check("system_shell_tool")
    assert r.mode == PermissionMode.DENY
    assert r.risk_level == RiskLevel.CRITICAL


# ── 阶段4: 成本分级 ─────────────────────────────────────────────
async def test_over_budget_auto_escalates_to_ask_and_high():
    c = _checker(cost_budget_hours=1.0)
    r = await c.check(
        "vasp_tool",
        cost_estimate={"cpu_hours": 50.0},
    )
    assert r.mode == PermissionMode.ASK
    assert r.risk_level == RiskLevel.HIGH
    assert r.cost_hours == 50.0
    assert any(x.startswith("cost:") for x in r.matched_rules)


async def test_within_budget_stays_unchanged():
    c = _checker(cost_budget_hours=10.0)
    r = await c.check("vasp_tool", cost_estimate={"cpu_hours": 2.0})
    assert r.mode == PermissionMode.ASK
    assert r.risk_level == RiskLevel.MEDIUM


# ── 阶段5: 信任自适应 ───────────────────────────────────────────
async def test_high_trust_auto_approves_medium():
    reset_trust()
    c = _checker(trust_adaptive=True)
    # 模拟多次批准把 trust 抬到 0.7 以上
    s = "sess-high"
    for _ in range(12):
        c = _checker(trust_adaptive=True)
        from huginn.permissions import _record_approval
        _record_approval(s, "approve")
    r = await c.check("vasp_tool", session_id=s)
    assert r.mode == PermissionMode.AUTO
    assert any(x.startswith("trust:") for x in r.matched_rules)


async def test_low_trust_keeps_medium_ask():
    reset_trust()
    c = _checker(trust_adaptive=True)
    s = "sess-low"
    from huginn.permissions import _record_approval
    for _ in range(4):
        _record_approval(s, "deny")
    r = await c.check("vasp_tool", session_id=s)
    # 低信任: medium 风险工具保持 ASK, 绝不被自动放行
    assert r.mode == PermissionMode.ASK
    assert r.risk_level == RiskLevel.MEDIUM


async def test_trust_does_not_auto_approve_high_or_critical():
    reset_trust()
    c = _checker(trust_adaptive=True)
    s = "sess-high2"
    from huginn.permissions import _record_approval
    for _ in range(12):
        _record_approval(s, "approve")
    # high 风险 (bash 含副作用) 即便高信任也不放行
    r = await c.check("bash_tool", args={"command": "rm -rf /tmp/x"}, session_id=s)
    assert r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert r.mode == PermissionMode.ASK or r.mode == PermissionMode.DENY


# ── StandingRulesStore: (tool, target) 常驻授权 ─────────────────
def test_standing_rule_grant_and_match():
    store = StandingRulesStore()
    store.grant("s1", "file_write_tool", "/tmp/*")
    assert store.is_granted("s1", "file_write_tool", "/tmp/a.txt")
    assert not store.is_granted("s1", "file_write_tool", "/etc/passwd")
    assert not store.is_granted("s1", "file_edit_tool", "/tmp/a.txt")


def test_standing_rule_wildcard_target():
    store = StandingRulesStore()
    store.grant("s1", "bash_tool")
    assert store.is_granted("s1", "bash_tool", "anything")


def test_standing_rule_singleton_reset():
    reset_standing_rules_store()
    store = get_standing_rules_store()
    store.grant("s1", "file_write_tool", "/tmp/*")
    assert store.list_rules("s1")
    reset_standing_rules_store()
    assert not get_standing_rules_store().list_rules("s1")
