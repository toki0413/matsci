"""JEV 接入接线契约单测.

覆盖:
  - 开关层 (_enabled): 隐私外发闸 + jev_enabled 两级 short-circuit, 默认全关.
  - 客户端 (JevClient): Noul batch 解析 / 无 key fail-open / 传输异常 fail-open.
  - 工具子集扩展 (jev_expand_subset): 按阈值+置信度挑选, 排除常驻 CORE.
  - compute_effective_subset 集成: keyword 无命中时才走 JEV, 命中则不调.
  - gate.jev_adapter: review→pass(advisory), block→block.

默认全关, 均为确定性/零网络. 关键外部依赖 (JEV 传输/开关) 一律 mock.
"""
from __future__ import annotations

from huginn.runtime.jev import _enabled as jev_enabled_mod
from huginn.runtime.jev.client import JevClient
from huginn.security.gate import GateChain, jev_adapter


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeTransport:
    """可注入 transport: 记录请求, 按调用返回给定响应/抛出."""

    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc
        self.calls: list[tuple] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        if self._exc is not None:
            raise self._exc
        return _FakeResp(self._payload)


# ── 开关层 ─────────────────────────────────────────────────────


def test_egress_blocked_by_local_only(monkeypatch):
    monkeypatch.setattr(jev_enabled_mod, "_privacy_level", lambda: "local_only")
    assert jev_enabled_mod.jev_egress_allowed() is False
    # jev_enabled 触发短路: 即便开关开着也禁用
    monkeypatch.setattr(jev_enabled_mod, "_flag_enabled", lambda key, default=False: True)
    assert jev_enabled_mod.jev_enabled("jev_tool_router") is False


def test_egress_blocked_by_redact(monkeypatch):
    monkeypatch.setattr(jev_enabled_mod, "_privacy_level", lambda: "redact")
    assert jev_enabled_mod.jev_egress_allowed() is False


def test_egress_allowed_on_off_tier(monkeypatch):
    monkeypatch.setattr(jev_enabled_mod, "_privacy_level", lambda: "off")
    assert jev_enabled_mod.jev_egress_allowed() is True
    # 隐私允许 + 开关开 → 启用
    monkeypatch.setattr(jev_enabled_mod, "_flag_enabled", lambda key, default=False: True)
    assert jev_enabled_mod.jev_enabled("jev_tool_router") is True


def test_jev_disabled_by_default_flag(monkeypatch):
    monkeypatch.setattr(jev_enabled_mod, "_privacy_level", lambda: "off")
    monkeypatch.setattr(jev_enabled_mod, "_flag_enabled", lambda key, default=False: False)
    assert jev_enabled_mod.jev_enabled("jev_tool_router") is False


# ── 客户端 ─────────────────────────────────────────────────────


def test_noul_batch_parses(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k-test")
    transport = _FakeTransport(
        payload={
            "answers": {
                "vasp_tool": {"noul": 0.91, "confidence": 0.8},
                "lammps_tool": {"noul": 0.1, "confidence": 0.9},
            }
        }
    )
    c = JevClient(transport=transport)
    out = c.noul_batch({"task": "x"}, {"vasp_tool": "rel?", "lammps_tool": "rel?"})
    assert out["vasp_tool"]["p"] == 0.91
    assert out["lammps_tool"]["p"] == 0.1
    # 请求里带上了 questions {type: noul}
    req = transport.calls[0]
    assert req["json"]["questions"]["vasp_tool"]["type"] == "noul"


def test_client_fail_open_without_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_MANAGED_KEY", raising=False)
    transport = _FakeTransport(payload={"answers": {}})
    c = JevClient(transport=transport)
    assert c.available is False
    assert c.noul_batch({"task": "x"}, {"a": "rel?"}) is None  # fail-open
    assert transport.calls == []  # 没 key 不发请求


def test_client_fail_open_on_transport_error(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k-test")
    c = JevClient(transport=_FakeTransport(exc=RuntimeError("down")))
    assert c.noul_batch({"task": "x"}, {"a": "rel?"}) is None
    assert c.noul({"task": "x"}, "rel?") is None


def test_client_parse_malformed_answers(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k-test")
    c = JevClient(transport=_FakeTransport(payload={"something": "weird"}))
    assert c.noul_batch({"task": "x"}, {"a": "rel?"}) == {}


# ── 工具子集扩展 ───────────────────────────────────────────────


def test_jev_expand_subset_requires_flag_and_client(monkeypatch):
    import huginn.runtime.jev.tool_router as tr

    monkeypatch.setattr(tr, "jev_enabled", lambda key: False)
    picked, meta = tr.jev_expand_subset("some task", ["vasp_tool", "analysis_tool"])
    assert picked == [] and meta["eval"] is False


def test_jev_expand_subset_picks_by_threshold(monkeypatch):
    import huginn.runtime.jev.tool_router as tr

    monkeypatch.setattr(tr, "jev_enabled", lambda key: True)
    # 给候选工具一律返回描述, 避免依赖本机 ToolRegistry 是否注册这些工具
    monkeypatch.setattr(tr, "_tool_description", lambda name: f"用法描述:{name}")

    class _Fake:
        available = True

        def noul_batch(self, state, instructions):
            # 只对 vasp_tool 高置信命中, lammps_tool 未达阈值, analysis_tool 在 CORE 不该进候选
            return {
                "vasp_tool": {"p": 0.9, "confidence": 0.8},
                "lammps_tool": {"p": 0.55, "confidence": 0.7},
            }

    picked, meta = tr.jev_expand_subset(
        "novel heterogeneous task", ["vasp_tool", "lammps_tool", "analysis_tool"],
        client=_Fake(),
    )
    assert picked == ["vasp_tool"]
    assert meta["eval"] is True


def test_jevn_expand_subset_skips_analysis_when_core(monkeypatch):
    import huginn.runtime.jev.tool_router as tr

    monkeypatch.setattr(tr, "jev_enabled", lambda key: True)

    class _Fake:
        available = True

        def noul_batch(self, state, instructions):
            return {"analysis_tool": {"p": 0.99, "confidence": 0.9}}

    # analysis_tool 是常驻 CORE, 不出现在候选 → 不会被重复加入指导结果
    picked, _ = tr.jev_expand_subset(
        "t", ["analysis_tool"], client=_Fake()
    )
    assert all(t != "analysis_tool" for t in picked)


# ── compute_effective_subset 集成 ──────────────────────────────


def test_subset_known_domain_keeps_rule_and_core(monkeypatch):
    import huginn.runtime.task_tool_router as ttr

    monkeypatch.setattr(ttr, "CORE_TOOL_NAMES", ["file_read_tool", "analysis_tool"])
    # 已知域 "vasp ..." 命中 → 规则子集 + core, 不触发任何 JEV (默认 flag off)
    out = ttr.compute_effective_subset(
        "compute band structure of Si",
        ["vasp_tool", "file_read_tool", "analysis_tool"],
    )
    assert "vasp_tool" in out
    assert "file_read_tool" in out


def test_subset_unknown_domain_expands_via_jev(monkeypatch):
    import huginn.runtime.jev.tool_router as jtr
    import huginn.runtime.task_tool_router as ttr

    # 让 compute_effective_subset 内部 import 到的 jev_expand_subset 返回额外工具
    monkeypatch.setattr(
        jtr, "jev_expand_subset",
        lambda msg, avail, **kw: (["knowledge_tool"], {"source": "jev:advisory"}),
    )
    monkeypatch.setattr(ttr, "CORE_TOOL_NAMES", ["file_read_tool", "analysis_tool"])
    out = ttr.compute_effective_subset(
        "this is a totally novel unknown request with no keyword",
        ["file_read_tool", "analysis_tool", "knowledge_tool"],
    )
    assert "knowledge_tool" in out


# ── gate 归一化 ─────────────────────────────────────────────────


def test_jev_adapter_review_is_pass():
    gate = jev_adapter(classify=lambda ctx: ("review", "low-ish confidence"), requires={"jev_guardrail"})
    # review → 统一门禁视为 pass (advisory 不拦截)
    assert gate.evaluate(None).status == "pass"


def test_jev_adapter_block_is_block():
    gate = jev_adapter(classify=lambda ctx: ("block", "low confidence, escalate"), requires=set())
    assert gate.evaluate(None).status == "block"


def test_jev_adapter_in_chain_with_requires(monkeypatch):
    chain = GateChain()
    chain.register(
        jev_adapter(classify=lambda ctx: ("pass", "ok"), requires={"jev_guardrail"})
    )
    chain.set_available("jev_guardrail", False)
    assert chain.evaluate_all(None)[0].status == "skip"  # 依赖未满足 → 停用
    chain.set_available("jev_guardrail", True)
    assert chain.evaluate_all(None)[0].status == "pass"
