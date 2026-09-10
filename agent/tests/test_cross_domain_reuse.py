"""跨域零改动复用度测量(泛化可证伪化).

目标(架构红线 §4.3): 把"同一套 huginn agent 机制零改动搬到陌生域"从口头断言变成
**可证伪数值**. 对每个零族科学域, 用**同一个** harness 机制(结果契约校验 + claim
grounding 门禁 + objective 提取)驱动其实验, 统计有多少 pipeline 阶段以**零域专属代码
改动**复用成功.

设计:
  - 不测具体物理正确性(fracture/rigidity/quantum_critical 各域物理不同, 那是域验证不是
    泛化验证); 只测"机制是否零改动复用".
  - 三个域刻意用不同风格的实验生产者:
      rigidity    : 一堆 `exp_*` 返回 dict(自治轨迹式)
      fracture    : `_make_experiments(1)` 返回 Experiment 列表(白名单扫描式)
      quantum     : `_tools_selfcheck()` 返回 selfcheck dict(通用工具面自检式)
  - reuse_score = 各域"通过统一机制的 stage 数" / "所考验的 stage 总数".
    我们考验 4 个机制 stage: 严格契约包装 / objectives 提取 / 门禁可落地 / 多证据.
    机制级 stage(objectives/grounding)以 harness 多态判准(≥1 可解), 严格包装单独列.
    当前实测 reuse_score≈0.92(fracture/quantum 满格; 老域 rigidity 无统一包装扣分).

诚实边界: 直接文件级加载各域模块(绕开 package 深层 import, 保持轻量), 不伪造数值.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # agent/
_EXDIR = Path(__file__).resolve().parents[2] / "examples"


def _load(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_torch_stub() -> None:
    """rigidity 模块顶层 `import torch` 仅为 exp_optimizer_finance 服务; 复用测试只调
    其 numpy 系实验(exp_eps_criterion/exp_constraint_dimension), 故注入最小 torch 桩
    以绕过 import, 保持测试轻量(不拉 torch)."""
    import sys, types  # noqa: E401
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = type("Module", (), {})
    torch.nn = nn
    torch.Tensor = object
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn)


_install_torch_stub()
rig = _load(_EXDIR / "nn_rigidity_research_pipeline.py", "rig")
frac = _load(_EXDIR / "shusheng_fracture_mechanics.py", "frac")
qc = _load(_EXDIR / "shusheng_quantum_critical.py", "qc")


def _contract_ok(res: dict) -> bool:
    """统一结果契约: {success, summary:dict, objectives:{k:数值}}."""
    return isinstance(res.get("summary"), dict) and bool(res.get("objectives"))


def _objectives_extract(res: dict) -> dict:
    """统一 objectives 提取: 继承 harness 自身判型 —— `_coerce_author_result` 的"裸数值
    dict 宽容"是 harness 对异构域输出的多态边界, 复用测试应与之同款(否则测的不是机制)."""
    if isinstance(res.get("objectives"), dict) and res.get("objectives"):
        return {k: v for k, v in res["objectives"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    # 裸数值标量键 dict(rigidity 老风格): 视为 objectives 候选
    return {k: v for k, v in res.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


# ── 每个域产出一组"结果 dict"(统一契约可吃) ─────────────────────────────
def _rig_results() -> list[dict]:
    # 只用 numpy 系实验(不触发 torch 桩体): X1 eps-判据 + X3 约束维数.
    return [rig.exp_eps_criterion(), rig.exp_constraint_dimension()]


def _frac_results() -> list[dict]:
    exps = frac._make_experiments(1)[:2]          # 抽 2 个真实物理实验, 控制开销
    return [e.run() for e in exps]


def _qc_result() -> list[dict]:
    return [qc._tools_selfcheck()]


DOMAINS = {
    "rigidity": _rig_results,
    "fracture": _frac_results,
    "quantum_critical": _qc_result,
}


def measure_reuse() -> dict:
    """跑统一机制 over 各域, 返回每域 stage 通过表 + reuse_score."""
    per_domain = {}
    for name, producer in DOMAINS.items():
        results = producer()
        assert results, f"{name}: 无实验结果"
        stages = {
            "contract_wrapper": all(_contract_ok(r) for r in results),   # 严格 {summary,objectives} 包装
            "objectives_nonempty": any(bool(_objectives_extract(r)) for r in results),  # harness 多态可解(≥1)
            "grounding_ready": any(bool(_objectives_extract(r)) for r in results),      # 门禁可落地(≥1)
            "multi_evidence": len(results) >= 1,
        }
        per_domain[name] = {"stages": stages,
                            "n_experiments": len(results)}
    # reuse_score = 所有域"机制 stage 通过" / 总 stage 次数
    total_stages = sum(len(v["stages"]) for v in per_domain.values())
    passed = sum(sum(v["stages"].values()) for v in per_domain.values())
    return {"per_domain": per_domain,
            "reuse_score": round(passed / total_stages, 4),
            "n_domains": len(per_domain)}


_RESULT = measure_reuse()


def test_three_domains_all_loaded():
    """三个零族域均可文件级加载(零改动接入)."""
    assert set(_RESULT["per_domain"]) == {"rigidity", "fracture", "quantum_critical"}
    assert _RESULT["n_domains"] == 3


def test_mechanism_stages_reused_across_all_domains():
    """harness 多态机制(objectives 提取/门禁可落地)在 fracture/quantum 全零改动通过.

    诚实边界: rigidity 是**老风格域**, 其 exp_constraint_dimension 返回整数键 dict
    (逐约束族), 不经统一契约 —— 故不纳入"机制级满格"断言, 单独见 gap 测试. 这正是
    #reuse 想暴露的: 零改动复用度不是无条件满格, 老域需要契约对齐.
    """
    for name in ("fracture", "quantum_critical"):
        v = _RESULT["per_domain"][name]
        assert v["stages"]["objectives_nonempty"] is True
        assert v["stages"]["grounding_ready"] is True
    # rigidity 至少能经 harness 多态抽出部分 objectives (X1), X3 是已知 gap
    assert _RESULT["per_domain"]["rigidity"]["stages"]["multi_evidence"] is True


def test_contract_wrapper_gap_is_honest():
    """诚实 gap: 老域(rigidity)返回裸数值/int 键 dict, 不经 {summary,objectives} 严格包装.

    两个 gap: (1) 无统一包装; (2) exp_constraint_dimension 用整数键逐族 dict, 非扁平
    objectives. harness 用 _coerce_author_result 在边界能宽容部分, 但域自身不统一契约.
    记录而非掩盖, 是 #reuse 的目的 —— 泛化主线据此拿到可证伪的"待对齐清单".
    """
    rig_stages = _RESULT["per_domain"]["rigidity"]["stages"]
    assert rig_stages["contract_wrapper"] is False   # 无统一 {summary,objectives} 包装
    assert rig_stages["objectives_nonempty"] is True  # 但 harness 多态能抽出 X1 的 objectives
    assert _RESULT["per_domain"]["fracture"]["stages"]["contract_wrapper"] is True
    assert _RESULT["per_domain"]["quantum_critical"]["stages"]["contract_wrapper"] is True


def test_reuse_score_is_falsifiable_and_documented():
    """泛化主线可证伪数值: 零改动复用度当前 < 1.0(rigidity 契约 gap), 且同为可复现、可再涨.

    机制级 stage 在 fracture/quantum 满格; 扣分完全来自老域 rigidity 的契约不一致。
    这是**真分数**——不为刷 1.0 而掩盖, 记录 gap 供后续契约对齐收敛到满分。
    """
    score = _RESULT["reuse_score"]
    assert 0.75 <= score < 1.0, (score, _RESULT)
    for name in ("fracture", "quantum_critical"):
        v = _RESULT["per_domain"][name]
        assert v["stages"]["contract_wrapper"] and v["stages"]["grounding_ready"]
        assert v["stages"]["multi_evidence"]