# 分层流式结算 · 生产化（阈值参数化 + 跨 run 稳定度先验）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 [layered-streaming-settlement-spec.md](../layered-streaming-settlement-spec.md) §9 的两个待办落地：P-A/P-B/P-C 的硬编码阈值参数化；把一次 run 的分数高原沉淀为可传递、可注入的下次 run 预算先验（诚实钳制，不激进）。

**Architecture:** 两处接口接缝，均遵守"新视角只许注册进聚合头，不新增 `out.*` 字段"的架构纪律（`tests/test_arch_cleanliness.py::test_research_outcome_fields_are_frozen` 执法）：
1. **A1 参数化**：`run_research_program` 新增 4 个可选参数（均有与现常量相同的默认值），替换 `program.py` 中 4 处常量使用点；`replan_gate.decide_replan_skip` / `early_stop_gate.stability_check` 纯函数已是参数化接口，只透传。
2. **A2 先验**：新模块 `huginn/research/prior_store.py`，两个纯函数 —— `extract_prior(out)` 把一次 run 的稳定度观察沉淀为 dict；`resolve_early_stop_args(prior)` 把先验映射为早停预算参数，**只朝着更保守方向平移**（min_layers 单调不减、钳制下限 2），绝不因上次稳定而激进提前终止。

**Tech Stack:** Python 3 (dataclasses, asyncio), pytest, 现有 Huginn research 管线。

---

### Task 0: 基线预检

**Files:**
- Run only (no changes)

- [ ] **Step 1: 跑研究线全量基线**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py tests/test_arch_cleanliness.py tests/test_decision_gate.py tests/test_tool_surface_unification.py tests/test_self_harness.py tests/test_research_planning_team.py tests/test_evolution_drivers.py tests/test_capability_introspection.py tests/test_harness_ledger.py tests/test_harness_significance_gate.py tests/test_exploration.py -q --no-cov
```

Expected: `180 passed`（含既有 P-A/P-B/P-C 全部测试）。

- [ ] **Step 2: 确认 4 个待参数化常量的使用点**

```bash
cd /workspace/agent && grep -n "_STREAM_SUMMARY_CHARS\|_REPLAN_SIM\|_EARLY_STOP_MIN_LAYERS\|_EARLY_STOP_MARGIN" huginn/research/program.py
```

Expected: 4 处常量定义（`program.py` 顶部 ~L86-95）+ 各 1 处使用点（`_stream_rows` distil 调用 / `_replan_decision` / `_settle_layer_check`）。

---

### Task 1: A1 · 阈值参数化（stream_summary_chars / replan_similarity / early_stop_min_layers / early_stop_margin）

**Files:**
- Modify: `huginn/research/program.py`（4 个常量使用点 → 新参数；函数签名）
- Test: `tests/test_aggregation_head.py`（追加 4 个参数化测试）

- [ ] **Step 1: 写失败测试（4 个参数化行为测试）**

追加到 `tests/test_aggregation_head.py` 文件末尾：

```python
# ── 生产化 A1: 阈值参数化 ─────────────────────────────────────────────────
def _run_(v: float):
    return lambda: {"summary": {"y": v}, "objectives": {"score": v}}


def test_a1_stream_summary_chars_parameterized():
    """P-A 单条摘要上限参数化: 传 stream_summary_chars 有界生效(默认 1200 不变)."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 stream", [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0))],
    )
    out = run_research_program(
        goal="a1 stream", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan, layer_epochs=True, stream_summary_chars=80,
    )
    rows = [r for lay in out.consolidated["stream_view"] for r in lay["rows"]]
    assert rows, "stream_view 应有记录"
    assert all(len(r["summary"]) <= 80 + 80 for r in rows), "自定义上限生效(有界)"


def test_a1_replan_similarity_parameterized():
    """P-B 重叠阈值参数化: 调高到 0.9 后, Jaccard 0.83 的方向不再视为冗余 → 真实执行."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 sim",
        [SubResearch("a", "sweep beta ceramic domain thick", _run_(10.0)),
         SubResearch("e", "sweep beta ceramic domain thick dense", _run_(8.0),
                     depends_on=["a"])],
    )
    out = run_research_program(
        goal="a1 sim", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, replan_gate=True, replan_similarity=0.9,
        max_parallel=1,
    )
    assert "e" in out.cache, "0.83 < 0.9 → 不冗余, 必须执行"


def test_a1_early_stop_margin_parameterized():
    """P-C margin 参数化: 放宽到 0.5 后, rel=0.4 的移动也视为高原 → 提前终止."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 margin",
        [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run_(14.0), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run_(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="a1 margin", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, early_stop_margin=0.5,
        max_parallel=1,
    )
    assert out.consolidated["early_stop"]["verdict"] == "early_stopped"
    assert "e" not in out.cache, "margin=0.5 视 rel=0.4 为高原 → 剩余层终止"


def test_a1_early_stop_min_layers_parameterized():
    """P-C 防早停参数化: min_layers=1 时 1 层结算即可查稳定并终止."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 minlayers",
        [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run_(10.1), depends_on=["a"])],
    )
    out = run_research_program(
        goal="a1 minlayers", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, early_stop_min_layers=1,
        max_parallel=1,
    )
    assert out.consolidated["early_stop"]["verdict"] == "early_stopped"
    assert "c" not in out.cache, "min_layers=1 → 层0 结算即查稳定"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py -q --no-cov -k "a1"
```

Expected: 4 个失败（`TypeError: run_research_program() got an unexpected keyword argument 'stream_summary_chars'`）—— 参数尚未定义。

- [ ] **Step 3: 实现参数化（program.py 4 处替换）**

① 函数签名 `huginn/research/program.py`，在 `early_stop_gate: bool = False,` 参数（~L210-213）后追加：

```python
    stream_summary_chars: int = _STREAM_SUMMARY_CHARS,  # A1: P-A 单条层摘要上限(字符)
    replan_similarity: float = _REPLAN_SIM,             # A1: P-B 假说重叠阈值(0..1)
    early_stop_min_layers: int = _EARLY_STOP_MIN_LAYERS, # A1: P-C 防早停最小观测层数
    early_stop_margin: float = _EARLY_STOP_MARGIN,      # A1: P-C 分数高原容差
) -> ResearchOutcome:
```

② P-A 流式入账（`_stream_rows` 的 `distill_tool_output` 调用，~L399）：`max_chars=_STREAM_SUMMARY_CHARS` → `max_chars=stream_summary_chars`。

③ P-B `_replan_decision`（~L296-301）：

```python
        return decide_replan_skip(
            name, i, prior, cache, _hypotheses,
            already_decided=set(cache) | {r["name"] for r in _replan_log},
            similarity=replan_similarity, tol=_WM_TOL,
        )
```

④ P-C `_settle_layer_check`（~L333）：

```python
        _st = stability_check(settled_layers, min_layers=early_stop_min_layers,
                              margin=early_stop_margin)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py -q --no-cov -k "a1"
```

Expected: `4 passed`。

- [ ] **Step 5: 默认值回归（零行为变化）**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py -q --no-cov -k "pa or pb or pc or pabc or parallel or stress"
```

Expected: 全部通过（默认值 = 原常量，行为不变）。

- [ ] **Step 6: Commit**

```bash
git add huginn/research/program.py tests/test_aggregation_head.py
git commit -m "feat: P-A/P-B/P-C 阈值参数化(stream/replan/early_stop 可配置, 默认零变化)"
```

---

### Task 2: A2 · 先验提取（extract_prior）

**Files:**
- Create: `huginn/research/prior_store.py`
- Test: `tests/test_aggregation_head.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_aggregation_head.py` 末尾：

```python
# ── 生产化 A2: 跨 run 稳定度先验 ─────────────────────────────────────────
def test_a2_extract_prior_from_out():
    """从一次稳定终止的 run 提取可复用先验(高原料纯函数)."""
    from huginn.research.prior_store import extract_prior
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a2 extract",
        [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run_(10.1), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run_(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="a2 extract", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, max_parallel=1,
    )
    prior = extract_prior(out)
    assert prior["applicable"] is True
    assert prior["source"] == "layered_settlement"
    assert prior["plateau"]["layer_index"] == 1
    assert prior["plateau"]["top"] == "c"
    assert prior["goal"] == "a2 extract"


def test_a2_extract_prior_not_applicable_for_plain_run():
    """未启用早停的 run → 提取出不可用先验(诚实标 applicable=False)."""
    from huginn.research.prior_store import extract_prior
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a2 plain", [SubResearch("a", "sweep alpha metallic alloy", _run_(1.0))],
    )
    out = run_research_program(
        goal="a2 plain", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    assert extract_prior(out)["applicable"] is False
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py -q --no-cov -k "a2"
```

Expected: `ModuleNotFoundError: No module named 'huginn.research.prior_store'`。

- [ ] **Step 3: 实现 extract_prior（新建文件）**

`huginn/research/prior_store.py` 完整内容：

```python
"""A2 · 跨 run 稳定度先验 (cross-run settlement prior) —— 把一次 run 的分数高原
沉淀为可传递、可注入的预算先验.

背景: 同类目标域反复研究时, "这个领域通常在第几层进入分数高原"是**可复用**的调度
经验. 本模块把一次 run 的稳定度观察提取成纯 dict(extract_prior), 再由下一次 run
以保守方式注入(resolve_early_stop_args) —— 但它只是**预算参数先验**, 绝不参与、
更不替代任何真实实验.

诚实红线(不可逾越):
  1. extract_prior 只读 out.consolidated(聚合头唯一出口), 不新增 out.* 字段;
  2. 先验只能把早停参数推向**更保守**方向(min_layers 单调不减), 永不因"上次稳定"
     而激进提前终止 —— 领域可能漂移, 先验不为历史背书;
  3. applicable=False 的字典可安全注入(等价无先验), 不抛错、不改行为.
"""
from __future__ import annotations

from typing import Any


def extract_prior(out: Any) -> dict[str, Any]:
    """从 ResearchOutcome 提取一次 run 的稳定度先验 (纯函数, 无副作用).

    只消费 out.consolidated["early_stop"] —— 未启用早停/无观测 → applicable=False.
    返回说明:
      - applicable: 是否有可复用的高原观察(early_stopped 才为 True);
      - plateau: {layer_index, top, score, relative_change} 首次判稳的那一层;
      - goal: 目标原文(跨 run 匹配建议用归一化 slug, 由调用方决定).
    """
    con = getattr(out, "consolidated", None) or {}
    es = con.get("early_stop") or {}
    st = es.get("stability") or {}
    stopped = es.get("verdict") == "early_stopped"
    layer_index = es.get("stopped_after_layer")
    if not stopped or layer_index is None or not st:
        return {
            "source": "layered_settlement",
            "applicable": False,
            "reason": "no_early_stopped",
            "goal": getattr(out, "converred", "") or "",
        }
    return {
        "source": "layered_settlement",
        "applicable": True,
        "goal": getattr(out, "converred", "") or "",
        "plateau": {
            "layer_index": int(layer_index),
            "top": st.get("top"),
            "score": st.get("score"),
            "relative_change": st.get("relative_change"),
        },
    }


def resolve_early_stop_args(
    prior: dict | None,
    *,
    default_min_layers: int = 2,
    default_margin: float = 0.02,
) -> dict[str, Any]:
    """把先验映射为早停参数(纯函数, 确定性).

    规则(全部可证伪):
      - prior 为空/不可用 → 返回默认参数, note="no_prior";
      - prior 可用(early_stopped) → min_layers = plateau.layer_index + 1,
        且钳制在 [default_min_layers, 4] —— 上次 N 层才稳定, 这次至少等 N 层
        才允许查稳定(**更保守**, 防领域漂移误停); margin 原样传默认(不因先验放宽).
    """
    if not prior or not prior.get("applicable"):
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "no_prior"}
    plateau = prior.get("plateau") or {}
    idx = plateau.get("layer_index")
    if idx is None:
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "prior_without_plateau"}
    min_layers = max(default_min_layers, min(int(idx) + 1, 4))
    return {"min_layers": min_layers, "margin": default_margin,
            "source": "cross_run_prior", "note": f"plateau_layer={idx}"}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py -q --no-cov -k "a2"
```

Expected: `2 passed`。

- [ ] **Step 5: Commit**

```bash
git add huginn/research/prior_store.py tests/test_aggregation_head.py
git commit -m "feat: 跨 run 稳定度先验提取(extract_prior, 聚合头出口独读不新增 out.*)"
```

---

### Task 3: A2 · 先验注入（resolve 接线 + program.py 消费）

**Files:**
- Modify: `huginn/research/program.py`（`prior` 参数 + P-C 状态块 + `_early_stop_head` 加 `prior_used`）
- Test: `tests/test_aggregation_head.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_aggregation_head.py` 末尾：

```python
def test_a3_prior_injection_relaxes_min_layers():
    """先验注入: 上次 plateau 在第 2 层 → 本次 min_layers 保守提高到 3,
    3 层场景不再提前终止(全执行), 且 prior_used 记录进聚合视图."""
    from huginn.research.planning import SubResearch, build_research_plan

    # 3 层场景: 层0=10, 层1=10.1(rel 0.01 高原) → 默认 min_layers=2 会在层1 后终止
    plan = build_research_plan(
        "a3 inject",
        [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run_(10.1), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run_(8.0), depends_on=["c"])],
    )
    # 人工构造"上次 plateau 在第 2 层"的先验 → resolve 得 min_layers=3(更保守)
    prior = {"source": "layered_settlement", "applicable": True,
             "plateau": {"layer_index": 2, "top": "g", "score": 15.1,
                         "relative_change": 0.0067}}
    out = run_research_program(
        goal="a3 inject", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True,
        prior=prior, max_parallel=1,
    )
    es = out.consolidated["early_stop"]
    assert es["prior_used"]["note"] == "plateau_layer=2"
    assert es["prior_used"]["min_layers"] == 3
    assert es["verdict"] != "early_stopped", "min_layers=3 > 3 层场景可观测层数 → 不终止"
    assert "e" in out.cache, "先验把等待拉长 → 剩余层真实执行"


def test_a3_prior_none_keeps_defaults():
    """无先验 → 不注入, 默认 min_layers=2, prior_used 记为未使用."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a3 none", [SubResearch("a", "sweep alpha metallic alloy", _run_(1.0))],
    )
    out = run_research_program(
        goal="a3 none", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True,
    )
    es = out.consolidated["early_stop"]
    assert es["prior_used"]["note"] == "no_prior"
    assert es["prior_used"]["min_layers"] == 2
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py -q --no-cov -k "a3"
```

Expected: `KeyError: 'prior_used'`（`_early_stop_head` 尚无此键）—— 参数也尚未定义。

- [ ] **Step 3: 实现注入接线（program.py）**

① 函数签名：在 `early_stop_margin: float = _EARLY_STOP_MARGIN,` 后（Task 1 新加的行）追加：

```python
    prior: dict | None = None,                      # A2: 跨 run 稳定度先验(prior_store.extract_prior 产物).
                                                    # 只保守调整早停预算参数(min_layers 单调不减), 绝不参与实验.
) -> ResearchOutcome:
```

② P-C 状态块（`_early_stop_meta` 定义处，~L313-316 之后）追加先验解析：

```python
    # A2: 先验注入 —— 只把早停参数推向更保守方向(上次 N 层才稳定, 这次至少等 N 层).
    from huginn.research.prior_store import resolve_early_stop_args
    _prior_args = resolve_early_stop_args(
        prior, default_min_layers=_EARLY_STOP_MIN_LAYERS, default_margin=_EARLY_STOP_MARGIN)
    _early_stop_min_layers = int(_prior_args["min_layers"])
    _early_stop_margin = float(_prior_args["margin"])
```

③ `_settle_layer_check` 内 `stability_check(...)` 调用（Task 1 已改）：

```python
        _st = stability_check(settled_layers, min_layers=_early_stop_min_layers,
                              margin=_early_stop_margin)
```

（注意：此块把 Task 1 的 `min_layers=early_stop_min_layers, margin=early_stop_margin` 替换为解析后的局部量，保证 prior 注入优先于参数。）

④ `_early_stop_head` 构建处（~L780）追加键：

```python
        _early_stop_head = {
            "enabled": _early_stop_enabled,
            "layers": len(_layers_map) if _early_stop_enabled else 0,
            "stopped_after_layer": _early_stop_meta.get("stopped_after_layer"),
            "skipped": len(_early_stop_meta.get("skipped", [])),
            "log": list(_early_stop_meta.get("skipped", [])),   # 完整终止名单(可证伪)
            "prior_used": {"min_layers": _early_stop_min_layers,
                           "margin": _early_stop_margin,
                           "source": _prior_args["source"],
                           "note": _prior_args["note"]},
            "verdict": ("no_early_stop" if not _early_stop_enabled
                        else _early_stop_meta.get("verdict", "checked_and_continued")),
        }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py -q --no-cov -k "a3"
```

Expected: `2 passed`。

- [ ] **Step 5: 架构纪律 + 全量回归**

```bash
cd /workspace/agent && python -m pytest tests/test_arch_cleanliness.py tests/test_aggregation_head.py -q --no-cov
```

Expected: 全绿 —— `test_research_outcome_fields_are_frozen` 通过（未新增 `out.*` 字段，先验走 `consolidated.early_stop.prior_used`）。

- [ ] **Step 6: Commit**

```bash
git add huginn/research/program.py tests/test_aggregation_head.py
git commit -m "feat: 跨 run 稳定度先验注入(prior 保守调整早停参数, prior_used 入聚合视图)"
```

---

### Task 4: 规格文档收尾 + 总回归

**Files:**
- Modify: `docs/layered-streaming-settlement-spec.md`（§9 里程碑勾选）

- [ ] **Step 1: 勾选生产化里程碑**

在 `docs/layered-streaming-settlement-spec.md` 第 9 节，把：

```markdown
- [ ] 生产化: `min_layers` / `margin` / `skip_similarity` 从常量提为可配置参数；
     跨 run 的稳定度学习（把一次 run 的分数高原当作下个 run 的预算先验）。
```

替换为：

```markdown
- [x] 生产化: 阈值已参数化(stream_summary_chars / replan_similarity /
     early_stop_min_layers / early_stop_margin)；稳定度先验已沉淀
     (`prior_store.extract_prior` → `run_research_program(prior=...)` 保守注入,
     prior_used 入聚合视图)。计划: `docs/superpowers/plans/2026-09-09-settlement-productionize.md`。
```

- [ ] **Step 2: 全量研究线回归**

```bash
cd /workspace/agent && python -m pytest tests/test_aggregation_head.py tests/test_arch_cleanliness.py tests/test_decision_gate.py tests/test_tool_surface_unification.py tests/test_self_harness.py tests/test_research_planning_team.py tests/test_evolution_drivers.py tests/test_capability_introspection.py tests/test_harness_ledger.py tests/test_harness_significance_gate.py tests/test_exploration.py -q --no-cov
```

Expected: `188 passed`（180 基线 + 4 a1 + 2 a2 + 2 a3）。

- [ ] **Step 3: Commit**

```bash
git add docs/layered-streaming-settlement-spec.md
git commit -m "docs: 标记分层流式结算生产化里程碑完成"
```

---

## Self-Review

**Spec coverage（spec §9 待办 → 任务）：**
- 阈值参数化（min_layers/margin/skip_similarity 提为参数）→ Task 1（覆盖全部 4 个阈值，含 stream_summary_chars）。
- 跨 run 稳定度学习（高原 → 下个 run 预算先验）→ Task 2（提取）+ Task 3（保守注入 + 审计）。
- 架构纪律（不新增 out.* 字段）→ Task 3 Step 5 以 `test_research_outcome_fields_are_frozen` 强制验证。

**Placeholder scan：** 全部步骤含完整代码与精确预期；无 TBD/TODO。

**Type consistency：** Task 1 新参数名 `stream_summary_chars`/`replan_similarity`/`early_stop_min_layers`/`early_stop_margin` 在测试与实现中逐一对应；Task 3 用 `_early_stop_min_layers`/`_early_stop_margin` 局部量替换 Task 1 参数名（prior 注入优先），`_early_stop_head` 的 `prior_used` 键在测试 (`test_a3_prior_injection_relaxes_min_layers`) 与实现中命名一致。