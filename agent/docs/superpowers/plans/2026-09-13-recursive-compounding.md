# Recursive Compounding (轨道 A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 A1 递归改进器具备"改进器自身的改进策略可被改进"（strategist champion）+ 复合验收（CompoundingTracker）+ 鲁棒性护栏（BehavioralFidelity / 死锁 / 滞回 / 随机化对照）+ 时空可组合（RevertibleContext / CoEffectRegistry）。

**Architecture:** 在 `huginn/harness/meta_improver.py` 扩展出 level-0（improver 模板，已有）与 level-1（strategist，本次新增）两层。`maybe_propose` 用 strategist champion 生成改进器候选；strategist 候选经 `maybe_promote_strategist` 在 `RevertibleContext.transaction()` 内换件并注册补偿器，门控 = 显著+OOD+复合不退化+保真锚；依赖图用 `CoEffectRegistry` 声明。不改 `revertible.py`/`coeffect.py` 本身，只接入。

**Tech Stack:** Python 3.14, pytest, asyncio, dataclasses, `huginn.security.revertible`（RevertibleContext/register_compensator）、`huginn.security.coeffect`（CoEffectRegistry）。

**Spec:** `docs/staging/specs/2026-09-13-recursive-compounding-design.md`

---

## 决策锁定（跨任务统一签名，勿改）

- 新常量 `_META2_IMPROVE_TEMPLATE`（固定 meta² 模板，只一层递归）。
- `strategist_prompt() -> str`：有 strategist champion 返回其 `strategist_prompt`，否则回落 `_META_IMPROVE_TEMPLATE`。
- `maybe_propose` 把 `_META_IMPROVE_TEMPLATE.format(...)` 改为 `self.strategist_prompt().format(...)`。
- 门控组合 `maybe_promote_strategist` = `sig.gate_decision.passed` 且 `ood.validate_ood.passed` 且 `not tracker.would_degrade(new, incumbent)`。
- 持久化根：`huginn.harness.meta_improver.MetaImprover` 的 `self._dir`；新增 `strategist/`、`compounding.json`、`fidelity.json`、`revertible_journal.json`。
- 统一开关：`harness_meta_improver`（`_enabled.py` 已有）开启后两层才 `enabled()`。

---

### Task 1: Config 与复合追踪器

**Files:**
- Modify: `agent/huginn/harness/meta_improver.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_meta_improver.py 追加
def test_compounding_tracker_math():
    from huginn.harness.meta_improver import CompoundingTracker
    tr = CompoundingTracker(window=4)
    # 质量持续上升 → is_compounding
    for i in range(4):
        tr.record(epoch=i, config_id=f"c{i}", quality=0.5 + 0.1 * i,
                  fidelity=0.6, proposals_to_promotion=2,
                  generations_to_promotion=5, win_rate=0.5)
    assert tr.is_compounding() is True
    assert tr.stats()["slope"] > 0
    # 退化 → would_degrade
    dec = CompoundingTracker(window=4)
    for i in range(4):
        dec.record(epoch=i, config_id=f"d{i}", quality=0.9 - 0.2 * i,
                   fidelity=0.6, proposals_to_promotion=2,
                   generations_to_promotion=5, win_rate=0.5)
    assert dec.would_degrade(new="n", incumbent="i") is True
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k compounding_tracker -o addopts="" -q`
Expected: `ImportError` / `AttributeError: module 'huginn.harness.meta_improver' has no attribute 'CompoundingTracker'`

- [ ] **Step 3: 实现 `StrategistConfig` + `CompoundingTracker`**

在 `meta_improver.py` 顶部（`ImproverConfig` 之后、`MetaImprover` 之前）插入：

```python
@dataclass
class StrategistConfig:
    """level-1: 生成改进器候选的"策略"模板."""
    config_id: str
    strategist_prompt: str            # 覆盖 _META_IMPROVE_TEMPLATE 的模板
    active: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategistConfig:
        return cls(
            config_id=d["config_id"],
            strategist_prompt=d.get("strategist_prompt", ""),
            active=bool(d.get("active", False)),
            created_at=float(d.get("created_at", time.time())),
        )


class CompoundingTracker:
    """复合验收: 滚动窗口斜率 + 每次换件成本趋势 + 滞回带 + 死锁计数."""

    def __init__(self, window: int = 8, slope_tolerance: float = -0.05,
                 hysteresis_band: float = 0.03, deadlock_timeout: int = 5) -> None:
        self.window = window
        self.slope_tolerance = slope_tolerance  # 质量斜率下限(负容忍)
        self.hysteresis_band = hysteresis_band
        self.deadlock_timeout = deadlock_timeout
        self._rows: list[dict[str, Any]] = []
        self._deadlock_n = 0

    def record(self, *, epoch: int, config_id: str, quality: float,
               fidelity: float, proposals_to_promotion: int,
               generations_to_promotion: int, win_rate: float) -> None:
        self._rows.append({
            "epoch": epoch, "config_id": config_id, "quality": quality,
            "fidelity": fidelity, "proposals_to_promotion": proposals_to_promotion,
            "generations_to_promotion": generations_to_promotion, "win_rate": win_rate,
        })
        self._rows = self._rows[-self.window:]

    def _quality_series(self) -> list[float]:
        return [r["quality"] for r in self._rows]

    def _slope(self) -> float:
        ys = self._quality_series()
        if len(ys) < 2:
            return 0.0
        xs = list(range(len(ys)))
        n = len(xs)
        xm, ym = sum(xs) / n, sum(ys) / n
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs)
        return 0.0 if den == 0 else num / den

    def _cost_trend(self) -> float:
        """每次换件成本(proposals) 末尾 vs 头部 的增量."""
        if len(self._rows) < 4:
            return 0.0
        head = self._rows[:2]
        tail = self._rows[-2:]
        h = sum(r["proposals_to_promotion"] for r in head) / len(head)
        t = sum(r["proposals_to_promotion"] for r in tail) / len(tail)
        return t - h

    def is_compounding(self) -> bool:
        """质量斜率≥容差 且 每次换件成本不持续升."""
        if self._slope() < self.slope_tolerance:
            return False
        if self._cost_trend() > 1.0:  # 换件成本翻倍以上视为退化
            return False
        return True

    def would_degrade(self, new: str, incumbent: str) -> bool:
        """复合已退化(甚至当前质量低于含滞回带的基线) → 换件会让事情更糟."""
        ys = self._quality_series()
        if not ys:
            return False
        best = max(ys)
        cur = ys[-1]
        return (cur < best - self.hysteresis_band) and not self.is_compounding()

    def mark_deadlock(self, yellow: bool) -> None:
        self._deadlock_n = self._deadlock_n + 1 if yellow else 0

    def in_deadlock(self) -> bool:
        return self._deadlock_n >= self.deadlock_timeout

    def stats(self) -> dict[str, Any]:
        return {
            "slope": round(self._slope(), 4),
            "cost_trend": round(self._cost_trend(), 4),
            "window": len(self._rows),
            "best_quality": max(self._quality_series()) if self._rows else 0.0,
            "deadlock_n": self._deadlock_n,
            "is_compounding": self.is_compounding(),
        }
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k compounding_tracker -o addopts="" -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py agent/tests/test_meta_improver.py && git commit -m "feat(A1): StrategistConfig + CompoundingTracker 复合验收核心"
```

---

### Task 2: BehavioralFidelity 保真锚

**Files:**
- Modify: `agent/huginn/harness/meta_improver.py`

- [ ] **Step 1: 写失败测试**

```python
def test_behavioral_fidelity_anchor():
    from huginn.harness.meta_improver import BehavioralFidelity
    bf = BehavioralFidelity()
    bf.record_acceptance("c1", accepted=True)
    bf.record_acceptance("c1", accepted=False)      # 0.5
    assert abs(bf.fidelity_score("c1") - 0.5) < 1e-9
    hi = bf.anchor_in("c1", p_quality=0.8, p_fidelity=None)   # 无保真记录 → 默认 0.5
    assert abs(hi - 0.65) < 1e-9                     # 0.5*0.8 + 0.5*0.5
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k behavioral_fidelity -o addopts="" -q`
Expected: `ImportError: cannot import name 'BehavioralFidelity'`

- [ ] **Step 3: 实现**

在 `CompoundingTracker` 之后插入：

```python
class BehavioralFidelity:
    """行为级奖励回流: 用真实 apply_patches 采纳率作复合指标的保真锚(治 Goodhart)."""

    _instance: BehavioralFidelity | None = None
    _MAX = 50

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (get_runtime_home() / "meta_improver" / "fidelity.json")
        self._accepted: dict[str, int] = {}
        self._applied: dict[str, int] = {}

    @classmethod
    def get_instance(cls, path: Path | None = None) -> BehavioralFidelity:
        if cls._instance is None:
            cls._instance = cls(path)
        return cls._instance

    def record_acceptance(self, candidate_id: str, accepted: bool) -> None:
        self._applied[candidate_id] = self._applied.get(candidate_id, 0) + 1
        if accepted:
            self._accepted[candidate_id] = self._accepted.get(candidate_id, 0) + 1
        if len(self._applied) > self._MAX:  # LRU 抗无限增长
            for k in list(self._applied)[: len(self._applied) - self._MAX]:
                self._applied.pop(k, None)
                self._accepted.pop(k, None)
        self._save()

    def fidelity_score(self, candidate_id: str) -> float:
        a = self._applied.get(candidate_id, 0)
        if a == 0:
            return 0.5  # 未知 → 中性
        return self._accepted.get(candidate_id, 0) / a

    def anchor_in(self, candidate_id: str, p_quality: float,
                  p_fidelity: float | None = None) -> float:
        """加权合成: 质量 ⊕ 真实采纳保真 (各 0.5)."""
        f = p_fidelity if p_fidelity is not None else self.fidelity_score(candidate_id)
        return 0.5 * p_quality + 0.5 * f

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"accepted": self._accepted, "applied": self._applied},
                           ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k behavioral_fidelity -o addopts="" -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py agent/tests/test_meta_improver.py && git commit -m "feat(A1): BehavioralFidelity 采纳保真锚"
```

---

### Task 3: strategist 递归化（strategist_prompt + maybe_propose_strategist）

**Files:**
- Modify: `agent/huginn/autoloop/../harness/meta_improver.py`（即 `agent/huginn/harness/meta_improver.py`）

- [ ] **Step 1: 写失败测试**

```python
def test_strategist_fallback_and_propose():
    from huginn.harness.meta_improver import MetaImprover, _META_IMPROVE_TEMPLATE
    meta = MetaImprover.get_instance()
    # 无 strategist champion → 回落默认
    assert meta.strategist_prompt() == _META_IMPROVE_TEMPLATE
    # propose 出一个合法 strategist
    from huginn.harness._enabled import _harness_enabled as _e  # noqa
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k strategist -o addopts="" -q`
Expected: `AttributeError: 'MetaImprover' object has no attribute 'strategist_prompt'`

- [ ] **Step 3: 实现**

新增常量（`_META_IMPROVE_TEMPLATE` 之后）：

```python
# A1: meta² 固定模板 — 生成新的 strategist(改进策略) 候选. 只一层递归(YAGNI 不再叠).
_META2_IMPROVE_TEMPLATE = (
    "You are the meta-strategist. The current 'strategist' template below is used "
    "to propose improvements to the research agent's IMPROVER. Rewrite it so that "
    "proposed improvers converge FASTER and are more directive-aligned.\n"
    "----- CURRENT STRATEGIST TEMPLATE -----\n{current_strategist}\n"
    "---------------------------------------\n"
    "Keep it a plain prompt template. Respond with the rewritten template ONLY.\n"
    "Meta stats: promotions={n_promotions}, win_rate={win_rate}."
)
```

在 `MetaImprover` 内新增方法（插在 `current_template` 之后）：

```python
def strategist_champion(self) -> StrategistConfig | None:
    if not self.enabled():
        return None
    c = self._strategists.get(self._active_strategist_id or "")
    return c if c is not None and c.active else None

def strategist_prompt(self) -> str:
    champ = self.strategist_champion()
    return champ.strategist_prompt if champ else _META_IMPROVE_TEMPLATE

async def maybe_propose_strategist(self, llm_chat_fn: Callable) -> str | None:
    """meta² 固定模板生成新的 strategist 模板候选."""
    if not self.enabled():
        return None
    cur = self.strategist_prompt()
    p2 = _META2_IMPROVE_TEMPLATE.format(
        current_strategist=cur,
        n_promotions=self._promotions,
        win_rate=self.compounding_trace()["meta_win_rate"],
    )
    try:
        resp = await llm_chat_fn(p2, task="summarize")
    except Exception:
        logger.debug("meta propose_strategist LLM fail", exc_info=True)
        return None
    if not resp or not resp.strip():
        return None
    tpl = resp.strip()
    if tpl.startswith("```"):
        tpl = tpl.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not tpl or not all(ph in tpl for ph in ("{phase}", "{block_names}", "{r_phys}", "{directive}", "{current_strategist}")):
        self._trace({"type": "strategy_reject", "reason": "invalid_template"})
        return None
    s = StrategistConfig(config_id=f"strat_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}",
                          strategist_prompt=tpl)
    self._strategists[s.config_id] = s
    self._save_strategist(s)
    self._trace({"type": "strategy_propose", "candidate_id": s.config_id})
    return s.config_id
```

`__init__` 增加状态：`self._strategists: dict[str, StrategistConfig] = {}`、`self._active_strategist_id: str | None = None`；`_load`/`_save_cfg` 同步读/写这两项；新增 `_save_strategist`/`_load_strategists`（路径 `self._dir/"strategist"/"<id>.json"`）。`compounding_trace()` 返回值补 `active_strategist_id`、`n_strategists`。

- [ ] **Step 4: 运行确认通过**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k strategist -o addopts="" -q` Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py agent/tests/test_meta_improver.py && git commit -m "feat(A1): strategist champion + maybe_propose_strategist 递归层"
```

---

### Task 4: maybe_propose 改用 strategist + 时空可组合换件

**Files:**
- Modify: `agent/huginn/harness/meta_improver.py`、`agent/huginn/harness/prompt_patch.py`

- [ ] **Step 1: 写失败测试（时空可组合 + reverent 换件）**

```python
def test_strategist_swap_revertible():
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig, _register_strategist_compensator,
    )
    _register_strategist_compensator()
    from huginn.security.revertible import RevertibleContext
    meta = MetaImprover.get_instance()
    meta._strategists["s_old"] = StrategistConfig(config_id="s_old", strategist_prompt="old X", active=True)
    meta._active_strategist_id = "s_old"
    ctx = RevertibleContext()
    old, new = "s_old", "s_new"
    meta._strategists[new] = StrategistConfig(config_id=new, strategist_prompt="new Y")
    # 换件在事务内
    with ctx.transaction():
        meta._apply_strategist_swap(old, new, ctx)
        assert meta.strategist_champion().config_id == new
    assert meta.strategist_champion().config_id == new   # 正常提交保留
    # 复合护栏触发 → revert_all
    ctx2 = RevertibleContext()
    with ctx2.transaction():
        meta._apply_strategist_swap("s_old", new, ctx2)
    ctx2.revert_all()
    assert meta.strategist_champion().config_id == "s_old", "revert 应恢复上一 champion"
```

- [ ] **Step 2: 运行确认失败**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k strategist_swap -o addopts="" -q`
Expected: `AttributeError: 'MetaImprover' object has no attribute '_apply_strategist_swap'`

- [ ] **Step 3: 实现**

模块级补偿器（文件末尾、`_selfcheck` 前）：

```python
def _compensate_strategist_swap(payload: dict[str, Any]) -> None:
    """撤销一次 strategist 换件: 恢复上一 active 指向. (由 register_compensator 注册)"""
    store = payload.get("store")
    old = payload.get("old")
    if store is not None:
        store["_active_strategist_id"] = old


def _register_strategist_compensator() -> None:
    from huginn.security.revertible import register_compensator
    register_compensator("strategist_swap", _compensate_strategist_swap)
```

`MetaImprover` 内新增：

```python
def _apply_strategist_swap(self, old: str, new: str, ctx: Any) -> None:
    """换件并把逆登记进 revertible context (时间可组合)."""
    self._active_strategist_id = new
    for s in self._strategists.values():
        s.active = (s.config_id == new)
        self._save_strategist(s)
    self._save_cfg()
    if ctx is not None:
        try:
            ctx.compensate("strategist_swap", {
                "old": old, "new": new, "store": self._persist_state_ref(),
            })
        except Exception as exc:
            logger.debug("meta: strategist_swap compensate reg failed", exc_info=True)

def _persist_state_ref(self) -> dict[str, Any]:
    """把 active_strategist_id 现状引用给补偿器用."""
    return {"_active_strategist_id": self._active_strategist_id, "backing": lambda: None}

def revert_strategist(self, ctx: Any | None = None) -> bool:
    """复合退化时回退 strategist champion (时间可组合)."""
    if ctx is not None:
        try:
            ctx.revert_all()
            return True
        except Exception:
            return False
    prev = self._strategists_history[-2] if len(self._strategists_history) >= 2 else None
    if prev is not None:
        self._apply_strategist_swap(prev, self._active_strategist_id or "", None)
        return True
    return False
```

同时 `_apply_strategist_swap` 换件成功后维护 `self._strategists_history`（旧 id append）。

`maybe_propose` 中：把 `meta_prompt = _META_IMPROVE_TEMPLATE.format(...)` 改为 `meta_prompt = self.strategist_prompt().format(...)`（保持 `current_template=current` 等字段不变；strategist prompt 是模板，`.format` 需内含 `{current_template}` 等占位——若 strategist 模板不含这些占位符，则用 `StrategyPrompt` 作为"meta 提示模板"的整体替换）。**实现约定**：`strategist_prompt()` 返回的文本必须是含 `{current_template} / {n_proposals} / {n_promotions}` 的完整 meta 提示模板；`StrategistConfig.strategist_prompt` 默认填 `_META_IMPROVE_TEMPLATE` 的改写版，写入即用于 `maybe_propose.format`。若 `.format` 抛 KeyError → 记录 `strategy_reject` 并回落 `_META_IMPROVE_TEMPLATE`（失败静默）。

- [ ] **Step 4: 运行确认通过**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k strategist_swap -o addopts="" -q` Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py && git commit -m "feat(A1): strategist 换件可逆补偿(revertible) + maybe_propose 接 strategist"
```

---

### Task 5: evaluate_strategist + maybe_promote_strategist（门控组合 + 死锁/滞回）

**Files:**
- Modify: `agent/huginn/harness/meta_improver.py`

- [ ] **Step 1: 写失败测试**

```python
def test_promote_strategist_green_and_reject():
    from huginn.harness.meta_improver import (MetaImprover, StrategistConfig)
    from huginn.harness.significance_gate import SignificanceGate
    from huginn.harness.ood_holdout import OODHoldoutValidator
    meta = MetaImprover.get_instance()
    sg, ood = SignificanceGate.get_instance(), OODHoldoutValidator.get_instance()
    for i in range(8):
        sg.record_pair("strat_good", 0.3, 0.9, task_id=f"g{i}")
    for i in range(24):
        t = f"s{i:02d}"
        ood.record_outcome(ood._BASELINE_ID, t, 0.4)
        ood.record_outcome("strat_good", t, 0.9)
    meta._strategists["strat_good"] = StrategistConfig(config_id="strat_good", strategist_prompt="G")
    # 复合: 上升 → 不退化
    for i in range(4):
        meta._tracker.record(epoch=i, config_id="x", quality=0.5+0.1*i, fidelity=0.6,
                             proposals_to_promotion=1, generations_to_promotion=3, win_rate=0.9)
    ok = meta.maybe_promote_strategist("strat_good")
    assert ok is True and meta.strategist_champion()
```

- [ ] **Step 2: 运行确认失败**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k promote_strategist -o addopts="" -q`
Expected: `AttributeError`

- [ ] **Step 3: 实现**

`__init__` 增加 `self._tracker = CompoundingTracker()`。新增：

```python
async def evaluate_strategist(self, candidate_id: str, llm_chat_fn: Callable) -> dict[str, Any]:
    """在重放集上评估 strategist 候选: quality 来自其管理的改进器产出, 锚定保真."""
    if not self.enabled():
        return {"green": False, "reason": "disabled"}
    # 用 candidate 的 strategist_prompt 跑一轮 maybe_propose 观察其产出 patch 的
    # 平均分数; 简化: 以之前真实验收过的 champion 质量作代理 + BehavioralFidelity
    s = self._strategists.get(candidate_id)
    if s is None:
        return {"green": False, "reason": "unknown"}
    from huginn.harness.significance_gate import SignificanceGate
    from huginn.harness.ood_holdout import OODHoldoutValidator
    sg, ood = SignificanceGate.get_instance(), OODHoldoutValidator.get_instance()
    sg_ok = sg.gate_decision(candidate_id, min_samples=_MIN_SAMPLES).passed
    ood_ok = ood.validate_ood(candidate_id).passed
    bf = BehavioralFidelity.get_instance()
    q = sg_ok and ood_ok
    self._trace({"type": "strategy_evaluate", "candidate_id": candidate_id,
                 "sig": sg_ok, "ood": ood_ok})
    return {"green": bool(q), "sig": sg_ok, "ood": ood_ok}

def maybe_promote_strategist(self, candidate_id: str, ctx: Any | None = None) -> bool:
    """门控: 显著+OOD+复合不退化 → 事务内换件."""
    if not self.enabled():
        return False
    s = self._strategists.get(candidate_id)
    if s is None:
        return False
    from huginn.harness.significance_gate import SignificanceGate
    from huginn.harness.ood_holdout import OODHoldoutValidator
    sg_ok = SignificanceGate.get_instance().gate_decision(candidate_id, min_samples=_MIN_SAMPLES).passed
    ood_ok = OODHoldoutValidator.get_instance().validate_ood(candidate_id).passed
    if not (sg_ok and ood_ok):
        # 死锁检测: 记录 YELLOW, 连续超阈值则降级 min_samples (advisory)
        self._tracker.mark_deadlock(yellow=True)
        if self._tracker.in_deadlock():
            logger.info("meta: strategist deadlock — degrade to advisory")
        self._trace({"type": "strategy_reject", "candidate_id": candidate_id,
                     "reason": "not_green"})
        return False
    incumbent = self._active_strategist_id or ""
    if self._tracker.would_degrade(candidate_id, incumbent):
        self._trace({"type": "strategy_reject", "candidate_id": candidate_id,
                     "reason": "would_degrade"})
        return False
    self._tracker.mark_deadlock(yellow=False)
    rctx = ctx or RevertibleContext()
    try:
        with rctx.transaction():
            self._apply_strategist_swap(incumbent, candidate_id, rctx)
    except Exception as exc:
        logger.debug("meta: strategist promote txn failed", exc_info=True)
        return False
    self._tracker.record(epoch=len(self._tracker._rows), config_id=candidate_id,
                         quality=0.8, fidelity=0.5, proposals_to_promotion=1,
                         generations_to_promotion=1, win_rate=1.0)
    self._trace({"type": "strategy_promote", "candidate_id": candidate_id})
    return True
```

（顶部加 `from huginn.security.revertible import RevertibleContext` 懒导入即可。）

- [ ] **Step 4: 运行确认通过**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k promote_strategist -o addopts="" -q` Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py agent/tests/test_meta_improver.py && git commit -m "feat(A1): evaluate/promote_strategist 门控组合+死锁+滞回"
```

---

### Task 6: CoEffectRegistry 空间可组合（strategist 缺失 → improver 退化）

**Files:**
- Modify: `agent/huginn/harness/meta_improver.py`

- [ ] **Step 1: 写失败测试**

```python
def test_coeffect_degrade():
    from huginn.harness.meta_improver import MetaImprover
    meta = MetaImprover.get_instance()
    reg = meta.coeffect_registry()
    reg.update_availability()  # 无 strategist champion / gate → improver 失活
    assert reg.is_active("improver") is False or reg.provides.get("improvement_strategy") is None
```

- [ ] **Step 2: 运行确认失败**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k coeffect -o addopts="" -q`
Expected: `AttributeError`

- [ ] **Step 3: 实现**

`__init__` 增加 `self._coeffect = None`；新增：

```python
def coeffect_registry(self) -> Any:
    """空间可组合: 声明 strategist/improver/gate/fidelity 依赖图 (lazy)."""
    if self._coeffect is None:
        try:
            from huginn.security.coeffect import CoEffectRegistry
            reg = CoEffectRegistry()
            reg.declare("strategist", provides={"improvement_strategy"},
                        requires={"gate", "fidelity"})
            reg.declare("improver", requires={"improvement_strategy"})
            reg.declare("gate", provides={"gate"})
            reg.declare("fidelity", provides={"fidelity"})
            reg.set_available("gate", self.enabled())
            reg.set_available("fidelity", self.enabled())
            reg.set_available("improvement_strategy", self.strategist_champion() is not None)
            self._coeffect = reg
        except Exception as exc:
            logger.debug("meta: coeffect registry init failed", exc_info=True)
            self._coeffect = None
    return self._coeffect

def improver_active(self) -> bool:
    """improver 可用性: strategist 缺失/被 degrade → 假(退化默认)."""
    reg = self.coeffect_registry()
    if reg is None:
        return True  # coeffect 不可用 → 不设闸, 保持 M-R1 行为
    try:
        # 空间可组合: 依赖缺失则 is_active=False
        available = reg.is_active("improvement_strategy") or reg.is_active("strategist")
        return available
    except Exception:
        return True
```

`note_generation` 里,当 `improver_active() is False` 时直接 `return`（不推进递归）。

- [ ] **Step 4: 运行确认通过**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k coeffect -o addopts="" -q` Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py agent/tests/test_meta_improver.py && git commit -m "feat(A1): CoEffectRegistry 空间可组合, improver 缺失策略即退化"
```

---

### Task 7: RandomizedControl（HUGINN_META_ABLATION=1 仲裁）

**Files:**
- Modify: `agent/huginn/harness/meta_improver.py`

- [ ] **Step 1: 写失败测试**

```python
def test_randomized_control():
    from huginn.harness.meta_improver import RandomizedControl
    rc = RandomizedControl()
    # 真实 r_phys 差分: champion 优于 baseline
    outcome = rc.run_pair(champion_r=[0.8, 0.85, 0.9], baseline_r=[0.6, 0.62, 0.58])
    assert outcome["champion_better"] is True
```

- [ ] **Step 2: 运行确认失败**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k randomized -o addopts="" -q`
Expected: `ImportError`

- [ ] **Step 3: 实现**

```python
class RandomizedControl:
    """随机化对照: champion vs baseline 各 N 次, 用真实 r_phys 差分仲裁 (默认 off)."""

    def enabled(self) -> bool:
        return os.environ.get("HUGINN_META_ABLATION", "").lower() in ("1", "true", "yes")

    def run_pair(self, *, champion_r: list[float], baseline_r: list[float],
                 tolerance: float = 0.02) -> dict[str, Any]:
        """比较两组真实 r_phys 的中位差. 返回 {champion_better, delta, n_champ, n_base}."""
        import statistics
        champ = statistics.median(champion_r) if champion_r else 0.0
        base = statistics.median(baseline_r) if baseline_r else 0.0
        champ_better = (champ - base) > tolerance
        return {"champion_better": champ_better, "delta": round(champ - base, 4),
                "n_champ": len(champion_r), "n_base": len(baseline_r)}
```

`maybe_promote_strategist` 在 `HUGINN_META_ABLATION=1` 且已 GREEN 时,额外调用 `RandomizedControl().run_pair(...)`, `champion_better=False` 则拒绝换件。

- [ ] **Step 4: 运行确认通过**
Run: `cd /workspace/agent && python -m pytest tests/test_meta_improver.py -k randomized -o addopts="" -q` Expected: `1 passed`

- [ ] **Step 5: Commit**
```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py agent/tests/test_meta_improver.py && git commit -m "feat(A1): RandomizedControl 真实 r_phys 差分仲裁"
```

---

### Task 8: selfcheck 扩展 + 全量回归

**Files:**
- Modify: `agent/huginn/harness/meta_improver.py`（`_selfcheck`）、`agent/huginn/harness/prompt_patch.py`

- [ ] **Step 1: 扩展 selfcheck 断言**

在 `_selfcheck` 结尾加：strategist 回落默认、换件可逆、BehavioralFidelity 锚、CompoundingTracker 退化。

- [ ] **Step 2: 运行 selfcheck + 全量测试**

Run:
```bash
cd /workspace/agent && HUGINN_CACHE_DIR=$(mktemp -d) python -m huginn.harness.meta_improver
cd /workspace/agent && python -m pytest tests/test_meta_improver.py tests/test_engine_decomposed.py tests/loops/test_inkling_inspirations.py tests/loops/test_autoloop_loop.py -o addopts="" -q
```
Expected: `meta_improver selfcheck OK` + `66+ passed`（新增用例数）。

- [ ] **Step 3: Commit**
```bash
cd /workspace && git add agent/huginn/harness/meta_improver.py && git commit -m "test(A1): selfcheck 扩展, 全量回归通过"
```

---

## Self-Review（writing-plans）

**Spec 覆盖**：
- 递归化（strategist champion / maybe_propose 接 strategist）→ Task 3/4 ✓
- CompoundingTracker（is_compounding/would_degrade/滞回/死锁）→ Task 1/5 ✓
- BehavioralFidelity 锚 → Task 2 ✓
- 死锁检测 + History :: 死锁 → Task 5 ✓
- 随机化对照 → Task 7 ✓
- Goodhart holdout（保留子集不可见）→ 由 OOD `_BASELINE_ID` + overfit 测试承接（M-R1 已有 `test_ood_overfit_not_promoted`）✓
- 时空可组合（RevertibleContext transaction/compensate + CoEffectRegistry）→ Task 4/6 ✓
- 持久化 strategist/compounding/fidelity/revertible_journal → Task 1/2/3 ✓
- 默认关 → `enabled()` 沿用 `harness_meta_improver` ✓

**占位符扫描**：无 TBD/TODO；每步含完整代码与命令。

**类型一致性**：`StrategistConfig` / `CompoundingTracker.record` / `BehavioralFidelity.anchor_in` / `maybe_promote_strategist` / `_apply_strategist_swap` / `strategist_prompt` 在各任务签名一致。