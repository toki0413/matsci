"""C-Space (Capability Workspace) — 外部化全局工作区 (对齐 GWT/J-Space + S-Space).

存在目的 (为什么一个科考 agent 需要它):
  长程科研的硬伤是"此刻在想的、想过的、能证明的"三者不一致。Anthropic 的 J-Space(内部
  全局工作区, 用 J-lens 读)给我们的启发是: 推理依赖一小撮**当前在场**的概念。本模块把
  "在场"**外置**成可读/可控/可推理/可证伪的工作区, 与既有组件咬合:
    - concept     : 概念在场 —— kg/命题 + trace 数值,
    - state       : 状态在场 —— law_model.world_model_card + reconcile 可对账 (S-Space 近似),
    - interaction : 可读激活 —— interaction_explain 交互基元 (结构性可读).
  完整设计见 docs/cspace-workspace-spec.md。

  核心操作 (对齐 J-Space 的 report/control/reason/audit):
    - :meth:`CSpace.probe`    report: 按词扩散激活, 返回点亮排序.
    - :meth:`CSpace.pin` / `suppress`   control: 钉住 / 压制一个在场.
    - :meth:`CSpace.readout`  reason: 把点亮在场折叠成紧凑上下文.
    - :meth:`CSpace.broadcast` audit: 过 claim_grounding 才落 trace; 无凭据永不确认在场.

  治理规则 (机械可测, 见测试节 10):
    - 每个 Being 必须有 source 证据指针; 缺证据 → falsifiable=False → probe 激活恒 0.
    - 广播/进 trace 必须过声明门禁; 拿 activation 只做召回, 不替代证据.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

# 允许的 Being kind (在场通道)
KINDS = ("concept", "state", "interaction")
# 读上下文时激活阈值以下的在场不写进 readout(避免刷屏)
_DEF_READOUT_ACT = 0.0


@dataclass
class Being:
    """工作区里的一个"在场"单元 (concept/state/interaction)."""
    id: str
    kind: str                                              # KINDS 之一
    payload: dict = field(default_factory=dict)
    source: str = ""                                       # 证据指针(进 trace/门禁)
    falsifiable: bool = True                               # 无证据声称 → 不确认在场
    activation: float = 0.0
    pinned: bool = False
    suppressed: bool = False

    def as_trace(self) -> str:
        return json.dumps({"id": self.id, "kind": self.kind,
                           **self.payload, "source": self.source}, ensure_ascii=False)


class CSpace:
    """外部化全局工作区编排器 (单入口: 读/控/推理/审计)."""

    def __init__(self, *, verify: Callable[[str, list[str]], dict] | None = None) -> None:
        self.beings: dict[str, Being] = {}
        # 关联图: spreading activation 的召回边 (id -> {neighbor: weight})
        self.assoc: dict[str, dict[str, float]] = defaultdict(dict)
        self.trace: list[str] = []
        self.verify = verify or _default_verifier()

    # ── 注册 / 关联 ──────────────────────────────────────────────
    def register(self, id: str, kind: str, *, payload: dict | None = None,
                 source: str = "", falsifiable: bool = True) -> Being:
        if kind not in KINDS:
            raise ValueError(f"unknown kind '{kind}'; 允许 {KINDS}")
        b = Being(id=id, kind=kind, payload=dict(payload or {}),
                  source=source, falsifiable=falsifiable and bool(source))
        self.beings[id] = b
        return b

    def associate(self, a: str, b: str, weight: float = 1.0) -> None:
        self.assoc[a][b] = weight
        self.assoc[b][a] = weight

    # ── control: 钉住 / 压制 ────────────────────────────────────
    def pin(self, id: str) -> "CSpace":
        if id in self.beings:
            self.beings[id].pinned = True
        return self

    def suppress(self, id: str) -> "CSpace":
        if id in self.beings:
            self.beings[id].suppressed = True
        return self

    def wake(self, id: str | None = None) -> "CSpace":
        """解除钉住/压制 (id 为 None 则全部重置)."""
        keys = [id] if id is not None else list(self.beings)
        for k in keys:
            if k in self.beings:
                self.beings[k].pinned = False
                self.beings[k].suppressed = False
        return self

    # ── report: 激活扩散 → 点亮排序 ─────────────────────────────
    def _seed(self, text: str) -> dict[str, float]:
        """按词把命中的 Being 点亮(仅召回, 不替代证据)."""
        low = (text or "").lower()
        tokens = set(re.findall(r"[0-9a-z_]+", low))
        act: dict[str, float] = {}
        for bid, b in self.beings.items():
            hay = " ".join([bid, b.kind, *map(str, b.payload.keys())]).lower()
            hits = sum(1 for t in tokens if t and t in hay)
            if hits:
                act[bid] = float(hits)
        return act

    def _spread(self, act: dict[str, float], *, steps: int = 2) -> dict[str, float]:
        out = dict(act)
        for _ in range(steps):
            nxt = dict(out)
            for bid, val in out.items():
                for nb, w in self.assoc.get(bid, {}).items():
                    nxt[nb] = max(nxt.get(nb, 0.0), val * w)
            out = nxt
        return out

    def probe(self, text: str) -> dict[str, Any]:
        """report: 运行一次读, 回写每个 Being 的 activation, 返回点亮排序."""
        act = self._spread(self._seed(text))
        for bid, b in self.beings.items():
            v = float(act.get(bid, 0.0))
            if b.suppressed:
                v = -1.0                     # 压制优先于钉住: 不能靠钉住"强行想到被压下之物"
            elif b.pinned:
                v = 9.99
            if not b.falsifiable:
                v = 0.0                    # 治理: 无凭据的声称不确认在场
            b.activation = v
        lit = sorted((b for b in self.beings.values() if b.activation > 0.0),
                     key=lambda b: -b.activation)
        return {"lit": [b.id for b in lit], "activation": {b.id: b.activation for b in lit}}

    def readout(self, *, min_act: float = _DEF_READOUT_ACT) -> dict[str, Any]:
        """reason: 把点亮在场折叠成紧凑上下文(供提示/决策引用)."""
        lit = [b for b in self.beings.values() if b.activation > min_act and b.falsifiable]
        lit.sort(key=lambda b: -b.activation)
        return {
            "at_hand": [
                {"id": b.id, "kind": b.kind, "activation": round(b.activation, 3),
                 "source": b.source}
                for b in lit
            ],
            "pending_no_source": [b.id for b in self.beings.values()
                                  if not b.falsifiable],
        }

    # ── audit: 广播必须过声明门禁才落 trace ─────────────────────
    def broadcast(self, statement: str) -> dict[str, Any]:
        """把结论声明广播进工作区; 未过 grounding 门禁则拒绝, 不进 trace."""
        g = self.verify(statement, self.trace)
        if g["verdict"] != "pass":
            return {"verified": False, "unsubstantiated": g.get("unsubstantiated", [])}
        self.trace.append(json.dumps({"type": "cspace_broadcast", "claim": statement},
                                     ensure_ascii=False))
        return {"verified": True}

    # ── 便捷: 从既有治理组件把"在场"灌进来 ─────────────────────
    def add_state(self, being_id: str, model, *, state: dict | None = None,
                  action: dict | None = None) -> Being:
        """从 LawModel 灌一条 S-Space 状态在场 (falsifiable=law card 可对账)."""
        from huginn.research.law_model import LawAction, LawState, world_model_card
        card = world_model_card(model)
        if state is not None:
            pred = model.predict(LawState(dict(state), domain=getattr(model, "domain", "")),
                                 LawAction(dict(action or {}) or {}))
            payload = {"law": model.law(), "predicted": pred.as_dict(), "card": card}
        else:
            payload = {"law": model.law(), "card": card}
        # 域科学契约(量纲/有效域)随状态 Being 携带 —— "状态在场"自证其法律约束.
        payload["scientific_contract"] = card.get("contract")
        # 具身可信: law 三件套在 → 可通过 reconcile 对账 → 可证伪
        b = self.register(being_id, "state", payload=payload,
                          source=card["truth_reference"], falsifiable=card["falsifiable"])
        return b

    def add_interaction(self, being_id: str, prim: dict[tuple, float]) -> Being:
        """从 interaction_explain 灌一条可读交互在场 (结构性基元, 天然可证伪)."""
        from huginn.research.interaction_explain import interaction_trace
        payload = {"primitives": interaction_trace(prim)}
        return self.register(being_id, "interaction", payload=payload,
                             source="interaction_explain (Σφ 结构)", falsifiable=True)


def _default_verifier() -> Callable[[str, list[str]], dict]:
    """缺省声明门禁: 统一从产品单一实现取 (program.grounding_verifier)."""
    from huginn.research.program import grounding_verifier
    return grounding_verifier()


# 兼容性别名: 入口唯一 (读/控/推理/审计 都经 CSpace).
GlobalWorkspace = CSpace
__all__ = ["Being", "CSpace", "GlobalWorkspace", "KINDS"]