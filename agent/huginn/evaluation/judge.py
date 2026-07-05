"""Judge model 评估系统 — LLM-as-judge 双盲 arena.

JudgeEvaluator: 用独立 verification LLM 给两个 agent 输出打分, 选出 winner.
BlindArena:     双盲包装, 随机打乱 A/B 顺序, 记录历史, 维护 ELO 评分.

设计参考 Moonshine 三槽: judge 用独立 LLM, 避免被评模型自评的确认偏差.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from huginn.evaluation.arena_store import ArenaRecord, ArenaStore, now_ts

logger = logging.getLogger(__name__)

# ELO 参数
_ELO_K = 32.0
_ELO_DEFAULT = 1000.0


@dataclass
class JudgeResult:
    """一次 judge 评估的结果."""

    winner: str  # "A" / "B" / "tie"
    reasoning: str
    scores: dict[str, float] = field(default_factory=dict)  # {"A": 8.5, "B": 7.2}
    raw_response: str = ""


class JudgeEvaluator:
    """用独立 LLM 做 judge, 评估两个 agent 输出.

    model_router 不为空时优先取 verification model; 否则用传入的 model.
    两者都没有时降级到规则比较 (按长度+关键词), 不让流程卡死.
    """

    JUDGE_PROMPT = """你是一个材料科学专家，评估以下两个回答的质量。

评判标准: {criteria}

回答 A:
{answer_a}

回答 B:
{answer_b}

请从以下维度打分 (0-10): 准确性, 完整性, 推理深度, 实用性.
然后给出最终判断: 哪个回答更好 (A / B / tie), 并简述理由.

严格按以下 JSON 格式返回, 不要加任何额外文字:
{{"scores_a": {{"accuracy": <float>, "completeness": <float>, "reasoning": <float>, "utility": <float>}}, "scores_b": {{...}}, "winner": "A"|"B"|"tie", "reasoning": "<简要理由>"}}"""

    def __init__(
        self,
        model: Any = None,
        model_router: Any = None,
        criteria: str = "准确性、完整性、推理深度、实用性",
    ) -> None:
        self._model = model
        self._router = model_router
        self._criteria = criteria

    def _resolve_model(self) -> Any:
        """拿 judge 用的 LLM. router 优先, 没有就用裸 model."""
        if self._router is not None:
            try:
                return self._router.select_verification()
            except Exception:
                logger.warning("select_verification 失败, 退回裸 model", exc_info=True)
        return self._model

    async def evaluate(
        self, answer_a: str, answer_b: str, criteria: str | None = None
    ) -> JudgeResult:
        """评估两个回答, 返回 winner + reasoning + scores."""
        llm = self._resolve_model()
        crit = criteria or self._criteria

        if llm is None:
            # 降级: 没 LLM 就用朴素启发式比一下, 别让流程卡死
            return self._heuristic_judge(answer_a, answer_b, crit)

        prompt = self.JUDGE_PROMPT.format(
            criteria=crit, answer_a=answer_a, answer_b=answer_b
        )
        try:
            from langchain_core.messages import HumanMessage

            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = str(resp.content)
        except Exception as exc:
            logger.warning("judge LLM 调用失败, 退回启发式: %s", exc)
            return self._heuristic_judge(answer_a, answer_b, crit)

        return self._parse_judge(raw)

    def _parse_judge(self, raw: str) -> JudgeResult:
        """从 LLM 响应里抠出 winner/scores/reasoning, 容错解析."""
        text = raw.strip()
        # 去掉 markdown ```json 围栏
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            data = json.loads(text)
            winner = str(data.get("winner", "tie")).strip().upper()
            if winner not in ("A", "B", "TIE"):
                winner = "TIE" if "tie" in winner.lower() else "tie"
            sa = data.get("scores_a", {})
            sb = data.get("scores_b", {})
            # 综合分 = 各维度均值, 方便排序
            avg_a = sum(sa.values()) / len(sa) if sa else 0.0
            avg_b = sum(sb.values()) / len(sb) if sb else 0.0
            reasoning = str(data.get("reasoning", ""))
            return JudgeResult(
                winner=winner.lower(),
                reasoning=reasoning,
                scores={"A": round(avg_a, 2), "B": round(avg_b, 2)},
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError):
            # 兜底: 正则抠 winner
            import re

            m = re.search(r"winner[:\s]*\"?(A|B|tie)\"?", raw, re.IGNORECASE)
            winner = m.group(1).lower() if m else "tie"
            return JudgeResult(
                winner=winner,
                reasoning="raw parse fallback",
                scores={"A": 0.0, "B": 0.0},
                raw_response=raw,
            )

    @staticmethod
    def _heuristic_judge(answer_a: str, answer_b: str, criteria: str) -> JudgeResult:
        """没 LLM 时的降级: 按长度 + 材料科学关键词密度比.

        ponytail: 纯启发式, 只在 LLM 完全不可用时兜底, 不该当真做排名.
        """
        matsci_kw = ("材料", "相图", "晶体", "电子结构", "DFT", "势函数", "缺陷", "扩散")
        score = lambda t: (  # noqa: E731
            min(len(t) / 500.0, 1.0) * 5
            + sum(t.count(k) for k in matsci_kw) * 0.5
        )
        sa = round(min(score(answer_a), 10.0), 2)
        sb = round(min(score(answer_b), 10.0), 2)
        if abs(sa - sb) < 0.5:
            winner = "tie"
        else:
            winner = "a" if sa > sb else "b"
        return JudgeResult(
            winner=winner,
            reasoning="heuristic fallback (no judge LLM available)",
            scores={"A": sa, "B": sb},
        )


# ── ELO ────────────────────────────────────────────────────────


def expected_score(elo_a: float, elo_b: float) -> float:
    """A 对 B 的预期胜率. 0~1."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_a: float, elo_b: float, result: str, k: float = _ELO_K
) -> tuple[float, float]:
    """根据对局结果更新 ELO.

    result: "A"=A赢, "B"=B赢, "tie"=平.
    返回 (new_elo_a, new_elo_b).
    """
    exp_a = expected_score(elo_a, elo_b)
    if result.upper() == "A":
        score_a = 1.0
    elif result.upper() == "B":
        score_a = 0.0
    else:
        score_a = 0.5
    score_b = 1.0 - score_a
    new_a = elo_a + k * (score_a - exp_a)
    new_b = elo_b + k * (score_b - (1.0 - exp_a))
    return round(new_a, 2), round(new_b, 2)


class BlindArena:
    """双盲 arena: 随机打乱 A/B 顺序, 记录历史, 维护 ELO."""

    def __init__(
        self,
        judge: JudgeEvaluator | None = None,
        store: ArenaStore | None = None,
        seed: int | None = None,
    ) -> None:
        self._judge = judge or JudgeEvaluator()
        self._store = store
        self._rng = random.Random(seed)
        # 内存里的 ELO 缓存, 没接 store 时也能用
        self._elo: dict[str, float] = {}
        if self._store is not None:
            # 启动时从历史里恢复 ELO
            self._elo = dict(self._store.latest_elo())

    def get_elo(self, model: str) -> float:
        return self._elo.get(model, _ELO_DEFAULT)

    def leaderboard(self) -> list[tuple[str, float]]:
        """ELO 从高到低排序."""
        return sorted(self._elo.items(), key=lambda x: -x[1])

    async def battle(
        self,
        model_a: str,
        answer_a: str,
        model_b: str,
        answer_b: str,
        criteria: str | None = None,
    ) -> ArenaRecord:
        """跑一场双盲对决, 更新 ELO, 记录历史.

        双盲: 随机决定谁当 "A" 谁当 "B" 喂给 judge, 避免 judge 偏向先出现的.
        记录时还原成真实模型名.
        """
        # 随机打乱顺序
        flip = self._rng.random() < 0.5
        if flip:
            judge_first, judge_second = answer_b, answer_a
            first_model, second_model = model_b, model_a
        else:
            judge_first, judge_second = answer_a, answer_b
            first_model, second_model = model_a, model_b

        result = await self._judge.evaluate(
            judge_first, judge_second, criteria=criteria
        )

        # judge 返回的 winner 是相对于 "第一参数" 的 A/B
        # flip 过的话要把 winner 映射回真实模型
        judge_winner = result.winner  # "a" / "b" / "tie" (相对 judge 第一参数)
        if judge_winner == "tie":
            true_winner = "tie"
        elif flip:
            # judge 看到的 A 其实是 model_b, B 是 model_a
            true_winner = "b" if judge_winner == "a" else "a"
        else:
            true_winner = judge_winner

        # ELO 更新 (用真实模型)
        elo_a = self.get_elo(model_a)
        elo_b = self.get_elo(model_b)
        new_elo_a, new_elo_b = update_elo(elo_a, elo_b, true_winner)
        self._elo[model_a] = new_elo_a
        self._elo[model_b] = new_elo_b

        # scores 也还原成真实模型名
        scores = {
            model_a: result.scores.get("A" if not flip else "B", 0.0),
            model_b: result.scores.get("B" if not flip else "A", 0.0),
        }

        rec = ArenaRecord(
            timestamp=now_ts(),
            model_a=model_a,
            model_b=model_b,
            winner=true_winner.upper(),
            reasoning=result.reasoning,
            scores=scores,
            elo_a=new_elo_a,
            elo_b=new_elo_b,
            meta={
                "judge_raw": result.raw_response[:500],
                "flipped": flip,
            },
        )

        if self._store is not None:
            self._store.record(rec)

        return rec
