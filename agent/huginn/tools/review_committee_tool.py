"""学术预审委员会工具 —— 调度 5 个 reviewer persona 并行审稿, 汇总修订建议.

审稿人角色 (对应 .huginn/personas/ 下的 persona 文件):
  - editor_desk_reject : 主编预筛, 给 1-10 分初筛评分
  - reviewer_1_theory  : 理论贡献质询师
  - reviewer_2_method  : 方法论审查员
  - reviewer_3_literature : 文献对话审查专家
  - reviewer_4_logic   : 逻辑链条审查师

全面审稿模式下, 调度顺序为:
  Editor → Reviewer 1 (理论) → Reviewer 3 (文献) → Reviewer 2 (方法) → Reviewer 4 (逻辑)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)

# mode -> persona 名字的映射; full 模式下按 FULL_REVIEW_ORDER 全调
MODE_PERSONA_MAP: dict[str, str] = {
    "theory": "reviewer_1_theory",
    "method": "reviewer_2_method",
    "literature": "reviewer_3_literature",
    "logic": "reviewer_4_logic",
    "desk_reject": "editor_desk_reject",
}

# 全面审稿时的调度顺序 (汇总结果也按这个顺序呈现)
# Editor 先做预筛, 然后理论 → 文献 → 方法 → 逻辑
FULL_REVIEW_ORDER: list[str] = [
    "editor_desk_reject",
    "reviewer_1_theory",
    "reviewer_3_literature",
    "reviewer_2_method",
    "reviewer_4_logic",
]


class ReviewCommitteeToolInput(BaseModel):
    paper_content: str = Field(
        ..., description="待审稿的论文全文 (纯文本或 Markdown)"
    )
    mode: Literal[
        "full", "theory", "method", "literature", "logic", "desk_reject"
    ] = Field(
        default="full",
        description=(
            "审稿模式: full=全面审稿(5 个全调); "
            "theory=只调 reviewer_1; method=只调 reviewer_2; "
            "literature=只调 reviewer_3; logic=只调 reviewer_4; "
            "desk_reject=只调 editor"
        ),
    )
    target_journal: str | None = Field(
        default=None,
        description="目标期刊名称, 用于 Editor 评估契合度; 不提供则跳过契合度评估",
    )


# 综合判断的合成 prompt, 把各 reviewer 报告喂给 LLM, 让它提炼出
# top_3_issues / revision_priority / overall_recommendation
_SYNTHESIS_SYSTEM_PROMPT = """你是学术预审委员会的汇总审稿人。你将收到若干位审稿人针对同一篇论文的审查报告。

你的任务:
1. 综合所有审稿报告, 提炼出最需要先改的三个问题 (top_3_issues)。
2. 给出修改优先级排序 (revision_priority), 按从高到低排列。
3. 给出总体投稿建议 (overall_recommendation), 只能从以下四个值中选一个:
   - accept          : 几乎无需修改, 可接受
   - minor_revision  : 小修即可
   - major_revision  : 大修
   - reject          : 建议拒稿

判断 overall_recommendation 时参考主编初筛评分: 8-10 倾向 accept/minor_revision; 5-7 倾向 major_revision; 4 分及以下倾向 reject。但要结合各 reviewer 的具体意见综合判断, 不要机械套用分数。

输出必须是严格的 JSON, 不要加 markdown 代码块标记, 不要加任何解释文字。JSON 格式如下:
{
  "top_3_issues": ["问题1", "问题2", "问题3"],
  "revision_priority": ["优先级最高的修改项", "次高", "..."],
  "overall_recommendation": "accept | minor_revision | major_revision | reject"
}"""


class ReviewCommitteeTool(HuginnTool):
    """调度 5 个 reviewer persona 并行审稿, 汇总修订建议."""

    name = "review_committee_tool"
    category = "search"
    description = (
        "学术预审委员会: 调度 5 个资深审稿人 persona (主编预筛/理论/方法/文献/逻辑) "
        "并行审稿, 输出初筛评分、各维度审查报告、最需先改的三个问题、修改优先级和总体建议。"
        "支持全面审稿 (mode=full) 或按维度单独审查。"
    )
    input_schema = ReviewCommitteeToolInput
    # 只调 LLM, 不写文件, 设为只读
    read_only = True

    async def call(
        self, args: ReviewCommitteeToolInput, context: ToolContext
    ) -> ToolResult:
        # 1. 确定要调度的 persona 列表
        persona_names = self._select_personas(args.mode)
        if not persona_names:
            return ToolResult(
                data=None,
                success=False,
                error=f"未知审稿模式: {args.mode}",
            )

        # 2. 取 persona + LLM 客户端
        try:
            personas = self._load_personas(persona_names, context)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"加载 reviewer persona 失败: {exc}",
            )

        try:
            model = self._get_model(context)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"初始化 LLM 客户端失败: {exc}",
            )

        # 3. 并行调度各 reviewer
        tasks = [
            self._run_reviewer(name, personas[name], args, model)
            for name in persona_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reviews: dict[str, str] = {}
        errors: dict[str, str] = {}
        for name, res in zip(persona_names, results):
            if isinstance(res, Exception):
                logger.warning("reviewer %s 审稿失败: %s", name, res)
                errors[name] = f"{type(res).__name__}: {res}"
            else:
                reviews[name] = res

        # 4. 解析主编初筛评分
        editor_score = self._parse_editor_score(reviews.get("editor_desk_reject"))

        # 5. 合成 top_3_issues / revision_priority / overall_recommendation
        synthesis = await self._synthesize(args, reviews, editor_score, model)

        # 6. 按 FULL_REVIEW_ORDER 排序 reviews, 保证输出顺序稳定
        ordered_reviews = {
            name: reviews[name]
            for name in FULL_REVIEW_ORDER
            if name in reviews
        }
        # 单维度模式下调用的 persona 可能不在 FULL_REVIEW_ORDER 之外 (不会发生), 兜底补上
        for name, text in reviews.items():
            if name not in ordered_reviews:
                ordered_reviews[name] = text

        data: dict[str, Any] = {
            "editor_score": editor_score,
            "reviews": ordered_reviews,
            "top_3_issues": synthesis.get("top_3_issues", []),
            "revision_priority": synthesis.get("revision_priority", []),
            "overall_recommendation": synthesis.get(
                "overall_recommendation",
                self._fallback_recommendation(editor_score),
            ),
        }
        if errors:
            data["reviewer_errors"] = errors

        return ToolResult(data=data, success=True)

    # ------------------------------------------------------------------ helpers

    def _select_personas(self, mode: str) -> list[str]:
        """根据 mode 决定要调度哪些 persona."""
        if mode == "full":
            return list(FULL_REVIEW_ORDER)
        persona = MODE_PERSONA_MAP.get(mode)
        return [persona] if persona else []

    def _load_personas(
        self, names: list[str], context: ToolContext
    ) -> dict[str, Any]:
        """通过 PersonaManager 加载需要的 persona.

        workspace 优先用 context.workspace, 没有就退到 cwd —— 这样 persona
        文件放在项目根 .huginn/personas/ 下时能被扫到.
        """
        from huginn.personas import PersonaManager

        workspace = getattr(context, "workspace", None) or None
        manager = PersonaManager(workspace=workspace) if workspace else PersonaManager()

        loaded: dict[str, Any] = {}
        missing: list[str] = []
        for name in names:
            persona = manager.get(name)
            # PersonaManager.get 找不到会回退到 default, 这里要严格判断
            if persona is None or persona.name != name:
                missing.append(name)
            else:
                loaded[name] = persona

        if missing:
            raise RuntimeError(
                f"未找到 persona: {missing}. 请确认 .huginn/personas/ 下存在对应的 .md 文件."
            )
        return loaded

    def _get_model(self, context: ToolContext) -> Any:
        """拿一个 LangChain chat model, 优先用 context.config."""
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.3, max_tokens=8000)

    async def _run_reviewer(
        self,
        persona_name: str,
        persona: Any,
        args: ReviewCommitteeToolInput,
        model: Any,
    ) -> str:
        """用单个 persona 的 system_prompt 审稿, 返回审查报告文本."""
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = persona.system_prompt
        user_prompt = self._build_user_prompt(args, persona_name)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # 优先 ainvoke, 没有就退到同步 invoke + to_thread
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)

        content = response.content if hasattr(response, "content") else str(response)
        # 个别 provider 返回 list[ContentBlock], 拼成纯文本
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return content

    def _build_user_prompt(
        self, args: ReviewCommitteeToolInput, persona_name: str
    ) -> str:
        """构造喂给 reviewer 的 user prompt."""
        lines = ["请对以下论文进行审查, 严格按照你的角色和输出格式要求给出报告。"]
        if args.target_journal:
            lines.append(f"目标期刊: {args.target_journal}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("【论文内容】")
        lines.append("=" * 60)
        lines.append(args.paper_content)
        lines.append("")
        lines.append("=" * 60)
        lines.append("请开始审查。记住: 批评要尖锐直接并指向具体段落, 禁止讨好型套话, 禁止捏造文献, 写得好的地方也要公正指出。")
        return "\n".join(lines)

    def _parse_editor_score(self, editor_report: str | None) -> int | None:
        """从主编报告里抠出初筛评分 (1-10).

        主编 persona 被要求输出 '初筛评分: X/10' 这一行, 这里用正则兜底匹配.
        """
        if not editor_report:
            return None
        # 匹配 "初筛评分: 7/10" / "初筛评分:7/10" / "初筛评分：7" 等变体
        match = re.search(
            r"初筛评分\s*[:：]\s*(\d{1,2})\s*(?:/\s*10)?",
            editor_report,
        )
        if not match:
            # 兜底: 找 "X/10" 这种最显眼的写法
            match = re.search(r"\b(\d{1,2})\s*/\s*10\b", editor_report)
        if not match:
            return None
        try:
            score = int(match.group(1))
        except (ValueError, IndexError):
            return None
        # 钳到 1-10
        if score < 1:
            return 1
        if score > 10:
            return 10
        return score

    async def _synthesize(
        self,
        args: ReviewCommitteeToolInput,
        reviews: dict[str, str],
        editor_score: int | None,
        model: Any,
    ) -> dict[str, Any]:
        """调一次 LLM 把各 reviewer 报告合成 top_3_issues / 优先级 / 总体建议."""
        from langchain_core.messages import HumanMessage, SystemMessage

        if not reviews:
            return {
                "top_3_issues": [],
                "revision_priority": [],
                "overall_recommendation": self._fallback_recommendation(editor_score),
            }

        # 拼各 reviewer 报告
        report_parts = []
        if editor_score is not None:
            report_parts.append(f"【主编初筛评分】{editor_score}/10")
        for name, text in reviews.items():
            report_parts.append(f"【{name} 报告】\n{text}")
        combined = "\n\n".join(report_parts)

        messages = [
            SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=combined),
        ]

        try:
            if hasattr(model, "ainvoke"):
                response = await model.ainvoke(messages)
            else:
                response = await asyncio.to_thread(model.invoke, messages)
            content = response.content if hasattr(response, "content") else str(response)
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            parsed = self._parse_synthesis_json(content)
            if parsed is not None:
                return parsed
        except Exception as exc:
            logger.warning("合成审稿结论失败, 退回分数兜底: %s", exc)

        # 解析失败就按分数兜底
        return {
            "top_3_issues": [],
            "revision_priority": [],
            "overall_recommendation": self._fallback_recommendation(editor_score),
        }

    def _parse_synthesis_json(self, content: str) -> dict[str, Any] | None:
        """从 LLM 回复里抠 JSON. 容忍前后多余文字和 ```json 代码块."""
        # 先剥 markdown 代码块
        text = content.strip()
        if text.startswith("```"):
            # 去掉首行 ``` 或 ```json
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        # 直接解析
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 兜底: 找第一个 { 到最后一个 } 之间的内容
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        recommendation = str(data.get("overall_recommendation", "")).strip().lower()
        if recommendation not in {"accept", "minor_revision", "major_revision", "reject"}:
            recommendation = self._fallback_recommendation(
                self._extract_score_from_data(data)
            )

        top_3 = data.get("top_3_issues", [])
        priority = data.get("revision_priority", [])
        if not isinstance(top_3, list):
            top_3 = [str(top_3)] if top_3 else []
        if not isinstance(priority, list):
            priority = [str(priority)] if priority else []

        return {
            "top_3_issues": [str(item) for item in top_3][:3],
            "revision_priority": [str(item) for item in priority],
            "overall_recommendation": recommendation,
        }

    def _extract_score_from_data(self, data: dict[str, Any]) -> int | None:
        """合成 JSON 里偶尔会塞 editor_score 字段, 兜底用一下."""
        score = data.get("editor_score")
        if isinstance(score, (int, float)):
            return int(score)
        return None

    def _fallback_recommendation(self, editor_score: int | None) -> str:
        """LLM 合成失败时, 按主编分数兜底给总体建议."""
        if editor_score is None:
            return "major_revision"
        if editor_score >= 8:
            return "minor_revision"
        if editor_score >= 5:
            return "major_revision"
        return "reject"

    def estimate_cost(self, args: ReviewCommitteeToolInput) -> dict[str, float] | None:
        # 5 个 reviewer 各调一次 LLM + 1 次合成, 估算为 6 次 LLM 调用
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.05}
