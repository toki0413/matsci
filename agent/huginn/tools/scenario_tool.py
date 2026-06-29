"""场景工具选择器 —— 元工具, 帮 LLM 一次性拿到某场景的工具集和调用顺序.

LLM 识别用户意图后, 调一次 scenario_tool 就能拿到该场景推荐工具列表 +
有序调用链 + workflow 模板, 不用每次都从头逐个挑工具. 这样能显著减少
LLM 在工具选择上的 token 消耗和决策延迟.

实现思路:
  1. 预置 14 个常见材料科研场景 → 工具 bundle 映射
  2. 优先用 LLM 把用户描述匹配到最接近的 scenario_type
  3. LLM 失败 / 低置信度时, 退到关键词兜底匹配
  4. 返回推荐工具 + 有序调用链 + workflow 模板 + rationale
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


# ---------------------------------------------------------------------------
# 预置场景 → 工具 bundle
# ---------------------------------------------------------------------------
# 每个场景包含:
#   recommended_tools : 无序的工具名清单 (供 LLM 参考)
#   tool_chain        : 建议的调用顺序 (从前到后)
#   workflow_template : 简短的 workflow 模板描述
#   rationale         : 为什么选这些工具
#
# 工具名严格对齐 tools/__init__.py 注册的真实工具, 不硬编码不存在的工具.
# defect_calculation 没用 defect_tool (该项目里没有独立 defect_tool),
# 而是复用 structure_tool 的 create_defect action.
SCENARIO_TOOL_BUNDLES: dict[str, dict[str, Any]] = {
    "dft_structure_optimization": {
        "recommended_tools": [
            "structure_tool",
            "vasp_tool",
            "validate_tool",
        ],
        "tool_chain": [
            "structure_tool",
            "vasp_tool",
            "validate_tool",
        ],
        "workflow_template": (
            "structure_tool(分析/准备结构) → vasp_tool(relax) → "
            "validate_tool(校验受力/能量收敛)"
        ),
        "rationale": (
            "结构优化需要先解析输入结构, 再用 VASP 做 ISIF=3 弛豫, "
            "最后校验 EDIFF/EDIFFG 是否真收敛."
        ),
    },
    "dft_band_calculation": {
        "recommended_tools": [
            "structure_tool",
            "vasp_tool",
            "validate_tool",
            "report_tool",
        ],
        "tool_chain": [
            "structure_tool",
            "vasp_tool",
            "validate_tool",
            "report_tool",
        ],
        "workflow_template": (
            "structure_tool(准备结构) → vasp_tool(relax → scf → band) → "
            "validate_tool(校验能带/DOS) → report_tool(出报告)"
        ),
        "rationale": (
            "能带计算是 relax→SCF→band 三段式, 中间要校验 Fermi 能级和带隙, "
            "最后用 report_tool 汇总结果."
        ),
    },
    "aimd_simulation": {
        "recommended_tools": [
            "structure_tool",
            "lammps_tool",
            "ml_potential_tool",
            "validate_tool",
        ],
        "tool_chain": [
            "structure_tool",
            "ml_potential_tool",
            "lammps_tool",
            "validate_tool",
        ],
        "workflow_template": (
            "structure_tool(准备结构) → ml_potential_tool(加载/校验势函数) → "
            "lammps_tool(AIMD/MD 跑轨迹) → validate_tool(校验温度/能量漂移)"
        ),
        "rationale": (
            "AIMD 通常用 MLP 加速, 需要先确认势函数可用, 再丢给 LAMMPS 跑轨迹, "
            "最后看能量/温度是否稳定."
        ),
    },
    "defect_calculation": {
        "recommended_tools": [
            "structure_tool",
            "vasp_tool",
            "validate_tool",
        ],
        "tool_chain": [
            "structure_tool",
            "vasp_tool",
            "structure_tool",
            "vasp_tool",
            "validate_tool",
        ],
        "workflow_template": (
            "structure_tool(分析完美结构) → vasp_tool(relax 完美) → "
            "structure_tool(create_defect 建超胞+缺陷) → "
            "vasp_tool(relax 缺陷) → validate_tool(校验形成能)"
        ),
        "rationale": (
            "点缺陷计算要先驰豫完美结构, 再造缺陷超胞, 跑缺陷驰豫, "
            "最后校验形成能和 charge transition level. "
            "structure_tool 的 create_defect action 已经覆盖造缺陷的活."
        ),
    },
    "ml_potential_training": {
        "recommended_tools": [
            "vasp_tool",
            "ml_potential_tool",
            "active_learning_tool",
            "validate_tool",
        ],
        "tool_chain": [
            "vasp_tool",
            "ml_potential_tool",
            "active_learning_tool",
            "validate_tool",
        ],
        "workflow_template": (
            "vasp_tool(DFT 标定训练集) → ml_potential_tool(训练 NEP/SNAP/...) → "
            "active_learning_tool(主动学习扩样) → validate_tool(校验势函数精度)"
        ),
        "rationale": (
            "训练 MLP 需要 DFT 标定数据, 训练完用主动学习迭代扩样, "
            "最后用 validate_tool 校验能量/受力 RMSE."
        ),
    },
    "literature_review": {
        "recommended_tools": [
            "web_search_tool",
            "rag_tool",
            "science_literature_search_arxiv",
            "science_literature_search_openalex",
            "gap_analysis_tool",
            "report_tool",
        ],
        "tool_chain": [
            "science_literature_search_arxiv",
            "science_literature_search_openalex",
            "web_search_tool",
            "rag_tool",
            "gap_analysis_tool",
            "report_tool",
        ],
        "workflow_template": (
            "science_literature_search_arxiv + openalex(学术库检索) → "
            "web_search_tool(补充网络资料) → rag_tool(本地知识库检索) → "
            "gap_analysis_tool(识别研究空白) → report_tool(综述报告)"
        ),
        "rationale": (
            "文献调研要先用 arXiv/OpenAlex 拉论文, 再补网络搜索和本地 RAG, "
            "最后用 gap_analysis 找研究空白并出综述."
        ),
    },
    "paper_review": {
        "recommended_tools": [
            "review_committee_tool",
            "report_tool",
        ],
        "tool_chain": [
            "review_committee_tool",
            "report_tool",
        ],
        "workflow_template": (
            "review_committee_tool(5 个 reviewer 并行审稿) → "
            "report_tool(汇总修订建议报告)"
        ),
        "rationale": (
            "学术预审走 review_committee_tool 的 5 reviewer 链路, "
            "拿到 top_3_issues / 修订优先级后用 report_tool 出报告."
        ),
    },
    "hypothesis_generation": {
        "recommended_tools": [
            "hypothesis_generator_tool",
            "gap_analysis_tool",
            "design_plan_tool",
        ],
        "tool_chain": [
            "gap_analysis_tool",
            "hypothesis_generator_tool",
            "design_plan_tool",
        ],
        "workflow_template": (
            "gap_analysis_tool(从文献找空白) → "
            "hypothesis_generator_tool(生成可证伪假设 + 映射 workflow) → "
            "design_plan_tool(出执行计划)"
        ),
        "rationale": (
            "假设生成要走 '文献空白 → 科学假设 → 可执行 workflow' 三步, "
            "hypothesis_generator_tool 内部已串好, 配合 gap_analysis 和 design_plan."
        ),
    },
    "materials_screening": {
        "recommended_tools": [
            "materials_database_tool",
            "structure_tool",
            "active_learning_tool",
            "gp_tool",
            "report_tool",
        ],
        "tool_chain": [
            "materials_database_tool",
            "structure_tool",
            "gp_tool",
            "active_learning_tool",
            "report_tool",
        ],
        "workflow_template": (
            "materials_database_tool(MP/OQMD 拉候选) → "
            "structure_tool(解析/筛结构) → gp_tool(代理模型预测) → "
            "active_learning_tool(下一批实验建议) → report_tool(排名报告)"
        ),
        "rationale": (
            "材料筛选先从数据库拉候选, 用 GP 代理模型快速预测性质, "
            "再用主动学习建议下一批, 最后输出排名."
        ),
    },
    "mechanical_properties": {
        "recommended_tools": [
            "vasp_tool",
            "lammps_tool",
            "ml_potential_tool",
            "validate_tool",
            "report_tool",
        ],
        "tool_chain": [
            "vasp_tool",
            "ml_potential_tool",
            "lammps_tool",
            "validate_tool",
            "report_tool",
        ],
        "workflow_template": (
            "vasp_tool(DFT 标定/弹性常数) → ml_potential_tool(训练势函数) → "
            "lammps_tool(大体系应力-应变) → validate_tool(校验力学稳定性) → "
            "report_tool(出报告)"
        ),
        "rationale": (
            "力学性质常需大体系, 用 DFT 标定 + MLP 加速, 再跑 LAMMPS 应力应变, "
            "校验 Born 稳定性后出报告."
        ),
    },
    "high_throughput_screening": {
        "recommended_tools": [
            "materials_database_tool",
            "high_throughput_tool",
            "gp_tool",
            "active_learning_tool",
            "report_tool",
        ],
        "tool_chain": [
            "materials_database_tool",
            "high_throughput_tool",
            "gp_tool",
            "active_learning_tool",
            "report_tool",
        ],
        "workflow_template": (
            "materials_database_tool(批量拉候选) → "
            "high_throughput_tool(批量提交/聚合) → gp_tool(代理模型加速) → "
            "active_learning_tool(下一批建议) → report_tool(排名报告)"
        ),
        "rationale": (
            "高通量筛选要走批量化路径: 数据库批量拉候选, high_throughput_tool 批量跑, "
            "GP 代理加速, 主动学习迭代, 最后出排名."
        ),
    },
    "synthesis_planning": {
        "recommended_tools": [
            "active_learning_tool",
            "gap_analysis_tool",
            "report_tool",
        ],
        "tool_chain": [
            "gap_analysis_tool",
            "active_learning_tool",
            "report_tool",
        ],
        "workflow_template": (
            "gap_analysis_tool(从历史实验找空白) → "
            "active_learning_tool(recommend 下一批合成条件) → "
            "report_tool(合成建议报告)"
        ),
        "rationale": (
            "合成规划用 active_learning_tool 在历史实验上做贝叶斯推荐, "
            "配合 gap_analysis 找还没探索的区域, 最后出报告."
        ),
    },
    "plasma_simulation": {
        "recommended_tools": [
            "plasma_tool",
            "structure_tool",
            "report_tool",
        ],
        "tool_chain": [
            "plasma_tool",
            "structure_tool",
            "plasma_tool",
            "report_tool",
        ],
        "workflow_template": (
            "plasma_tool(sheath_model 算基础参数) → "
            "structure_tool(壁材料/靶材结构) → "
            "plasma_tool(pic/fluid/arc 主仿真 + transport 后处理) → "
            "report_tool(等离子体仿真报告)"
        ),
        "rationale": (
            "等离子体仿真先用 plasma_tool 算 Debye 长度/鞘层电位定参数, "
            "再用结构工具确认壁材料, 然后跑 PIC/MHD/弧等离子体主仿真, "
            "最后做输运系数后处理并出报告."
        ),
    },
    "fusion_plasma_analysis": {
        "recommended_tools": [
            "plasma_tool",
            "materials_database_tool",
            "report_tool",
        ],
        "tool_chain": [
            "materials_database_tool",
            "plasma_tool",
            "plasma_tool",
            "report_tool",
        ],
        "workflow_template": (
            "materials_database_tool(查壁材料/面向等离子体材料) → "
            "plasma_tool(transport_coefficients 算 Spitzer 输运) → "
            "plasma_tool(wave_dispersion 分析 Alfvén/whistler) → "
            "report_tool(聚变等离子体分析报告)"
        ),
        "rationale": (
            "聚变等离子体分析要先从材料库查面向等离子体材料 (W/Be/CFC), "
            "再用 plasma_tool 算高温下的 Spitzer 输运和波色散, 最后汇总. "
            "等离子体参数通常 n~1e20, T~10keV, B~5T."
        ),
    },
    "reaction_pathway": {
        "recommended_tools": [
            "structure_tool",
            "ml_potential_tool",
            "neb_tool",
            "report_tool",
        ],
        "tool_chain": [
            "structure_tool",
            "ml_potential_tool",
            "neb_tool",
            "neb_tool",
            "report_tool",
        ],
        "workflow_template": (
            "structure_tool(准备初末态结构) → "
            "ml_potential_tool(relax 初末态, 快速弛豫) → "
            "neb_tool(neb 跑 CI-NEB 找最小能量路径) → "
            "neb_tool(mep_analyze 算势垒/能量剖面) → "
            "report_tool(反应路径报告)"
        ),
        "rationale": (
            "反应路径/相变路径分析需要先弛豫初末态, 再用 NEB 找最小能量路径, "
            "然后做 MEP 分析拿正反向势垒和鞍点能量. ml_potential_tool 比 VASP "
            "快几个数量级, 适合做初筛; 需要精确势垒再换 vasp_tool 重跑."
        ),
    },
    "diffusion_barrier": {
        "recommended_tools": [
            "structure_tool",
            "neb_tool",
            "vasp_tool",
            "report_tool",
        ],
        "tool_chain": [
            "structure_tool",
            "neb_tool",
            "vasp_tool",
            "report_tool",
        ],
        "workflow_template": (
            "structure_tool(建扩散初末态: 占位 + 间隙位) → "
            "neb_tool(neb 用 vasp 评估器跑 NEB) → "
            "vasp_tool(对鞍点做精确 SCF 单点) → "
            "report_tool(扩散势垒报告)"
        ),
        "rationale": (
            "扩散势垒要精确, 通常用 DFT 评估. 先用 structure_tool 造扩散初末态, "
            "neb_tool 内部调 vasp_tool 做 NEB 各 image 单点, 鞍点处再用 "
            "vasp_tool 跑一次精细 SCF 校验能量. 扩散势垒一般在 0.1-2 eV."
        ),
    },
}


# 关键词兜底匹配表 (按优先级排序, 命中即返回)
# 用中文/英文关键词覆盖最常见的说法. LLM 不可用时也能撑住.
# 关键词尽量用单字词或短词组, 避免 "优化结构" 这种被空格/标点断开的写法漏匹配.
_KEYWORD_FALLBACK: list[tuple[list[str], str]] = [
    (["结构优化", "优化结构", "晶体结构", "弛豫", "relax", "structure optimization"], "dft_structure_optimization"),
    (["能带", "band", "dos", "电子结构", "electronic structure"], "dft_band_calculation"),
    (["aimd", "ab initio md", "第一性原理分子动力学", "从头分子动力学"], "aimd_simulation"),
    (["defect", "缺陷", "vacancy", "空位", "substitution", "替位", "interstitial", "间隙"], "defect_calculation"),
    (["训练势函数", "mlp", "ml potential", "机器学习势", "nep", "snap", "gap ", "ace ", "train potential"], "ml_potential_training"),
    (["文献", "literature", "综述", "review paper", "调研"], "literature_review"),
    (["审稿", "paper review", "review my paper", "审查", "预审", "manuscript"], "paper_review"),
    (["假设", "hypothesis", "research idea", "科学问题"], "hypothesis_generation"),
    (["筛选材料", "screening materials", "材料筛选", "screen candidates"], "materials_screening"),
    (["力学", "mechanical", "弹性", "elastic", "应力", "stress", "应变", "strain", "硬度", "hardness"], "mechanical_properties"),
    (["高通量", "high throughput", "htvs", "大规模筛选"], "high_throughput_screening"),
    (["合成", "synthesis", "实验设计", "doe", "next experiment"], "synthesis_planning"),
    (["等离子体", "plasma", "pic", "mhd", "鞘层", "sheath", "弧等离子", "arc plasma", "电弧"], "plasma_simulation"),
    (["聚变", "fusion", "tokamak", "托卡马克", "等离子体材料", "plasma facing"], "fusion_plasma_analysis"),
    (["反应路径", "reaction pathway", "mep", "最小能量路径", "过渡态", "transition state", "neb"], "reaction_pathway"),
    (["扩散势垒", "diffusion barrier", "扩散路径", "diffusion pathway", "migration barrier", "扩散能垒"], "diffusion_barrier"),
]


_SCENARIO_MATCH_SYSTEM_PROMPT = """你是材料科研场景分类器. 你的任务是把用户的自然语言场景描述映射到最接近的预设场景类型.

预设场景类型 (只能从这些里选一个):
- dft_structure_optimization : DFT 结构优化 (弛豫晶体结构)
- dft_band_calculation       : DFT 能带/DOS 计算
- aimd_simulation            : 从头分子动力学 (AIMD)
- defect_calculation         : 点缺陷计算 (空位/替位/间隙)
- ml_potential_training      : 机器学习势函数训练
- literature_review          : 文献综述/调研
- paper_review               : 论文审查/预审
- hypothesis_generation      : 科学假设生成
- materials_screening        : 材料筛选
- mechanical_properties      : 力学性质计算
- high_throughput_screening  : 高通量筛选
- synthesis_planning         : 合成规划/实验设计
- plasma_simulation          : 等离子体仿真 (PIC/MHD/鞘层/弧等离子体)
- fusion_plasma_analysis     : 聚变等离子体分析 (输运/波/面向等离子体材料)
- reaction_pathway           : 反应路径/最小能量路径 (NEB/MEP/过渡态)
- diffusion_barrier          : 扩散势垒 (原子扩散/迁移路径)

判断要点:
1. 看用户的核心动作 (优化/算能带/跑 MD/查文献/审稿/...)
2. 看对象 (结构/缺陷/势函数/论文/材料库/...)
3. 拿不准时选最接近的一个, 不要回 "unknown"

输出必须是严格的 JSON, 不要加 markdown 代码块标记, 不要加任何解释文字. 格式:
{
  "scenario_type": "上面 14 个之一",
  "confidence": 0.0-1.0 之间的浮点数,
  "rationale": "一句话说明为什么选这个场景"
}"""


class ScenarioToolInput(BaseModel):
    scenario: str = Field(
        default="",
        description=(
            "用户场景描述, 自然语言. 例如 '我要优化 Si 的晶体结构' / "
            "'帮我调研高熵合金文献' / '审查我的论文'. "
            "predict action 下可留空, 纯靠历史预测."
        ),
    )
    action: Literal["match", "predict"] = Field(
        default="match",
        description=(
            "match: 匹配场景并返回工具 bundle (默认). "
            "predict: 基于历史预测 top-3 意图, 不阻塞, 纯预测."
        ),
    )


class ScenarioTool(HuginnTool):
    """元工具: 根据用户场景描述, 返回该场景的推荐工具集 + 调用链 + workflow 模板."""

    name = "scenario_tool"
    category = "meta"
    description = (
        "场景工具选择器 (元工具). 输入用户场景描述 (如 '优化结构'/'调研文献'/'审查论文'), "
        "返回该场景的推荐工具列表、有序调用链、workflow 模板和 rationale. "
        "用于让 LLM 一次性拿到某场景的工具集, 避免逐个挑选工具. "
        "action='predict' 时基于历史预测 top-3 意图, 给 agent 看下一步可能要做什么."
    )
    input_schema = ScenarioToolInput
    # 只做 LLM 匹配 + 查表, 不写文件, 不跑计算
    read_only = True

    async def call(
        self, args: ScenarioToolInput, context: ToolContext
    ) -> ToolResult:
        # predict action: 走 IntentSpeculator, 纯预测不阻塞
        if args.action == "predict":
            return await self._handle_predict(args, context)

        scenario_text = (args.scenario or "").strip()
        if not scenario_text:
            return ToolResult(
                data=None,
                success=False,
                error="scenario 不能为空",
            )

        # 1. 先用 LLM 匹配 scenario_type
        scenario_type: str | None = None
        match_rationale: str = ""
        match_confidence: float = 0.0
        match_source: str = ""

        try:
            model = self._get_model(context)
            llm_result = await self._llm_match(scenario_text, model)
            if llm_result is not None:
                stype, conf, rationale = llm_result
                # 只在置信度够高且类型在预置表里时才采信 LLM 结果
                if stype in SCENARIO_TOOL_BUNDLES and conf >= 0.5:
                    scenario_type = stype
                    match_rationale = rationale
                    match_confidence = conf
                    match_source = "llm"
        except Exception as exc:
            # LLM 不可用就退到关键词兜底, 别让整个工具挂掉
            logger.warning("scenario_tool LLM 匹配失败, 退回关键词兜底: %s", exc)

        # 2. LLM 没给出可用结果 → 关键词兜底
        if scenario_type is None:
            scenario_type = self._keyword_fallback(scenario_text)
            if scenario_type is not None:
                match_source = "keyword"
                match_confidence = 0.6  # 兜底匹配给个中等置信度
                match_rationale = "关键词兜底匹配"
            else:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"无法识别场景类型, scenario='{scenario_text}'. "
                        "可用场景: " + ", ".join(SCENARIO_TOOL_BUNDLES.keys())
                    ),
                )

        # 3. 查表拿工具 bundle
        bundle = SCENARIO_TOOL_BUNDLES[scenario_type]
        data: dict[str, Any] = {
            "scenario_type": scenario_type,
            "recommended_tools": list(bundle["recommended_tools"]),
            "tool_chain": list(bundle["tool_chain"]),
            "workflow_template": bundle["workflow_template"],
            "rationale": bundle["rationale"],
            # 匹配元信息, 方便 debug / 上层决策
            "match_info": {
                "source": match_source,
                "confidence": match_confidence,
                "match_rationale": match_rationale,
            },
            # 列出全部可用场景, 给 LLM 一个备选菜单
            "available_scenarios": list(SCENARIO_TOOL_BUNDLES.keys()),
        }
        return ToolResult(data=data, success=True)

    # ------------------------------------------------------------------ helpers

    async def _handle_predict(
        self, args: ScenarioToolInput, context: ToolContext
    ) -> ToolResult:
        """predict action: 调 IntentSpeculator 返回 top-3 预测.

        不阻塞, 纯预测, 给 agent 看下一步可能要做什么.
        scenario 字段当 query 用, 留空就纯靠历史预测.
        """
        from huginn.agents.speculator import IntentSpeculator

        speculator = IntentSpeculator.shared()
        query = (args.scenario or "").strip() or None
        try:
            preds = speculator.predict(query)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"speculator predict failed: {exc}",
            )

        return ToolResult(
            data={
                "action": "predict",
                "query": query,
                "predictions": [
                    {
                        "scenario_name": p.scenario_name,
                        "score": p.score,
                        "recommended_tools": p.recommended_tools,
                        "confidence": p.confidence,
                    }
                    for p in preds
                ],
                "stats": speculator.stats(),
            },
            success=True,
        )

    def _get_model(self, context: ToolContext) -> Any:
        """拿一个 LangChain chat model, 优先用 context.config."""
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        # temperature 调低一点, 场景分类要确定性
        return get_model(config=config, temperature=0.1, max_tokens=500)

    async def _llm_match(
        self, scenario_text: str, model: Any
    ) -> tuple[str, float, str] | None:
        """调一次 LLM 把场景描述映射到 scenario_type. 失败返回 None."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_SCENARIO_MATCH_SYSTEM_PROMPT),
            HumanMessage(content=f"用户场景描述: {scenario_text}"),
        ]

        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)

        content = response.content if hasattr(response, "content") else str(response)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        return self._parse_match_json(content)

    def _parse_match_json(self, content: str) -> tuple[str, float, str] | None:
        """从 LLM 回复里抠 JSON. 容忍前后多余文字和 ```json 代码块."""
        text = content.strip()
        # 剥 markdown 代码块
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        # 直接解析
        data: dict[str, Any] | None = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 兜底: 抓第一个 { 到最后一个 }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict):
            return None

        stype = str(data.get("scenario_type", "")).strip().lower()
        if not stype:
            return None

        # 置信度解析, 容忍各种写法
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        # 钳到 [0, 1]
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0

        rationale = str(data.get("rationale", "")).strip()
        return stype, conf, rationale

    def _keyword_fallback(self, scenario_text: str) -> str | None:
        """关键词兜底匹配. 命中第一个就返回, 按预置优先级顺序."""
        # 统一转小写做大小写不敏感匹配
        lowered = scenario_text.lower()
        for keywords, scenario_type in _KEYWORD_FALLBACK:
            for kw in keywords:
                # 中文关键词直接 in 匹配; 英文关键词用小写边界匹配避免误伤
                if any(ord(c) > 127 for c in kw):
                    if kw in scenario_text:
                        return scenario_type
                else:
                    if kw in lowered:
                        return scenario_type
        return None

    def estimate_cost(self, args: ScenarioToolInput) -> dict[str, float] | None:
        # 只调一次 LLM 做分类, 成本可忽略
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.001}
