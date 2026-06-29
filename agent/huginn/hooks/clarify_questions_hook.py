"""USER_PROMPT_SUBMIT 钩子: 信息不足时生成结构化追问, 降低无效迭代.

设计思路 (参考 v0/nudge/Questions 机制):
- 输入不足时 AI 先追问对齐意图, 而非猜测执行
- 每个会话最多追问一次, 避免循环
- 纯规则检测 + 模板生成, 不调 LLM, 零成本
- 只在"明显信息不足"时触发, 避免过度打扰

检测维度:
1. 目标模糊: 含"分析/研究/探索"等动词但无具体对象(材料/文件/体系)
2. 参数缺失: 含"计算/求解/拟合"等动词但缺必要参数(数值/范围/方法)
3. 输出未定: 含"生成/做/给"等动词但没说输出格式(图/报告/数据)
4. 过短: prompt <12 字符且无具体名词

追问模板按维度生成, 最多 3 个. 写入 ctx.metadata["clarify_questions"],
agent.chat() 取到后直接返回不进循环, 等用户回答.
"""

from __future__ import annotations

import logging
import re

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 触发追问的模糊动词. 命中这些词但缺少具体对象时, 认为意图模糊.
_VAGUE_VERBS_CN = (
    "分析一下", "研究一下", "探索一下", "看看", "帮我看看", "帮我分析",
    "帮我研究", "处理一下", "弄一下", "搞一下",
)
_VAGUE_VERBS_EN = (
    r"analyze\s*(?:it|this|that)?\s*$",
    r"look\s+into\s+(?:it|this)",
    r"check\s+(?:it|this|that)\s*$",
)

# 计算类动词, 需要明确参数才不追问
_COMPUTE_VERBS_CN = ("计算", "求解", "拟合", "优化", "积分", "解方程")
_COMPUTE_VERBS_EN = ("calculate", "solve", "fit", "optimize", "integrate")

# 生成类动词, 需要明确输出格式
# 注意: "给" 作为介词太常见 ("importance 给 0.9"), 移除避免误触发
# 注意: "做" 作为泛指动词太常见 ("做任务/做计算/做分析"), 不是真正的生成指令, 移除
_GENERATE_VERBS_CN = ("生成", "画", "写")
_GENERATE_VERBS_EN = ("generate", "draw", "write", "create")

# 工具引用标记: prompt 里出现这些说明用户已明确指定工具, 意图清晰不追问
_TOOL_REFERENCES = (
    "_tool",  # 通配: xxx_tool 都算
    "remember", "recall",  # 记忆工具
    "orchestrate", "design_plan", "nudge", "design_atom", "generative_design",
)

# 具体名词标记: 出现这些词认为有具体对象, 不追问目标
_CONCRETE_NOUNS_CN = (
    "硅", "锗", "碳", "铜", "铁", "钛", "铝", "镍", "钴", "锌",
    "GaAs", "GaN", "ZnO", "TiO2", "SiC", "MoS2", "石墨烯",
    "晶体", "结构", "带隙", "能带", "晶格", "原子", "分子",
    "VASP", "LAMMPS", "Gaussian", "INCAR", "POSCAR",
    "文件", "数据", "矩阵", "方程", "函数",
)
_CONCRETE_NOUNS_EN = (
    "silicon", "germanium", "copper", "iron", "crystal", "lattice",
    "bandgap", "band gap", "structure", "file", "data", "matrix",
    "equation", "function", "VASP", "LAMMPS", "Gaussian",
)

# 输出格式标记: 出现这些词认为输出已定, 不追问格式
_OUTPUT_FORMATS_CN = ("图", "表", "报告", "列表", "JSON", "CSV", "Markdown", "LaTeX")
_OUTPUT_FORMATS_EN = ("plot", "chart", "figure", "table", "report", "json", "csv")

# 追问模板, 按维度生成
_QUESTION_TEMPLATES = {
    "goal": "请明确你要{verb}的具体对象是什么？（材料名/文件路径/体系名称）",
    "params": "请提供关键参数（如数值范围/方法/精度要求）？",
    "output": "请说明期望的输出格式（图表/报告/数据/代码）？",
    "vague": "请更详细地描述你的需求，包括目标、输入和期望输出。",
}

# 最小 prompt 长度, 低于此值且无具体名词时触发追问
_MIN_PROMPT_LEN = 12


class ClarifyQuestionsHook:
    """USER_PROMPT_SUBMIT 钩子: 信息不足时生成结构化追问.

    纯规则检测 + 模板生成, 不调 LLM. 每个会话(thread_id)最多追问一次.
    """

    def __init__(self) -> None:
        # 已追问过的 thread_id 集合, 避免循环追问
        self._asked_threads: set[str] = set()
        # 英文模糊动词预编译
        self._vague_en_pattern = re.compile(
            "|".join(_VAGUE_VERBS_EN), re.IGNORECASE
        )
        self._compute_en_pattern = re.compile(
            r"\b(" + "|".join(_COMPUTE_VERBS_EN) + r")\b", re.IGNORECASE
        )
        self._generate_en_pattern = re.compile(
            r"\b(" + "|".join(_GENERATE_VERBS_EN) + r")\b", re.IGNORECASE
        )

    async def __call__(self, ctx: HookContext) -> HookContext | None:
        try:
            message = ctx.metadata.get("user_message")
            if not isinstance(message, str) or not message.strip():
                return None

            # 同一 thread 只追问一次, 避免循环
            thread_id = ""
            try:
                thread_id = str(ctx.metadata.get("thread_id", "") or "")
            except Exception:
                pass
            if thread_id and thread_id in self._asked_threads:
                return None

            questions = self._detect_and_generate(message)
            if not questions:
                return None

            # 标记已追问, 写入 metadata 给 agent.chat() 取
            if thread_id:
                self._asked_threads.add(thread_id)
            ctx.metadata["clarify_questions"] = questions
        except Exception:
            # 规则匹配不该挂, 真挂了也别拖垮 agent
            logger.warning("ClarifyQuestionsHook raised", exc_info=True)
        return None

    def _detect_and_generate(self, text: str) -> list[str]:
        """检测信息完整度, 不足时生成追问. 返回空列表表示不追问."""
        text_stripped = text.strip()
        text_lower = text.lower()
        questions: list[str] = []

        # 用户明确引用了工具名(vasp_tool/remember/...), 意图清晰不追问
        if any(t in text_lower for t in _TOOL_REFERENCES):
            return questions

        # 多步骤任务标记: 用户已列步骤或明确要执行, 不追问
        # 例如 "帮我做以下任务: 1) 算硅带隙 2) 算铜电导率" 不应被拦
        _MULTI_STEP_MARKERS = (
            "1)", "2)", "3)", "首先", "然后", "最后", "任务",
            "step 1", "step 2", "first,", "then,", "finally,",
        )
        # 执行续接标记: 用户在催促继续执行, 不追问
        # 例如 "现在可以执行 SCF 计算了" 不应被拦
        _EXEC_CONT_MARKERS = (
            "现在可以", "继续", "执行", "开始", "已经", "完成",
            "proceed", "continue", "go ahead",
        )
        if any(m in text for m in _MULTI_STEP_MARKERS) or any(
            m in text_lower for m in _EXEC_CONT_MARKERS
        ):
            return questions

        # 1) 检测具体名词: 有具体对象就不追问目标
        has_concrete = any(n in text for n in _CONCRETE_NOUNS_CN) or any(
            n.lower() in text_lower for n in _CONCRETE_NOUNS_EN
        )

        # 2) 检测模糊动词
        has_vague_cn = any(v in text for v in _VAGUE_VERBS_CN)
        has_vague_en = bool(self._vague_en_pattern.search(text))
        has_compute = any(v in text for v in _COMPUTE_VERBS_CN) or bool(
            self._compute_en_pattern.search(text)
        )
        has_generate = any(v in text for v in _GENERATE_VERBS_CN) or bool(
            self._generate_en_pattern.search(text)
        )

        # 3) 检测输出格式
        has_output_format = any(f in text for f in _OUTPUT_FORMATS_CN) or any(
            f.lower() in text_lower for f in _OUTPUT_FORMATS_EN
        )

        # 4) 过短且无具体名词 → 极模糊, 直接追问
        if len(text_stripped) < _MIN_PROMPT_LEN and not has_concrete:
            questions.append(_QUESTION_TEMPLATES["vague"])
            return questions

        # 5) 模糊动词 + 无具体对象 → 追问目标
        if (has_vague_cn or has_vague_en) and not has_concrete:
            questions.append(_QUESTION_TEMPLATES["goal"])

        # 6) 计算类动词 + 无具体名词(参数) → 追问参数
        #    有具体名词说明给了材料/体系, 可能也给了参数, 不追问
        #    长消息(>=30字)通常已含完整上下文, 不追问参数
        if has_compute and not has_concrete and len(text_stripped) < 30:
            if _QUESTION_TEMPLATES["params"] not in questions:
                questions.append(_QUESTION_TEMPLATES["params"])

        # 7) 生成类动词 + 无输出格式 → 追问输出
        #    有具体名词说明用户已明确了对象, 输出格式可以推断, 不追问
        if has_generate and not has_output_format and not has_concrete:
            questions.append(_QUESTION_TEMPLATES["output"])

        # 最多 3 个追问, 避免刷屏
        return questions[:3]
