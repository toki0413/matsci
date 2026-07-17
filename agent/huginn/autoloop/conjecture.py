"""Conjecture generation — Moonshine 跨域猜想生成范式.

设计参考: Moonshine 数学 agent 的"猜想生成"思路. 不是让模型凭空联想,
而是把跨领域类比推理拆成有约束的三步流水线 —— 从已知问题里提取结构模式,
迁移到新领域, 再生成可检验的猜想. 每一步都有明确的输入输出, 出错了能
降级到模板, 拿不到 LLM 也能跑.

三步:
1. extract_pattern — 从源问题抽取抽象结构模式
   e.g. "doping increases conductivity in semiconductors"
        → "杂质引入 提升 电子输运性质"
2. transfer_domain — 把模式迁移到目标领域
   e.g. 迁到 "battery cathodes"
        → "异价取代 调控 锂离子扩散率"
3. generate_conjecture — 生成带 statement / prediction / rationale 的猜想

每步都把结果写到 research_log, 串成演化树:
- extract  → OPEN_QUESTION (从已知里提炼出的待探索结构)
- transfer → BRIDGE        (连接源领域与目标领域的桥梁)
- generate → CONJECTURE    (最终可检验猜想)

两种模式:
- template (model=None): 关键词匹配 + 领域知识表做泛化, 确定性, 测试用
- LLM-enhanced (model 传入): 调 LLM 做更丰富的提取/迁移/生成, 失败降级到模板

典型用法:

    gen = get_conjecture_generator()
    result = gen.run(
        source_problem="doping increases conductivity in semiconductors",
        source_domain="semiconductors",
        target_domain="battery cathodes",
    )
    print(result["conjecture"]["statement"])
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 关键词表: 模板模式靠这些做模式提取 ────────────────────────────────────
# 不全, 覆盖大部分常见场景够用, 后面遇到新的再补
# R15 评估: spec 建议改走 SignalHub 信号驱动, 但 SignalHub 是系统信号路由
# (events/tool_errors/context_overflow), 不做 NLP 关键词匹配 — 替代不适用.
# 真正升级路径是 LLM extract_pattern + RAG recall, 当前 LLM 路径已不依赖这些表.
# 保留模板表供无 LLM 场景 (测试/离线) 使用. ponytail: 删了反而破坏 test_conjecture_engine.


# 因果触发词 → 方向分类. 遍历时先匹配到的优先, 所以常见词放前面
_CAUSAL_TRIGGERS: dict[str, str] = {
    # increases
    "增加": "increases", "提升": "increases", "提高": "increases",
    "增强": "increases", "促进": "increases", "增大": "increases",
    "increases": "increases", "enhances": "increases",
    "improves": "increases", "promotes": "increases", "boosts": "increases",
    # decreases
    "降低": "decreases", "减小": "decreases", "减弱": "decreases",
    "抑制": "decreases", "削减": "decreases",
    "decreases": "decreases", "reduces": "decreases",
    "suppresses": "decreases", "diminishes": "decreases",
    # modifies (中性, 方向不明确)
    "改变": "modifies", "调控": "modifies", "影响": "modifies",
    "调节": "modifies", "调制": "modifies",
    "modifies": "modifies", "affects": "modifies",
    "alters": "modifies", "controls": "modifies",
}

# 具体动作 → 抽象动作. 提取时把领域术语泛化成跨域通用的概念
_ACTION_ABSTRACTIONS: dict[str, str] = {
    "掺杂": "杂质引入",
    "doping": "杂质引入",
    "dopant": "杂质引入",
    "退火": "热处理",
    "annealing": "热处理",
    "anneal": "热处理",
    "应变": "晶格畸变",
    "strain": "晶格畸变",
    "合金化": "组分调制",
    "alloying": "组分调制",
    "alloy": "组分调制",
    "缺陷工程": "缺陷工程",
    "defect engineering": "缺陷工程",
    "缺陷": "缺陷工程",
    "表面修饰": "表面态调控",
    "surface modification": "表面态调控",
    "passivation": "表面态调控",
    "插层": "层间插体",
    "intercalation": "层间插体",
    "氢化": "化学键合",
    "hydrogenation": "化学键合",
    "氧化": "氧化态调控",
    "oxidation": "氧化态调控",
    "还原": "氧化态调控",
    "reduction": "氧化态调控",
}

# 具体属性 → 抽象属性类. 跟动作一样, 泛化到跨域可比的层次
_PROPERTY_ABSTRACTIONS: dict[str, str] = {
    "导电率": "电子输运性质",
    "电导率": "电子输运性质",
    "conductivity": "电子输运性质",
    "载流子浓度": "电子输运性质",
    "carrier concentration": "电子输运性质",
    "迁移率": "电子输运性质",
    "mobility": "电子输运性质",
    "带隙": "电子结构",
    "band gap": "电子结构",
    "bandgap": "电子结构",
    "态密度": "电子结构",
    "dos": "电子结构",
    "离子电导率": "离子输运性质",
    "ionic conductivity": "离子输运性质",
    "扩散系数": "离子输运性质",
    "diffusion coefficient": "离子输运性质",
    "容量": "储能特性",
    "capacity": "储能特性",
    "比容量": "储能特性",
    "电压": "储能特性",
    "voltage": "储能特性",
    "催化活性": "表面反应活性",
    "catalytic activity": "表面反应活性",
    "吸附能": "表面反应活性",
    "adsorption energy": "表面反应活性",
    "硬度": "力学性能",
    "hardness": "力学性能",
    "弹性模量": "力学性能",
    "elastic modulus": "力学性能",
    "热导率": "热输运性质",
    "thermal conductivity": "热输运性质",
    "塞贝克": "热输运性质",
    "seebeck": "热输运性质",
    "磁化": "磁学性质",
    "magnetic": "磁学性质",
    "磁化率": "磁学性质",
    "居里温度": "磁学性质",
    "curie": "磁学性质",
    "超导": "超导特性",
    "superconduct": "超导特性",
}

# 方向 → 中文动词, 拼模式文本时用
_DIR_ZH: dict[str, str] = {
    "increases": "提升",
    "decreases": "降低",
    "modifies": "调控",
}


# ── 领域知识查表 (R14 已砍硬编码表, 改走 RAG recall) ───────────────────
# 历史: 8 领域表 (~107 行硬编码) 是经验主义, 与 RAG recall 重复.
# 现 _lookup_domain 走 recall_context(category="knowledge_seed", query=domain).
# RAG 没数据时返回空 dict, 模板路径降级到抽象概念本身 (结构仍正确).


class ConjectureGenerator:
    """跨域猜想生成器 — Moonshine 三步流水线.

    extract_pattern → transfer_domain → generate_conjecture

    model=None 时走模板, model 传入且非 mock 时优先调 LLM, 失败降级到模板.
    每步结果都写 research_log, 靠 parent_id 串成演化树:
    OPEN_QUESTION → BRIDGE → CONJECTURE.
    """

    def __init__(self) -> None:
        # 缓存上一步的结果, 方便链式调用和调试回溯
        self._last_pattern: dict[str, Any] | None = None
        self._last_transfer: dict[str, Any] | None = None
        self._last_conjecture: dict[str, Any] | None = None

    # ── pipeline steps ──────────────────────────────────────────────

    def extract_pattern(
        self,
        source_problem: str,
        source_domain: str,
        model: Any = None,
        domain_context: str | None = None,
    ) -> dict[str, Any]:
        """从已知问题抽取抽象结构模式.

        e.g. "doping increases conductivity in semiconductors"
             → pattern: "杂质引入 提升 电子输运性质"

        model 传入且非 mock 时调 LLM 做更丰富的提取, 否则走关键词模板.
        domain_context 是从 KG 查到的源领域上下文, 传给 LLM 做提示增强;
        走模板时忽略它 (模板靠关键词, 不需要额外上下文).
        结果写 research_log (OPEN_QUESTION).
        """
        if model is not None and self._is_real_model(model):
            try:
                result = self._llm_extract(
                    source_problem, source_domain, model, domain_context
                )
            except Exception:
                logger.debug("LLM extract failed, fallback to template", exc_info=True)
                result = self._template_extract(source_problem, source_domain)
        else:
            result = self._template_extract(source_problem, source_domain)

        # 提取出的模式本身就是一个待探索的开放问题
        log_id = self._log_research(
            "open_question",
            f"模式提取: {result.get('abstract_pattern', '')[:60]}",
            (
                f"源问题: {source_problem}\n"
                f"源领域: {source_domain}\n"
                f"抽象模式: {result.get('abstract_pattern', '')}\n"
                f"动作: {result.get('action', '')}\n"
                f"属性: {result.get('property', '')}\n"
                f"方向: {result.get('direction', '')}\n"
                f"机制: {result.get('mechanism', '')}\n"
                f"方法: {result.get('method', '')}"
            ),
            tags=["autoloop", "conjecture", "pattern_extraction", source_domain],
            metadata={
                "source_problem": source_problem,
                "source_domain": source_domain,
                "method": result.get("method"),
                "has_kg_context": bool(domain_context),
            },
        )
        result["log_id"] = log_id
        self._last_pattern = result
        return result

    def transfer_domain(
        self,
        pattern: dict[str, Any],
        target_domain: str,
        model: Any = None,
    ) -> dict[str, Any]:
        """把抽象模式迁移到目标领域.

        e.g. 模式 "杂质引入 调控 电子输运性质" 迁到 "battery cathodes"
             → "异价取代 调控 锂离子扩散率"

        结果写 research_log (BRIDGE), parent 指向 extract_pattern 的记录.
        """
        if model is not None and self._is_real_model(model):
            try:
                result = self._llm_transfer(pattern, target_domain, model)
            except Exception:
                logger.debug("LLM transfer failed, fallback to template", exc_info=True)
                result = self._template_transfer(pattern, target_domain)
        else:
            result = self._template_transfer(pattern, target_domain)

        # 迁移本身就是一座桥, 连接源领域和目标领域
        parent_id = pattern.get("log_id")
        log_id = self._log_research(
            "bridge",
            f"跨域迁移: {pattern.get('source_domain', '?')} → {target_domain}",
            (
                f"抽象模式: {pattern.get('abstract_pattern', '')}\n"
                f"源领域: {pattern.get('source_domain', '')}\n"
                f"目标领域: {target_domain}\n"
                f"迁移模式: {result.get('transferred_pattern', '')}\n"
                f"概念映射: {json.dumps(result.get('domain_mapping', {}), ensure_ascii=False)}\n"
                f"类比说明: {result.get('analogy_notes', '')}\n"
                f"方法: {result.get('method', '')}"
            ),
            parent_id=parent_id,
            tags=["autoloop", "conjecture", "domain_transfer", target_domain],
            metadata={
                "source_domain": pattern.get("source_domain"),
                "target_domain": target_domain,
                "method": result.get("method"),
            },
        )
        result["log_id"] = log_id
        result["parent_log_id"] = parent_id
        self._last_transfer = result
        return result

    def generate_conjecture(
        self,
        transfer_result: dict[str, Any],
        model: Any = None,
        prompt_level: int = 1,
        known_solutions: list[str] | None = None,
    ) -> dict[str, Any]:
        """从迁移结果生成可检验猜想.

        返回 statement (猜想陈述), prediction (可证伪预测),
        rationale (类比依据), confidence (置信度).

        prompt_level 控制提示策略:
        - 0: 纯自由生成, 不加任何额外提示
        - 1: 领域知识表提示 (默认, 性价比最高)
        - 2: 分步引导 + 遗忘已知解法 (需要传 known_solutions)

        结果写 research_log (CONJECTURE), parent 指向 transfer 的 BRIDGE 记录.
        """
        if model is not None and self._is_real_model(model):
            try:
                result = self._llm_generate(
                    transfer_result, model,
                    prompt_level=prompt_level,
                    known_solutions=known_solutions,
                )
            except Exception:
                logger.debug("LLM generate failed, fallback to template", exc_info=True)
                result = self._template_generate(transfer_result)
        else:
            result = self._template_generate(transfer_result)

        parent_id = transfer_result.get("log_id")
        log_id = self._log_research(
            "conjecture",
            result.get("statement", "")[:80],
            (
                f"猜想陈述: {result.get('statement', '')}\n\n"
                f"可证伪预测: {result.get('prediction', '')}\n\n"
                f"类比依据: {result.get('rationale', '')}\n\n"
                f"置信度: {result.get('confidence', 'medium')}\n"
                f"方法: {result.get('method', '')}"
            ),
            parent_id=parent_id,
            status="proposed",
            tags=["autoloop", "conjecture", transfer_result.get("target_domain", "")],
            metadata={
                "domain": transfer_result.get("target_domain"),
                "confidence": result.get("confidence", "medium"),
                "method": result.get("method"),
            },
        )
        result["log_id"] = log_id
        result["parent_log_id"] = parent_id
        self._last_conjecture = result
        # 写回 KG: 猜想作为 FACT 节点, 连上源/目标领域. 失败不影响主流程
        result["kg_node_id"] = self._write_conjecture_to_kg(result, transfer_result)
        return result

    def run(
        self,
        source_problem: str,
        source_domain: str,
        target_domain: str,
        model: Any = None,
        prompt_level: int = 1,
        known_solutions: list[str] | None = None,
    ) -> dict[str, Any]:
        """完整流水线: extract → transfer → generate.

        三步串行执行, 每步结果传给下一步. 研究日志里的记录靠 parent_id
        串成一棵树: OPEN_QUESTION → BRIDGE → CONJECTURE.

        prompt_level 透传给 generate_conjecture (0/1/2, 默认 1).
        """
        # 先从 KG 捞源领域相关实体, 给模式提取多一份上下文 (KG 没建也无所谓)
        domain_context = self._fetch_domain_context(source_domain)
        pattern = self.extract_pattern(
            source_problem, source_domain, model, domain_context=domain_context
        )
        transfer = self.transfer_domain(pattern, target_domain, model)
        conjecture = self.generate_conjecture(
            transfer, model,
            prompt_level=prompt_level,
            known_solutions=known_solutions,
        )

        return {
            "source_problem": source_problem,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "pattern": pattern,
            "transfer": transfer,
            "conjecture": conjecture,
            "log_chain": {
                "pattern_id": pattern.get("log_id"),
                "transfer_id": transfer.get("log_id"),
                "conjecture_id": conjecture.get("log_id"),
            },
            "method": pattern.get("method", "template"),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def forget_then_generate(
        self,
        source_problem: str,
        source_domain: str,
        target_domain: str,
        known_solutions: list[str],
        model: Any = None,
    ) -> dict[str, Any]:
        """遗忘-重生模式: 先遗忘已知解法, 再从第一性原理重新推理.

        等价于 run(prompt_level=2, known_solutions=known_solutions).
        单独暴露出来是因为语义上跟普通 run 有区别 — 强制跳出已有思路.
        """
        return self.run(
            source_problem=source_problem,
            source_domain=source_domain,
            target_domain=target_domain,
            model=model,
            prompt_level=2,
            known_solutions=known_solutions,
        )

    # ── template fallbacks ─────────────────────────────────────────

    def _template_extract(
        self, source_problem: str, source_domain: str
    ) -> dict[str, Any]:
        """关键词匹配做模式提取. 匹配不到的用默认泛化兜底."""
        text = source_problem.lower()

        # 找动作: 遍历关键词表, 第一个命中的就用
        action = "结构调制"
        action_original = "(未识别)"
        for key, abstract in _ACTION_ABSTRACTIONS.items():
            if key in text:
                action = abstract
                action_original = key
                break

        # 找属性, 同理
        prop = "材料性质"
        prop_original = "(未识别)"
        for key, abstract in _PROPERTY_ABSTRACTIONS.items():
            if key in text:
                prop = abstract
                prop_original = key
                break

        # 找因果方向
        direction = "modifies"
        for trig, dir_val in _CAUSAL_TRIGGERS.items():
            if trig in text:
                direction = dir_val
                break

        abstract_pattern = f"{action} {_DIR_ZH.get(direction, '调控')} {prop}"

        return {
            "abstract_pattern": abstract_pattern,
            "source_problem": source_problem,
            "source_domain": source_domain,
            "action": action,
            "action_original": action_original,
            "property": prop,
            "property_original": prop_original,
            "direction": direction,
            "mechanism": (
                f"通过 {action} 引入结构或化学变化, "
                f"进而{_DIR_ZH.get(direction, '调控')} {prop}"
            ),
            "method": "template",
        }

    def _template_transfer(
        self, pattern: dict[str, Any], target_domain: str
    ) -> dict[str, Any]:
        """用领域知识表把抽象模式落到目标领域."""
        action = pattern.get("action", "结构调制")
        prop = pattern.get("property", "材料性质")
        direction = pattern.get("direction", "modifies")
        abstract_pattern = pattern.get("abstract_pattern", "")
        source_domain = pattern.get("source_domain", "")

        domain_info = _lookup_domain(target_domain)
        # 未知领域就用抽象概念本身, 起码结构是对的
        target_action = domain_info.get(action, action)
        target_property = domain_info.get(prop, prop)

        transferred_pattern = (
            f"{target_action} {_DIR_ZH.get(direction, '调控')} {target_property}"
        )

        return {
            "transferred_pattern": transferred_pattern,
            "abstract_pattern": abstract_pattern,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "domain_mapping": {
                "source_action": action,
                "target_action": target_action,
                "source_property": prop,
                "target_property": target_property,
                "direction": direction,
            },
            "analogy_notes": (
                f"将 {source_domain} 中的 '{action}' 类比为 "
                f"{target_domain} 中的 '{target_action}', "
                f"属性 '{prop}' 对应 '{target_property}'"
            ),
            "method": "template",
        }

    def _template_generate(
        self, transfer_result: dict[str, Any]
    ) -> dict[str, Any]:
        """模板拼出猜想陈述 / 预测 / 依据."""
        transferred = transfer_result.get("transferred_pattern", "")
        target_domain = transfer_result.get("target_domain", "")
        source_domain = transfer_result.get("source_domain", "")
        mapping = transfer_result.get("domain_mapping", {})
        analogy = transfer_result.get("analogy_notes", "")
        abstract = transfer_result.get("abstract_pattern", "")

        statement = (
            f"在{target_domain}中, {transferred}, "
            f"其内在机制可类比{source_domain}中的对应现象"
        )

        prediction = (
            f"若在{target_domain}体系中实施 {mapping.get('target_action', '相应调控')}, "
            f"应可观测到 {mapping.get('target_property', '目标性质')} 的系统性变化, "
            f"且变化趋势与{source_domain}中的已知规律一致"
        )

        rationale = (
            f"该猜想基于跨域类比推理: 从{source_domain}的已知规律中提取结构模式 "
            f"'{abstract}', 迁移到{target_domain}后得到 '{transferred}'. "
            f"类比依据: {analogy}. "
            f"注意: 跨域迁移可能存在失效边界, 建议先做小范围计算验证."
        )

        return {
            "statement": statement,
            "prediction": prediction,
            "rationale": rationale,
            "domain": target_domain,
            "confidence": "medium",
            "method": "template",
        }

    # ── LLM enhanced ────────────────────────────────────────────────

    def _llm_extract(
        self, source_problem: str, source_domain: str, model: Any,
        domain_context: str | None = None,
    ) -> dict[str, Any]:
        """调 LLM 做模式提取, 返回结构化结果."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # KG 查到的领域上下文拼进 prompt, 帮 LLM 落到已知实体上
        context_block = ""
        if domain_context:
            context_block = f"\nKnown domain context from KG:\n{domain_context}\n"
        messages = [
            SystemMessage(content=(
                "You are a materials science pattern extractor. "
                "Given a known problem or result, extract the abstract structural "
                "pattern by generalizing the specific action and property. "
                "Output ONLY a JSON object with keys: "
                "abstract_pattern, action, property, "
                "direction (increases|decreases|modifies), mechanism. "
                "No markdown, no explanation."
            )),
            HumanMessage(content=(
                f"Source problem: {source_problem}\n"
                f"Source domain: {source_domain}\n"
                f"{context_block}"
                f"Extract the abstract pattern."
            )),
        ]
        text = self._invoke_model(model, messages)
        parsed = self._parse_json(text)

        # LLM 没返回的字段用模板补
        fallback = self._template_extract(source_problem, source_domain)
        return {
            "abstract_pattern": parsed.get("abstract_pattern", fallback["abstract_pattern"]),
            "source_problem": source_problem,
            "source_domain": source_domain,
            "action": parsed.get("action", fallback["action"]),
            "action_original": fallback["action_original"],
            "property": parsed.get("property", fallback["property"]),
            "property_original": fallback["property_original"],
            "direction": parsed.get("direction", fallback["direction"]),
            "mechanism": parsed.get("mechanism", fallback["mechanism"]),
            "method": "llm",
        }

    def _llm_transfer(
        self, pattern: dict[str, Any], target_domain: str, model: Any
    ) -> dict[str, Any]:
        """调 LLM 做跨域迁移."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # 查源-目标域共享的数学结构, 作为提示增强.
        # 把 LLM 的隐式类比 (依赖训练时见过的同构对) 变成显式提示.
        shared = self._lookup_shared_structure(
            pattern.get("source_domain", ""), target_domain
        )

        messages = [
            HumanMessage(content=self._render_transfer_prompt(
                pattern, target_domain, shared
            )),
        ]
        text = self._invoke_model(model, messages)
        parsed = self._parse_json(text)

        fallback = self._template_transfer(pattern, target_domain)
        return {
            "transferred_pattern": parsed.get(
                "transferred_pattern", fallback["transferred_pattern"]
            ),
            "abstract_pattern": pattern.get("abstract_pattern", ""),
            "source_domain": pattern.get("source_domain", ""),
            "target_domain": target_domain,
            "domain_mapping": parsed.get("domain_mapping", fallback["domain_mapping"]),
            "analogy_notes": parsed.get("analogy_notes", fallback["analogy_notes"]),
            "method": "llm",
        }

    @staticmethod
    def _lookup_shared_structure(src_name: str, tgt_name: str) -> list[str]:
        """查源-目标域共享的数学结构标签.
        ponytail: 只在 _REGISTRY 命中时返回非空, 避免硬塞虚假结构."""
        try:
            from huginn.ml.transfer_registry import _REGISTRY, shared_structure
            src = next((d for d in _REGISTRY if d.name == src_name), None)
            tgt = next((d for d in _REGISTRY if d.name == tgt_name), None)
            if src and tgt:
                return shared_structure(src, tgt)
        except Exception:
            logger.debug("shared_structure lookup failed", exc_info=True)
        return []

    @staticmethod
    def _render_transfer_prompt(
        pattern: dict[str, Any], target_domain: str, shared: list[str]
    ) -> str:
        """结构化提示模板 — 各字段独立填充, 避免 prompt injection.

        ponytail: 模板用三引号 + 占位符, 不用 f-string 拼接.
        升级: 当 shared 包含 Lean 表达式时, 改为 JSON schema."""
        shared_block = (
            f"SHARED STRUCTURE: {', '.join(shared)}\n"
            f"NOTE: Source and target are structurally isomorphic. "
            f"Anchor the transfer on the shared structure, not on "
            f"composition or material type."
            if shared else ""
        )
        return f"""You are a materials science cross-domain transfer specialist.

Given an abstract pattern and a target domain, instantiate the pattern in the target domain using domain-appropriate terminology.

ABSTRACT PATTERN: {pattern.get('abstract_pattern', '')}
ACTION: {pattern.get('action', '')}
PROPERTY: {pattern.get('property', '')}
DIRECTION: {pattern.get('direction', '')}
SOURCE DOMAIN: {pattern.get('source_domain', '')}
TARGET DOMAIN: {target_domain}
{shared_block}

Output ONLY a JSON object with keys: transferred_pattern, domain_mapping (object with source_action, target_action, source_property, target_property), analogy_notes. No markdown, no explanation."""

    def _llm_generate(
        self,
        transfer_result: dict[str, Any],
        model: Any,
        prompt_level: int = 1,
        known_solutions: list[str] | None = None,
    ) -> dict[str, Any]:
        """调 LLM 生成可检验猜想. prompt_level 决定提示策略."""
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt, user_prompt = self._build_generation_prompt(
            transfer_result, prompt_level, known_solutions
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        text = self._invoke_model(model, messages)
        parsed = self._parse_json(text)

        fallback = self._template_generate(transfer_result)
        return {
            "statement": parsed.get("statement", fallback["statement"]),
            "prediction": parsed.get("prediction", fallback["prediction"]),
            "rationale": parsed.get("rationale", fallback["rationale"]),
            "domain": transfer_result.get("target_domain", ""),
            "confidence": parsed.get("confidence", "medium"),
            "method": "llm",
            "prompt_level": prompt_level,
        }

    def _build_generation_prompt(
        self,
        transfer_result: dict[str, Any],
        prompt_level: int,
        known_solutions: list[str] | None,
    ) -> tuple[str, str]:
        """按 prompt_level 构建系统提示 + 用户提示.

        Level 0: 纯自由生成, 不加额外上下文.
        Level 1: 附带领域知识表, 帮 LLM 落到具体术语.
        Level 2: 分步引导推理 + 遗忘已知解法, 强制从第一性原理出发.
        """
        transferred = transfer_result.get("transferred_pattern", "")
        target_domain = transfer_result.get("target_domain", "")
        source_domain = transfer_result.get("source_domain", "")
        domain_mapping = json.dumps(
            transfer_result.get("domain_mapping", {}), ensure_ascii=False
        )
        analogy_notes = transfer_result.get("analogy_notes", "")

        # 所有级别共用的输出格式要求
        output_spec = (
            "Output ONLY a JSON object with keys: "
            "statement (falsifiable conjecture), "
            "prediction (specific observable result), "
            "rationale (grounded reasoning), "
            "confidence (low|medium|high). No markdown."
        )

        base_user = (
            f"Transferred pattern: {transferred}\n"
            f"Target domain: {target_domain}\n"
            f"Source domain: {source_domain}\n"
            f"Domain mapping: {domain_mapping}\n"
            f"Analogy notes: {analogy_notes}\n"
        )

        if prompt_level == 0:
            # 纯自由生成
            system = (
                "You are a materials science conjecture generator. "
                "Given a transferred pattern, formulate a testable conjecture. "
                + output_spec
            )
            return system, base_user + "Generate a testable conjecture."

        if prompt_level == 1:
            # 领域知识表提示
            domain_info = _lookup_domain(target_domain)
            system = (
                "You are a materials science conjecture generator. "
                "Use the provided domain knowledge to ground your conjecture "
                "in domain-appropriate terminology and mechanisms. "
                + output_spec
            )
            user = (
                base_user
                + f"Domain knowledge: {json.dumps(domain_info, ensure_ascii=False)}\n"
                + "Generate a testable conjecture grounded in this domain knowledge."
            )
            return system, user

        # Level 2: 分步引导 + 遗忘已知解法
        domain_info = _lookup_domain(target_domain)
        forget_section = ""
        if known_solutions:
            forget_section = (
                "\n\nIMPORTANT: Ignore the following known solutions. "
                "Do NOT reproduce or build upon them. "
                "Reason from first principles to find a NOVEL approach:\n"
                + "\n".join(f"- {s}" for s in known_solutions)
            )

        system = (
            "You are a materials science conjecture generator using "
            "first-principles reasoning. Follow the step-by-step instructions. "
            "Ignore any known solutions provided and derive a genuinely novel conjecture. "
            + output_spec
        )
        user = (
            base_user
            + f"Domain knowledge: {json.dumps(domain_info, ensure_ascii=False)}\n\n"
            "Step 1: Identify the core physical mechanism that connects "
            f"the transferred pattern '{transferred}' to {target_domain}.\n"
            "Step 2: Consider what observable quantity would change and in which direction.\n"
            "Step 3: State the underlying assumption and its boundary conditions.\n"
            "Step 4: Formulate a falsifiable conjecture and a specific prediction.\n"
            f"Step 5: Rate your confidence (low|medium|high)."
            f"{forget_section}\n\n"
            "Output the JSON object with your final conjecture."
        )
        return system, user

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _is_real_model(model: Any) -> bool:
        """检测是不是 MagicMock (测试注入的)."""
        return not hasattr(model, "_mock_name")

    @staticmethod
    def _invoke_model(model: Any, messages: list) -> str:
        """同步调 LLM, 处理 async 上下文. 失败抛异常给调用方 catch."""
        import asyncio

        try:
            asyncio.get_running_loop()
            # 已经在 event loop 里, 不能 asyncio.run, 用同步 invoke
            resp = model.invoke(messages)
        except RuntimeError:
            resp = asyncio.run(model.ainvoke(messages))
        return str(resp.content).strip()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """从 LLM 输出里抠 JSON. 处理 markdown 代码块包裹的情况."""
        if not text:
            return {}
        # 去掉 ```json ... ``` 包裹
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    def _log_research(
        self,
        record_type: Any,
        title: str,
        content: str,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "proposed",
    ) -> str | None:
        """写研究日志, 返回 record id. 出错返回 None, 不影响主流程.

        record_type 接受 str 或 RecordType, 内部统一转成 RecordType.
        跟 hypothesis_loop 里 _log_research 的思路一样, 但这里把 str
        显式转成 RecordType 再传给 add(), 避免下游 .value 调用炸掉.
        """
        try:
            from huginn.research_log import RecordType, get_research_log

            # str → RecordType, 已经是 RecordType 就跳过
            if isinstance(record_type, str) and not isinstance(record_type, RecordType):
                record_type = RecordType(record_type)

            record = get_research_log().add(
                record_type=record_type,
                title=title,
                content=content,
                parent_id=parent_id,
                status=status,
                tags=tags or [],
                metadata=metadata or {},
            )
            return record.id
        except Exception:
            logger.debug("research log write failed", exc_info=True)
            return None

    def _write_conjecture_to_kg(
        self, result: dict[str, Any], transfer_result: dict[str, Any]
    ) -> str | None:
        """把猜想写进知识图谱, 返回节点 id.

        猜想作为 FACT 节点, 两条边串起跨域关系:
          source MATERIAL --DERIVED_FROM--> conjecture --APPLIES--> target MATERIAL
        任何一步炸了都返回 None, 不影响猜想生成主流程.
        """
        try:
            from huginn.kg.entities import EntityType, Relation

            kg = get_kg()
            if kg is None:
                return None

            statement = result.get("statement", "")
            source_domain = transfer_result.get("source_domain", "")
            target_domain = transfer_result.get("target_domain", "")

            conjecture_id = kg.add_entity(
                label=statement[:80] or "cross-domain conjecture",
                entity_type=EntityType.FACT,
                source="conjecture",
                confidence=0.6,
                prediction=result.get("prediction", ""),
                rationale=result.get("rationale", ""),
                confidence_level=result.get("confidence", "medium"),
            )

            # 源领域 MATERIAL --DERIVED_FROM--> 猜想
            if source_domain:
                src_id = kg.add_entity(
                    source_domain, EntityType.MATERIAL, source="conjecture"
                )
                kg.add_relation(
                    src_id, Relation.DERIVED_FROM, conjecture_id, source="conjecture"
                )

            # 猜想 --APPLIES--> 目标领域 MATERIAL
            if target_domain:
                dst_id = kg.add_entity(
                    target_domain, EntityType.MATERIAL, source="conjecture"
                )
                kg.add_relation(
                    conjecture_id, Relation.APPLIES, dst_id, source="conjecture"
                )

            kg.save()
            return conjecture_id
        except Exception:
            logger.debug("conjecture KG write-back failed", exc_info=True)
            return None

    def _fetch_domain_context(self, domain: str) -> str | None:
        """从 KG 查源领域相关实体, 拼成文本给模式提取用.

        查不到或 KG 不可用就返回 None, 让 extract_pattern 走默认流程.
        """
        try:
            kg = get_kg()
            if kg is None:
                return None
            result = kg.query(domain, depth=1, top_k=8)
            nodes = result.get("nodes") or []
            if not nodes:
                return None
            return kg.to_text({n["id"] for n in nodes})
        except Exception:
            logger.debug("KG domain context fetch failed", exc_info=True)
            return None


# ── 领域知识查表 ──────────────────────────────────────────────────────


def _lookup_domain(domain: str) -> dict[str, str]:
    """R14: 改走 RAG recall (knowledge_seed) 查领域知识.
    历史是查硬编码 _DOMAIN_KNOWLEDGE 表 (8 领域), 现表已删, 走 RAG.
    RAG 返回的 content 若是合法 JSON dict 则直接用, 否则塞进 {"system": domain, "raw": content}.
    RAG 失败/无数据返回空 dict — 模板路径降级到抽象概念本身.
    ponytail: 升级路径是 RAG 召回时做 embedding 相似度排序, 当前精确 query 匹配.
    """
    try:
        from huginn.metacog import recall_context
        results = recall_context(
            category="knowledge_seed",
            query=domain,
            top_k=3,
        )
    except Exception:
        return {}
    if not results:
        return {}
    # 合并多条 recall 结果. 每条 content 尝试解析成 dict, 合并到一起.
    merged: dict[str, str] = {"system": domain}
    for r in results:
        content = r.get("content", "") if isinstance(r, dict) else ""
        if not content:
            continue
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, str):
                        merged[k] = v
        except (json.JSONDecodeError, ValueError):
            # 非结构化内容, 留作 raw 注释
            merged.setdefault("raw", "")
            if merged["raw"]:
                merged["raw"] += " | "
            merged["raw"] += content[:200]
    return merged


# ── 模块级单例 ────────────────────────────────────────────────────────
# 跟 research_log 的 get_research_log 一个套路: 双检锁懒加载.
# 测试要隔离的话直接 new 一个 ConjectureGenerator() 就行, 别碰这个单例.

_conjecture_generator_singleton: ConjectureGenerator | None = None
_singleton_lock = threading.Lock()


def get_conjecture_generator() -> ConjectureGenerator:
    """拿模块级单例 ConjectureGenerator. 线程安全地懒加载."""
    global _conjecture_generator_singleton
    if _conjecture_generator_singleton is None:
        with _singleton_lock:
            # 双检锁, 避免两个线程同时过了第一道 None 检查
            if _conjecture_generator_singleton is None:
                _conjecture_generator_singleton = ConjectureGenerator()
                logger.info("conjecture generator singleton initialized")
    return _conjecture_generator_singleton


# KG 单例: 跟 get_research_log 一样的套路, 落在 ~/.huginn.
# 不可用 (networkx 缺失、目录没权限) 时返回 None, 调用方自己兜底.
_kg_singleton: Any = None
_kg_singleton_lock = threading.Lock()


def get_kg() -> Any:
    """拿模块级单例 ProjectKnowledgeGraph. 线程安全懒加载, 不可用时返回 None."""
    global _kg_singleton
    if _kg_singleton is None:
        with _kg_singleton_lock:
            if _kg_singleton is None:
                try:
                    from huginn.kg.graph import ProjectKnowledgeGraph

                    _kg_singleton = ProjectKnowledgeGraph(Path.home() / ".huginn")
                except Exception:
                    logger.debug("KG singleton init failed", exc_info=True)
                    return None
    return _kg_singleton


# ── v6 G49: Moonshine 三步结构主义重构 ──────────────────────────────
# 用户思想: 结构先于对象. extract_pattern 抽取的是文字模式, 升级为
# extract_physical_structure 抽取 PhysicalStructure (形式化结构关系).
# transfer_domain 接受 PhysicalStructure, 锁定 relation_type 不变, 允许不同
# 实现者填充槽位. generate_conjecture 断言同构保持 (validate_structure_preservation).

# 旧 6 类文字结构 → 5 类预定义结构的映射
_TEXT_TO_STRUCTURE_TYPE: dict[str, str] = {
    "pde": "band_symmetry",
    "variational": "interface_binding",
    "conservation": "defect_chemistry",
    "geometric": "catalytic_geometry",
    "statistical": "percolation_topology",
    # 默认兜底
    "unknown": "catalytic_geometry",
}


def extract_physical_structure(
    source_problem: str,
    source_domain: str,
    model: Any = None,
    domain_context: str | None = None,
) -> "tuple[Any, dict[str, Any]]":
    """G49: 从已知问题抽取 PhysicalStructure (替代纯文字 extract_pattern).

    返回 (physical_structure, pattern_dict).
    pattern_dict 跟旧 extract_pattern 兼容 (含 abstract_pattern/action/property/
    direction/mechanism/method), physical_structure 是新增的形式化结构.

    ponytail: 文字 pattern 用旧 extract_pattern (已有, 关键词模板+LLM),
    再用关键词映射到预定义 PhysicalStructure. 升级路径是 LLM 直接产
    PhysicalStructure (relation_expr + constraints).
    """
    gen = ConjectureGenerator()
    pattern = gen.extract_pattern(
        source_problem, source_domain, model=model, domain_context=domain_context,
    )

    # 从 pattern 关键词推断 relation_type
    text = (
        pattern.get("abstract_pattern", "") + " " +
        pattern.get("action", "") + " " +
        pattern.get("property", "") + " " +
        source_problem
    ).lower()
    relation_type = _infer_relation_type(text)

    # 用预定义结构作模板, implementor_slots 用 pattern 的 action/property 填
    from huginn.metacog.physical_structure import PREDEFINED_STRUCTURES
    template = PREDEFINED_STRUCTURES.get(relation_type)
    if template is None:
        # 兜底: catalytic_geometry
        template = PREDEFINED_STRUCTURES["catalytic_geometry"]

    # 构造 PhysicalStructure — 复用 template 的 relation_expr/constraints,
    # implementor_slots 填 pattern 的抽象 action/property
    from huginn.metacog.physical_structure import PhysicalStructure
    physical = PhysicalStructure(
        relation_type=template.relation_type,
        relation_expr=template.relation_expr,
        implementor_slots=dict(template.implementor_slots),
        constraints=list(template.constraints),
        relative_anchors=dict(template.relative_anchors),
    )
    # action/property 进 metadata (pattern 里已有, 不重复存)
    pattern["physical_structure_type"] = relation_type
    return physical, pattern


def transfer_with_structure(
    physical_structure: "Any",
    pattern: dict[str, Any],
    target_domain: str,
    model: Any = None,
) -> "tuple[Any, dict[str, Any]]":
    """G49: 把 PhysicalStructure 迁移到目标领域 — 锁定结构关系, 允许不同实现者.

    返回 (target_physical_structure, transfer_result_dict).
    target_physical_structure 跟 source 同 relation_type (结构不变),
    implementor_slots 用目标领域的实现者填充.

    ponytail: 当前直接复用 source 的 implementor_slots (不真替换), 因为
    目标领域实现者识别需要 LLM + 领域知识. 升级路径是 LLM 给目标领域候选
    实现者, 再调 enumerate_implementors 枚举.
    """
    gen = ConjectureGenerator()
    transfer = gen.transfer_domain(pattern, target_domain, model=model)

    # target PhysicalStructure: 同 relation_type, implementor_slots 暂时复用
    # (真实迁移需要 LLM 识别目标领域实现者)
    from huginn.metacog.physical_structure import PhysicalStructure
    target_physical = PhysicalStructure(
        relation_type=physical_structure.relation_type,
        relation_expr=physical_structure.relation_expr,
        implementor_slots=dict(physical_structure.implementor_slots),
        constraints=list(physical_structure.constraints),
        relative_anchors=dict(physical_structure.relative_anchors),
    )
    transfer["target_physical_structure_type"] = target_physical.relation_type
    return target_physical, transfer


def generate_conjecture_with_structure(
    source_physical: "Any",
    target_physical: "Any",
    transfer_result: dict[str, Any],
    model: Any = None,
    prompt_level: int = 1,
    known_solutions: list[str] | None = None,
) -> "tuple[bool, dict[str, Any]]":
    """G49: 生成猜想并断言同构保持 (validate_structure_preservation).

    返回 (is_isomorphic, conjecture_result_dict).
    is_isomorphic = True 表示 source/target 结构保持, 猜想可作为同构保持的
    跨域迁移; False 表示结构破坏, 猜想需重新生成.

    ponytail: 当前 source/target implementor_slots 相同 (transfer_with_structure
    没真替换), 必然 trivial mapping. 升级路径是 transfer_with_structure 真做
    实现者替换, 这里才能真验证同构.
    """
    from huginn.metacog.physical_structure import StructureMapping, validate_structure_preservation
    gen = ConjectureGenerator()
    conjecture = gen.generate_conjecture(
        transfer_result, model=model,
        prompt_level=prompt_level, known_solutions=known_solutions,
    )

    # 断言同构保持
    mapping = StructureMapping(
        source=source_physical,
        target=target_physical,
        slot_replacements={},
    )
    is_isomorphic = validate_structure_preservation(mapping)
    conjecture["is_structure_preserved"] = is_isomorphic
    conjecture["structure_violations"] = mapping.violation_detail
    return is_isomorphic, conjecture


def _infer_relation_type(text: str) -> str:
    """从 pattern 文字推断 5 类预定义 relation_type."""
    text_lower = text.lower()
    # band_symmetry: 电子结构/能带/带隙
    if any(k in text_lower for k in (
        "band", "能带", "带隙", "symmetry", "对称", "topolog", "电子结构",
    )):
        return "band_symmetry"
    # interface_binding: 界面/结合/异质
    if any(k in text_lower for k in (
        "interface", "界面", "binding", "结合", "hetero", "异质", "adhesion",
    )):
        return "interface_binding"
    # percolation_topology: 逾渗/网络/连通
    if any(k in text_lower for k in (
        "percolat", "逾渗", "network", "网络", "connect", "连通", "transport",
    )):
        return "percolation_topology"
    # defect_chemistry: 缺陷/掺杂/电荷
    if any(k in text_lower for k in (
        "defect", "缺陷", "dop", "掺杂", "charge", "电荷", "vacanc", "空位",
    )):
        return "defect_chemistry"
    # catalytic_geometry: 催化/吸附/活性位
    if any(k in text_lower for k in (
        "catal", "催化", "adsorp", "吸附", "active site", "活性位",
    )):
        return "catalytic_geometry"
    # 默认
    return "catalytic_geometry"


__all__ = [
    "ConjectureGenerator", "get_conjecture_generator",
    "extract_physical_structure", "transfer_with_structure",
    "generate_conjecture_with_structure",
]
