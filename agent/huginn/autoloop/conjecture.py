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


# ── 领域知识表: 模板迁移时把抽象概念落到具体领域 ──────────────────────────
# key 全用小写, _lookup_domain 做大小写不敏感匹配
# 每个 entry: system=体系名称, 然后是 "抽象action/property" → "领域术语"
_DOMAIN_KNOWLEDGE: dict[str, dict[str, str]] = {
    "semiconductors": {
        "system": "半导体晶格",
        "杂质引入": "doping (掺杂)",
        "热处理": "annealing (退火)",
        "晶格畸变": "strain engineering (应变工程)",
        "组分调制": "alloying (合金化)",
        "缺陷工程": "defect passivation (缺陷钝化)",
        "表面态调控": "surface passivation (表面钝化)",
        "电子输运性质": "carrier concentration & conductivity (载流子浓度与导电率)",
        "电子结构": "band gap & DOS (带隙与态密度)",
    },
    "battery cathodes": {
        "system": "层状氧化物正极",
        "杂质引入": "aliovalent substitution (异价取代)",
        "热处理": "calcination (煅烧)",
        "晶格畸变": "chemical pressure (化学压力)",
        "组分调制": "solid-solution tuning (固溶调控)",
        "缺陷工程": "oxygen vacancy engineering (氧空位工程)",
        "表面态调控": "surface coating (表面包覆)",
        "电子输运性质": "electronic conductivity (电子电导率)",
        "离子输运性质": "Li-ion diffusivity (锂离子扩散率)",
        "储能特性": "specific capacity & voltage (比容量与电压)",
    },
    "battery anodes": {
        "system": "负极材料",
        "杂质引入": "heteroatom doping (杂原子掺杂)",
        "热处理": "carbonization (碳化)",
        "晶格畸变": "interlayer spacing tuning (层间距调控)",
        "组分调制": "composite engineering (复合工程)",
        "缺陷工程": "defect generation (缺陷引入)",
        "表面态调控": "SEI modification (SEI修饰)",
        "电子输运性质": "electron transport (电子输运)",
        "离子输运性质": "Li-ion diffusion (锂离子扩散)",
        "储能特性": "specific capacity (比容量)",
    },
    "perovskites": {
        "system": "ABX3 钙钛矿",
        "杂质引入": "A/B-site substitution (A/B位取代)",
        "热处理": "sintering (烧结)",
        "晶格畸变": "tolerance factor tuning (容忍因子调控)",
        "组分调制": "compositional engineering (组分工程)",
        "缺陷工程": "oxygen vacancy (氧空位)",
        "表面态调控": "surface termination (表面端接)",
        "电子输运性质": "carrier mobility (载流子迁移率)",
        "电子结构": "band structure (能带结构)",
        "储能特性": "dielectric response (介电响应)",
    },
    "catalysts": {
        "system": "催化材料",
        "杂质引入": "single-atom modification (单原子修饰)",
        "热处理": "calcination (煅烧)",
        "晶格畸变": "lattice strain (晶格应变)",
        "组分调制": "alloy catalyst (合金催化)",
        "缺陷工程": "active defect sites (活性缺陷位点)",
        "表面态调控": "surface functionalization (表面功能化)",
        "电子输运性质": "electron transfer (电子转移)",
        "表面反应活性": "adsorption energy (吸附能)",
    },
    "thermoelectrics": {
        "system": "热电材料",
        "杂质引入": "carrier optimization (载流子优化)",
        "热处理": "sintering (烧结)",
        "晶格畸变": "phonon scattering engineering (声子散射工程)",
        "组分调制": "band convergence (能带收敛)",
        "缺陷工程": "point defect scattering (点缺陷散射)",
        "表面态调控": "grain boundary engineering (晶界工程)",
        "电子输运性质": "Seebeck coefficient (塞贝克系数)",
        "热输运性质": "lattice thermal conductivity (晶格热导率)",
    },
    "superconductors": {
        "system": "超导材料",
        "杂质引入": "chemical doping (化学掺杂)",
        "热处理": "oxygen annealing (氧退火)",
        "晶格畸变": "chemical pressure (化学压力)",
        "组分调制": "compositional tuning (组分调控)",
        "缺陷工程": "pinning centers (钉扎中心)",
        "表面态调控": "surface doping (表面掺杂)",
        "电子输运性质": "critical current density (临界电流密度)",
        "超导特性": "Tc (临界温度)",
    },
    "magnetic materials": {
        "system": "磁性材料",
        "杂质引入": "magnetic ion doping (磁性离子掺杂)",
        "热处理": "annealing (退火)",
        "晶格畸变": "spin-lattice coupling (自旋-晶格耦合)",
        "组分调制": "magnetic alloying (磁性合金化)",
        "缺陷工程": "domain wall pinning (畴壁钉扎)",
        "表面态调控": "surface anisotropy (表面各向异性)",
        "电子输运性质": "spin transport (自旋输运)",
        "磁学性质": "magnetization & coercivity (磁化与矫顽力)",
    },
    "2d materials": {
        "system": "二维材料",
        "杂质引入": "substitutional doping (取代掺杂)",
        "热处理": "thermal annealing (热退火)",
        "晶格畸变": "strain engineering (应变工程)",
        "组分调制": "alloying (合金化)",
        "缺陷工程": "vacancy engineering (空位工程)",
        "表面态调控": "functionalization (功能化)",
        "电子输运性质": "carrier mobility (载流子迁移率)",
        "电子结构": "band gap (带隙)",
    },
}


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
    """大小写不敏感地查领域知识表. 匹配不到返回空 dict."""
    domain = domain.lower().strip()
    if domain in _DOMAIN_KNOWLEDGE:
        return _DOMAIN_KNOWLEDGE[domain]
    # 模糊: 包含关系, 比如 "oxide cathodes" 能匹中 "battery cathodes"
    for key, info in _DOMAIN_KNOWLEDGE.items():
        if key in domain or domain in key:
            return info
    return {}


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


__all__ = ["ConjectureGenerator", "get_conjecture_generator"]
