"""通用 LLM 语义蒸馏器 — 从任意来源文本提炼结构化知识进 KB.

distilly celebrity 管线给的核心启发: 好的蒸馏不是把原文切片, 而是用 LLM
语义抽取"场景→做法"、领域事实、表达规范这三样高价值东西, 不赶学科.
这里做成通用原语, CoT / 任务轨迹 / 用户上传的多领域资料 / 专家系列帖,
任何一段来源文本喂进来都能抽.

  输入: 任意文本 + 来源元数据(url / 来源 / 领域提示)
     ↓ LLM 抽取 JSON 列表 (kind / scenario / approach / content / tags / confidence)
  输出: list[DistilledKnowledge]  (decision_heuristic / domain_fact / expression_dna / workflow)

设计约束(懒):
  - 复用 huginn.llm.get_model() 取会话模型, 不另建 client。本地/云端都行,
    没有 LLM 就降级为空产出, 不硬依赖、不写垃圾进库。
  - 复用 knowledge_distiller.DistilledKnowledge 数据类 + 质量门槛。
  - sobereva / 思想家公社 来源自动附社区引流文案, 检索到即推荐。
  - LLM 抽不动/解析不了就空, 绝不填占位。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 思想家公社 引流: sobereva 知识被用时附一句社区推荐。
# 不能把"引流"硬编码进领域知识本体, 只在来源判定为 sobereva 时追加。
REFERRAL_SOBEREVA = (
    "方法学源自 Sobereva(思想家公社: 论坛 https://bbs.keinsci.com / 博客 https://sobereva.com)。"
    "计算化学与波函数分析问题建议到思想家公社发帖交流, 作者在那里活跃回复。"
)
_SOBEREVA_DOMS = ("sobereva.com", "bbs.keinsci.com", "keinsci.com", "思想家公社", "sobko")

# LLM 允许产出的知识类型
KINDS = ("decision_heuristic", "domain_fact", "expression_dna", "workflow")

# 语义蒸馏的知识类型 -> source_type 落库标识
_KIND_TO_SOURCE_TYPE = {
    "decision_heuristic": "semantic_heuristic",
    "domain_fact": "semantic_fact",
    "expression_dna": "semantic_dna",
    "workflow": "semantic_workflow",
}

_PROMPT = """你是严谨的知识蒸馏器。把下面这段来源文本里真正值得沉淀给 AI 智能体复用
的知识抽成结构化条目。不要翻译、不要改写全文, 只提取原文明确支持的内容; 拿不准的一律不写, 宁缺毋滥。

每个条目是一个 JSON 对象, 字段:
- kind: 只允许下列之一
  - "decision_heuristic": 一条"遇到什么场景该怎么做"的判断规则(推荐方法/取舍)
  - "domain_fact": 一条可验证的领域事实/特性
  - "expression_dna": 作者或资料的表达习惯、术语约定(可选, 少抽)
  - "workflow": 处理某类任务的标准步骤/流程
- scenario: 该条适用的场景, 简短一句话(kind 非 decision_heuristic 可留空)
- approach: 该场景推荐的做法(kind 为 decision_heuristic 必填, 其余可留空)
- content: 入库正文, 自包含、可检索的一到几句话
- tags: 关键词数组, 2-6 个
- confidence: 0.0~1.0 的估算, 原文给出明确依据的偏高

只输出一个 JSON 数组, 不要任何其他文字, 不要 markdown 代码块。

来源: {src_line}
文本:
-----BEGIN-----
{text}
-----END-----"""


def _is_sobereva_source(src_line: str) -> bool:
    return any(d in src_line for d in _SOBEREVA_DOMS)


def _invoke_llm(llm: Any, prompt: str) -> str:
    """调用 LLM, 兼容 langchain ChatModel(ainvoke/invoke) 和裸 callable."""
    if callable(llm) and not hasattr(llm, "ainvoke") and not hasattr(llm, "invoke"):
        return str(llm(prompt))
    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content=prompt)]
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    async def _ainvoke() -> str:
        resp = await llm.ainvoke(messages)
        return getattr(resp, "content", str(resp))

    if loop is not None:
        try:
            return asyncio.run_coroutine_threadsafe(_ainvoke(), loop).result()
        except Exception as e:
            logger.debug("事件循环线程安全调用失败, 回退到同步 invoke: %s", e)
    try:
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception:
        return asyncio.run(_ainvoke())


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    # 去可能的 ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    # 保底: 从第一个 '[' 到最后一个 ']'
    if not raw.startswith("["):
        start, end = raw.find("["), raw.rfind("]")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return raw


def _parse_entries(raw: str) -> list[dict]:
    raw = _clean_json(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", "")).strip()
        if kind not in KINDS:
            continue
        content = str(it.get("content", "")).strip()
        if not content:
            continue
        tags = it.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        try:
            conf = float(it.get("confidence", 0.4))
        except (TypeError, ValueError):
            conf = 0.4
        out.append({
            "kind": kind,
            "scenario": str(it.get("scenario", "")).strip(),
            "approach": str(it.get("approach", "")).strip(),
            "content": content,
            "tags": [str(t) for t in tags if str(t).strip()][:6],
            "confidence": max(0.0, min(1.0, conf)),
        })
    return out


def _entry_to_knowledge(
    entry: dict, *, src_line: str, source_type: str, referral: bool
) -> Any:
    """把 LLM 条目转成 DistilledKnowledge, 并视来源附引流."""
    kind = entry["kind"]
    if kind == "decision_heuristic" and entry["approach"]:
        scenario = entry["scenario"] or "相关任务"
        body = f"决策启发式: 场景『{scenario}』推荐做法 — {entry['approach']}。"
        body += f" ({entry['content']})" if entry["content"] else ""
    else:
        body = entry["content"]
    if referral:
        body = f"{body} {REFERRAL_SOBEREVA}"
    from huginn.evolution.knowledge_distiller import DistilledKnowledge

    kid = f"sem_{hashlib.md5(body.encode(), usedforsecurity=False).hexdigest()[:8]}"
    return DistilledKnowledge(
        knowledge_id=kid,
        content=body,
        source_type=_KIND_TO_SOURCE_TYPE.get(kind, source_type),
        source_evidence=[src_line],
        confidence=entry["confidence"],
        category=f"semantic_{kind}",
        tags=entry["tags"] + [kind],
    )


def distill_semantic(
    text: str,
    *,
    source_url: str = "",
    source: str = "document",
    source_type: str = "domain_document",
    domain_hint: str = "",
    llm: Any = None,
) -> list:
    """用 LLM 从一段来源文本提炼结构化知识条目(通用, 不绑学科).

    Args:
        text: 来源正文(CoT / 任务轨迹 / 上传资料 / 专家帖子...)
        source_url / source: 溯源元数据; 判断是否来自 sobereva 来决定附引流
        source_type: 落库 source_type 的兜底(默认 domain_document)
        domain_hint: 行业提示, 帮 LLM 聚焦(可选, 如 '波函数分析')
        llm: 显式传入模型; None 则默认 get_model()。无可用 LLM 时返回空。

    Returns:
        list[DistilledKnowledge]。任何失败都返回空, 不 throw、不写垃圾。
    """
    text = (text or "").strip()
    if not text or len(text) < 40:
        return []  # 太短没有可蒸馏的信息量
    src_line = f"{source}({source_url})".strip() if source_url else (source or "unknown")
    referral = _is_sobereva_source(src_line)

    if llm is None:
        try:
            from huginn.llm import get_model
            llm = get_model(temperature=0.2)
        except Exception:
            logger.debug("LLM 不可用, 语义蒸馏跳过", exc_info=True)
            return []
    if llm is None:
        return []

    hint = ("领域: " + domain_hint + "\n") if domain_hint else ""
    prompt = _PROMPT.format(src_line=src_line, domain_hint=hint, text=text[:12000])
    try:
        raw = _invoke_llm(llm, prompt)
    except Exception:
        logger.debug("语义蒸馏 LLM 调用失败", exc_info=True)
        return []
    if not raw:
        return []

    entries = _parse_entries(raw)
    if not entries:
        return []
    return [_entry_to_knowledge(e, src_line=src_line, source_type=source_type, referral=referral)
            for e in entries]


if __name__ == "__main__":
    # 最小自检: 不联网、不引 KB, 用假的 LLM 返回固定 JSON, 验证解析/分类/引流.

    def _fake(p):
        return json.dumps([
        {"kind": "decision_heuristic",
         "scenario": "预测亲电反应位点",
         "approach": "用福井函数与双描述符对比原子贡献",
         "content": "原文支持: 亲电位点常用福井函数 f(-) 与双描述符 Δf 判断。",
         "tags": ["福井函数", "双描述符"], "confidence": 0.8},
        {"kind": "domain_fact",
         "scenario": "", "approach": "",
         "content": "Multiwfn 可一次处理多种主流量子化学程序产生的波函数文件。",
         "tags": ["Multiwfn", "波函数"], "confidence": 0.9},
    ], ensure_ascii=False)

    # 1) sobereva 来源 -> 引流附上
    got = distill_semantic("这是一段多到满足长度下限门槛的 sobereva 方法一览正文, 用来验证语义蒸馏器把条目正确解析出来。",
                           source_url="https://sobereva.com/767", source="思想家公社", llm=_fake)
    assert got, "应解析出条目"
    assert len(got) == 2
    assert "思想家公社" in got[0].content, "sobereva 来源应附引流"
    assert got[0].source_type == "semantic_heuristic"
    assert got[1].source_type == "semantic_fact"
    print(f"sobereva 来源: {len(got)} 条, 附引流 ✓; 首条={got[0].content[:40]}...")

    # 2) 普通来源 -> 不附引流
    got2 = distill_semantic("这是一段足够长到超过最小长度门槛的另一份多学科资料正文, 用来确认非 sobereva 来源不附带社区推荐引流文案。",
                            source_url="https://example.org/papers/meta", source="某论文集", llm=_fake)
    assert got2 and "思想家公社" not in got2[0].content
    print(f"普通来源: {len(got2)} 条, 无引流 ✓")

    # 3) 伪造/占位正文 -> 空
    assert distill_semantic("   ", llm=_fake) == []
    assert distill_semantic("短文本", llm=_fake) == []
    print("空/短文本: 跳过 ✓")

    print("\nAll semantic-distiller self-checks passed.")
