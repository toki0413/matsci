"""Agent middleware: dangling-tool-call repair and rate limiting."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from huginn.utils.session_context import get_thread_id

logger = logging.getLogger(__name__)


class FixDanglingToolCallsMiddleware(AgentMiddleware):
    """Patch orphan tool_calls left behind by summarization compaction.

    deepagents' built-in PatchToolCallsMiddleware only runs before_agent,
    but summarization can drop ToolMessages mid-turn, leaving orphan
    AIMessage.tool_calls that make DeepSeek reject with 400.  This
    middleware patches at the wrap_model_call layer instead.
    """

    def _patch_messages(self, messages: list) -> list:
        if not messages:
            return messages
        # 先剥离 file/非 text multimodal block — DeepSeek 等纯文本 model
        # 不支持 type=file, deepagents read_file 读 PDF/binary 会返回
        # {"type": "file", "base64": ...} block, 直接发给 DeepSeek 触发 400.
        # 转成 text 占位让 model 知道这里有个文件但内容不可见.
        patched = list(messages)
        _changed_blocks = False
        for _i, _msg in enumerate(patched):
            _content = getattr(_msg, "content", None)
            if not isinstance(_content, list):
                continue
            _new_blocks = []
            _msg_changed = False
            for _block in _content:
                if isinstance(_block, dict) and _block.get("type") not in ("text", None):
                    _btype = _block.get("type", "unknown")
                    _mime = _block.get("mime_type", "")
                    _b64 = _block.get("base64", "") or ""
                    _new_blocks.append({
                        "type": "text",
                        "text": f"[{_btype} content omitted: mime={_mime}, {len(_b64)} chars base64 — model does not support multimodal]",
                    })
                    _msg_changed = True
                else:
                    _new_blocks.append(_block)
            if _msg_changed:
                try:
                    _msg.content = _new_blocks
                    _changed_blocks = True
                except Exception:
                    pass
        # 再处理 orphan tool_calls
        answered_ids = {
            getattr(msg, "tool_call_id", None)
            for msg in patched
            if hasattr(msg, "type") and msg.type == "tool"
        }
        has_orphan = any(
            tc.get("id") is not None and tc["id"] not in answered_ids
            for msg in patched
            if isinstance(msg, AIMessage)
            for tc in (*msg.tool_calls, *getattr(msg, "invalid_tool_calls", []))
        )
        if not has_orphan and not _changed_blocks:
            return messages
        if not has_orphan:
            return patched
        for msg in patched:
            if not isinstance(msg, AIMessage):
                continue
            for tc in (*msg.tool_calls, *getattr(msg, "invalid_tool_calls", [])):
                tc_id = tc.get("id")
                if tc_id is None or tc_id in answered_ids:
                    continue
                name = tc.get("name") or "unknown"
                content = (
                    f"Tool call {name} (id={tc_id}) was cancelled — "
                    f"summarization compaction removed its result."
                )
                # 插到 AIMessage 后面紧跟的 ToolMessage 队列尾部, 不能 append
                # 到消息列表末尾 — OpenAI/DeepSeek 要求 tool 响应紧跟其调用,
                # 中间隔了别的消息会 400 (Step 3 序列错乱 σ₉).
                insert_at = patched.index(msg) + 1
                while insert_at < len(patched) and getattr(patched[insert_at], "type", None) == "tool":
                    insert_at += 1
                patched.insert(insert_at, ToolMessage(content=content, name=name, tool_call_id=tc_id))
                answered_ids.add(tc_id)
        return patched

    def wrap_model_call(self, request, handler):
        request.messages = self._patch_messages(request.messages)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.messages = self._patch_messages(request.messages)
        return await handler(request)

    # Also patch at before_agent — _get_model_input_state runs before
    # wrap_model_call, so we need to fix orphan tool calls earlier.
    def before_agent(self, request, handler=None):
        if hasattr(request, 'messages') and request.messages:
            request.messages = self._patch_messages(request.messages)
        return handler(request) if handler else None

    async def abefore_agent(self, request, handler=None):
        if hasattr(request, 'messages') and request.messages:
            request.messages = self._patch_messages(request.messages)
        return await handler(request) if handler else None

    # deepagents middleware protocol requires all four methods.
    # Tool-call layer doesn't need orphan patching — passthrough.
    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


class RateLimitMiddleware(AgentMiddleware):
    """Token-rate limiter at the model_call layer.

    Guards against runaway generation by checking before each model
    call and recording usage from the returned AIMessage afterwards.
    """

    def __init__(self) -> None:
        from huginn.security.rate_limiter import get_rate_limiter

        self._limiter = get_rate_limiter()

    def _estimate_tokens(self, messages: list) -> int:
        # ponytail: LangChain content 可能是 str / list[multipart] / None.
        # 之前用 str(content) 兜底, list 时会 str(list) 把整个 repr 算进去,
        # 一条 multimodal message 就能把 estimate 撑到 12M chars 触发误拦.
        # 正确做法: str 直接算字符, list 累加每个 block 的 text, None 用 repr 但限长.
        total = 0
        for msg in messages or []:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(str(block.get("text", block.get("content", ""))))
                    else:
                        total += len(str(block))
            elif content is not None:
                total += len(str(content))
            else:
                total += min(len(str(msg)), 1000)
        return max(total // 4, 1)

    def _extract_usage(self, result: Any) -> tuple[int, int]:
        from huginn.security.rate_limiter import _extract_usage as _extract

        return _extract(result)

    def wrap_model_call(self, request, handler):
        _tid = get_thread_id() or "default"
        ok, reason = self._limiter.check_allowed(
            "agent", self._estimate_tokens(getattr(request, "messages", [])),
            thread_id=_tid,
        )
        if not ok:
            from huginn.security.rate_limiter import RateLimitExceeded

            raise RateLimitExceeded(reason, reason="limit_exceeded")
        result = handler(request)
        in_tok, out_tok = self._extract_usage(result)
        self._limiter.record_usage("agent", in_tok, out_tok, thread_id=_tid)
        return result

    async def awrap_model_call(self, request, handler):
        _tid = get_thread_id() or "default"
        ok, reason = self._limiter.check_allowed(
            "agent", self._estimate_tokens(getattr(request, "messages", [])),
            thread_id=_tid,
        )
        if not ok:
            from huginn.security.rate_limiter import RateLimitExceeded

            raise RateLimitExceeded(reason, reason="limit_exceeded")
        result = await handler(request)
        in_tok, out_tok = self._extract_usage(result)
        self._limiter.record_usage("agent", in_tok, out_tok, thread_id=_tid)
        return result

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


class DeliverableCoverageMiddleware(AgentMiddleware):
    """内容级 deliverable 覆盖检查 — SearchOS SOCM + LA4VLA grounding.

    把 INSTRUCTIONS.md 里 implicit 的 task requirements 变成 explicit
    coverage state. 每轮 model call 前对比 INSTRUCTIONS 和 report.md,
    缺失的物理量作为 frontier task 注入消息列表前面.

    ponytail: 不用 LLM 解析 task description (成本高 + 不确定), 用正则
    提取 "derive/constrain/upper limits on X and Y" 模式的物理量.
    LA4VLA 思想: language→action grounding 在 harness 层强制, 不让 agent
    走 "图生成了就 done" 的 visual shortcut.

    升级路径: 漏报严重时加 synonym dict (mass→μ/mu, coupling→g), 或换
    LLM 解析 task description. 当前接受一定 noise (误报比漏报好).
    """

    # 两条 pattern 抓 "X and Y" 物理量对:
    #   (1) 动词触发: "derive/constrain/estimate/predict/classify X and Y"
    #   (2) 列举触发: "classifications such as X and Y" (Material_000 模式)
    # X/Y 限制最多 5 个词避免匹配整个从句; [\w\-/] 让 "self-interaction"
    # 和 "metal/insulator" 这种带连字符/斜杠的复合词被当成一个词.
    _QUANTITY_PATTERNS = [
        re.compile(
            r'(?:upper limits? on|limits? on|derive|constrain|estimate|'
            r'calculate|determine|measure|compute|obtain|provide|report|'
            r'analyze|investigate|study|explore|examine|fit|infer|extract|'
            r'bound|restrict|classify|categorize|predict|identify|characterize)'
            r'\s+([\w\-/]+(?:\s+[\w\-/]+){0,4}?)\s+and\s+'
            r'([\w\-/]+(?:\s+[\w\-/]+){0,4}?)'
            r'(?=[.,;)]|\s+(?:to|in|for|by|using|via|with|from|thereby|thus|hence)\s|$)',
            re.IGNORECASE,
        ),
        re.compile(
            r'(?:classifications?|categories?|properties|predictions?|types?|labels?)'
            r'\s+such\s+as\s+([\w\-/]+(?:\s+[\w\-/]+){0,4}?)\s+and\s+'
            r'([\w\-/]+(?:\s+[\w\-/]+){0,4}?)'
            r'(?=[.,;)]|\s+(?:to|in|for|by|using|via|with|from|thereby|thus|hence)\s|$)',
            re.IGNORECASE,
        ),
    ]

    _STOPWORDS = frozenset({
        "the", "a", "an", "of", "and", "or", "to", "in", "for", "by",
        "with", "from", "on", "at", "is", "are", "was", "were", "be",
        "been", "being", "this", "that", "these", "those", "as", "it",
        # 代词/限定词 — 会让 keyword 误匹配 report 里无关行
        "their", "its", "our", "your", "his", "her", "such", "other",
        "some", "many", "most", "all", "both", "each", "every", "any",
        "specific", "general", "overall", "main", "primary", "novel",
        "new", "different", "similar", "certain", "particular",
    })

    # meta 语言黑名单 — 这些词出现在 quantity 里说明不是物理量
    # (从 INSTRUCTIONS 的 meta 句子 "study the related work and data" 等误提取)
    _META_TERMS = frozenset({
        "related work", "data", "deliverables", "report", "written",
        "complete", "instructions", "workspace", "files", "code",
        "figures", "results", "methodology", "discussion", "fully",
        "task", "phase", "tool", "call",
    })

    # 数值模式 — 行里有真实数值才算 "真的分析了", future work / caveat 无数值 = missing
    # 优先匹配小数/科学计数法/单位/比较符; 最后兜底纯整数 (2位+), 排除列表编号 "1. " / "1."
    # ponytail: 纯整数兜底会放过 "Figure 28" 这类 false positive, 但 ±1 窗口 + keyword
    # 过滤已经把风险压到可接受范围. 漏报 (agent 分析了但 middleware 仍提醒, 浪费 token)
    # 比误报 (agent 没分析但 middleware 不提醒, 丢分) 好 — 前者浪费 token, 后者丢分.
    _NUMERIC_PATTERN = re.compile(
        r"\d+\.\d+"  # 小数 5.2 / 1.20 (排除 "1." 列表编号)
        r"|\d+e[\-+]?\d+"  # 科学计数法 1e-20
        r"|\d+\s*[×x*]\s*10"  # 1.2 × 10^...
        r"|\d+\s*(?:<|≤|>|≥|±)"  # 比较符 μ < 5 / ± 0.02
        r"|\d+\s*(?:ev|gev|mev|kev|m☉|msun|myr|gyr|yr|kg|hz)\b"  # 数字+单位
        r"|\d{2,}(?!\.\d)(?!\.\s)(?!\.$)"  # 2位+整数, 排除小数/列表编号
    )

    def _extract_quantities(self, text: str) -> list[str]:
        """从 INSTRUCTIONS text 提取 candidate physical quantities."""
        quantities: list[str] = []
        for pat in self._QUANTITY_PATTERNS:
            for m in pat.finditer(text):
                for g in m.groups():
                    q = g.strip().lower()
                    if 3 <= len(q) <= 80:
                        # 过滤 meta 语言 (related work / data / deliverables 等)
                        if any(term in q for term in self._META_TERMS):
                            continue
                        quantities.append(q)
        return quantities

    def _extract_keywords(self, phrase: str) -> list[str]:
        """从短语提取所有非停用词作为关键词, 用于宽松匹配.

        ponytail: 取所有实词而非前 N 个, 这样 "statistically rigorous upper
        limits on ulb masses" 里的 "ulb" / "masses" 都能匹配 report 里的
        "ULB mass". 按空白/连字符/斜杠拆分让 "metal/insulator" 拆成 metal +
        insulator, "d/g/i-wave" 拆出 wave (d/g/i 太短被过滤, 但 wave/anisotropy
        足以匹配). 升级路径: 加 stemmer (masses→mass) 或 synonym dict.
        """
        words = [
            w for w in re.split(r"[\s\-/]+", phrase)
            if w.lower() not in self._STOPWORDS and len(w) > 2
        ]
        if not words:
            return [phrase]
        # 所有实词 + 整个短语, 任一匹配即视为 covered
        return words + [phrase]

    def _check_coverage(
        self, instructions_text: str, report_text: str
    ) -> list[str]:
        """返回 report.md 里缺失的物理量列表.

        判定逻辑 (行级 ±1 窗口数值检测):
        - keyword 不出现 = missing
        - keyword 出现但 ±1 行窗口内无数值 (future work / caveat) = missing
        - keyword 出现且 ±1 行窗口内有数值 = covered

        ponytail: 按行而非按段落, 避免列表项跨匹配 (L220 self-interactions
        和 L224 3Myr 在同一列表段落但语义无关). ±1 行窗口容忍数值跟在
        keyword 下一行的情况. 升级路径: NLI 模型判断 keyword 是否被定量分析.
        """
        required = self._extract_quantities(instructions_text)
        if not required:
            return []
        lines = report_text.lower().split("\n")
        missing: list[str] = []
        for q in required:
            keywords = self._extract_keywords(q)
            covered = False
            for i, line in enumerate(lines):
                if not any(kw in line for kw in keywords):
                    continue
                # ±1 行窗口
                window = "\n".join(lines[max(0, i - 1):i + 2])
                if self._NUMERIC_PATTERN.search(window):
                    covered = True
                    break
            if not covered:
                missing.append(q)
        return missing

    def _build_frontier_msg(self, missing: list[str]) -> str:
        return (
            "[FRONTIER TASK — DeliverableCoverageMiddleware]\n"
            "report.md 缺失以下 INSTRUCTIONS.md 明确要求的物理量:\n"
            + "".join(f"  - {m}\n" for m in missing)
            + "\n每个缺失量 = 该 criterion 0 分. 必须在 report.md 补充定量结果 "
            "(numeric value + unit). 不要写 'left for future work', "
            "必须给出数值结果或上界. 完成后再次确认所有量都已覆盖."
        )

    def _inject_frontier(self, request) -> None:
        """读 INSTRUCTIONS.md + report.md, 缺失量注入 frontier task."""
        try:
            cwd = Path.cwd()
            instructions = cwd / "INSTRUCTIONS.md"
            report = cwd / "report" / "report.md"
            if not instructions.exists() or not report.exists():
                return  # report 还没写, Phase 3 强制写 report 的逻辑在 system prompt
            inst_text = instructions.read_text(encoding="utf-8")
            report_text = report.read_text(encoding="utf-8")
            missing = self._check_coverage(inst_text, report_text)
            if not missing:
                return
            frontier = self._build_frontier_msg(missing)
            # prepend SystemMessage, 不累积 — messages 每轮从 state 重建
            msgs = getattr(request, "messages", None)
            if msgs is None:
                return
            request.messages = [SystemMessage(content=frontier)] + list(msgs)
            logger.info(f"DeliverableCoverage injected frontier: {missing}")
        except Exception as e:
            logger.debug(f"DeliverableCoverage inject skipped: {e}")

    def wrap_model_call(self, request, handler):
        self._inject_frontier(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._inject_frontier(request)
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


# ── self-check ────────────────────────────────────────────────

def _self_check() -> int:
    """assert-based demo: 验证 DeliverableCoverageMiddleware 正则提取 + 覆盖检查."""
    m = DeliverableCoverageMiddleware()

    # 场景 1: Astronomy_000 真实 INSTRUCTIONS — 提取 "X and Y" 物理量
    inst = (
        "To constrain the properties of ultralight bosons by developing and "
        "applying a novel Bayesian statistical framework. This framework "
        "translates the physics of black hole superradiance into a probabilistic "
        "model. The goal is to derive statistically rigorous upper limits on "
        "ULB masses and self-interaction coupling strengths, thereby using "
        "astrophysical data to probe fundamental particle physics."
    )
    quantities = m._extract_quantities(inst)
    # 应该提取出 mass 相关和 coupling 相关
    assert any("mass" in q for q in quantities), f"mass not extracted: {quantities}"
    assert any("coupling" in q or "self-interaction" in q for q in quantities), \
        f"coupling not extracted: {quantities}"
    print(f"[CHECK] extracted quantities: {quantities}")

    # 场景 2: report 缺 coupling (future work 段落无数值) — 应报 missing
    report_missing_coupling = (
        "# Bayesian Constraints on Ultralight Bosons\n\n"
        "## Results\n"
        "We derive an upper limit on the ULB mass μ < 5.2e-20 eV at 95% CL.\n"
        "The mass constraint is robust across both IRAS and M33 datasets.\n\n"
        "## Future Work\n"
        "Including boson self-interactions would allow us to constrain "
        "not just the mass but also the coupling strength of ultralight bosons.\n"
    )
    missing = m._check_coverage(inst, report_missing_coupling)
    assert any("coupling" in x or "self-interaction" in x for x in missing), \
        f"coupling should be missing (future work, no numeric): {missing}"
    print(f"[CHECK] missing (coupling in future work): {missing}")

    # 场景 3: report 全覆盖 (mass + coupling 都有数值) — 应返回空
    report_full = (
        "# Bayesian Constraints on Ultralight Bosons\n\n"
        "## Mass Constraints\n"
        "ULB mass μ < 5.2e-20 eV at 95% CL.\n\n"
        "## Self-Interaction Coupling\n"
        "The self-interaction coupling strength g is constrained to g < 1.3e-17.\n"
        "Coupling bounds are derived from the superradiance condition.\n"
    )
    missing_full = m._check_coverage(inst, report_full)
    assert missing_full == [], f"should be fully covered: {missing_full}"
    print(f"[CHECK] missing (full coverage): {missing_full}")

    # 场景 4: frontier message 格式
    msg = m._build_frontier_msg(["self-interaction coupling strengths"])
    assert "FRONTIER TASK" in msg
    assert "self-interaction coupling strengths" in msg
    assert "0 分" in msg
    print(f"[CHECK] frontier msg OK")

    # 场景 5: Material_000 INSTRUCTIONS — "classifications such as X and Y" 模式
    # X = metal/insulator (斜杠复合词), Y = d/g/i-wave anisotropy (连字符+斜杠)
    mat_inst = (
        "The output is a list of candidate materials predicted to be "
        "altermagnets with high probability, along with their electronic "
        "structure properties confirmed by first-principles calculations "
        "(e.g. 50 newly discovered altermagnets with classifications such "
        "as metal/insulator and d/g/i-wave anisotropy)."
    )
    mat_q = m._extract_quantities(mat_inst)
    assert any("metal" in q or "insulator" in q for q in mat_q), \
        f"metal/insulator not extracted: {mat_q}"
    assert any("anisotropy" in q or "wave" in q for q in mat_q), \
        f"d/g/i-wave anisotropy not extracted: {mat_q}"
    print(f"[CHECK] Material_000 extracted: {mat_q}")

    # 场景 6: Material_000 report 缺分类 — 应报 missing
    mat_report_missing = (
        "# Altermagnetic Materials Discovery\n\n"
        "## Results\n"
        "The GNN model achieves ROC-AUC = 0.486 on the candidate set.\n"
        "Top-50 precision is 0.080 with 4 true positives identified.\n\n"
        "## Discussion\n"
        "The model struggles due to extreme class imbalance.\n"
    )
    mat_missing = m._check_coverage(mat_inst, mat_report_missing)
    assert len(mat_missing) >= 1, \
        f"should report missing classifications: {mat_missing}"
    print(f"[CHECK] Material_000 missing (no classification): {mat_missing}")

    # 场景 7: Material_000 report 有分类数值 — 应 covered
    mat_report_full = (
        "# Altermagnetic Materials Discovery\n\n"
        "## Classification Results\n"
        "Of 50 discovered altermagnets, 32 are metals and 18 are insulators.\n"
        "Anisotropy analysis: 12 d-wave, 8 g-wave, 10 i-wave patterns.\n"
    )
    mat_missing_full = m._check_coverage(mat_inst, mat_report_full)
    assert mat_missing_full == [], \
        f"should be fully covered: {mat_missing_full}"
    print(f"[CHECK] Material_000 full coverage: {mat_missing_full}")

    print("[MIDDLEWARES] self-check OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
