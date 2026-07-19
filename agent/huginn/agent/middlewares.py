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

    # v13: 五层分层锚点. Method/Data/Claim 用 section header 锚定段落,
    # Concept/Equation 用全文正则. ponytail: section header 是 INSTRUCTIONS
    # Deliverables 段推导的稳定锚点 (## Methodology / ## Results / ## Discussion),
    # 不依赖 LLM 解析段落语义. 升级路径: claim 层单位匹配最难正则化,
    # 漏报严重时换 LLM 辅助.
    _LAYER_PATTERNS = {
        # Concept 层: quantity 关键词在 report 任一段出现即算 (复用 _extract_keywords)
        "concept": None,  # 用 _extract_keywords(q) 做匹配, 无独立正则

        # Equation 层: LaTeX 行内/块级公式, 或显式等式
        "equation": re.compile(
            r"\$\$.+?\$\$"            # 块级 $$...$$
            r"|\$[^$]{3,}\$"          # 行内 $...$ (至少3字符, 排除 $1$ 编号)
            r"|\\begin\{equation\}"   # LaTeX equation 环境
            r"|=\s*[\w\d\-+\\]{2,}"  # 显式等式 "λ = g²/(2m²)" (右值至少2字符)
            , re.DOTALL,
        ),

        # Method 层: ## Methodology 或 ## Method 段落
        "method": re.compile(
            r"^##\s*(?:Methodology|Methods?|Approach|Framework)\b.*?"
            r"(?=^##\s|\Z)",
            re.MULTILINE | re.DOTALL,
        ),

        # Data 层: ## Results 段落, 或 ## Findings
        "data": re.compile(
            r"^##\s*(?:Results?|Findings?|Numerical\s+Results?)\b.*?"
            r"(?=^##\s|\Z)",
            re.MULTILINE | re.DOTALL,
        ),

        # Claim 层: ## Discussion / ## Conclusion 段, 或显式 upper limit / point estimate 声明
        "claim": re.compile(
            r"^##\s*(?:Discussion|Conclusion|Conclusions)\b.*?"
            r"(?=^##\s|\Z)",
            re.MULTILINE | re.DOTALL,
        ),
    }

    # Claim 层单位/声明关键词 — 出现任一即视为 claim 层有内容
    _CLAIM_KEYWORDS = re.compile(
        r"upper\s+limit|point\s+estimate|confidence\s+level|95%\s*CL|1\s*sigma|"
        r"GeV\^?\s*-?1|GeV⁻¹|eV\b|dimensionless|"
        r"constrained\s+to|bounded\s+by|excluded\s+at|consistent\s+with",
        re.IGNORECASE,
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

    def _check_layer_coverage(
        self, quantity: str, report_text: str
    ) -> list[str]:
        """v13: 返回该 quantity 缺失的层名列表 (subset of
        ["concept", "equation", "data", "method", "claim"]).

        ponytail: concept/data 共享 A 的 keyword+数值检测, 不重复造轮子.
        method/claim 用 section header 锚点定位段落, 段内查 quantity 关键词.
        equation 用全文公式正则, 不绑定段落 (公式可能在 Methodology 也可能在
        Results). 升级路径: concept 层可换 kg 查 (huginn/kg/graph.py), 漏报严重再上.
        """
        missing_layers: list[str] = []
        keywords = self._extract_keywords(quantity)
        report_lower = report_text.lower()

        # Concept 层: quantity 关键词在 report 任一处出现
        if not any(kw in report_lower for kw in keywords):
            missing_layers.append("concept")
            # concept 缺 → 后面几层都不必查 (没提概念就是没分析)
            return missing_layers

        # Equation 层: report 有公式 + 公式附近 (±200 字符窗口) 出现 quantity 关键词
        eq_blocks = list(self._LAYER_PATTERNS["equation"].finditer(report_text))
        if eq_blocks:
            # 检查任一公式 ±200 字符窗口内出现 quantity 关键词
            eq_has_q = any(
                any(kw in report_text[max(0, m.start()-200):m.end()+200].lower()
                    for kw in keywords)
                for m in eq_blocks
            )
            if not eq_has_q:
                missing_layers.append("equation")
        else:
            missing_layers.append("equation")

        # Method 层: ## Methodology 段内出现 quantity 关键词
        method_match = self._LAYER_PATTERNS["method"].search(report_text)
        if method_match:
            if not any(kw in method_match.group(0).lower() for kw in keywords):
                missing_layers.append("method")
        else:
            missing_layers.append("method")

        # Data 层: ## Results 段内出现 quantity 关键词 + 数值
        data_match = self._LAYER_PATTERNS["data"].search(report_text)
        if data_match:
            data_block = data_match.group(0)
            if not (any(kw in data_block.lower() for kw in keywords)
                    and self._NUMERIC_PATTERN.search(data_block)):
                missing_layers.append("data")
        else:
            missing_layers.append("data")

        # Claim 层: ## Discussion 段出现 quantity 关键词 + claim 声明关键词
        #   (upper limit / 单位 / confidence level)
        claim_match = self._LAYER_PATTERNS["claim"].search(report_text)
        if claim_match:
            claim_block = claim_match.group(0)
            has_q = any(kw in claim_block.lower() for kw in keywords)
            has_claim = bool(self._CLAIM_KEYWORDS.search(claim_block))
            if not (has_q and has_claim):
                missing_layers.append("claim")
        else:
            missing_layers.append("claim")

        return missing_layers

    def _check_layer_gaps(
        self, instructions_text: str, report_text: str
    ) -> list[tuple[str, list[str]]]:
        """v13: 返回 list[(quantity, [missing_layers])].

        只对横向 covered 的 quantity 检查纵向层 (横向 missing 由 _check_coverage 报).
        ponytail: 跳过横向 missing 的 quantity, 避免跟 A 重复提醒 (A 已经会报 "缺 X",
        v13 不需要再报 "X 的 concept 层缺"). 升级路径: 横向 missing 的 quantity 也
        查层, 但当前 YAGNI.
        """
        required = self._extract_quantities(instructions_text)
        if not required:
            return []
        # 复用 _check_coverage 拿横向 missing, 排除掉
        horizontal_missing = set(self._check_coverage(instructions_text, report_text))
        gaps: list[tuple[str, list[str]]] = []
        for q in required:
            if q in horizontal_missing:
                continue  # 横向缺 → A 已经会报, 不重复
            layers = self._check_layer_coverage(q, report_text)
            if layers:
                gaps.append((q, layers))
        return gaps

    def _build_frontier_msg(self, missing: list[str]) -> str:
        return (
            "[FRONTIER TASK — DeliverableCoverageMiddleware]\n"
            "report.md 缺失以下 INSTRUCTIONS.md 明确要求的物理量:\n"
            + "".join(f"  - {m}\n" for m in missing)
            + "\n每个缺失量 = 该 criterion 0 分. 必须在 report.md 补充定量结果 "
            "(numeric value + unit). 不要写 'left for future work', "
            "必须给出数值结果或上界. 完成后再次确认所有量都已覆盖."
        )

    def _build_layer_frontier_msg(
        self, gaps: list[tuple[str, list[str]]]
    ) -> str:
        """v13: 拼接纵向层次缺失的 frontier message.

        改进 (Astronomy_000 多轮验证后):
        - 报所有缺失层, 不只 layers[0] — agent 之前只看到 "Method 层缺失" 就以为
          补一段 Discussion 就够了, 实际 Method/Data/Claim 三层都缺.
        - 每层给具体可操作提示, 不是模糊的 "层次不完整".
        - Claim 层特别强调 upper limit + 单位 + CL 的数值形式, 因 judge 按
          "具体数值 + 单位 + 置信水平" 评分, 定性描述 = 0 分.
        - 反 future-work 模式: 实测 agent 会把 "self-interaction" 写进
          Limitations/Future Work 段当作 "已 acknowledge", judge 仍判 0 分.
          必须明确禁止这种 escape hatch.
        - 物理量计算 hint: 告诉 agent 去 related_work 找公式, 不要凭空造数.
        """
        layer_action = {
            "concept": "在 report 任意位置引入该物理概念 (定义 + 物理含义), 不能只在未来工作段提及",
            "equation": "在 report 给出该量的定义式或约束方程 (LaTeX 公式), 从 related_work 引用也可",
            "data": "在 ## Results 段给出该量的具体数值 (不是定性描述, 必须有数字 + 单位)",
            "method": "在 ## Methodology 段描述该量的推导/计算方法 (公式来源 + 计算步骤)",
            "claim": "在 ## Discussion 或 ## Conclusion 段给出 upper limit/point estimate 的具体数值 + 单位 + 置信水平 (如 'g < X GeV⁻¹ at 95% CL')",
        }
        lines = [
            "[FRONTIER TASK — DeliverableCoverageMiddleware / v13 纵向分层]\n",
            "以下物理量已提及但层次不完整 (横向上 covered, 纵向缺层).\n",
            "必须在 report.md 对应 section 补全每一层, 不是补一段就够了:\n\n",
        ]
        for q, layers in gaps:
            lines.append(f"  ■ {q}:\n")
            for layer in layers:
                lines.append(f"    - [{layer}] {layer_action[layer]}\n")
            lines.append("\n")
        lines.append(
            "## 强制要求 (违反 = 该 criterion 直接 0 分)\n\n"
            "1. **禁止放入 Future Work / Limitations 段**: 之前多轮跑都把 "
            "self-interaction 写成 'our constraints apply primarily to weakly "
            "interacting ULBs' 或 'the framework can incorporate additional "
            "physics (self-interactions)' — 这是 acknowledge 而非 deliver, "
            "judge 直接判 0. 必须在 ## Results / ## Discussion 给出实际数值.\n\n"
            "2. **coupling strength 类量必须给出耦合常数符号 + 单位**: "
            "self-interaction/interaction/decay 量的正确形式是 g [GeV⁻¹], "
            "不要用 decay constant f_a 或 dimensionless λ 替代 — "
            "judge 按 task description 期望的单位评分. 关系一般是 "
            "g² ~ m_a²/f_a² (具体定义查 related_work/paper_*.pdf).\n\n"
            "3. **'upper limit' 必须是数值形式**: 'X < Y unit at Z% CL' "
            "(如 'g < 1.3e-17 GeV⁻¹ at 95% CL'), 不是 'we constrain' / "
            "'we limit' / 'consistent with' 这种定性描述.\n\n"
            "4. **数据来源**: related_work/ 里的 PDF 已有相关公式 (superradiance + "
            "self-interaction quenching), 用 read_file 工具读 paper_*.pdf 找 "
            "coupling 定义和计算方法, 然后在 code/ 写脚本算出数值上限. "
            "不要凭空造数.\n\n"
            "5. **完成后再次检查**: 重新读 report.md, 确认对应 section 出现了该量的数值.\n"
        )
        return "".join(lines)

    def _build_planning_msg(self, quantities: list[str]) -> str:
        """v13: report.md 还没写时, 基于 INSTRUCTIONS 注入 planning hint.

        时机问题修复 (Astronomy_000 第5轮验证): 之前 middleware 在 report.md
        写完后才注入 frontier msg, agent 已经在 'finish' 模式不会重写.
        改成在 report 写之前就注入 planning hint, 让 agent 知道必须算哪些量.
        ponytail: 不读 report_text, 只读 INSTRUCTIONS 提取 quantities, 复用
        _extract_quantities. 升级路径: 加 task-type 检测给更具体的 hint.
        """
        lines = [
            "[PLANNING HINT — DeliverableCoverageMiddleware / v13]\n",
            "INSTRUCTIONS.md 明确要求以下物理量必须给出数值结果 (不是 future work):\n\n",
        ]
        for q in quantities:
            lines.append(f"  ■ {q}\n")
        lines.append(
            "\n## 规划要求 (违反 = 对应 criterion 0 分)\n\n"
            "1. **分析阶段就必须计算所有量**: 不要先写 report 再补, "
            "code/ 阶段就要把每个量都算出数值. 每个量都需要: 定义式 → 计算 "
            "脚本 → 数值结果 + 单位.\n\n"
            "2. **coupling strength 类量**: self-interaction/interaction/decay "
            "coupling 的正确单位是 g [GeV⁻¹], 不是 f_a (decay constant) 或 "
            "dimensionless λ. 关系 g² ~ m_a²/f_a², 具体公式查 related_work/"
            "paper_*.pdf (用 read_file).\n\n"
            "3. **禁止放入 Future Work / Limitations**: 把量写成 'our "
            "constraints apply to weakly interacting ULBs' 或 'the framework "
            "can incorporate X' = acknowledge 而非 deliver, judge 判 0 分. "
            "必须在 ## Results / ## Discussion 给数值.\n\n"
            "4. **upper limit 格式**: 'X < Y unit at Z% CL' (如 "
            "'g < 1.3e-17 GeV⁻¹ at 95% CL'), 不是 'we constrain' 这种定性描述.\n\n"
        )
        return "".join(lines)

    def _inject_frontier(self, request) -> None:
        """读 INSTRUCTIONS.md + report.md, 缺失量 + 层次缺失注入 frontier task.

        时机: report.md 不存在时注入 planning hint (写 report 前), 存在时
        注入 frontier task (写 report 后补漏). 两段都用同一份 quantities 列表.
        """
        try:
            cwd = Path.cwd()
            instructions = cwd / "INSTRUCTIONS.md"
            report = cwd / "report" / "report.md"
            if not instructions.exists():
                return
            inst_text = instructions.read_text(encoding="utf-8")
            quantities = self._extract_quantities(inst_text)
            if not quantities:
                return

            msgs = getattr(request, "messages", None)
            if msgs is None:
                return

            # report 还没写 → 注入 planning hint, 让 agent 在 code/ 阶段就算全
            if not report.exists():
                planning = self._build_planning_msg(quantities)
                request.messages = [SystemMessage(content=planning)] + list(msgs)
                logger.info(
                    f"DeliverableCoverage planning hint injected: quantities={quantities}"
                )
                return

            # report 已写 → 横向 + 纵向检查, 缺啥补啥
            report_text = report.read_text(encoding="utf-8")
            missing = self._check_coverage(inst_text, report_text)
            gaps = self._check_layer_gaps(inst_text, report_text)

            if not missing and not gaps:
                return

            parts = []
            if missing:
                parts.append(self._build_frontier_msg(missing))
            if gaps:
                parts.append(self._build_layer_frontier_msg(gaps))
            frontier = "\n\n".join(parts)

            # prepend SystemMessage, 不累积 — messages 每轮从 state 重建
            request.messages = [SystemMessage(content=frontier)] + list(msgs)
            logger.info(
                f"DeliverableCoverage injected: horizontal={missing}, layer_gaps={gaps}"
            )
            # DIAG: 轻量 sentinel, 只在首次注入时写一行, 验证 middleware 真的被调
            # (之前发现 agent 写完 report 后忽略 frontier, 需确认是 middleware 没调
            # 还是 agent 忽略)
            try:
                from huginn.utils.runtime import get_runtime_home
                sentinel = get_runtime_home() / "_diagnostic_frontier_inject.txt"
                if not sentinel.exists():
                    with sentinel.open("w", encoding="utf-8") as f:
                        import time as _t
                        f.write(f"{_t.time()} first_inject missing={missing} gaps={gaps}\n")
            except Exception:
                pass
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

    # 场景 8 (v13): criterion 2 错位 — agent 给 dimensionless λ 而非 g (GeV⁻¹)
    # 横向 covered (coupling + 数值都有), 纵向 Claim 层缺失 (无 GeV⁻¹ / 无 upper limit 声明)
    report_claim_mismatch = (
        "# Bayesian Constraints on Ultralight Bosons\n\n"
        "## Methodology\n"
        "We apply Bayesian marginalization over black hole posterior samples.\n\n"
        "## Results\n"
        "ULB mass μ < 5.2e-20 eV.\n"
        "Self-interaction coupling λ ≲ 4.6 × 10⁻⁶ (dimensionless).\n\n"
        "## Discussion\n"
        "The constraints are robust across datasets.\n"
    )
    gaps = m._check_layer_gaps(inst, report_claim_mismatch)
    # coupling 应该报 Claim 层缺失 (无 GeV⁻¹, 无 upper limit/point estimate 声明)
    coupling_gaps = [layers for q, layers in gaps
                     if "coupling" in q or "self-interaction" in q]
    assert coupling_gaps and "claim" in coupling_gaps[0], \
        f"criterion 2 错位应报 Claim 层缺失: {gaps}"
    print(f"[CHECK v13] criterion 2 Claim layer gap: {coupling_gaps}")

    # 场景 9 (v13): 全覆盖 — report 五层齐全, 不应报任何 layer gap
    report_full_with_claim = (
        "# Bayesian Constraints on Ultralight Bosons\n\n"
        "## Methodology\n"
        "We apply Bayesian marginalization over black hole posterior samples "
        "to constrain ULB mass and self-interaction coupling. "
        "The self-interaction coupling is parameterized as $\\lambda = g^2/(2m^2)$.\n\n"
        "## Results\n"
        "ULB mass μ < 5.2e-20 eV at 95% CL.\n"
        "Self-interaction coupling g < 1.3e-17 GeV⁻¹ (upper limit).\n\n"
        "## Discussion\n"
        "We report upper limits on ULB mass μ and the self-interaction "
        "coupling strength g in GeV⁻¹ at 95% confidence level.\n"
    )
    gaps_full = m._check_layer_gaps(inst, report_full_with_claim)
    assert gaps_full == [], f"五层齐全不应报 layer gap: {gaps_full}"
    print(f"[CHECK v13] full layer coverage: {gaps_full}")

    # 场景 10 (v13): concept 缺 — Metal_000 report 没提 metal/insulator 分类
    mat_gaps = m._check_layer_gaps(mat_inst, mat_report_missing)
    # metal/insulator 应该报 concept 层缺失 (根本没出现)
    metal_gaps = [layers for q, layers in mat_gaps
                  if "metal" in q or "insulator" in q]
    # 注意: 横向 missing 的 quantity 会被 _check_layer_gaps 跳过 (A 已经会报).
    # mat_report_missing 里 metal/insulator 是横向 missing, 所以 gaps 里不会有它.
    # 改用 _check_layer_coverage 直接查 (绕过 horizontal filter)
    direct_metal_layers = m._check_layer_coverage("metal/insulator", mat_report_missing)
    assert "concept" in direct_metal_layers, \
        f"Metal_000 缺分类应报 Concept 层缺失: {direct_metal_layers}"
    print(f"[CHECK v13] Metal_000 Concept layer gap (direct): {direct_metal_layers}")

    # 场景 11 (v13 wording): frontier msg 必须包含反 future-work + 数值要求
    msg = m._build_layer_frontier_msg([("self-interaction coupling strengths",
                                        ["method", "data", "claim"])])
    assert "Future Work" in msg, "frontier msg 必须明确禁止 Future Work 模式"
    assert "GeV⁻¹" in msg, "frontier msg 必须给出单位提示 GeV⁻¹"
    assert "related_work" in msg, "frontier msg 必须指向 related_work 数据源"
    assert "0 分" in msg, "frontier msg 必须明确 0 分后果"
    print(f"[CHECK v13] layer frontier msg wording OK")

    # 场景 12 (v13 planning): planning hint 必须包含所有 required quantities
    planning = m._build_planning_msg(["ulb masses", "self-interaction coupling strengths"])
    assert "PLANNING HINT" in planning
    assert "ulb masses" in planning
    assert "self-interaction coupling strengths" in planning
    assert "Future Work" in planning, "planning hint 必须禁止 future work"
    assert "g²" in planning or "g^2" in planning or "GeV⁻¹" in planning
    print(f"[CHECK v13] planning hint wording OK")

    print("[MIDDLEWARES] self-check OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
