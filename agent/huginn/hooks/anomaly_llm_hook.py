"""PRT Level 1 — LLM 异常判定钩子.

Level 0 用规则扫工具输出, 只能抓硬信号(error / 关键词 / 标准值偏差).
真实对话里的"软异常"——输入数据可疑、多源数据对不上、计算结果物理上
不合理但工具没报错——规则根本扫不出来. Level 1 在 PostToolUse 后塞一次
小模型(默认 deepseek-chat)做判定, 让 LLM 兜底这类软异常.

默认关闭, 通过 HUGINN_PRT_LEVEL1=1 在 factory 注册时开启.
LLM 调用失败 / 超时一律静默跳过, 绝不影响 agent 主流程.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 强制观察的高风险工具: 即使规则初筛没发现可疑信号, 也要调 LLM 判定.
# 这些工具能产生错误的数值结果(负带隙/发散/量级离谱), 规则不好抓.
_FORCE_WATCH_TOOLS = {"validate_tool", "numerical_tool", "structure_tool"}

# 同一个 tool 多久内不重复判定(秒). 防止短时间连续调用刷屏烧钱.
_THROTTLE_SECONDS = 10

# prompt 里 args / result 截断长度, 控制单次 token 量
_MAX_ARG_CHARS = 600
_MAX_RESULT_CHARS = 1200

# 规则初筛: result/error 文本里出现这些关键词就认为可疑, 才调 LLM 复判.
# 正常调用(90%+)不会命中这些词, 不花 LLM 钱.
_SUSPICION_PATTERNS = [
    "error", "fail", "exception", "traceback", "nan", "none", "null",
    "inf", "空", "无法", "不能", "invalid", "not found", "missing",
    "warning", "warn", "异常", "不合理",
]

# LLM 返回的 category 合法值
_VALID_CATEGORIES = {
    "INPUT_DATA",
    "DATA_CONFLICT",
    "TOOL_FAILURE",
    "COMPUTATION_RESULT",
    "NONE",
}


def _quick_suspicion_check(tool_name: str, result: Any, error: BaseException | None) -> bool:
    """规则初筛: 判断工具结果是否有可疑信号.

    正常调用返回 False (跳过 LLM, 省钱), 可疑返回 True (调 LLM 复判).
    高风险工具(validate/numerical/structure)始终返回 True, 因为数值
    类的软异常(负带隙/量级离谱)规则不好抓, 必须靠 LLM.
    """
    # 高风险工具强制观察
    if tool_name in _FORCE_WATCH_TOOLS:
        return True
    # 工具报错, 直接可疑
    if error is not None:
        return True
    # result 文本里找可疑关键词
    text = ""
    try:
        text = json.dumps(result, ensure_ascii=False, default=str).lower()
    except Exception:
        text = str(result).lower()
    for pattern in _SUSPICION_PATTERNS:
        if pattern in text:
            return True
    return False


# 判定 prompt. 加了分类边界澄清和 few-shot 示例, 让 LLM 区分
# INPUT_DATA(输入参数有问题) vs COMPUTATION_RESULT(工具没报错但结果不合理)
# vs DATA_CONFLICT(用户给的值和工具返回的值对不上).
_JUDGE_SYSTEM = (
    "你是材料计算工具的异常检测助手。判断一次工具调用是否异常。\n"
    "异常类型:\n"
    "- INPUT_DATA: 工具的输入参数有问题(文件路径不存在/格式不对/缺关键字段/单位错误)\n"
    "- DATA_CONFLICT: 多源或前后数据互相矛盾(用户给的值和工具返回的值对不上, 或两个来源给出不同值)\n"
    "- TOOL_FAILURE: 工具执行失败或返回错误(报错/异常/崩溃)\n"
    "- COMPUTATION_RESULT: 工具成功执行但结果物理上不合理(负带隙/发散/不收敛/量级离谱)\n"
    "- NONE: 正常\n"
    "\n"
    "关键区分:\n"
    "- 工具报错(error/exception) → TOOL_FAILURE\n"
    "- 输入参数格式/路径有问题 → INPUT_DATA\n"
    "- 工具没报错但结果数值不合理 → COMPUTATION_RESULT\n"
    "- 用户提问里给了一个值, 工具返回了另一个明显不同的值 → DATA_CONFLICT\n"
    "- 用户提问里给了两个矛盾值, 让工具判断 → DATA_CONFLICT (如果工具没识别出冲突)\n"
    "\n"
    "示例:\n"
    "工具: validate_tool, 入参: {\"file\": \"/dev/null\"}, 返回: {\"error\": \"file not found\"}\n"
    "→ {\"anomaly\": true, \"category\": \"INPUT_DATA\", \"reason\": \"输入文件路径不存在\"}\n"
    "工具: validate_tool, 入参: {\"bandgap\": -1.2}, 返回: {\"valid\": true, \"bandgap\": -1.2}\n"
    "→ {\"anomaly\": true, \"category\": \"COMPUTATION_RESULT\", \"reason\": \"带隙为负值, 物理上不合理\"}\n"
    "工具: materials_database_tool, 用户提问: \"硅带隙是 1.5 eV 吗\", 返回: {\"band_gap\": 1.12}\n"
    "→ {\"anomaly\": true, \"category\": \"DATA_CONFLICT\", \"reason\": \"用户说 1.5 eV, 工具返回 1.12 eV, 数据冲突\"}\n"
    "工具: rag_tool, 入参: {\"action\": \"get\", \"doc_id\": \"nonexistent-123\"}, 返回: {\"error\": \"document not found\"}\n"
    "→ {\"anomaly\": true, \"category\": \"TOOL_FAILURE\", \"reason\": \"文档不存在, 工具报错\"}\n"
    "\n"
    '只返回 JSON: '
    '{"anomaly": true/false, "category": "INPUT_DATA|DATA_CONFLICT|TOOL_FAILURE|COMPUTATION_RESULT|NONE", "reason": "一句话说明"}'
)


class AnomalyLLMHook:
    """PostToolUse 钩子: 用小模型判定工具结果是否异常, 异常则登记进 AnomalyLog.

    跟 Level 0 的 AnomalyDetectionHook 互补: Level 0 抓硬信号, 这里抓软异常.
    判定结果复用现有 4 类 category, detection_method 标为 llm_judgment 以示区别,
    不新增 category 类型.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        # 模型懒加载, 第一次调用时才建. 没配 key 时 import 阶段不会炸.
        self._model: Any = None
        # 初始化失败后永久跳过, 不在每次工具调用时重试, 避免反复打挂掉的 endpoint.
        self._model_init_failed = False
        # 节流表: {throttle_key: last_check_ts}.
        # thread_id 不在 post hook 的 context 里, 暂时按 (thread_id?, tool_name) 维度
        # 节流; 如果 ctx.metadata 里有 thread_id 就用, 没有就退化为只按 tool_name.
        self._last_check: dict[str, float] = {}

    async def __call__(self, ctx: HookContext) -> HookContext | None:
        try:
            # 所有工具都过规则初筛, 可疑的才调 LLM (省成本).
            # 高风险工具(validate/numerical/structure)强制观察, 不过初筛.
            if not _quick_suspicion_check(ctx.tool_name, ctx.result, ctx.error):
                return None

            # 节流: 同一 thread + tool 在窗口内只判一次
            throttle_key = self._throttle_key(ctx)
            now = time.monotonic()
            last = self._last_check.get(throttle_key, 0.0)
            if now - last < _THROTTLE_SECONDS:
                return None
            self._last_check[throttle_key] = now

            model = self._get_model()
            if model is None:
                return None

            verdict = await self._judge(ctx)
            if verdict is None:
                return None

            if verdict.get("anomaly") and verdict.get("category") in (
                "INPUT_DATA",
                "DATA_CONFLICT",
                "TOOL_FAILURE",
                "COMPUTATION_RESULT",
            ):
                self._log_anomaly(ctx, verdict)
        except Exception:
            # 任何意外都不能把 agent 主流程搞挂
            logger.warning("AnomalyLLMHook raised", exc_info=True)
        return None

    # ---- 模型管理 ----

    def _get_model(self) -> Any:
        """懒加载小模型. 优先 deepseek-chat, 失败一次后不再重试."""
        if self._model is not None:
            return self._model
        if self._model_init_failed:
            return None
        try:
            from huginn.models.registry import create_langchain_model

            # deepseek-chat 便宜快, 没配 DEEPSEEK_API_KEY 会抛 ValueError, 静默跳过.
            # temperature=0 让判定稳定, max_tokens 收紧省成本.
            self._model = create_langchain_model(
                provider="deepseek",
                model_name="deepseek-chat",
                temperature=0.0,
                max_tokens=200,
            )
            logger.info("AnomalyLLMHook: deepseek-chat 就绪")
        except Exception as exc:
            # 没 key / 网络挂 / 包没装都属于这类, 记 debug 就行, 别吵
            logger.debug("AnomalyLLMHook 模型初始化失败, 跳过: %s", exc)
            self._model_init_failed = True
            return None
        return self._model

    # ---- 判定 ----

    async def _judge(self, ctx: HookContext) -> dict | None:
        """调小模型判定, 返回解析后的 dict, 失败返回 None."""
        from langchain_core.messages import HumanMessage, SystemMessage

        args_text = self._stringify(ctx.args)[:_MAX_ARG_CHARS]
        result_text = self._stringify(ctx.result)[:_MAX_RESULT_CHARS]
        # 工具报错也喂给 LLM, 它能判成 TOOL_FAILURE
        error_text = str(ctx.error) if ctx.error else ""
        # 用户提问也喂给 LLM, 让它能识别"用户给的值"和"工具返回的值"
        # 是否冲突 (DATA_CONFLICT). 没有就只看工具自身的 args/result.
        user_msg = ""
        try:
            user_msg = str(ctx.metadata.get("user_message", "") or "")
        except Exception:
            pass
        user_msg = user_msg[:_MAX_ARG_CHARS]

        user_prompt = f"工具名: {ctx.tool_name}\n入参: {args_text}\n返回: {result_text}\n"
        if error_text:
            user_prompt += f"异常信息: {error_text[:_MAX_ARG_CHARS]}\n"
        if user_msg:
            user_prompt += f"用户提问: {user_msg}\n"
        user_prompt += "\n请判定是否有异常, 返回 JSON。"

        try:
            resp = await self._model.ainvoke(
                [
                    SystemMessage(content=_JUDGE_SYSTEM),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as exc:
            logger.debug("AnomalyLLMHook ainvoke 失败: %s", exc)
            return None

        content = resp.content if isinstance(resp.content, str) else ""
        return self._parse_verdict(content)

    @staticmethod
    def _parse_verdict(content: str) -> dict | None:
        """从 LLM 回复里抠 JSON, 容错 markdown 围栏和多余文本."""
        text = content.strip()
        if not text:
            return None
        # 去掉 ```json ... ``` 围栏
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
        # 模型偶尔在 JSON 前后加废话, 抠第一个 {...}
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        # 字段校验 + 归一化
        category = str(data.get("category", "NONE")).upper().strip()
        if category not in _VALID_CATEGORIES:
            category = "NONE"
        data["category"] = category
        data["anomaly"] = bool(data.get("anomaly", False))
        return data

    # ---- 登记 ----

    def _log_anomaly(self, ctx: HookContext, verdict: dict) -> None:
        from huginn.anomaly_log import Anomaly

        reason = str(verdict.get("reason", "")).strip() or "LLM 判定异常未给理由"
        category = verdict["category"]
        # severity 粗分: TOOL_FAILURE / COMPUTATION_RESULT 偏高, 其余中等
        severity = (
            "HIGH"
            if category in ("TOOL_FAILURE", "COMPUTATION_RESULT")
            else "MEDIUM"
        )

        # thread_id 从 hook context 的 metadata 拿, 没有就留空
        # 测试脚本靠这个字段把 anomaly 关联到具体会话
        thread_id = ""
        try:
            thread_id = str(ctx.metadata.get("thread_id", "") or "")
        except Exception:
            pass
        # user_message 也存一份, 方便事后排查"用户给的值"和"工具返回的值"
        # 到底哪里冲突. 截断避免 context_snapshot 爆炸.
        user_msg = ""
        try:
            user_msg = str(ctx.metadata.get("user_message", "") or "")[:_MAX_ARG_CHARS]
        except Exception:
            pass

        self._store.log(
            Anomaly(
                id="",
                ts=datetime.now(),
                category=category,
                severity=severity,
                description=f"[LLM判定] {ctx.tool_name}: {reason[:200]}",
                detection_method="llm_judgment",
                source="tool_output",
                context_snapshot={
                    "thread_id": thread_id,
                    "user_message": user_msg,
                    "tool": ctx.tool_name,
                    "args": self._shrink(ctx.args),
                    "result": self._stringify(ctx.result)[:_MAX_RESULT_CHARS],
                    "duration_ms": ctx.duration_ms,
                    "llm_reason": reason,
                },
                unresolved_dimensions=[
                    f"LLM 判定 {category}, 需复核: {reason[:120]}"
                ],
            )
        )

    # ---- 工具方法 ----

    @staticmethod
    def _throttle_key(ctx: HookContext) -> str:
        """节流 key. 有 thread_id 就按 (thread, tool), 没有就只按 tool."""
        thread_id = ""
        try:
            thread_id = str(ctx.metadata.get("thread_id", "") or "")
        except Exception:
            pass
        return f"{thread_id}:{ctx.tool_name}" if thread_id else ctx.tool_name

    @staticmethod
    def _stringify(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    @staticmethod
    def _shrink(args: Any) -> dict:
        """args 快照, 砍超长字段避免 context_snapshot 爆炸."""
        if not isinstance(args, dict):
            return {"_raw": str(args)[:200]}
        out: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 200:
                out[k] = v[:200] + "…"
            else:
                out[k] = v
        return out
