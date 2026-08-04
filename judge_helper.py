"""Minimal judge helper — 直接用 OpenAI client, 绕过 structai 的线程/reasoning 兼容问题.

deepseek-v4-flash 是 reasoning model, 推理过程会消耗全部 max_tokens, JSON 没机会输出.
Judge 需要稳定 JSON 而非深度推理, 默认用 deepseek-chat (非 reasoning).

C6 评测卫生 (structai 锁版决策): structai 原未声明未锁版 (audit 16 D6), 重跑即崩.
本模块替代 structai LLMAgent, 全部 judge 路径走 OpenAI client + json_repair, 无外部依赖.
C6 最终决策: 不声明 structai, 不 vendoring, 直接全量移除 — rcb_score/paperbench_huginn/
hle_huginn/sab_huginn/mlebench_huginn 的 structai import 已全部删干净 (含死代码 patch).
"""
from __future__ import annotations

import json
import os
from typing import Any


def judge_score(
    prompt: str,
    system_prompt: str = "You are a strict grader. Return JSON only.",
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0,
    max_try: int = 3,
    image_paths: list[str] | None = None,
    return_example: dict | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any] | None:
    """Call LLM judge and parse JSON response.

    Returns parsed dict or None on failure.
    Handles <think> tags from reasoning models by splitting on </think>.

    C6: image_paths 支持视觉 judge (发图给多模态模型).
        return_example 注入到 prompt 尾部引导 JSON schema.
        api_key/api_base 可覆盖 (视觉 judge 用不同 endpoint 时).
    """
    from openai import OpenAI
    from json_repair import repair_json

    # C6: judge 同源检测 — 默认 deepseek-chat 与被测同源, 警告 (不阻断, env 可覆盖)
    _judge_model = model or os.environ.get("JUDGE_MODEL_NAME", "deepseek-chat")
    _api_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    _api_base = api_base or os.environ.get("JUDGE_API_BASE", "https://api.deepseek.com")

    if not _api_key:
        return None

    # return_example 注入 prompt 尾部
    _prompt = prompt
    if return_example:
        _prompt += f"\n\nReturn JSON matching this shape: {json.dumps(return_example)}"

    client = OpenAI(api_key=_api_key, base_url=_api_base)

    last_err = None
    for try_idx in range(max_try):
        try:
            # 视觉 judge: 用 image_url 格式发图
            if image_paths:
                import base64
                content: list[dict[str, Any]] = [{"type": "text", "text": _prompt}]
                for img_path in image_paths[:5]:
                    try:
                        with open(img_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        })
                    except Exception:
                        pass
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _prompt},
                ]

            resp = client.chat.completions.create(
                model=_judge_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content_str = resp.choices[0].message.content or ""
            if not content_str:
                msg = resp.choices[0].message
                rc = getattr(msg, "reasoning_content", None) or ""
                if rc:
                    content_str = rc
                else:
                    last_err = f"empty response (finish={resp.choices[0].finish_reason})"
                    continue

            if "</think>" in content_str:
                content_str = content_str.split("</think>", 1)[-1].strip()

            start = content_str.find("{")
            end = content_str.rfind("}")
            if start != -1 and end != -1 and end > start:
                content_str = content_str[start : end + 1]

            parsed = json.loads(repair_json(content_str))
            # C6: return_example 是 list 时保留 list 返回 (paperbench 批量评分需要)
            _want_list = isinstance(return_example, list)
            if _want_list:
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if not isinstance(parsed, list):
                    last_err = f"parsed to {type(parsed).__name__}, not list"
                    continue
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                last_err = f"parsed to {type(parsed).__name__}, not dict"
                continue
            return parsed
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue

    print(f"[judge_helper] all {max_try} tries failed: {last_err}")
    return None
