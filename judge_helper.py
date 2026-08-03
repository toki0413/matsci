"""Minimal judge helper — 直接用 OpenAI client, 绕过 structai 的线程/reasoning 兼容问题.

deepseek-v4-flash 是 reasoning model, 推理过程会消耗全部 max_tokens, JSON 没机会输出.
Judge 需要稳定 JSON 而非深度推理, 默认用 deepseek-chat (非 reasoning).
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
) -> dict[str, Any] | None:
    """Call LLM judge and parse JSON response.

    Returns parsed dict or None on failure.
    Handles <think> tags from reasoning models by splitting on </think>.
    """
    from openai import OpenAI
    from json_repair import repair_json

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    api_base = os.environ.get("JUDGE_API_BASE", "https://api.deepseek.com")
    # 默认 deepseek-chat: 非 reasoning, 直接输出 JSON, 不会用推理吃光 token.
    # deepseek-v4-flash 的 reasoning 会消耗 16000+ chars, max_tokens 扛不住.
    model = model or os.environ.get("JUDGE_MODEL_NAME", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=api_base)

    last_err = None
    for try_idx in range(max_try):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            if not content:
                # reasoning model 偶尔把输出放 reasoning_content
                msg = resp.choices[0].message
                rc = getattr(msg, "reasoning_content", None) or ""
                if rc:
                    content = rc
                else:
                    last_err = f"empty response (finish={resp.choices[0].finish_reason})"
                    continue

            # 剥离 <think>...</think> (reasoning model 可能带)
            if "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()

            # 从 content 中提取 JSON — 模型可能前后带解释文本
            # 找第一个 { 和最后一个 }
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]

            parsed = json.loads(repair_json(content))
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
