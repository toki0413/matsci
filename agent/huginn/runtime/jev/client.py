"""JEV 系统一决策模型容错客户端.

按 TypeSafe ``/v1/systemone`` 契约, 用 httpx 直连 (不依赖 typesafe-sdk,
避免新增硬依赖). 三种原语都实现, 且**全部失败容错**:

  - 无 API key / 无网络 / 超时 / 响应异常 → 返回 None (调用方自行判空降级).
  - 解析只读 ``answers[<name>].<field>``, 字段缺失按 None/0 兜底.
  - transport 可注入, 便于单测 mock 而无需真实网络.

端点: ``POST {base_url}/v1/systemone``
  body: ``{"state": state, "model": model?, "questions": {name: {type, ...}}}``
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_TIMEOUT = 5.0

# JEV Choice 允许的最大选项数 (保守略小于 255 留余量)
_MAX_CHOICE_OPTIONS = 250


def _api_key() -> str | None:
    """读取 TypeSafe API key. 优先 TYPESAFE_API_KEY, 兼容 TYPESAFE_MANAGED_KEY."""
    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_MANAGED_KEY")
    return key or None


def _model_name() -> str | None:
    """可选显式模型名, 缺省交由服务端默认."""
    return os.environ.get("TYPESAFE_MODEL") or None


class JevClient:
    """第三方 JEV API 的极简容错封装."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: Any | None = None,
    ) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or _api_key()
        self._model = model or _model_name()
        self._timeout = timeout
        # transport 可注入一个可调对象 post(url, headers, json, timeout)->response,
        # 用于单测 mock; 默认 None 时惰性用 httpx.
        self._transport = transport

    # ── 连通性 ────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """无 API key 视为不可用 (fail-open: 调用方应跳过 JEV)."""
        return bool(self._api_key)

    # ── 原语 ─────────────────────────────────────────────────────
    def noul(self, state: Mapping[str, Any], instruction: str) -> float | None:
        """Noul: 单一 yes/no 命题 → 0..1 概率. 失败返回 None."""
        out = self.noul_batch(state, {"q": instruction})
        return (out.get("q") or {}).get("p") if out else None

    def noul_batch(
        self, state: Mapping[str, Any], instructions: Mapping[str, str]
    ) -> dict[str, dict[str, Any]]:
        """并行 Noul: {命题名: 命题} → {命题名: {"p", "confidence", "raw"}}."""
        questions = {
            name: {"type": "noul", "instructions": instr}
            for name, instr in instructions.items()
        }
        return self._roundtrip(state, questions)

    def choice(
        self,
        state: Mapping[str, Any],
        options: list[str],
        *,
        instruction: str,
    ) -> dict[str, Any] | None:
        """Choice: 从候选里选一 → {"choice","probabilities","confidence"}."""
        if len(options) > _MAX_CHOICE_OPTIONS:
            options = options[:_MAX_CHOICE_OPTIONS]
        questions = {
            "q": {
                "type": "choice",
                "instructions": instruction,
                "options": list(options),
            }
        }
        out = self._roundtrip(state, questions)
        ans = (out or {}).get("q")
        return ans if isinstance(ans, dict) else None

    def score(
        self,
        state: Mapping[str, Any],
        instruction: str,
        *,
        levels: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Score: 按自定义等级打分 → {"score","probabilities","confidence"}."""
        q: dict[str, Any] = {"type": "score", "instructions": instruction}
        if levels:
            q["levels"] = list(levels)
        out = self._roundtrip(state, {"q": q})
        ans = (out or {}).get("q")
        return ans if isinstance(ans, dict) else None

    # ── 内部 ─────────────────────────────────────────────────────
    def _roundtrip(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]] | None:
        """发一次请求, 解析 answers. 任何异常 → None (fail-open)."""
        if not self.available or not state or not questions:
            return None
        body: dict[str, Any] = {"state": dict(state), "questions": dict(questions)}
        if self._model:
            body["model"] = self._model
        try:
            payload = self._post(body)
        except Exception:  # noqa: BLE001 — 网络/超时/状态码异常一律 fail-open
            logger.debug("jev: request failed (fail-open)", exc_info=True)
            return None
        return self._parse_answers(payload)

    def _post(self, body: Mapping[str, Any]) -> Any:
        url = f"{self._base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._transport is not None:
            resp = self._transport(url=url, headers=headers, json=body, timeout=self._timeout)
            return resp.json()
        import httpx

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _parse_answers(payload: Any) -> dict[str, dict[str, Any]]:
        """容错解析 answers. 结构不符 → 空 dict (调用方按无结果处理)."""
        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for name, ans in answers.items():
            if not isinstance(ans, dict):
                continue
            p = ans.get("noul")
            entry: dict[str, Any] = {
                "confidence": ans.get("confidence"),
                "raw": ans,
            }
            try:
                entry["p"] = float(p) if p is not None else 0.0
            except (TypeError, ValueError):
                entry["p"] = 0.0
            # Choice / Score 原语字段透传 (供上层调用方扩展)
            if "choice" in ans:
                entry["choice"] = ans["choice"]
            if "score" in ans:
                entry["score"] = ans["score"]
            out[name] = entry
        return out
