"""adaptive_call —— 模型无关的"遇错升级"LLM 调用适配器.

搭配 text_to_json(鲁棒提取), 解决不同模型对结构化输出遵循差异极大的问题
(书生 s1 ~10%, DeepSeek 100%). 关键设计: **不在代码里查模型名 / 不加 per-model
白名单**, 而是用一套对任意端点都成立的**试探-降级**策略:

  plan[0]: 携带可选 reasoning 扩展(nextra, 如 {"thinking_mode":True}) + 放宽 max_tokens
           若端点返回 4xx(4xx==该字段不支持/密钥错) → 不硬试, 直接跳到下一档
  plan[1]: 去掉 nextra + 追加硬护栏 prompt(只输出单个合法 JSON, 禁思考/围栏/前后文)
           → 通过"更严指令"逼模型收敛到干净 JSON
  每档: status 2xx 且 robust_extract 按 schema 解析成功 → 返回; 否则继续下一档

传输(HTTP 怎么发)由调用方注入 raw_call(body)->(status,text), 本模块只管
"该不该重试 / 换哪档 / 怎么判成功", 因此对任意 OpenAI 兼容端点通用, 且可单测.

Pure function, 零网络/零 numpy; 失败返回 None + 原因(供诊断), 不抛异常.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from huginn.metacog.text_to_json import robust_extract

# 硬护栏: 追加到 system 末尾, 逼迫"只输出一个合法 JSON"
_HARDEN = (
    "\nHARD RULES: Output ONLY a single well-formed JSON object, nothing else. "
    "No thinking, no reasoning, no markdown fences, no text before or after. "
    "The first character must be '{' and the last must be '}'."
)


def _mk_body(system: str, user: str, max_tokens: int,
             extra: dict | None) -> dict:
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    if extra:
        body.update(extra)
    return body


def robust_chat(
    raw_call: Callable[[dict], tuple[int, str]],
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    extract: Callable[[str, dict], dict | None] | None = None,
    base_max_tokens: int = 2048,
    nextra: dict | None = None,
    harden_text: str | None = None,
    max_attempts: int = 3,
) -> dict:
    """通用调用: 逐档试探, 任一档解析成功即返回; 全过则返回失败 + 原因.

    返回 dict: {"ok": bool, "extracted": dict|None, "status": int,
                "text": str(末档原文), "attempts": int}
    """
    extract = extract or robust_extract
    harden_text = harden_text or _HARDEN

    plans: list[dict] = []
    if nextra:
        plans.append({"extra": nextra, "harden": False, "mt": base_max_tokens * 2,
                      "sys": system})
    plans.append({"extra": None, "harden": True, "mt": base_max_tokens,
                  "sys": system + harden_text})
    # 若 nextra 成功但解析失败, 保留一次"去掉 nextra + 硬护栏"档(上面已含);
    # 若 user 上此外再把 nextra 重试会退化为 plan2, 故不额外重复.

    last_status: int = 0
    last_text: str = ""
    attempts = 0
    for p in plans:
        if attempts >= max_attempts:
            break
        body = _mk_body(p["sys"], user, p["mt"], p["extra"])
        try:
            status, text = raw_call(body)
        except Exception:  # noqa: BLE001 — 传输异常按 0(不可用) 处理, 跳到下档
            status, text = 0, ""
        last_status, last_text = status, text
        attempts += 1
        if not (200 <= status < 300):
            continue  # 4xx/5xx: 该档 payload 不可用 → 下档(4xx 字段不支持也在这被略过)
        got = extract(text, schema)
        if got:
            return {"ok": True, "extracted": got, "status": status,
                    "text": text, "attempts": attempts}
    # 全部失败: 尝试用宽松提取兜底看末档是否有可救结果(例如 200 但 schema 苛刻)
    if 200 <= last_status < 300 and last_text:
        got = extract(last_text, schema)  # robust_extract 已含宽松分支
        if got:
            return {"ok": True, "extracted": got, "status": last_status,
                    "text": last_text, "attempts": attempts}
    return {"ok": False, "extracted": None, "status": last_status,
            "text": last_text, "attempts": attempts}


# ── self-check(注入假 raw_call, 不烧真实模型) ────────────────────

# ── 桥接: 把 agent 现有的 langchain `model.invoke(messages)` 包装成 raw_call ──
# 这样 adaptive_call 不需知道是哪个端点/模型, 直接接现有传输.
# 关键降级点: 若 body 里带了 endpoint 不支持的额外参数(如 thinking_mode 对非推理
# 类模型), langchain invoke 抛 TypeError -> 我们返回 400 -> robust_chat 判定该档
# 不可用, 自动降到硬护栏档。无需 per-model 判断。


def langchain_raw_call(model, *, resp_to_text=None, build_messages=None):
    """把 langchain 模型包装成 raw_call(body)->(status,text).

    body: {"messages":[{"role":"system|user","content":...}], "max_tokens",
           ...其它 body 字段(extra 会原样透传 invoke)}
    """
    def _bm(s, u):
        if build_messages:
            return build_messages(s, u)
        from huginn.metacog.step_evaluator import _build_messages
        return _build_messages(s, u)

    def _rt(r):
        if resp_to_text:
            return resp_to_text(r)
        from huginn.metacog.step_evaluator import _resp_to_text
        return _resp_to_text(r)

    def raw_call(body):
        sys_c = user_c = ""
        for m in body.get("messages") or []:
            r = m.get("role")
            c = m.get("content", "") or ""
            if r == "system":
                sys_c = c
            elif r == "user":
                user_c = c
        extra = {k: v for k, v in body.items() if k not in ("messages", "max_tokens")}
        try:
            msgs = _bm(sys_c, user_c)
            resp = model.invoke(msgs, **extra) if extra else model.invoke(msgs)
            return 200, _rt(resp)
        except TypeError:
            # 额外参数不被支持(如 thinking 字段对非推理模型) -> 视为该档不可用
            return 400, "extra params unsupported"
        except Exception as e:  # noqa: BLE001 — 传输异常, 交 robust_chat 判定
            return 0, str(e)
    return raw_call


def robust_invoke(model, *, system, user, schema=None, extract=None,
                  base_max_tokens: int = 2048, nextra=None):
    """直接用现有 langchain 模型跑"遇错升级". 返回 robust_chat 的 dict."""
    return robust_chat(
        langchain_raw_call(model),
        system=system, user=user, schema=schema, extract=extract,
        base_max_tokens=base_max_tokens, nextra=nextra,
    )


def _selfcheck() -> None:
    schema = {"conductivity": float, "mobility": float}

    # 档1(带 nextra)就干净成功 —— 不触发降级
    def ok1(body):
        return 200, '{"conductivity": 1.2, "mobility": 80.0}'
    r = robust_chat(ok1, system="s", user="u", schema=schema, nextra={"thinking_mode": True})
    assert r["ok"] and r["extracted"]["conductivity"] == 1.2 and r["attempts"] == 1

    # 4xx 字段不支持(nextra) → 自动去掉降级到硬护栏档
    def ok_nextra_400(body):
        if "thinking_mode" in body:
            return 400, '{"error": "unknown param"}'
        return 200, '{"conductivity": 0.8, "mobility": 60.0}'
    r = robust_chat(ok_nextra_400, system="s", user="u", schema=schema,
                    nextra={"thinking_mode": True})
    assert r["ok"] and r["extracted"]["conductivity"] == 0.8 and r["attempts"] == 2

    # 无 nextra / 干净输出第一档即成功
    def ok_plain(body):
        return 200, '{"conductivity": 1.0, "mobility": 100.0}'
    r = robust_chat(ok_plain, system="s", user="u", schema=schema)
    assert r["ok"] and r["extracted"]["mobility"] == 100.0 and r["attempts"] == 1

    # 解析失败(档1散文) → 硬护栏档重试成功(两档来自 nextra 存在)
    def loose_first(body):
        if "thinking_mode" in body:
            return 200, "We think conductivity should be around one point two ..."
        return 200, '{"conductivity": 1.2, "mobility": 90.0}'
    r = robust_chat(loose_first, system="s", user="u", schema=schema,
                    nextra={"thinking_mode": True})
    assert r["ok"] and r["extracted"]["conductivity"] == 1.2 and r["attempts"] == 2

    # 全程失败 → ok=False, 不抛
    def always_fail(body):
        return 500, "server error"
    r = robust_chat(always_fail, system="s", user="u", schema=schema)
    assert not r["ok"] and r["status"] == 500

    # 200 但 内容完全无关 → 宽松也救不回 → ok=False
    def irrelevant(body):
        return 200, "hello world nothing here"
    r = robust_chat(irrelevant, system="s", user="u", schema=schema)
    assert not r["ok"]

    print("OK adaptive_call self-check passed (model-agnostic escalate-and-degrade)")


if __name__ == "__main__":
    _selfcheck()
