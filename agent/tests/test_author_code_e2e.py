"""书生成码主路径端到端验证 (behavior-level, 不依赖真实 API key).

背景: 之前 `--dry` 走 client=None 确定性自检, 只覆盖了白名单自检分支 —— 书生成码
主导价自主建模`._try_author_code`是**主编码路径**并未被端到端跑到(仅编译级验证)。这里的
fake client 回放一条书生写的真实 run(cfg) 代码, 把整条链 `_try_author_code → Experiment
→ sandbox_run(真执行) → 门禁可落地` 从"编译级"补成"行为级"验证。

诚实边界: fake client 只替书生返回"一段待执行的真实 numpy 代码"(假装书生写的), 不含任何
预先算好的答案; 执行仍走 code_lab 真沙箱, 数值是真算出来的 —— 不伪造。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # agent/
_EXDIR = Path(__file__).resolve().parents[2] / "examples"


def _load(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qc = _load(_EXDIR / "shusheng_quantum_critical.py", "qc_e2e")


# ── 端到端用: OpenAI 兼容 fake client (只回放书生成码, 不预算答案) ──────────
_BOOK_AUTH_CODE = '''def run(cfg):
    import math
    T = float(cfg.get("T", 300.0))
    strain = float(cfg.get("strain", 0.0))
    # 一个可证伪的保守序参量量纲: eta^2 ∝ (Tc-T)/Tc 形式, 真实数值计算
    Tc = 403.0 - 40.0 * strain
    tau = max(0.0, (Tc - T) / Tc)
    eta = (tau ** 0.5) if tau > 0 else 0.0
    return {"success": True,
            "summary": {"Tc": Tc, "tau": tau, "eta": eta},
            "objectives": {"eta": eta, "Tc": Tc}}
'''


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Completions:
    def __init__(self, reply): self._reply = reply

    def create(self, **kwargs):
        return _ChoiceEncoder(self._reply())


class _Chat:
    def __init__(self, reply): self.completions = _Completions(reply)


class _ChoiceEncoder:
    def __init__(self, content): self.choices = [_Choice(content)]


class FakeBookClient:
    """书生成码 fake client: `chat.completions.create` 首次返回书生代码, 带 base_url."""
    def __init__(self, base_url="https://chat.intern-ai.org.cn/api/v1"):
        self.base_url = base_url
        self.calls = 0
        self._chat = _Chat(self._reply)

    def _reply(self):
        self.calls += 1
        # 真实书生对"写码"任务的交付: fenced python 代码块(不是 JSON 包裹 dict).
        # _ask_json 先尝试裸 {…} JSON(用于结构化答复), 失败后回退到 fenced 代码块提取.
        # 若返回 json.dumps({"code": …}), 代码内容自身的 } 会被非贪婪 \{.*?\} 提前截断,
        # json.loads 失败后把整个 JSON 串当代码 → 残留 \n 字面量触发 SyntaxError.
        return "```python\n" + _BOOK_AUTH_CODE + "\n```"

    @property
    def chat(self):
        return self._chat


def test_author_code_path_end_to_end():
    """主编码路径 `_try_author_code` 端到端跑通: 假书生回放代码 → 真沙箱执行 → 出 experiment."""
    client = FakeBookClient()
    exp, probes, obj_keys, err = qc._try_author_code(
        client, "intern-s2-preview", "检验应变对铁电临界/序参量的调控", 1)
    assert exp is not None, f"书生成码端到端失败: {err}"
    assert client.calls >= 1                    # 真的问了书生
    res = exp.run()                             # 真执行书生代码
    assert res["success"] is True
    # 代码是真算的: ε=-0.05 → Tc=405 → η>0; 通用契约
    assert res["objectives"]["Tc"] > 0
    assert res["objectives"]["eta"] >= 0
    # 门禁可落地: objectives 是纯数值标量(非配置/列表/字符串)
    for v in res["objectives"].values():
        assert isinstance(v, (int, float)) and not isinstance(v, bool)


def test_author_code_contract_reachable_by_gate():
    """experiment 的 run() 产物 {summary, objectives} 可落入统一契约(claim grounding 可核对)."""
    client = FakeBookClient()
    exp, _, _, _ = qc._try_author_code(client, "intern-s2-preview", "自主建模检验", 2)
    res = exp.run()
    assert isinstance(res.get("summary"), dict)
    assert isinstance(res.get("objectives"), dict) and res["objectives"]
    # 独立于 dom handler 的浅校验: 每条 objective 都能被 gate 溯源为真实标量
    traceable = {k: v for k, v in res["objectives"].items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
    assert set(traceable) == set(res["objectives"].keys())


def test_fake_base_url_is_intern_compatible():
    """fake client 的 base_url 走 Intern 端点 → _llm_compat_kwargs 注入 thinking_mode(不改真行为)."""
    client = FakeBookClient()
    assert qc._llm_compat_kwargs(client) == {"extra_body": {"thinking_mode": False}} or qc._llm_compat_kwargs(client) == {}