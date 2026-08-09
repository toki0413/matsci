"""huginn.causal 子包的 smoke import 测试.

覆盖 huginn/causal/ 下所有 .py 模块 (除 __init__.py):
  - visual_scm / predict_intervention / llm_generate_scm /
    visual_causal_chain / counterfactual_render
  - 每个模块一个 test_import_<module_name>, 验证可正常 import.
  - 对有明确 class/function 的模块, 额外补轻量实例化/调用测试.

依赖说明:
  - predict_intervention 直接 import pydantic; visual_causal_chain /
    counterfactual_render / llm_generate_scm 直接 import scipy.
  - visual_scm 自身不依赖 pydantic/scipy, 但 huginn.causal 包的 __init__
    会 import 上述兄弟模块, 因此导入 huginn.causal.* 任一子模块都会触发
    pydantic + scipy. 任一缺失则整个子包不可用, 用 _require_causal_deps()
    统一 importorskip 跳过.
  - 本测试环境 (pytest 用的 Python 3.14) 缺 scipy, 因此本文件全部 skip,
    在装齐依赖的环境里会全部运行.

只读不写源码, 不修改 huginn/ 下任何文件.
"""

from __future__ import annotations

import pytest


def _require_causal_deps() -> None:
    """统一跳过: huginn.causal 子包需要 pydantic + scipy 才能正常 import.

    子包 __init__ 依次导入 visual_scm (纯 stdlib)、predict_intervention
    (pydantic)、visual_causal_chain / counterfactual_render (scipy), 任一
    缺失都会让 `import huginn.causal.<任意子模块>` 失败.
    """
    pytest.importorskip("pydantic")
    pytest.importorskip("scipy")


# ── 1. counterfactual_render ───────────────────────────────────


def test_import_counterfactual_render():
    _require_causal_deps()
    import huginn.causal.counterfactual_render  # noqa: F401


# ── 2. llm_generate_scm ────────────────────────────────────────


def test_import_llm_generate_scm():
    _require_causal_deps()
    import huginn.causal.llm_generate_scm  # noqa: F401


# ── 3. predict_intervention ────────────────────────────────────


def test_import_predict_intervention():
    _require_causal_deps()
    import huginn.causal.predict_intervention  # noqa: F401


# ── 4. visual_causal_chain ─────────────────────────────────────


def test_import_visual_causal_chain():
    _require_causal_deps()
    import huginn.causal.visual_causal_chain  # noqa: F401


# ── 5. visual_scm ──────────────────────────────────────────────


def test_import_visual_scm():
    # visual_scm 自身不需 pydantic/scipy, 但包 __init__ 需要, 故同样跳过
    _require_causal_deps()
    import huginn.causal.visual_scm  # noqa: F401


def test_visual_scm_variable_edge():
    # Variable / Edge 是 dataclass, 能用必填字段正常构造
    _require_causal_deps()
    from huginn.causal.visual_scm import Edge, Variable

    v = Variable(name="T", type="condition", unit="K")
    assert v.name == "T"
    assert v.unit == "K"
    e = Edge(cause="T", effect="particle_size", mechanism="arrhenius")
    assert e.cause == "T"
    assert e.effect == "particle_size"


def test_visual_scm_list_templates():
    # 内置 4 个领域模板: sintering / ostwald_ripening / diffusion / phase_transition
    _require_causal_deps()
    from huginn.causal.visual_scm import get_template, list_templates

    names = list_templates()
    assert isinstance(names, list)
    assert "sintering" in names
    tpl = get_template("sintering")
    assert tpl is not None
    assert tpl.name == "sintering"
