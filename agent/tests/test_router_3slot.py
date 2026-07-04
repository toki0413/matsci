"""3-slot router 的测试.

Moonshine 三槽: main / verification / archival. 验证独立验证模型的回退、
优先选择, 以及 from_env 能正确注册带 verification 标签的模型.
用 mock 模型, 不碰真实 API.
"""

from __future__ import annotations

import pytest

from huginn.models.router import ModelRouter


class _FakeModel:
    """简单 mock, 只需要一个 name 属性方便断言."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"_FakeModel({self.name})"


# ── verification 槽 ────────────────────────────────────────


def test_select_verification_fallback() -> None:
    # 没注册 verification 模型, 应该退回 default
    router = ModelRouter()
    router.register("default", _FakeModel("default"), tags={"default"})
    model = router.select_verification()
    assert model.name == "default"


def test_select_verification_dedicated() -> None:
    # 注册了 verification 模型, 应该优先选它
    router = ModelRouter()
    router.register("default", _FakeModel("default"), tags={"default"})
    router.register("verifier", _FakeModel("verifier"), tags={"verification"})
    model = router.select_verification()
    assert model.name == "verifier"


# ── archival 槽 ───────────────────────────────────────────


def test_select_archival_fallback() -> None:
    # 没注册 archival 模型, 退回 default
    router = ModelRouter()
    router.register("default", _FakeModel("default"), tags={"default"})
    model = router.select_archival()
    assert model.name == "default"


def test_select_archival_dedicated() -> None:
    # 注册了 archival 模型, 优先选它
    router = ModelRouter()
    router.register("default", _FakeModel("default"), tags={"default"})
    router.register("archiver", _FakeModel("archiver"), tags={"archival"})
    model = router.select_archival()
    assert model.name == "archiver"


# ── has_dedicated_verification ─────────────────────────────


def test_has_dedicated_verification_false() -> None:
    router = ModelRouter()
    router.register("default", _FakeModel("default"), tags={"default"})
    assert router.has_dedicated_verification() is False


def test_has_dedicated_verification_true() -> None:
    router = ModelRouter()
    router.register("default", _FakeModel("default"), tags={"default"})
    router.register("verifier", _FakeModel("verifier"), tags={"verification"})
    assert router.has_dedicated_verification() is True


# ── from_env 注册 verification ─────────────────────────────


def test_from_env_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    # mock 掉 create_langchain_model, 避免真的去连 provider
    from huginn.models import router as router_mod

    def _fake_create(*args, **kwargs):
        return _FakeModel("env-verifier")

    monkeypatch.setattr(router_mod, "create_langchain_model", _fake_create)
    monkeypatch.setenv("HUGINN_MODEL_VERIFICATION", "openai:gpt-4o")
    router = ModelRouter.from_env()
    assert "verification" in router.list_models()
    assert router.has_dedicated_verification() is True
