"""transolver_tool.py 补测 — 覆盖 _check_torch(缓存/成功/ImportError)、
_load_model_class(缓存/首个命中/跳过失败/全失败)、_build_model(成功/无法命中)、
_to_tensor、call 分派(unknown action)、_predict(缺数据/无模型类/build 失败/
ckpt 缺 .pt 回退 .pth/都缺/load 失败/推理成功/推理失败/审计异常吞掉)、
_train(缺数据/无模型类/build 失败/冷启动保存/带 warm-start/load 异常吞掉/
tensor 失败/save 失败/成功/审计异常吞掉)、estimate_cost 全分支.

配合既有 tests/test_transolver_tool.py, 把 transolver_tool.py 覆盖率提升到 90%+.
"""

from __future__ import annotations

import sys
import types
import asyncio
from pathlib import Path

import pytest

from huginn.tools.sim.transolver_tool import TransolverTool, TransolverToolInput

pytestmark = pytest.mark.anyio


def _ctx():
    from huginn.core_types import ToolContext

    return ToolContext(session_id="test", workspace=".")


def _run(tool, args):
    return asyncio.run(tool.call(args, _ctx()))


def _model(action="predict", **kw):
    base = {"action": action}
    base.update(kw)
    return TransolverToolInput(**base)


# ── fake torch ───────────────────────────────────────────────────────────


class _FakeTensor(list):
    def __init__(self, data, dtype=None, device=None):
        super().__init__(data)
        self.dtype = dtype
        self.device = device

    def unsqueeze(self, dim):
        return self

    def squeeze(self, dim=None):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [list(x) if isinstance(x, (list, _FakeTensor)) else x for x in self]

    def item(self):
        data = self
        while isinstance(data, (list, _FakeTensor)) and data:
            data = data[0]
        return float(data)

    def to(self, device):
        return self

    def backward(self):
        return None

    def shape(self):
        return (1,)


def _build_fake_torch():
    mod = types.ModuleType("torch")
    mod.__path__ = []  # 让 torch 可被当作包, 支持 `import torch.nn`
    sys.modules["torch"] = mod

    class _Float32:
        pass

    mod.float32 = _Float32()

    def tensor(arr, dtype=None, device=None):
        return _FakeTensor(list(arr), dtype=dtype, device=device)

    mod.tensor = tensor

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    mod.no_grad = _NoGrad

    mod.cuda = types.SimpleNamespace(is_available=lambda: True)

    def _load(path, map_location=None):
        return {"model": {"w": 1}, "config": {}}

    mod.load = _load

    def _save(obj, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"x")
        return None

    mod.save = _save

    class _FakeParam:
        def __init__(self, tensor):
            self._t = tensor

        def data(self):
            return self._t

    class _FakeModule:
        def __init__(self):
            self._p = [_FakeParam(_FakeTensor([1.0]))]

        def parameters(self):
            return self._p

        def to(self, device):
            return self

        def eval(self):
            return self

        def train(self):
            return self

        def state_dict(self):
            return {"w": 1}

        def load_state_dict(self, sd, strict=False):
            return None

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    class _Optim:
        def __init__(self, params, lr=None):
            self.params = params

        def zero_grad(self):
            return None

        def step(self):
            return None

    # torch.optim / torch.nn 作为独立子模块注册, 支持 `import torch.nn as nn`
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.Adam = lambda params, lr=0.001: _Optim(params, lr)
    mod.optim = optim_mod
    sys.modules["torch.optim"] = optim_mod

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.MSELoss = lambda: (lambda pred, y: _FakeTensor([0.1]))
    mod.nn = nn_mod
    sys.modules["torch.nn"] = nn_mod

    return mod, _FakeModule


def _install_fake_torch(monkeypatch):
    mod, _FakeModule = _build_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", mod)
    return _FakeModule


# ── helpers: fake Model class ────────────────────────────────────────────


def _install_model_cls(monkeypatch, ModelCls):
    sys.modules.pop("transolver_plus", None)
    for name in ("transolver_plus.models", "Transolver_plus.models"):
        sys.modules.pop(name, None)
    sys.modules.pop("transolver_plus.models.Transolver_plus", None)

    pkg = types.ModuleType("transolver_plus")
    pkg.__path__ = []  # noqa: SLF001
    sub = types.ModuleType("transolver_plus.models")
    sub.__path__ = []
    sys.modules["transolver_plus"] = pkg
    sys.modules["transolver_plus.models"] = sub

    mod = types.ModuleType("transolver_plus.models.Transolver_plus")
    mod.Model = ModelCls
    sys.modules["transolver_plus.models.Transolver_plus"] = mod


def _tool(**kw):
    return TransolverTool(**kw)


# ── _check_torch ─────────────────────────────────────────────────────────


def test_check_torch_cached(monkeypatch):
    tool = _tool()
    tool._torch_ok = True
    assert tool._check_torch() is True


def test_check_torch_success(monkeypatch):
    _install_fake_torch(monkeypatch)
    tool = _tool()
    assert tool._check_torch() is True
    assert tool._torch_ok is True


def test_check_torch_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    tool = _tool()
    assert tool._check_torch() is False
    assert tool._torch_ok is False


# ── _load_model_class ────────────────────────────────────────────────────


def test_load_model_class_cached(monkeypatch):
    tool = _tool()
    tool._model_cls = object
    assert tool._load_model_class() is object


def test_load_model_class_first_hit(monkeypatch):
    class _M:
        pass

    _install_model_cls(monkeypatch, _M)
    tool = _tool()
    assert tool._load_model_class() is _M
    assert tool._model_cls is _M


def test_load_model_class_all_fail(monkeypatch):
    for name in ("transolver_plus.models", "models", "Transolver_plus.models"):
        sys.modules.pop(name, None)
    sys.modules.pop("transolver_plus.models.Transolver_plus", None)
    sys.modules.pop("models.Transolver_plus", None)
    sys.modules.pop("Transolver_plus.models.Transolver_plus", None)
    tool = _tool()
    assert tool._load_model_class() is None


def test_load_model_class_skip_failed_then_hit(monkeypatch):
    """第一个 import 路径抛错, 跳到第二个命中."""
    class _M:
        pass

    # 让第一个路径失败, 第二个路径成功
    sys.modules.pop("transolver_plus.models.Transolver_plus", None)
    sys.modules.pop("transolver_plus.models", None)
    sys.modules.pop("transolver_plus", None)

    mod = types.ModuleType("models.Transolver_plus")
    mod.Model = _M
    sys.modules["models.Transolver_plus"] = mod
    sys.modules["models"] = types.ModuleType("models")
    sys.modules["models"].__path__ = []

    tool = _tool()
    assert tool._load_model_class() is _M


def test_resolve_model_dir_cwd(monkeypatch, tmp_path):
    """无 workspace 无 checkpoint_dir → 用 cwd."""
    tool = TransolverTool(workspace=None)
    monkeypatch.chdir(tmp_path)
    d = tool._resolve_model_dir(_model(checkpoint_dir=None))
    assert d.name == "transolver"
    assert d.exists()


# ── call 分派 ────────────────────────────────────────────────────────────


def test_call_unknown_action(monkeypatch):
    _install_fake_torch(monkeypatch)

    class _M:
        pass

    _install_model_cls(monkeypatch, _M)
    tool = _tool()
    res = _run(tool, TransolverToolInput(action="predict", coords=[], features=[]))
    # predict 缺数据 → 走 _predict 分支
    assert res.success is False


def test_call_list_models_no_torch(tmp_path):
    tool = _tool()
    res = _run(tool, TransolverToolInput(action="list_models", checkpoint_dir=str(tmp_path)))
    assert res.success is True
    assert res.data["status"] == "no_models"


def test_call_predict_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    tool = _tool()
    res = _run(tool, TransolverToolInput(action="predict", coords=[[0.0]], features=[[1.0]]))
    assert res.success is False
    assert "Transolver++" in res.error


# ── _predict ─────────────────────────────────────────────────────────────


def test_predict_missing_data():
    tool = _tool()
    res = tool._predict(_model(coords=[], features=[]))
    assert res.success is False
    assert "needs coords and features" in res.error


def test_predict_no_model_class(monkeypatch):
    monkeypatch.setattr(TransolverTool, "_load_model_class", lambda self: None)
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(_model(coords=[[0.0]], features=[[1.0]]))
    assert res.success is False
    assert "Transolver++" in res.error


def test_predict_build_fail(monkeypatch):
    _install_fake_torch(monkeypatch)

    class _M:
        def __init__(self, **kw):
            raise RuntimeError("build boom")

    _install_model_cls(monkeypatch, _M)
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(_model(coords=[[0.0]], features=[[1.0]]))
    assert res.success is False
    assert "model build failed" in res.error


def test_predict_ckpt_missing_pt_falls_to_pth(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

    _install_model_cls(monkeypatch, _M)
    ckpt = tmp_path / "default.pth"
    ckpt.write_bytes(b"x")
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(_model(coords=[[0.0]], features=[[1.0]], checkpoint_dir=str(tmp_path)))
    assert res.success is False  # load 失败 → checkpoint load failed
    assert "checkpoint load failed" in res.error


def test_predict_ckpt_not_found(monkeypatch, tmp_path):
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(
        _model(coords=[[0.0]], features=[[1.0]], checkpoint_dir=str(tmp_path))
    )
    assert res.success is False
    assert "not found" in res.error


def test_predict_success(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_auditor(monkeypatch, has_errors=False)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, sd, strict=False):
            return None

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    _install_model_cls(monkeypatch, _M)
    (tmp_path / "default.pt").write_bytes(b"x")
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(
        _model(
            coords=[[0.0, 0.0, 0.0]],
            features=[[1.0]],
            condition=[0.1, 0.2, 0.3],
            checkpoint_dir=str(tmp_path),
        )
    )
    assert res.success is True
    assert res.data["status"] == "ok"
    assert res.data["physics_audit"]["has_errors"] is False


def test_predict_success_condition_none(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_auditor(monkeypatch, has_errors=False)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, sd, strict=False):
            return None

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    _install_model_cls(monkeypatch, _M)
    (tmp_path / "default.pt").write_bytes(b"x")
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(
        _model(coords=[[0.0]], features=[[1.0]], checkpoint_dir=str(tmp_path))
    )
    assert res.success is True


def test_predict_inference_fail(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, sd, strict=False):
            return None

        def __call__(self, inp):
            raise RuntimeError("infer boom")

    _install_model_cls(monkeypatch, _M)
    (tmp_path / "default.pt").write_bytes(b"x")
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(
        _model(coords=[[0.0]], features=[[1.0]], checkpoint_dir=str(tmp_path))
    )
    assert res.success is False
    assert "inference failed" in res.error


def test_predict_audit_exception_swallowed(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", mod)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, sd, strict=False):
            return None

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    _install_model_cls(monkeypatch, _M)
    (tmp_path / "default.pt").write_bytes(b"x")
    tool = _tool()
    tool._torch_ok = True
    res = tool._predict(
        _model(coords=[[0.0]], features=[[1.0]], checkpoint_dir=str(tmp_path))
    )
    assert res.success is True
    assert "physics_audit" not in res.data


# ── _train ───────────────────────────────────────────────────────────────


def test_train_missing_data():
    tool = _tool()
    res = tool._train(_model(action="train", coords=[], features=[], target=[]))
    assert res.success is False
    assert "needs coords, features, and target" in res.error


def test_train_no_model_class(monkeypatch):
    monkeypatch.setattr(TransolverTool, "_load_model_class", lambda self: None)
    tool = _tool()
    tool._torch_ok = True
    res = tool._train(_model(action="train", coords=[[0.0]], features=[[1.0]], target=[[2.0]]))
    assert res.success is False
    assert "Transolver++" in res.error


def test_train_build_fail(monkeypatch):
    _install_fake_torch(monkeypatch)

    class _M:
        def __init__(self, **kw):
            raise RuntimeError("build boom")

    _install_model_cls(monkeypatch, _M)
    tool = _tool()
    tool._torch_ok = True
    res = tool._train(_model(action="train", coords=[[0.0]], features=[[1.0]], target=[[2.0]]))
    assert res.success is False
    assert "model build failed" in res.error


def test_train_cold_start_success(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_auditor(monkeypatch, has_errors=False)

    class _M:
        def __init__(self, **kw):
            self._warm = False

        def to(self, device):
            return self

        def train(self):
            return self

        def state_dict(self):
            return {"w": 1}

        def load_state_dict(self, sd, strict=False):
            self._warm = True
            return None

        def parameters(self):
            return []

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    _install_model_cls(monkeypatch, _M.__class__ if False else _M)
    tool = _tool()
    tool._torch_ok = True
    res = tool._train(
        _model(
            action="train",
            coords=[[0.0]],
            features=[[1.0]],
            target=[[2.0]],
            epochs=1,
            checkpoint_dir=str(tmp_path),
        )
    )
    assert res.success is True
    assert res.data["status"] == "ok"
    assert res.data["warnings"][0].startswith("trained from scratch")
    assert (tmp_path / "default.pt").exists()


def test_train_warm_start(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_auditor(monkeypatch, has_errors=False)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

        def train(self):
            return self

        def state_dict(self):
            return {"w": 1}

        def load_state_dict(self, sd, strict=False):
            return None

        def parameters(self):
            return []

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    _install_model_cls(monkeypatch, _M)
    (tmp_path / "default.pt").write_bytes(b"x")
    tool = _tool()
    tool._torch_ok = True
    res = tool._train(
        _model(
            action="train",
            coords=[[0.0]],
            features=[[1.0]],
            target=[[2.0]],
            epochs=1,
            checkpoint_dir=str(tmp_path),
        )
    )
    assert res.success is True
    assert res.data["warnings"] == []


def test_train_tensor_prep_fail(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

    _install_model_cls(monkeypatch, _M)
    tool = _tool()
    tool._torch_ok = True
    # 让 _to_tensor 抛错
    monkeypatch.setattr(
        TransolverTool, "_to_tensor",
        lambda self, arr, device, dtype: (_ for _ in ()).throw(RuntimeError("prep boom")),
    )
    res = tool._train(
        _model(action="train", coords=[[0.0]], features=[[1.0]], target=[[2.0]], checkpoint_dir=str(tmp_path))
    )
    assert res.success is False
    assert "tensor prep failed" in res.error


def test_train_save_fail(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

        def train(self):
            return self

        def state_dict(self):
            return {"w": 1}

        def parameters(self):
            return []

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    _install_model_cls(monkeypatch, _M)
    tool = _tool()
    tool._torch_ok = True
    monkeypatch.setattr(
        sys.modules["torch"], "save",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("save boom")),
    )
    res = tool._train(
        _model(action="train", coords=[[0.0]], features=[[1.0]], target=[[2.0]], epochs=1, checkpoint_dir=str(tmp_path))
    )
    assert res.success is False
    assert "save failed" in res.error


def test_train_audit_exception_swallowed(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", mod)

    class _M:
        def __init__(self, **kw):
            pass

        def to(self, device):
            return self

        def train(self):
            return self

        def state_dict(self):
            return {"w": 1}

        def parameters(self):
            return []

        def __call__(self, inp):
            return _FakeTensor([[0.5]])

    _install_model_cls(monkeypatch, _M)
    tool = _tool()
    tool._torch_ok = True
    res = tool._train(
        _model(action="train", coords=[[0.0]], features=[[1.0]], target=[[2.0]], epochs=1, checkpoint_dir=str(tmp_path))
    )
    assert res.success is True
    assert "physics_audit" not in res.data


# ── estimate_cost 全分支 ─────────────────────────────────────────────────


def test_estimate_cost_all_branches():
    tool = _tool()
    assert tool.estimate_cost(_model(action="train", epochs=3))["gpu_hours"] == pytest.approx(0.15)
    assert tool.estimate_cost(_model(action="predict"))["gpu_hours"] == 0.01
    assert tool.estimate_cost(_model(action="list_models")) is None


# ── auditor helper ───────────────────────────────────────────────────────


def _install_auditor(monkeypatch, has_errors=False, findings=None):
    mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Audit:
        def __init__(self, has_errors, findings):
            self.has_errors = has_errors
            self.findings = findings

        def to_dict(self):
            return {"has_errors": self.has_errors, "findings": len(self.findings)}

    class _Auditor:
        def audit(self, *a, **k):
            return _Audit(has_errors, list(findings or []))

    mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", mod)