"""测试专用: 把重依赖包装成命名空间包替身 (只给 `__path__`), 绕过其 `__init__`.

`test_prompt_contract` 与 `test_model_surface` 都要在不拉起 langchain/pydantic/rich
重栈的前提下 import `prompt_segments` / `prompt_builder`. 两份复制曾各自漂移 (一处
stub 了 `huginn.plugins`, 另一处没有, 于是后者 import 时会执行 `plugins/__init__` 的
重依赖), 故收成单一实现.

- 已加载的真实包不覆盖 (全量套件里其它测试可能已 import 真实包 —— 那正是想要的).
- 替身带 PEP 562 `__getattr__`: 一旦有人访问替身没有的属性 (如 `HuginnAgent`), 就当场
  升级为真实包, 避免同进程后续 import 拿到假包 (unknown location).
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types


def ensure_namespace_pkg(
    name: str,
    path: pathlib.PurePath,
    subpackages: tuple[str, ...] = (),
) -> None:
    """把 `name` 及其 `subpackages` 装成命名空间包替身 (幂等)."""
    _install(name, path)
    for sub in subpackages:
        _install(f"{name}.{sub}", path / sub)


def _install(name: str, path: pathlib.PurePath) -> None:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__path__", None):
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]

    def _upgrade(attr: str):
        if sys.modules.get(name) is pkg:
            del sys.modules[name]
        real = importlib.import_module(name)
        try:
            return getattr(real, attr)
        except AttributeError:
            sub = importlib.import_module(f"{name}.{attr}")
            setattr(real, attr, sub)
            return sub

    pkg.__getattr__ = _upgrade
    sys.modules[name] = pkg
