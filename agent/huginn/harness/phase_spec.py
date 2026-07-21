"""H4 试点: BUILTIN_SPECS 可演化.

把 subagent.py 的 BUILTIN_SPECS 4 个 SubagentSpec 抽成 PhaseRegistry,
用户可在 .huginn/phase_registry.json 写 override 覆盖 baseline.
SubagentDispatch.__init__ 在 toggle 开启时从 registry 取 specs.

试点只覆盖 BUILTIN_SPECS (spec H4 第 1 步, ~80 行). 7 个 phase 方法体
的 PhaseSpec 抽取留给 H4 后续分批 (spec H4 第 2-8 步). VALID_ACTIONS
改 property 留给 H4 第 9 步.

toggle: cfg.feature_flags.harness_phase_evolve (默认 off).

数学: PhaseSpec = {name, prompt_template, tool_whitelist, postcondition,
fallback}. SubagentSpec = PhaseSpec − postcondition − fallback (subagent.py:41-71),
是 PhaseSpec 的子集. 试点首选 BUILTIN_SPECS 因为 SubagentSpec 已是 dataclass,
register_spec() 实例级注册机制已有 (subagent.py:316), 零成本接入.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _harness_enabled(key: str, default: bool = False) -> bool:
    """读 cfg.feature_flags.<key>, mtime 自动 reload. 默认 off.

    跟 prompt_patch._harness_enabled 同实现, 不抽公共 util (两个文件各自独立).
    """
    try:
        from huginn.config import get_config
        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        return bool(ff.get(key, default))
    except Exception:
        return default


@dataclass
class PhaseSpec:
    """一个 phase / subagent 的可演化字段.

    SubagentSpec 是 PhaseSpec 的子集 (没 postcondition / fallback),
    试点首选 BUILTIN_SPECS, 这俩字段留空. 7 phase 接入时才用.
    """
    name: str
    prompt_template: str  # 对应 SubagentSpec.system_prompt
    tool_whitelist: list[str]  # 对应 SubagentSpec.allowed_tools
    postcondition: str = ""  # subagent 不用, phase 才用
    fallback: str = ""  # 同上
    max_tool_calls: int = 10
    max_iterations: int = 5
    summary_format: str = "free"
    description: str = ""


class PhaseRegistry:
    """单例. 存 .huginn/phase_registry.json 覆盖 baseline.

    baseline = 当前 BUILTIN_SPECS 镜像 (subagent.py:115-183). 不写盘,
    用户写 override 才落盘. 跨 iter 状态持久: 进程内单例 + 磁盘文件.
    跟 _get_evolution (engine.py:460) 同模式懒加载.

    ponytail: 不做拓扑排序检查 (BUILTIN_SPECS 试点无 phase 间依赖),
    升级路径: 7 phase 全接入后加 topo sort (spec H4 安全边界第 5 条).
    """
    _instance: "PhaseRegistry | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        cache_dir = Path(
            os.environ.get("HUGINN_CACHE_DIR", Path.home() / ".huginn")
        )
        self._reg_path = cache_dir / "phase_registry.json"
        self._overrides: dict[str, dict[str, Any]] = {}
        self._load_overrides()

    @classmethod
    def get_instance(cls) -> "PhaseRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_overrides(self) -> None:
        try:
            if self._reg_path.exists():
                d = json.loads(self._reg_path.read_text(encoding="utf-8"))
                if isinstance(d, dict) and isinstance(d.get("subagents"), dict):
                    self._overrides = d["subagents"]
        except Exception:
            logger.debug("phase_registry load fail", exc_info=True)

    def _save_overrides(self) -> None:
        try:
            self._reg_path.write_text(
                json.dumps(
                    {"subagents": self._overrides},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("phase_registry save fail", exc_info=True)

    def get_subagent_spec(self, name: str) -> Any:
        """从 baseline + override 合成 SubagentSpec. 不存在返回 None.

        返回类型是 SubagentSpec (subagent.py:42), 不写死避免循环 import.
        """
        from huginn.agents.subagent import SubagentDispatch, SubagentSpec
        baseline: SubagentSpec | None = SubagentDispatch.BUILTIN_SPECS.get(name)
        if baseline is None:
            return None
        ov = self._overrides.get(name)
        if not ov:
            return baseline
        return SubagentSpec(
            name=name,
            description=ov.get("description", baseline.description),
            system_prompt=ov.get("system_prompt", baseline.system_prompt),
            allowed_tools=ov.get("allowed_tools", baseline.allowed_tools),
            max_tool_calls=ov.get("max_tool_calls", baseline.max_tool_calls),
            max_iterations=ov.get("max_iterations", baseline.max_iterations),
            summarize_result=ov.get(
                "summarize_result", baseline.summarize_result
            ),
            summary_format=ov.get(
                "summary_format", baseline.summary_format
            ),
            max_depth=ov.get("max_depth", baseline.max_depth),
        )

    def register_subagent_override(
        self, name: str, spec_dict: dict[str, Any]
    ) -> None:
        """注册 / 更新一个 subagent spec override. 落盘."""
        with self._lock:
            self._overrides[name] = spec_dict
        self._save_overrides()

    def list_subagent_specs(self) -> list[str]:
        """列出所有可用 subagent spec 名 (baseline + override 合并)."""
        from huginn.agents.subagent import SubagentDispatch
        names = set(SubagentDispatch.BUILTIN_SPECS.keys())
        names.update(self._overrides.keys())
        return sorted(names)


def get_subagent_specs_for_dispatch() -> dict[str, Any] | None:
    """SubagentDispatch.__init__ 调: toggle 开启时返回 registry 合成的 specs,
    否则返回 None (caller 用 class attr baseline).

    返回 dict[name, SubagentSpec] 或 None. SubagentSpec 类型不写死避免循环 import.
    """
    if not _harness_enabled("harness_phase_evolve"):
        return None
    try:
        reg = PhaseRegistry.get_instance()
        from huginn.agents.subagent import SubagentDispatch
        return {
            name: reg.get_subagent_spec(name)
            for name in SubagentDispatch.BUILTIN_SPECS.keys()
        }
    except Exception:
        logger.debug("get_subagent_specs_for_dispatch fail", exc_info=True)
        return None


def _selfcheck() -> None:
    """H4 试点 selfcheck: PhaseRegistry 覆盖 + 持久化 + 未覆盖字段回退 baseline."""
    import shutil
    import tempfile

    from huginn.agents.subagent import SubagentDispatch

    import huginn.harness.phase_spec as ps

    # 1. toggle off: SubagentDispatch 用 class attr baseline
    d = SubagentDispatch()
    assert "explore" in d._specs
    assert d._specs["explore"].system_prompt.startswith(
        "You are an exploration agent"
    ), "baseline should be loaded"
    print("1. SubagentDispatch baseline load OK")

    # 2. toggle on + 注册 override + 持久化
    orig = ps._harness_enabled
    ps._harness_enabled = lambda key, default=False: (
        True if key == "harness_phase_evolve" else default
    )
    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    ps.PhaseRegistry._instance = None

    reg = ps.PhaseRegistry.get_instance()
    reg.register_subagent_override("explore", {
        "system_prompt": "PATCHED explore prompt",
        "max_tool_calls": 99,
    })
    # reset singleton 重读, 验证持久化
    ps.PhaseRegistry._instance = None
    reg2 = ps.PhaseRegistry.get_instance()
    spec = reg2.get_subagent_spec("explore")
    assert spec is not None
    assert spec.system_prompt == "PATCHED explore prompt", (
        f"override not applied: {spec.system_prompt}"
    )
    assert spec.max_tool_calls == 99, (
        f"override max_tool_calls not applied: {spec.max_tool_calls}"
    )
    # 未 override 的字段保留 baseline
    assert "file_read_tool" in spec.allowed_tools, (
        f"baseline allowed_tools lost: {spec.allowed_tools}"
    )
    print("2. PhaseRegistry override + persistence + baseline fallback OK")

    # 3. 不存在的 spec 名 → None
    assert reg2.get_subagent_spec("nonexistent") is None
    print("3. unknown subagent name → None OK")

    # 4. get_subagent_specs_for_dispatch (SubagentDispatch 接入点)
    specs = ps.get_subagent_specs_for_dispatch()
    assert specs is not None, "toggle on should return specs dict"
    assert set(specs.keys()) == set(SubagentDispatch.BUILTIN_SPECS.keys())
    assert specs["explore"].system_prompt == "PATCHED explore prompt"
    print("4. get_subagent_specs_for_dispatch OK")

    # 清理
    shutil.rmtree(tmp, ignore_errors=True)
    del os.environ["HUGINN_CACHE_DIR"]
    ps.PhaseRegistry._instance = None
    ps._harness_enabled = orig
    print("H4 phase_spec 试点 selfcheck OK (4/4)")


if __name__ == "__main__":
    _selfcheck()
