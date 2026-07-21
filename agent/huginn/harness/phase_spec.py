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
from dataclasses import dataclass, field
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

    H4 phase 分批加的字段 (dispatch_table / persona) 只有对应 phase 用,
    其他 phase 留空. 不抽 per-phase dataclass 避免 class 膨胀
    (ponytail: 统一 PhaseSpec 容纳所有 phase 字段, 升级路径: 字段超 15 个
    再拆 per-phase dataclass + registry 多表).
    """
    name: str
    prompt_template: str = ""  # 对应 SubagentSpec.system_prompt
    tool_whitelist: list[str] = field(default_factory=list)  # 对应 SubagentSpec.allowed_tools
    postcondition: str = ""  # subagent 不用, phase 才用
    fallback: str = ""  # 同上
    max_tool_calls: int = 10
    max_iterations: int = 5
    summary_format: str = "free"
    description: str = ""
    # H4 phase 分批: phase 特定可演化字段
    dispatch_table: dict[str, list[str]] = field(default_factory=dict)
    persona: str = ""
    # phase 特定配置 (env name, marker, 阈值等). 用 dict 避免 PhaseSpec 字段膨胀.
    # ponytail: 字段超 15 个再拆 per-phase dataclass. 升级路径见 class docstring.
    extra: dict[str, Any] = field(default_factory=dict)


# 7 phase baseline: 当前 hardcode 镜像. 分批接入, 先填 _execute + _report.
# ponytail: baseline 不写盘, 用户写 override 才落盘. 跟 subagent baseline 同模式.
_PHASE_BASELINE: dict[str, PhaseSpec] = {
    "_execute": PhaseSpec(
        name="_execute",
        description="mode → executor dispatch table",
        dispatch_table={
            "coder": ["_execute_coder", "description"],
            "workflow": ["_execute_workflow", "description"],
            "dynamic_workflow": ["_execute_dynamic_workflow", "plan"],
            "explore": ["_execute_explore", "description"],
            "skill": ["_execute_skill", "plan"],
            "visual_inspect": ["_execute_visual_inspect", "description"],
        },
    ),
    "_report": PhaseSpec(
        name="_report",
        description="scientific report generation",
        persona="tutor",
    ),
    "_hypothesize": PhaseSpec(
        name="_hypothesize",
        description="hypothesis generation from context",
        extra={
            "branch_incubator_env": "HUGINN_USE_BRANCH_INCUBATOR",
            "selected_marker": "SELECTED:",
        },
    ),
    "_plan": PhaseSpec(
        name="_plan",
        description="plan generation from hypothesis",
        extra={
            "reject_tokens": ["no", "n", "cancel", "reject", "decline", "stop", "abort"],
        },
    ),
    "_perceive": PhaseSpec(
        name="_perceive",
        description="workspace perception",
        extra={
            "signal_routes": [
                "perception_error",
                "perception_conflict",
                "perception_converged",
            ],
        },
    ),
    "_validate": PhaseSpec(
        name="_validate",
        description="execution result validation",
        extra={
            "reviewer_threshold": 0.5,
            "matworldbench_categories": ["structure", "thermo", "electronic"],
            "needs_retry_threshold": 0.5,
        },
    ),
    "_learn": PhaseSpec(
        name="_learn",
        description="Feynman learning + memory consolidation",
        extra={
            "importance_default": 0.6,
            "importance_max": 0.9,
        },
    ),
}


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
        self._phase_overrides: dict[str, dict[str, Any]] = {}
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
                if isinstance(d, dict):
                    if isinstance(d.get("subagents"), dict):
                        self._overrides = d["subagents"]
                    if isinstance(d.get("phases"), dict):
                        self._phase_overrides = d["phases"]
        except Exception:
            logger.debug("phase_registry load fail", exc_info=True)

    def _save_overrides(self) -> None:
        try:
            self._reg_path.write_text(
                json.dumps(
                    {
                        "subagents": self._overrides,
                        "phases": self._phase_overrides,
                    },
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

    # ── phase spec (H4 phase 分批) ──────────────────────────────

    def get_phase_spec(self, phase_name: str) -> PhaseSpec | None:
        """从 baseline + override 合成 PhaseSpec. 不存在返回 None."""
        baseline = _PHASE_BASELINE.get(phase_name)
        if baseline is None:
            return None
        ov = self._phase_overrides.get(phase_name)
        if not ov:
            return baseline
        # 合并 override: 只覆盖 override 里有的字段, 其余回退 baseline.
        # dispatch_table 做 key 级 merge (用户只覆盖想改的 mode, 其余保留 baseline),
        # extra 做 key 级 merge (用户只覆盖想改的 key), 其他字段整体替换.
        merged_dispatch = {**baseline.dispatch_table}
        if "dispatch_table" in ov:
            merged_dispatch.update(ov["dispatch_table"])
        merged_extra = {**baseline.extra}
        if "extra" in ov:
            merged_extra.update(ov["extra"])
        return PhaseSpec(
            name=phase_name,
            description=ov.get("description", baseline.description),
            prompt_template=ov.get("prompt_template", baseline.prompt_template),
            tool_whitelist=ov.get("tool_whitelist", baseline.tool_whitelist),
            postcondition=ov.get("postcondition", baseline.postcondition),
            fallback=ov.get("fallback", baseline.fallback),
            dispatch_table=merged_dispatch,
            persona=ov.get("persona", baseline.persona),
            extra=merged_extra,
        )

    def register_phase_override(
        self, phase_name: str, spec_dict: dict[str, Any]
    ) -> None:
        """注册 / 更新一个 phase spec override. 落盘."""
        with self._lock:
            self._phase_overrides[phase_name] = spec_dict
        self._save_overrides()

    def list_phase_specs(self) -> list[str]:
        """列出所有可用 phase spec 名 (baseline + override 合并)."""
        names = set(_PHASE_BASELINE.keys())
        names.update(self._phase_overrides.keys())
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


def get_phase_dispatch_table() -> dict[str, list[str]] | None:
    """engine._execute 调: toggle 开启时返回 _execute phase 的 dispatch_table,
    否则返回 None (caller 用 hardcode if/elif).

    返回 dict[mode, [method_name, arg_mode]] 或 None.
    arg_mode = "description" | "plan", 决定 executor 传 plan 还是 description.
    """
    if not _harness_enabled("harness_phase_evolve"):
        return None
    try:
        spec = PhaseRegistry.get_instance().get_phase_spec("_execute")
        return spec.dispatch_table if spec else None
    except Exception:
        logger.debug("get_phase_dispatch_table fail", exc_info=True)
        return None


def get_phase_persona(phase_name: str) -> str | None:
    """engine._report 等调: toggle 开启时返回 phase 的 persona,
    否则返回 None (caller 用 hardcode).

    返回 persona name str 或 None.
    """
    if not _harness_enabled("harness_phase_evolve"):
        return None
    try:
        spec = PhaseRegistry.get_instance().get_phase_spec(phase_name)
        return spec.persona if spec and spec.persona else None
    except Exception:
        logger.debug("get_phase_persona fail", exc_info=True)
        return None


def get_phase_extra(phase_name: str, key: str, default: Any = None) -> Any:
    """engine phase 方法调: toggle 开启时返回 phase extra 里的 key 值,
    否则返回 default (caller 用 hardcode).

    用于 _hypothesize (env name, marker), _plan (confirm/reject tokens) 等
    phase 特定配置. extra 做 key 级 merge, 用户只覆盖想改的 key.
    """
    if not _harness_enabled("harness_phase_evolve"):
        return default
    try:
        spec = PhaseRegistry.get_instance().get_phase_spec(phase_name)
        if spec and spec.extra:
            return spec.extra.get(key, default)
        return default
    except Exception:
        logger.debug("get_phase_extra fail", exc_info=True)
        return default


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

    # 5. phase spec baseline: _execute dispatch_table + _report persona
    reg3 = ps.PhaseRegistry.get_instance()
    exec_spec = reg3.get_phase_spec("_execute")
    assert exec_spec is not None, "_execute baseline should exist"
    assert "coder" in exec_spec.dispatch_table, "coder mode missing"
    assert exec_spec.dispatch_table["coder"] == ["_execute_coder", "description"], (
        f"coder dispatch entry wrong: {exec_spec.dispatch_table['coder']}"
    )
    assert exec_spec.dispatch_table["dynamic_workflow"] == ["_execute_dynamic_workflow", "plan"], (
        f"dynamic_workflow dispatch entry wrong: {exec_spec.dispatch_table['dynamic_workflow']}"
    )
    report_spec = reg3.get_phase_spec("_report")
    assert report_spec is not None, "_report baseline should exist"
    assert report_spec.persona == "tutor", f"_report persona wrong: {report_spec.persona}"
    print("5. phase spec baseline (_execute dispatch_table + _report persona) OK")

    # 6. phase spec override + 持久化
    reg3.register_phase_override("_execute", {
        "dispatch_table": {
            "coder": ["_execute_coder", "description"],
            "custom_mode": ["_execute_explore", "description"],
        },
    })
    ps.PhaseRegistry._instance = None
    reg4 = ps.PhaseRegistry.get_instance()
    exec_spec2 = reg4.get_phase_spec("_execute")
    assert "custom_mode" in exec_spec2.dispatch_table, "override custom_mode missing"
    # 未 override 的 mode 保留 baseline
    assert "workflow" in exec_spec2.dispatch_table, "baseline workflow lost after override"
    assert exec_spec2.dispatch_table["custom_mode"] == ["_execute_explore", "description"]
    print("6. phase spec override + persistence + baseline fallback OK")

    # 7. get_phase_dispatch_table helper (engine._execute 接入点)
    dt = ps.get_phase_dispatch_table()
    assert dt is not None, "toggle on should return dispatch_table"
    assert "custom_mode" in dt, "custom_mode missing from helper"
    assert dt["coder"] == ["_execute_coder", "description"]
    print("7. get_phase_dispatch_table helper OK")

    # 8. get_phase_persona helper (engine._report 接入点)
    # override _report persona
    reg4.register_phase_override("_report", {"persona": "reviewer"})
    ps.PhaseRegistry._instance = None
    persona = ps.get_phase_persona("_report")
    assert persona == "reviewer", f"persona override failed: {persona}"
    # 不存在的 phase → None
    assert ps.get_phase_persona("nonexistent") is None
    print("8. get_phase_persona helper OK")

    # 9. toggle off: helpers 返回 None (caller 用 hardcode)
    ps._harness_enabled = lambda key, default=False: False
    assert ps.get_phase_dispatch_table() is None, "toggle off should return None"
    assert ps.get_phase_persona("_report") is None, "toggle off should return None"
    assert ps.get_phase_extra("_hypothesize", "selected_marker", "X") == "X", (
        "toggle off get_phase_extra should return default"
    )
    print("9. toggle off helpers → None OK")

    # 10. 全 7 phase baseline 存在 + extra 字段正确
    ps._harness_enabled = lambda key, default=False: (
        True if key == "harness_phase_evolve" else default
    )
    ps.PhaseRegistry._instance = None
    reg5 = ps.PhaseRegistry.get_instance()
    expected_phases = {
        "_execute", "_report", "_hypothesize", "_plan", "_perceive",
        "_validate", "_learn",
    }
    actual_phases = set(reg5.list_phase_specs())
    assert expected_phases <= actual_phases, (
        f"missing phases: {expected_phases - actual_phases}"
    )
    # _hypothesize extra
    hyp_spec = reg5.get_phase_spec("_hypothesize")
    assert hyp_spec.extra.get("selected_marker") == "SELECTED:"
    assert hyp_spec.extra.get("branch_incubator_env") == "HUGINN_USE_BRANCH_INCUBATOR"
    # _plan extra
    plan_spec = reg5.get_phase_spec("_plan")
    assert "no" in plan_spec.extra.get("reject_tokens", [])
    assert "abort" in plan_spec.extra.get("reject_tokens", [])
    # _validate extra
    val_spec = reg5.get_phase_spec("_validate")
    assert val_spec.extra.get("reviewer_threshold") == 0.5
    assert "structure" in val_spec.extra.get("matworldbench_categories", [])
    # _learn extra
    learn_spec = reg5.get_phase_spec("_learn")
    assert learn_spec.extra.get("importance_default") == 0.6
    assert learn_spec.extra.get("importance_max") == 0.9
    print("10. 全 7 phase baseline + extra 字段 OK")

    # 11. get_phase_extra helper + override (key 级 merge)
    reg5.register_phase_override("_hypothesize", {
        "extra": {"selected_marker": "CHOSEN:"},
    })
    ps.PhaseRegistry._instance = None
    # override 的 key 生效
    assert ps.get_phase_extra("_hypothesize", "selected_marker") == "CHOSEN:"
    # 未 override 的 key 保留 baseline
    assert ps.get_phase_extra("_hypothesize", "branch_incubator_env") == "HUGINN_USE_BRANCH_INCUBATOR"
    print("11. get_phase_extra helper + extra key 级 merge OK")

    # 12. _validate extra override (reviewer_threshold)
    reg5.register_phase_override("_validate", {
        "extra": {"reviewer_threshold": 0.7},
    })
    ps.PhaseRegistry._instance = None
    assert ps.get_phase_extra("_validate", "reviewer_threshold") == 0.7
    # 未 override 的 matworldbench_categories 保留 baseline
    cats = ps.get_phase_extra("_validate", "matworldbench_categories", [])
    assert "structure" in cats, f"baseline categories lost: {cats}"
    print("12. _validate extra override OK")

    # 清理
    shutil.rmtree(tmp, ignore_errors=True)
    del os.environ["HUGINN_CACHE_DIR"]
    ps.PhaseRegistry._instance = None
    ps._harness_enabled = orig
    print("H4 phase_spec selfcheck OK (12/12)")


if __name__ == "__main__":
    _selfcheck()
