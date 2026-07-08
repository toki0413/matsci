"""压缩感知智能预取 — 根据溯源记录预测下一步需要的文件, 跨压缩保留关键状态.

对话压缩后 agent 不知道 "接下来该读哪个文件". 这里从 provenance 最近活动推断
管线阶段 (relax -> static -> band/dos -> analysis), 把可能用到的文件路径和
key_properties 预先塞进上下文, 让压缩后的 agent 仍能接着干.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from huginn.provenance.registry import ProvenanceEntry, ProvenanceRegistry

logger = logging.getLogger(__name__)

# 管线阶段顺序, 用来判断当前进度 (idx+1)/total
_PIPELINE_STAGES = ["structure", "relax", "static", "band", "dos", "analysis"]


def _entries(registry: ProvenanceRegistry) -> list[ProvenanceEntry]:
    # registry 没公开迭代接口, 直接拿内部列表
    return list(getattr(registry, "_entries", []))


def _sibling(path: str, name: str) -> str:
    """取 path 同目录下的 name, path 为空时只返回 name."""
    if not path:
        return name
    return str(Path(path).parent / name)


class SmartPrefetcher:
    """根据 provenance 最近活动预测接下来可能需要的文件."""

    def __init__(self, registry: ProvenanceRegistry) -> None:
        self.registry = registry

    def predict_needed_files(self) -> list[str]:
        """预测下一步可能需要的文件路径.

        规则:
          a. 最近产出的 OUTCAR/trajectory 等很可能被分析
          b. 最近做了 relax, 下一步可能要优化后的 POSCAR (CONTCAR)
          c. 最近做了 static/scf, 下一步可能要 band/dos 的输入结构
          d. 离线 artifact (.txt 卸载文件) 预取
        """
        entries = _entries(self.registry)
        if not entries:
            return []

        predicted: list[str] = []
        seen: set[str] = set()
        recent = entries[-15:]  # 看最近 15 条就够了, 太老的没参考价值

        def _add(p: str) -> None:
            if p and p not in seen:
                predicted.append(p)
                seen.add(p)

        # a. 最近产出的文件 (OUTCAR/trajectory 等) 很可能被分析
        for e in recent:
            fmt = (e.file_format or "").lower()
            fname = Path(e.file_path).name.lower() if e.file_path else ""
            if (
                fmt in ("outcar", "trajectory", "traj", "log")
                or "outcar" in fname
                or "trajectory" in fname
            ):
                _add(e.file_path)

        # 最近一条的输出文件, 大概率下一步要读
        _add(entries[-1].file_path)

        # b. 最近做了 relax, 下一步可能需要优化后的 POSCAR (CONTCAR)
        for e in recent:
            action = str(e.parameters.get("action", "")).lower()
            if "relax" in action:
                _add(_sibling(e.file_path, "CONTCAR"))
                _add(_sibling(e.file_path, "POSCAR"))
                break  # 一次 relax 的预测就够

        # c. 最近做了 static/scf, 下一步可能需要 band/dos 输入结构
        for e in recent:
            action = str(e.parameters.get("action", "")).lower()
            if "static" in action or "scf" in action:
                _add(_sibling(e.file_path, "CONTCAR"))
                _add(_sibling(e.file_path, "POSCAR"))
                break

        # d. 离线 artifact (磁盘卸载的 .txt) 预取关键值
        for e in recent:
            if e.file_path and e.file_path.endswith(".txt"):
                _add(e.file_path)

        return predicted[:20]  # 上限 20 个, 避免上下文膨胀

    def prefetch_to_context(self) -> str:
        """生成上下文块: 预测文件路径 + 从 provenance 提取的 key_properties + 提示."""
        needed = self.predict_needed_files()
        if not needed:
            return ""

        lines: list[str] = ["### Predicted files (smart prefetch):"]
        lines.append("你可能需要这些文件:")
        for f in needed:
            props = ""
            entry = self.registry.find_by_path(f)
            if entry and entry.key_properties:
                props = " | " + ", ".join(
                    f"{k}={v}" for k, v in list(entry.key_properties.items())[:4]
                )
            lines.append(f"  - {f}{props}")

        return "\n".join(lines)

    def detect_pipeline_stage(self) -> tuple[str, int]:
        """检测当前管线阶段和进度, 返回 (stage_name, stage_index).

        按最近 20 条记录里出现过的最高阶段判断, idx 对应 _PIPELINE_STAGES 下标.
        没有记录时返回 ("idle", -1).
        """
        entries = _entries(self.registry)
        if not entries:
            return ("idle", -1)

        latest_stage = "structure"
        for e in entries[-20:]:
            action = str(e.parameters.get("action", "")).lower()
            tool = (e.produced_by or "").lower()
            fmt = (e.file_format or "").lower()
            fname = Path(e.file_path).name.lower() if e.file_path else ""

            if "relax" in action or "relax" in tool:
                latest_stage = "relax"
            elif "static" in action or "scf" in action:
                latest_stage = "static"
            elif "band" in action or "band" in tool or "band" in fname:
                latest_stage = "band"
            elif "dos" in action or "dos" in tool or "dos" in fname:
                latest_stage = "dos"
            elif any(k in tool for k in ("analysis", "analyze", "plot", "extract")):
                latest_stage = "analysis"

        idx = (
            _PIPELINE_STAGES.index(latest_stage)
            if latest_stage in _PIPELINE_STAGES
            else 0
        )
        return (latest_stage, idx)


def enhance_compact_attachments(
    messages: list[Any], registry: ProvenanceRegistry
) -> str:
    """在 _extract_compact_attachments 基础上追加 provenance + 预取信息.

    增加:
      - 从 provenance 获取所有活跃文件的 key_properties (不依赖消息内容)
      - 预测下一步需要的文件
      - 当前管线阶段和进度
    返回 base + 追加内容. registry 为空时只返回 base.
    """
    # 延迟导入, 避免 context <-> smart_prefetch 循环引用
    from huginn.utils.context import _extract_compact_attachments

    base = _extract_compact_attachments(messages)
    lines: list[str] = [base] if base else []

    entries = _entries(registry)
    if not entries:
        return "\n".join(lines)

    # 1. 从 provenance 获取所有活跃文件的 key_properties (不依赖消息内容)
    prov_lines: list[str] = ["### Provenance key properties:"]
    for e in entries[-15:]:
        if e.key_properties:
            kv = ", ".join(f"{k}={v}" for k, v in e.key_properties.items())
            fname = Path(e.file_path).name if e.file_path else "?"
            prov_lines.append(f"  - {fname} ({e.produced_by}): {kv}")
    if len(prov_lines) > 1:
        lines.append("\n".join(prov_lines))

    # 2. 预测下一步需要的文件
    prefetcher = SmartPrefetcher(registry)
    prefetch_block = prefetcher.prefetch_to_context()
    if prefetch_block:
        lines.append(prefetch_block)

    # 3. 当前管线阶段和进度
    stage, idx = prefetcher.detect_pipeline_stage()
    total = len(_PIPELINE_STAGES)
    progress = f"{idx + 1}/{total}" if idx >= 0 else f"0/{total}"
    lines.append(f"### Pipeline stage: {stage} ({progress})")

    return "\n".join(lines)


if __name__ == "__main__":
    # 自检: 造假 registry, 验证预测/阶段/增强附件
    reg = ProvenanceRegistry()
    reg.register(
        file_path="/work/Si.poscar",
        produced_by="structure_tool",
        input_files=[],
        parameters={"action": "load"},
        file_format="poscar",
        key_properties={"formula": "Si"},
    )
    reg.register(
        file_path="/work/OUTCAR",
        produced_by="vasp_tool",
        input_files=["/work/Si.poscar"],
        parameters={"action": "relax", "encut": 520},
        file_format="outcar",
        key_properties={"energy": -10.5, "converged": True},
    )

    pf = SmartPrefetcher(reg)

    # relax 后应预测到 CONTCAR/POSCAR
    needed = pf.predict_needed_files()
    assert "/work/OUTCAR" in needed, "应预取最近产出的 OUTCAR"
    assert any("CONTCAR" in p for p in needed), "relax 后应预测 CONTCAR"

    ctx = pf.prefetch_to_context()
    assert "Predicted files" in ctx
    assert "你可能需要这些文件" in ctx
    # OUTCAR 有 key_properties, 应附带 energy
    assert "energy=-10.5" in ctx

    stage, idx = pf.detect_pipeline_stage()
    assert stage == "relax", f"应为 relax 阶段, 实际 {stage}"
    assert idx == 1

    # enhance_compact_attachments: base + provenance + prefetch + stage
    enhanced = enhance_compact_attachments([], reg)
    assert "Provenance key properties" in enhanced
    assert "energy=-10.5" in enhanced
    assert "Pipeline stage" in enhanced
    assert "relax" in enhanced

    # 空 registry 时 enhance 应回退为 base (空消息 -> 空字符串)
    empty = ProvenanceRegistry()
    assert enhance_compact_attachments([], empty) == ""

    print("smart_prefetch self-check OK")
