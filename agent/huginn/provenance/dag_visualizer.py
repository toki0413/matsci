"""科学工作流 DAG 可视化 — 把 ProvenanceRegistry 渲染成 Mermaid 图.

对话被压缩后 agent 容易丢 "我跑到哪了" 的全局视图, 一张 DAG 图能快速重建.
三个粒度:
  - visualize_dag: 完整流程图, 节点=工具调用, 边=文件依赖, 收敛状态上色
  - visualize_timeline: 按时间排列的时间线
  - to_mermaid_for_context: 精简版, 只留最近 10 个节点, 塞进 agent 上下文
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from huginn.provenance.registry import ProvenanceEntry, ProvenanceRegistry

logger = logging.getLogger(__name__)


def _entries(registry: ProvenanceRegistry) -> list[ProvenanceEntry]:
    # registry 没公开迭代接口, 直接拿内部列表 (同项目内部, 够用)
    return list(getattr(registry, "_entries", []))


def _esc(label: str) -> str:
    """转义 Mermaid 节点标签里的特殊字符.

    双引号包裹的标签里只需要处理双引号和反斜杠, 其余字符 (等号/括号) 都安全.
    """
    return label.replace("\\", "\\\\").replace('"', "&quot;")


def _node_label(entry: ProvenanceEntry) -> str:
    """工具名 + 输出文件名 + 关键参数/属性摘要, 用 <br/> 换行."""
    parts: list[str] = [entry.produced_by or "unknown"]
    fname = Path(entry.file_path).name if entry.file_path else "?"
    parts.append(fname)
    action = entry.parameters.get("action")
    if action:
        parts.append(str(action))
    # 挑几个最常用的参数, 标签太长 Mermaid 渲染会挤
    for pk in ("encut", "kpoints", "functional", "ediff"):
        if pk in entry.parameters:
            parts.append(f"{pk}={entry.parameters[pk]}")
    # 关键物理量, 让 agent 一眼看到能量/带隙/收敛
    for kk in ("energy", "band_gap", "converged", "forces_max"):
        if kk in entry.key_properties:
            parts.append(f"{kk}={entry.key_properties[kk]}")
    return _esc("<br/>".join(str(p) for p in parts))


def _converged(entry: ProvenanceEntry) -> bool | None:
    """从 key_properties 读收敛状态, 拿不到返回 None (无法判断)."""
    val = entry.key_properties.get("converged")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    return None


def visualize_dag(registry: ProvenanceRegistry, max_nodes: int = 30) -> str:
    """从 ProvenanceRegistry 生成 Mermaid 流程图.

    节点 = 工具调用 (工具名 + 参数摘要), 边 = 文件依赖 (input_files -> output_files).
    收敛/未收敛用 style class 分别上色 (绿/红). 节点超过 max_nodes 只取最近的.
    """
    entries = _entries(registry)
    if not entries:
        return 'graph TD\n  empty["(no provenance entries)"]'

    # 只取最近 max_nodes 条, 图太大渲染不动也没法看
    if len(entries) > max_nodes:
        entries = entries[-max_nodes:]

    # 路径 -> 节点 id, 只在本次渲染范围里连边
    path_to_id: dict[str, str] = {}
    for i, e in enumerate(entries):
        path_to_id[e.file_path] = f"N{i}"

    lines: list[str] = ["graph TD"]
    converged_ids: list[str] = []
    unconverged_ids: list[str] = []
    source_counter = 0

    for i, e in enumerate(entries):
        nid = f"N{i}"
        label = _node_label(e)
        lines.append(f'  {nid}["{label}"]')
        conv = _converged(e)
        if conv is True:
            converged_ids.append(nid)
        elif conv is False:
            unconverged_ids.append(nid)

        # 输入文件 -> 当前节点; 没注册的源文件单独画一个节点
        for inp in e.input_files:
            if inp in path_to_id:
                lines.append(f"  {path_to_id[inp]} --> {nid}")
            else:
                sid = f"S{source_counter}"
                source_counter += 1
                sname = Path(inp).name if inp else "?"
                lines.append(f'  {sid}["{_esc(sname)}"]')
                lines.append(f"  {sid} --> {nid}")

    # 收敛状态样式: 绿=收敛, 红=未收敛
    if converged_ids:
        lines.append("  classDef conv fill:#c8e6c9,stroke:#388e3c")
        lines.append(f"  class {','.join(converged_ids)} conv")
    if unconverged_ids:
        lines.append("  classDef noconv fill:#ffcdd2,stroke:#d32f2f")
        lines.append(f"  class {','.join(unconverged_ids)} noconv")

    return "\n".join(lines)


def visualize_timeline(registry: ProvenanceRegistry) -> str:
    """生成时间线视图, 按时间排列所有计算.

    用 Mermaid timeline 语法, 按日期分 section, 每条计算一个事件.
    时间字段用 HHhMM 避免冒号 (Mermaid timeline 用 : 做分隔符).
    """
    entries = _entries(registry)
    if not entries:
        return "timeline\n    title (no entries)"

    sorted_entries = sorted(entries, key=lambda e: e.produced_at)

    lines: list[str] = ["timeline", "    title 科学计算工作流时间线"]
    current_date = ""
    for e in sorted_entries:
        dt = datetime.fromtimestamp(e.produced_at)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%Hh%M")
        if date_str != current_date:
            current_date = date_str
            lines.append(f"    section {date_str}")

        action = e.parameters.get("action", "")
        fname = Path(e.file_path).name if e.file_path else "?"
        desc = f"{e.produced_by}"
        if action:
            desc += f" {action}"
        desc += f" -> {fname}"
        if e.key_properties:
            kv = ", ".join(f"{k}={v}" for k, v in list(e.key_properties.items())[:3])
            desc += f" ({kv})"
        # 冒号是 timeline 分隔符, 描述里不能出现
        desc = desc.replace(":", " -")

        lines.append(f"        {time_str} : {_esc(desc)}")

    return "\n".join(lines)


def to_mermaid_for_context(registry: ProvenanceRegistry) -> str:
    """精简版 DAG, 只保留最近 10 个节点, 适合插入 agent 上下文.

    标签只留工具名 + 输出文件名, 不带参数/属性, 控制 token 量.
    """
    entries = _entries(registry)
    if not entries:
        return ""

    recent = entries[-10:]
    path_to_id: dict[str, str] = {e.file_path: f"N{i}" for i, e in enumerate(recent)}

    lines: list[str] = ["graph TD"]
    for i, e in enumerate(recent):
        nid = f"N{i}"
        fname = Path(e.file_path).name if e.file_path else "?"
        label = _esc(f"{e.produced_by}<br/>{fname}")
        lines.append(f'  {nid}["{label}"]')
        for inp in e.input_files:
            if inp in path_to_id:
                lines.append(f"  {path_to_id[inp]} --> {nid}")

    return "\n".join(lines)


if __name__ == "__main__":
    # 自检: 造一个假 registry, 跑三个函数, 验证输出结构
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
    reg.register(
        file_path="/work/band.dat",
        produced_by="vasp_tool",
        input_files=["/work/OUTCAR"],
        parameters={"action": "band"},
        file_format="dat",
        key_properties={"band_gap": 0.7, "converged": True},
    )
    reg.register(
        file_path="/work/bad.dat",
        produced_by="vasp_tool",
        input_files=["/work/OUTCAR"],
        parameters={"action": "scf"},
        file_format="dat",
        key_properties={"converged": False},
    )

    dag = visualize_dag(reg)
    assert "graph TD" in dag
    assert "N0" in dag and "N1" in dag
    assert "-->" in dag, "DAG 应该有依赖边"
    assert "conv" in dag, "应标记收敛节点"
    assert "noconv" in dag, "应标记未收敛节点"
    # 标签里的特殊字符要转义, 不能有裸双引号
    assert dag.count('"') % 2 == 0

    tl = visualize_timeline(reg)
    assert tl.startswith("timeline")
    assert "section" in tl
    # 时间线描述里不能有冒号 (会被 Mermaid 当分隔符)
    ev_lines = [l for l in tl.splitlines() if " : " in l]
    assert ev_lines, "应至少有一条事件"
    for el in ev_lines:
        # 事件文本在最后一个 ' : ' 之后, 不应再含冒号
        event_text = el.split(" : ", 1)[1]
        assert ":" not in event_text, f"事件文本含冒号: {event_text}"

    ctx = to_mermaid_for_context(reg)
    assert ctx.startswith("graph TD")
    assert ctx.count("\n") <= 12, "精简版应只留最近 10 个节点"

    # 空注册表不应崩
    empty = ProvenanceRegistry()
    assert "no provenance" in visualize_dag(empty)
    assert visualize_timeline(empty).startswith("timeline")
    assert to_mermaid_for_context(empty) == ""

    print("dag_visualizer self-check OK")
