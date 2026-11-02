"""共享颗粒堆积数据加载 — comsol / abaqus / openfoam 三个仿真工具的共同入口.

DRY: 三处 ``_load_packing_data`` 是逐字相同的实现 (inline dict 或相对 work_dir
的 JSON 路径 → objects 列表). 任一处的路径解析语义变化都会让三个工具对同一份
输入给出不同结果, 因此收敛为单一实现.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_packing_objects(
    packing_data: dict[str, Any] | str | None, work_dir: Path
) -> list[dict[str, Any]]:
    """把 packing_data 读成 objects 列表.

    packing_data:
      - None        → [] (未提供即无颗粒)
      - str         → JSON 文件路径 (相对路径按 work_dir 解析)
      - dict        → 直接取其中的 ``objects``
    """
    if packing_data is None:
        return []
    if isinstance(packing_data, str):
        path = Path(packing_data)
        if not path.is_absolute():
            path = work_dir / path
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = packing_data
    return data.get("objects", [])
