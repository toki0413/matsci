"""计算溯源注册表 — 追踪文件产出关系, 构建科学计算的 DAG.

每个工具调用产生的文件自动注册:
  (tool_name, inputs, parameters) → (file_path, format, key_properties)

agent 可以查询:
  - "Si 结构的弛豫结果在哪?" → 通过 provenance 链找到 OUTCAR
  - "哪些计算用了 PBE 泛函?" → 按参数查询
  - "这个结构的能量是多少?" → 从 key_properties 直接取

与 OpenHarness 的 Unified File Shortcut 区别:
  - OH 只记录 "文件在哪"
  - 我们记录 "文件是怎么来的, 从什么来, 包含什么"
  这让 agent 在压缩后仍能找到关键文件, 不依赖对话历史.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceEntry:
    """一次文件产出的溯源记录."""

    file_path: str
    produced_by: str  # tool name
    produced_at: float  # timestamp
    input_files: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    file_format: str = ""  # poscar/outcar/cif/...
    key_properties: dict[str, Any] = field(default_factory=dict)
    # 能量/带隙/晶格常数等关键值, 直接存在这里,
    # agent 压缩后仍可查询, 不需要重新解析文件.

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "produced_by": self.produced_by,
            "produced_at": self.produced_at,
            "input_files": list(self.input_files),
            "parameters": self.parameters,
            "file_format": self.file_format,
            "key_properties": self.key_properties,
        }


class ProvenanceRegistry:
    """全局溯源注册表, 进程级单例."""

    _instance: ProvenanceRegistry | None = None

    def __init__(self) -> None:
        self._entries: list[ProvenanceEntry] = []
        self._by_path: dict[str, ProvenanceEntry] = {}
        self._by_tool: dict[str, list[ProvenanceEntry]] = {}

    @classmethod
    def shared(cls) -> ProvenanceRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        file_path: str,
        produced_by: str,
        input_files: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        file_format: str = "",
        key_properties: dict[str, Any] | None = None,
    ) -> ProvenanceEntry:
        """注册一条产出记录."""
        entry = ProvenanceEntry(
            file_path=file_path,
            produced_by=produced_by,
            produced_at=time.time(),
            input_files=input_files or [],
            parameters=parameters or {},
            file_format=file_format,
            key_properties=key_properties or {},
        )
        self._entries.append(entry)
        self._by_path[file_path] = entry
        self._by_tool.setdefault(produced_by, []).append(entry)

        # 限制总量, 超过 500 条删最旧的
        if len(self._entries) > 500:
            old = self._entries.pop(0)
            self._by_path.pop(old.file_path, None)

        return entry

    def find_by_path(self, path: str) -> ProvenanceEntry | None:
        return self._by_path.get(path)

    def find_by_tool(self, tool_name: str) -> list[ProvenanceEntry]:
        return self._by_tool.get(tool_name, [])

    def find_by_format(self, fmt: str) -> list[ProvenanceEntry]:
        return [e for e in self._entries if e.file_format == fmt]

    def find_by_property(self, key: str, value: Any = None) -> list[ProvenanceEntry]:
        """按 key_properties 查找. value=None 时只要有这个 key 就行."""
        results = []
        for e in self._entries:
            if key in e.key_properties:
                if value is None or e.key_properties[key] == value:
                    results.append(e)
        return results

    def get_lineage(self, file_path: str, depth: int = 5) -> list[ProvenanceEntry]:
        """获取文件的溯源链: 它是从哪些文件来的, 那些文件又是从哪来的."""
        chain: list[ProvenanceEntry] = []
        visited: set[str] = set()
        current = self._by_path.get(file_path)
        while current and depth > 0 and current.file_path not in visited:
            chain.append(current)
            visited.add(current.file_path)
            if current.input_files:
                # 取第一个输入文件的溯源
                current = self._by_path.get(current.input_files[0])
            else:
                break
            depth -= 1
        return chain

    def query(self, query_str: str) -> list[dict[str, Any]]:
        """自然语言式查询 (简单版): 按格式/工具/属性搜索."""
        q = query_str.lower()
        results = []
        for e in self._entries:
            score = 0
            if q in e.file_path.lower():
                score += 2
            if q in e.produced_by.lower():
                score += 2
            if q in e.file_format.lower():
                score += 1
            for k, v in e.key_properties.items():
                if q in k.lower() or q in str(v).lower():
                    score += 3
            if score > 0:
                results.append((score, e.to_dict()))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:20]]

    def summary(self) -> dict[str, Any]:
        """当前注册表摘要, 给 agent 上下文用."""
        return {
            "total_files": len(self._entries),
            "by_tool": {k: len(v) for k, v in self._by_tool.items()},
            "by_format": {},
            "recent": [e.to_dict() for e in self._entries[-10:]],
        }

    def to_context_block(self) -> str:
        """生成可插入上下文的状态块, 跨压缩保留."""
        if not self._entries:
            return ""
        lines = ["### Provenance registry (active files):"]
        # 只列最近 10 个, 带 key_properties
        for e in self._entries[-10:]:
            props = ""
            if e.key_properties:
                props = " | " + ", ".join(
                    f"{k}={v}" for k, v in e.key_properties.items()
                )
            lines.append(f"  - {e.file_path} ({e.file_format or '?'}) by {e.produced_by}{props}")
        return "\n".join(lines)


def register_tool_output(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
) -> None:
    """从工具调用中自动提取文件路径和关键属性, 注册到溯源表.

    在 ToolAdapter._run_post_checks 里调用.
    """
    try:
        reg = ProvenanceRegistry.shared()

        # 从 tool_input 提取输入文件
        input_files: list[str] = []
        for key in ("file_path", "working_dir", "poscar_path", "structure_file"):
            val = tool_input.get(key)
            if val and isinstance(val, str):
                input_files.append(val)

        # 从 tool_output 提取产出文件和关键属性
        if not isinstance(tool_output, dict):
            return

        result = tool_output.get("result", tool_output)
        if not isinstance(result, dict):
            return

        # 提取文件路径
        output_paths: list[str] = []
        for key in ("output_file", "outcar_path", "trajectory_file", "file_path", "saved_to"):
            val = result.get(key)
            if val and isinstance(val, str):
                output_paths.append(val)

        # 提取关键属性
        key_props: dict[str, Any] = {}
        for key in (
            "energy", "total_energy", "free_energy", "E0",
            "band_gap", "lattice_constant", "converged",
            "forces_max", "stress_max", "pressure",
            "spacegroup", "volume", "density",
            "n_atoms", "formula",
        ):
            val = result.get(key)
            if val is not None:
                key_props[key] = val

        # 提取参数
        params: dict[str, Any] = {}
        for key in ("action", "encut", "ediff", "kpoints", "functional", "basis_set", "method"):
            val = tool_input.get(key)
            if val is not None:
                params[key] = val

        # 推断文件格式
        fmt = ""
        for key in ("file_format", "format"):
            val = result.get(key)
            if val:
                fmt = str(val)
                break
        if not fmt and output_paths:
            ext = Path(output_paths[0]).suffix.lstrip(".").lower()
            fmt = ext

        # 注册每个产出文件
        for path in output_paths:
            reg.register(
                file_path=path,
                produced_by=tool_name,
                input_files=input_files,
                parameters=params,
                file_format=fmt,
                key_properties=key_props,
            )
    except Exception:
        logger.debug("register_tool_output failed (non-fatal)", exc_info=True)
