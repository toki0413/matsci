"""本地结构库 —— 预存常用晶体结构，免去反复查 Materials Project。

查 Si / GaN / Cu 这类热门结构时先看本地，命中直接返回，省掉一次
外部 API 往返。没命中再走 materials_database_tool 调 API。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 数据文件就在同级的 data 目录下
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "common_structures.json"


class LocalStructureDB:
    """从 common_structures.json 加载的本地结构库，线程安全。

    查询入口:
        db.get("mp-149")   # 按 mp_id 查
        db.get("Si")        # 按化学式查
        db.get("SiO2")      # 按公式查
    """

    _singleton_lock = threading.Lock()
    _singleton: LocalStructureDB | None = None

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._lock = threading.RLock()
        # mp_id -> structure dict
        self._by_mp_id: dict[str, dict[str, Any]] = {}
        # formula -> list of structure dict (一个 formula 可能有多个相)
        self._by_formula: dict[str, list[dict[str, Any]]] = {}
        self._loaded = False
        self._load()

    @classmethod
    def shared(cls) -> LocalStructureDB:
        """进程级单例，避免每个工具各读一遍 JSON。"""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def _load(self) -> None:
        """从 JSON 加载结构数据，建索引。"""
        with self._lock:
            self._by_mp_id.clear()
            self._by_formula.clear()
            if not self._db_path.exists():
                logger.debug("local structure db not found: %s", self._db_path)
                self._loaded = True
                return
            try:
                raw = json.loads(self._db_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("failed to load local structure db: %s", exc)
                self._loaded = True
                return
            for mp_id, struct in raw.get("structures", {}).items():
                self._by_mp_id[mp_id] = struct
                formula = struct.get("formula_pretty") or struct.get("formula") or ""
                if formula:
                    self._by_formula.setdefault(formula, []).append(struct)
            self._loaded = True
            logger.debug(
                "local structure db loaded: %d structures", len(self._by_mp_id)
            )

    def reload(self) -> None:
        """重新从磁盘加载（add 了新结构后想刷新缓存时用）。"""
        self._load()

    def get(self, key: str) -> dict[str, Any] | None:
        """按 mp_id 或化学式查结构，命中返回 dict，否则 None。

        key 可以是 "mp-149"（mp_id）或 "Si" / "BaTiO3"（formula）。
        多个相匹配时返回第一个（按 JSON 里的顺序）。
        """
        if not key:
            return None
        key = key.strip()
        with self._lock:
            # 先按 mp_id 精确匹配
            if key in self._by_mp_id:
                return self._by_mp_id[key]
            # 再按 formula 匹配（不区分大小写）
            lower = key.lower()
            for formula, structs in self._by_formula.items():
                if formula.lower() == lower:
                    return structs[0]
            return None

    def list(self) -> list[str]:
        """列出所有可用的 mp_id。"""
        with self._lock:
            return sorted(self._by_mp_id.keys())

    def list_formulas(self) -> list[str]:
        """列出所有可用的化学式。"""
        with self._lock:
            return sorted(self._by_formula.keys())

    def add(self, structure: dict[str, Any]) -> bool:
        """往本地库加一个结构（内存 + 落盘）。

        structure 至少要有 mp_id 和 formula，缺了就不加。
        """
        mp_id = structure.get("mp_id")
        if not mp_id:
            return False
        with self._lock:
            self._by_mp_id[mp_id] = structure
            formula = structure.get("formula_pretty") or structure.get("formula") or ""
            if formula:
                lst = self._by_formula.setdefault(formula, [])
                # 同 mp_id 不重复加
                if not any(s.get("mp_id") == mp_id for s in lst):
                    lst.append(structure)
            # 落盘
            try:
                self._save()
            except Exception as exc:
                logger.warning("failed to persist local structure db: %s", exc)
            return True

    def _save(self) -> None:
        """把当前内存里的结构写回 JSON。"""
        if not self._db_path.parent.exists():
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "_meta": {
                "description": "常用晶体结构本地库",
                "version": 1,
                "source": "Materials Project + user additions",
            },
            "structures": dict(sorted(self._by_mp_id.items())),
        }
        self._db_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_mp_id)
