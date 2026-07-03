"""本地数据脱敏 + 云端训练 opt-out 守卫.

用户开启后, 发往云端 LLM 的本地敏感数据 (结构坐标 / MD 轨迹 / 计算结果 /
对话历史) 会被替换成摘要形式, 同时给 API 请求加 opt-out 训练 header.
三档级别:

  - off:        不脱敏, 正常发云端 (默认)
  - redact:     敏感本地数据脱敏后发云端
  - local_only: 完全本地, 不发云端 (强制走本地模型)

级别通过 FeatureFlags 的 privacy_* flag 互斥控制, 运行时改动不写盘.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from functools import reduce
from math import gcd
from typing import Any

logger = logging.getLogger(__name__)


class PrivacyGuard:
    """本地数据脱敏 + 云端训练 opt-out 守卫."""

    LEVELS = {
        "off": "不脱敏, 正常发云端 (默认)",
        "redact": "敏感本地数据脱敏后发云端 (结构→化学式, 轨迹→统计, 结果→关键值)",
        "local_only": "完全本地, 不发云端 (强制走本地模型)",
    }

    # 数据生命周期分级 —— 参考 Moonshine Voice 的隐私数据生命周期管理.
    # permanent:  公开数据 (已发表论文, 公开数据库), 永不脱敏, 永久保留
    # temporary:  实验原始数据 (POSCAR/轨迹/计算结果), 脱敏后发云端, 会话结束后本地清除
    # ephemeral:  凭证/内部路径/密钥, 绝不发云端, 用完立即清除
    DATA_TIERS = {
        "permanent": {"redact": False, "retain_after_session": True},
        "temporary": {"redact": True, "retain_after_session": False},
        "ephemeral": {"redact": True, "retain_after_session": False, "never_cloud": True},
    }

    _singleton_lock = threading.Lock()
    _singleton: "PrivacyGuard | None" = None

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 已脱敏次数 / 各类型计数, 给 summary 用
        self._redact_count: int = 0
        self._type_counts: Counter[str] = Counter()
        # 临时数据注册表: 记录哪些数据需要在会话结束后清除
        self._ephemeral_keys: set[str] = set()

    @classmethod
    def shared(cls) -> "PrivacyGuard":
        """进程级单例."""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # ── 级别读写 ──────────────────────────────────────────────

    def get_level(self) -> str:
        """读当前隐私级别. 从 FeatureFlags 反推, 三个 flag 互斥."""
        from huginn.feature_flags import FeatureFlags

        ff = FeatureFlags.shared()
        # 优先级 local_only > redact > off
        if ff.is_enabled("privacy_local_only"):
            return "local_only"
        if ff.is_enabled("privacy_redact"):
            return "redact"
        return "off"

    def set_level(self, level: str) -> None:
        """设隐私级别. 三个 flag 互斥, 只让对应的一个 True."""
        if level not in self.LEVELS:
            raise ValueError(
                f"未知隐私级别: {level}. 可选: {list(self.LEVELS)}"
            )
        from huginn.feature_flags import FeatureFlags

        ff = FeatureFlags.shared()
        # 一次性把三个 flag 都设到位, 保证互斥
        ff.toggle("privacy_off", level == "off")
        ff.toggle("privacy_redact", level == "redact")
        ff.toggle("privacy_local_only", level == "local_only")
        logger.info("privacy level set to '%s'", level)

    # ── 云端开关 ──────────────────────────────────────────────

    def should_send_to_cloud(self) -> bool:
        """local_only 时返回 False, 其他级别 True."""
        return self.get_level() != "local_only"

    def should_use_local(self, provider: str | None) -> bool:
        """local_only 模式下, 当前 provider 是云端就要切本地.
        返回 True 表示 "需要切到本地". 本地 provider 直接返回 False.
        """
        if self.get_level() != "local_only":
            return False
        # ollama / vllm / local 都算本地, 不用切
        return provider not in ("ollama", "vllm", "local", None)

    # ── 数据生命周期分级 ────────────────────────────────────

    def classify_data(self, content: str, content_type: str = "auto") -> str:
        """根据内容类型和数据特征判定生命周期等级.

        返回 permanent / temporary / ephemeral 之一.
        策略:
          - 凭证/密钥/路径 → ephemeral (绝不发云端)
          - 结构/轨迹/计算结果 → temporary (脱敏后可发, 会话后清除)
          - 其他 (含公开论文摘录) → permanent
        """
        if content_type == "auto":
            content_type = self._detect_type(content)

        # 凭证/密钥特征: API key, token, password, 路径
        if re.search(
            r"(api[_-]?key|secret|token|password|passwd|credential|"
            r"\.env|/home/|C:\\Users\\|/root/)",
            content, re.I,
        ):
            return "ephemeral"

        # 实验原始数据: 结构/轨迹/计算结果 → temporary
        if content_type in ("structure", "trajectory", "calculation_result"):
            return "temporary"

        # 对话历史可能含敏感信息 → temporary
        if content_type == "conversation":
            return "temporary"

        # 其余 (公开论文, 理论知识) → permanent
        return "permanent"

    def get_retention_policy(self, tier: str) -> dict[str, Any]:
        """返回某等级的保留策略 dict."""
        return self.DATA_TIERS.get(tier, self.DATA_TIERS["temporary"])

    def should_redact_by_tier(self, content: str, content_type: str = "auto") -> bool:
        """根据数据分级判断是否需要脱敏, 比级别开关更细粒度.

        ephemeral 等级在任何 privacy level 下都强制脱敏.
        temporary 在 redact/local_only 下脱敏.
        permanent 永不脱敏.
        """
        tier = self.classify_data(content, content_type)
        policy = self.get_retention_policy(tier)
        if not policy.get("redact", False):
            return False
        if policy.get("never_cloud", False):
            return True
        return self.get_level() != "off"

    def register_ephemeral(self, key: str) -> None:
        """注册一个临时数据 key, purge_session 时会清除."""
        with self._lock:
            self._ephemeral_keys.add(key)

    def purge_session(self) -> list[str]:
        """会话结束时清除所有临时数据, 返回被清除的 key 列表."""
        with self._lock:
            purged = list(self._ephemeral_keys)
            self._ephemeral_keys.clear()
            # 重置脱敏计数
            self._redact_count = 0
            self._type_counts.clear()
        if purged:
            logger.info("purged %d ephemeral data entries", len(purged))
        return purged

    # ── 核心脱敏 ──────────────────────────────────────────────

    def redact_for_cloud(
        self,
        content: str,
        content_type: str = "auto",
    ) -> str:
        """对一段文本做脱敏. content_type 不传时自动检测.

        off 级别原样返回, redact / local_only 才真脱敏.
        local_only 实际不会发云端, 但也走一遍以备万一.
        """
        level = self.get_level()
        if level == "off":
            return content

        ctype = content_type
        if ctype == "auto":
            ctype = self._detect_type(content)

        with self._lock:
            self._redact_count += 1
            self._type_counts[ctype] += 1

        if ctype == "structure":
            return self._redact_structure(content)
        if ctype == "trajectory":
            return self._redact_trajectory(content)
        if ctype == "calculation_result":
            return self._redact_calculation(content)
        if ctype == "conversation":
            return self._redact_conversation(content)
        # 未知类型原样返回, 至少记个数
        return content

    # ── 类型自动检测 ─────────────────────────────────────────

    def _detect_type(self, content: str) -> str:
        """根据关键词 / 正则猜内容类型. 都不匹配返回 'unknown'."""
        # 结构: POSCAR / CIF 关键词
        if re.search(
            r"\b(ATOMS|Cartesian|Direct|CONTVAR|POSCAR|_cell_length)\b",
            content,
            re.I,
        ):
            return "structure"
        # CIF 特征字段
        if "_symmetry_space_group_name" in content or (
            "loop_" in content and "_atom_site" in content
        ):
            return "structure"
        # 轨迹: 多帧 / Step / TIMESTEP, 单独看 Step 要 ≥3 行才像轨迹
        if re.search(r"\bTIMESTEP\b", content):
            return "trajectory"
        if re.search(r"\b(Step|frame|FRAME)\b", content):
            step_hits = re.findall(r"Step\s*\d+", content)
            frame_hits = re.findall(r"frame\s*\d+", content, re.I)
            if len(step_hits) >= 3 or len(frame_hits) >= 3:
                return "trajectory"
        # 计算结果: 能量 / 带隙 / 力 / 应力 / 费米能
        if re.search(
            r"(free energy|electronic energy|band gap|bandgap|"
            r"maximum force|stress|E-fermi|TOTAL ENERGY|without entropy)",
            content,
            re.I,
        ):
            return "calculation_result"
        # 对话: user/assistant 标记
        if re.search(
            r"(User:|Assistant:|user:|assistant:|<\|user\|>|<\|assistant\|>)",
            content,
        ):
            return "conversation"
        return "unknown"

    # ── 结构脱敏 ─────────────────────────────────────────────

    def _redact_structure(self, content: str) -> str:
        """POSCAR / CIF → 化学式 + 原子数 + 空间群(尽力) + 晶系(尽力)."""
        formula = "?"
        n_atoms = 0
        space_group = "?"
        crystal_system = "?"

        lines = content.splitlines()
        # POSCAR: 第 6 行元素名, 第 7 行各元素计数
        if len(lines) >= 7:
            elem_line = lines[5].split()
            count_line = lines[6].split()
            try:
                counts = [int(x) for x in count_line]
                # 元素名行可能是数字 (老 POSCAR 没元素名), 此时退化
                if elem_line and all(not x.isdigit() for x in elem_line):
                    parts = []
                    total = 0
                    for e, c in zip(elem_line, counts):
                        parts.append(f"{e}{c if c > 1 else ''}")
                        total += c
                    formula = self._normalize_formula("".join(parts))
                    n_atoms = total
                else:
                    n_atoms = sum(counts)
                    formula = f"M{n_atoms}"
            except (ValueError, IndexError):
                pass

        # CIF: 抓 _chemical_formula_sum 和 _symmetry_space_group_name
        m = re.search(
            r"_chemical_formula_sum\s+['\"]?([A-Za-z0-9]+)['\"]?",
            content,
            re.I,
        )
        if m:
            formula = self._normalize_formula(m.group(1))
        m2 = re.search(
            r"_symmetry_space_group_name_H-M\s+['\"]?([^'\"\s]+)['\"]?",
            content,
            re.I,
        )
        if m2:
            space_group = m2.group(1)
        m3 = re.search(
            r"_symmetry_cell_setting\s+([A-Za-z]+)", content, re.I
        )
        if m3:
            crystal_system = m3.group(1).lower()

        # 空间群 → 晶系兜底推断
        if space_group != "?" and crystal_system == "?":
            crystal_system = self._guess_crystal_system(space_group)

        return (
            f"[STRUCTURE: formula={formula}, n_atoms={n_atoms}, "
            f"space_group={space_group}, crystal_system={crystal_system}]"
        )

    @staticmethod
    def _normalize_formula(raw: str) -> str:
        """把 Mg2O2 合并成 MgO, Si2 约简成 Si. 简单 GCD 约简."""
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", raw)
        pairs: list[tuple[str, int]] = []
        for e, c in tokens:
            n = int(c) if c else 1
            pairs.append((e, n))
        if not pairs:
            return raw
        # 合并同名元素
        merged: dict[str, int] = {}
        for e, n in pairs:
            merged[e] = merged.get(e, 0) + n
        counts = list(merged.values())
        g = reduce(gcd, counts, 0)
        if g <= 1:
            return "".join(
                f"{e}{n if n > 1 else ''}" for e, n in merged.items()
            )
        return "".join(
            f"{e}{n // g if n // g > 1 else ''}" for e, n in merged.items()
        )

    @staticmethod
    def _guess_crystal_system(sg: str) -> str:
        """按空间群号兜底猜晶系. sg 可能是数字也可能是 HM 符号."""
        m = re.search(r"(\d+)", sg)
        if not m:
            return "?"
        n = int(m.group(1))
        if 1 <= n <= 2:
            return "triclinic"
        if 3 <= n <= 15:
            return "monoclinic"
        if 16 <= n <= 74:
            return "orthorhombic"
        if 75 <= n <= 142:
            return "tetragonal"
        if 143 <= n <= 167:
            return "trigonal"
        if 168 <= n <= 194:
            return "hexagonal"
        if 195 <= n <= 230:
            return "cubic"
        return "?"

    # ── 轨迹脱敏 ─────────────────────────────────────────────

    def _redact_trajectory(self, content: str) -> str:
        """MD 轨迹 → 原子数 + 帧数 + 时间步 + 能量范围 + 温度范围."""
        n_atoms = 0
        n_frames = 0
        dt = "?"
        e_min: float | None = None
        e_max: float | None = None
        t_min: float | None = None
        t_max: float | None = None

        # LAMMPS dump: NUMBER OF ATOMS: N
        m = re.search(r"NUMBER OF ATOMS:\s*(\d+)", content)
        if m:
            n_atoms = int(m.group(1))
        # 帧数: TIMESTEP 出现次数, 备用 Step 行数
        n_frames = len(re.findall(r"^TIMESTEP\s*$", content, re.M))
        if n_frames == 0:
            n_frames = len(re.findall(r"^Step\s+\d+", content, re.M))

        # 时间步
        m = re.search(r"dt\s*=\s*([\d.]+)\s*(fs|ps|ns)?", content)
        if m:
            dt = f"{m.group(1)}{m.group(2) or 'fs'}"

        # 能量 (eV), 抓所有形如 E=...eV 的值
        energies = re.findall(r"E\s*=?\s*(-?\d+\.\d+)\s*eV", content)
        if energies:
            try:
                vals = [float(e) for e in energies]
                e_min, e_max = min(vals), max(vals)
            except ValueError:
                pass

        # 温度 (K)
        temps = re.findall(r"T\s*=?\s*(-?\d+\.\d+)\s*K", content)
        if temps:
            try:
                vals = [float(t) for t in temps]
                t_min, t_max = min(vals), max(vals)
            except ValueError:
                pass

        e_str = (
            f"({e_min}, {e_max}) eV" if e_min is not None else "?"
        )
        t_str = f"({t_min}, {t_max}) K" if t_min is not None else "?"
        return (
            f"[TRAJECTORY: n_atoms={n_atoms}, n_frames={n_frames}, "
            f"dt={dt}, E_range={e_str}, T_range={t_str}]"
        )

    # ── 计算结果脱敏 ─────────────────────────────────────────

    def _redact_calculation(self, content: str) -> str:
        """DFT / MD 输出 → 类型 + 关键标量 + 方法."""
        # 先判类型
        calc_type = "dft"
        if re.search(r"\b(md|md_run|molecular dynamics|nvt|npt)\b", content, re.I):
            calc_type = "md"
        elif re.search(r"\b(relax|vc-relax|optimization)\b", content, re.I):
            calc_type = "dft_relax"
        elif re.search(r"\b(scf|static)\b", content, re.I):
            calc_type = "dft_scf"

        energy = self._extract_first(
            r"(?:free energy|TOTAL ENERGY|energy without entropy|E0)"
            r"\s*=?\s*(-?\d+\.\d+)\s*eV",
            content,
        )
        band_gap = self._extract_first(
            r"(?:band gap|bandgap|Bg|Kohn-Sham gap)\s*=?\s*(-?\d+\.\d+)\s*eV",
            content,
        )
        max_force = self._extract_first(
            r"(?:maximum force|max force|FORCES:max)\s*=?\s*(-?\d+\.\d+)\s*(?:eV/Å|eV/A)",
            content,
        )
        functional = self._extract_first(
            r"(?:functional|XC|GGA|LDA)\s*=?\s*([A-Za-z0-9]+)",
            content,
        ) or "?"

        parts = [f"type={calc_type}"]
        if energy is not None:
            parts.append(f"energy={energy} eV")
        if band_gap is not None:
            parts.append(f"band_gap={band_gap} eV")
        if max_force is not None:
            parts.append(f"max_force={max_force} eV/Å")
        parts.append(f"functional={functional}")
        return f"[RESULT: {', '.join(parts)}]"

    @staticmethod
    def _extract_first(pattern: str, text: str) -> str | None:
        m = re.search(pattern, text, re.I)
        return m.group(1) if m else None

    # ── 对话脱敏 ─────────────────────────────────────────────

    def _redact_conversation(self, content: str) -> str:
        """对话历史 → 保留最近 3 轮原文, 更早的压缩成摘要标记."""
        # 按 user/assistant 标记切分轮次
        turns = re.split(
            r"(?=User:|Assistant:|user:|assistant:|<\|user\|>|<\|assistant\|>)",
            content,
        )
        turns = [t for t in turns if t.strip()]
        n = len(turns)
        if n <= 3:
            return content
        recent = turns[-3:]
        earlier_count = n - 3
        marker = (
            f"[CONVERSATION: 3 recent turns kept, "
            f"{earlier_count} earlier turns summarized]"
        )
        return marker + "\n" + "".join(recent)

    # ── 消息列表脱敏 ─────────────────────────────────────────

    def redact_messages_for_cloud(self, messages: list[Any]) -> list[Any]:
        """对 LangChain 消息列表逐条脱敏.

        - system 消息不动 (系统提示没有用户数据)
        - tool 消息按内容自动检测类型
        - 非字符串 content (多模态) 不动
        - 单条挂了不影响其他消息
        """
        if self.get_level() == "off":
            return messages
        out: list[Any] = []
        for msg in messages:
            try:
                role = getattr(msg, "type", "") or msg.__class__.__name__.lower()
                # system 提示没有用户数据, 不动
                if role == "system":
                    out.append(msg)
                    continue
                content = getattr(msg, "content", None)
                if not isinstance(content, str):
                    # 非字符串 (多模态 list/dict) 不动
                    out.append(msg)
                    continue
                redacted = self.redact_for_cloud(content, "auto")
                # LangChain 消息有 copy(update=...), 用它替换 content
                if hasattr(msg, "copy"):
                    out.append(msg.copy(update={"content": redacted}))
                else:
                    try:
                        msg.content = redacted  # type: ignore[misc]
                    except Exception:
                        pass
                    out.append(msg)
            except Exception:
                # 单条挂了不能影响其他消息
                logger.warning("redact message failed", exc_info=True)
                out.append(msg)
        return out

    # ── opt-out header ───────────────────────────────────────

    def apply_opt_out_headers(
        self,
        provider: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """给 API 请求加 opt-out 训练 header. 本地 provider 不加.

        OpenAI / Anthropic 默认不用 API 数据训练, 这里加自定义 header
        作为内部审计标记, 不影响请求语义.
        """
        out = dict(headers or {})
        # 本地 provider 不发任何 opt-out header
        if provider in ("ollama", "vllm", "local"):
            return out
        # 通用标记
        out["X-Data-Usage"] = "no-training"
        if provider == "openai":
            out.setdefault("X-OpenAI-Data-Usage", "no-training")
        elif provider == "anthropic":
            out.setdefault("X-Anthropic-Data-Usage", "no-training")
        return out

    # ── 状态汇总 ─────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """返回当前级别 + 已脱敏次数 + 各类型脱敏统计."""
        with self._lock:
            counts = dict(self._type_counts)
            total = self._redact_count
        level = self.get_level()
        return {
            "level": level,
            "level_description": self.LEVELS[level],
            "redact_count": total,
            "type_counts": counts,
            "send_to_cloud": self.should_send_to_cloud(),
        }
