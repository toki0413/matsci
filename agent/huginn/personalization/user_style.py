"""用户语言偏好学习系统 — 基于实际对话逐步定制 agent 通信风格.

不是过度推断, 而是从用户真实消息里抽特征, 用 EMA 平滑更新 profile,
confidence 低时不注入 system prompt 避免瞎猜. profile 只存本地 SQLite.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── 术语词典 ──────────────────────────────────────────────────────
# 这些词表是特征提取的硬规则, 按专业程度分级. 命中数越高, 用户越专业.

# 专业黑话 (DFT/VASP 等领域术语), 命中 ≥3 个判 expert
_EXPERT_TERMS = {
    # DFT / VASP
    "dft", "encut", "ismear", "pbe", "brillouin", "vasp", "kpoints",
    "potcar", "poscar", "incar", "lda", "gga", "hse06", "bse", "gw",
    "soc", "dos", "pdos", "scf", "neb", "npt", "nvt", "vcrelax",
    "hubbard", "bader", "fermi", "phonon", "eos", "born effective",
    # 计算软件
    "lammps", "cp2k", "quantum espresso", "abinit",
    # 物理量 / 概念
    "bandgap", "kpoint", "spin-orbit", "paw",
}

# 技术词 (中文材料学术语), 命中 ≥2 个判 technical
_TECHNICAL_TERMS_ZH = {
    "带隙", "晶格", "优化", "能带", "态密度", "弛豫", "电子结构",
    "缺陷", "表面", "界面", "异质结", "铁电", "压电", "应力", "应变",
    "杨氏模量", "热导", "电导", "磁矩", "相图", "固溶体", "掺杂",
    "能带结构", "费米面", "声子谱", "弹性常数", "介电", "波函数",
}

# 通用词, 命中任一判 intermediate
_INTERMEDIATE_TERMS = {
    "计算", "模拟", "材料", "性质", "结构", "参数", "模型", "结果",
    "分析", "预测", "实验", "理论", "方法", "数据", "能量", "温度",
}

# 中英文术语对, 用于 preferred_terms 提取
# 当用户在 msg 里用 en 形式, 记 preferred_terms[zh] = en
_ZH_EN_TERM_PAIRS = {
    "带隙": "bandgap",
    "第一性原理": "DFT",
    "晶格": "lattice",
    "能带": "band structure",
    "态密度": "DOS",
    "弛豫": "relaxation",
    "电子结构": "electronic structure",
    "缺陷": "defect",
    "表面": "surface",
    "界面": "interface",
    "异质结": "heterojunction",
    "铁电": "ferroelectric",
    "压电": "piezoelectric",
    "应力": "stress",
    "应变": "strain",
    "杨氏模量": "Young's modulus",
    "分子动力学": "MD",
    "蒙特卡洛": "MC",
    "优化": "optimization",
    "收敛": "convergence",
    "声子": "phonon",
    "费米": "Fermi",
    "布里渊区": "Brillouin zone",
}

# 正式用语标记
_FORMAL_MARKERS = {"请", "您好", "感谢", "谢谢", "劳驾", "麻烦", "烦请"}

# 口语语气词
_CASUAL_MARKERS = {"嗨", "嗯", "哈", "吧", "呢", "啊", "哎", "哟", "嘛", "呀"}


@dataclass
class UserStyleProfile:
    """用户语言偏好 profile, 由 StyleLearner 通过 EMA 更新."""

    vocabulary_level: str = "intermediate"  # expert / technical / intermediate / beginner
    formality: str = "neutral"              # formal / neutral / casual
    verbosity: str = "moderate"             # concise / moderate / detailed
    language: str = "zh"                    # zh / en / mixed
    preferred_terms: dict[str, str] = field(default_factory=dict)  # zh_term -> 偏好的形式
    avoid_terms: list[str] = field(default_factory=list)
    response_format: str = "markdown"       # markdown / plain / table_heavy
    code_style: str = "commented"           # commented / minimal / verbose
    confidence: float = 0.0
    sample_count: int = 0


# 通过 EMA 投票更新的分类维度
_CATEGORICAL_DIMS = ("vocabulary_level", "formality", "verbosity", "language")

# 反馈里抠避免用词的正则
_AVOID_PATTERNS = [
    re.compile(r"[避免别][要用说]([^\s，。、,.;:！!？?]+)"),
    re.compile(r"不要[用说]([^\s，。、,.;:！!？?]+)"),
    re.compile(r'[""\u2018\u2019\u201c\u201d]([^"\u2018\u2019\u201c\u201d]+)[""\u2018\u2019\u201c\u201d]'),
]


class StyleLearner:
    """观察对话, 学习用户语言偏好.

    分类维度用加权投票实现 EMA (新观察权重 0.1, 老分布权重 0.9),
    避免单次观察剧烈改变 profile. preferred_terms 按累计频次提取,
    avoid_terms 从用户显式反馈里抠. profile 持久化到本地 SQLite, 不上传.
    """

    def __init__(self, storage_path: str = ":memory:"):
        self._storage_path = storage_path
        # RLock 允许同线程重入, observe() 内部调 save() 不会死锁
        self._lock = threading.RLock()
        # 分类维度的加权分数: _scores[dim][value] = float
        self._scores: dict[str, dict[str, float]] = {
            dim: {} for dim in _CATEGORICAL_DIMS
        }
        # 术语使用频次: _term_usage[zh_term] = {"zh": n, "en": m}
        self._term_usage: dict[str, dict[str, int]] = {}
        # 手动设置的维度, 学习时不覆盖
        self._manual_overrides: set[str] = set()
        self._profile = UserStyleProfile()

        if storage_path != ":memory:":
            # 目录不存在 sqlite3.connect 会炸, 先建好
            Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
        self.load()

    # ── 观察 ────────────────────────────────────────────────────

    def observe(
        self,
        user_message: str,
        agent_response: str,
        user_feedback: str | None = None,
    ) -> None:
        """观察一次对话, 抽特征并 EMA 更新 profile."""
        # flag 关掉时不学习, 直接返回
        try:
            from huginn.feature_flags import FeatureFlags
            if not FeatureFlags.shared().is_enabled("personalization"):
                return
        except Exception:
            # flag 层挂了不能带挂业务, 继续走原逻辑
            pass

        if not user_message:
            return
        with self._lock:
            self._update_vocabulary(user_message)
            self._update_formality(user_message)
            self._update_verbosity(user_message)
            self._update_language(user_message)
            self._update_preferred_terms(user_message)
            if user_feedback:
                self._apply_feedback_text(user_feedback)
            self._profile.sample_count += 1
            self._profile.confidence = min(1.0, self._profile.sample_count / 20)
            self._sync_scores_to_profile()
            self._save_unlocked()

    def _update_vocabulary(self, msg: str) -> None:
        msg_lower = msg.lower()
        expert_hits = sum(1 for t in _EXPERT_TERMS if t in msg_lower)
        tech_hits = sum(1 for t in _TECHNICAL_TERMS_ZH if t in msg)
        inter_hits = sum(1 for t in _INTERMEDIATE_TERMS if t in msg)
        if expert_hits >= 3:
            level = "expert"
        elif tech_hits >= 2:
            level = "technical"
        elif inter_hits >= 1:
            level = "intermediate"
        else:
            level = "beginner"
        self._ema_vote("vocabulary_level", level)

    def _update_formality(self, msg: str) -> None:
        formal_hits = sum(1 for m in _FORMAL_MARKERS if m in msg)
        casual_hits = sum(1 for m in _CASUAL_MARKERS if m in msg)
        if formal_hits >= 1 and casual_hits == 0:
            level = "formal"
        elif casual_hits >= 1:
            level = "casual"
        else:
            level = "neutral"
        self._ema_vote("formality", level)

    def _update_verbosity(self, msg: str) -> None:
        n = len(msg)
        if n < 50:
            level = "concise"
        elif n <= 200:
            level = "moderate"
        else:
            level = "detailed"
        self._ema_vote("verbosity", level)

    def _update_language(self, msg: str) -> None:
        zh = sum(1 for c in msg if "\u4e00" <= c <= "\u9fff")
        en = sum(1 for c in msg if c.isascii() and c.isalpha())
        if zh == 0 and en == 0:
            level = "zh"
        elif zh == 0:
            level = "en"
        elif en == 0:
            level = "zh"
        elif zh > en * 4:
            level = "zh"
        elif en > zh * 4:
            level = "en"
        else:
            level = "mixed"
        self._ema_vote("language", level)

    def _ema_vote(self, dim: str, value: str) -> None:
        """EMA 投票: 新观察加 0.1, 老分布整体衰减 0.9.

        手动设过的维度不参与学习, 避免被 EMA 洗掉.
        """
        if dim in self._manual_overrides:
            return
        scores = self._scores[dim]
        for v in scores:
            scores[v] *= 0.9
        scores[value] = scores.get(value, 0.0) + 0.1

    def _sync_scores_to_profile(self) -> None:
        """把 _scores 里得分最高的值写回 profile 字段."""
        for dim in _CATEGORICAL_DIMS:
            if dim in self._manual_overrides:
                continue
            scores = self._scores[dim]
            if not scores:
                continue
            best_val, best_score = max(scores.items(), key=lambda kv: kv[1])
            if best_score > 0:
                setattr(self._profile, dim, best_val)

    # ── 术语偏好 ────────────────────────────────────────────────

    def _update_preferred_terms(self, msg: str) -> None:
        """统计用户实际用的术语形式, 累计 ≥2 次且有偏好才写进 profile."""
        msg_lower = msg.lower()
        for zh, en in _ZH_EN_TERM_PAIRS.items():
            en_lower = en.lower()
            used_zh = zh in msg
            used_en = en_lower in msg_lower
            if not (used_zh or used_en):
                continue
            bucket = self._term_usage.setdefault(zh, {"zh": 0, "en": 0})
            if used_zh:
                bucket["zh"] += 1
            if used_en:
                bucket["en"] += 1
            # 至少 2 次才记, 避免单次偶然
            total = bucket["zh"] + bucket["en"]
            if total < 2:
                continue
            if bucket["en"] > bucket["zh"]:
                self._profile.preferred_terms[zh] = en
            elif bucket["zh"] > bucket["en"]:
                # 用户更偏好中文形式
                self._profile.preferred_terms[zh] = zh
            # 平局不更新, 等下次观察打破

    # ── 反馈 ────────────────────────────────────────────────────

    def _apply_feedback_text(self, feedback: str) -> None:
        """从自然语言反馈里抠避免用词.

        支持 "避免用X" / "不要说X" / "别用X" / "X" (引号包裹) 这几种写法.
        自然语言提取是 best-effort, 精确控制走 apply_feedback(term=...) API.
        """
        # 常见尾部描述词, "避免用X这个词" → "X"
        _trailing_desc = ("这个词", "这种话", "这些话", "这个词儿", "这种词", "这句话")
        for pat in _AVOID_PATTERNS:
            for m in pat.finditer(feedback):
                term = m.group(1).strip().strip('"\'\u201c\u201d\u2018\u2019')
                for suffix in _trailing_desc:
                    if term.endswith(suffix):
                        term = term[: -len(suffix)]
                        break
                if term and term not in self._profile.avoid_terms:
                    self._profile.avoid_terms.append(term)

    def apply_feedback(
        self,
        term: str | None = None,
        action: str | None = None,
        dimension: str | None = None,
        value: str | None = None,
    ) -> None:
        """显式反馈接口, 给 HTTP API / tool 调用.

        - {term, action="avoid"}: 把 term 加进 avoid_terms
        - {dimension, value}: 手动设某维度, 覆盖学习结果
        """
        with self._lock:
            if term and action == "avoid":
                if term not in self._profile.avoid_terms:
                    self._profile.avoid_terms.append(term)
            if dimension and value:
                self._set_dimension_unlocked(dimension, value)
            self._save_unlocked()

    def set_preference(self, dimension: str, value: str) -> bool:
        """手动设某维度, 覆盖学习结果. 返回是否设置成功."""
        with self._lock:
            ok = self._set_dimension_unlocked(dimension, value)
            if ok:
                self._save_unlocked()
            return ok

    def _set_dimension_unlocked(self, dimension: str, value: str) -> bool:
        """实际设维度逻辑, 调用方必须持锁."""
        if not hasattr(self._profile, dimension):
            return False
        if dimension in _CATEGORICAL_DIMS:
            self._manual_overrides.add(dimension)
            self._scores.pop(dimension, None)
        # avoid_terms 是 list, value 当单个词追加
        if dimension == "avoid_terms":
            if value not in self._profile.avoid_terms:
                self._profile.avoid_terms.append(value)
        else:
            setattr(self._profile, dimension, value)
        return True

    # ── 访问 ────────────────────────────────────────────────────

    def get_profile(self) -> UserStyleProfile:
        """返回当前 profile 的深拷贝, 外部改不影响内部状态."""
        with self._lock:
            return copy.deepcopy(self._profile)

    def get_style_directive(self) -> str:
        """把 profile 拼成 system prompt 片段. confidence=0 时返回空."""
        # flag 关掉时不注入风格指令
        try:
            from huginn.feature_flags import FeatureFlags
            if not FeatureFlags.shared().is_enabled("personalization"):
                return ""
        except Exception:
            # flag 层挂了不能带挂业务, 继续走原逻辑
            pass

        with self._lock:
            p = copy.deepcopy(self._profile)
        if p.confidence <= 0:
            return ""

        lines = ["## 通信风格定制"]
        vocab_hint = {
            "expert": "用 DFT/VASP/ENCUT 等专业术语, 不解释基础概念",
            "technical": "可以用带隙/晶格/优化等技术词, 复杂概念简要说明",
            "intermediate": "用计算/模拟/材料等通用词, 关键术语给一句话解释",
            "beginner": "用日常用语, 避免专业黑话, 必要时打比方",
        }.get(p.vocabulary_level, "")
        lines.append(f"- 用户专业程度: {p.vocabulary_level}（{vocab_hint}）")
        lines.append(f"- 正式程度: {p.formality}")
        verbose_hint = {
            "concise": "直接给答案, 少废话",
            "moderate": "答案 + 必要解释",
            "detailed": "完整推导 + 解释",
        }.get(p.verbosity, "")
        lines.append(f"- 详细程度: {p.verbosity}（{verbose_hint}）")

        lang_hint = {
            "zh": "中文为主",
            "en": "英文为主",
            "mixed": "中英混合",
        }.get(p.language, "中文为主")
        terms_hint = ""
        if p.preferred_terms:
            parts = []
            for zh, form in list(p.preferred_terms.items())[:5]:
                if form == zh:
                    parts.append(f"用'{zh}'不用英文")
                else:
                    parts.append(f"用'{form}'不用'{zh}'")
            terms_hint = ", 专业术语: " + "; ".join(parts)
        lines.append(f"- 语言: {lang_hint}{terms_hint}")

        if p.avoid_terms:
            avoid_str = "、".join(f'"{t}"' for t in p.avoid_terms)
            lines.append(f"- 避免用词: {avoid_str}")

        fmt_hint = {
            "markdown": "markdown",
            "plain": "纯文本",
            "table_heavy": "markdown, 多用表格",
        }.get(p.response_format, "markdown")
        lines.append(f"- 格式: {fmt_hint}")

        if p.code_style == "minimal":
            lines.append("- 代码: 只写关键部分, 不加注释")
        elif p.code_style == "verbose":
            lines.append("- 代码: 完整可跑, 加详细注释")

        return "\n".join(lines)

    def reset(self) -> None:
        """重置 profile 到初始状态, 清掉所有学习结果和手动设置."""
        with self._lock:
            self._scores = {dim: {} for dim in _CATEGORICAL_DIMS}
            self._term_usage = {}
            self._manual_overrides = set()
            self._profile = UserStyleProfile()
            self._save_unlocked()

    # ── 持久化 ──────────────────────────────────────────────────

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        """实际写盘逻辑, 调用方必须持锁. :memory: 模式直接跳过."""
        if self._storage_path == ":memory:":
            return
        try:
            conn = sqlite3.connect(self._storage_path)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS style_profile "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                payload = {
                    "profile": asdict(self._profile),
                    "scores": self._scores,
                    "term_usage": self._term_usage,
                    "manual_overrides": list(self._manual_overrides),
                }
                conn.execute(
                    "INSERT OR REPLACE INTO style_profile (key, value) VALUES (?, ?)",
                    ("default", json.dumps(payload, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.warning("StyleLearner save 失败", exc_info=True)

    def load(self) -> None:
        """从 SQLite 加载 profile. 文件不存在或损坏时保持默认值."""
        if self._storage_path == ":memory:":
            return
        try:
            conn = sqlite3.connect(self._storage_path)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS style_profile "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                row = conn.execute(
                    "SELECT value FROM style_profile WHERE key = ?",
                    ("default",),
                ).fetchone()
            finally:
                conn.close()
            if row is None:
                return
            payload = json.loads(row[0])
            self._profile = UserStyleProfile(**payload["profile"])
            self._scores = payload.get("scores", {})
            for dim in _CATEGORICAL_DIMS:
                self._scores.setdefault(dim, {})
            self._term_usage = payload.get("term_usage", {})
            self._manual_overrides = set(payload.get("manual_overrides", []))
        except Exception:
            logger.warning("StyleLearner load 失败, 用默认 profile", exc_info=True)
