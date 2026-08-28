"""决策启发式蒸馏器 — 从 sobereva 方法一览帖提炼 '场景→方法' 判断规则。

借鉴 distilly 的 decision-heuristics 思路: 专家知识不止是 '是什么',
更值钱的是 '该怎么做' 的场景判断。sobereva 的系列'方法一览'帖
(反应位点预测 / 化学键分析 / 电子激发分析 / 芳香性 / 弱相互作用 …)
每一篇都是结构化的决策规则源: 章节标题 = 场景/方法类别, 正文 = 何时用。

这里做一个轻量结构化抽取器 (无 LLM, 稳定且零成本):
  1. 从帖子正文抓带 '可/用于/衡量/分析/包括' 的章节标题或条目
  2. 把 '场景 → 罗列方法' 的列表抽成一条决策规则
  3. 输出结构化 JSON, 供 seed 预置分发 (agent 检索到即可按场景选方法)

ponytail: 这是标题/条目级抽取, 不做 LLM 语义合并; 对 <500 方法一览帖
规模足够。升级路径: 换 LLM 逐帖语义抽取, 能合并跨帖同场景规则。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DecisionRule:
    """一条 '场景 → 方法' 决策规则."""

    rule_id: str
    scenario: str          # 场景描述, 如 '预测亲电反应位点'
    methods: list[str]     # 该场景可用的方法
    source: str            # 溯源 URL
    note: str = ""         # 注意事项/上下文
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 场景动作词: 只有明确的"分析需求"才构成决策场景;
# 章节标题/普通条目不是需求(如方法名、步骤名、转化格式) — 不抽规则.
_SCENARIO_REQ = re.compile(r"(计算|预测|衡量|评价|判断|分析|研究|考察|识别|表征|判定|估计)")
# 纯"什么是什么/怎么操作"的标题不构成决策场景
_SCENARIO_NON_REQ = re.compile(r"(转化|第一步|第二步|第\d步|关键词|代码|命令$|脚本$|源文件|片段|方法$)")
# 场景动作词: 章节/条目里出现这些往往意味着"该场景用什么方法"
# 行首场景标题: "X反应：" "衡量X的方法：" "X分析" 等
_BULLET_RE = re.compile(r"^[•\-*]\s*(.+)$")
_SECTION_RE = re.compile(r"^#{1,4}\s+(.+)$")


def _clean(text: str) -> str:
    return re.sub(r"[*#`>]", "", text).strip()


def _is_scenario_line(text: str) -> bool:
    """判断是否构成'决策场景': 明确的分析需求(计算/预测/衡量...),
    且不是单纯的操作/步骤标题. 无需求词或属步骤标题都不算场景."""
    if not text or len(text) > 25:
        return False
    if _SCENARIO_NON_REQ.search(text):
        return False
    return bool(_SCENARIO_REQ.search(text))


def _extract_methods(follow: list[str]) -> list[str]:
    """从场景行后续行收集方法名: 逗号/、分隔的短词, 去掉无信息项."""
    out: list[str] = []
    for line in follow:
        line = line.strip()
        if not line or line.startswith(("#", "-"), 0, 1) and False:
            continue
        # 去掉常见噪音词
        line_clean = re.sub(r"（.*?）|\(.*?\)", "", line)
        parts = re.split(r"[,，、;；]", line_clean)
        for p in parts:
            p = _clean(p)
            if not p or len(p) > 30:
                continue
            if _is_noise_word(p):
                continue
            out.append(p)
    return out


# 决策规则方法白名单: 只有业界/sobereva 公认的方法词才算有效方法.
# 白名单外的一律不算, 避免把章节描述/操作步骤/格式名当方法.
_METHOD_WHITELIST = (
    # 波函数/量子化学分析
    "福井函数", "双描述符", "ALIE", "LEAE", "静电势", "ESP", "NCI", "NTO", "NICS",
    "Hirshfeld", "IRI", "IGMH", "mIGM", "amIGM", "IGM", "AdNDP", "ELF", "LOL",
    "键级", "Mulliken", "Schmidt", "NBO", "TDDFT", "IFCT", "软化度", "亲电", "亲核",
    "空穴", "电子", "多中心键", "BLA", "BOD", "NAdO", "ETS", "电荷分解", "轨道",
    "键临界点", "AIM", "ESP面", "Fukui", "QTAIM", "DORI", "DB6", "mayer", "Wiberg",
    # DFT 泛函 / 基组
    "PBE0", "PBE", "B3LYP", "M06-2X", "M06-2X", "r2SCAN", "SCAN", "TPSS", "TPSS-h",
    "MN15", "wB97", "CAM", "Gaussian", "ORCA", "CP2K", "xTB", "GROMACS", "AMBER",
    "NWChem", "Molpro", "Psi4", "VASP", "DFT", "DZVP", "def2", "cc-pVDZ", "aug-cc",
)


def _is_asc_rule_ok(rule: DecisionRule) -> bool:
    """规则有效性: 场景是明确分析需求, 且至少一个方法命中白名单."""
    if not _SCENARIO_REQ.search(rule.scenario) and not re.match(r"^(亲|亲核|自由基|谐|色)", rule.scenario):
        return False
    return any(any(w in m for w in _METHOD_WHITELIST) for m in rule.methods)


def _is_noise_word(w: str) -> bool:
    return w.lower() in {"方法", "包括", "包括：", "如下", "以及", "以及\n", "等等"}


def extract_rules(post_text: str, source_url: str) -> list[DecisionRule]:
    """从一篇方法一览帖抽取决策规则."""
    lines = [line.rstrip() for line in post_text.splitlines()]
    # 跳过 YAML frontmatter (首行 --- 到下一个 ---)
    start_body = 0
    if lines and lines[0].strip() == "---":
        for k in range(1, len(lines)):
            if lines[k].strip() == "---":
                start_body = k + 1
                break
    rules: list[DecisionRule] = []
    seen: set[tuple[str, ...]] = set()

    # 场景行两种形态:
    #  A) 归纳列表 "• 亲电反应：方法A、方法B" (冒号同行 + 后续续行)
    #  B) 章节标题 "### 2.1 福井函数" 本身 = 方法, 归到父场景
    pending_scenario: str | None = None  # 当前进入的子场景(用于 B 形态)

    i = start_body
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith(("HTTP", "http", "http", "<", "**PS")):
            i += 1
            continue

        # 形态 A: "场景：方法"
        if line.startswith("•") or line.startswith("- ") or line.startswith("* "):
            content = _clean(line.lstrip("•-* "))
            if "：" in content or ":" in content:
                scenario, _, rest = content.partition("：") if "：" in content else content.partition(":")
                scenario = scenario.strip()
                # 局限章节的"因素"是反例(什么会不准), 不是决策场景, 跳过
                if re.search(r"(效应|因素|扭曲|局限|不足|注意)$", scenario):
                    pending_scenario = None
                    i += 1
                    continue
                # 冒号前是短标签(1-12字)就算场景, 不必含动作词 —
                # 像 '亲电反应：福井函数、双描述符' 这类归纳条目动作词在冒号后.
                if 1 <= len(scenario) <= 12:
                    # 收集同场景方法 (冒号后 + 后续续行, 直到空行/新场景)
                    methods = _collect_methods(rest, lines, i + 1)
                    if methods:
                        key = (scenario, tuple(methods))
                        if key not in seen:
                            seen.add(key)
                            rules.append(DecisionRule(
                                rule_id=_rule_id(scenario, methods),
                                scenario=scenario, methods=methods,
                                source=source_url, tags=["heuristic", "sobko"],
                            ))
                    pending_scenario = scenario
                elif _is_scenario_line(scenario):
                    pending_scenario = scenario
                else:
                    pending_scenario = None
            else:
                pending_scenario = None
            i += 1
            continue

        # 形态 B: 章节标题 -> 归入当前 pending_scenario (若在归纳区)
        s = _SECTION_RE.match(line)
        if s:
            sec = _clean(s.group(1))
            # 剔除编号前缀 "2.1 "
            sec_name = re.sub(r"^\d+(\.\d+)*\s*", "", sec)
            # 新的大场景: 章节标题含动作词, 作为独立规则(单方法)
            if _is_scenario_line(sec_name) and pending_scenario is None:
                pending_scenario = sec_name
            else:
                # 章节名本身可能是方法, 但不是"场景"; 不单独成规则
                pass
        i += 1
    return rules


def _collect_methods(first: str, lines: list[str], start: int) -> list[str]:
    """从'场景：'冒号后的首段 + 后续续行收集方法名."""
    raw = [first]
    j = start
    while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith("•") and not lines[j].lstrip().startswith("- "):
        raw.append(lines[j].strip())
        j += 1
    joined = "".join(raw)
    # 去掉括号内说明, 专注方法名词
    joined = re.sub(r"（.*?）|\(.*?\)", "", joined)
    parts = re.split(r"[,，、;；\n]", joined)
    methods = []
    for p in parts:
        p = _clean(p)
        # 方法名是短名词短语(2-12字). 排除: 过短 / 过长 / 含日常动词/连接的句子碎片.
        if not p or len(p) > 12 or len(p) < 2 or _is_noise_word(p):
            continue
        if not _is_plausible_method(p):
            continue
        methods.append(p)
    return methods[:8]  # 限制条数防过宽


def _is_plausible_method(text: str) -> bool:
    """方法名应为名词短语; 排除含谓词/日常口语连接词的正文句子碎片.

    例如 '用鼠标左键拖动' / '是极小基' / '记录了程序的源代码' 这类
    是描述句, 不是方法名, 过滤掉. 真正的像 '福井函数' / '双描述符' /
    'M06-2X' / 'ALIE' 都通过. ponytail: 关键词黑名单够用, 不引入 POS tagger.
    """
    return not any(k in text for k in ("鼠标", "然后", "但是", "因为", "如果", "使得",
                                       "用", "让", "是指", "这是", "那是", "记录了",
                                       "告诉", "比如", "强烈", "可选", "失效", "推荐",
                                       "当手头"))


def _rule_id(scenario: str, methods: list[str]) -> str:
    seed = f"{scenario}|{','.join(methods)}"
    return f"dec_{abs(hash(seed)) & 0xffffffff:08x}"


def distill_posts(posts_dir: Path, out_path: Path) -> tuple[int, list[DecisionRule]]:
    """扫描 posts 目录下所有 index.md, 抽决策规则并写出 JSONL.

    Returns: (写入条数, 全部规则).
    """
    all_rules: list[DecisionRule] = []
    for md in sorted(posts_dir.glob("**/index.md")):
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        m = re.search(r"^url:\s*(.+)$", text, re.M)
        url = m.group(1).strip() if m else md.parent.name
        try:
            rules = extract_rules(text, url)
        except Exception:
            continue
        # 决策规则质量门槛: 只保留"明确需求 -> 白名单方法"的规则
        rules = [r for r in rules if _is_asc_rule_ok(r)]
        # 方法列表也剪到白名单命中项, 输出更精确
        for r in rules:
            r.methods = [
                m for m in r.methods
                if any(w in m for w in _METHOD_WHITELIST)
            ][:8]
        all_rules.extend(rules)

    # 全局去重 (同场景近似方法合并)
    uniq: dict[tuple[str, ...], DecisionRule] = {}
    for r in all_rules:
        key = (r.scenario, tuple(sorted(r.methods)))
        uniq.setdefault(key, r)
    final = list(uniq.values())

    with out_path.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(final), final
