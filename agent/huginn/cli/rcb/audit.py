"""rcb/audit.py: RCB 评测裁决纯函数族.

从 rcb_runner.py 剥离出的「评测裁决」逻辑 — 这些函数不依赖 llama/agent 状态,
只做机械比对/统计/门控判定, 是 RCB 评测专属的合规裁决层, 与 agent 通用主循环解耦.

本模块顶层不依赖 rcb_runner, 可被任意顺序 import, 无循环依赖.
rcb_runner 通过 `from huginn.cli.rcb.audit import ...` 引用, 并 re-export 保持向后兼容.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from huginn.utils.runtime import HUGINN_DIR_NAME


def _rcb_drift_check(evals_history: list) -> tuple[bool, str]:
    """v16: RCB 专用 drift 检查 — window=2, unsure 也算 drift 信号.

    旧 detect_drift window=3 且只认 on_track=false, Math_003 案例 4 轮全 unsure
    但 window 不触发, agent 一路跑到 TASK COMPLETE.

    ponytail: 配合 evidence_quality — 2 步 unsure 且至少 1 步 evidence=low
    才触发, 减误报. 升级路径: 加 task_metrics 加权.
    """
    if len(evals_history) < 2:
        return False, ""
    last_two = evals_history[-2:]
    if all(getattr(e, "on_track", "") in ("false", "unsure") for e in last_two):
        ev_low = any(
            getattr(e, "evidence_quality", "") in ("low", "") for e in last_two
        )
        if ev_low:
            return True, (
                f"RCB drift: 2 consecutive unsure/false with low evidence "
                f"(iter {len(evals_history)-1}, {len(evals_history)})"
            )
    return False, ""


def _extract_exact_components(checklist: str) -> list[str]:
    """从 checklist 文本抽 [EXACT] 标记的组件名.

    ponytail: 纯正则, 不调 LLM — 机械比对的前提是规则确定.
    匹配 '[EXACT]' 后到行尾/分号/句号的文本, strip 后作组件名.
    升级路径: Step 1 直接输出结构化 JSON checklist 时换 parser.
    """
    if not checklist:
        return []
    out = []
    seen = set()
    for m in re.finditer(r"\[EXACT\]\s*([^\n;]+)", checklist):
        name = m.group(1).strip().rstrip(".,;:")
        if not name or len(name) > 120:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _scan_implementation_traces(ws: Path, components: list[str]) -> dict:
    """扫 ws 下产物文件 (.py/.md/.json/.txt/.sh/.yaml), 检查每个 [EXACT] 组件是否出现.

    ponytail: 子串匹配 + case-insensitive. 简单但够用 — RCB 任务的 [EXACT] 组件
    名通常是显式术语 (e.g. 'GVAE encoder', 'C2ST classifier'), 在实现里会留下痕迹.
    升级路径: 用 AST 解析 code/*.py 抽函数/类名做更精确匹配.
    返回 {component_name: bool}.
    """
    if not components:
        return {}
    exts = {".py", ".md", ".json", ".txt", ".sh", ".yaml", ".yml"}
    corpus_parts = []
    for ext in exts:
        for p in ws.rglob(f"*{ext}"):
            # 跳过 .huginn/ 内部 trace/cache — 那是观测不是产物
            if ".huginn" in p.parts:
                continue
            try:
                corpus_parts.append(
                    p.read_text(encoding="utf-8", errors="ignore").lower()
                )
            except OSError:
                continue
    corpus = "\n".join(corpus_parts)
    return {c: (c.lower() in corpus) for c in components}


def _parse_substitute_headers(report_md: Path) -> list[dict]:
    """解析 report.md 顶部 METHOD SUBSTITUTE 声明.

    返回 [{replaced, reason, raw}, ...]. 约定 header 形如:
        METHOD SUBSTITUTE: <X> replaced <Y> because <reason>
    ponytail: 只扫 report.md 前 50 行 — header 应在顶部, 全文搜易误匹配正文.
    """
    if not report_md.exists():
        return []
    try:
        head = report_md.read_text(encoding="utf-8", errors="ignore").splitlines()[:50]
    except OSError:
        return []
    pat = re.compile(
        r"METHOD\s+SUBSTITUTE:\s*(.+?)\s+replaced\s+(.+?)\s+because\s+(.+)",
        re.IGNORECASE,
    )
    out = []
    for line in head:
        m = pat.search(line)
        if m:
            out.append(
                {
                    "replaced": m.group(1).strip(),
                    "reason": m.group(3).strip(),
                    "raw": line.strip(),
                }
            )
    return out


def _count_failed_attempts(ws: Path, evals_history: list, component: str) -> int:
    """统计 [EXACT] 组件的失败尝试次数.

    ponytail: 两个来源取最大值 —
      (a) evals_history 里 on_track=false 且 attempted 文本含组件名;
      (b) .huginn/meta_trace.jsonl 里 on_track=false 行的 attempted 含组件名.
    升级路径: 用 LLM 读 attempted 文本判语义相关性, 而非子串匹配.
    """
    key = component.lower()
    n = 0
    # (a) in-memory evals
    for ev in evals_history or []:
        try:
            on_track = str(getattr(ev, "on_track", "")).lower()
            attempted = str(getattr(ev, "attempted", "") or "").lower()
            if on_track == "false" and key in attempted:
                n += 1
        except Exception:
            continue
    # (b) on-disk trace — resume/跨进程场景 evals_history 未必含全部历史
    trace_path = ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
    if trace_path.exists():
        try:
            with trace_path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if (
                        str(entry.get("on_track", "")).lower() == "false"
                        and key in str(entry.get("attempted", "") or "").lower()
                    ):
                        n += 1
        except OSError:
            pass
    return n


async def _step2_substitution_audit(
    ws: Path,
    checklist: str,
    evals_history: list,
    stream_chat_fn,
    *,
    max_remediate: int = 1,
    variant_min_failures: int = 2,
) -> dict:
    """A3: silent substitution 结构性拦截.

    Step-2 结束机械比对「[EXACT] 组件 ↔ code/实现痕迹」, 缺失即回退执行.
    禁止未尝试标 [VARIANT] — ≥2 次失败 + 报错才允许降级.

    返回 audit 报告 dict:
      {exact_components, missing, substitutions, variant_blocked, remediated,
       unresolved, raw_log}
    ponytail: 不调 LLM 做语义判断 — 路线图 N5 明确「不再堆 prompt 级规劝」,
    机械比对的意义就是规则确定不可被 LLM 说服. 回退执行用一次 chat 注入强提示,
    失败也不循环 (max_remediate=1).
    """
    log = []
    components = _extract_exact_components(checklist)
    log.append(f"extracted {len(components)} [EXACT] components")
    if not components:
        return {
            "exact_components": [],
            "missing": [],
            "substitutions": [],
            "variant_blocked": [],
            "remediated": [],
            "unresolved": [],
            "raw_log": log,
        }

    traces = _scan_implementation_traces(ws, components)
    report_md = ws / "report" / "report.md"
    subs = _parse_substitute_headers(report_md)
    sub_names = {s["replaced"].lower() for s in subs}

    missing = []
    variant_blocked = []
    for name, found in traces.items():
        if found:
            continue
        # 缺失: 已有 SUBSTITUTE header 记录则视为合规降级
        if name.lower() in sub_names:
            continue
        # 无 SUBSTITUTE — 检查失败次数是否达 variant 阈值
        n_fail = _count_failed_attempts(ws, evals_history, name)
        if n_fail >= variant_min_failures:
            # 达到降级阈值, 但仍未声明 SUBSTITUTE — 提示补声明
            variant_blocked.append(
                {
                    "component": name,
                    "failures": n_fail,
                    "issue": "达到降级阈值但未声明 METHOD SUBSTITUTE",
                }
            )
            continue
        # 未尝试就缺失 — silent substitution, 必须回退执行
        missing.append({"component": name, "failures": n_fail})

    log.append(
        f"missing={len(missing)} variant_blocked={len(variant_blocked)} subs={len(subs)}"
    )
    if not missing and not variant_blocked:
        return {
            "exact_components": components,
            "missing": [],
            "substitutions": subs,
            "variant_blocked": [],
            "remediated": [],
            "unresolved": [],
            "raw_log": log,
        }

    # 回退执行: 一次性注入强提示. ponytail: max_remediate=1 防无限循环.
    remediated = []
    unresolved = list(missing) + list(variant_blocked)
    for _ in range(max_remediate):
        if not unresolved:
            break
        prompt = (
            "STEP-2 SUBSTITUTION AUDIT FAILED — 以下 [EXACT] 组件既无实现痕迹, "
            "也未在 report/report.md 顶部声明 METHOD SUBSTITUTE:\n"
        )
        for item in unresolved:
            tag = "MISSING" if item in missing else "VARIANT_BLOCKED"
            prompt += (
                f"  [{tag}] {item['component']} (失败 {item.get('failures', 0)} 次)\n"
            )
        prompt += (
            "\n这是结构性拦截 — 不允许 silent substitution.\n"
            "对每个组件, 必须 EITHER:\n"
            "  (a) 用 code_tool / bash_tool 实际实现并跑通 (产物文件里留下痕迹); OR\n"
            "  (b) 在 report/report.md 顶部添加 header 行:\n"
            "      'METHOD SUBSTITUTE: <组件名> replaced <替代方案> because <原因 + ≥2 次失败的报错摘要>'\n"
            "未尝试 (失败 0 次) 的组件不允许标 VARIANT — 先实际尝试.\n"
            "现在补做或补声明. 这是最后一次回退执行机会."
        )
        log.append(f"remediate attempt: {len(unresolved)} items")
        try:
            await stream_chat_fn(prompt, "step2_audit_remediate")
        except Exception as e:
            log.append(f"remediate chat failed: {e}")
            break
        # 重扫
        traces = _scan_implementation_traces(ws, [it["component"] for it in unresolved])
        report_md = ws / "report" / "report.md"
        subs = _parse_substitute_headers(report_md)
        sub_names = {s["replaced"].lower() for s in subs}
        still = []
        for item in unresolved:
            name = item["component"]
            if traces.get(name, False) or name.lower() in sub_names:
                remediated.append(name)
            else:
                still.append(item)
        unresolved = still
        log.append(
            f"after remediate: remediated={len(remediated)} unresolved={len(unresolved)}"
        )
        if not unresolved:
            break

    return {
        "exact_components": components,
        "missing": missing,
        "substitutions": subs,
        "variant_blocked": variant_blocked,
        "remediated": remediated,
        "unresolved": unresolved,
        "raw_log": log,
    }


# A2: 产物级门控 — 路线图 P1-A2 / 12 报告 P1-1.
# ResearchClaw remediation task 最小实现: outputs/ 无真实 metrics 文件时
# 禁止虚写 Results, 触发 blocker remediate task.
_PLACEHOLDER_TOKENS = (
    "expected",
    "todo",
    "placeholder",
    "tbd",
    "n/a",
    "not implemented",
)


def _scan_real_metrics(ws: Path) -> list[Path]:
    """扫 outputs/ 下真实 metrics 文件 (非空 + 非占位).

    ponytail: 扩展名白名单 (.json/.csv/.npy/.txt/.yaml) + 大小 > 0 +
    内容不含 placeholder token (case-insensitive 子串). 二进制 (.npy) 只查大小.
    升级路径: 用 schema 校验 JSON 字段是否含数值列, 而非子串过滤.
    """
    out_dir = ws / "outputs"
    if not out_dir.exists():
        return []
    out: list[Path] = []
    for p in out_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".json", ".csv", ".npy", ".txt", ".yaml", ".yml"):
            continue
        try:
            if p.stat().st_size == 0:
                continue
        except OSError:
            continue
        if p.suffix.lower() == ".npy":
            out.append(p)
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        # 占位文件 (整文件只含 placeholder token) 不算真实 metrics
        stripped = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        if any(tok in stripped for tok in _PLACEHOLDER_TOKENS) and len(stripped) < 200:
            # 短文件 + 含占位 token = 占位文件; 长文件含 token 可能是正常叙述
            continue
        out.append(p)
    return out


# P1-B4: report.md 数值标记 lint — 检测未标 [EXECUTED]/[EXPECTED]/[NOT EXECUTED] 的数值
_B4_NUMERIC_RE = re.compile(r"\b\d+\.?\d*(?:[eE][-+]?\d+)?\b")
_B4_MARKERS = ("[EXECUTED]", "[EXPECTED]", "[NOT EXECUTED]")


def _lint_report_markers(report_path: Path) -> dict:
    """B4: 扫描 report.md, 统计数值声明的标记情况.

    返回 {total_numbers, tagged, untagged, untagged_samples, marker_counts}.
    ponytail: 句子级扫描 — 数值所在句子含 marker 即算 tagged.
      ceiling: 不区分 marker 是否真实 (agent 可能瞎标), 只检测存在性.
      升级路径: 跟 outputs/ 文件交叉验证 [EXECUTED] 数值是否真有产物支撑.
    """
    if not report_path.exists():
        return {
            "total_numbers": 0,
            "tagged": 0,
            "untagged": 0,
            "untagged_samples": [],
            "marker_counts": {},
        }
    try:
        text = report_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {
            "total_numbers": 0,
            "tagged": 0,
            "untagged": 0,
            "untagged_samples": [],
            "marker_counts": {},
        }
    # 按行扫描 (句子级太粗, 行级够用)
    lines = text.splitlines()
    total = 0
    tagged = 0
    untagged_samples: list[str] = []
    marker_counts = dict.fromkeys(_B4_MARKERS, 0)
    for line in lines:
        nums = _B4_NUMERIC_RE.findall(line)
        if not nums:
            continue
        # 排除纯结构行 (markdown 表头/分隔符/列表标记)
        stripped = line.strip()
        if stripped.startswith(("|", "---", "##", "#", "- ", "* ")):
            # 表格行和标题行的数值仍要检查, 但列表标记行宽松
            pass
        # 排除年份/版本号 (2007, v1.0) — 启发式: 4 位数字 1900-2099
        real_nums = [
            n
            for n in nums
            if not (len(n) == 4 and n.isdigit() and 1900 <= int(n) <= 2099)
        ]
        if not real_nums:
            continue
        total += len(real_nums)
        line_upper = line.upper()
        has_marker = any(m in line_upper for m in _B4_MARKERS)
        if has_marker:
            tagged += len(real_nums)
            for m in _B4_MARKERS:
                if m in line_upper:
                    marker_counts[m] += 1
        else:
            if len(untagged_samples) < 5:
                untagged_samples.append(stripped[:120])
    return {
        "total_numbers": total,
        "tagged": tagged,
        "untagged": total - tagged,
        "untagged_samples": untagged_samples,
        "marker_counts": marker_counts,
    }


async def _step2_outputs_gate(
    ws: Path,
    stream_chat_fn,
    *,
    max_remediate: int = 1,
) -> dict:
    """A2: 产物级门控 — outputs/ 无真实 metrics 文件时禁止虚写 Results.

    返回 {has_real_metrics, metrics_files, remediated, blocker, raw_log}.
    ponytail: 不调 LLM 做语义判断 — 纯文件存在性 + 占位 token 子串过滤.
    blocker=True 时 Step 3 应降权 Results claim 并标注「无产物支撑」.
    """
    log = []
    metrics = _scan_real_metrics(ws)
    log.append(f"initial metrics files: {len(metrics)}")
    if metrics:
        return {
            "has_real_metrics": True,
            "metrics_files": [str(p) for p in metrics],
            "remediated": False,
            "blocker": False,
            "raw_log": log,
        }

    # 触发 blocker remediate — ResearchClaw 风格的 remediation task.
    remediated = False
    for _ in range(max_remediate):
        log.append("triggering outputs remediate (blocker task)")
        prompt = (
            "STEP-2 OUTPUTS GATE FAILED — outputs/ 目录无真实 metrics 文件.\n"
            "禁止虚写 Results / Discussion — 没有产物支撑的数值声明将被 Step 3 降权.\n\n"
            "现在必须 EITHER:\n"
            "  (a) 用 code_tool / bash_tool 实际跑一次实验, 把结果写入 outputs/*.json "
            '(至少包含一个数值字段, e.g. {"loss": 0.5, "rmse": 0.1}); OR\n'
            "  (b) 若任务确实无法执行 (e.g. 数据缺失/模型太大), 在 report/report.md 顶部添加:\n"
            "      'EXECUTION BLOCKER: <原因>'\n"
            '      并在 outputs/blocker.json 写 {"reason": "...", "attempted": [...]}.\n'
            "诚实失败 > 沉默虚写. 现在补做或声明 blocker."
        )
        try:
            await stream_chat_fn(prompt, "step2_outputs_gate_remediate")
        except Exception as e:
            log.append(f"remediate chat failed: {e}")
            break
        metrics = _scan_real_metrics(ws)
        log.append(f"after remediate: {len(metrics)} metrics files")
        if metrics:
            remediated = True
            break

    blocker = not bool(metrics)
    return {
        "has_real_metrics": bool(metrics),
        "metrics_files": [str(p) for p in metrics],
        "remediated": remediated,
        "blocker": blocker,
        "raw_log": log,
    }


# --- 主循环决策函数 (从 rcb_runner.py 剥离) -----------------------------------

def _should_retry_execute(
    verdict: str,
    beta_1: int,
    gap_type: str,
) -> bool:
    """Step3→Step2 回退触发判断 (v14 拓扑许可).

    拓扑许可: β_1>0 (Meta-Trace 存在循环回退路径) 才允许回退.
    gap 类型: numeric_recompute / exact_component_missing 才回退,
              text_description 不回退 (文字补完在 Step 3 内 OVERWRITE report.md 即可).
    verdict: fix_needed 和 fail 都允许回退 — fail + 具体 gap 说明 critique
             找到了可修问题, 放弃重试等于 0 分, 重试至少有机会.
    """
    if verdict not in ("fix_needed", "fail"):
        return False
    if beta_1 <= 0:
        return False
    return gap_type in ("numeric_recompute", "exact_component_missing")


def _derive_gap_type(object_verdict: dict) -> str:
    """从 adversarial_critique (object mode) dict 推断 gap_type.

    object mode 不直接返回 gap_type (只有 critique_decision 的 CritiqueResult 才有),
    按 red flag 类型反推: implausible/recomputed → numeric_recompute,
    substitution/missing → exact_component_missing, 否则 fix_needed → text_description.
    ponytail: 规则推断是廉价代理, 升级路径是 LLM 在 object mode 也直接返回 gap_type.
    """
    if not object_verdict:
        return "none"
    if object_verdict.get("recomputed_red_flags") or object_verdict.get("implausible_metrics"):
        return "numeric_recompute"
    if object_verdict.get("silent_substitutions") or object_verdict.get("missing_components"):
        return "exact_component_missing"
    if object_verdict.get("overall_verdict") == "fix_needed":
        return "text_description"
    return "none"


def _infer_beta_1_simple(ws: Path) -> int:
    """β_1 简易推断 — 数 meta_trace.jsonl 行数.

    ponytail: 真正的 β_1 计算在 v14 Task 4 (networkx cycle_basis), 未实现前用
    'trace 已有 ≥3 条 entry 则视为存在循环路径' 的代理. 二值返回, 不假装算精确值.
    升级路径: 接入 trace_topology.compute_betti 后替换.
    """
    _trace = ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
    if not _trace.exists():
        return 0
    try:
        with _trace.open(encoding="utf-8") as _f:
            _n = sum(1 for _line in _f if _line.strip())
    except Exception:
        return 0
    return 1 if _n >= 3 else 0


def _write_directive_rejection(
    ws: Path, gap_type: str, verdict: str, retry_count: int,
) -> None:
    """回退上限触发 — 写 directive_rejections.jsonl.

    spec §"回退次数上限": retry 2 次仍 fix_needed 时强制 finalize 并留痕.
    """
    import time as _t
    _rej_path = ws / HUGINN_DIR_NAME / "directive_rejections.jsonl"
    _rej_path.parent.mkdir(parents=True, exist_ok=True)
    _entry = {
        "ts": _t.time(),
        "reason": "step3_retry_limit_reached",
        "retry_count": retry_count,
        "final_verdict": verdict,
        "gap_type": gap_type,
    }
    with _rej_path.open("a", encoding="utf-8") as _f:
        _f.write(json.dumps(_entry, ensure_ascii=False) + "\n")


# G28: parse MAE/R2/RMSE/accuracy claims from report.md, compare to outputs/.
# flag >10% deviation, breaks LLM critique circular reasoning.
# ponytail: regex extract, no LLM. ceiling: semantic parse needs LLM.
_METRIC_RE = re.compile(
    r"\b(MAE|RMSE|R2|R²|MSE|accuracy|loss|F1|AUC|RMS)\b\s*[:=]\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def _recompute_report_metrics(report_text: str, ws: Path) -> list[dict]:
    """Compare report claimed metrics vs outputs/ actual, return >10% deviation flags."""
    flags: list[dict] = []
    claimed = {m.group(1).upper(): float(m.group(2))
               for m in _METRIC_RE.finditer(report_text)}
    if not claimed:
        return flags
    outputs_dir = ws / "outputs"
    if not outputs_dir.exists():
        return flags
    actual: dict[str, float] = {}
    for f in outputs_dir.rglob("*"):
        if not f.is_file() or f.suffix not in (".txt", ".json", ".csv", ".md"):
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            for m in _METRIC_RE.finditer(txt):
                k = m.group(1).upper()
                if k not in actual:
                    actual[k] = float(m.group(2))
        except Exception:
            continue
    for k, claim_val in claimed.items():
        if k not in actual:
            continue
        ref = actual[k]
        if abs(ref) < 1e-12:
            continue
        dev = abs(claim_val - ref) / abs(ref)
        if dev > 0.10:
            flags.append({
                "metric": k, "claimed": claim_val, "actual": ref,
                "deviation_pct": round(dev * 100, 1),
            })
    return flags
