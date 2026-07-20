"""Sanity gate — 机械检查 outputs/ 防止占位式执行.

治 PaperBench pinn 的"假执行"问题: agent 跑了 0.117s 的假训练,
final_loss 小数点后 16 位完全相同 (L-BFGS 没跑但 agent 没察觉),
照写 README 宣称完整扫描. step_verifier 重工具前缀不含 code/bash_tool,
对这种假执行完全无感.

4 项机械检查:
  float_dedup       多个 final_loss/loss 值 16 位精度重复 → fail
  training_time     training_time 字段 < 1.0s → fail
  loss_monotone     loss 曲线无单调下降区间 → fail
  placeholder_text  含 "placeholder"/"Expected"/"TODO"/"dummy" 字样 → fail

接入点: BenchmarkOrchestrator.run() 每轮 chat() 后, deliverable 全齐时调.
fail → 注入修复指令, 不放行 _is_done.

ponytail: 只做机械字符串/数值检查, 不调 LLM. 升级路径: 接领域断言表
(loss 范围 / L2RE 阈值 / C2ST 区间, 见 analysis_20260717/15:166).
天花板: 只扫 outputs/*.json, 不扫 .csv/.txt/.log. 不懂语义.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 假执行嫌疑字段
_LOSS_FIELDS = ("final_loss", "loss", "train_loss", "val_loss", "mse", "mae")
_TIME_FIELDS = ("training_time", "train_time", "runtime", "elapsed")
_PLACEHOLDER_MARKERS = ("placeholder", "Expected", "TODO", "dummy", "stub", "not_implemented")

# 单调下降区间最小长度 (loss 曲线至少连续 3 点下降)
_MIN_MONOTONE_LEN = 3


def _collect_loss_values(data: Any) -> list[float]:
    """递归收集 dict/list 里所有 loss-like 字段的 float 值."""
    out: list[float] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in _LOSS_FIELDS and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out.extend(_collect_loss_values(v))
    elif isinstance(data, list):
        for item in data:
            out.extend(_collect_loss_values(item))
    return out


def _collect_time_values(data: Any) -> list[float]:
    """递归收集 training_time-like 字段."""
    out: list[float] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in _TIME_FIELDS and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out.extend(_collect_time_values(v))
    elif isinstance(data, list):
        for item in data:
            out.extend(_collect_time_values(item))
    return out


def _collect_loss_curves(data: Any) -> list[list[float]]:
    """收集 loss 曲线 (list of floats, 长度 >= _MIN_MONOTONE_LEN)."""
    out: list[list[float]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in _LOSS_FIELDS and isinstance(v, list):
                nums = [x for x in v if isinstance(x, (int, float))]
                if len(nums) >= _MIN_MONOTONE_LEN:
                    out.append([float(x) for x in nums])
            else:
                out.extend(_collect_loss_curves(v))
    elif isinstance(data, list):
        for item in data:
            out.extend(_collect_loss_curves(item))
    return out


def _has_monotone_decrease(curve: list[float]) -> bool:
    """曲线是否有连续 _MIN_MONOTONE_LEN 个点严格下降."""
    if len(curve) < _MIN_MONOTONE_LEN:
        return False
    run = 0
    for i in range(1, len(curve)):
        if curve[i] < curve[i - 1]:
            run += 1
            if run >= _MIN_MONOTONE_LEN - 1:
                return True
        else:
            run = 0
    return False


def _float_dedup(values: list[float]) -> bool:
    """多个 loss 值在 16 位精度重复 → True (有重复)."""
    if len(values) < 2:
        return False
    # 16 位精度字符串化
    strs = [f"{v:.16g}" for v in values]
    return len(set(strs)) < len(strs)


def _check_placeholder_text(text: str) -> bool:
    """文本含 placeholder 标记 → True."""
    lower = text.lower()
    return any(marker.lower() in lower for marker in _PLACEHOLDER_MARKERS)


def check_sanity(workspace: Path | str) -> dict[str, Any]:
    """机械检查 workspace/submission/outputs/*.json.

    返回:
        {"passed": bool, "reason": str, "fix_prompt": str, "checks": {...}}
        passed=True: 通过所有检查 (或无 outputs 可查)
        passed=False: 有占位式执行嫌疑, fix_prompt 给修复指令
    """
    ws = Path(workspace)
    outputs_dir = ws / "submission" / "outputs"
    if not outputs_dir.exists():
        return {
            "passed": True,
            "reason": "no outputs/ dir",
            "fix_prompt": "",
            "checks": {},
        }

    json_files = list(outputs_dir.glob("*.json"))
    if not json_files:
        return {
            "passed": True,
            "reason": "no .json in outputs/",
            "fix_prompt": "",
            "checks": {},
        }

    checks: dict[str, Any] = {}
    all_losses: list[float] = []
    all_times: list[float] = []
    all_curves: list[list[float]] = []
    placeholder_hits: list[str] = []

    for jf in json_files:
        try:
            raw = jf.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw)
        except Exception:
            continue

        # placeholder 文本检查 (扫原始 JSON 字符串)
        if _check_placeholder_text(raw):
            placeholder_hits.append(jf.name)

        all_losses.extend(_collect_loss_values(data))
        all_times.extend(_collect_time_values(data))
        all_curves.extend(_collect_loss_curves(data))

    # 1. float dedup
    checks["float_dedup"] = {
        "passed": not _float_dedup(all_losses),
        "n_values": len(all_losses),
    }

    # 2. training_time < 1.0s
    short_times = [t for t in all_times if t < 1.0]
    checks["training_time"] = {
        "passed": len(short_times) == 0,
        "n_short": len(short_times),
        "n_total": len(all_times),
    }

    # 3. loss monotone (至少一条曲线有单调下降区间)
    has_monotone = any(_has_monotone_decrease(c) for c in all_curves)
    checks["loss_monotone"] = {
        "passed": has_monotone or len(all_curves) == 0,
        "n_curves": len(all_curves),
    }

    # 4. placeholder text
    checks["placeholder_text"] = {
        "passed": len(placeholder_hits) == 0,
        "hits": placeholder_hits,
    }

    # 汇总
    failed = [name for name, r in checks.items() if not r["passed"]]
    if not failed:
        return {
            "passed": True,
            "reason": "all checks passed",
            "fix_prompt": "",
            "checks": checks,
        }

    # [check_name] 前缀让 reason 可解析 (selfcheck / orchestrator 注入时都能 grep)
    reason_parts = []
    if "float_dedup" in failed:
        reason_parts.append(
            f"[float_dedup] 多个 loss 值在 16 位精度重复 ({checks['float_dedup']['n_values']} 个值)"
        )
    if "training_time" in failed:
        reason_parts.append(
            f"[training_time] {checks['training_time']['n_short']}/{checks['training_time']['n_total']} 个 training_time < 1.0s"
        )
    if "loss_monotone" in failed:
        reason_parts.append(
            f"[loss_monotone] {checks['loss_monotone']['n_curves']} 条 loss 曲线均无单调下降区间"
        )
    if "placeholder_text" in failed:
        reason_parts.append(
            f"[placeholder_text] 文件含 placeholder/Expected/TODO 字样: {checks['placeholder_text']['hits']}"
        )

    reason = "; ".join(reason_parts)
    fix_prompt = (
        "SANITY GATE FAIL: 检测到占位式执行嫌疑. " + reason + ".\n\n"
        "Do this NOW:\n"
        "1. 打开 outputs/*.json, 检查数值是否真实 (非重复/非占位)\n"
        "2. 如果 training_time < 1.0s, 说明训练没真跑 — 增加 epochs/iters 重跑\n"
        "3. 如果 loss 值重复, 说明多个配置没真跑 (只跑了一次复制) — 分别跑每个配置\n"
        "4. 删除所有 placeholder/Expected/TODO 字样, 写真实数值\n"
        "5. 重跑 reproduce.sh 直到 sanity gate 通过\n\n"
        "Do NOT claim completion until sanity gate passes."
    )

    return {
        "passed": False,
        "reason": reason,
        "fix_prompt": fix_prompt,
        "checks": checks,
    }


# ── selfcheck ──────────────────────────────────────────────

def _selfcheck() -> None:
    """6 项 assert 验证 sanity_gate 核心行为."""
    import tempfile

    print("[sanity_gate] running self-check...")

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)

        # 1. 无 outputs/ → passed
        r = check_sanity(ws)
        assert r["passed"] is True, f"1. 无 outputs 应 passed, got {r}"

        # 2. 空 outputs/ → passed
        (ws / "submission" / "outputs").mkdir(parents=True)
        r = check_sanity(ws)
        assert r["passed"] is True, f"2. 空 outputs 应 passed, got {r}"

        # 3. float dedup: 两个相同 final_loss → fail
        (ws / "submission" / "outputs" / "r1.json").write_text(
            json.dumps({"final_loss": 3.306041717529297, "training_time": 100.0})
        )
        (ws / "submission" / "outputs" / "r2.json").write_text(
            json.dumps({"final_loss": 3.306041717529297, "training_time": 200.0})
        )
        r = check_sanity(ws)
        assert r["passed"] is False, f"3. float dedup 应 fail, got {r}"
        assert "float_dedup" in r["reason"], f"3. reason 应含 float_dedup, got {r['reason']}"

        # 4. training_time < 1.0s → fail
        (ws / "submission" / "outputs" / "r1.json").write_text(
            json.dumps({"final_loss": 0.1, "training_time": 0.117})
        )
        (ws / "submission" / "outputs" / "r2.json").write_text(
            json.dumps({"final_loss": 0.2, "training_time": 0.036})
        )
        r = check_sanity(ws)
        assert r["passed"] is False, f"4. training_time 应 fail, got {r}"
        assert "training_time" in r["reason"], f"4. reason 应含 training_time, got {r['reason']}"

        # 5. placeholder text → fail
        (ws / "submission" / "outputs" / "r1.json").write_text(
            json.dumps({"final_loss": 0.1, "training_time": 100.0, "note": "placeholder result"})
        )
        (ws / "submission" / "outputs" / "r2.json").write_text(
            json.dumps({"final_loss": 0.2, "training_time": 200.0})
        )
        r = check_sanity(ws)
        assert r["passed"] is False, f"5. placeholder 应 fail, got {r}"
        assert "placeholder" in r["reason"], f"5. reason 应含 placeholder, got {r['reason']}"

        # 6. 全部合法 → passed
        (ws / "submission" / "outputs" / "r1.json").write_text(
            json.dumps({
                "final_loss": 0.5,
                "training_time": 100.0,
                "loss_curve": [1.0, 0.8, 0.6, 0.5],
            })
        )
        (ws / "submission" / "outputs" / "r2.json").write_text(
            json.dumps({
                "final_loss": 0.3,
                "training_time": 200.0,
                "loss_curve": [1.0, 0.7, 0.5, 0.3],
            })
        )
        r = check_sanity(ws)
        assert r["passed"] is True, f"6. 合法应 passed, got {r}"

    print("[sanity_gate] self-check OK (6/6)")


if __name__ == "__main__":
    _selfcheck()
