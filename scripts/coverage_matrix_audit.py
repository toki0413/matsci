"""红/蓝覆盖矩阵只读门禁 (可复用 / CI).

把 ``docs/threat_model.md`` §6 "Red / Blue Coverage Matrix" 表格建模成
"攻击面 → (红队探针, 蓝队校验)" 的机器可读视图, 做两件事:

1. **控制完整性 (蓝队视角)**: 表里引用的每个测试资产必须真实存在 —— 资产被
   删除/改名后, 声称的覆盖会静默失效, 只有一条 `file not found` 才暴露.
2. **覆盖回归 (门禁)**: 与基线对比, 任一攻击面的覆盖等级不得下降
   (dual → single/none, 或整行被删). 补测只增不减, 防止治理面回退.

覆盖等级 (两轴): none=0, single=1 (仅红或仅蓝), dual=2 (红蓝俱全).

只读诊断, 不改任何代码。

用法:
  python scripts/coverage_matrix_audit.py                    基础盘点
  python scripts/coverage_matrix_audit.py --json             追加机器可读 JSON
  python scripts/coverage_matrix_audit.py --write-baseline scripts/coverage_matrix_baseline.json
  python scripts/coverage_matrix_audit.py --fail-on-regression \\
      --baseline scripts/coverage_matrix_baseline.json        CI 门禁
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MATRIX_DOC = _REPO_ROOT / "docs" / "threat_model.md"
_SECTION_RE = re.compile(r"^##\s*6\.", re.M)
_TICK_RE = re.compile(r"`([^`]+)`")

# 覆盖等级排名 (越高越完备). single 不分红/蓝, 都算单边.
_RANK = {"none": 0, "single": 1, "dual": 2}


# ── 解析 §6 表格 ─────────────────────────────────────────────────────


def _extract_section(text: str) -> str:
    """截出 §6 一节正文 (到下一个 '## ' 或文末)."""
    m = _SECTION_RE.search(text)
    if not m:
        raise ValueError("docs/threat_model.md 未找到 '## 6. Red / Blue Coverage Matrix'")
    rest = text[m.end():]
    nxt = rest.find("\n## ")
    return rest if nxt == -1 else rest[:nxt]


def _split_row(line: str) -> list[str]:
    """把 markdown 表行按 '|' 拆成单元格 (去掉首尾空单元)."""
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(c) <= set("-: ") and "-" in c for c in cells)


def parse_matrix(text: str) -> list[dict]:
    """从 §6 表格解析出每个攻击面的两轴资产.

    列: 攻击面 | 层级 | 红队探针 | 蓝队校验 | 缺口.
    资产 = 反引号包裹的 token (去掉备注括号).
    """
    section = _extract_section(text)
    rows: list[dict] = []
    header_seen = False
    for raw in section.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = _split_row(line)
        if len(cells) < 5:
            continue
        if not header_seen:
            # 表头行 (含 "攻击面")
            if "攻击面" in cells[0]:
                header_seen = True
            continue
        if _is_separator(cells):
            continue
        surface, level, red_cell, blue_cell = cells[0], cells[1], cells[2], cells[3]
        rows.append({
            "surface": surface,
            "level": level,
            "red": _TICK_RE.findall(red_cell),
            "blue": _TICK_RE.findall(blue_cell),
        })
    return rows


# ── 资产解析 ─────────────────────────────────────────────────────────


def resolve_asset(token: str, root: Path = _REPO_ROOT) -> Path | None:
    """把引用 token 解析到仓库内真实文件; 找不到返回 None."""
    token = token.strip()
    if not token:
        return None
    # 1) 仓库根相对路径
    cand = root / token
    if cand.is_file():
        return cand
    # 2) agent/ 下相对路径 (多数测试资产写的是省略前缀的 tests/... 形式)
    cand = root / "agent" / token
    if cand.is_file():
        return cand
    # 3) 按 basename 在 agent/ 下搜 (表里多数只写文件名)
    name = Path(token).name
    agent_dir = root / "agent"
    if agent_dir.is_dir():
        for hit in agent_dir.rglob(name):
            if hit.is_file():
                return hit
    return None


def classify(red: list[str], blue: list[str]) -> str:
    if red and blue:
        return "dual"
    if red or blue:
        return "single"
    return "none"


def audit(root: Path = _REPO_ROOT) -> dict:
    """返回 {surfaces, broken_refs, summary}."""
    text = _MATRIX_DOC.read_text(encoding="utf-8")
    rows = parse_matrix(text)

    surfaces: dict[str, dict] = {}
    broken: list[dict] = []
    for r in rows:
        surface = r["surface"]
        red_ok, blue_ok = [], []
        for side, tokens, store in (("red", r["red"], red_ok),
                                    ("blue", r["blue"], blue_ok)):
            for tok in tokens:
                hit = resolve_asset(tok, root)
                if hit is None:
                    broken.append({"surface": surface, "side": side, "token": tok})
                else:
                    store.append(tok)
        coverage = classify(red_ok, blue_ok)
        surfaces[surface] = {
            "level": r["level"],
            "red": len(red_ok),
            "blue": len(blue_ok),
            "coverage": coverage,
            "rank": _RANK[coverage],
            "red_assets": red_ok,
            "blue_assets": blue_ok,
        }

    counts = {"dual": 0, "single": 0, "none": 0}
    for s in surfaces.values():
        counts[s["coverage"]] += 1
    summary = {"total": len(surfaces), **counts,
               "broken_refs": len(broken)}
    return {"surfaces": surfaces, "broken_refs": broken, "summary": summary}


# ── 打印 ─────────────────────────────────────────────────────────────


def _print_report(result: dict) -> None:
    surfaces = result["surfaces"]
    print("=" * 78)
    print("红/蓝覆盖矩阵 (§6)")
    print("-" * 78)
    print(f"{'攻击面':<40} {'层级':<9} {'红':>3} {'蓝':>3}  覆盖")
    print("-" * 78)
    for name, s in surfaces.items():
        print(f"{name:<40} {s['level']:<9} {s['red']:>3} {s['blue']:>3}  {s['coverage']}")
    print("-" * 78)
    su = result["summary"]
    print(f"攻击面 {su['total']} 个 | dual={su['dual']} single={su['single']} "
          f"none={su['none']} | 断链引用 {su['broken_refs']}")
    if result["broken_refs"]:
        print("\n[断链引用 → 声称的覆盖已静默失效]")
        for b in result["broken_refs"]:
            print(f"  - {b['surface']} [{b['side']}] -> {b['token']} (文件不存在)")
    print("=" * 78)


# ── 门禁 ─────────────────────────────────────────────────────────────


def run_gate(result: dict, baseline_path: Path) -> int:
    """基于基线的回归门禁. 返回退出码 (0 通过 / 1 失败)."""
    if result["broken_refs"]:
        print("\n[CI] 存在断链引用 (资产被删/改名) → 退出码 1")
        for b in result["broken_refs"]:
            print(f"  - {b['surface']} [{b['side']}] -> {b['token']}")
        return 1

    try:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"\n[CI] 基线文件缺失: {baseline_path} → 退出码 1")
        return 1
    except json.JSONDecodeError as e:
        print(f"\n[CI] 基线文件损坏: {baseline_path} ({e}) → 退出码 1")
        return 1

    base_surfaces = base.get("surfaces", {})
    cur = result["surfaces"]
    problems: list[str] = []

    for name, b in base_surfaces.items():
        b_rank = int(b.get("rank", _RANK.get(b.get("coverage", "none"), 0)))
        if name not in cur:
            problems.append(f"攻击面被移除: {name} (基线 {b.get('coverage')})")
            continue
        c_rank = cur[name]["rank"]
        if c_rank < b_rank:
            problems.append(
                f"覆盖回退: {name} {b.get('coverage')} -> {cur[name]['coverage']}"
            )

    new_surfaces = sorted(n for n in cur if n not in base_surfaces)
    if new_surfaces:
        print(f"\n[CI] 新增攻击面 (未纳入基线, 仅提示): {new_surfaces}")

    if problems:
        print("\n[CI] 覆盖矩阵回归 → 退出码 1")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\n[CI] 覆盖矩阵无回归, 门禁通过")
    return 0


# ── main ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="红/蓝覆盖矩阵只读门禁")
    ap.add_argument("--json", action="store_true",
                    help="输出机器可读 JSON 摘要")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="门禁: 断链引用 / 相对基线的覆盖回退 → 退出码 1")
    ap.add_argument("--baseline", metavar="PATH", default=None,
                    help="基线 JSON 路径")
    ap.add_argument("--write-baseline", metavar="PATH", default=None,
                    help="把本次矩阵写入 PATH 作为基线 (维护者扩面后更新用)")
    args = ap.parse_args(argv)

    result = audit()
    _print_report(result)

    if args.json:
        print("\n[JSON]\n" + json.dumps(result, ensure_ascii=False))

    if args.write_baseline:
        with open(args.write_baseline, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "surfaces": {
                        name: {
                            "level": s["level"],
                            "coverage": s["coverage"],
                            "rank": s["rank"],
                            "red": s["red"],
                            "blue": s["blue"],
                        }
                        for name, s in result["surfaces"].items()
                    },
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"\n[基线已写入] {args.write_baseline}")

    if args.fail_on_regression:
        if not args.baseline:
            print("\n[CI] --fail-on-regression 需要 --baseline PATH → 退出码 1")
            return 1
        return run_gate(result, Path(args.baseline))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
