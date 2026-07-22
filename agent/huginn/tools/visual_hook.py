"""Auto-render numerical tool results as charts for VLM analysis.

When a tool returns structured numerical data (energy values, spectra, etc.),
this hook generates a quick matplotlib chart and attaches it as base64 to
the tool result. The agent loop can then pass it to a VLM for visual analysis.

Also extracts structured "visual primitives" — numerical deictic pointers
(peak positions, trends, anomalies) — that give text-only LLMs concrete
coordinates to reason about, inspired by Thinking with Visual Primitives'
"point while it reasons" principle.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tool names that produce numerical data suitable for visualization
_VISUALIZABLE_PATTERNS = {
    "thermo_tool": "phase_diagram",     # formation energies -> scatter
    "band_structure": "line_plot",       # band structure -> line plot
    "dos": "line_plot",                  # density of states -> line plot
    "phonon": "line_plot",               # phonon dispersion -> line plot
    "mechanical_tool": "stress_strain",  # stress-strain curve
    "benchmark": "bar_chart",            # benchmark results -> bar chart
    "evolution": "convergence",          # evolution convergence
}

# Max base64 image size to avoid context bloat (256KB)
_MAX_IMAGE_BYTES = 256 * 1024


def should_visualize(tool_name: str, output: dict[str, Any]) -> bool:
    """Check if a tool's output is suitable for auto-visualization."""
    if not output or not output.get("result"):
        return False
    result = output["result"]
    if not isinstance(result, dict):
        return False

    for pattern in _VISUALIZABLE_PATTERNS:
        if pattern in tool_name.lower():
            return True

    for key in ("energies", "bands", "dos", "frequencies", "stress", "strain", "scores"):
        if key in result and isinstance(result[key], (list, dict)):
            return True

    # F1: code_tool / bash_tool 输出也触发 — RCB 跑分路径里 agent 用 code_tool
    # 算完打印 MAE/RMSE/R2 等数值, 没视觉反馈等于盲跑. 解 v8 视觉升级在 RCB 死路径.
    # 检测策略: stdout 里 ≥3 个可解析数字 → 视为可视化的数值序列.
    # ponytail: 启发式 regex, 不上 AST 解析. 升级路径: LLM 判是否适合可视化.
    if tool_name in ("code_tool", "bash_tool", "shell_tool"):
        stdout = result.get("stdout") or result.get("output") or ""
        if isinstance(stdout, str):
            nums = _NUMERIC_LINE_RE.findall(stdout)
            if len(nums) >= 3:
                return True

    return False


# F1: 匹配 "key: value" / "key = value" / "key value" 形式的数值行.
# 抓 MAE: 0.05 / RMSE=0.12 / R2 0.89 / loss 1.23e-4 等. 不抓纯叙述数字.
import re as _re
_NUMERIC_LINE_RE = _re.compile(
    r"[\w\-]{1,30}\s*[:=]?\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)",
    _re.ASCII,
)

# F1: 解析 "metric: value" 对. 要求 key 是字母/下划线开头, value 是数字.
# 用 findall 提取所有匹配, 同名 key 保留第一次出现 (避免 R2 多次出现画重叠).
_METRIC_PAIR_RE = _re.compile(
    r"(?P<key>[A-Za-z_][\w\-]{0,29})\s*[:=]\s*(?P<val>-?\d+\.?\d*(?:[eE][+-]?\d+)?)",
    _re.ASCII,
)


def _extract_metric_pairs(stdout: str) -> list[tuple[str, float]]:
    """从 stdout 抽 (key, value) 对. 同名只保留第一次."""
    seen: set[str] = set()
    pairs: list[tuple[str, float]] = []
    # 文件扩展名 / 常见非指标 key — 出现在 "foo.py:10" 这类 file:line 模式里
    _NON_METRIC_KEYS = {
        "line", "file", "version", "pid", "size", "len", "count", "n",
        "py", "js", "cpp", "c", "h", "json", "md", "txt", "log",
        "error", "warning", "info", "debug",
    }
    for m in _METRIC_PAIR_RE.finditer(stdout):
        key = m.group("key")
        if key in seen:
            continue
        # 排除 "foo.py:10" 这种 file:line — key 前一个字符是 "." 表示是文件扩展名
        if m.start() > 0 and stdout[m.start() - 1] == ".":
            continue
        try:
            val = float(m.group("val"))
        except ValueError:
            continue
        if key.lower() in _NON_METRIC_KEYS:
            continue
        seen.add(key)
        pairs.append((key, val))
    return pairs


def render_tool_output(tool_name: str, output: dict[str, Any]) -> str | None:
    """Generate a quick chart from tool output, return as base64 string.

    Returns None if rendering fails or data is unsuitable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.debug("matplotlib not available, skipping visualization")
        return None

    result = output.get("result", {})
    if not isinstance(result, dict):
        return None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    plotted = False

    # Band structure / DOS / phonon -> line plot
    for key in ("bands", "dos", "frequencies"):
        data = result.get(key)
        if isinstance(data, list) and data:
            if isinstance(data[0], list):
                for line in data[:20]:  # cap at 20 lines
                    if isinstance(line, list):
                        ax.plot(line, linewidth=0.8)
            else:
                ax.plot(data, linewidth=1.2)
            ax.set_xlabel("k-path" if key == "bands" else key)
            ax.set_ylabel(key)
            plotted = True
            break

    # Energies -> scatter/bar
    if not plotted:
        energies = result.get("energies") or result.get("energy")
        if isinstance(energies, list) and energies:
            labels = result.get("labels", [f"#{i}" for i in range(len(energies))])
            ax.bar(labels[:20], energies[:20])
            ax.set_ylabel("Energy (eV)")
            plotted = True
        elif isinstance(energies, (int, float)):
            ax.text(0.5, 0.5, f"E = {energies:.4f} eV", ha="center", va="center", fontsize=16)
            plotted = True

    # Stress-strain
    if not plotted:
        stress = result.get("stress")
        strain = result.get("strain")
        if isinstance(stress, list) and isinstance(strain, list):
            ax.plot(strain, stress, "b-o", markersize=4)
            ax.set_xlabel("Strain")
            ax.set_ylabel("Stress (GPa)")
            plotted = True

    # Scores -> bar chart
    if not plotted:
        scores = result.get("scores")
        if isinstance(scores, dict):
            names = list(scores.keys())[:10]
            values = [scores[n] for n in names]
            ax.barh(names, values)
            ax.set_xlabel("Score")
            plotted = True

    # F1: code_tool / bash_tool stdout → 解析 "key: value" 对画 bar chart
    # 不画 line chart — RCB agent 的 stdout 通常是离散指标 (MAE/RMSE/R2), 不是时序.
    if not plotted and tool_name in ("code_tool", "bash_tool", "shell_tool"):
        stdout = result.get("stdout") or result.get("output") or ""
        if isinstance(stdout, str):
            pairs = _extract_metric_pairs(stdout)
            if pairs:
                labels, values = zip(*pairs[:10])
                ax.bar(labels, values)
                ax.set_ylabel("Value")
                ax.tick_params(axis="x", rotation=30)
                plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"{tool_name} result", fontsize=12, fontweight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    if len(b64) > _MAX_IMAGE_BYTES:
        return None  # too large for context

    return b64


def extract_visual_primitives(tool_name: str, output: dict[str, Any]) -> str:
    """Extract coordinate-tagged visual primitives from tool output.

    Inspired by DeepSeek's Thinking with Visual Primitives: instead of vague
    "peak at index 3", give the LLM normalized coordinates <point>[[x,y]]</point>
    that anchor its reasoning to the data shape — bridging the Reference Gap.

    Also draws on 3D Primitives are a Spatial Language: structured primitives
    (here numerical, not geometric) serve as a bridge between text reasoning
    and visual understanding. This is the text-only path of our Mirage
    strategy — we don't need the LLM to see the image, we give it the
    primitives that describe the image.

    Coordinate system: normalized to 0-999 (same as DeepSeek paper), where
    x = data index (0=first point, 999=last point), y = data value (0=min, 999=max).
    This lets a text-only LLM "point" at specific locations in the data.
    """
    result = output.get("result", {})
    if not isinstance(result, dict):
        return ""

    lines: list[str] = []

    # 1D numeric lists: bands, dos, frequencies, energies, stress, strain
    for key in ("bands", "dos", "frequencies", "energies", "stress", "strain"):
        data = result.get(key)
        if not isinstance(data, list) or not data:
            continue
        # Handle nested lists (e.g. bands = [[...], [...]]) — flatten to summary
        if isinstance(data[0], list):
            sub_lines = []
            for bi, band in enumerate(data[:5]):
                try:
                    nums = [float(v) for v in band if isinstance(v, (int, float))]
                except (ValueError, TypeError):
                    continue
                if not nums:
                    continue
                pk, pk_xy = _to_coord(nums, max(range(len(nums)), key=lambda i: nums[i]))
                mn, mn_xy = _to_coord(nums, min(range(len(nums)), key=lambda i: nums[i]))
                sub_lines.append(
                    f"  band{bi}: peak=<point>[{pk_xy}]</point>({nums[pk]:.4f}), "
                    f"min=<point>[{mn_xy}]</point>({nums[mn]:.4f})"
                )
            if sub_lines:
                lines.append(f"[{key}] {len(data)} bands:\n" + "\n".join(sub_lines))
            continue
        try:
            nums = [float(v) for v in data if isinstance(v, (int, float))]
        except (ValueError, TypeError):
            continue
        if not nums:
            continue

        n = len(nums)
        peak_idx = max(range(n), key=lambda i: nums[i])
        min_idx = min(range(n), key=lambda i: nums[i])
        mean_v = sum(nums) / n
        std_v = (sum((x - mean_v) ** 2 for x in nums) / n) ** 0.5

        # 坐标化: 归一化到 0-999, 让 LLM 可以"指向"数据位置
        peak_xy = _normalize_coord(peak_idx, n, nums[peak_idx], nums)
        min_xy = _normalize_coord(min_idx, n, nums[min_idx], nums)

        # trend: compare first half mean vs second half mean
        mid = n // 2
        if mid > 0:
            first_half = sum(nums[:mid]) / mid
            second_half = sum(nums[mid:]) / (n - mid)
            if second_half > first_half * 1.05:
                trend = "increasing"
            elif second_half < first_half * 0.95:
                trend = "decreasing"
            else:
                trend = "approximately flat"
        else:
            trend = "unknown"

        # anomalies: points > 2 std from mean, with coordinates
        anomalies = []
        if std_v > 0:
            for i, v in enumerate(nums):
                if abs(v - mean_v) > 2 * std_v:
                    ax, ay = _normalize_coord(i, n, v, nums)
                    anomalies.append(f"<point>[{ax},{ay}]</point>={v:.4f}")
        anomalies_str = ", ".join(anomalies[:5]) if anomalies else "none"

        lines.append(
            f"[{key}] n={n}, peak=<point>[{peak_xy}]</point>({nums[peak_idx]:.4f}), "
            f"min=<point>[{min_xy}]</point>({nums[min_idx]:.4f}), "
            f"mean={mean_v:.4f}, std={std_v:.4f}, "
            f"trend={trend}, anomalies={anomalies_str}"
        )

        # 导数分析: 拐点 (二阶导符号变化) 和 FWHM 估计.
        # 材料科学中拐点常对应相变/临界点, FWHM 表征峰展宽程度.
        if n >= 5:
            deriv_lines = _extract_derivative_primitives(nums, key)
            lines.extend(deriv_lines)

    # Dict scores: top/bottom items with box coordinates
    scores = result.get("scores")
    if isinstance(scores, dict) and scores:
        try:
            items = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
            # 坐标化: 每个分数项占一个虚拟 x 位置, y 归一化
            vals = [float(v) for _, v in items]
            v_min, v_max = min(vals), max(vals)
            v_range = v_max - v_min if v_max != v_min else 1.0
            top_parts = []
            for k, v in items[:3]:
                yi = int((float(v) - v_min) / v_range * 999)
                top_parts.append(f"{k}=<point>[{yi}]</point>={float(v):.3f}")
            bot_parts = []
            for k, v in items[-2:]:
                yi = int((float(v) - v_min) / v_range * 999)
                bot_parts.append(f"{k}=<point>[{yi}]</point>={float(v):.3f}")
            lines.append(f"[scores] top3: {', '.join(top_parts)}; bottom: {', '.join(bot_parts)}")
        except (ValueError, TypeError):
            pass

    # 2D 数据 primitives: EDS mapping / phase_field / 2D 矩阵
    # 之前只处理 1D (bands/dos/energies), 2D 数据(相场/元素分布)没有坐标化 primitives,
    # text-only LLM 对 2D 形态"失明". 加 2D 分支让 LLM 通过质心/覆盖率坐标推理空间分布.
    lines.extend(_extract_2d_primitives(result))

    # F1: code_tool / bash_tool stdout 指标 primitives — 让 text-only LLM
    # 看到 "MAE=<point>[700]</point>=0.05" 这种坐标化锚点, 配合 v8 _measure_nearest_primitive
    # 的变体 5 解析. 不画图也有结构化视觉锚点.
    if tool_name in ("code_tool", "bash_tool", "shell_tool"):
        stdout = result.get("stdout") or result.get("output") or ""
        if isinstance(stdout, str):
            pairs = _extract_metric_pairs(stdout)
            if pairs:
                vals = [v for _, v in pairs]
                v_min, v_max = min(vals), max(vals)
                v_range = v_max - v_min if v_max != v_min else 1.0
                parts = []
                for k, v in pairs[:8]:
                    # 单坐标 (1D): x 归一化到 0-999, 让 LLM 指向指标值高度
                    yi = int((v - v_min) / v_range * 999)
                    parts.append(f"{k}=<point>[{yi}]</point>={v:.4f}")
                lines.append("[metrics] " + ", ".join(parts))

    if not lines:
        return ""

    return "\n".join(lines)


def _extract_2d_primitives(result: dict[str, Any]) -> list[str]:
    """从 2D 工具结果提取坐标化 primitives.

    覆盖 EDS mapping (元素质心 + 覆盖率) 和 phase_field (相分布 + domain 形态).
    坐标归一化到 0-999, 跟 1D primitives 同格式, LLM 可以跨维度引用.
    ponytail: 只处理结构化字段, 不重建 2D 矩阵. 升级: 等高线 + 热点检测.
    """
    out: list[str] = []
    measurements = result.get("measurements") or result

    # EDS mapping: elements 字段 — 元素质心坐标 + 覆盖率 + hotspots
    elements = measurements.get("elements") if isinstance(measurements, dict) else None
    if isinstance(elements, dict) and elements:
        img_shape = measurements.get("image_shape") or [1, 1]
        h, w = float(img_shape[0]), float(img_shape[1])
        parts: list[str] = []
        for elem, stats in elements.items():
            if not isinstance(stats, dict):
                continue
            try:
                cx, cy = stats.get("centroid_px", [0, 0])
                cov = float(stats.get("coverage_fraction", 0))
                # 质心归一化到 0-999
                nx = int(float(cx) / w * 999) if w > 0 else 0
                ny = int(float(cy) / h * 999) if h > 0 else 0
                cov_y = int(cov * 999)
                parts.append(
                    f"  {elem}: centroid=<point>[{nx},{ny}]</point>, "
                    f"coverage=<point>[{cov_y}]</point>={cov * 100:.1f}%"
                )
                # hotspots: top-N 聚集域质心, 比 overall centroid 更精确
                hotspots = stats.get("hotspots") or []
                for hi, hs in enumerate(hotspots[:3]):
                    if not isinstance(hs, dict):
                        continue
                    hcx, hcy = hs.get("centroid_px", [0, 0])
                    area = int(hs.get("area_px2", 0))
                    hnx = int(float(hcx) / w * 999) if w > 0 else 0
                    hny = int(float(hcy) / h * 999) if h > 0 else 0
                    parts.append(
                        f"    hotspot{hi+1}=<point>[{hnx},{hny}]</point> "
                        f"(area={area}px²)"
                    )
            except (ValueError, TypeError, IndexError):
                continue
        if parts:
            out.append(f"[EDS_mapping] {len(elements)} elements:\n" + "\n".join(parts))
            overlaps = measurements.get("overlaps")
            if isinstance(overlaps, dict) and overlaps:
                ov_parts = []
                for pair, ov in overlaps.items():
                    if isinstance(ov, dict):
                        iou = float(ov.get("iou", 0))
                        iou_y = int(iou * 999)
                        ov_parts.append(f"{pair}: IoU=<point>[{iou_y}]</point>={iou:.2f}")
                if ov_parts:
                    out.append("  overlaps: " + ", ".join(ov_parts))

    # phase_field: volume_fractions + morphology + interface
    vol_fracs = measurements.get("volume_fractions") if isinstance(measurements, dict) else None
    if isinstance(vol_fracs, dict) and vol_fracs:
        pf_shape = measurements.get("image_shape") or [1, 1]
        pf_h, pf_w = float(pf_shape[0]), float(pf_shape[1])
        parts = []
        for phase, frac in vol_fracs.items():
            try:
                y = int(float(frac) * 999)
                parts.append(f"{phase}=<point>[{y}]</point>={float(frac) * 100:.1f}%")
            except (ValueError, TypeError):
                continue
        if parts:
            out.append(f"[phase_field] volume_fractions: {', '.join(parts)}")

        interface = measurements.get("interface_pixel_fraction")
        if isinstance(interface, (int, float)):
            iy = int(float(interface) * 999)
            out.append(f"  interface_fraction=<point>[{iy}]</point>={float(interface) * 100:.2f}%")

        morphology = measurements.get("morphology")
        if isinstance(morphology, dict) and morphology:
            morph_parts = []
            for phase, morph in morphology.items():
                if not isinstance(morph, dict):
                    continue
                n_dom = int(morph.get("n_domains", 0))
                if n_dom > 0:
                    mean_area = float(morph.get("mean_domain_area_px2", 0))
                    morph_parts.append(f"{phase}: domains={n_dom}, mean_area={mean_area:.0f}px²")
                    # top domain 质心坐标化, 让 LLM 指向最大 domain 位置
                    top_cents = morph.get("top_domain_centroids_px") or []
                    for ci, c in enumerate(top_cents[:3]):
                        try:
                            cx, cy = c
                            nx = int(float(cx) / pf_w * 999) if pf_w > 0 else 0
                            ny = int(float(cy) / pf_h * 999) if pf_h > 0 else 0
                            morph_parts.append(
                                f"    {phase} top{ci+1}=<point>[{nx},{ny}]</point>"
                            )
                        except (ValueError, TypeError, IndexError):
                            continue
            if morph_parts:
                out.append("  morphology: " + "; ".join(morph_parts))

    return out


def _normalize_coord(idx: int, n: int, val: float, all_vals: list[float]) -> str:
    """归一化数据点到 0-999 坐标空间 (DeepSeek 格式).
    x: 数据索引位置 (0=第一个点, 999=最后一个点)
    y: 数据值位置 (0=最小值, 999=最大值)
    返回 "x,y" 字符串."""
    x = int(idx / max(n - 1, 1) * 999)
    v_min = min(all_vals)
    v_max = max(all_vals)
    v_range = v_max - v_min if v_max != v_min else 1.0
    y = int((val - v_min) / v_range * 999)
    return f"{x},{y}"


def _to_coord(nums: list[float], idx: int) -> tuple[int, str]:
    """返回 (原始索引, 归一化坐标) 用于嵌套列表."""
    xy = _normalize_coord(idx, len(nums), nums[idx], nums)
    return idx, xy


def _extract_derivative_primitives(nums: list[float], key: str) -> list[str]:
    """提取导数相关原语: 拐点 + FWHM 估计.

    拐点 = 二阶差分符号变化点, 常对应相变/临界温度.
    FWHM = 半峰全宽, 用线性插值估算峰的展宽程度.
    """
    lines: list[str] = []
    n = len(nums)
    if n < 5:
        return lines

    # 二阶差分: sign 变化 = 拐点
    inflections: list[str] = []
    for i in range(1, n - 1):
        d2 = (nums[i + 1] - 2 * nums[i] + nums[i - 1])
        d2_prev = (nums[i] - 2 * nums[i - 1] + nums[i - 2]) if i >= 2 else 0
        if d2_prev != 0 and (d2 > 0) != (d2_prev > 0):
            xy = _normalize_coord(i, n, nums[i], nums)
            inflections.append(f"<point>[{xy}]</point>={nums[i]:.4f}")
    if inflections:
        lines.append(
            f"  [{key}] inflections({len(inflections)}): {', '.join(inflections[:5])}"
        )

    # FWHM: 以最大峰为基准, 找半峰高位置的左右边界
    peak_idx = max(range(n), key=lambda i: nums[i])
    peak_val = nums[peak_idx]
    baseline = min(nums)
    half_max = baseline + (peak_val - baseline) * 0.5
    # 向左找半峰交叉
    left_i = peak_idx
    for i in range(peak_idx, -1, -1):
        if nums[i] < half_max:
            left_i = i
            break
    # 向右找半峰交叉
    right_i = peak_idx
    for i in range(peak_idx, n):
        if nums[i] < half_max:
            right_i = i
            break
    fwhm = right_i - left_i
    if fwhm > 0:
        lxy = _normalize_coord(left_i, n, nums[left_i], nums)
        rxy = _normalize_coord(right_i, n, nums[right_i], nums)
        lines.append(
            f"  [{key}] FWHM~{fwhm} points, "
            f"left=<point>[{lxy}]</point>, right=<point>[{rxy}]</point>"
        )

    return lines


def extract_comparative_primitives(
    baseline: dict[str, Any], current: dict[str, Any]
) -> str:
    """比较两轮工具输出, 生成差分原语.

    用于 validate 阶段: 上轮结果 vs 本轮结果, 突出变化.
    - 峰值位移: peak position shift
    - 新特征: 本轮有但上轮没有的峰/异常
    - 趋势变化: 上轮 increasing → 本轮 decreasing
    """
    bl_result = baseline.get("result", {})
    cr_result = current.get("result", {})
    if not isinstance(bl_result, dict) or not isinstance(cr_result, dict):
        return ""

    lines: list[str] = []
    for key in ("bands", "dos", "frequencies", "energies", "stress", "strain"):
        bl_data = bl_result.get(key)
        cr_data = cr_result.get(key)
        if not isinstance(bl_data, list) or not isinstance(cr_data, list):
            continue
        if isinstance(bl_data[0] if bl_data else 0, list):
            continue  # 跳过嵌套, 只比 1D
        try:
            bl_nums = [float(v) for v in bl_data if isinstance(v, (int, float))]
            cr_nums = [float(v) for v in cr_data if isinstance(v, (int, float))]
        except (ValueError, TypeError):
            continue
        if not bl_nums or not cr_nums:
            continue

        bl_peak = max(bl_nums)
        cr_peak = max(cr_nums)
        shift = cr_peak - bl_peak
        if abs(shift) > 1e-6:
            lines.append(
                f"[{key}] peak_shift: {bl_peak:.4f} → {cr_peak:.4f} (Δ={shift:+.4f})"
            )

        # 谷值位移: 材料科学中 valley shift 同样重要 (如能带极小值)
        bl_val = min(bl_nums)
        cr_val = min(cr_nums)
        vshift = cr_val - bl_val
        if abs(vshift) > 1e-6:
            lines.append(
                f"[{key}] valley_shift: {bl_val:.4f} → {cr_val:.4f} (Δ={vshift:+.4f})"
            )

        # 新异常: 本轮有 >2σ 的点但上轮没有
        cr_mean = sum(cr_nums) / len(cr_nums)
        cr_std = (sum((x - cr_mean) ** 2 for x in cr_nums) / len(cr_nums)) ** 0.5
        bl_mean = sum(bl_nums) / len(bl_nums)
        bl_std = (sum((x - bl_mean) ** 2 for x in bl_nums) / len(bl_nums)) ** 0.5
        new_anomalies = 0
        if cr_std > 0 and bl_std > 0:
            for v in cr_nums:
                if abs(v - cr_mean) > 2 * cr_std and abs(v - bl_mean) <= 2 * bl_std:
                    new_anomalies += 1
        if new_anomalies:
            lines.append(f"[{key}] new_anomalies: {new_anomalies} points outside baseline 2σ")

    # QW2: 2D 比较分支 — EDS centroid shift / coverage diff / phase_field vol_frac diff
    # 之前只比 1D, 2D 数据 (相场/元素分布) 的空间变化对 text-only LLM 不可见.
    lines.extend(_comparative_2d_primitives(bl_result, cr_result))

    if not lines:
        return ""
    return "### Comparative Visual Primitives (vs last run)\n" + "\n".join(lines)


def _comparative_2d_primitives(bl_result: dict[str, Any], cr_result: dict[str, Any]) -> list[str]:
    """2D 差分原语: EDS 质心位移 / 覆盖率变化 / 相场体积分数差.

    跟 _extract_2d_primitives 配对: 单轮提取结构化 primitives, 本函数做跨轮 diff.
    坐标归一化到 0-999, 让 LLM 通过 <point> 引用空间位移.
    ponytail: 只处理结构化字段, 不重建 2D 矩阵. 升级: 光流 / 域配准.
    """
    out: list[str] = []

    # EDS: elements 字典, 每元素 {centroid_px, coverage_fraction, hotspots}
    bl_meas = bl_result.get("measurements") or bl_result
    cr_meas = cr_result.get("measurements") or cr_result
    bl_els = bl_meas.get("elements") if isinstance(bl_meas, dict) else None
    cr_els = cr_meas.get("elements") if isinstance(cr_meas, dict) else None
    if isinstance(bl_els, dict) and isinstance(cr_els, dict):
        bl_shape = bl_meas.get("image_shape") or [1, 1]
        cr_shape = cr_meas.get("image_shape") or bl_shape
        bl_h, bl_w = float(bl_shape[0]), float(bl_shape[1])
        cr_h, cr_w = float(cr_shape[0]), float(cr_shape[1])
        parts = []
        for elem in set(bl_els) & set(cr_els):
            be, ce = bl_els[elem], cr_els[elem]
            if not (isinstance(be, dict) and isinstance(ce, dict)):
                continue
            try:
                bcx, bcy = be.get("centroid_px", [0, 0])
                ccx, ccy = ce.get("centroid_px", [0, 0])
                bnx = int(float(bcx) / bl_w * 999) if bl_w > 0 else 0
                bny = int(float(bcy) / bl_h * 999) if bl_h > 0 else 0
                cnx = int(float(ccx) / cr_w * 999) if cr_w > 0 else 0
                cny = int(float(ccy) / cr_h * 999) if cr_h > 0 else 0
                dx = cnx - bnx
                dy = cny - bny
                if dx or dy:
                    parts.append(
                        f"  {elem} centroid_shift: "
                        f"<point>[{bnx},{bny}]</point> → <point>[{cnx},{cny}]</point> "
                        f"(Δx={dx:+d}, Δy={dy:+d})"
                    )
                bl_cov = float(be.get("coverage_fraction", 0))
                cr_cov = float(ce.get("coverage_fraction", 0))
                if abs(cr_cov - bl_cov) > 1e-4:
                    parts.append(
                        f"  {elem} coverage: {bl_cov * 100:.1f}% → {cr_cov * 100:.1f}% "
                        f"(Δ={(cr_cov - bl_cov) * 100:+.2f}%)"
                    )
            except (ValueError, TypeError, IndexError):
                continue
        # baseline 有但本轮消失 / 本轮新增的元素
        only_bl = set(bl_els) - set(cr_els)
        only_cr = set(cr_els) - set(bl_els)
        if only_bl:
            parts.append(f"  lost_elements: {sorted(only_bl)}")
        if only_cr:
            parts.append(f"  new_elements: {sorted(only_cr)}")
        if parts:
            out.append("[EDS_mapping] 2D diff:\n" + "\n".join(parts))

    # phase_field: volume_fractions + interface_pixel_fraction
    bl_pf = bl_meas.get("volume_fractions") if isinstance(bl_meas, dict) else None
    cr_pf = cr_meas.get("volume_fractions") if isinstance(cr_meas, dict) else None
    if isinstance(bl_pf, dict) and isinstance(cr_pf, dict):
        parts = []
        for phase in set(bl_pf) & set(cr_pf):
            try:
                bf = float(bl_pf[phase])
                cf = float(cr_pf[phase])
                if abs(cf - bf) > 1e-4:
                    parts.append(
                        f"  {phase}: {bf * 100:.1f}% → {cf * 100:.1f}% "
                        f"(Δ={(cf - bf) * 100:+.2f}%)"
                    )
            except (ValueError, TypeError):
                continue
        bl_iface = bl_meas.get("interface_pixel_fraction") if isinstance(bl_meas, dict) else None
        cr_iface = cr_meas.get("interface_pixel_fraction") if isinstance(cr_meas, dict) else None
        if isinstance(bl_iface, (int, float)) and isinstance(cr_iface, (int, float)):
            if abs(float(cr_iface) - float(bl_iface)) > 1e-4:
                parts.append(
                    f"  interface: {float(bl_iface) * 100:.2f}% → "
                    f"{float(cr_iface) * 100:.2f}% "
                    f"(Δ={(float(cr_iface) - float(bl_iface)) * 100:+.3f}%)"
                )
        if parts:
            out.append("[phase_field] 2D diff:\n" + "\n".join(parts))

    return out


def enrich_with_visual(tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
    """Post-process tool output: add _visual_base64 + _visual_primitives."""
    if not should_visualize(tool_name, output):
        return output

    b64 = render_tool_output(tool_name, output)
    primitives = extract_visual_primitives(tool_name, output)

    if b64:
        output["_visual_base64"] = b64

    if primitives:
        # Structured visual primitives as deictic pointers for text LLM reasoning.
        # Thinking with Visual Primitives: "point while it reasons" — give
        # coordinates, not vague descriptions. 3D Primitives: primitives as a
        # spatial language bridging text and visual. Mirage effect: text LLMs
        # have latent visual reasoning; these primitives activate it without
        # requiring actual image input.
        # When a VLM is available, _visual_base64 provides the actual image;
        # when not, _visual_primitives gives the LLM enough structure to
        # "visualize" the data shape through coordinates alone.
        output["_visual_hint"] = (
            "Visual primitives (coordinate-tagged, DeepSeek format):\n"
            f"{primitives}\n"
            "Coordinates are normalized 0-999: x=data position, y=value.\n"
            "<point>[x,y]</point> tags are deictic pointers — reason about\n"
            "data shape by referencing these coordinates. Where are peaks?\n"
            "What does the trend imply? Do anomalies suggest physics or noise?\n"
            "\n"
            "Mirage activation: these primitives are the data's shape in coordinate form.\n"
            "Use them to 'see' the curve without an image — peaks at <point>[x,y]</point>\n"
            "are real features, trends describe the overall slope, anomalies flag\n"
            "outliers worth investigating. Point at coordinates while reasoning."
        )

    # self_check: Nullmax 启发 — 让红队/PhaseGate 能判断视觉估算可信度.
    # 之前只有图像输入侧有 self_check (visual_to_symbols_structured), 工具产出的
    # 数值 primitives 没有. 加 _visual_self_check 字段, RedTeamReviewer 能读它
    # 判断可视化结论是否可信.
    sc = _estimate_data_confidence(output)
    if sc:
        output["_visual_self_check"] = sc

    return output


def _estimate_data_confidence(output: dict[str, Any]) -> dict[str, Any]:
    """估算工具产出的数值数据可视化置信度.

    1D 数据 (bands/dos/energies): 数据点数 / 信噪比 / 异常比例
    2D 数据 (EDS/phase_field): 像素覆盖率 / 域数量 / 重叠度
    ponytail: 启发式加权, 不调 LLM. 升级: 让 LLM 判断数据质量.
    """
    result = output.get("result", {})
    if not isinstance(result, dict):
        return {}

    score = 0.0
    caveats: list[str] = []

    # 1D 数据
    for key in ("bands", "dos", "frequencies", "energies", "stress", "strain"):
        data = result.get(key)
        if not isinstance(data, list) or not data:
            continue
        if isinstance(data[0], list):
            continue  # 嵌套不估
        try:
            nums = [float(v) for v in data if isinstance(v, (int, float))]
        except (ValueError, TypeError):
            continue
        if not nums:
            continue
        n = len(nums)
        mean_v = sum(nums) / n
        std_v = (sum((x - mean_v) ** 2 for x in nums) / n) ** 0.5
        # 信噪比: std/mean 越大噪声越多
        snr = abs(mean_v / std_v) if std_v > 0 else 10.0
        # 异常比例
        n_anom = sum(1 for v in nums if abs(v - mean_v) > 2 * std_v) if std_v > 0 else 0
        anom_rate = n_anom / n if n > 0 else 0

        if n >= 20:
            score += 0.15
        elif n >= 5:
            score += 0.08
            caveats.append(f"few_points_{key}: {n} 个数据点, 趋势分析可靠性降低")
        else:
            caveats.append(f"too_few_points_{key}: {n} 个数据点, 无法可靠分析")

        if snr >= 3.0:
            score += 0.1
        elif snr < 1.0:
            caveats.append(f"low_snr_{key}: 信噪比 {snr:.2f}, 数据噪声大")

        if anom_rate > 0.1:
            caveats.append(f"high_anomaly_rate_{key}: {anom_rate * 100:.1f}% 数据点异常")

    # 2D 数据 (EDS)
    measurements = result.get("measurements") or result
    elements = measurements.get("elements") if isinstance(measurements, dict) else None
    if isinstance(elements, dict) and elements:
        total_cov = 0.0
        n_elems = 0
        for elem, stats in elements.items():
            if isinstance(stats, dict):
                cov = float(stats.get("coverage_fraction", 0))
                total_cov += cov
                n_elems += 1
                # hotspots 存在说明做了连通域分析, 置信度高
                if stats.get("hotspots"):
                    score += 0.05
        avg_cov = total_cov / n_elems if n_elems > 0 else 0
        if avg_cov < 0.05:
            caveats.append(f"low_coverage_eds: 平均覆盖率 {avg_cov * 100:.1f}%, 元素信号弱")
        elif avg_cov > 0.5:
            score += 0.1
        if n_elems == 0:
            caveats.append("no_elements_eds: 未检测到元素")

    # 2D 数据 (phase_field)
    vol_fracs = measurements.get("volume_fractions") if isinstance(measurements, dict) else None
    if isinstance(vol_fracs, dict) and vol_fracs:
        morphology = measurements.get("morphology") if isinstance(measurements, dict) else None
        if isinstance(morphology, dict):
            n_total_doms = 0
            for phase, morph in morphology.items():
                if isinstance(morph, dict):
                    n_total_doms += int(morph.get("n_domains", 0))
                    if morph.get("top_domain_centroids_px"):
                        score += 0.05
            if n_total_doms == 0:
                caveats.append("no_domains_phase_field: 未检测到相域")
            elif n_total_doms >= 3:
                score += 0.1

    if not caveats:
        score += 0.2  # 无 caveats 加分

    return {
        "confidence": round(min(score, 1.0), 2),
        "caveats": caveats,
        "note": "数值数据可视化置信度估算, 让红队判断可视化结论可信度",
    }
