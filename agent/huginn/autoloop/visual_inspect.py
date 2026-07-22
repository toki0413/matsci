"""VisualInspect — 视觉检查函数模块.

从 engine.py 抽出的 5 个方法 (原 _execute_visual_inspect + 4 个辅助).
通过 engine 实例访问 workspace/_last_visual_context/_visual_base64,
无自身状态, 纯函数 + engine 注入.

调用点: engine._execute 里 mode == "visual_inspect" 分支.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


async def execute_visual_inspect(
    engine: Any, description: str, context: dict[str, Any]
) -> dict[str, Any]:
    """Path C: 交互式视觉检查. 让 agent 主动调用视觉工具检查上一轮结果.

    这是 OpenThinkIMG 式的工具调用路径 — agent 在推理过程中主动选择
    "放大图表某区域"或"测量某数据点", 而不是被动接收预处理好的视觉基元.
    使用已有的 image_analysis_tool / visual_hook 基础设施, 不新建工具.

    description 解析: "zoom into band 3 near [500,800]" / "measure peak at [999,999]"
    坐标是 0-999 归一化的视觉原语坐标 (路径 B 格式).
    """
    result: dict[str, Any] = {
        "mode": "visual_inspect",
        "description": description,
        "actions": [],
    }

    visual_ctx = getattr(engine, "_last_visual_context", "")
    visual_base64 = getattr(engine, "_visual_base64", "")

    desc_lower = description.lower()

    # 动作 -1 (G2): rotate — SE(3) 群作用, 不依赖 visual_base64
    # agent 能调 "rotate 90° z" 或 "chain: rotate 90° z; zoom [...]"
    # 从 cognitive_maps 拿当前结构, 调 rotate + to_composite_token
    # 返回旋转后的 <point3d> 文本给 LLM (C 实验工程化)
    if desc_lower.startswith("rotate"):
        # 解析 "rotate 90° z" / "rotate z 90" / "rotate 90 around z"
        axis_match = re.search(r"\b([xyz])\b", desc_lower)
        angle_match = re.search(r"(-?\d+(?:\.\d+)?)\s*°?", description)
        if not axis_match or not angle_match:
            result["actions"].append({
                "action": "rotate",
                "error": "cannot parse axis/angle, expect 'rotate 90° z' or 'rotate z 90'",
                "description": description,
            })
            return result
        axis = axis_match.group(1)
        angle = float(angle_match.group(1))

        # 从 engine._state.cognitive_maps 拿最后一个 map
        state = getattr(engine, "_state", None)
        cog_maps = getattr(state, "cognitive_maps", {}) if state else {}
        if not cog_maps:
            result["actions"].append({
                "action": "rotate",
                "error": "no cognitive_map in engine state, build one first",
                "description": description,
            })
            return result

        try:
            from huginn.metacog.structure_cognitive_map import StructureCognitiveMap
            # 取最后一个 map (按插入顺序)
            last_map_id = list(cog_maps.keys())[-1]
            m = StructureCognitiveMap.from_engine_state_dict(cog_maps[last_map_id])
            # 调 SE(3) rotate
            m_rotated = m.rotate(axis=axis, angle=angle, degrees=True)
            # 投影成 <point3d> 复合 token
            token = m_rotated.to_composite_token()
            # 也算 Hodge 看拓扑有没有变
            hodge = m_rotated.hodge_decomposition()
            result["actions"].append({
                "action": "rotate",
                "map_id": last_map_id,
                "axis": axis,
                "angle": angle,
                "n_atoms": token["n_atoms"],
                "species": token["species"],
                "point3d_primitives": token["text"],
                "coords_rotated": token["coords"].tolist(),
                "hodge": hodge["note"],
                "note": f"SE(3) rotate {angle}° around {axis}: <point3d> + coords synced (C experiment)",
            })
        except Exception as e:
            result["actions"].append({
                "action": "rotate",
                "error": f"rotate failed: {e}",
                "description": description,
            })
        return result

    # 动作 0 (QW3): chain — 必须在 visual_ctx 检查之前, 否则空 visual_ctx 时
    # "chain: rotate ..." 跑不了. 子描述自己检查 visual_ctx (rotate 不需要).
    if desc_lower.startswith("chain"):
        body = re.sub(r"^\s*chain\s*:?\s*", "", description, flags=re.IGNORECASE).strip()
        sub_descs = [s.strip() for s in re.split(r"\s*;\s*|\s+then\s+", body, flags=re.IGNORECASE) if s.strip()]
        trajectory: list[dict[str, Any]] = []
        for sd in sub_descs:
            sub_result = await execute_visual_inspect(engine, sd, context)
            sub_actions = sub_result.get("actions", [])
            first = sub_actions[0] if sub_actions else {}
            trajectory.append({
                "step": sd,
                "action": first.get("action", "unknown"),
                "note": first.get("note", ""),
                "sub_result": sub_result,
            })
        result["actions"].append({
            "action": "chain",
            "description": description,
            "trajectory": trajectory,
            "n_steps": len(trajectory),
            "note": f"Chain of {len(trajectory)} visual_inspect steps",
        })
        return result

    if not visual_ctx and not visual_base64:
        return {
            **result,
            "success": False,
            "error": "No visual data from previous iteration to inspect",
        }

    # 动作 1: zoom — 放大某区域
    if "zoom" in desc_lower:
        coords = re.findall(r"\[?(\d+)\s*,\s*(\d+)\]?", description)
        if len(coords) >= 2:
            x1, y1 = int(coords[0][0]), int(coords[0][1])
            x2, y2 = int(coords[1][0]), int(coords[1][1])
            action_result = {
                "action": "zoom",
                "region": [x1, y1, x2, y2],
                "note": f"Zoomed into region [{x1},{y1}]-[{x2},{y2}]",
            }
            if visual_base64:
                try:
                    import base64 as b64
                    import io as _io
                    from PIL import Image

                    img_data = b64.b64decode(visual_base64)
                    img = Image.open(_io.BytesIO(img_data))
                    w, h = img.size
                    px1 = int(x1 / 999 * w)
                    py1 = int(y1 / 999 * h)
                    px2 = int(x2 / 999 * w)
                    py2 = int(y2 / 999 * h)
                    cropped = img.crop((px1, py1, px2, py2))
                    buf = _io.BytesIO()
                    cropped.save(buf, format="PNG")
                    cropped_bytes = buf.getvalue()
                    action_result["cropped_image"] = b64.b64encode(cropped_bytes).decode()[:10000]
                    action_result["crop_size"] = [px2 - px1, py2 - py1]

                    # 反馈通道: 调 vision_describe 把 cropped 图转结构化 JSON.
                    # 让 DeepSeek 读描述而不是原始像素. 失败不阻塞, 只缺 description 字段.
                    try:
                        from huginn.tools.vision_describe_tool import describe_image_bytes
                        desc_result = describe_image_bytes(
                            cropped_bytes,
                            question=f"Zoomed region [{x1},{y1}]-[{x2},{y2}] of the previous image. Describe what's visible.",
                        )
                        if desc_result.get("available"):
                            action_result["description"] = desc_result
                        else:
                            action_result["description_unavailable"] = desc_result.get("error", "unknown")
                            action_result["description_tier"] = desc_result.get("tier")
                    except Exception as e:
                        action_result["description_unavailable"] = f"vision_describe failed: {e}"
                except ImportError:
                    action_result["note"] += " (PIL not available, coordinates only)"
                except Exception as e:
                    action_result["note"] += f" (crop failed: {e})"
            result["actions"].append(action_result)

    # 动作 2: measure — 测量某点或区域的数据值
    if "measure" in desc_lower:
        coords = re.findall(r"\[?(\d+)\s*,\s*(\d+)\]?", description)
        if coords:
            x, y = int(coords[0][0]), int(coords[0][1])
            measured = measure_nearest_primitive(x, y, visual_ctx)
            result["actions"].append({
                "action": "measure",
                "coordinate": [x, y],
                "note": f"Measured at <point>[{x},{y}]</point>",
                "nearest_primitive": measured,
                "visual_context_snippet": visual_ctx[:300] if visual_ctx else "",
            })

    # 动作 3: annotate — 标注结构特征
    if "annotate" in desc_lower:
        annotation = annotate_visual_features(description, visual_base64, visual_ctx)
        result["actions"].append({
            "action": "annotate",
            "description": description,
            "note": annotation["note"],
            "features": annotation.get("features", []),
            "visual_context": visual_ctx[:500] if visual_ctx else "",
        })
        if annotation.get("tool_output"):
            result["actions"][-1]["tool_output"] = annotation["tool_output"]

    # 动作 4: compare — 比较两组数据
    if "compare" in desc_lower:
        comparison = compare_visual_data(description, visual_ctx)
        result["actions"].append({
            "action": "compare",
            "description": description,
            "visual_context": visual_ctx[:500] if visual_ctx else "",
            "note": comparison["note"],
            "diff": comparison.get("diff", {}),
        })

    # 默认: 记录检查请求
    else:
        result["actions"].append({
            "action": "inspect",
            "description": description,
            "visual_context": visual_ctx[:500] if visual_ctx else "",
        })

    # 生成新的视觉基元 (基于检查动作的输出)
    new_primitives = []
    for action in result["actions"]:
        if "note" in action:
            new_primitives.append(f"[{action['action']}] {action['note']}")
    result["visual_summary"] = "\n".join(new_primitives)
    result["success"] = True

    # 用 enrich_with_visual 给这次检查也生成视觉基元
    try:
        from huginn.tools.visual_hook import enrich_with_visual

        enriched = enrich_with_visual("visual_inspect", {"result": result})
        if "_visual_hint" in enriched:
            result["_visual_hint"] = enriched["_visual_hint"]
    except Exception:
        pass

    return result


def measure_nearest_primitive(x: int, y: int, visual_ctx: str) -> dict[str, Any]:
    """v8: 从 visual_ctx 解析 <point>[x,y]</point> 原语, 找最接近 (x,y) 的点.

    visual_hook.py 生成 5 种格式变体 (B1 鲁棒化):
      1. <point>[x,y]</point>(value)       — peak/min (band/dos/phonon)
      2. <point>[x,y]</point>=value         — anomalies
      3. <point>[x,y]</point>=value%        — phase_field / coverage
      4. <point>[x,y]</point> value         — inflections (空格分隔)
      5. key=<point>[y]</point>=value       — scores (单坐标, 只有 y)

    单坐标 (变体 5) 的 x 默认 0, 距离只算 y 差.

    返回最近点的坐标 + 数值 + 上下文. 无原语返回空 dict.
    """
    if not visual_ctx:
        return {}

    primitives: list[dict[str, Any]] = []

    # 变体 1-3: <point>[x,y]</point>(value) / =value / =value%
    for m in re.finditer(
        r"<point>\[(\d+),(\d+)\]</point>(?:\(([\d.\-eE]+)\)|=([\d.\-eE]+%?)|[\s]+([\d.\-eE]+))",
        visual_ctx,
    ):
        px, py = int(m.group(1)), int(m.group(2))
        val = m.group(3) or m.group(4) or m.group(5)
        val_clean = val.rstrip("%") if val else None
        primitives.append({
            "coordinate": [px, py],
            "value": float(val_clean) if val_clean else None,
            "raw_value": val,
        })

    # 变体 5: key=<point>[y]</point>=value (单坐标)
    for m in re.finditer(
        r"(\w+)=<point>\[(\d+)\]</point>=([\d.\-eE]+%?)",
        visual_ctx,
    ):
        key = m.group(1)
        py = int(m.group(2))
        val = m.group(3).rstrip("%")
        primitives.append({
            "coordinate": [0, py],
            "value": float(val) if val else None,
            "raw_value": m.group(3),
            "label": key,
        })

    if not primitives:
        return {}

    # 找最接近 (x, y) 的点
    best = None
    best_dist = float("inf")
    for p in primitives:
        px, py = p["coordinate"]
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_dist:
            best_dist = d
            best = {**p, "distance": int(best_dist ** 0.5)}

    if best is None:
        return {}

    # 找上下文行 (含该 point 的行)
    coord_str = f"<point>[{best['coordinate'][0]},{best['coordinate'][1]}]</point>"
    for line in visual_ctx.split("\n"):
        if coord_str in line:
            best["context"] = line.strip()[:200]
            break
    return best


def annotate_visual_features(
    description: str, visual_base64: str, visual_ctx: str
) -> dict[str, Any]:
    """v8: 标注结构特征. 有图片调 image_analysis_tool, 无图片用文本特征.

    B2 增强: 无图片时解析 visual_ctx 段落结构 + 趋势 + 异常聚类,
    不只做单点 regex 提取. 让文本路径也有结构化标注.

    ponytail: 优先用已有 image_analysis_tool (defect_detect/phase_field 场景),
    失败降级到 visual_ctx 文本特征提取. 不新建工具.
    """
    features: list[str] = []
    tool_output: dict[str, Any] | None = None
    # 有图片 → 调 image_analysis_tool 做真正结构标注
    if visual_base64:
        try:
            from huginn.tools.registry import ToolRegistry
            import base64 as b64
            import io as _io

            img_tool = ToolRegistry.get("image_analysis_tool")
            if img_tool:
                from PIL import Image
                img_data = b64.b64decode(visual_base64)
                img = Image.open(_io.BytesIO(img_data))
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = b64.b64encode(buf.getvalue()).decode()
                desc_lower = description.lower()
                if "defect" in desc_lower or "缺陷" in description:
                    scene = "defect_detect"
                elif "phase" in desc_lower or "相" in description:
                    scene = "phase_field"
                else:
                    scene = "sem_analysis"
                res = img_tool.call({
                    "image_base64": img_b64,
                    "scene": scene,
                    "task_description": description,
                })
                if res and getattr(res, "success", False):
                    tool_output = res.data if hasattr(res, "data") else res
                    features.append(f"{scene}: tool analysis done")
                else:
                    features.append(f"{scene}: tool returned no result")
        except Exception as e:
            features.append(f"tool_annotation_failed: {e}")
    # 文本特征提取 (B2 增强: 段落结构 + 趋势 + 异常聚类)
    if visual_ctx:
        structured = extract_text_visual_features(visual_ctx)
        features.extend(structured["features"])
        if structured["summary"]:
            if tool_output is None:
                tool_output = {"text_analysis": structured["summary"]}
            else:
                tool_output["text_analysis"] = structured["summary"]
    note = "Annotated " + ", ".join(features[:5]) if features else "No features found"
    return {"note": note, "features": features, "tool_output": tool_output}


def extract_text_visual_features(visual_ctx: str) -> dict[str, Any]:
    """B2: 从 visual_ctx 提取结构化文本特征 — 段落 + 趋势 + 异常聚类.

    visual_ctx 按段落组织, 每个 [section] 是一个数据集. 解析:
    - section 列表 (band/dos/phonon/scores/phase_field/...)
    - 每段的 trend (increasing/decreasing/flat)
    - 异常点聚类 (相邻异常归为一组)
    - 关键数值 (peak/min/mean/std)
    """
    features: list[str] = []
    summary_parts: list[str] = []
    sections: list[dict[str, Any]] = []

    for line in visual_ctx.split("\n"):
        line = line.strip()
        if not line:
            continue
        sec_match = re.match(r"\[(\w+)\]", line)
        if sec_match:
            sec_name = sec_match.group(1)
            sec: dict[str, Any] = {"name": sec_name, "raw": line[:200]}

            trend_m = re.search(r"trend=(\w+)", line)
            if trend_m:
                sec["trend"] = trend_m.group(1)
                features.append(f"{sec_name}.trend={trend_m.group(1)}")

            peak_m = re.search(r"peak=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", line)
            if peak_m:
                sec["peak"] = float(peak_m.group(1))
            min_m = re.search(r"min=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", line)
            if min_m:
                sec["min"] = float(min_m.group(1))

            mean_m = re.search(r"mean=([\d.\-eE]+)", line)
            if mean_m:
                sec["mean"] = float(mean_m.group(1))
            std_m = re.search(r"std=([\d.\-eE]+)", line)
            if std_m:
                sec["std"] = float(std_m.group(1))

            anom_m = re.search(r"anomalies=([^,\n]+(?:,\s*[^,\n]+)*)", line)
            if anom_m and anom_m.group(1).strip() != "none":
                anom_str = anom_m.group(1)
                anom_count = anom_str.count("<point>")
                sec["anomaly_count"] = anom_count
                if anom_count > 0:
                    features.append(f"{sec_name}.anomalies={anom_count}")

            sections.append(sec)
            parts = [f"{sec_name}"]
            if "trend" in sec:
                parts.append(f"trend={sec['trend']}")
            if "peak" in sec and "min" in sec:
                parts.append(f"range=[{sec['min']:.4f}, {sec['peak']:.4f}]")
            if "anomaly_count" in sec and sec["anomaly_count"] > 0:
                parts.append(f"{sec['anomaly_count']} anomalies")
            summary_parts.append(", ".join(parts))

    summary = "; ".join(summary_parts) if summary_parts else ""
    return {"features": features, "summary": summary, "sections": sections}


def compare_visual_data(description: str, visual_ctx: str) -> dict[str, Any]:
    """v8: 比较两组数据. 用文本级差分 (peak/min/anomaly 数量级).

    ponytail: visual_hook.extract_comparative_primitives 已有, 但需 baseline+current
    两个 dict, visual_ctx 是文本. 这里做文本级差分: 解析两组 <point> 原语, 算峰值位移
    /新异常. 升级路径才换真正的 baseline vs current dict 比较.
    """
    if not visual_ctx:
        return {"note": "No visual context to compare", "diff": {}}
    peaks = re.findall(r"peak=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", visual_ctx)
    mins = re.findall(r"min=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", visual_ctx)
    anomalies = re.findall(r"anomalies=([^,\n]+)", visual_ctx)
    diff: dict[str, Any] = {}
    if peaks:
        peak_vals = [float(p) for p in peaks if p]
        diff["peak_range"] = [min(peak_vals), max(peak_vals)]
        diff["peak_count"] = len(peak_vals)
    if mins:
        min_vals = [float(m) for m in mins if m]
        diff["min_range"] = [min(min_vals), max(min_vals)]
    if anomalies:
        diff["anomaly_count"] = sum(1 for a in anomalies if a.strip() and a.strip() != "none")
    compare_match = re.search(
        r"compare\s+(\w+\s*\d*)\s+(?:and|with|vs\.?)\s+(\w+\s*\d*)",
        description, re.IGNORECASE,
    )
    target = None
    if compare_match:
        target = f"{compare_match.group(1).strip()} vs {compare_match.group(2).strip()}"
    note_parts = []
    if diff:
        note_parts.append(f"Found {diff.get('peak_count', 0)} peaks, {diff.get('anomaly_count', 0)} anomalies")
    if target:
        note_parts.append(f"Requested: {target}")
    if not note_parts:
        note_parts.append("Comparison recorded (no quantitative data to diff)")
    return {"note": "; ".join(note_parts), "diff": diff}


# === 自检 ===

if __name__ == "__main__":
    import asyncio
    from types import SimpleNamespace

    # 1) execute_visual_inspect: 无视觉数据 → 返 failure
    async def _test_no_visual():
        eng = SimpleNamespace(_last_visual_context="", _visual_base64="")
        r = await execute_visual_inspect(eng, "zoom into band 3", {})
        assert r["success"] is False
        assert "No visual data" in r["error"]
    asyncio.run(_test_no_visual())

    # 2) execute_visual_inspect: zoom 动作 (无 base64, 只记坐标)
    async def _test_zoom():
        eng = SimpleNamespace(
            _last_visual_context="[band] peak=<point>[500,800]</point>(2.5)",
            _visual_base64="",
        )
        r = await execute_visual_inspect(eng, "zoom into [100,100] [200,200]", {})
        assert r["success"] is True
        assert len(r["actions"]) == 1
        assert r["actions"][0]["action"] == "zoom"
        assert r["actions"][0]["region"] == [100, 100, 200, 200]
    asyncio.run(_test_zoom())

    # 3) measure_nearest_primitive: 5 种格式变体
    ctx = (
        "[band] peak=<point>[500,800]</point>(2.5)\n"
        "[dos] anomalies=<point>[300,400]</point>=1.2\n"
        "[phase] coverage=<point>[700,200]</point>=85%\n"
        "[scores] f1=<point>[900]</point>=0.75\n"
    )
    # 变体 1: <point>[x,y]</point>(value)
    m = measure_nearest_primitive(510, 810, ctx)
    assert m["coordinate"] == [500, 800]
    assert m["value"] == 2.5
    # 变体 2: <point>[x,y]</point>=value
    m = measure_nearest_primitive(290, 410, ctx)
    assert m["coordinate"] == [300, 400]
    assert m["value"] == 1.2
    # 变体 3: <point>[x,y]</point>=value%
    m = measure_nearest_primitive(710, 210, ctx)
    assert m["coordinate"] == [700, 200]
    assert m["value"] == 85.0
    # 变体 5: key=<point>[y]</point>=value (单坐标, x=0)
    m = measure_nearest_primitive(0, 900, ctx)
    assert m["coordinate"] == [0, 900]
    assert m["value"] == 0.75
    assert m["label"] == "f1"

    # 4) measure_nearest_primitive: 空上下文 → {}
    assert measure_nearest_primitive(0, 0, "") == {}

    # 5) extract_text_visual_features: 段落 + 趋势 + 异常
    ctx2 = (
        "[band] trend=increasing peak=<point>[500,800]</point>(2.5) "
        "min=<point>[100,200]</point>(0.1) anomalies=<point>[700,300]</point>,<point>[750,350]</point>\n"
        "[dos] trend=flat\n"
    )
    r = extract_text_visual_features(ctx2)
    assert "band.trend=increasing" in r["features"]
    assert "band.anomalies=2" in r["features"]
    assert "dos.trend=flat" in r["features"]
    assert "band" in r["summary"]
    assert "dos" in r["summary"]
    # sections 结构
    band_sec = next(s for s in r["sections"] if s["name"] == "band")
    assert band_sec["trend"] == "increasing"
    assert band_sec["peak"] == 2.5
    assert band_sec["min"] == 0.1
    assert band_sec["anomaly_count"] == 2

    # 6) compare_visual_data: 文本级差分
    ctx3 = (
        "[band] peak=<point>[500,800]</point>(2.5) peak=<point>[600,900]</point>(3.1) "
        "min=<point>[100,200]</point>(0.1) anomalies=<point>[700,300]</point>\n"
    )
    r = compare_visual_data("compare band 3 and band 5", ctx3)
    assert r["diff"]["peak_count"] == 2
    assert r["diff"]["peak_range"] == [2.5, 3.1]
    assert r["diff"]["anomaly_count"] == 1
    assert "band 3 vs band 5" in r["note"]

    # 7) compare_visual_data: 空上下文 → 默认 note
    r = compare_visual_data("compare something", "")
    assert r["note"] == "No visual context to compare"
    assert r["diff"] == {}

    # 8) annotate_visual_features: 无图片无上下文 → "No features found"
    r = annotate_visual_features("annotate defects", "", "")
    assert "No features found" in r["note"]
    assert r["features"] == []

    # 9) annotate_visual_features: 文本特征提取
    r = annotate_visual_features("annotate band structure", "", ctx2)
    assert any("band.trend" in f for f in r["features"])
    assert r["tool_output"] is not None
    assert "text_analysis" in r["tool_output"]

    # 10) execute_visual_inspect: measure 动作调 measure_nearest_primitive
    async def _test_measure():
        eng = SimpleNamespace(
            _last_visual_context="[band] peak=<point>[500,800]</point>(2.5)",
            _visual_base64="",
        )
        r = await execute_visual_inspect(eng, "measure peak at [510,810]", {})
        assert r["success"] is True
        assert len(r["actions"]) == 1
        assert r["actions"][0]["action"] == "measure"
        assert r["actions"][0]["nearest_primitive"]["value"] == 2.5
    asyncio.run(_test_measure())

    # 11) execute_visual_inspect: zoom 动作有 base64 时调 vision_describe
    #     构造一个真实的小 PNG base64, 验证 description 字段被填充 (无论 Tier 是否可用)
    async def _test_zoom_with_vision_describe():
        import base64 as b64
        import io as _io
        from PIL import Image
        buf = _io.BytesIO()
        Image.new("RGB", (50, 50), (200, 100, 50)).save(buf, format="PNG")
        visual_b64 = b64.b64encode(buf.getvalue()).decode()
        eng = SimpleNamespace(
            _last_visual_context="[band] peak=<point>[500,800]</point>(2.5)",
            _visual_base64=visual_b64,
        )
        r = await execute_visual_inspect(eng, "zoom into [100,100] [200,200]", {})
        assert r["success"] is True
        action = r["actions"][0]
        assert action["action"] == "zoom"
        assert action["region"] == [100, 100, 200, 200]
        assert "crop_size" in action
        # vision_describe 调过 (成功走 description, 失败走 description_unavailable)
        assert "description" in action or "description_unavailable" in action, (
            f"zoom 后既无 description 也无 description_unavailable: {action.keys()}"
        )
    asyncio.run(_test_zoom_with_vision_describe())

    print("all self-checks passed")
