"""VisualInspectMixin - visual_inspect 方法族, 从 engine.py 下沉.

P2 slim-down: 5 个 visual inspect 方法从 engine.py 迁入, 定义为 mixin class.
engine 通过多继承接入, 方法内通过 self 访问 engine 状态字段
(_last_visual_context / _visual_base64) 和同级 visual 方法
(_measure_nearest_primitive / _annotate_visual_features /
_extract_text_visual_features / _compare_visual_data).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class VisualInspectMixin:
    """visual_inspect 方法族. 通过 self 访问 engine 状态."""

    async def _execute_visual_inspect(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Path C: 交互式视觉检查. 让 agent 主动调用视觉工具检查上一轮结果.

        这是 OpenThinkIMG 式的工具调用路径 — agent 在推理过程中主动选择
        "放大图表某区域"或"测量某数据点", 而不是被动接收预处理好的视觉基元.
        使用已有的 image_analysis_tool / visual_hook 基础设施, 不新建工具.

        description 解析: "zoom into band 3 near [500,800]" / "measure peak at [999,999]"
        坐标是 0-999 归一化的视觉原语坐标 (路径 B 格式).
        """
        import re

        result: dict[str, Any] = {
            "mode": "visual_inspect",
            "description": description,
            "actions": [],
        }

        # 获取上一轮的视觉基元和 base64 图片
        visual_ctx = getattr(self, "_last_visual_context", "")
        visual_base64 = getattr(self, "_visual_base64", "")

        if not visual_ctx and not visual_base64:
            return {
                **result,
                "success": False,
                "error": "No visual data from previous iteration to inspect",
            }

        # 解析 description 中的动作
        desc_lower = description.lower()

        # 动作 1: zoom — 放大某区域
        if "zoom" in desc_lower:
            # 提取坐标 [x1,y1,x2,y2] 或 [x,y]
            coords = re.findall(r"\[?(\d+)\s*,\s*(\d+)\]?", description)
            if len(coords) >= 2:
                x1, y1 = int(coords[0][0]), int(coords[0][1])
                x2, y2 = int(coords[1][0]), int(coords[1][1])
                # 把 0-999 坐标转成数据索引 (如果有上一轮的原始数据)
                action_result = {
                    "action": "zoom",
                    "region": [x1, y1, x2, y2],
                    "note": f"Zoomed into region [{x1},{y1}]-[{x2},{y2}]",
                }
                # 如果有 visual_base64, 调用 image_analysis_tool 做真正的区域分析
                if visual_base64:
                    try:
                        from huginn.tools.registry import ToolRegistry

                        img_tool = ToolRegistry.get("image_analysis_tool")
                        if img_tool:
                            # 裁剪 base64 图片到指定区域并分析
                            import base64 as b64
                            import io as _io

                            try:
                                from PIL import Image

                                img_data = b64.b64decode(visual_base64)
                                img = Image.open(_io.BytesIO(img_data))
                                w, h = img.size
                                # 0-999 → pixel coordinates
                                px1 = int(x1 / 999 * w)
                                py1 = int(y1 / 999 * h)
                                px2 = int(x2 / 999 * w)
                                py2 = int(y2 / 999 * h)
                                cropped = img.crop((px1, py1, px2, py2))
                                buf = _io.BytesIO()
                                cropped.save(buf, format="PNG")
                                cropped_b64 = b64.b64encode(buf.getvalue()).decode()
                                action_result["cropped_image"] = cropped_b64[
                                    :10000
                                ]  # limit size
                                action_result["crop_size"] = [px2 - px1, py2 - py1]
                            except ImportError:
                                action_result[
                                    "note"
                                ] += " (PIL not available, coordinates only)"
                            except Exception as e:
                                action_result["note"] += f" (crop failed: {e})"
                    except Exception:
                        logger.debug("image crop action failed", exc_info=True)
                result["actions"].append(action_result)

        # 动作 2: measure — 测量某点或区域的数据值
        # v8 补全: 解析 visual_ctx 里的 <point>[x,y]</point> 原语, 找最接近的数据点
        elif "measure" in desc_lower:
            coords = re.findall(r"\[?(\d+)\s*,\s*(\d+)\]?", description)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                # 从 visual_ctx 解析所有 <point>[x,y]</point>=value 原语, 找最近的
                measured = self._measure_nearest_primitive(x, y, visual_ctx)
                result["actions"].append(
                    {
                        "action": "measure",
                        "coordinate": [x, y],
                        "note": f"Measured at <point>[{x},{y}]</point>",
                        "nearest_primitive": measured,
                        "visual_context_snippet": (
                            visual_ctx[:300] if visual_ctx else ""
                        ),
                    }
                )

        # 动作 3: annotate — 标注结构特征
        # v8 补全: 有图片时调 image_analysis_tool 做真正结构标注, 无图片用文本特征
        elif "annotate" in desc_lower:
            annotation = self._annotate_visual_features(description, visual_base64, visual_ctx)
            result["actions"].append(
                {
                    "action": "annotate",
                    "description": description,
                    "note": annotation["note"],
                    "features": annotation.get("features", []),
                    "visual_context": visual_ctx[:500] if visual_ctx else "",
                }
            )
            if annotation.get("tool_output"):
                result["actions"][-1]["tool_output"] = annotation["tool_output"]

        # 动作 4: compare — 比较两组数据
        # v8 补全: 用 extract_comparative_primitives 做真正差分
        elif "compare" in desc_lower:
            comparison = self._compare_visual_data(description, visual_ctx)
            result["actions"].append(
                {
                    "action": "compare",
                    "description": description,
                    "visual_context": visual_ctx[:500] if visual_ctx else "",
                    "note": comparison["note"],
                    "diff": comparison.get("diff", {}),
                }
            )

        # 默认: 记录检查请求
        else:
            result["actions"].append(
                {
                    "action": "inspect",
                    "description": description,
                    "visual_context": visual_ctx[:500] if visual_ctx else "",
                }
            )

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

    def _measure_nearest_primitive(
        self, x: int, y: int, visual_ctx: str
    ) -> dict[str, Any]:
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
        import re

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
                "coordinate": [0, py],  # 单坐标 x=0
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

    def _annotate_visual_features(
        self, description: str, visual_base64: str, visual_ctx: str
    ) -> dict[str, Any]:
        """v8: 标注结构特征. 有图片调 image_analysis_tool, 无图片用文本特征.

        B2 增强: 无图片时解析 visual_ctx 段落结构 + 趋势 + 异常聚类,
        不只做单点 regex 提取. 让文本路径也有结构化标注.

        ponytail: 优先用已有 image_analysis_tool (defect_detect/phase_field 场景),
        失败降级到 visual_ctx 文本特征提取. 不新建工具.
        """
        import re

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
                    # 根据描述选场景: 含 "defect" → defect_detect, 含 "phase" → phase_field
                    desc_lower = description.lower()
                    if "defect" in desc_lower or "缺陷" in description:
                        scene = "defect_detect"
                    elif "phase" in desc_lower or "相" in description:
                        scene = "phase_field"
                    else:
                        scene = "sem_analysis"  # 默认 SEM 分析
                    # 调工具 (同步 call, 不是 async)
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
            structured = self._extract_text_visual_features(visual_ctx)
            features.extend(structured["features"])
            if structured["summary"]:
                # 如果有 tool_output, 把文本 summary 作为补充; 否则作为主 note
                if tool_output is None:
                    tool_output = {"text_analysis": structured["summary"]}
                else:
                    tool_output["text_analysis"] = structured["summary"]
        note = "Annotated " + ", ".join(features[:5]) if features else "No features found"
        return {"note": note, "features": features, "tool_output": tool_output}

    def _extract_text_visual_features(self, visual_ctx: str) -> dict[str, Any]:
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
            # section 标题: [section_name] ...
            sec_match = re.match(r"\[(\w+)\]", line)
            if sec_match:
                sec_name = sec_match.group(1)
                sec: dict[str, Any] = {"name": sec_name, "raw": line[:200]}

                # trend
                trend_m = re.search(r"trend=(\w+)", line)
                if trend_m:
                    sec["trend"] = trend_m.group(1)
                    features.append(f"{sec_name}.trend={trend_m.group(1)}")

                # peak / min
                peak_m = re.search(r"peak=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", line)
                if peak_m:
                    sec["peak"] = float(peak_m.group(1))
                min_m = re.search(r"min=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", line)
                if min_m:
                    sec["min"] = float(min_m.group(1))

                # mean / std
                mean_m = re.search(r"mean=([\d.\-eE]+)", line)
                if mean_m:
                    sec["mean"] = float(mean_m.group(1))
                std_m = re.search(r"std=([\d.\-eE]+)", line)
                if std_m:
                    sec["std"] = float(std_m.group(1))

                # anomalies (可能多个, 用逗号分隔)
                anom_m = re.search(r"anomalies=([^,\n]+(?:,\s*[^,\n]+)*)", line)
                if anom_m and anom_m.group(1).strip() != "none":
                    anom_str = anom_m.group(1)
                    anom_count = anom_str.count("<point>")
                    sec["anomaly_count"] = anom_count
                    if anom_count > 0:
                        features.append(f"{sec_name}.anomalies={anom_count}")

                sections.append(sec)
                # summary 行
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

    def _compare_visual_data(
        self, description: str, visual_ctx: str
    ) -> dict[str, Any]:
        """v8: 比较两组数据. 用 extract_comparative_primitives 做差分.

        ponytail: visual_hook.extract_comparative_primitives 已有, 直接复用.
        但它需要 baseline + current 两个 dict, visual_ctx 是文本. 这里做文本级
        差分: 解析两组 <point> 原语, 算峰值位移/新异常. 升级路径才换真正的
        baseline vs current dict 比较.
        """
        if not visual_ctx:
            return {"note": "No visual context to compare", "diff": {}}
        # 文本级差分: 找 visual_ctx 里的关键指标, 算数量级
        import re
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
        # 检查描述里有没有指定比较对象 (e.g. "compare band 3 and band 5")
        compare_match = re.search(r"compare\s+(\w+\s*\d*)\s+(?:and|with|vs\.?)\s+(\w+\s*\d*)", description, re.IGNORECASE)
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


