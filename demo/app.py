"""Huginn · HF Space 演示壳 (Gradio).

一个 Space 内用四个标签页承载全部演示面板:
  1. 缺度追问取证   — 输入一批文献报道值 → 按自由度分组/缺度/取证/门禁
  2. 判别对比       — 未分组 vs 分组的判定差异
  3. MCP 能力       — 3 个 MCP server 工具清单 + 图片主色交互
  4. 系统概览       — 版本 / 能力 / 组成

运行:
    pip install gradio numpy Pillow
    python app.py          # HF Space 默认入口 (launch on 0.0.0.0)
"""
from __future__ import annotations

import os

import gradio as gr

import demo_core as core


# ───────────────────────── 输入解析 ─────────────────────────


def _parse_rows(text: str, papers: bool = False) -> list[dict]:
    """解析 `value,method,unit` 行 (逗号分隔, 每行一条报道值).

    example:
        4.10, DFT-PBE, eV
        5.81, experiment, eV
    """
    rows: list[dict] = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts or parts[0] == "":
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        row = {
            "value": value,
            "method": parts[1] if len(parts) > 1 else "",
            "unit": parts[2] if len(parts) > 2 else "",
            "note": "",
            "doi": f"demo/{i}_{value}",
        }
        rows.append(row)
    return rows


# ───────────────────────── 面板回调 ─────────────────────────


def panel1_default() -> str:
    return core.decompose_values()


def panel1_custom(text: str) -> str:
    rows = _parse_rows(text)
    if not rows:
        return "⚠ 未解析到有效行。格式：`数值, 方法, 单位`（每行一条）。"
    return core.decompose_values(rows)


def panel2_default() -> str:
    return core.contrast_flatten()


def panel2_custom(text: str) -> str:
    rows = _parse_rows(text)
    if not rows:
        return "⚠ 未解析到有效行。格式：`数值, 方法, 单位`（每行一条）。"
    return core.contrast_flatten(rows)


def panel3_tools() -> str:
    return core.mcp_server_tools()


def panel3_colors(image_path: str | None) -> str:
    if not image_path:
        return "上传图片 (或留空) 后点击「提取主色」。"
    try:
        return core.mcp_vision_colors(image_path)
    except Exception as exc:  # noqa: BLE001 - Space 容错
        return f"⚠ 无法解析该图：{exc}"


def panel4_overview() -> str:
    return core.system_overview()


# ───────────────────────── 组装 ─────────────────────────


def build() -> gr.Blocks:
    with gr.Blocks(title="Huginn · Scientific Agent Demo") as demo:
        gr.Markdown(
            "# Huginn · Intelligence for Materials Discovery\n"
            "一个面向科研的 LLM agent：把互相矛盾的文献数值**按真实物理自由度拆开归因**，"
            "**暴露缺度**，**对象级取证**，并执行**门禁不变量**（不假装一致、不捏造自由度）。"
            "本演示零 LLM / 零凭据 / 确定性，可离线复现。"
        )

        with gr.Tabs():
            with gr.Tab("① 缺度追问取证"):
                gr.Markdown("### 看默认样例 —— 7 条报道值如何被拆开归因")
                gr.Markdown(panel1_default())
                gr.Markdown("### 或贴上你自己的数据 (`数值, 方法, 单位`)")
                row_in = gr.Textbox(
                    lines=4,
                    value=("5.81, experiment, eV\n5.79, experiment, eV\n"
                           "4.10, DFT-PBE, eV\n4.05, DFT-PBE, eV\n"
                           "6.00, HSE06, eV\n6.02, HSE06, eV"),
                    label="报道值 (每行一条: 数值, 方法, 单位)",
                )
                row_btn = gr.Button("拆开归因 + 取证 + 门禁")
                row_out = gr.Markdown()
                row_btn.click(panel1_custom, row_in, row_out)

            with gr.Tab("② 判别对比"):
                gr.Markdown("### 未分组 vs 分组的判定差异 —— 为什么「分组」更诚实")
                gr.Markdown(panel2_default())
                gr.Markdown("### 跑你的数据")
                fp_in = gr.Textbox(
                    lines=4,
                    value=("5.81, experiment, eV\n5.79, experiment, eV\n"
                           "4.10, DFT-PBE, eV\n4.05, DFT-PBE, eV\n"
                           "6.00, HSE06, eV\n6.02, HSE06, eV"),
                    label="报道值 (每行一条: 数值, 方法, 单位)",
                )
                fp_btn = gr.Button("对比判定")
                fp_out = gr.Markdown()
                fp_btn.click(panel2_custom, fp_in, fp_out)

            with gr.Tab("③ MCP 能力"):
                gr.Markdown("### 3 个可独立发布的 MCP server（mat-db / math-anything / vision-pixel）")
                gr.Markdown(panel3_tools())
                gr.Markdown("### 试 vision-pixel 取主色（已预置示例图，或上传你自己的）")
                img_in = gr.Image(
                    type="filepath", value="assets/sample_crystal.jpg",
                    label="图片（默认示例：Li2O 晶体质感图）",
                )
                img_btn = gr.Button("提取主色")
                img_out = gr.Markdown()
                img_btn.click(panel3_colors, img_in, img_out)

            with gr.Tab("④ 系统概览"):
                gr.Markdown(panel4_overview())

            with gr.Tab("⑤ 能力集装箱全貌"):
                gr.Markdown(
                    "### 真实注册能力清单（原子 + 复合，可导出/复用/组合）\n"
                    "点击「加载能力清单」离线注册工具池后展示完整清单（约几秒）。"
                )
                cap_btn = gr.Button("加载能力清单")
                cap_out = gr.Markdown("（未加载）")
                cap_btn.click(core.capability_manifest, [], cap_out)

            with gr.Tab("⑥ 符号数学 × Lean4 形式化"):
                gr.Markdown("### 输入 sympy 表达式 → 求导/积分 → 机械翻译成 Lean 4 源码")
                lean_expr = gr.Textbox(value="sin(x)**2 + cos(x)**2", label="表达式 (e.g. `x**2 * exp(x)`)")
                lean_op = gr.Radio(["diff", "integrate"], value="diff", label="操作")
                lean_btn = gr.Button("计算并翻译成 Lean 4")
                lean_out = gr.Markdown()
                lean_btn.click(
                    lambda t, o: core.symbolic_to_lean(t, op=o),
                    [lean_expr, lean_op], lean_out,
                )

        gr.Markdown(
            "\n---\n*Huginn · 不假装一致，也不捏造自由度。源码: "
            "<https://github.com/toki0413/matsci>*"
        )
    return demo


# ───────────────────────── 入口 ─────────────────────────


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    build().launch(server_name="0.0.0.0", server_port=port)