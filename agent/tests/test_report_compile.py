"""Phase 5c LaTeX 编译测试.

4 测:
  1. pdflatex 不存在时降级 (mock shutil.which → None)
  2. 正常编译 mock (mock subprocess.run 两次返回 0 + 造 .pdf)
  3. 二次编译失败报错 (第一次 ok, 第二次非零)
  4. engine 切换 (xelatex)
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from huginn.tools.report_tool import ReportTool, ReportToolInput


class TestCompilePdf:
    """compile_pdf action 行为."""

    @pytest.mark.asyncio
    async def test_engine_not_found_degrades(self) -> None:
        """shutil.which 返回 None → success=False + 安装提示."""
        tool = ReportTool()
        args = ReportToolInput(
            action="compile_pdf",
            tex_source=r"\documentclass{article}\begin{document}Hi\end{document}",
            engine="pdflatex",
        )
        with patch("shutil.which", return_value=None):
            result = await tool.call(args, context=None)
        assert not result.success
        assert "not found" in (result.error or "") or "PATH" in (result.error or "")

    @pytest.mark.asyncio
    async def test_successful_compile_mock(self, tmp_path) -> None:
        """mock subprocess.run 两次都返回 0, 造一个 .pdf 文件."""
        tool = ReportTool()
        args = ReportToolInput(
            action="compile_pdf",
            tex_source=r"\documentclass{article}\begin{document}Hi\end{document}",
            engine="pdflatex",
        )

        def fake_run(cmd, cwd, capture_output, text, timeout):
            # 造一个 .pdf 文件
            tex_path = cmd[-1]
            from pathlib import Path

            pdf_path = Path(tex_path).with_suffix(".pdf")
            pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")
            log_path = Path(tex_path).with_suffix(".log")
            log_path.write_text("This is a fake log.", encoding="utf-8")
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "ok"
            mock_proc.stderr = ""
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/pdflatex"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            result = await tool.call(args, context=None)
        assert result.success, f"compile should succeed: {result.error}"
        assert result.data["pdf_base64"] is not None
        assert result.data["pdf_size_bytes"] > 0
        assert result.data["engine"] == "pdflatex"

    @pytest.mark.asyncio
    async def test_second_pass_failure(self) -> None:
        """第一次 ok, 第二次非零 → success=False."""
        tool = ReportTool()
        args = ReportToolInput(
            action="compile_pdf",
            tex_source=r"\documentclass{article}\begin{document}Hi\end{document}",
            engine="pdflatex",
        )

        call_count = [0]

        def fake_run(cmd, cwd, capture_output, text, timeout):
            call_count[0] += 1
            mock_proc = MagicMock()
            if call_count[0] == 1:
                mock_proc.returncode = 0
                mock_proc.stdout = "pass 1 ok"
            else:
                mock_proc.returncode = 1
                mock_proc.stderr = "pass 2 failed"
                mock_proc.stdout = ""
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/pdflatex"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            result = await tool.call(args, context=None)
        assert not result.success
        assert "Pass 2" in (result.error or "") or "failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_engine_switch_xelatex(self) -> None:
        """engine=xelatex 时, shutil.which 检测 xelatex."""
        tool = ReportTool()
        args = ReportToolInput(
            action="compile_pdf",
            tex_source=r"\documentclass{article}\begin{document}Hi\end{document}",
            engine="xelatex",
        )

        which_calls: list[str] = []

        def fake_which(name):
            which_calls.append(name)
            return None  # 模拟 xelatex 不存在

        with patch("shutil.which", side_effect=fake_which):
            result = await tool.call(args, context=None)
        assert not result.success
        assert "xelatex" in which_calls
