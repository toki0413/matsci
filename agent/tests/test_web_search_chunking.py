"""锁定 web_search_tool 的 P1-6 正文清洗/切块升级.

- ``_html_to_text``: 去 <nav>/<header>/<footer>/<aside> 噪音 + 块级换行 + 实体解码
- ``_chunk_text``: 按段落/句子边界切块, 不跨段拼上下文, 超长单句硬切

全部本地可跑, 不打真实网络.
"""

from __future__ import annotations

from huginn.tools.web_search_tool import WebSearchTool


class TestHtmlToTextP16:
    def test_removes_nav_header_footer_aside_noise(self):
        html = (
            "<html><head><title>T</title></head><body>"
            "<nav>Home About Contact</nav>"
            "<header>Site banner ad</header>"
            "<article>This is the real article body content.</article>"
            "<aside>Related links sidebar</aside>"
            "<footer>Copyright 2026</footer>"
            "</body></html>"
        )
        text = WebSearchTool._html_to_text(html)
        assert "real article body" in text
        assert "Home About" not in text
        assert "banner ad" not in text
        assert "Related links" not in text
        assert "Copyright" not in text

    def test_removes_script_style_blocks(self):
        html = "<style>.x{}</style><script>alert(1)</script><p>visible text</p>"
        text = WebSearchTool._html_to_text(html)
        assert text == "visible text"

    def test_decodes_html_entities(self):
        text = WebSearchTool._html_to_text("<p>a&amp;b &lt;c&gt; &quot;d&quot;</p>")
        assert "a&b <c> \"d\"" in text

    def test_block_tags_become_newlines(self):
        text = WebSearchTool._html_to_text("<p>para one</p><p>para two</p>")
        assert "para one" in text
        assert "para two" in text
        assert "\n" in text

    def test_empty_input(self):
        assert WebSearchTool._html_to_text("") == ""


class TestChunkTextP16:
    def test_keeps_paragraphs_as_semantic_units(self):
        text = "para one sentence. para one second. \n\n para two content here."
        chunks = WebSearchTool._chunk_text(text, size=60)
        # 两个段落分别成块, 不跨段拼接
        assert len(chunks) == 2
        assert "para one" in chunks[0] and "para one second" in chunks[0]
        assert "para two" in chunks[1]

    def test_single_short_paragraph_single_chunk(self):
        text = "Only a short paragraph here."
        chunks = WebSearchTool._chunk_text(text, size=600)
        assert chunks == ["Only a short paragraph here."]

    def test_oversized_sentence_hard_split(self):
        long = "x" * 100
        text = long
        chunks = WebSearchTool._chunk_text(text, size=30)
        assert all(len(c) <= 30 for c in chunks)
        assert "".join(chunks) == long

    def test_empty_input(self):
        assert WebSearchTool._chunk_text("", size=600) == []
        assert WebSearchTool._chunk_text("   \n\n  ", size=600) == []

    def test_no_chunk_exceeds_size(self):
        text = ("sentence one here. " * 5 + "\n\n" + "sentence two here. " * 5)
        chunks = WebSearchTool._chunk_text(text, size=50)
        assert all(len(c) <= 50 for c in chunks)
