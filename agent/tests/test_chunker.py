"""chunker 模块的单元测试.

覆盖点:
  * RecursiveCharacterChunker — 优先级分隔符切分、overlap 行为、小文本单块、递归回退到字符级
  * MarkdownChunker — 按标题切分、heading_path 元数据、大段子切分、无标题回退
  * StructureChunker — CIF 检测、POSCAR 检测、元数据提取 (空间群/晶格)、未知格式回退
  * AutoChunker — 按扩展名和内容选择正确 chunker
  * Chunk dataclass — 元数据字段
  * _merge_small_chunks — 小块合并
"""

from __future__ import annotations

import pytest

from huginn.knowledge.chunker import (
    AutoChunker,
    BaseChunker,
    Chunk,
    MarkdownChunker,
    RecursiveCharacterChunker,
    StructureChunker,
)


# ════════════════════════════════════════════════════════════════════
# Chunk 数据类
# ════════════════════════════════════════════════════════════════════


class TestChunkDataclass:
    def test_default_metadata_is_empty(self):
        # 不传 metadata 时应该为空 dict
        c = Chunk(text="hello")
        assert c.text == "hello"
        assert c.metadata == {}

    def test_metadata_fields_preserved(self):
        # 自定义 metadata 字段都能保留
        meta = {"source": "paper.md", "chunk_index": 3, "element_type": "text"}
        c = Chunk(text="content", metadata=meta)
        assert c.metadata["source"] == "paper.md"
        assert c.metadata["chunk_index"] == 3
        assert c.metadata["element_type"] == "text"

    def test_each_chunk_has_independent_metadata(self):
        # 默认 factory 应保证各实例 metadata 互不影响
        c1 = Chunk(text="a")
        c2 = Chunk(text="b")
        c1.metadata["x"] = 1
        assert "x" not in c2.metadata


# ════════════════════════════════════════════════════════════════════
# RecursiveCharacterChunker
# ════════════════════════════════════════════════════════════════════


class TestRecursiveCharacterChunker:
    @pytest.mark.asyncio
    async def test_small_text_returns_single_chunk(self):
        # 文本短于 chunk_size 时直接返回单块
        chunker = RecursiveCharacterChunker(chunk_size=800, overlap=100)
        chunks = await chunker.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].metadata["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_splits_by_priority_separator(self):
        # 用 \\n\\n 分隔的段落, chunk_size=30 应按段落切分 (不带 overlap)
        text = (
            "First paragraph here.\n\n"
            "Second paragraph here.\n\n"
            "Third paragraph here."
        )
        chunker = RecursiveCharacterChunker(chunk_size=30, overlap=0)
        chunks = await chunker.chunk(text)

        assert len(chunks) == 3
        # 每块是干净的段落, 不含双换行
        assert "First" in chunks[0].text
        assert "Second" in chunks[1].text
        assert "Third" in chunks[2].text
        assert "\n\n" not in chunks[0].text
        # chunk_index 连续递增
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]
        assert all(c.metadata["separator"] == "recursive" for c in chunks)

    @pytest.mark.asyncio
    async def test_overlap_between_adjacent_chunks(self):
        # overlap=5 时后一块开头应包含前一块末尾的 overlap 字符
        text = (
            "First paragraph here.\n\n"
            "Second paragraph here.\n\n"
            "Third paragraph here."
        )
        chunker = RecursiveCharacterChunker(chunk_size=30, overlap=5)
        chunks = await chunker.chunk(text)

        assert len(chunks) == 3
        overlap_tail = chunks[0].text[-5:]
        assert chunks[1].text.startswith(overlap_tail)

    @pytest.mark.asyncio
    async def test_recursive_fallback_to_char_split(self):
        # 没有任何分隔符的长字符串应回退到字符级切分
        text = "A" * 100
        chunker = RecursiveCharacterChunker(chunk_size=20, overlap=0)
        chunks = await chunker.chunk(text)

        assert len(chunks) == 5
        assert all(c.text == "A" * 20 for c in chunks)
        assert all(c.metadata["separator"] == "recursive" for c in chunks)

    @pytest.mark.asyncio
    async def test_char_split_with_overlap(self):
        # 字符级切分也要支持 overlap
        text = "B" * 50
        chunker = RecursiveCharacterChunker(chunk_size=20, overlap=5)
        chunks = await chunker.chunk(text)

        # 50 字符, 步长 15 (20-5): 起点 0, 15, 30 -> 3 块 (30+20=50 到末尾即停)
        assert len(chunks) == 3
        assert chunks[0].text == "B" * 20
        # 每块都正好 20 字符
        assert all(len(c.text) == 20 for c in chunks)
        # 后一块开头应与前一块末尾重叠
        assert chunks[1].text.startswith(chunks[0].text[-5:])

    @pytest.mark.asyncio
    async def test_custom_separators(self):
        # 自定义分隔符优先级应生效
        text = "part1||part2||part3"
        chunker = RecursiveCharacterChunker(
            chunk_size=10, overlap=0, separators=["||", ""]
        )
        chunks = await chunker.chunk(text)
        assert len(chunks) >= 2
        # 至少把 "part1" 单独切出来
        assert "part1" in chunks[0].text

    @pytest.mark.asyncio
    async def test_kwargs_override_chunk_size(self):
        # 通过 kwargs 临时覆盖 chunk_size
        chunker = RecursiveCharacterChunker(chunk_size=800, overlap=0)
        text = "x" * 50
        chunks = await chunker.chunk(text, chunk_size=20)
        assert len(chunks) > 1
        assert all(len(c.text) <= 20 for c in chunks)

    def test_default_separators_priority(self):
        # 确认默认分隔符优先级顺序
        chunker = RecursiveCharacterChunker()
        assert chunker.separators[0] == "\n\n\n"
        assert chunker.separators[1] == "\n\n"
        assert chunker.separators[-1] == ""


# ════════════════════════════════════════════════════════════════════
# MarkdownChunker
# ════════════════════════════════════════════════════════════════════


class TestMarkdownChunker:
    @pytest.mark.asyncio
    async def test_splits_by_headings(self):
        # 每段内容 > 100 字符, 避免 _merge_small_chunks 把它们合并
        text = (
            "# Title\n\n"
            + "Intro text content here. " * 5 + "\n\n"
            + "## Methods\n\n"
            + "Method details described here. " * 5 + "\n\n"
            + "### SCF\n\n"
            + "SCF convergence info detailed. " * 5
        )
        chunker = MarkdownChunker(chunk_size=800, overlap=50)
        chunks = await chunker.chunk(text)

        assert len(chunks) == 3
        # 每块对应一个标题
        assert chunks[0].metadata["heading_title"] == "Title"
        assert chunks[1].metadata["heading_title"] == "Methods"
        assert chunks[2].metadata["heading_title"] == "SCF"
        assert all(c.metadata["element_type"] == "text" for c in chunks)

    @pytest.mark.asyncio
    async def test_heading_path_metadata(self):
        # heading_path 应反映标题层级, 每段内容 > 100 字符避免合并
        text = (
            "# Title\n\n"
            + "Intro text content here. " * 5 + "\n\n"
            + "## Methods\n\n"
            + "Method details described here. " * 5 + "\n\n"
            + "### SCF\n\n"
            + "SCF convergence info detailed. " * 5
        )
        chunker = MarkdownChunker(chunk_size=800, overlap=50)
        chunks = await chunker.chunk(text)

        assert chunks[0].metadata["heading_path"] == ["Title"]
        assert chunks[1].metadata["heading_path"] == ["Title", "Methods"]
        assert chunks[2].metadata["heading_path"] == ["Title", "Methods", "SCF"]
        # heading_level 与 # 数量一致
        assert chunks[0].metadata["heading_level"] == 1
        assert chunks[1].metadata["heading_level"] == 2
        assert chunks[2].metadata["heading_level"] == 3

    @pytest.mark.asyncio
    async def test_heading_path_pops_on_level_up(self):
        # 从 ### 回到 ## 时, heading_path 应弹出深层标题; 每段 > 100 字符避免合并
        text = (
            "## A\n\n"
            + "content a section text here. " * 5 + "\n\n"
            + "### B\n\n"
            + "content b section text here. " * 5 + "\n\n"
            + "## C\n\n"
            + "content c section text here. " * 5
        )
        chunker = MarkdownChunker(chunk_size=800, overlap=50)
        chunks = await chunker.chunk(text)

        assert chunks[0].metadata["heading_path"] == ["A"]
        assert chunks[1].metadata["heading_path"] == ["A", "B"]
        # 回到 level 2 后, "B" 被弹出, "C" 接在 "A" 后
        assert chunks[2].metadata["heading_path"] == ["A", "C"]

    @pytest.mark.asyncio
    async def test_large_section_sub_chunked(self):
        # 单个标题下内容过大时, 应子切分且共享 heading_path
        big_body = "Sentence about SCF. " * 200
        text = f"# Big Section\n\n{big_body}"
        chunker = MarkdownChunker(chunk_size=200, overlap=20)
        chunks = await chunker.chunk(text)

        assert len(chunks) > 1
        # 所有子块都继承同一 heading_path / heading_title
        assert all(c.metadata["heading_path"] == ["Big Section"] for c in chunks)
        assert all(c.metadata["heading_title"] == "Big Section" for c in chunks)
        assert all(c.metadata["heading_level"] == 1 for c in chunks)

    @pytest.mark.asyncio
    async def test_no_headings_falls_back_to_recursive(self):
        # 无标题时回退到 RecursiveCharacterChunker, 不带 heading_path
        text = "Plain text without headings. " * 50
        chunker = MarkdownChunker(chunk_size=200, overlap=20)
        chunks = await chunker.chunk(text)

        assert len(chunks) >= 1
        # 回退路径不会写 heading_path
        assert all("heading_path" not in c.metadata for c in chunks)

    @pytest.mark.asyncio
    async def test_empty_heading_content_skipped(self):
        # 紧邻的两个标题, 中间无内容应跳过空块
        text = "# A\n\n## B\n\nreal content here."
        chunker = MarkdownChunker(chunk_size=800, overlap=50)
        chunks = await chunker.chunk(text)

        # A 无内容被跳过, 只剩 B 的内容
        assert len(chunks) == 1
        assert chunks[0].text == "real content here."
        assert chunks[0].metadata["heading_path"] == ["A", "B"]


# ════════════════════════════════════════════════════════════════════
# StructureChunker
# ════════════════════════════════════════════════════════════════════


class TestStructureChunker:
    @pytest.mark.asyncio
    async def test_cif_detection_and_metadata(self):
        cif = (
            "data_Si\n"
            "_symmetry_space_group_name_H-M   'Fd-3m'\n"
            "_cell_length_a 5.43\n"
            "_cell_length_b 5.43\n"
            "_cell_length_c 5.43\n"
            "loop_\n"
            "_atom_site_fract_x\n"
            "0.0 0.0 0.0\n"
            "0.25 0.25 0.25\n"
        )
        chunker = StructureChunker()
        chunks = await chunker.chunk(cif)

        assert len(chunks) >= 1
        assert all(c.metadata["format"] == "cif" for c in chunks)
        assert all(c.metadata["element_type"] == "structure" for c in chunks)

        info = chunks[0].metadata["structure_info"]
        assert info["space_group"] == "Fd-3m"
        assert info["lattice_a"] == 5.43
        assert info["lattice_b"] == 5.43
        assert info["lattice_c"] == 5.43
        # 原子坐标行被解析出来
        assert info["atom_count"] == 2

    @pytest.mark.asyncio
    async def test_cif_splits_into_sections(self):
        # data_ 块和 loop_ 块应被切成不同 section
        cif = (
            "data_Si\n"
            "_cell_length_a 5.43\n"
            "loop_\n"
            "_atom_site_fract_x\n"
            "0.0 0.0 0.0\n"
        )
        chunker = StructureChunker()
        chunks = await chunker.chunk(cif)

        # data_ 段 + loop_ 段
        assert len(chunks) == 2
        assert "data_Si" in chunks[0].text
        assert "loop_" in chunks[1].text
        # chunk_index 连续
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]

    @pytest.mark.asyncio
    async def test_poscar_detection_and_metadata(self):
        poscar = (
            "My comment\n"
            "1.0\n"
            "5.0 0.0 0.0\n"
            "0.0 6.0 0.0\n"
            "0.0 0.0 7.0\n"
            "Si\n"
            "2\n"
            "Direct\n"
            "0.0 0.0 0.0\n"
            "0.5 0.5 0.5\n"
        )
        chunker = StructureChunker()
        chunks = await chunker.chunk(poscar)

        assert len(chunks) == 1
        assert chunks[0].metadata["format"] == "poscar"
        assert chunks[0].metadata["element_type"] == "structure"

        info = chunks[0].metadata["structure_info"]
        assert info["comment"] == "My comment"
        assert info["scale"] == "1.0"
        assert info["lattice_vectors"] == [
            [5.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 7.0],
        ]

    @pytest.mark.asyncio
    async def test_poscar_cartesian_keyword_detected(self):
        # 用 Cartesian 关键字也应识别为 POSCAR
        poscar = (
            "Cu slab\n"
            "1.0\n"
            "3.6 0.0 0.0\n"
            "0.0 3.6 0.0\n"
            "0.0 0.0 3.6\n"
            "Cu\n1\nCartesian\n0 0 0\n"
        )
        chunker = StructureChunker()
        chunks = await chunker.chunk(poscar)
        assert chunks[0].metadata["format"] == "poscar"

    @pytest.mark.asyncio
    async def test_unknown_format_falls_back(self):
        # 既不是 CIF 也不是 POSCAR 的文本应回退到递归切分
        text = "Just some plain text. " * 100
        chunker = StructureChunker()
        chunks = await chunker.chunk(text)

        assert len(chunks) >= 1
        # 回退路径不带 format 标记
        assert all("format" not in c.metadata for c in chunks)

    @pytest.mark.asyncio
    async def test_poscar_short_text_handled(self):
        # 极短的 POSCAR 文本不会崩
        chunker = StructureChunker()
        chunks = await chunker.chunk("Direct\n")
        assert chunks[0].metadata["format"] == "poscar"


# ════════════════════════════════════════════════════════════════════
# AutoChunker
# ════════════════════════════════════════════════════════════════════


class TestAutoChunker:
    @pytest.mark.asyncio
    async def test_cif_by_filename(self):
        cif = "data_Si\n_cell_length_a 5.0\n"
        auto = AutoChunker()
        chunks = await auto.chunk(cif, filename="structure.cif")

        assert chunks[0].metadata["format"] == "cif"
        assert chunks[0].metadata["source"] == "structure.cif"

    @pytest.mark.asyncio
    async def test_markdown_by_filename(self):
        md = "# Title\n\nContent here.\n"
        auto = AutoChunker()
        chunks = await auto.chunk(md, filename="paper.md")

        assert chunks[0].metadata["source"] == "paper.md"
        assert "heading_path" in chunks[0].metadata

    @pytest.mark.asyncio
    async def test_poscar_by_filename(self):
        poscar = (
            "comment\n1.0\n5.0 0.0 0.0\n0.0 6.0 0.0\n0.0 0.0 7.0\n"
            "Si\n2\nDirect\n0.0 0.0 0.0\n"
        )
        auto = AutoChunker()
        chunks = await auto.chunk(poscar, filename="struct.poscar")

        assert chunks[0].metadata["format"] == "poscar"
        assert chunks[0].metadata["source"] == "struct.poscar"

    @pytest.mark.asyncio
    async def test_vasp_extension_routed_to_structure(self):
        poscar = (
            "comment\n1.0\n5.0 0.0 0.0\n0.0 6.0 0.0\n0.0 0.0 7.0\n"
            "Si\n1\nDirect\n0.0 0.0 0.0\n"
        )
        auto = AutoChunker()
        chunks = await auto.chunk(poscar, filename="out.vasp")
        assert chunks[0].metadata["format"] == "poscar"

    @pytest.mark.asyncio
    async def test_plain_text_by_filename(self):
        auto = AutoChunker()
        chunks = await auto.chunk("Just plain text.", filename="notes.txt")
        assert chunks[0].metadata["source"] == "notes.txt"
        # 纯文本走递归切分, 不带 format
        assert "format" not in chunks[0].metadata

    @pytest.mark.asyncio
    async def test_cif_by_content_detection(self):
        # 不给扩展名, 靠内容识别 CIF
        cif = "data_Si\n_cell_length_a 5.0\n"
        auto = AutoChunker()
        chunks = await auto.chunk(cif, filename="")
        assert chunks[0].metadata["format"] == "cif"

    @pytest.mark.asyncio
    async def test_markdown_by_content_detection(self):
        # 不给扩展名, 靠内容识别 markdown
        md = "# Title\n\nContent.\n"
        auto = AutoChunker()
        chunks = await auto.chunk(md, filename="")
        assert "heading_path" in chunks[0].metadata

    @pytest.mark.asyncio
    async def test_plain_text_by_content_detection(self):
        # 无特征内容走递归切分
        auto = AutoChunker()
        chunks = await auto.chunk("Some unstructured text.", filename="")
        assert len(chunks) >= 1
        assert "format" not in chunks[0].metadata


# ════════════════════════════════════════════════════════════════════
# _merge_small_chunks
# ════════════════════════════════════════════════════════════════════


class TestMergeSmallChunks:
    def test_small_chunks_merged_into_previous(self):
        chunker = RecursiveCharacterChunker()
        chunks = [
            Chunk(text="A" * 50, metadata={"chunk_index": 0}),
            Chunk(text="B" * 50, metadata={"chunk_index": 1}),
            Chunk(text="C" * 200, metadata={"chunk_index": 2}),
        ]
        merged = chunker._merge_small_chunks(chunks, min_size=100)

        # 前两块 (各 50) 合并成一块, 第三块保留
        assert len(merged) == 2
        assert "A" * 50 in merged[0].text
        assert "B" * 50 in merged[0].text
        assert merged[1].text == "C" * 200

    def test_large_chunks_not_merged(self):
        chunker = RecursiveCharacterChunker()
        chunks = [
            Chunk(text="A" * 200, metadata={"chunk_index": 0}),
            Chunk(text="B" * 200, metadata={"chunk_index": 1}),
        ]
        merged = chunker._merge_small_chunks(chunks, min_size=100)
        assert len(merged) == 2

    def test_empty_input(self):
        chunker = RecursiveCharacterChunker()
        assert chunker._merge_small_chunks([]) == []

    def test_single_chunk_unchanged(self):
        chunker = RecursiveCharacterChunker()
        chunks = [Chunk(text="A" * 50, metadata={"chunk_index": 0})]
        merged = chunker._merge_small_chunks(chunks, min_size=100)
        assert len(merged) == 1
        assert merged[0].text == "A" * 50

    def test_merge_uses_newline_separator(self):
        # 合并时应以换行连接
        chunker = RecursiveCharacterChunker()
        chunks = [
            Chunk(text="first", metadata={"chunk_index": 0}),
            Chunk(text="second", metadata={"chunk_index": 1}),
        ]
        merged = chunker._merge_small_chunks(chunks, min_size=100)
        assert len(merged) == 1
        assert merged[0].text == "first\nsecond"


# ════════════════════════════════════════════════════════════════════
# BaseChunker 默认行为
# ════════════════════════════════════════════════════════════════════


class TestBaseChunker:
    @pytest.mark.asyncio
    async def test_base_chunk_returns_none(self):
        # BaseChunker.chunk 只是占位, 返回 None
        base = BaseChunker()
        result = await base.chunk("text")
        assert result is None
