"""content_parts 模块的单元测试.

覆盖点:
  * TextPart — to_openai_format / to_anthropic_format / from_dict
  * ImageURLPart — URL 格式、data URI 格式、from_file、to_anthropic (base64 提取)
  * StructurePart — to_openai_format (代码块)、from_dict 全字段
  * PlotPart — from_file、to_openai / to_anthropic 格式
  * ContentPart.from_dict — 按 type 分发, 未知类型回退 TextPart
  * content_to_message — 字符串直通、list 转 openai/anthropic
  * mark_as_temp — 设置 _no_save 标志
  * Registry — 所有子类自动注册
"""

from __future__ import annotations

import base64

import pytest

from huginn.content_parts import (
    ContentPart,
    ImageURLPart,
    PlotPart,
    StructurePart,
    TextPart,
    content_to_message,
)


# ════════════════════════════════════════════════════════════════════
# TextPart
# ════════════════════════════════════════════════════════════════════


class TestTextPart:
    def test_to_openai_format(self):
        part = TextPart(text="hello world")
        assert part.to_openai_format() == {"type": "text", "text": "hello world"}

    def test_to_anthropic_format(self):
        part = TextPart(text="hello world")
        assert part.to_anthropic_format() == {"type": "text", "text": "hello world"}

    def test_from_dict(self):
        part = ContentPart.from_dict({"type": "text", "text": "from dict"})
        assert isinstance(part, TextPart)
        assert part.text == "from dict"

    def test_from_dict_defaults_empty_text(self):
        part = ContentPart.from_dict({"type": "text"})
        assert isinstance(part, TextPart)
        assert part.text == ""

    def test_default_type_is_text(self):
        part = TextPart(text="x")
        assert part.type == "text"


# ════════════════════════════════════════════════════════════════════
# ImageURLPart
# ════════════════════════════════════════════════════════════════════


class TestImageURLPart:
    def test_url_to_openai_format(self):
        part = ImageURLPart(image_url="https://example.com/img.png", detail="high")
        assert part.to_openai_format() == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png", "detail": "high"},
        }

    def test_url_default_detail_is_auto(self):
        part = ImageURLPart(image_url="https://example.com/img.png")
        assert part.to_openai_format()["image_url"]["detail"] == "auto"

    def test_data_uri_to_anthropic_extracts_base64(self):
        # data URI 应被解析成 Anthropic 的 base64 source
        part = ImageURLPart(image_url="data:image/png;base64,SGVsbG8=")
        result = part.to_anthropic_format()
        assert result == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "SGVsbG8=",
            },
        }

    def test_jpeg_data_uri_media_type(self):
        part = ImageURLPart(image_url="data:image/jpeg;base64,abc==")
        result = part.to_anthropic_format()
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "abc=="

    def test_url_to_anthropic_format(self):
        # 普通 URL 在 Anthropic 侧用 url source
        part = ImageURLPart(image_url="https://example.com/img.png")
        assert part.to_anthropic_format() == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.png"},
        }

    def test_from_dict_with_dict_image_url(self):
        part = ContentPart.from_dict(
            {"type": "image_url", "image_url": {"url": "https://x.com/i.png", "detail": "high"}}
        )
        assert isinstance(part, ImageURLPart)
        assert part.image_url == "https://x.com/i.png"
        assert part.detail == "high"

    def test_from_dict_with_str_image_url(self):
        part = ContentPart.from_dict(
            {"type": "image_url", "image_url": "https://x.com/i.png"}
        )
        assert isinstance(part, ImageURLPart)
        assert part.image_url == "https://x.com/i.png"

    def test_from_file_png(self, tmp_path):
        # 从本地 PNG 文件构造, base64 编码后可还原
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        png_path = tmp_path / "test.png"
        png_path.write_bytes(raw)

        part = ImageURLPart.from_file(png_path)
        assert part.image_url.startswith("data:image/png;base64,")
        b64 = part.image_url.split(",", 1)[1]
        assert base64.b64decode(b64) == raw

    def test_from_file_jpeg_media_type(self, tmp_path):
        raw = b"\xff\xd8\xff" + b"\x00" * 50
        jpg_path = tmp_path / "pic.jpg"
        jpg_path.write_bytes(raw)

        part = ImageURLPart.from_file(jpg_path)
        assert part.image_url.startswith("data:image/jpeg;base64,")


# ════════════════════════════════════════════════════════════════════
# StructurePart
# ════════════════════════════════════════════════════════════════════


class TestStructurePart:
    def test_to_openai_format_is_code_block(self):
        part = StructurePart(format="cif", content="data_test")
        result = part.to_openai_format()
        assert result == {"type": "text", "text": "```cif\ndata_test\n```"}

    def test_to_anthropic_format_is_code_block(self):
        part = StructurePart(format="poscar", content="Si\n1.0")
        result = part.to_anthropic_format()
        assert result["type"] == "text"
        assert "```poscar\nSi\n1.0\n```" == result["text"]

    def test_from_dict_all_fields(self):
        part = ContentPart.from_dict(
            {
                "type": "structure",
                "format": "poscar",
                "content": "Si",
                "formula": "Si2",
                "space_group": "P1",
            }
        )
        assert isinstance(part, StructurePart)
        assert part.format == "poscar"
        assert part.content == "Si"
        assert part.formula == "Si2"
        assert part.space_group == "P1"

    def test_from_dict_defaults(self):
        part = ContentPart.from_dict({"type": "structure"})
        assert isinstance(part, StructurePart)
        assert part.format == "cif"
        assert part.content == ""

    def test_default_type_is_structure(self):
        part = StructurePart(content="x")
        assert part.type == "structure"


# ════════════════════════════════════════════════════════════════════
# PlotPart
# ════════════════════════════════════════════════════════════════════


class TestPlotPart:
    def test_from_file(self, tmp_path):
        raw = b"\x89PNG" + b"\x00" * 50
        plot_path = tmp_path / "band.png"
        plot_path.write_bytes(raw)

        part = PlotPart.from_file(
            plot_path, title="Band Structure", x_label="k-points", y_label="Energy (eV)"
        )
        assert part.image_url.startswith("data:image/png;base64,")
        assert part.title == "Band Structure"
        assert part.x_label == "k-points"
        assert part.y_label == "Energy (eV)"

        # base64 可还原
        b64 = part.image_url.split(",", 1)[1]
        assert base64.b64decode(b64) == raw

    def test_to_openai_format_uses_high_detail(self):
        part = PlotPart(image_url="data:image/png;base64,abc", title="DOS")
        result = part.to_openai_format()
        assert result == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc", "detail": "high"},
        }

    def test_to_anthropic_format_data_uri(self):
        part = PlotPart(image_url="data:image/png;base64,abc")
        result = part.to_anthropic_format()
        assert result == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
        }

    def test_to_anthropic_format_url(self):
        part = PlotPart(image_url="https://example.com/plot.png")
        result = part.to_anthropic_format()
        assert result == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/plot.png"},
        }

    def test_from_dict_all_fields(self):
        part = ContentPart.from_dict(
            {
                "type": "plot",
                "image_url": "data:image/png;base64,abc",
                "title": "T",
                "x_label": "X",
                "y_label": "Y",
            }
        )
        assert isinstance(part, PlotPart)
        assert part.title == "T"
        assert part.x_label == "X"
        assert part.y_label == "Y"

    def test_default_type_is_plot(self):
        part = PlotPart(image_url="x")
        assert part.type == "plot"


# ════════════════════════════════════════════════════════════════════
# ContentPart.from_dict 分发
# ════════════════════════════════════════════════════════════════════


class TestContentPartFromDict:
    def test_dispatches_to_text(self):
        p = ContentPart.from_dict({"type": "text", "text": "hi"})
        assert isinstance(p, TextPart)
        assert p.text == "hi"

    def test_dispatches_to_image_url(self):
        p = ContentPart.from_dict({"type": "image_url", "image_url": {"url": "https://x.com/i.png"}})
        assert isinstance(p, ImageURLPart)
        assert p.image_url == "https://x.com/i.png"

    def test_dispatches_to_structure(self):
        p = ContentPart.from_dict({"type": "structure", "format": "cif", "content": "x"})
        assert isinstance(p, StructurePart)
        assert p.format == "cif"

    def test_dispatches_to_plot(self):
        p = ContentPart.from_dict({"type": "plot", "image_url": "data:image/png;base64,abc", "title": "T"})
        assert isinstance(p, PlotPart)
        assert p.title == "T"

    def test_unknown_type_falls_back_to_text(self):
        # 未知 type 应回退到 TextPart
        p = ContentPart.from_dict({"type": "unknown_type", "text": "fallback"})
        assert isinstance(p, TextPart)
        assert p.text == "fallback"

    def test_missing_type_defaults_to_text(self):
        p = ContentPart.from_dict({"text": "no type field"})
        assert isinstance(p, TextPart)
        assert p.text == "no type field"


# ════════════════════════════════════════════════════════════════════
# content_to_message
# ════════════════════════════════════════════════════════════════════


class TestContentToMessage:
    def test_string_passthrough(self):
        # 字符串原样返回
        assert content_to_message("hello") == "hello"

    def test_list_openai_conversion(self):
        parts = [TextPart(text="hi"), ImageURLPart(image_url="https://x.com/i.png")]
        msg = content_to_message(parts, provider="openai")
        assert msg == [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://x.com/i.png", "detail": "auto"}},
        ]

    def test_list_anthropic_conversion(self):
        parts = [TextPart(text="hi"), ImageURLPart(image_url="https://x.com/i.png")]
        msg = content_to_message(parts, provider="anthropic")
        assert msg == [
            {"type": "text", "text": "hi"},
            {"type": "image", "source": {"type": "url", "url": "https://x.com/i.png"}},
        ]

    def test_default_provider_is_openai(self):
        parts = [TextPart(text="hi")]
        msg = content_to_message(parts)
        assert msg == [{"type": "text", "text": "hi"}]

    def test_empty_list(self):
        assert content_to_message([], provider="openai") == []
        assert content_to_message([], provider="anthropic") == []

    def test_mixed_parts_conversion(self):
        # 混合多种 ContentPart 也能转
        parts = [
            TextPart(text="see structure"),
            StructurePart(format="cif", content="data_Si"),
            PlotPart(image_url="data:image/png;base64,abc"),
        ]
        msg = content_to_message(parts, provider="openai")
        assert len(msg) == 3
        assert msg[0]["type"] == "text"
        assert msg[1]["type"] == "text"
        assert "```cif" in msg[1]["text"]
        assert msg[2]["type"] == "image_url"


# ════════════════════════════════════════════════════════════════════
# mark_as_temp
# ════════════════════════════════════════════════════════════════════


class TestMarkAsTemp:
    def test_sets_no_save_flag(self):
        part = TextPart(text="temp")
        assert part._no_save is False
        part.mark_as_temp()
        assert part._no_save is True

    def test_returns_self_for_chaining(self):
        part = ImageURLPart(image_url="https://x.com/i.png")
        ret = part.mark_as_temp()
        assert ret is part
        assert part._no_save is True

    def test_default_no_save_is_false(self):
        assert TextPart(text="x")._no_save is False
        assert StructurePart(content="x")._no_save is False
        assert PlotPart(image_url="x")._no_save is False


# ════════════════════════════════════════════════════════════════════
# Registry (自动注册)
# ════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_all_subclasses_registered(self):
        # __init_subclass__ 应把每个子类按 type 默认值注册
        assert ContentPart._registry["text"] is TextPart
        assert ContentPart._registry["image_url"] is ImageURLPart
        assert ContentPart._registry["structure"] is StructurePart
        assert ContentPart._registry["plot"] is PlotPart

    def test_registry_keys_count(self):
        # 至少注册了 4 种类型
        assert len(ContentPart._registry) >= 4

    def test_from_dict_uses_registry(self):
        # from_dict 应通过 registry 找到对应类
        assert ContentPart._registry.get("structure") is StructurePart
        p = ContentPart.from_dict({"type": "structure", "content": "abc"})
        assert isinstance(p, StructurePart)


# ════════════════════════════════════════════════════════════════════
# 基类 ContentPart 行为
# ════════════════════════════════════════════════════════════════════


class TestBaseContentPart:
    def test_base_to_openai_format_returns_type(self):
        base = ContentPart()
        assert base.to_openai_format() == {"type": "text"}

    def test_base_to_anthropic_format_returns_type(self):
        base = ContentPart()
        assert base.to_anthropic_format() == {"type": "text"}

    def test_base_from_dict_raises_not_implemented(self):
        # 直接调 _from_dict 应抛 NotImplementedError
        with pytest.raises(NotImplementedError):
            ContentPart._from_dict({"type": "text"})
