"""本地视觉解码器 `huginn.vision.local_decoder` 的测试。

不真起 ollama, 全部用 mock 把网络层换成可控假对象, 覆盖:
无模型 / 模型可用 / 前缀匹配 / 解码成功失败 / 图片编码 / 缓存清空。
"""

from __future__ import annotations

from unittest import mock

from huginn.vision import local_decoder as ld


def _clear():
    ld.clear_cache()


def test_available_false_when_no_model():
    _clear()
    with mock.patch.object(ld, "_list_models", return_value=[]):
        assert ld.available() is False


def test_available_true_and_caches():
    _clear()
    with mock.patch.object(ld, "_list_models", return_value=["qwen2.5-vl:7b"]):
        assert ld.available() is True
        # 第二次命中缓存, 不再探 /api/tags
        assert ld.available() is True
        assert ld._list_models.call_count == 1  # type: ignore[attr-defined]


def test_find_model_prefix_matches_largest():
    _clear()
    names = ["embeddings", "qwen2.5", "qwen2.5-vl:7b", "qwen2.5-vl:14b"]
    with mock.patch.object(ld, "_list_models", return_value=names):
        # 无精确匹配时取前缀下名字最长者
        assert ld._find_model(ld._ollama_host()) == "qwen2.5-vl:14b"


def test_find_model_prefers_exact():
    _clear()
    names = ["qwen2.5-vl:7b", "qwen2.5-vl"]
    with mock.patch.object(ld, "_list_models", return_value=names):
        assert ld._find_model(ld._ollama_host()) == "qwen2.5-vl"


def test_decode_returns_none_without_model():
    _clear()
    with mock.patch.object(ld, "available", return_value=False):
        assert ld.decode_image("/tmp/a.png") is None


def test_decode_success_returns_text():
    _clear()
    body = mock.Mock()
    body.read.return_value = '{"message": {"content": " 针状晶粒, 沿晶界分布  "}}'.encode()
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = body
    with (
        mock.patch.object(ld, "available", return_value=True),
        mock.patch.object(ld, "_find_model", return_value="qwen2.5-vl:7b"),
        mock.patch.object(ld, "_encode_image", return_value="aGk="),
        mock.patch.object(ld.urllib.request, "urlopen", return_value=ctx),
    ):
        assert ld.decode_image(b"\x89PNG") == "针状晶粒, 沿晶界分布"


def test_decode_network_failure_returns_none():
    _clear()
    with (
        mock.patch.object(ld, "available", return_value=True),
        mock.patch.object(ld, "_find_model", return_value="qwen2.5-vl:7b"),
        mock.patch.object(ld, "_encode_image", return_value="aGk="),
        mock.patch.object(
            ld.urllib.request, "urlopen", side_effect=OSError("connection refused")
        ),
    ):
        assert ld.decode_image("/tmp/a.png") is None


def test_encode_image_bytes_and_path(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\x00")
    assert ld._encode_image(bytes(p.read_bytes())) == "iVBORwA="
    assert ld._encode_image(str(p)) == "iVBORwA="
    assert ld._encode_image(tmp_path / "missing.png") is None


def test_clear_cache_empties():
    _clear()
    with mock.patch.object(ld, "_list_models", return_value=["qwen2.5-vl:7b"]):
        assert ld.available() is True
        ld.clear_cache()
        assert ld._CACHE == {}
