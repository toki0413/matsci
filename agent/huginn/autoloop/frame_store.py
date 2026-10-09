"""FrameStore — VISTA 式无损观测记忆 (默认关, 由 visual_frame_memory flag 控制).

与 execution_ledger 互补: 台账只留"看到什么数"的紧凑数值快照并丢掉原帧;
本 store 把每次 execute 产出的视觉帧**原样**落盘, 按稳定 frame_id 建索引,
支持后续无损回看 / 裁剪任意区域 / 读取指定坐标的精确像素值 —— 即 VISTA 的
lossless visual memory + active visual inspection 在我们架构里的对应物.

设计 (ponytail):
- 纯本地文件, 一个 index.json + 每帧一个文件, 不引数据库; workspace 内的
  `.huginn/frames/` 与既有 `.huginn/alignment_dataset.json` 同范式.
- PIL 只用于区域裁剪/取像素的**按需**路径; 未安装时 capture 仍原样落盘,
  region/read_pixels 优雅返回 {"error": ...}, 不让观测记忆带挂主循环.
- 全部 best-effort: 任何异常都不向调用方抛, 返回 None / {"error": ...}.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 魔数 → 扩展名 (只覆盖最常见的几种; 其余按 bin 存, 不影响回看)
_MAGIC_EXT: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF8", "gif"),
    (b"BM", "bmp"),
    (b"RIFF", "webp"),
)


def decode_b64_image(value: Any) -> bytes | None:
    """把 base64 字符串 (可带 data: URL 前缀) 解成原始字节; 失败返回 None."""
    if not value:
        return None
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("data:") and "," in text[:64]:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=False)
    except Exception:  # 防御: 非法 base64 → 当作无帧
        logger.debug("frame b64 decode failed", exc_info=True)
        return None


def _guess_ext(raw: bytes) -> str:
    for magic, ext in _MAGIC_EXT:
        if raw.startswith(magic):
            return ext
    return "bin"


class FrameStore:
    """按 frame_id 索引的无损视觉帧存储 (workspace/.huginn/frames/)."""

    def __init__(self, workspace: str | Path) -> None:
        self.dir = Path(workspace) / ".huginn" / "frames"
        self.index_path = self.dir / "index.json"
        self._index: list[dict[str, Any]] | None = None

    # ── 索引读写 ────────────────────────────────────────────────────
    def _load(self) -> list[dict[str, Any]]:
        if self._index is not None:
            return self._index
        try:
            if self.index_path.is_file():
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self._index = data if isinstance(data, list) else []
            else:
                self._index = []
        except Exception:  # 防御: 损坏索引当空, 不阻断
            logger.debug("frame index load failed", exc_info=True)
            self._index = []
        return self._index

    def _persist(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(
                json.dumps(self._load(), ensure_ascii=False), encoding="utf-8"
            )
        except Exception:  # 防御: 落盘失败不影响已捕获的帧文件
            logger.debug("frame index persist failed", exc_info=True)

    # ── 捕获 ────────────────────────────────────────────────────────
    def capture(
        self,
        image_b64: Any,
        *,
        turn: Any = None,
        tool: str = "",
        kind: str = "image",
        note: str = "",
    ) -> dict[str, Any] | None:
        """原样存一帧, 返回索引记录 (含 frame_id); 无有效图像 → None."""
        raw = decode_b64_image(image_b64)
        if not raw:
            return None
        idx = self._load()
        fid = len(idx) + 1
        ext = _guess_ext(raw)
        fname = f"frame_{fid:04d}.{ext}"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / fname).write_bytes(raw)
        except Exception:  # 防御: 写盘失败 → 不落索引, 保持索引与实际文件一致
            logger.debug("frame write failed", exc_info=True)
            return None

        w = h = None
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
        except Exception:  # 防御: 尺寸读取失败不影响存储
            logger.debug("frame size probe failed", exc_info=True)

        rec = {
            "frame_id": fid,
            "file": fname,
            "turn": turn,
            "tool": tool,
            "kind": kind,
            "ts": time.time(),
            "bytes": len(raw),
            "w": w,
            "h": h,
            "note": note,
        }
        idx.append(rec)
        self._persist()
        return dict(rec)

    # ── 查询 ────────────────────────────────────────────────────────
    def frames(self) -> list[dict[str, Any]]:
        """按捕获顺序列出全部帧记录 (深拷贝浅层字段)."""
        return [dict(r) for r in self._load()]

    def get_bytes(self, frame_id: Any) -> bytes | None:
        for r in self._load():
            if r.get("frame_id") == frame_id:
                p = self.dir / str(r.get("file", ""))
                if p.is_file():
                    return p.read_bytes()
                return None
        return None

    def get_record(self, frame_id: Any) -> dict[str, Any] | None:
        for r in self._load():
            if r.get("frame_id") == frame_id:
                return dict(r)
        return None

    def last_frame_id(self) -> int | None:
        idx = self._load()
        return idx[-1]["frame_id"] if idx else None

    # ── 主动检视: 区域回看 / 像素读取 ─────────────────────────────────
    def region(self, frame_id: Any, box: list[float]) -> dict[str, Any]:
        """裁剪 frame 的 box 区域, 返回裁剪图 base64 (无损回看).

        box = [x0, y0, x1, y1]. 四个值都落在 [0,1] 时按**归一化比例**解释,
        否则按**像素**坐标. 缺 PIL / 帧不存在 → {"error": ...}.
        """
        raw = self.get_bytes(frame_id)
        if raw is None:
            return {"error": f"frame {frame_id} not found"}
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGB")
                W, H = im.size
                x0, y0, x1, y1 = (float(v) for v in box[:4])
                if all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
                    x0, y0, x1, y1 = x0 * W, y0 * H, x1 * W, y1 * H
                left, top = int(x0), int(y0)
                right = max(int(x1), left + 1)
                bottom = max(int(y1), top + 1)
                crop = im.crop((left, top, right, bottom))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                return {
                    "frame_id": frame_id,
                    "box": [x0, y0, x1, y1],
                    "source_size": [W, H],
                    "crop_size": [crop.width, crop.height],
                    "image_b64": base64.b64encode(buf.getvalue()).decode(),
                }
        except Exception as e:  # 防御: 裁剪失败返回错误, 不抛
            return {"error": f"region failed: {e}"}

    def read_pixels(
        self, frame_id: Any, points: list[list[float]], *, normalized: bool = False
    ) -> dict[str, Any]:
        """读取若干坐标的精确 RGB (VISTA read_pixels 同构).

        points: [[x, y], ...]. normalized=True 时按 0-999 归一化坐标换算成像素
        (对齐 visual_hook 的 <point>[x,y]</point> 原语). 越界点标 in_bounds=False.
        """
        raw = self.get_bytes(frame_id)
        if raw is None:
            return {"error": f"frame {frame_id} not found"}
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGB")
                W, H = im.size
                out: list[dict[str, Any]] = []
                for p in points or []:
                    if not isinstance(p, list | tuple) or len(p) < 2:
                        continue
                    fx, fy = float(p[0]), float(p[1])
                    if normalized:
                        px, py = int(fx / 999.0 * (W - 1)), int(fy / 999.0 * (H - 1))
                    else:
                        px, py = int(fx), int(fy)
                    in_bounds = 0 <= px < W and 0 <= py < H
                    rgb = list(im.getpixel((px, py))) if in_bounds else None
                    out.append(
                        {
                            "x": px,
                            "y": py,
                            "in_bounds": in_bounds,
                            "rgb": rgb,
                            "hex": ("#%02X%02X%02X" % tuple(rgb)) if rgb else None,
                        }
                    )
                return {"frame_id": frame_id, "size": [W, H], "pixels": out}
        except Exception as e:  # 防御: 取像素失败返回错误, 不抛
            return {"error": f"read_pixels failed: {e}"}


if __name__ == "__main__":
    # self-check: 落一帧 → 回看/裁剪/取像素 全链路. 无 PIL 时只验证捕获+读取原始字节.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = FrameStore(td)
        assert store.frames() == [], "fresh store should be empty"

        # 造一张 3x2 PNG (优先 PIL; 无 PIL 用最小合法 PNG 字节)
        made = False
        try:
            import io

            from PIL import Image

            im = Image.new("RGB", (3, 2), (10, 20, 30))
            im.putpixel((1, 0), (200, 100, 50))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            made = True
        except Exception:
            b64 = base64.b64encode(
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
            ).decode()

        rec = store.capture(b64, turn=1, tool="code_lab")
        assert rec and rec["frame_id"] == 1, f"capture failed: {rec}"
        assert store.frames()[0]["tool"] == "code_lab"
        assert store.get_bytes(1) is not None
        assert store.last_frame_id() == 1
        assert store.capture("") is None, "empty input should not store"

        if made:
            px = store.read_pixels(1, [[1, 0], [99, 99]])
            assert px["size"] == [3, 2], px
            assert px["pixels"][0]["rgb"] == [200, 100, 50], px
            assert px["pixels"][1]["in_bounds"] is False, px
            rg = store.region(1, [0.0, 0.0, 1.0, 1.0])
            assert rg.get("crop_size") == [3, 2], rg
        print("FrameStore self-check PASS")