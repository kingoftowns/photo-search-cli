"""On-demand thumbnail generation with a disk cache.

Used by the FastAPI service to serve browser-friendly JPEGs for photos in
the source directory (including HEIC originals from iPhones). The cache is
keyed by ``sha1(file_path) + size`` so repeat requests are cheap.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

# Register HEIC support once at import time.
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


def _cache_key(file_path: str, size: int) -> str:
    return f"{hashlib.sha1(file_path.encode()).hexdigest()}_{size}.jpg"


class ThumbnailCache:
    """Generates and caches JPEG thumbnails on disk."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, file_path: str, size: int) -> Path:
        return self.cache_dir / _cache_key(file_path, size)

    def get_or_generate(self, file_path: str, size: int = 400) -> Path:
        """Return a path to the cached thumbnail, generating if needed."""
        dest = self.get_path(file_path, size)
        if dest.is_file() and dest.stat().st_size > 0:
            return dest

        src = Path(file_path)
        if not src.is_file():
            raise FileNotFoundError(file_path)

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            tmp = dest.with_suffix(".tmp")
            im.save(tmp, format="JPEG", quality=82, optimize=True)
            os.replace(tmp, dest)
        return dest


def transcode_to_jpeg(file_path: str, max_dim: Optional[int] = None) -> bytes:
    """Open a source photo and return its JPEG bytes, optionally resized.

    Used to serve HEIC originals to browsers that can't render them.
    """
    with Image.open(file_path) as im:
        im = ImageOps.exif_transpose(im)
        if max_dim is not None:
            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue()
