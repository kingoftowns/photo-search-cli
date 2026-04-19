"""URL-safe base64 encoding for absolute file paths, plus traversal safety.

File paths contain slashes and unicode, so we pass them as urlsafe base64
tokens in the URL. ``_resolve_safe`` additionally verifies the decoded
path lives under the configured photos root so a crafted token can't be
used to read arbitrary files on the API host.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

from fastapi import HTTPException, Request


def encode_path(file_path: str) -> str:
    return base64.urlsafe_b64encode(file_path.encode("utf-8")).decode("ascii").rstrip("=")


def decode_path(token: str) -> str:
    padding = "=" * (-len(token) % 4)
    try:
        return base64.urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path token: {exc}")


def resolve_safe(request: Request, token: str) -> Path:
    """Decode ``token`` and return a path guaranteed to be under photos_root.

    If the decoded path lives under any configured
    ``photos.source_dir_aliases`` prefix, the prefix is rewritten to the
    real ``photos_root`` before the traversal check. This keeps tokens
    minted on a workstation (e.g. ``/Volumes/voyager2/Photos/...``) valid
    when the API serves the same files from a different mount
    (e.g. ``/photos``) without requiring a reindex.
    """
    file_path = decode_path(token)
    root: Path = request.app.state.photos_root
    cfg = request.app.state.cfg

    for alias in cfg.photos.source_dir_aliases:
        alias_norm = alias.rstrip("/")
        if not alias_norm:
            continue
        if file_path == alias_norm or file_path.startswith(alias_norm + "/"):
            file_path = str(root) + file_path[len(alias_norm):]
            break

    resolved = Path(file_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path outside photos root")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Photo not found")
    return resolved
