"""FastAPI service powering the photo-search web UI.

Run locally with::

    uvicorn api:app --reload --port 8000

The package is organised as:

- :mod:`api.app`        -- FastAPI app + lifespan + CORS
- :mod:`api.routes`     -- HTTP endpoints
- :mod:`api.schemas`    -- Pydantic response models
- :mod:`api.paths`      -- URL-safe base64 path encoding + safety
- :mod:`api.thumbnails` -- on-demand thumbnail cache

Imports the existing :mod:`photo_search` core package for config, storage,
and embedding — this layer only adds HTTP plumbing.
"""

from api.app import app

__all__ = ["app"]
