"""FastAPI application wiring: lifespan, CORS, shared state."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import api, binary
from api.thumbnails import ThumbnailCache
from photo_search.config import load_config
from photo_search.embed import TextEmbedder
from photo_search.storage import PostgresStorage, QdrantStorage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    pg = PostgresStorage(cfg.postgres.connection_string)
    qd = QdrantStorage(
        url=cfg.qdrant.url,
        collection_name=cfg.qdrant.collection_name,
        vector_size=cfg.qdrant.vector_size,
    )
    embedder = TextEmbedder(
        base_url=cfg.ollama.base_url,
        model=cfg.ollama.embedding_model,
    )
    cache_dir = os.environ.get(
        "PHOTO_SEARCH_THUMB_CACHE", "/var/cache/photo-thumbs"
    )
    try:
        thumb_cache = ThumbnailCache(cache_dir)
    except PermissionError:
        fallback = Path.home() / ".cache" / "photo-search-thumbs"
        logger.warning("Thumb cache %s not writable, using %s", cache_dir, fallback)
        thumb_cache = ThumbnailCache(fallback)

    app.state.cfg = cfg
    app.state.pg = pg
    app.state.qd = qd
    app.state.embedder = embedder
    app.state.thumbs = thumb_cache
    app.state.photos_root = Path(cfg.photos.source_dir).resolve()
    logger.info(
        "photo-search API ready (photos_root=%s, qdrant=%s, postgres=%s)",
        app.state.photos_root,
        cfg.qdrant.url,
        cfg.postgres.connection_string.split("@")[-1],
    )
    try:
        yield
    finally:
        pg.close()


def _build_app() -> FastAPI:
    application = FastAPI(title="photo-search", version="0.1.0", lifespan=lifespan)

    origins = os.environ.get(
        "PHOTO_SEARCH_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    application.include_router(api)
    application.include_router(binary)
    return application


app = _build_app()
