"""HTTP endpoints for the photo-search web UI."""

from __future__ import annotations

import mimetypes
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from api.paths import decode_path, resolve_safe
from api.schemas import (
    FaceIdentity,
    FacesResponse,
    LocationSuggestion,
    LocationsResponse,
    PhotoResult,
    SearchResponse,
    StatusResponse,
)
from api.thumbnails import ThumbnailCache, transcode_to_jpeg
from photo_search.embed import TextEmbedder
from photo_search.geo import split_location_name
from photo_search.storage import PostgresStorage, QdrantStorage

# Routes mounted under /api/* (metadata endpoints)
api = APIRouter(prefix="/api")

# Routes served at / (binary routes — so <img src="/thumbs/..."> works)
binary = APIRouter()


# ---------------------------------------------------------------------------
# /api/* routes
# ---------------------------------------------------------------------------


@api.get("/health")
async def health() -> dict[str, str]:
    # async so the liveness/readiness probe runs on the event loop and
    # can't be starved by sync endpoints (thumbnail generation) that
    # saturate the default threadpool.
    return {"status": "ok"}


@api.get("/search", response_model=SearchResponse)
def search(
    request: Request,
    q: Optional[str] = Query(
        None,
        description=(
            "Natural-language query. Optional: if omitted or blank, results "
            "are browsed by filter alone, ordered by date_taken descending."
        ),
    ),
    person: Optional[list[str]] = Query(
        None,
        description=(
            "Restrict to labeled people. Repeat the parameter or pass a "
            "comma-separated list; all listed people must appear (AND)."
        ),
    ),
    city: Optional[str] = Query(None, description="Lowercase city (e.g. 'woodcrest')"),
    region: Optional[str] = Query(
        None, description="Lowercase state/region (e.g. 'california')"
    ),
    country_code: Optional[str] = Query(
        None, description="ISO2 country code, uppercase (e.g. 'US')"
    ),
    year: Optional[int] = Query(None, ge=1900, le=2100),
    after: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD) lower bound"),
    before: Optional[str] = Query(None, description="ISO date (YYYY-MM-DD) upper bound"),
    top: int = Query(60, ge=1, le=200),
) -> SearchResponse:
    qd: QdrantStorage = request.app.state.qd

    filters: dict[str, Any] = {}
    if person:
        people = [p.strip() for raw in person for p in raw.split(",") if p.strip()]
        if people:
            filters["person"] = people
    if city:
        filters["city"] = city.strip().lower()
    if region:
        filters["region"] = region.strip().lower()
    if country_code:
        filters["country_code"] = country_code.strip().upper()
    if year is not None:
        filters["year"] = year
    if after:
        filters["date_from"] = after
    if before:
        filters["date_to"] = before

    query_text = (q or "").strip()
    # Strip surrounding quotes so `"BEAVERS"` matches captions that include
    # the literal word.  Dense search handles either form fine; it's the
    # keyword leg below that would otherwise miss when the user quotes.
    dequoted = query_text.strip('"').strip("'").strip()

    if query_text:
        embedder: TextEmbedder = request.app.state.embedder
        try:
            vec = embedder.embed_query(query_text)
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        dense_hits = qd.search(
            query_vector=vec, limit=top, filters=filters or None
        )

        # Hybrid: surface photos whose caption literally contains the query
        # text.  Dense vectors buried these for short proper-noun queries
        # (e.g. 'beavers' matched animal photos before a team-name jersey).
        #
        # IMPORTANT: pass the same filters to the keyword leg so person /
        # location / date constraints are honored. Without this, selecting
        # "Henry" + searching "spacex" would return every SpaceX photo
        # (not just Henry's) since Postgres ILIKE alone doesn't know about
        # faces.  Oversample from Postgres so the Qdrant filter still has
        # room after dropping non-matching rows.
        pg: PostgresStorage = request.app.state.pg
        keyword_paths = pg.keyword_match_file_paths(dequoted, limit=200)
        keyword_hits = (
            qd.retrieve_by_file_paths(keyword_paths, filters=filters or None)
            if keyword_paths
            else []
        )

        # Merge keyword hits first (capped so they can't flood the page for
        # broad queries), then fill with dense hits not already included.
        # Assign keyword hits score=1.0 so they sort first in the UI grid.
        keyword_cap = min(len(keyword_hits), max(0, top // 2))
        keyword_slice = keyword_hits[:keyword_cap]
        for r in keyword_slice:
            r.score = 1.0

        seen: set[str] = {r.file_path for r in keyword_slice}
        merged = list(keyword_slice)
        for r in dense_hits:
            if r.file_path in seen:
                continue
            merged.append(r)
            if len(merged) >= top:
                break
        hits = merged
    elif filters:
        hits = qd.browse(limit=top, filters=filters)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a query (q) or at least one filter.",
        )

    results = [PhotoResult.from_search(h) for h in hits]
    return SearchResponse(query=query_text, count=len(results), results=results)


@api.get("/faces", response_model=FacesResponse)
def faces(request: Request) -> FacesResponse:
    pg: PostgresStorage = request.app.state.pg
    rows = pg.get_face_identities()
    identities = [
        FaceIdentity(
            label=r["label"],
            display_name=r.get("display_name") or r["label"],
            sample_count=r.get("sample_count") or 0,
        )
        for r in rows
    ]
    return FacesResponse(count=len(identities), faces=identities)


@api.get("/locations", response_model=LocationsResponse)
def locations(
    request: Request,
    prefix: str = Query("", description="Case-insensitive substring to match"),
    limit: int = Query(20, ge=1, le=100),
) -> LocationsResponse:
    pg: PostgresStorage = request.app.state.pg
    rows = pg.list_locations(prefix=prefix, limit=limit)
    suggestions = []
    for r in rows:
        city, region, cc = split_location_name(r["location_name"])
        suggestions.append(
            LocationSuggestion(
                display=r["location_name"],
                city=city,
                region=region,
                country_code=cc,
                photo_count=r["photo_count"],
            )
        )
    return LocationsResponse(count=len(suggestions), locations=suggestions)


@api.get("/photos/{token}")
def photo_metadata(request: Request, token: str) -> JSONResponse:
    file_path = decode_path(token)
    pg: PostgresStorage = request.app.state.pg
    row = pg.get_photo(file_path)
    if row is None:
        raise HTTPException(status_code=404, detail="Photo not indexed")
    for k, v in list(row.items()):
        if isinstance(v, datetime):
            row[k] = v.isoformat()
    row["path_token"] = token
    row["thumb_url"] = f"/thumbs/{token}"
    row["original_url"] = f"/originals/{token}"
    return JSONResponse(row)


@api.get("/status", response_model=StatusResponse)
def status(request: Request) -> StatusResponse:
    pg: PostgresStorage = request.app.state.pg
    qd: QdrantStorage = request.app.state.qd
    counts = pg.get_all_statuses()
    try:
        qcount = qd.count()
    except Exception:
        qcount = 0
    return StatusResponse(qdrant_vectors=qcount or 0, **counts)


# ---------------------------------------------------------------------------
# Binary routes (thumbs + originals, served at the root for clean URLs)
# ---------------------------------------------------------------------------


@binary.get("/thumbs/{token}")
def thumb(
    request: Request,
    token: str,
    size: int = Query(400, ge=64, le=2048),
) -> FileResponse:
    src = resolve_safe(request, token)
    cache: ThumbnailCache = request.app.state.thumbs
    try:
        path = cache.get_or_generate(str(src), size=size)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@binary.get("/originals/{token}")
def original(request: Request, token: str) -> Response:
    src = resolve_safe(request, token)
    suffix = src.suffix.lower()
    if suffix in {".heic", ".heif"}:
        data = transcode_to_jpeg(str(src))
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    media_type, _ = mimetypes.guess_type(src.name)
    return FileResponse(
        src,
        media_type=media_type or "application/octet-stream",
        headers={"Cache-Control": "public, max-age=3600"},
    )
