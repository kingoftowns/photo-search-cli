"""Pydantic response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from api.paths import encode_path
from photo_search.models import SearchResult


class PhotoResult(BaseModel):
    file_path: str
    file_name: str
    score: float
    caption: Optional[str] = None
    faces: list[str] = []
    date_taken: Optional[datetime] = None
    location_name: Optional[str] = None
    camera: Optional[str] = None
    path_token: str
    thumb_url: str
    original_url: str

    @classmethod
    def from_search(cls, r: SearchResult) -> "PhotoResult":
        token = encode_path(r.file_path)
        return cls(
            file_path=r.file_path,
            file_name=r.file_name,
            score=r.score,
            caption=r.caption,
            faces=r.faces,
            date_taken=r.date_taken,
            location_name=r.location_name,
            camera=r.camera,
            path_token=token,
            thumb_url=f"/thumbs/{token}",
            original_url=f"/originals/{token}",
        )


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[PhotoResult]


class FaceIdentity(BaseModel):
    label: str
    display_name: str
    sample_count: int


class FacesResponse(BaseModel):
    count: int
    faces: list[FaceIdentity]


class LocationSuggestion(BaseModel):
    display: str
    city: Optional[str] = None
    region: Optional[str] = None
    country_code: Optional[str] = None
    photo_count: int


class LocationsResponse(BaseModel):
    count: int
    locations: list[LocationSuggestion]


class StatusResponse(BaseModel):
    total: int
    exif_extracted: int
    faces_extracted: int
    faces_classified: int
    captioned: int
    embedded: int
    errors: int
    qdrant_vectors: int
