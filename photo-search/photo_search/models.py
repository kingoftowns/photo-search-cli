"""Pydantic models for data flowing through the photo-search pipeline.

These models represent every stage of processing -- from raw EXIF metadata
extraction, through face detection/identification and captioning, to the
final indexed record stored in Qdrant and the search results returned to
the user.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PhotoMetadata(BaseModel):
    """Extracted metadata from a photo file (EXIF and filesystem)."""

    file_path: str
    file_name: str
    file_size_bytes: int
    file_type: str  # extension like "HEIC", "JPG"
    date_taken: Optional[datetime] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    camera: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    iso: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    orientation: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}


class DetectedFace(BaseModel):
    """A face detected in a photo by InsightFace."""

    bbox: tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    embedding: list[float]  # 512-dim ArcFace embedding as list

    model_config = {"arbitrary_types_allowed": True}


class IdentifiedFace(DetectedFace):
    """A detected face that has been matched to a known identity."""

    label: str = "unknown"
    similarity: float = 0.0


class PhotoCaption(BaseModel):
    """VLM-generated caption for a photo."""

    caption: str
    model: str
    generation_time_seconds: float


class IndexedPhoto(BaseModel):
    """Complete record for a fully processed photo.

    Aggregates metadata, detected/identified faces, the VLM caption,
    reverse-geocoded location, and the text embedding vector that gets
    stored in Qdrant.
    """

    metadata: PhotoMetadata
    faces: list[IdentifiedFace] = Field(default_factory=list)
    caption: Optional[PhotoCaption] = None
    location_name: Optional[str] = None
    text_embedding: Optional[list[float]] = None  # 768-dim nomic embedding


class IndexingStatus(BaseModel):
    """Per-file tracking of pipeline stages.

    Each boolean flag corresponds to one stage in the indexing pipeline.
    A file is considered fully indexed when ``embedded`` is True.
    """

    file_path: str
    exif_extracted: bool = False
    faces_extracted: bool = False
    faces_classified: bool = False
    captioned: bool = False
    embedded: bool = False
    error: Optional[str] = None
    last_updated: Optional[datetime] = None


class SearchResult(BaseModel):
    """A single search result returned from a Qdrant vector query."""

    file_path: str
    file_name: str
    score: float
    caption: Optional[str] = None
    faces: list[str] = Field(default_factory=list)
    date_taken: Optional[datetime] = None
    location_name: Optional[str] = None
    camera: Optional[str] = None
