"""Configuration loading and validation for photo-search.

Loads settings from config.yaml with environment variable overrides for
sensitive values. Config file is resolved from the current working directory
first, then from the package directory as a fallback.

Environment variable overrides use the PHOTO_SEARCH_ prefix with double
underscore for nesting, e.g.:
    PHOTO_SEARCH_POSTGRES__CONNECTION_STRING=postgresql://...
    PHOTO_SEARCH_OLLAMA__BASE_URL=http://...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PhotosConfig(BaseModel):
    """Configuration for photo source directory and file type filtering."""

    source_dir: str
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".heic", ".jpg", ".jpeg", ".png", ".tiff"]
    )
    skip_extensions: list[str] = Field(
        default_factory=lambda: [".mov", ".mp4", ".aac", ".m4a", ".aae", ".gif"]
    )


class OllamaConfig(BaseModel):
    """Configuration for Ollama vision and embedding model endpoints."""

    base_url: str = "http://localhost:11434"
    vision_model: str = "qwen2.5vl:7b"
    embedding_model: str = "nomic-embed-text"
    request_timeout: int = 120


class FacesConfig(BaseModel):
    """Configuration for InsightFace face detection and recognition."""

    model_pack: str = "buffalo_l"
    similarity_threshold: float = 0.4
    min_face_size: int = 20


class QdrantConfig(BaseModel):
    """Configuration for the Qdrant vector database connection."""

    url: str = "http://localhost:6333"
    collection_name: str = "photos"
    vector_size: int = 768


class PostgresConfig(BaseModel):
    """Configuration for the PostgreSQL metadata database."""

    connection_string: str = "postgresql://photouser:changeme@localhost:5432/photosearch"


class GeocodingConfig(BaseModel):
    """Configuration for reverse geocoding of GPS coordinates."""

    enabled: bool = True


class PipelineConfig(BaseModel):
    """Configuration for the indexing pipeline runtime behavior."""

    batch_log_interval: int = 10
    max_retries: int = 3
    retry_delay: int = 5
    resize_max_dimension: int = 1536


class AppConfig(BaseSettings):
    """Root application configuration aggregating all subsections.

    Values are loaded from config.yaml and can be overridden by environment
    variables prefixed with PHOTO_SEARCH_. Nested keys use double underscores,
    e.g. PHOTO_SEARCH_POSTGRES__CONNECTION_STRING.
    """

    model_config = SettingsConfigDict(
        env_prefix="PHOTO_SEARCH_",
        env_nested_delimiter="__",
    )

    photos: PhotosConfig = Field(default_factory=PhotosConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    faces: FacesConfig = Field(default_factory=FacesConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    geocoding: GeocodingConfig = Field(default_factory=GeocodingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def _resolve_config_path(path: str | None) -> Path:
    """Resolve the config.yaml file path.

    Search order:
        1. Explicit path argument (if provided).
        2. config.yaml in the current working directory.
        3. config.yaml next to this Python package (the repo root).

    Raises:
        FileNotFoundError: If no config.yaml can be found in any location.
    """
    if path is not None:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Specified config file not found: {path}")

    # Current working directory
    cwd_candidate = Path.cwd() / "config.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate

    # Package directory (photo-search repo root, one level up from photo_search/)
    package_candidate = Path(__file__).resolve().parent.parent / "config.yaml"
    if package_candidate.is_file():
        return package_candidate

    raise FileNotFoundError(
        "config.yaml not found. Searched in:\n"
        f"  1. {cwd_candidate}\n"
        f"  2. {package_candidate}\n"
        "Create a config.yaml or pass an explicit path to load_config()."
    )


def load_config(path: str | None = None) -> AppConfig:
    """Load application configuration from YAML with environment variable overrides.

    Args:
        path: Optional explicit path to a config.yaml file. When None the
              function searches the current directory and then the package
              directory.

    Returns:
        A fully validated AppConfig instance.

    Raises:
        FileNotFoundError: If no config file can be located.
        yaml.YAMLError: If the config file contains invalid YAML.
        pydantic.ValidationError: If the loaded values fail validation.
    """
    config_path = _resolve_config_path(path)

    try:
        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        print(f"Error parsing {config_path}: {exc}", file=sys.stderr)
        raise

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"config.yaml must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )

    # Pydantic Settings will merge the dict values with env var overrides
    # automatically when we pass them as keyword arguments.
    return AppConfig(**raw)
