"""EXIF metadata extraction from photos with HEIC support."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pillow_heif import register_heif_opener

from photo_search.models import PhotoMetadata

register_heif_opener()

logger = logging.getLogger(__name__)

# EXIF tag IDs used for extraction
_TAG_DATETIME_ORIGINAL = 36867
_TAG_GPS_INFO = 34853
_TAG_MAKE = 271
_TAG_MODEL = 272
_TAG_FOCAL_LENGTH = 37386
_TAG_FNUMBER = 33437
_TAG_ISO = 34855
_TAG_ORIENTATION = 274


def _gps_to_decimal(gps_coords: tuple[Any, ...], gps_ref: str) -> float:
    """Convert GPS coordinates from DMS (degrees, minutes, seconds) to decimal.

    Args:
        gps_coords: Tuple of three values (degrees, minutes, seconds).
            Each value may be an IFDRational or a plain number.
        gps_ref: Reference direction - one of 'N', 'S', 'E', 'W'.

    Returns:
        Decimal degrees as a float. Negative for South and West.
    """
    degrees = float(gps_coords[0])
    minutes = float(gps_coords[1])
    seconds = float(gps_coords[2])

    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if gps_ref in ("S", "W"):
        decimal = -decimal

    return decimal


def _parse_exif_datetime(date_str: str) -> Optional[datetime]:
    """Parse an EXIF datetime string into a Python datetime.

    Handles multiple common EXIF datetime formats including optional timezone
    offsets and sub-second precision.

    Args:
        date_str: Raw datetime string from EXIF data.

    Returns:
        Parsed datetime object, or None if parsing fails.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Strip null bytes and whitespace that sometimes appear in EXIF data
    date_str = date_str.strip().strip("\x00")

    if not date_str:
        return None

    # Formats to try, ordered from most common to least common
    formats = [
        "%Y:%m:%d %H:%M:%S",           # Standard EXIF: "2024:06:15 14:30:00"
        "%Y-%m-%d %H:%M:%S",           # ISO-like: "2024-06-15 14:30:00"
        "%Y:%m:%d %H:%M:%S%z",         # With timezone: "2024:06:15 14:30:00+02:00"
        "%Y-%m-%d %H:%M:%S%z",         # ISO with tz: "2024-06-15 14:30:00+02:00"
        "%Y:%m:%dT%H:%M:%S",           # T-separated: "2024:06:15T14:30:00"
        "%Y-%m-%dT%H:%M:%S",           # ISO 8601: "2024-06-15T14:30:00"
        "%Y-%m-%dT%H:%M:%S%z",         # ISO 8601 with tz
        "%Y:%m:%d %H:%M:%S.%f",        # With sub-seconds
        "%Y-%m-%d %H:%M:%S.%f",        # ISO with sub-seconds
    ]

    # Some cameras embed timezone as "+HH:MM" after a space, e.g.
    # "2024:06:15 14:30:00 +02:00" -- collapse the space before the sign
    # so strptime %z can handle it.
    cleaned = date_str
    for sep in (" +", " -"):
        if sep in cleaned:
            idx = cleaned.rfind(sep)
            # Only treat it as a tz offset if the part after looks like HH:MM
            suffix = cleaned[idx + 1:]
            if len(suffix) in (5, 6) and ":" in suffix:
                cleaned = cleaned[:idx] + cleaned[idx].replace(" ", "") + suffix
                break

    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    logger.warning("Could not parse EXIF datetime: %r", date_str)
    return None


def extract_metadata(file_path: str) -> PhotoMetadata:
    """Extract EXIF metadata from a photo file.

    Opens the image, reads EXIF tags, and returns a PhotoMetadata model with
    whatever information could be successfully extracted. Individual field
    extraction failures are logged as warnings and never crash the function.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        A PhotoMetadata instance. Required fields (file_path, file_name,
        file_size_bytes, file_type) are always populated. Optional EXIF
        fields are set to None when extraction fails.

    Raises:
        FileNotFoundError: If the file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.
    """
    # -- File-system metadata (always available if file exists) ---------------
    abs_path = os.path.abspath(file_path)
    file_name = os.path.basename(abs_path)
    file_size_bytes = os.path.getsize(abs_path)

    ext = os.path.splitext(file_name)[1].upper().lstrip(".")
    file_type = ext if ext else "UNKNOWN"

    # -- Open the image -------------------------------------------------------
    image = Image.open(abs_path)

    # -- Collect raw EXIF dict ------------------------------------------------
    # Use getexif() as the primary interface — it handles HEIC IFD nesting.
    exif_data: dict[int, Any] = {}
    exif_ifd: dict[int, Any] = {}
    gps_ifd: dict[int, Any] = {}
    try:
        exif_obj = image.getexif()
        if exif_obj:
            exif_data = dict(exif_obj)
            # Sub-IFDs: ExifIFD (0x8769) contains DateTimeOriginal, FNumber, etc.
            try:
                exif_ifd = dict(exif_obj.get_ifd(0x8769))
            except Exception:
                pass
            # GPS IFD (0x8825) contains lat/lon/altitude.
            try:
                gps_ifd = dict(exif_obj.get_ifd(0x8825))
            except Exception:
                pass
    except Exception:
        # Fallback to _getexif() for older Pillow / plain JPEGs.
        try:
            raw = image._getexif()
            if raw is not None:
                exif_data = raw
        except (AttributeError, Exception):
            logger.warning("No EXIF data found in %s", file_path)

    # Merge ExifIFD into main dict so tag-based lookups work uniformly.
    # Sub-IFD tags take priority (they are the authoritative source).
    merged = {**exif_data, **exif_ifd}

    # -- Width and height -----------------------------------------------------
    width: Optional[int] = None
    height: Optional[int] = None
    try:
        width, height = image.size
    except Exception:
        logger.warning("Could not get image dimensions for %s", file_path)

    # -- DateTimeOriginal -----------------------------------------------------
    # Tag 36867 lives in the ExifIFD sub-IFD, so use `merged` which includes it.
    date_taken: Optional[datetime] = None
    try:
        raw_date = merged.get(_TAG_DATETIME_ORIGINAL)
        if raw_date is not None:
            date_taken = _parse_exif_datetime(str(raw_date))
    except Exception:
        logger.warning("Could not extract date_taken from %s", file_path)

    # -- GPS info -------------------------------------------------------------
    # GPS data lives in the GPS IFD (0x8825).  For HEIC files, `gps_ifd`
    # (populated via getexif().get_ifd(0x8825)) is the reliable source.
    # For JPEGs, exif_data[_TAG_GPS_INFO] may already be a nested dict.
    # We try gps_ifd first, then fall back to the old approach.
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    try:
        gps_dict: dict[str, Any] = {}

        if gps_ifd:
            # gps_ifd is keyed by numeric GPS sub-tag IDs (e.g. 1, 2, 3, 4).
            for key, val in gps_ifd.items():
                tag_name = GPSTAGS.get(key, key)
                gps_dict[tag_name] = val
        else:
            # Fallback: some Pillow/JPEG combos put a nested dict at _TAG_GPS_INFO.
            gps_info = exif_data.get(_TAG_GPS_INFO)
            if gps_info is not None and isinstance(gps_info, dict):
                for key, val in gps_info.items():
                    tag_name = GPSTAGS.get(key, key)
                    gps_dict[tag_name] = val

        if gps_dict:
            lat_dms = gps_dict.get("GPSLatitude")
            lat_ref = gps_dict.get("GPSLatitudeRef")
            lon_dms = gps_dict.get("GPSLongitude")
            lon_ref = gps_dict.get("GPSLongitudeRef")

            if lat_dms is not None and lat_ref is not None:
                gps_lat = _gps_to_decimal(lat_dms, str(lat_ref))
            if lon_dms is not None and lon_ref is not None:
                gps_lon = _gps_to_decimal(lon_dms, str(lon_ref))
    except Exception:
        logger.warning("Could not extract GPS data from %s", file_path, exc_info=True)

    # -- Camera make + model --------------------------------------------------
    camera: Optional[str] = None
    try:
        make = merged.get(_TAG_MAKE)
        model = merged.get(_TAG_MODEL)
        parts: list[str] = []
        if make:
            parts.append(str(make).strip())
        if model:
            model_str = str(model).strip()
            # Avoid redundant prefix: "Apple iPhone 15" instead of
            # "Apple Apple iPhone 15 Pro" when model already starts with make.
            if parts and model_str.lower().startswith(parts[0].lower()):
                parts = [model_str]
            else:
                parts.append(model_str)
        if parts:
            camera = " ".join(parts)
    except Exception:
        logger.warning("Could not extract camera info from %s", file_path)

    # -- Focal length ---------------------------------------------------------
    # Tag 37386 lives in ExifIFD, so use `merged`.
    focal_length: Optional[float] = None
    try:
        raw_fl = merged.get(_TAG_FOCAL_LENGTH)
        if raw_fl is not None:
            focal_length = float(raw_fl)
    except Exception:
        logger.warning("Could not extract focal length from %s", file_path)

    # -- Aperture (f-number) --------------------------------------------------
    # Tag 33437 lives in ExifIFD, so use `merged`.
    aperture: Optional[float] = None
    try:
        raw_fn = merged.get(_TAG_FNUMBER)
        if raw_fn is not None:
            aperture = float(raw_fn)
    except Exception:
        logger.warning("Could not extract aperture from %s", file_path)

    # -- ISO ------------------------------------------------------------------
    # Tag 34855 lives in ExifIFD, so use `merged`.
    iso: Optional[int] = None
    try:
        raw_iso = merged.get(_TAG_ISO)
        if raw_iso is not None:
            # Some cameras store a tuple; take the first value.
            if isinstance(raw_iso, (tuple, list)):
                iso = int(raw_iso[0])
            else:
                iso = int(raw_iso)
    except Exception:
        logger.warning("Could not extract ISO from %s", file_path)

    # -- Orientation ----------------------------------------------------------
    orientation: Optional[int] = None
    try:
        raw_orient = merged.get(_TAG_ORIENTATION)
        if raw_orient is not None:
            orientation = int(raw_orient)
    except Exception:
        logger.warning("Could not extract orientation from %s", file_path)

    return PhotoMetadata(
        file_path=abs_path,
        file_name=file_name,
        file_size_bytes=file_size_bytes,
        file_type=file_type,
        date_taken=date_taken,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        camera=camera,
        focal_length=focal_length,
        aperture=aperture,
        iso=iso,
        width=width,
        height=height,
        orientation=orientation,
    )
