"""Offline reverse geocoding using local GeoNames database."""

from __future__ import annotations

import logging
from typing import Optional

import reverse_geocoder as rg

logger = logging.getLogger(__name__)


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Convert GPS coordinates to a human-readable location name.

    Uses the ``reverse_geocoder`` package which bundles an offline GeoNames
    database, so no network access is required after the initial package
    install.

    Args:
        lat: Latitude in decimal degrees (positive = North, negative = South).
        lon: Longitude in decimal degrees (positive = East, negative = West).

    Returns:
        A formatted location string like ``"Irvine, California, US"``, or
        None if the lookup fails or produces no usable result.
    """
    try:
        results = rg.search((lat, lon))
    except Exception:
        logger.warning(
            "Reverse geocoding failed for (%.6f, %.6f)", lat, lon, exc_info=True
        )
        return None

    if not results:
        logger.warning("No reverse geocoding results for (%.6f, %.6f)", lat, lon)
        return None

    result = results[0]

    # Build location string from available components
    # Result dict keys: 'name' (city), 'admin1' (state/region), 'cc' (country)
    city: str = result.get("name", "").strip()
    admin1: str = result.get("admin1", "").strip()
    country_code: str = result.get("cc", "").strip()

    parts: list[str] = []
    if city:
        parts.append(city)
    if admin1:
        parts.append(admin1)
    if country_code:
        parts.append(country_code)

    if not parts:
        logger.warning(
            "Reverse geocoding returned empty result for (%.6f, %.6f)", lat, lon
        )
        return None

    return ", ".join(parts)
