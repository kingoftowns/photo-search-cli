"""Small lookup tables for location-filter UX.

`reverse_geocode()` stores locations as ``"City, AdminFullName, CountryCode"``
(e.g. ``"Irvine, California, US"``, ``"Florence, Tuscany, IT"``).  Users type
more convenient forms:

- ``"Laguna Beach, CA"``   -> match region ``"california"``
- ``"Zion National Park, Utah"`` -> match region ``"utah"``
- ``"Florence, Italy"``   -> match country_code ``"IT"``
- ``"Amsterdam, Netherlands"`` -> match country_code ``"NL"``

These helpers translate the user's token into the value actually stored in
Qdrant's payload, so the filter compares like-for-like.
"""
from __future__ import annotations

from typing import Optional

# Full lowercase name -> itself; 2-letter abbrev -> full lowercase name.
# Stored lowercase because we lowercase region in the payload too.
_US_STATES: dict[str, str] = {
    "al": "alabama",         "alabama": "alabama",
    "ak": "alaska",          "alaska": "alaska",
    "az": "arizona",         "arizona": "arizona",
    "ar": "arkansas",        "arkansas": "arkansas",
    "ca": "california",      "california": "california",
    "co": "colorado",        "colorado": "colorado",
    "ct": "connecticut",     "connecticut": "connecticut",
    "de": "delaware",        "delaware": "delaware",
    "dc": "district of columbia", "district of columbia": "district of columbia",
    "fl": "florida",         "florida": "florida",
    "ga": "georgia",         "georgia": "georgia",
    "hi": "hawaii",          "hawaii": "hawaii",
    "id": "idaho",           "idaho": "idaho",
    "il": "illinois",        "illinois": "illinois",
    "in": "indiana",         "indiana": "indiana",
    "ia": "iowa",            "iowa": "iowa",
    "ks": "kansas",          "kansas": "kansas",
    "ky": "kentucky",        "kentucky": "kentucky",
    "la": "louisiana",       "louisiana": "louisiana",
    "me": "maine",           "maine": "maine",
    "md": "maryland",        "maryland": "maryland",
    "ma": "massachusetts",   "massachusetts": "massachusetts",
    "mi": "michigan",        "michigan": "michigan",
    "mn": "minnesota",       "minnesota": "minnesota",
    "ms": "mississippi",     "mississippi": "mississippi",
    "mo": "missouri",        "missouri": "missouri",
    "mt": "montana",         "montana": "montana",
    "ne": "nebraska",        "nebraska": "nebraska",
    "nv": "nevada",          "nevada": "nevada",
    "nh": "new hampshire",   "new hampshire": "new hampshire",
    "nj": "new jersey",      "new jersey": "new jersey",
    "nm": "new mexico",      "new mexico": "new mexico",
    "ny": "new york",        "new york": "new york",
    "nc": "north carolina",  "north carolina": "north carolina",
    "nd": "north dakota",    "north dakota": "north dakota",
    "oh": "ohio",            "ohio": "ohio",
    "ok": "oklahoma",        "oklahoma": "oklahoma",
    "or": "oregon",          "oregon": "oregon",
    "pa": "pennsylvania",    "pennsylvania": "pennsylvania",
    "ri": "rhode island",    "rhode island": "rhode island",
    "sc": "south carolina",  "south carolina": "south carolina",
    "sd": "south dakota",    "south dakota": "south dakota",
    "tn": "tennessee",       "tennessee": "tennessee",
    "tx": "texas",           "texas": "texas",
    "ut": "utah",            "utah": "utah",
    "vt": "vermont",         "vermont": "vermont",
    "va": "virginia",        "virginia": "virginia",
    "wa": "washington",      "washington": "washington",
    "wv": "west virginia",   "west virginia": "west virginia",
    "wi": "wisconsin",       "wisconsin": "wisconsin",
    "wy": "wyoming",         "wyoming": "wyoming",
}

# Common name / alias -> ISO 3166-1 alpha-2.  Covers the countries we actually
# have photos in plus a bunch of likely-asked-for neighbours.  Intentionally
# small — extend as needed.
_COUNTRIES: dict[str, str] = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "america": "US",
    "canada": "CA",
    "mexico": "MX",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "ireland": "IE",
    "france": "FR",
    "germany": "DE",
    "italy": "IT",
    "spain": "ES",
    "portugal": "PT",
    "netherlands": "NL",
    "holland": "NL",
    "belgium": "BE",
    "luxembourg": "LU",
    "switzerland": "CH",
    "austria": "AT",
    "denmark": "DK",
    "sweden": "SE",
    "norway": "NO",
    "finland": "FI",
    "iceland": "IS",
    "poland": "PL",
    "czech republic": "CZ",
    "czechia": "CZ",
    "greece": "GR",
    "croatia": "HR",
    "slovenia": "SI",
    "hungary": "HU",
    "turkey": "TR",
    "russia": "RU",
    "japan": "JP",
    "china": "CN",
    "south korea": "KR",
    "korea": "KR",
    "taiwan": "TW",
    "thailand": "TH",
    "vietnam": "VN",
    "india": "IN",
    "australia": "AU",
    "new zealand": "NZ",
    "brazil": "BR",
    "argentina": "AR",
    "chile": "CL",
    "peru": "PE",
    "colombia": "CO",
    "costa rica": "CR",
    "cuba": "CU",
    "south africa": "ZA",
    "egypt": "EG",
    "morocco": "MA",
    "israel": "IL",
    "uae": "AE",
    "united arab emirates": "AE",
}


def resolve_state(text: str) -> Optional[str]:
    """Return the lowercase full US state name for *text*, or None.

    Accepts 2-letter abbreviations (``"CA"``) and full names
    (``"California"``, case-insensitive).
    """
    if not text:
        return None
    return _US_STATES.get(text.strip().lower())


def resolve_country(text: str) -> Optional[str]:
    """Return the ISO2 country code for *text*, or None.

    Accepts common country names and a few popular aliases.  Also passes
    through an already-valid 2-letter ISO code (e.g. ``"IT"``).
    """
    if not text:
        return None
    key = text.strip().lower()
    if key in _COUNTRIES:
        return _COUNTRIES[key]
    # Pass through 2-letter ISO codes that appear in our value set.
    if len(key) == 2 and key.upper() in _COUNTRIES.values():
        return key.upper()
    return None


def split_location_name(location_name: Optional[str]) -> tuple[
    Optional[str], Optional[str], Optional[str]
]:
    """Parse a ``"City, Region, CC"`` string into normalised payload fields.

    Returns ``(city_lower, region_lower, country_code_upper)`` with any
    missing component as ``None``.  Extra middle commas (rare) are joined
    back into ``region``.
    """
    if not location_name:
        return None, None, None

    parts = [p.strip() for p in location_name.split(",") if p.strip()]
    if not parts:
        return None, None, None

    city: Optional[str] = None
    region: Optional[str] = None
    country_code: Optional[str] = None

    if len(parts) == 1:
        city = parts[0]
    elif len(parts) == 2:
        city, tail = parts
        # Heuristic: if tail looks like an ISO2 code, treat as country.
        if len(tail) == 2 and tail.isupper():
            country_code = tail
        else:
            region = tail
    else:
        city = parts[0]
        country_code = parts[-1]
        region = ", ".join(parts[1:-1])

    return (
        city.lower() if city else None,
        region.lower() if region else None,
        country_code.upper() if country_code else None,
    )
