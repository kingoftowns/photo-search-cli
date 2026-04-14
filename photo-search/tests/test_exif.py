"""Tests for EXIF metadata extraction."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from photo_search.exif import _gps_to_decimal, _parse_exif_datetime, extract_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_jpeg(width: int = 100, height: int = 100) -> str:
    """Create a small temporary JPEG and return its path."""
    img = Image.new("RGB", (width, height), color=(200, 150, 100))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp, format="JPEG")
    tmp.close()
    return tmp.name


class _FakeExif(dict):
    """Dict subclass with get_ifd() for mocking image.getexif()."""

    def __init__(
        self,
        root_data: dict,
        exif_ifd: dict | None = None,
        gps_ifd: dict | None = None,
    ) -> None:
        super().__init__(root_data)
        self._ifds = {0x8769: exif_ifd or {}, 0x8825: gps_ifd or {}}

    def get_ifd(self, tag: int) -> dict:
        return self._ifds.get(tag, {})


# ---------------------------------------------------------------------------
# _gps_to_decimal
# ---------------------------------------------------------------------------

class TestGpsToDecimal:
    def test_north_east(self) -> None:
        """Irvine, CA is roughly 33.6846 N, 117.8265 W."""
        # 33 degrees, 41 minutes, 4.56 seconds North
        gps_coords = (33.0, 41.0, 4.56)
        result = _gps_to_decimal(gps_coords, "N")
        assert abs(result - 33.6846) < 0.001

    def test_south_hemisphere(self) -> None:
        """Southern hemisphere should produce negative latitude."""
        gps_coords = (33.0, 51.0, 54.0)
        result = _gps_to_decimal(gps_coords, "S")
        assert result < 0
        assert abs(result - (-33.865)) < 0.001

    def test_west_hemisphere(self) -> None:
        """Western hemisphere should produce negative longitude."""
        gps_coords = (117.0, 49.0, 35.4)
        result = _gps_to_decimal(gps_coords, "W")
        assert result < 0
        assert abs(result - (-117.8265)) < 0.001

    def test_east_hemisphere(self) -> None:
        """Eastern hemisphere should produce positive longitude."""
        gps_coords = (2.0, 17.0, 40.0)
        result = _gps_to_decimal(gps_coords, "E")
        assert result > 0
        expected = 2.0 + 17.0 / 60.0 + 40.0 / 3600.0
        assert abs(result - expected) < 0.0001

    def test_ifd_rational_values(self) -> None:
        """EXIF GPS values are often IFDRational objects that support float()."""
        # Simulate IFDRational with a mock that responds to float()
        mock_deg = MagicMock()
        mock_deg.__float__ = MagicMock(return_value=40.0)
        mock_min = MagicMock()
        mock_min.__float__ = MagicMock(return_value=26.0)
        mock_sec = MagicMock()
        mock_sec.__float__ = MagicMock(return_value=46.0)

        result = _gps_to_decimal((mock_deg, mock_min, mock_sec), "N")
        expected = 40.0 + 26.0 / 60.0 + 46.0 / 3600.0
        assert abs(result - expected) < 0.0001

    def test_zero_coordinates(self) -> None:
        """Zero coordinates (null island) should return 0.0."""
        result = _gps_to_decimal((0.0, 0.0, 0.0), "N")
        assert result == 0.0


# ---------------------------------------------------------------------------
# _parse_exif_datetime
# ---------------------------------------------------------------------------

class TestParseExifDatetime:
    def test_standard_exif_format(self) -> None:
        result = _parse_exif_datetime("2024:06:15 14:30:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30
        assert result.second == 0

    def test_iso_format(self) -> None:
        result = _parse_exif_datetime("2024-06-15 14:30:00")
        assert result is not None
        assert result.year == 2024

    def test_iso_8601_t_separator(self) -> None:
        result = _parse_exif_datetime("2024-06-15T14:30:00")
        assert result is not None
        assert result.year == 2024
        assert result.hour == 14

    def test_with_timezone_offset(self) -> None:
        result = _parse_exif_datetime("2024:06:15 14:30:00+02:00")
        assert result is not None
        assert result.year == 2024

    def test_with_subseconds(self) -> None:
        result = _parse_exif_datetime("2024:06:15 14:30:00.123456")
        assert result is not None
        assert result.microsecond == 123456

    def test_invalid_returns_none(self) -> None:
        assert _parse_exif_datetime("not a date") is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_exif_datetime("") is None

    def test_none_returns_none(self) -> None:
        # Type hint says str, but in practice EXIF can return None
        assert _parse_exif_datetime(None) is None  # type: ignore[arg-type]

    def test_null_bytes_stripped(self) -> None:
        result = _parse_exif_datetime("2024:06:15 14:30:00\x00")
        assert result is not None
        assert result.year == 2024

    def test_whitespace_stripped(self) -> None:
        result = _parse_exif_datetime("  2024:06:15 14:30:00  ")
        assert result is not None
        assert result.year == 2024


# ---------------------------------------------------------------------------
# extract_metadata — real JPEG (no EXIF)
# ---------------------------------------------------------------------------

class TestExtractMetadataBasic:
    def test_basic_jpeg_metadata(self) -> None:
        """A plain JPEG with no EXIF should still return file-level info."""
        path = _create_test_jpeg(120, 80)
        try:
            meta = extract_metadata(path)
            assert meta.file_name.endswith(".jpg")
            assert meta.file_type == "JPG"
            assert meta.file_size_bytes > 0
            assert meta.width == 120
            assert meta.height == 80
            assert os.path.isabs(meta.file_path)
        finally:
            os.unlink(path)

    def test_missing_exif_fields_are_none(self) -> None:
        """When there is no EXIF data, optional fields should be None."""
        path = _create_test_jpeg()
        try:
            meta = extract_metadata(path)
            assert meta.date_taken is None
            assert meta.gps_lat is None
            assert meta.gps_lon is None
            assert meta.camera is None
            assert meta.focal_length is None
            assert meta.aperture is None
            assert meta.iso is None
            assert meta.orientation is None
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self) -> None:
        """extract_metadata should raise for non-existent files."""
        with pytest.raises((FileNotFoundError, OSError)):
            extract_metadata("/tmp/definitely_does_not_exist_abc123xyz.jpg")


# ---------------------------------------------------------------------------
# extract_metadata — mocked EXIF
# ---------------------------------------------------------------------------

class TestExtractMetadataWithExif:
    def test_full_exif_extraction(self) -> None:
        """Mock a full set of EXIF tags and verify extraction."""
        path = _create_test_jpeg(200, 150)
        try:
            # Root IFD tags
            root_ifd = {
                271: "Apple",           # Make
                272: "iPhone 15 Pro",   # Model
                274: 1,                 # Orientation
            }
            # ExifIFD sub-tags (accessed via get_ifd(0x8769))
            exif_ifd = {
                36867: "2024:06:15 14:30:00",  # DateTimeOriginal
                37386: 6.765,                   # FocalLength
                33437: 1.78,                    # FNumber
                34855: 100,                     # ISOSpeedRatings
            }
            # GPS IFD sub-tags (accessed via get_ifd(0x8825))
            gps_ifd = {
                1: "N",                         # GPSLatitudeRef
                2: (33.0, 41.0, 4.56),          # GPSLatitude
                3: "W",                         # GPSLongitudeRef
                4: (117.0, 49.0, 35.4),         # GPSLongitude
            }
            fake_exif = _FakeExif(root_ifd, exif_ifd=exif_ifd, gps_ifd=gps_ifd)

            with patch("photo_search.exif.Image") as mock_image_mod:
                mock_img = MagicMock()
                mock_img.size = (200, 150)
                mock_img.getexif.return_value = fake_exif
                mock_image_mod.open.return_value = mock_img

                meta = extract_metadata(path)

            assert meta.date_taken is not None
            assert meta.date_taken.year == 2024
            assert meta.date_taken.month == 6
            assert meta.camera == "Apple iPhone 15 Pro"
            assert meta.focal_length is not None
            assert abs(meta.focal_length - 6.765) < 0.01
            assert meta.aperture is not None
            assert abs(meta.aperture - 1.78) < 0.01
            assert meta.iso == 100
            assert meta.orientation == 1
            assert meta.gps_lat is not None
            assert meta.gps_lat > 33.0
            assert meta.gps_lon is not None
            assert meta.gps_lon < 0  # West

        finally:
            os.unlink(path)

    def test_redundant_make_in_model(self) -> None:
        """If Model already starts with Make, don't duplicate it."""
        path = _create_test_jpeg()
        try:
            fake_exif = _FakeExif({
                271: "Apple",
                272: "Apple iPhone 15 Pro",
            })

            with patch("photo_search.exif.Image") as mock_image_mod:
                mock_img = MagicMock()
                mock_img.size = (100, 100)
                mock_img.getexif.return_value = fake_exif
                mock_image_mod.open.return_value = mock_img

                meta = extract_metadata(path)

            assert meta.camera == "Apple iPhone 15 Pro"
            # Must NOT be "Apple Apple iPhone 15 Pro"
            assert not meta.camera.startswith("Apple Apple")

        finally:
            os.unlink(path)

    def test_partial_exif_no_crash(self) -> None:
        """An image with some malformed EXIF fields should not crash."""
        path = _create_test_jpeg()
        try:
            # These tags live in ExifIFD
            fake_exif = _FakeExif({}, exif_ifd={
                36867: "not a valid date!!!",
                37386: "not_a_number",
                34855: None,
            })

            with patch("photo_search.exif.Image") as mock_image_mod:
                mock_img = MagicMock()
                mock_img.size = (100, 100)
                mock_img.getexif.return_value = fake_exif
                mock_image_mod.open.return_value = mock_img

                # This should NOT raise
                meta = extract_metadata(path)

            assert meta.file_name.endswith(".jpg")
            assert meta.date_taken is None  # invalid date -> None
        finally:
            os.unlink(path)

    def test_getexif_fallback_to_legacy(self) -> None:
        """If getexif() raises, fall back to _getexif()."""
        path = _create_test_jpeg()
        try:
            with patch("photo_search.exif.Image") as mock_image_mod:
                mock_img = MagicMock()
                mock_img.size = (100, 100)
                mock_img.getexif.side_effect = Exception("getexif broken")
                mock_img._getexif.return_value = {274: 6, 271: "Canon"}
                mock_image_mod.open.return_value = mock_img

                meta = extract_metadata(path)

            assert meta.orientation == 6
            assert meta.camera == "Canon"
        finally:
            os.unlink(path)
