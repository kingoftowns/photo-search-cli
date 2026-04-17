"""Tests for the indexing pipeline.

All external dependencies (storage backends, face detector, captioner,
embedder, geocoder) are mocked so that tests run fast and without
infrastructure.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from photo_search.config import AppConfig
from photo_search.models import (
    DetectedFace,
    IdentifiedFace,
    IndexedPhoto,
    IndexingStatus,
    PhotoCaption,
    PhotoMetadata,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_config(**overrides) -> AppConfig:
    """Build a minimal AppConfig suitable for testing."""
    defaults = {
        "photos": {
            "source_dir": "/fake/photos",
            "supported_extensions": [".jpg", ".jpeg", ".heic", ".png"],
            "skip_extensions": [".mov", ".mp4", ".aae"],
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "vision_model": "test-vision",
            "embedding_model": "test-embed",
            "request_timeout": 30,
        },
        "faces": {
            "model_pack": "buffalo_l",
            "similarity_threshold": 0.4,
            "min_face_size": 20,
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "collection_name": "test_photos",
            "vector_size": 768,
        },
        "postgres": {
            "connection_string": "postgresql://test:test@localhost/test",
        },
        "geocoding": {"enabled": True},
        "pipeline": {
            "batch_log_interval": 5,
            "max_retries": 1,
            "retry_delay": 0,
            "resize_max_dimension": 512,
        },
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def _make_metadata(file_path: str = "/fake/photos/img.jpg") -> PhotoMetadata:
    return PhotoMetadata(
        file_path=file_path,
        file_name=os.path.basename(file_path),
        file_size_bytes=1024,
        file_type="JPG",
        date_taken=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        gps_lat=37.7749,
        gps_lon=-122.4194,
        camera="Test Camera",
    )


def _make_detected_face() -> DetectedFace:
    return DetectedFace(
        bbox=(10.0, 20.0, 50.0, 50.0),
        confidence=0.99,
        embedding=[0.1] * 512,
    )


def _make_identified_face(label: str = "alice") -> IdentifiedFace:
    return IdentifiedFace(
        bbox=(10.0, 20.0, 50.0, 50.0),
        confidence=0.99,
        embedding=[0.1] * 512,
        label=label,
        similarity=0.85,
    )


def _make_caption() -> PhotoCaption:
    return PhotoCaption(
        caption="A sunset over the ocean",
        model="test-vision",
        generation_time_seconds=1.5,
    )


# ======================================================================
# Pipeline construction helper (mocks all I/O)
# ======================================================================

def _build_pipeline(config: AppConfig | None = None):
    """Construct an IndexingPipeline with all I/O components mocked."""
    if config is None:
        config = _make_config()

    with (
        patch("photo_search.pipeline.PostgresStorage") as MockPG,
        patch("photo_search.pipeline.QdrantStorage") as MockQdrant,
        patch("photo_search.pipeline.FaceDetector") as MockDetector,
        patch("photo_search.pipeline.FaceClassifier") as MockClassifier,
        patch("photo_search.pipeline.create_captioner") as MockCreateCaptioner,
        patch("photo_search.pipeline.TextEmbedder") as MockEmbedder,
    ):
        # Configure mock return values.
        mock_pg = MockPG.return_value
        mock_pg.get_face_identities.return_value = []
        mock_pg.get_indexing_status.return_value = None
        mock_pg.get_photo.return_value = None
        mock_pg.get_incomplete_files.return_value = []
        mock_pg.get_all_statuses.return_value = {"total": 0}
        mock_pg.get_files_with_errors.return_value = []

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.count.return_value = 0

        mock_detector = MockDetector.return_value
        mock_detector.detect_faces.return_value = []

        mock_classifier = MockClassifier.return_value
        mock_classifier.classify_faces.return_value = []

        mock_captioner = MagicMock()
        mock_captioner.caption_photo.return_value = _make_caption()
        MockCreateCaptioner.return_value = mock_captioner

        mock_embedder = MockEmbedder.return_value
        mock_embedder.embed_photo.return_value = (
            "search text",
            [0.5] * 768,
        )
        mock_embedder.embed_text.return_value = [0.5] * 768

        from photo_search.pipeline import IndexingPipeline

        pipeline = IndexingPipeline(config)

    return pipeline


# ======================================================================
# Tests: scan_photos
# ======================================================================

class TestScanPhotos:
    """Tests for file discovery."""

    def test_scan_photos(self) -> None:
        """os.walk results are filtered by supported extensions."""
        pipeline = _build_pipeline()

        walk_results = [
            ("/fake/photos", ["sub"], ["a.jpg", "b.mov", "c.png", "readme.txt"]),
            ("/fake/photos/sub", [], ["d.heic", "e.mp4"]),
        ]

        with patch("os.walk", return_value=walk_results):
            found = pipeline.scan_photos()

        assert found == [
            "/fake/photos/a.jpg",
            "/fake/photos/c.png",
            "/fake/photos/sub/d.heic",
        ]

    def test_scan_photos_case_insensitive(self) -> None:
        """Extensions are matched case-insensitively."""
        pipeline = _build_pipeline()

        walk_results = [
            ("/fake/photos", [], ["a.HEIC", "b.Jpg", "c.PNG", "d.heic"]),
        ]

        with patch("os.walk", return_value=walk_results):
            found = pipeline.scan_photos()

        assert len(found) == 4
        # All four files should be found regardless of case.
        basenames = [os.path.basename(f) for f in found]
        assert set(basenames) == {"a.HEIC", "b.Jpg", "c.PNG", "d.heic"}

    def test_scan_photos_empty_dir(self) -> None:
        """An empty directory returns an empty list."""
        pipeline = _build_pipeline()

        with patch("os.walk", return_value=[("/fake/photos", [], [])]):
            found = pipeline.scan_photos()

        assert found == []

    def test_scan_photos_skip_extensions(self) -> None:
        """Files with skip extensions are excluded even if otherwise supported."""
        config = _make_config(
            photos={
                "source_dir": "/fake",
                "supported_extensions": [".jpg"],
                "skip_extensions": [".jpg"],  # conflicting — skip wins
            }
        )
        pipeline = _build_pipeline(config)

        walk_results = [("/fake", [], ["photo.jpg"])]
        with patch("os.walk", return_value=walk_results):
            found = pipeline.scan_photos()

        # skip_extensions is checked first, so the file should be excluded.
        assert found == []


# ======================================================================
# Tests: get_pending_files
# ======================================================================

class TestGetPendingFiles:
    """Tests for resume / pending detection logic."""

    def test_get_pending_files_new(self) -> None:
        """Files not in indexing_status are considered pending."""
        pipeline = _build_pipeline()
        pipeline.pg.get_indexing_status.return_value = None

        pending = pipeline.get_pending_files(
            ["/fake/photos/a.jpg", "/fake/photos/b.jpg"]
        )

        assert len(pending) == 2

    def test_get_pending_files_partial(self) -> None:
        """Files with partial completion are pending."""
        pipeline = _build_pipeline()

        partial_status = IndexingStatus(
            file_path="/fake/photos/a.jpg",
            exif_extracted=True,
            faces_extracted=True,
            faces_classified=True,
            captioned=False,
            embedded=False,
        )

        def _side_effect(fp: str) -> Optional[IndexingStatus]:
            if fp == "/fake/photos/a.jpg":
                return partial_status
            return None

        pipeline.pg.get_indexing_status.side_effect = _side_effect

        # Without stage filter — uses 'embedded' as the check.
        pending = pipeline.get_pending_files(
            ["/fake/photos/a.jpg", "/fake/photos/b.jpg"]
        )
        assert "/fake/photos/a.jpg" in pending
        assert "/fake/photos/b.jpg" in pending

    def test_get_pending_files_complete(self) -> None:
        """Fully indexed files are not pending."""
        pipeline = _build_pipeline()

        complete = IndexingStatus(
            file_path="/fake/photos/done.jpg",
            exif_extracted=True,
            faces_extracted=True,
            faces_classified=True,
            captioned=True,
            embedded=True,
        )
        pipeline.pg.get_indexing_status.return_value = complete

        pending = pipeline.get_pending_files(["/fake/photos/done.jpg"])
        assert pending == []

    def test_get_pending_files_stage_filter(self) -> None:
        """Stage filter narrows the check to a specific boolean."""
        pipeline = _build_pipeline()

        status = IndexingStatus(
            file_path="/fake/photos/a.jpg",
            exif_extracted=True,
            faces_extracted=False,
            captioned=True,
            embedded=False,
        )
        pipeline.pg.get_indexing_status.return_value = status

        # Checking 'faces' stage — faces_extracted is False.
        pending = pipeline.get_pending_files(
            ["/fake/photos/a.jpg"], stage="faces"
        )
        assert len(pending) == 1

        # Checking 'caption' stage — captioned is True.
        pending = pipeline.get_pending_files(
            ["/fake/photos/a.jpg"], stage="caption"
        )
        assert pending == []


# ======================================================================
# Tests: process_photo
# ======================================================================

class TestProcessPhoto:
    """Tests for single-file processing."""

    @patch("photo_search.pipeline.extract_metadata")
    @patch("photo_search.pipeline.reverse_geocode", return_value="San Francisco, CA")
    def test_process_photo_full_success(
        self, mock_geocode, mock_exif
    ) -> None:
        """A photo processed through all stages ends fully indexed."""
        pipeline = _build_pipeline()
        mock_exif.return_value = _make_metadata()
        pipeline.face_detector.detect_faces.return_value = [_make_detected_face()]
        pipeline.face_classifier.classify_faces.return_value = [
            _make_identified_face()
        ]
        pipeline.captioner.caption_photo.return_value = _make_caption()
        pipeline.embedder.embed_photo.return_value = ("search text", [0.5] * 768)

        status = pipeline.process_photo("/fake/photos/img.jpg")

        assert status.exif_extracted is True
        assert status.faces_extracted is True
        assert status.faces_classified is True
        assert status.captioned is True
        assert status.embedded is True
        assert status.error is None

        # Verify storage calls.
        pipeline.pg.upsert_photo.assert_called_once()
        pipeline.pg.upsert_indexing_status.assert_called_once()
        pipeline.qdrant.upsert_photo.assert_called_once()

    @patch("photo_search.pipeline.extract_metadata")
    def test_process_photo_error_resilience(self, mock_exif) -> None:
        """An error in one stage is recorded but does not raise."""
        pipeline = _build_pipeline()
        mock_exif.side_effect = RuntimeError("EXIF parsing failed")

        # Should NOT raise.
        status = pipeline.process_photo("/fake/photos/broken.jpg")

        assert status.error is not None
        assert "exif" in status.error.lower()
        # Subsequent stages should not have run.
        assert status.faces_extracted is False
        assert status.captioned is False
        assert status.embedded is False

    @patch("photo_search.pipeline.extract_metadata")
    def test_process_photo_caption_error(self, mock_exif) -> None:
        """An error in the caption stage records the error and stops."""
        pipeline = _build_pipeline()
        mock_exif.return_value = _make_metadata()
        pipeline.face_detector.detect_faces.return_value = []
        pipeline.face_classifier.classify_faces.return_value = []
        pipeline.captioner.caption_photo.side_effect = TimeoutError("VLM timeout")

        status = pipeline.process_photo("/fake/photos/slow.jpg")

        assert status.exif_extracted is True
        assert status.faces_extracted is True
        assert status.captioned is False
        assert "caption" in status.error.lower()

    @patch("photo_search.pipeline.extract_metadata")
    def test_process_photo_stages_filtering(self, mock_exif) -> None:
        """When stages are restricted, only those stages run."""
        pipeline = _build_pipeline()
        mock_exif.return_value = _make_metadata()
        pipeline.face_detector.detect_faces.return_value = [_make_detected_face()]
        pipeline.face_classifier.classify_faces.return_value = [
            _make_identified_face()
        ]

        # Only run face stages.
        status = pipeline.process_photo(
            "/fake/photos/img.jpg", stages={"exif", "faces"}
        )

        assert status.exif_extracted is True
        assert status.faces_extracted is True
        assert status.faces_classified is True
        # Caption and embed should not have run.
        assert status.captioned is False
        assert status.embedded is False

    @patch("photo_search.pipeline.extract_metadata")
    def test_process_photo_skips_completed_stages(self, mock_exif) -> None:
        """Stages already marked complete are skipped on resume."""
        pipeline = _build_pipeline()

        # Simulate a file that already has exif and faces done.
        existing_status = IndexingStatus(
            file_path="/fake/photos/partial.jpg",
            exif_extracted=True,
            faces_extracted=True,
            faces_classified=True,
            captioned=False,
            embedded=False,
        )
        pipeline.pg.get_indexing_status.return_value = existing_status

        mock_exif.return_value = _make_metadata("/fake/photos/partial.jpg")
        pipeline.captioner.caption_photo.return_value = _make_caption()
        pipeline.embedder.embed_photo.return_value = ("text", [0.5] * 768)

        status = pipeline.process_photo("/fake/photos/partial.jpg")

        # EXIF and faces were already done so those methods should NOT
        # have been called (the pipeline should have skipped them).
        mock_exif.assert_called()  # called for embed stage to get metadata
        pipeline.face_detector.detect_faces.assert_not_called()
        pipeline.captioner.caption_photo.assert_called_once()

        assert status.captioned is True
        assert status.embedded is True


# ======================================================================
# Tests: run (dry_run)
# ======================================================================

class TestPipelineRun:
    """Tests for the main run loop."""

    def test_dry_run_returns_stats(self) -> None:
        """Dry run reports counts without processing any files."""
        pipeline = _build_pipeline()

        walk_results = [
            ("/fake/photos", [], ["a.jpg", "b.jpg", "c.jpg"]),
        ]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run(dry_run=True)

        assert stats["processed"] == 0
        assert stats["skipped"] == 3  # all 3 files are pending
        assert stats["total_scanned"] == 3
        # process_photo should never have been called.
        pipeline.pg.upsert_photo.assert_not_called()

    @patch("photo_search.pipeline.extract_metadata")
    @patch("photo_search.pipeline.reverse_geocode", return_value=None)
    def test_run_processes_pending(self, mock_geo, mock_exif) -> None:
        """Run processes all pending files and returns correct stats."""
        pipeline = _build_pipeline()
        mock_exif.return_value = _make_metadata()
        pipeline.captioner.caption_photo.return_value = _make_caption()
        pipeline.embedder.embed_photo.return_value = ("text", [0.5] * 768)

        walk_results = [("/fake/photos", [], ["a.jpg", "b.jpg"])]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run()

        assert stats["processed"] == 2
        assert stats["succeeded"] == 2
        assert stats["failed"] == 0

    @patch("photo_search.pipeline.extract_metadata")
    @patch("photo_search.pipeline.reverse_geocode", return_value=None)
    def test_run_with_file_filter(self, mock_geo, mock_exif) -> None:
        """File filter narrows processing to matching paths."""
        pipeline = _build_pipeline()
        mock_exif.return_value = _make_metadata()
        pipeline.captioner.caption_photo.return_value = _make_caption()
        pipeline.embedder.embed_photo.return_value = ("text", [0.5] * 768)

        walk_results = [
            ("/fake/photos", [], ["vacation.jpg", "work.jpg", "vacation2.jpg"]),
        ]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run(file_filter="vacation")

        assert stats["processed"] == 2  # only vacation*.jpg

    @patch("photo_search.pipeline.extract_metadata")
    @patch("photo_search.pipeline.reverse_geocode", return_value=None)
    def test_run_errors_only(self, mock_geo, mock_exif) -> None:
        """errors_only mode re-processes files that previously had errors."""
        pipeline = _build_pipeline()
        mock_exif.return_value = _make_metadata()
        pipeline.captioner.caption_photo.return_value = _make_caption()
        pipeline.embedder.embed_photo.return_value = ("text", [0.5] * 768)

        error_status = IndexingStatus(
            file_path="/fake/photos/err.jpg",
            exif_extracted=True,
            error="caption: timeout",
        )
        pipeline.pg.get_files_with_errors.return_value = [error_status]
        # After clearing, get_indexing_status returns None (fresh start).
        pipeline.pg.get_indexing_status.return_value = None

        walk_results = [("/fake/photos", [], ["err.jpg", "ok.jpg"])]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run(errors_only=True)

        assert stats["processed"] == 1
        pipeline.pg.clear_indexing_status.assert_called_once_with(
            "/fake/photos/err.jpg"
        )


# ======================================================================
# Tests: pipeline stages integration
# ======================================================================

class TestStagesFiltering:
    """Verify that stage subsets limit which processing runs."""

    @patch("photo_search.pipeline.extract_metadata")
    def test_faces_only_stages(self, mock_exif) -> None:
        """--faces-only maps to {exif, faces} stages."""
        pipeline = _build_pipeline()
        mock_exif.return_value = _make_metadata()
        pipeline.face_detector.detect_faces.return_value = [_make_detected_face()]
        pipeline.face_classifier.classify_faces.return_value = [
            _make_identified_face()
        ]

        walk_results = [("/fake/photos", [], ["a.jpg"])]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run(stages={"exif", "faces"})

        assert stats["processed"] == 1
        pipeline.captioner.caption_photo.assert_not_called()

    @patch("photo_search.pipeline.extract_metadata")
    def test_caption_only_stage(self, mock_exif) -> None:
        """--captions-only maps to {caption} stage."""
        pipeline = _build_pipeline()

        # File already has exif+faces done.
        existing = IndexingStatus(
            file_path="/fake/photos/a.jpg",
            exif_extracted=True,
            faces_extracted=True,
            faces_classified=True,
        )
        pipeline.pg.get_indexing_status.return_value = existing
        mock_exif.return_value = _make_metadata()
        pipeline.captioner.caption_photo.return_value = _make_caption()

        walk_results = [("/fake/photos", [], ["a.jpg"])]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run(stages={"caption"})

        assert stats["processed"] == 1
        pipeline.captioner.caption_photo.assert_called_once()
        pipeline.face_detector.detect_faces.assert_not_called()

    @patch("photo_search.pipeline.extract_metadata")
    @patch("photo_search.pipeline.reverse_geocode", return_value=None)
    def test_embed_only_stage(self, mock_geo, mock_exif) -> None:
        """--embed-only maps to {embed} stage."""
        pipeline = _build_pipeline()

        existing = IndexingStatus(
            file_path="/fake/photos/a.jpg",
            exif_extracted=True,
            faces_extracted=True,
            faces_classified=True,
            captioned=True,
        )
        pipeline.pg.get_indexing_status.return_value = existing
        pipeline.pg.get_photo.return_value = {"caption": "A sunset"}
        mock_exif.return_value = _make_metadata()
        pipeline.embedder.embed_photo.return_value = ("text", [0.5] * 768)

        walk_results = [("/fake/photos", [], ["a.jpg"])]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run(stages={"embed"})

        assert stats["processed"] == 1
        pipeline.embedder.embed_photo.assert_called_once()
        pipeline.captioner.caption_photo.assert_not_called()
        pipeline.face_detector.detect_faces.assert_not_called()


# ======================================================================
# Tests: parallel processing
# ======================================================================

class TestParallelProcessing:
    """Tests for the ThreadPoolExecutor-based parallel path."""

    @patch("photo_search.pipeline.extract_metadata")
    @patch("photo_search.pipeline.reverse_geocode", return_value=None)
    def test_concurrency_greater_than_one_processes_all_files(
        self, mock_geo, mock_exif
    ) -> None:
        """concurrency=4 should still process every pending file exactly once."""
        config = _make_config()
        config.pipeline.concurrency = 4
        pipeline = _build_pipeline(config)

        mock_exif.return_value = _make_metadata()
        pipeline.embedder.embed_photo.return_value = ("text", [0.5] * 768)

        # 10 fake files
        filenames = [f"img{i:02d}.jpg" for i in range(10)]
        walk_results = [("/fake/photos", [], filenames)]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run()

        assert stats["processed"] == 10
        assert stats["succeeded"] == 10
        assert stats["failed"] == 0
        # Every file should have been captioned exactly once
        assert pipeline.captioner.caption_photo.call_count == 10
        # upsert_indexing_status is called at least once per file
        assert pipeline.pg.upsert_indexing_status.call_count >= 10

    @patch("photo_search.pipeline.extract_metadata")
    @patch("photo_search.pipeline.reverse_geocode", return_value=None)
    def test_parallel_errors_are_isolated_per_file(
        self, mock_geo, mock_exif
    ) -> None:
        """One file's exception must not poison the others."""
        config = _make_config()
        config.pipeline.concurrency = 3
        pipeline = _build_pipeline(config)

        mock_exif.return_value = _make_metadata()
        pipeline.embedder.embed_photo.return_value = ("text", [0.5] * 768)

        call_paths: list[str] = []

        def caption_side_effect(path: str):
            call_paths.append(path)
            if path.endswith("bad.jpg"):
                raise RuntimeError("boom")
            return _make_caption()

        pipeline.captioner.caption_photo.side_effect = caption_side_effect

        walk_results = [("/fake/photos", [], ["a.jpg", "bad.jpg", "c.jpg"])]

        with patch("os.walk", return_value=walk_results):
            stats = pipeline.run()

        assert stats["processed"] == 3
        assert stats["succeeded"] == 2
        assert stats["failed"] == 1

    def test_concurrency_value_normalized(self) -> None:
        """concurrency <= 0 is clamped to 1 (sequential)."""
        config = _make_config()
        config.pipeline.concurrency = 0
        pipeline = _build_pipeline(config)

        with patch("os.walk", return_value=[]):
            stats = pipeline.run()

        # No files to process, just verify the run completed cleanly.
        assert stats["processed"] == 0


# ======================================================================
# Tests: PostgresStorage thread safety
# ======================================================================

class TestPostgresThreadSafety:
    """Verify PostgresStorage maintains one connection per thread."""

    def test_each_thread_gets_its_own_connection(self) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from photo_search.storage import PostgresStorage

        with patch("photo_search.storage.psycopg2") as mock_psycopg2:
            # Each call to psycopg2.connect returns a fresh MagicMock, so
            # we can count distinct connections by identity.
            mock_psycopg2.connect.side_effect = lambda *a, **kw: MagicMock(
                closed=False
            )

            pg = PostgresStorage("postgresql://test@localhost/test")
            connections: list[int] = []
            conns_lock = threading.Lock()
            # A barrier ensures all four workers hold distinct threads
            # simultaneously; otherwise the pool could reuse an idle
            # thread, its thread-local carrying over.
            barrier = threading.Barrier(4)

            def worker() -> None:
                barrier.wait()
                conn = pg._get_connection()
                with conns_lock:
                    connections.append(id(conn))

            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(worker) for _ in range(4)]
                for f in futures:
                    f.result()

            # 4 concurrent threads -> 4 distinct connections
            assert len(set(connections)) == 4
            assert mock_psycopg2.connect.call_count == 4

    def test_close_releases_all_connections(self) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from photo_search.storage import PostgresStorage

        opened: list[MagicMock] = []
        opened_lock = threading.Lock()

        def fake_connect(*args, **kwargs):
            m = MagicMock()
            m.closed = False
            with opened_lock:
                opened.append(m)
            return m

        with patch("photo_search.storage.psycopg2") as mock_psycopg2:
            mock_psycopg2.connect.side_effect = fake_connect

            pg = PostgresStorage("postgresql://test@localhost/test")

            barrier = threading.Barrier(3)

            def worker() -> None:
                barrier.wait()
                pg._get_connection()

            with ThreadPoolExecutor(max_workers=3) as ex:
                futures = [ex.submit(worker) for _ in range(3)]
                for f in futures:
                    f.result()

            pg.close()

            assert len(opened) == 3
            for conn in opened:
                conn.close.assert_called_once()
