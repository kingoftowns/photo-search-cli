"""Unit tests for :mod:`photo_search.faces`.

All tests are designed to run without InsightFace model files, a GPU, or a
database connection.  Heavy dependencies (InsightFace, cv2) are mocked where
necessary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from photo_search.faces import (
    FaceClassifier,
    FaceDetector,
    compute_centroid,
    crop_face,
)
from photo_search.models import DetectedFace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(seed: int = 0, dim: int = 512) -> list[float]:
    """Generate a deterministic pseudo-random embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_detected_face(
    seed: int = 0,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 80.0, 100.0),
    confidence: float = 0.99,
) -> DetectedFace:
    return DetectedFace(
        bbox=bbox,
        confidence=confidence,
        embedding=_make_embedding(seed),
    )


def _create_test_image(width: int = 200, height: int = 200) -> str:
    """Create a small temporary PNG and return its path."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp, format="PNG")
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# compute_centroid
# ---------------------------------------------------------------------------

class TestComputeCentroid:
    def test_single_embedding(self) -> None:
        emb = _make_embedding(42)
        centroid = compute_centroid([emb])
        norm = float(np.linalg.norm(centroid))
        assert abs(norm - 1.0) < 1e-5, f"Centroid should be unit vector, got norm={norm}"

    def test_multiple_embeddings(self) -> None:
        embeddings = [_make_embedding(i) for i in range(5)]
        centroid = compute_centroid(embeddings)
        norm = float(np.linalg.norm(centroid))
        assert abs(norm - 1.0) < 1e-5

    def test_identical_embeddings_return_same_direction(self) -> None:
        emb = _make_embedding(7)
        centroid = compute_centroid([emb, emb, emb])
        similarity = float(np.dot(centroid, np.array(emb, dtype=np.float32)))
        assert similarity > 0.99, "Centroid of identical vectors should point same direction"

    def test_result_is_ndarray(self) -> None:
        centroid = compute_centroid([_make_embedding(0)])
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (512,)

    def test_zero_vector_input(self) -> None:
        zeros = [0.0] * 512
        centroid = compute_centroid([zeros])
        # Should not raise — returns the zero vector as-is.
        assert centroid.shape == (512,)


# ---------------------------------------------------------------------------
# FaceClassifier
# ---------------------------------------------------------------------------

class TestFaceClassifierNoIdentities:
    def test_returns_unknown(self) -> None:
        clf = FaceClassifier(similarity_threshold=0.4)
        face = _make_detected_face(seed=1)
        result = clf.classify_face(face)
        assert result.label == "unknown"
        assert result.similarity == 0.0

    def test_preserves_bbox_and_confidence(self) -> None:
        clf = FaceClassifier()
        face = _make_detected_face(seed=2, bbox=(5.0, 10.0, 50.0, 60.0), confidence=0.95)
        result = clf.classify_face(face)
        assert result.bbox == face.bbox
        assert result.confidence == face.confidence

    def test_classify_faces_batch(self) -> None:
        clf = FaceClassifier()
        faces = [_make_detected_face(seed=i) for i in range(3)]
        results = clf.classify_faces(faces)
        assert len(results) == 3
        assert all(r.label == "unknown" for r in results)


class TestFaceClassifierWithIdentities:
    @pytest.fixture()
    def classifier(self) -> FaceClassifier:
        clf = FaceClassifier(similarity_threshold=0.4)
        # Create two distinct identities.
        emb_alice = np.array(_make_embedding(seed=100), dtype=np.float32)
        emb_bob = np.array(_make_embedding(seed=200), dtype=np.float32)
        clf.load_identities([
            {"label": "alice", "centroid_embedding": emb_alice},
            {"label": "bob", "centroid_embedding": emb_bob},
        ])
        return clf

    def test_classifies_matching_face(self, classifier: FaceClassifier) -> None:
        # A face whose embedding is identical to Alice's centroid.
        face = DetectedFace(
            bbox=(0.0, 0.0, 50.0, 50.0),
            confidence=0.98,
            embedding=_make_embedding(seed=100),
        )
        result = classifier.classify_face(face)
        assert result.label == "alice"
        assert result.similarity > 0.99

    def test_classifies_bob(self, classifier: FaceClassifier) -> None:
        face = DetectedFace(
            bbox=(0.0, 0.0, 50.0, 50.0),
            confidence=0.97,
            embedding=_make_embedding(seed=200),
        )
        result = classifier.classify_face(face)
        assert result.label == "bob"
        assert result.similarity > 0.99

    def test_below_threshold_returns_unknown(self, classifier: FaceClassifier) -> None:
        # Use a high threshold so nothing matches.
        classifier.similarity_threshold = 0.9999
        face = _make_detected_face(seed=999)
        result = classifier.classify_face(face)
        assert result.label == "unknown"

    def test_similarity_value_present(self, classifier: FaceClassifier) -> None:
        face = _make_detected_face(seed=300)
        result = classifier.classify_face(face)
        # Whether matched or not, similarity should be a float >= 0.
        assert isinstance(result.similarity, float)
        assert result.similarity >= 0.0


class TestFaceClassifierBelowThreshold:
    def test_high_threshold_rejects_all(self) -> None:
        clf = FaceClassifier(similarity_threshold=1.1)  # impossible to exceed
        centroid = np.array(_make_embedding(seed=0), dtype=np.float32)
        clf.load_identities([{"label": "person", "centroid_embedding": centroid}])
        face = _make_detected_face(seed=0)  # identical to centroid
        result = clf.classify_face(face)
        assert result.label == "unknown"

    def test_borderline_threshold(self) -> None:
        """Exactly-matching embedding should pass when threshold equals 1.0-epsilon."""
        clf = FaceClassifier(similarity_threshold=0.999)
        centroid = np.array(_make_embedding(seed=50), dtype=np.float32)
        clf.load_identities([{"label": "exact", "centroid_embedding": centroid}])
        face = DetectedFace(
            bbox=(0.0, 0.0, 50.0, 50.0),
            confidence=0.9,
            embedding=_make_embedding(seed=50),
        )
        result = clf.classify_face(face)
        assert result.label == "exact"
        assert result.similarity > 0.999


# ---------------------------------------------------------------------------
# crop_face
# ---------------------------------------------------------------------------

class TestCropFace:
    def test_basic_crop(self) -> None:
        img_path = _create_test_image(200, 200)
        cropped = crop_face(img_path, (50.0, 50.0, 60.0, 60.0), padding=0.0)
        assert isinstance(cropped, Image.Image)
        assert cropped.size == (60, 60)

    def test_padding_expands_region(self) -> None:
        img_path = _create_test_image(300, 300)
        # bbox (100, 100, 50, 50) with 0.5 padding → 25px extra on each side.
        cropped = crop_face(img_path, (100.0, 100.0, 50.0, 50.0), padding=0.5)
        w, h = cropped.size
        assert w == 100  # 50 + 25 + 25
        assert h == 100

    def test_clamps_to_image_bounds(self) -> None:
        """Bbox near the top-left corner with large padding must not go negative."""
        img_path = _create_test_image(100, 100)
        # bbox near origin — padding would push left/upper below 0.
        cropped = crop_face(img_path, (2.0, 2.0, 30.0, 30.0), padding=1.0)
        w, h = cropped.size
        # The crop should still succeed; dimensions are clamped to image bounds.
        assert w > 0 and h > 0
        assert w <= 100 and h <= 100

    def test_clamps_to_image_bounds_bottom_right(self) -> None:
        """Bbox near the bottom-right corner with padding must not exceed image size."""
        img_path = _create_test_image(100, 100)
        cropped = crop_face(img_path, (70.0, 70.0, 30.0, 30.0), padding=1.0)
        w, h = cropped.size
        assert w > 0 and h > 0
        assert w <= 100 and h <= 100

    def test_full_image_bbox(self) -> None:
        img_path = _create_test_image(150, 150)
        cropped = crop_face(img_path, (0.0, 0.0, 150.0, 150.0), padding=0.0)
        assert cropped.size == (150, 150)


# ---------------------------------------------------------------------------
# FaceDetector (mocked InsightFace)
# ---------------------------------------------------------------------------

class TestFaceDetectorMocked:
    """Tests for :class:`FaceDetector` that mock InsightFace internals."""

    @pytest.fixture()
    def mock_face_analysis(self) -> MagicMock:
        """Return a mock FaceAnalysis instance."""
        mock_app = MagicMock()
        mock_app.prepare = MagicMock()
        mock_app.get = MagicMock(return_value=[])
        return mock_app

    @patch("photo_search.faces.FaceAnalysis")
    def test_init_calls_prepare(self, mock_fa_cls: MagicMock) -> None:
        mock_app = MagicMock()
        mock_fa_cls.return_value = mock_app

        detector = FaceDetector(model_pack="buffalo_l", min_face_size=25)
        mock_fa_cls.assert_called_once_with(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        mock_app.prepare.assert_called_once_with(ctx_id=0, det_size=(640, 640))
        assert detector.min_face_size == 25

    @patch("photo_search.faces.Image")
    @patch("photo_search.faces.FaceAnalysis")
    def test_detect_returns_detected_faces(
        self, mock_fa_cls: MagicMock, mock_pil_image: MagicMock
    ) -> None:
        # Pillow open returns an RGB image that gets converted to BGR numpy array.
        fake_pil = MagicMock()
        fake_pil.convert.return_value = fake_pil
        fake_pil.__array__ = MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8)
        )
        mock_pil_image.open.return_value = fake_pil

        # Simulate one InsightFace Face result.
        fake_face = MagicMock()
        fake_face.bbox = np.array([10.0, 20.0, 90.0, 120.0])  # x1,y1,x2,y2
        fake_face.det_score = 0.95
        fake_face.embedding = np.random.randn(512).astype(np.float32)

        mock_app = MagicMock()
        mock_app.get.return_value = [fake_face]
        mock_fa_cls.return_value = mock_app

        detector = FaceDetector()
        results = detector.detect_faces("/fake/image.jpg")

        assert len(results) == 1
        det = results[0]
        assert det.bbox == (10.0, 20.0, 80.0, 100.0)  # converted to x,y,w,h
        assert abs(det.confidence - 0.95) < 1e-6
        assert len(det.embedding) == 512

    @patch("photo_search.faces.Image")
    @patch("photo_search.faces.FaceAnalysis")
    def test_detect_filters_small_faces(
        self, mock_fa_cls: MagicMock, mock_pil_image: MagicMock
    ) -> None:
        fake_pil = MagicMock()
        fake_pil.convert.return_value = fake_pil
        fake_pil.__array__ = MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8)
        )
        mock_pil_image.open.return_value = fake_pil

        small_face = MagicMock()
        small_face.bbox = np.array([0.0, 0.0, 10.0, 10.0])  # 10x10
        small_face.det_score = 0.8
        small_face.embedding = np.random.randn(512).astype(np.float32)

        mock_app = MagicMock()
        mock_app.get.return_value = [small_face]
        mock_fa_cls.return_value = mock_app

        detector = FaceDetector(min_face_size=20)
        results = detector.detect_faces("/fake/small.jpg")

        assert len(results) == 0

    @patch("photo_search.faces.Image")
    @patch("photo_search.faces.FaceAnalysis")
    def test_detect_handles_open_failure(
        self, mock_fa_cls: MagicMock, mock_pil_image: MagicMock
    ) -> None:
        mock_pil_image.open.side_effect = FileNotFoundError("not found")
        mock_fa_cls.return_value = MagicMock()

        detector = FaceDetector()
        results = detector.detect_faces("/fake/corrupted.jpg")

        assert results == []

    @patch("photo_search.faces.Image")
    @patch("photo_search.faces.FaceAnalysis")
    def test_detect_handles_exception(
        self, mock_fa_cls: MagicMock, mock_pil_image: MagicMock
    ) -> None:
        mock_pil_image.open.side_effect = RuntimeError("boom")
        mock_fa_cls.return_value = MagicMock()

        detector = FaceDetector()
        results = detector.detect_faces("/fake/error.jpg")

        assert results == []
