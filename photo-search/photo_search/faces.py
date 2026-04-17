"""Face detection, embedding extraction, and classification using InsightFace."""

from __future__ import annotations

import io
import logging

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.metrics.pairwise import cosine_similarity

from photo_search.models import DetectedFace, IdentifiedFace

logger = logging.getLogger(__name__)

# Register HEIC/HEIF support so PIL can open Apple photos.
register_heif_opener()


class FaceDetector:
    """Detect faces in images and extract 512-dim ArcFace embeddings using InsightFace."""

    def __init__(self, model_pack: str = "buffalo_l", min_face_size: int = 20) -> None:
        """Initialize the InsightFace FaceAnalysis app.

        Args:
            model_pack: InsightFace model pack name (e.g. ``"buffalo_l"``).
            min_face_size: Minimum face width/height in pixels.  Detections
                smaller than this on either axis are silently discarded.
        """
        self.min_face_size = min_face_size
        self._app = FaceAnalysis(
            name=model_pack,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info(
            "FaceDetector initialised with model_pack=%s, min_face_size=%d",
            model_pack,
            min_face_size,
        )

    def detect_faces(self, image_path: str) -> list[DetectedFace]:
        """Detect faces in *image_path* and return structured results.

        Each returned :class:`DetectedFace` contains:
        * ``bbox`` — ``(x, y, w, h)`` with top-left origin.
        * ``confidence`` — detector score.
        * ``embedding`` — 512-dim ArcFace embedding as ``list[float]``.

        Returns an empty list when the image cannot be read or detection fails.
        """
        try:
            # Read the entire file into memory first.  HEIC files use
            # random-access seeking via libheif, which can silently fail
            # over NFS (returning a valid but empty image).  Buffering
            # avoids this.
            with open(image_path, "rb") as fh:
                pil_img = Image.open(io.BytesIO(fh.read()))
            pil_img = pil_img.convert("RGB")
            img = np.array(pil_img)
            img = img[:, :, ::-1]  # RGB → BGR for InsightFace/OpenCV

            raw_faces = self._app.get(img)
        except Exception:
            logger.exception("Face detection failed for %s", image_path)
            return []

        detected: list[DetectedFace] = []
        for face in raw_faces:
            # InsightFace bbox is [x1, y1, x2, y2] (float).
            x1, y1, x2, y2 = face.bbox
            w = x2 - x1
            h = y2 - y1

            if w < self.min_face_size or h < self.min_face_size:
                logger.debug(
                    "Skipping small face (%dx%d) in %s", int(w), int(h), image_path
                )
                continue

            detected.append(
                DetectedFace(
                    bbox=(float(x1), float(y1), float(w), float(h)),
                    confidence=float(face.det_score),
                    embedding=[float(v) for v in face.embedding],
                )
            )

        logger.debug("Detected %d face(s) in %s", len(detected), image_path)
        return detected


class FaceClassifier:
    """Classify detected faces against a bank of known identities.

    Uses cosine similarity between ArcFace embeddings and stored identity
    centroids to assign labels.
    """

    def __init__(self, similarity_threshold: float = 0.4) -> None:
        """Create a classifier.

        Args:
            similarity_threshold: Minimum cosine similarity to accept a match.
        """
        self.similarity_threshold = similarity_threshold
        self.known_identities: dict[str, np.ndarray] = {}

    def load_identities(self, identities: list[dict]) -> None:  # noqa: ANN401
        """Load known identities from a list of dicts.

        Each dict must have at least:
        * ``"label"`` — unique string identifier (e.g. ``"michael"``).
        * ``"centroid_embedding"`` — centroid as :class:`numpy.ndarray`.
        """
        self.known_identities = {
            identity["label"]: np.asarray(identity["centroid_embedding"], dtype=np.float32)
            for identity in identities
        }
        logger.info("Loaded %d known identities", len(self.known_identities))

    def classify_face(self, face: DetectedFace) -> IdentifiedFace:
        """Match a single :class:`DetectedFace` to the closest known identity.

        Returns an :class:`IdentifiedFace` with ``label="unknown"`` when no
        identity exceeds *similarity_threshold* (or no identities are loaded).
        """
        if not self.known_identities:
            return IdentifiedFace(
                bbox=face.bbox,
                confidence=face.confidence,
                embedding=face.embedding,
                label="unknown",
                similarity=0.0,
            )

        face_vec = np.asarray(face.embedding, dtype=np.float32).reshape(1, -1)

        labels: list[str] = []
        centroids: list[np.ndarray] = []
        for label, centroid in self.known_identities.items():
            labels.append(label)
            centroids.append(centroid)

        centroid_matrix = np.vstack(centroids)  # (N, 512)
        similarities = cosine_similarity(face_vec, centroid_matrix).flatten()  # (N,)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.similarity_threshold:
            matched_label = labels[best_idx]
        else:
            matched_label = "unknown"

        return IdentifiedFace(
            bbox=face.bbox,
            confidence=face.confidence,
            embedding=face.embedding,
            label=matched_label,
            similarity=best_sim,
        )

    def classify_faces(self, faces: list[DetectedFace]) -> list[IdentifiedFace]:
        """Classify a batch of detected faces."""
        return [self.classify_face(face) for face in faces]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def compute_centroid(embeddings: list[list[float]]) -> np.ndarray:
    """Compute a unit-length centroid from a collection of embeddings.

    Args:
        embeddings: List of 512-dim embedding vectors (as plain lists).

    Returns:
        A 1-D :class:`numpy.ndarray` of shape ``(512,)`` with unit L2 norm.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    mean_vec = matrix.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return mean_vec.astype(np.float32)
    return (mean_vec / norm).astype(np.float32)


def crop_face(
    image_path: str,
    bbox: tuple[float, float, float, float],
    padding: float = 0.3,
) -> Image.Image:
    """Crop a face region from an image with configurable padding.

    Args:
        image_path: Path to the source image.
        bbox: ``(x, y, w, h)`` bounding box (top-left origin).
        padding: Fraction by which to expand the crop on every side
            (e.g. ``0.3`` adds 30 % of *w*/*h* as padding).

    Returns:
        A :class:`PIL.Image.Image` containing the cropped face region.
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    x, y, w, h = bbox
    pad_x = w * padding
    pad_y = h * padding

    left = max(0, x - pad_x)
    upper = max(0, y - pad_y)
    right = min(img_w, x + w + pad_x)
    lower = min(img_h, y + h + pad_y)

    return img.crop((left, upper, right, lower))
