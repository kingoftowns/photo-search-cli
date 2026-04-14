"""Text embedding generation using Ollama."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import ollama as ollama_lib

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Generate text embeddings via an Ollama embedding model.

    Provides both low-level ``embed_text`` for arbitrary strings and
    higher-level ``embed_photo`` / ``build_search_text`` helpers that compose
    a rich, multi-faceted description of a photo before embedding it.

    Args:
        base_url: Base URL for the Ollama API (e.g. ``http://localhost:11434``).
        model: Name of the Ollama embedding model. Defaults to
            ``nomic-embed-text`` which produces 768-dimensional vectors.
    """

    def __init__(self, base_url: str, model: str = "nomic-embed-text") -> None:
        self.base_url = base_url
        self.model = model
        self.client = ollama_lib.Client(host=base_url)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector (typically
            768 dimensions for nomic-embed-text).

        Raises:
            ConnectionError: If the Ollama server is unreachable.
            RuntimeError: If the embedding request fails or returns an
                unexpected response.
        """
        try:
            response = self.client.embed(model=self.model, input=text)
        except Exception as exc:
            exc_name = type(exc).__name__.lower()
            exc_msg = str(exc).lower()

            if "connect" in exc_name or "connect" in exc_msg:
                raise ConnectionError(
                    f"Could not connect to Ollama at {self.base_url}: {exc}"
                ) from exc

            raise RuntimeError(
                f"Ollama embedding failed: {exc}"
            ) from exc

        # Handle both dict and object response styles
        try:
            if isinstance(response, dict):
                embeddings = response["embeddings"]
            else:
                embeddings = response.embeddings
        except (KeyError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama embed response structure: {response!r}"
            ) from exc

        if not embeddings or not embeddings[0]:
            raise RuntimeError("Ollama returned empty embedding")

        return list(embeddings[0])

    @staticmethod
    def build_search_text(
        caption: Optional[str],
        face_labels: list[str],
        location: Optional[str],
        date_taken: Optional[datetime],
        camera: Optional[str],
    ) -> str:
        """Build a rich combined text from all photo facets for embedding.

        The resulting text is designed so that semantic search queries about
        any single facet (people, place, date, camera, scene content) will
        produce a strong match against the combined embedding.

        Args:
            caption: VLM-generated scene description.
            face_labels: Names of identified people (may be empty).
            location: Reverse-geocoded location string.
            date_taken: When the photo was taken.
            camera: Camera make/model string.

        Returns:
            A multi-line string suitable for embedding.
        """
        lines: list[str] = []

        # Caption line
        lines.append(caption if caption else "No description available")

        # People line
        if face_labels:
            lines.append(f"People: {', '.join(face_labels)}")
        else:
            lines.append("People: No people identified")

        # Location line
        lines.append(f"Location: {location or 'Unknown location'}")

        # Date line
        if date_taken is not None:
            formatted_date = date_taken.strftime("%B %d, %Y")
            lines.append(f"Date: {formatted_date}")
        else:
            lines.append("Date: Unknown date")

        # Camera line
        lines.append(f"Camera: {camera or 'Unknown camera'}")

        return "\n".join(lines)

    def embed_photo(
        self,
        caption: Optional[str],
        face_labels: list[str],
        location: Optional[str],
        date_taken: Optional[datetime],
        camera: Optional[str],
    ) -> tuple[str, list[float]]:
        """Build a combined search text for a photo and embed it.

        Convenience method that calls ``build_search_text`` followed by
        ``embed_text``.

        Args:
            caption: VLM-generated scene description.
            face_labels: Names of identified people.
            location: Reverse-geocoded location string.
            date_taken: When the photo was taken.
            camera: Camera make/model string.

        Returns:
            A tuple of ``(search_text, embedding_vector)`` where search_text
            is the combined string that was embedded and embedding_vector is
            the resulting list of floats.
        """
        search_text = self.build_search_text(
            caption=caption,
            face_labels=face_labels,
            location=location,
            date_taken=date_taken,
            camera=camera,
        )

        logger.debug(
            "Embedding photo text (%d chars) with model %s",
            len(search_text),
            self.model,
        )

        embedding = self.embed_text(search_text)
        return search_text, embedding
