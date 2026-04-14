"""Tests for text embedding."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from photo_search.embed import TextEmbedder


# ---------------------------------------------------------------------------
# build_search_text
# ---------------------------------------------------------------------------

class TestBuildSearchText:
    def test_full_fields(self) -> None:
        """All fields provided should produce a rich multi-line text."""
        text = TextEmbedder.build_search_text(
            caption="A family playing soccer in a park.",
            face_labels=["Alice", "Bob"],
            location="Irvine, California, US",
            date_taken=datetime(2024, 6, 15, 14, 30),
            camera="Apple iPhone 15 Pro",
        )
        assert "A family playing soccer in a park." in text
        assert "People: Alice, Bob" in text
        assert "Location: Irvine, California, US" in text
        assert "Date: June 15, 2024" in text
        assert "Camera: Apple iPhone 15 Pro" in text

    def test_minimal_caption_only(self) -> None:
        """Only caption provided, rest None/empty."""
        text = TextEmbedder.build_search_text(
            caption="A sunset over the ocean.",
            face_labels=[],
            location=None,
            date_taken=None,
            camera=None,
        )
        assert "A sunset over the ocean." in text
        assert "No people identified" in text
        assert "Unknown location" in text
        assert "Unknown date" in text
        assert "Unknown camera" in text

    def test_no_faces(self) -> None:
        """Empty face list should produce 'No people identified'."""
        text = TextEmbedder.build_search_text(
            caption="test",
            face_labels=[],
            location="Paris, Ile-de-France, FR",
            date_taken=datetime(2023, 12, 25),
            camera="Sony A7IV",
        )
        assert "No people identified" in text
        assert "Paris" in text

    def test_no_caption(self) -> None:
        """None caption should produce a placeholder."""
        text = TextEmbedder.build_search_text(
            caption=None,
            face_labels=["Charlie"],
            location=None,
            date_taken=None,
            camera=None,
        )
        assert "No description available" in text
        assert "People: Charlie" in text

    def test_multiple_faces(self) -> None:
        """Multiple faces should be comma-separated."""
        text = TextEmbedder.build_search_text(
            caption="Group photo.",
            face_labels=["Alice", "Bob", "Charlie"],
            location=None,
            date_taken=None,
            camera=None,
        )
        assert "People: Alice, Bob, Charlie" in text

    def test_date_formatting(self) -> None:
        """Verify the date is formatted as 'Month DD, YYYY'."""
        text = TextEmbedder.build_search_text(
            caption="test",
            face_labels=[],
            location=None,
            date_taken=datetime(2025, 1, 5),
            camera=None,
        )
        assert "Date: January 05, 2025" in text

    def test_returns_string(self) -> None:
        """Return value should always be a string."""
        text = TextEmbedder.build_search_text(
            caption=None,
            face_labels=[],
            location=None,
            date_taken=None,
            camera=None,
        )
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# embed_text (mocked Ollama)
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_embed_text_returns_list(self) -> None:
        """embed_text should return a list of floats."""
        fake_embedding = [0.1] * 768

        with patch("photo_search.embed.ollama_lib") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.Client.return_value = mock_client

            # Object-style response
            mock_response = MagicMock()
            mock_response.embeddings = [fake_embedding]
            mock_client.embed.return_value = mock_response

            embedder = TextEmbedder(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
            )
            result = embedder.embed_text("A test sentence.")

        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(v, float) for v in result)

    def test_embed_text_dict_response(self) -> None:
        """Handle dict-style Ollama response."""
        fake_embedding = [0.5] * 768

        with patch("photo_search.embed.ollama_lib") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.Client.return_value = mock_client

            mock_client.embed.return_value = {
                "embeddings": [fake_embedding]
            }

            embedder = TextEmbedder(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
            )
            result = embedder.embed_text("Another test.")

        assert len(result) == 768

    def test_embed_text_connection_error(self) -> None:
        """Connection failure should raise ConnectionError."""
        with patch("photo_search.embed.ollama_lib") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.Client.return_value = mock_client

            mock_client.embed.side_effect = Exception(
                "connect ECONNREFUSED 127.0.0.1:11434"
            )

            embedder = TextEmbedder(base_url="http://localhost:11434")

            with pytest.raises(ConnectionError, match="Could not connect"):
                embedder.embed_text("test")

    def test_embed_text_empty_response(self) -> None:
        """Empty embedding response should raise RuntimeError."""
        with patch("photo_search.embed.ollama_lib") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.embeddings = [[]]
            mock_client.embed.return_value = mock_response

            embedder = TextEmbedder(base_url="http://localhost:11434")

            with pytest.raises(RuntimeError, match="empty embedding"):
                embedder.embed_text("test")


# ---------------------------------------------------------------------------
# embed_photo (integration of build + embed)
# ---------------------------------------------------------------------------

class TestEmbedPhoto:
    def test_embed_photo_returns_tuple(self) -> None:
        """embed_photo should return (search_text, embedding_vector)."""
        fake_embedding = [0.25] * 768

        with patch("photo_search.embed.ollama_lib") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.embeddings = [fake_embedding]
            mock_client.embed.return_value = mock_response

            embedder = TextEmbedder(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
            )
            search_text, embedding = embedder.embed_photo(
                caption="Kids at a birthday party.",
                face_labels=["Alice"],
                location="Irvine, California, US",
                date_taken=datetime(2024, 3, 10),
                camera="Apple iPhone 15 Pro",
            )

        assert isinstance(search_text, str)
        assert "Kids at a birthday party." in search_text
        assert "People: Alice" in search_text
        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_embed_photo_passes_combined_text(self) -> None:
        """Verify the combined text is what gets sent to embed_text."""
        fake_embedding = [0.1] * 768

        with patch("photo_search.embed.ollama_lib") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.Client.return_value = mock_client

            mock_response = MagicMock()
            mock_response.embeddings = [fake_embedding]
            mock_client.embed.return_value = mock_response

            embedder = TextEmbedder(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
            )
            search_text, _ = embedder.embed_photo(
                caption="A test caption.",
                face_labels=[],
                location=None,
                date_taken=None,
                camera=None,
            )

        # Verify the text that was sent to the embed endpoint
        call_args = mock_client.embed.call_args
        sent_text = call_args.kwargs.get("input") or call_args[1].get("input")
        assert sent_text == search_text
