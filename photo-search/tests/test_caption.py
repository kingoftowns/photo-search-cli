"""Tests for VLM captioning."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from photo_search.caption import CAPTION_PROMPT, PhotoCaptioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(width: int = 100, height: int = 100, suffix: str = ".jpg") -> str:
    """Create a small temporary image and return its path."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fmt = "JPEG" if suffix.lower() in (".jpg", ".jpeg") else "PNG"
    img.save(tmp, format=fmt)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# CAPTION_PROMPT
# ---------------------------------------------------------------------------

class TestCaptionPrompt:
    def test_caption_prompt_exists(self) -> None:
        """CAPTION_PROMPT should be a non-empty string."""
        assert isinstance(CAPTION_PROMPT, str)
        assert len(CAPTION_PROMPT) > 50

    def test_caption_prompt_mentions_search(self) -> None:
        """Prompt should reference search indexing to guide the VLM."""
        assert "search" in CAPTION_PROMPT.lower()

    def test_caption_prompt_mentions_people(self) -> None:
        """Prompt should ask about people in the photo."""
        assert "people" in CAPTION_PROMPT.lower()


# ---------------------------------------------------------------------------
# _resize_image
# ---------------------------------------------------------------------------

class TestResizeImage:
    def test_small_image_no_resize(self) -> None:
        """An image smaller than max dim should be returned as-is (JPEG)."""
        path = _create_test_image(200, 150)
        try:
            captioner = PhotoCaptioner(
                base_url="http://localhost:11434",
                model="test-model",
                resize_max_dim=1536,
            )
            result = captioner._resize_image(path)
            # Small JPEG should not be resized — returns original path
            assert result == path
        finally:
            os.unlink(path)

    def test_large_image_gets_resized(self) -> None:
        """An image larger than max dim should be resized to fit."""
        path = _create_test_image(3000, 2000)
        try:
            captioner = PhotoCaptioner(
                base_url="http://localhost:11434",
                model="test-model",
                resize_max_dim=1536,
            )
            result = captioner._resize_image(path)
            assert result != path  # should return a new temp file

            # Verify the resized image dimensions
            resized = Image.open(result)
            w, h = resized.size
            assert max(w, h) <= 1536
            # Aspect ratio should be roughly preserved
            original_ratio = 3000 / 2000
            new_ratio = w / h
            assert abs(original_ratio - new_ratio) < 0.05

            os.unlink(result)
        finally:
            os.unlink(path)

    def test_heic_gets_converted(self) -> None:
        """HEIC files should always be converted to JPEG even if small."""
        # We can't easily create a real HEIC, so we create a JPEG and
        # rename it to .heic, then mock Image.open to handle it.
        path = _create_test_image(200, 150, suffix=".jpg")
        heic_path = path.replace(".jpg", ".heic")
        os.rename(path, heic_path)
        try:
            captioner = PhotoCaptioner(
                base_url="http://localhost:11434",
                model="test-model",
                resize_max_dim=1536,
            )
            # Mock Image.open to work with our fake HEIC
            with patch("photo_search.caption.Image") as mock_img_mod:
                mock_img = MagicMock()
                mock_img.size = (200, 150)
                mock_img.mode = "RGB"
                mock_img_mod.open.return_value = mock_img
                mock_img_mod.LANCZOS = Image.LANCZOS

                result = captioner._resize_image(heic_path)

            # Should be a temp JPEG file, not the original HEIC
            assert result != heic_path
            assert result.endswith(".jpg")

            # Clean up temp
            if os.path.exists(result):
                os.unlink(result)
        finally:
            if os.path.exists(heic_path):
                os.unlink(heic_path)

    def test_exact_boundary_no_resize(self) -> None:
        """Image with longest edge exactly at max dim should not be resized."""
        path = _create_test_image(1536, 1024)
        try:
            captioner = PhotoCaptioner(
                base_url="http://localhost:11434",
                model="test-model",
                resize_max_dim=1536,
            )
            result = captioner._resize_image(path)
            assert result == path
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# caption_photo (mocked Ollama)
# ---------------------------------------------------------------------------

class TestCaptionPhoto:
    def test_caption_photo_success(self) -> None:
        """Successful captioning returns a PhotoCaption with text and timing."""
        path = _create_test_image(200, 150)
        try:
            with patch("photo_search.caption.ollama_lib") as mock_ollama:
                mock_client = MagicMock()
                mock_ollama.Client.return_value = mock_client

                # Simulate a successful response
                mock_response = MagicMock()
                mock_response.message.content = (
                    "A family photo at the beach during sunset."
                )
                mock_client.chat.return_value = mock_response

                captioner = PhotoCaptioner(
                    base_url="http://localhost:11434",
                    model="qwen2.5vl:7b",
                    resize_max_dim=1536,
                )
                result = captioner.caption_photo(path)

            assert result.caption == "A family photo at the beach during sunset."
            assert result.model == "qwen2.5vl:7b"
            assert result.generation_time_seconds >= 0
        finally:
            os.unlink(path)

    def test_caption_photo_timeout(self) -> None:
        """Timeout from Ollama should raise TimeoutError."""
        path = _create_test_image(200, 150)
        try:
            with patch("photo_search.caption.ollama_lib") as mock_ollama:
                mock_client = MagicMock()
                mock_ollama.Client.return_value = mock_client

                # Simulate a timeout
                mock_client.chat.side_effect = Exception("request timeout exceeded")

                captioner = PhotoCaptioner(
                    base_url="http://localhost:11434",
                    model="qwen2.5vl:7b",
                    timeout=10,
                    resize_max_dim=1536,
                )

                with pytest.raises(TimeoutError, match="timed out"):
                    captioner.caption_photo(path)
        finally:
            os.unlink(path)

    def test_caption_photo_connection_error(self) -> None:
        """Connection failure should raise ConnectionError."""
        path = _create_test_image(200, 150)
        try:
            with patch("photo_search.caption.ollama_lib") as mock_ollama:
                mock_client = MagicMock()
                mock_ollama.Client.return_value = mock_client

                mock_client.chat.side_effect = Exception(
                    "connect ECONNREFUSED 127.0.0.1:11434"
                )

                captioner = PhotoCaptioner(
                    base_url="http://localhost:11434",
                    model="qwen2.5vl:7b",
                    resize_max_dim=1536,
                )

                with pytest.raises(ConnectionError, match="Could not connect"):
                    captioner.caption_photo(path)
        finally:
            os.unlink(path)

    def test_caption_photo_empty_response(self) -> None:
        """Empty response from VLM should raise RuntimeError."""
        path = _create_test_image(200, 150)
        try:
            with patch("photo_search.caption.ollama_lib") as mock_ollama:
                mock_client = MagicMock()
                mock_ollama.Client.return_value = mock_client

                mock_response = MagicMock()
                mock_response.message.content = ""
                mock_client.chat.return_value = mock_response

                captioner = PhotoCaptioner(
                    base_url="http://localhost:11434",
                    model="qwen2.5vl:7b",
                    resize_max_dim=1536,
                )

                with pytest.raises(RuntimeError, match="empty caption"):
                    captioner.caption_photo(path)
        finally:
            os.unlink(path)

    def test_caption_photo_dict_response(self) -> None:
        """Some Ollama SDK versions return dicts instead of objects."""
        path = _create_test_image(200, 150)
        try:
            with patch("photo_search.caption.ollama_lib") as mock_ollama:
                mock_client = MagicMock()
                mock_ollama.Client.return_value = mock_client

                # Return a plain dict response
                mock_client.chat.return_value = {
                    "message": {"content": "A dict-style response."}
                }

                captioner = PhotoCaptioner(
                    base_url="http://localhost:11434",
                    model="qwen2.5vl:7b",
                    resize_max_dim=1536,
                )
                result = captioner.caption_photo(path)

            assert result.caption == "A dict-style response."
        finally:
            os.unlink(path)
