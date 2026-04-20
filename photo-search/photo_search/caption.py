"""Photo captioning using a configurable vision-language model backend.

Two providers are currently supported:

* :class:`OllamaCaptioner` -- local inference via an Ollama server.
* :class:`AnthropicCaptioner` -- Anthropic's Claude API (vision-capable
  models such as ``claude-haiku-4-5``).

Both implementations share image preprocessing (HEIC conversion +
optional resize to bound prompt cost / inference time) and return a
:class:`PhotoCaption`. Use :func:`create_captioner` with an
:class:`~photo_search.config.AppConfig` to get the right implementation
for the currently-configured provider.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from typing import Optional

import ollama as ollama_lib
from PIL import Image
from pillow_heif import register_heif_opener

from photo_search.config import AppConfig
from photo_search.models import PhotoCaption

register_heif_opener()

logger = logging.getLogger(__name__)

CAPTION_PROMPT: str = (
    "Describe this photo in detail for a search index. Write a single "
    "paragraph of plain prose — no markdown headers (no '#' or '##'), no "
    "bullet lists, no section labels.\n\n"
    "Lead with the most distinctive, searchable details: proper nouns, "
    "team names, any text visible on signs/shirts/jerseys/banners "
    "(transcribe exactly, and include this text even when it's also a "
    "common word like an animal name — it's a proper noun in context), "
    "brand names, places, and the specific occasion.\n\n"
    "Then cover: who is in the photo and what they are doing; the "
    "setting/location (indoor/outdoor, type of place); time of day, "
    "lighting, and weather if apparent; notable objects, activities, "
    "sports, equipment; and the overall mood or occasion (birthday, "
    "vacation, practice, etc.). Be specific and concrete."
)


# ---------------------------------------------------------------------------
# Base class (shared image preprocessing)
# ---------------------------------------------------------------------------


class BaseCaptioner:
    """Abstract base for caption providers.

    Subclasses must implement :meth:`caption_photo`. The base class owns
    the shared HEIC-to-JPEG conversion and resize logic so that every
    provider sees a normalized JPEG.
    """

    def __init__(self, resize_max_dim: int = 1024) -> None:
        self.resize_max_dim = resize_max_dim

    # Subclasses override this.
    def caption_photo(self, image_path: str) -> PhotoCaption:  # pragma: no cover
        raise NotImplementedError

    def _resize_image(self, image_path: str) -> str:
        """Resize and/or convert an image for VLM consumption.

        HEIC images are always converted to JPEG (no supported provider
        reads HEIC natively). Other images are only converted if their
        longest edge exceeds ``resize_max_dim``.

        Args:
            image_path: Path to the source image.

        Returns:
            Path to a JPEG file. Either the original path (no conversion
            needed) or a new temporary file that the caller is
            responsible for cleaning up.
        """
        image = Image.open(image_path)
        width, height = image.size
        longest_edge = max(width, height)

        ext = os.path.splitext(image_path)[1].lower()
        is_heic = ext in (".heic", ".heif")

        needs_resize = longest_edge > self.resize_max_dim
        needs_conversion = is_heic

        if not needs_resize and not needs_conversion:
            return image_path

        if needs_resize:
            scale = self.resize_max_dim / longest_edge
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.debug(
                "Resized %s from %dx%d to %dx%d",
                image_path,
                width,
                height,
                new_width,
                new_height,
            )

        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="photo_caption_")
        os.close(tmp_fd)

        try:
            image.save(tmp_path, format="JPEG", quality=85)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return tmp_path


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


class OllamaCaptioner(BaseCaptioner):
    """Generate captions for photos using an Ollama vision-language model.

    Args:
        base_url: Base URL for the Ollama API (e.g. ``http://localhost:11434``).
        model: Name of the Ollama vision model to use.
        timeout: Request timeout in seconds for the Ollama API call.
        resize_max_dim: Maximum size of the longest image edge.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
        resize_max_dim: int = 1024,
    ) -> None:
        super().__init__(resize_max_dim=resize_max_dim)
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.client = ollama_lib.Client(host=base_url, timeout=timeout)

    def caption_photo(self, image_path: str) -> PhotoCaption:
        """Generate a caption for a single photo via Ollama.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
            TimeoutError: If the Ollama request exceeds the configured timeout.
            RuntimeError: If the model returns an unexpected response.
        """
        temp_path: Optional[str] = None
        try:
            resized_path = self._resize_image(image_path)
            temp_path = resized_path if resized_path != image_path else None

            logger.info(
                "Captioning %s with model %s (image: %s)",
                image_path,
                self.model,
                resized_path,
            )

            start = time.monotonic()

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": CAPTION_PROMPT,
                            "images": [resized_path],
                        }
                    ],
                )
            except Exception as exc:
                exc_name = type(exc).__name__.lower()
                exc_msg = str(exc).lower()

                if "timeout" in exc_name or "timeout" in exc_msg:
                    raise TimeoutError(
                        f"Ollama request timed out after {self.timeout}s "
                        f"while captioning {image_path}"
                    ) from exc

                if "connect" in exc_name or "connect" in exc_msg:
                    raise ConnectionError(
                        f"Could not connect to Ollama at {self.base_url}: {exc}"
                    ) from exc

                raise RuntimeError(
                    f"Ollama captioning failed for {image_path}: {exc}"
                ) from exc

            elapsed = time.monotonic() - start

            try:
                if isinstance(response, dict):
                    caption_text = response["message"]["content"]
                else:
                    caption_text = response.message.content
            except (KeyError, AttributeError, TypeError) as exc:
                raise RuntimeError(
                    f"Unexpected Ollama response structure: {response!r}"
                ) from exc

            if not caption_text or not isinstance(caption_text, str):
                raise RuntimeError(
                    f"Ollama returned empty caption for {image_path}"
                )

            logger.info(
                "Captioned %s in %.1fs (%d chars)",
                image_path,
                elapsed,
                len(caption_text),
            )

            return PhotoCaption(
                caption=caption_text.strip(),
                model=self.model,
                generation_time_seconds=round(elapsed, 3),
            )

        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.warning("Failed to clean up temp file: %s", temp_path)


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class AnthropicCaptioner(BaseCaptioner):
    """Generate captions via Anthropic's Claude API.

    The API key is read from the ``ANTHROPIC_API_KEY`` environment
    variable by the official SDK. Vision-capable models include
    ``claude-haiku-4-5``, ``claude-sonnet-4-5``, and ``claude-opus-4-6``;
    Haiku is the cost-effective default for bulk captioning.

    Args:
        model: Anthropic model id.
        max_tokens: Cap on generated caption length (in tokens).
        api_key: Optional explicit key. When ``None`` the SDK reads it
            from ``ANTHROPIC_API_KEY``.
        resize_max_dim: Maximum size of the longest image edge.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 300,
        api_key: Optional[str] = None,
        resize_max_dim: int = 1024,
    ) -> None:
        super().__init__(resize_max_dim=resize_max_dim)
        self.model = model
        self.max_tokens = max_tokens

        try:
            import anthropic  # imported lazily so Ollama-only installs work
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required for AnthropicCaptioner. "
                "Install it with: pip install anthropic"
            ) from exc

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def caption_photo(self, image_path: str) -> PhotoCaption:
        """Generate a caption for a single photo via the Anthropic API.

        Raises:
            ConnectionError: If the API is unreachable.
            TimeoutError: If the request times out.
            RuntimeError: If the API returns an unexpected response.
        """
        temp_path: Optional[str] = None
        try:
            resized_path = self._resize_image(image_path)
            temp_path = resized_path if resized_path != image_path else None

            with open(resized_path, "rb") as fh:
                img_b64 = base64.standard_b64encode(fh.read()).decode()

            logger.info(
                "Captioning %s with model %s (image: %s)",
                image_path,
                self.model,
                resized_path,
            )

            start = time.monotonic()

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": img_b64,
                                    },
                                },
                                {"type": "text", "text": CAPTION_PROMPT},
                            ],
                        }
                    ],
                )
            except self._anthropic.APITimeoutError as exc:
                raise TimeoutError(
                    f"Anthropic request timed out while captioning {image_path}"
                ) from exc
            except self._anthropic.APIConnectionError as exc:
                raise ConnectionError(
                    f"Could not connect to Anthropic API: {exc}"
                ) from exc
            except self._anthropic.APIStatusError as exc:
                raise RuntimeError(
                    f"Anthropic API error ({exc.status_code}) "
                    f"captioning {image_path}: {exc.message}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Anthropic captioning failed for {image_path}: {exc}"
                ) from exc

            elapsed = time.monotonic() - start

            # Extract text from the first text-type content block.
            caption_text: Optional[str] = None
            try:
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        caption_text = block.text
                        break
            except AttributeError as exc:
                raise RuntimeError(
                    f"Unexpected Anthropic response structure: {response!r}"
                ) from exc

            if not caption_text or not isinstance(caption_text, str):
                raise RuntimeError(
                    f"Anthropic returned empty caption for {image_path}"
                )

            logger.info(
                "Captioned %s in %.1fs (%d chars)",
                image_path,
                elapsed,
                len(caption_text),
            )

            return PhotoCaption(
                caption=caption_text.strip(),
                model=self.model,
                generation_time_seconds=round(elapsed, 3),
            )

        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.warning("Failed to clean up temp file: %s", temp_path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_captioner(config: AppConfig) -> BaseCaptioner:
    """Build a captioner from an :class:`AppConfig`.

    Dispatches on ``config.captioner.provider``:

    * ``"ollama"`` -> :class:`OllamaCaptioner` wired with the existing
      ``config.ollama`` values.
    * ``"anthropic"`` -> :class:`AnthropicCaptioner` wired with
      ``config.captioner.anthropic`` (API key from ``ANTHROPIC_API_KEY``).

    Args:
        config: Fully-resolved application configuration.

    Returns:
        A concrete :class:`BaseCaptioner` subclass ready for use.

    Raises:
        ValueError: If ``config.captioner.provider`` is unrecognized.
    """
    provider = config.captioner.provider
    resize_max_dim = config.pipeline.resize_max_dimension

    if provider == "ollama":
        return OllamaCaptioner(
            base_url=config.ollama.base_url,
            model=config.ollama.vision_model,
            timeout=config.ollama.request_timeout,
            resize_max_dim=resize_max_dim,
        )

    if provider == "anthropic":
        ac = config.captioner.anthropic
        return AnthropicCaptioner(
            model=ac.model,
            max_tokens=ac.max_tokens,
            resize_max_dim=resize_max_dim,
        )

    raise ValueError(f"Unknown captioner provider: {provider!r}")
