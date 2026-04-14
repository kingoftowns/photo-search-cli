"""Photo captioning using Ollama vision-language models."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

import ollama as ollama_lib
from PIL import Image
from pillow_heif import register_heif_opener

from photo_search.models import PhotoCaption

register_heif_opener()

logger = logging.getLogger(__name__)

CAPTION_PROMPT: str = (
    "Describe this photo in detail for a search index. Include:\n"
    "- People present and what they are doing\n"
    "- The setting/location (indoor/outdoor, type of place)\n"
    "- Time of day and weather if apparent\n"
    "- Notable objects, activities, sports, equipment\n"
    "- Any visible text (signs, shirts, etc.)\n"
    "- The overall mood or occasion (birthday, vacation, practice, etc.)\n"
    "Be specific and descriptive. This description will be used for semantic search."
)


class PhotoCaptioner:
    """Generate captions for photos using an Ollama vision-language model.

    The captioner handles image resizing (required for performance and to avoid
    exceeding model context limits) and HEIC-to-JPEG conversion (Ollama does not
    natively support HEIC).

    Args:
        base_url: Base URL for the Ollama API (e.g. ``http://localhost:11434``).
        model: Name of the Ollama vision model to use.
        timeout: Request timeout in seconds for the Ollama API call.
        resize_max_dim: Maximum size of the longest image edge. Images with a
            longer edge will be proportionally resized before captioning.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
        resize_max_dim: int = 1536,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.resize_max_dim = resize_max_dim
        self.client = ollama_lib.Client(host=base_url, timeout=timeout)

    def caption_photo(self, image_path: str) -> PhotoCaption:
        """Generate a caption for a single photo.

        The image is resized and converted to JPEG before being sent to the
        vision-language model. Temporary files are cleaned up after captioning.

        Args:
            image_path: Path to the image file to caption.

        Returns:
            A PhotoCaption containing the generated caption text, the model
            name, and the generation wall-clock time.

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

            # Extract caption text -- handle both dict and object responses
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
            # Clean up temporary resized file
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.warning("Failed to clean up temp file: %s", temp_path)

    def _resize_image(self, image_path: str) -> str:
        """Resize and/or convert an image for Ollama consumption.

        HEIC images are always converted to JPEG because Ollama does not
        support HEIC natively. Other images are only converted if their
        longest edge exceeds ``resize_max_dim``.

        Args:
            image_path: Path to the source image.

        Returns:
            Path to a JPEG file suitable for sending to Ollama. This is
            either the original path (if no conversion was needed) or a
            new temporary file path.
        """
        image = Image.open(image_path)
        width, height = image.size
        longest_edge = max(width, height)

        # Determine if the file is HEIC (Ollama cannot read HEIC directly)
        ext = os.path.splitext(image_path)[1].lower()
        is_heic = ext in (".heic", ".heif")

        needs_resize = longest_edge > self.resize_max_dim
        needs_conversion = is_heic

        if not needs_resize and not needs_conversion:
            return image_path

        # Resize if needed
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

        # Convert to RGB if necessary (e.g. RGBA, palette mode)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Save to a temporary JPEG file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="photo_caption_")
        os.close(tmp_fd)

        try:
            image.save(tmp_path, format="JPEG", quality=85)
        except Exception:
            # Clean up on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return tmp_path
