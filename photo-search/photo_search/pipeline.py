"""Photo indexing pipeline orchestrator with resume capability.

Wires together all processing modules (EXIF extraction, face detection,
captioning, geocoding, embedding) into a single resumable pipeline.
Per-file, per-stage status is persisted in PostgreSQL so the pipeline can
be interrupted at any point and resumed without re-processing completed
stages.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from photo_search.caption import BaseCaptioner, create_captioner
from photo_search.config import AppConfig
from photo_search.embed import TextEmbedder
from photo_search.exif import extract_metadata
from photo_search.faces import FaceClassifier, FaceDetector
from photo_search.geocode import reverse_geocode
from photo_search.models import (
    IndexedPhoto,
    IndexingStatus,
    PhotoCaption,
    PhotoMetadata,
)
from photo_search.storage import PostgresStorage, QdrantStorage

logger = logging.getLogger(__name__)
console = Console()

# All recognized pipeline stages in execution order.
ALL_STAGES = {"exif", "faces", "caption", "embed"}

# Mapping from stage name to the IndexingStatus boolean that tracks it.
_STAGE_TO_STATUS_FIELD: dict[str, str] = {
    "exif": "exif_extracted",
    "faces": "faces_extracted",
    "caption": "captioned",
    "embed": "embedded",
}


class IndexingPipeline:
    """Orchestrates the full photo indexing workflow.

    The pipeline is designed to be **resumable** at the granularity of
    individual files *and* individual stages.  After each stage completes
    for a given file the corresponding boolean flag is persisted in
    PostgreSQL so a restart will skip already-finished work.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize all sub-components from the application config.

        Args:
            config: Fully-resolved application configuration.
        """
        self.config = config
        self._interrupted = False

        # --- Storage ---
        self.pg = PostgresStorage(config.postgres.connection_string)
        self.pg.init_schema()

        self.qdrant = QdrantStorage(
            url=config.qdrant.url,
            collection_name=config.qdrant.collection_name,
            vector_size=config.qdrant.vector_size,
        )
        self.qdrant.ensure_collection()

        # --- Face processing ---
        self.face_detector = FaceDetector(
            model_pack=config.faces.model_pack,
            min_face_size=config.faces.min_face_size,
        )
        self.face_classifier = FaceClassifier(
            similarity_threshold=config.faces.similarity_threshold,
        )
        identities = self.pg.get_face_identities()
        self.face_classifier.load_identities(identities)

        # --- Caption / embed ---
        self.captioner: BaseCaptioner = create_captioner(config)
        logger.info(
            "Captioner provider: %s", config.captioner.provider
        )
        self.embedder = TextEmbedder(
            base_url=config.ollama.base_url,
            model=config.ollama.embedding_model,
        )

        logger.info("IndexingPipeline initialised")

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def scan_photos(self) -> list[str]:
        """Recursively discover photo files under the configured source dir.

        Files are matched by extension (case-insensitive).  Files whose
        extension appears in ``skip_extensions`` are silently ignored.

        Returns:
            A sorted list of absolute file paths.
        """
        source_dir = self.config.photos.source_dir
        supported = {ext.lower() for ext in self.config.photos.supported_extensions}
        skipped = {ext.lower() for ext in self.config.photos.skip_extensions}

        found: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(source_dir):
            for fname in filenames:
                if fname.startswith("._"):
                    continue  # macOS resource fork files
                ext = os.path.splitext(fname)[1].lower()
                if ext in skipped:
                    continue
                if ext in supported:
                    found.append(os.path.join(dirpath, fname))

        found.sort()
        logger.info("Scanned %d photos in %s", len(found), source_dir)
        return found

    # ------------------------------------------------------------------
    # Pending / resume logic
    # ------------------------------------------------------------------

    def get_pending_files(
        self,
        all_files: list[str],
        stage: str | None = None,
    ) -> list[str]:
        """Determine which files still need processing.

        Args:
            all_files: Complete list of discovered file paths.
            stage: If provided, only return files where this specific stage
                is ``False``.  When ``None`` the check uses ``embedded``
                (the terminal stage).

        Returns:
            A list of file paths that require further processing.
        """
        status_field = _STAGE_TO_STATUS_FIELD.get(stage, "embedded") if stage else "embedded"

        pending: list[str] = []
        for fp in all_files:
            status = self.pg.get_indexing_status(fp)
            if status is None:
                # Brand-new file — not yet tracked at all.
                pending.append(fp)
            elif not getattr(status, status_field, False):
                pending.append(fp)

        logger.info(
            "Found %d pending files (stage=%s) out of %d total",
            len(pending),
            stage or "all",
            len(all_files),
        )
        return pending

    # ------------------------------------------------------------------
    # Single-file processing
    # ------------------------------------------------------------------

    def process_photo(
        self,
        file_path: str,
        stages: set[str] | None = None,
    ) -> IndexingStatus:
        """Process a single photo through all (or specified) pipeline stages.

        Processing is **stage-gated**: a stage is skipped when its
        corresponding ``IndexingStatus`` flag is already ``True`` *and* the
        stage is not explicitly requested via *stages*.

        On error within any stage the exception message is recorded in
        ``status.error`` and the remaining stages are skipped — but the
        exception is **not** re-raised so the pipeline can continue with
        the next file.

        Args:
            file_path: Absolute path to the image file.
            stages: Optional subset of stages to run.  ``None`` means run
                all stages.

        Returns:
            The updated :class:`IndexingStatus` for this file.
        """
        active_stages = stages if stages is not None else ALL_STAGES

        # Retrieve or bootstrap status tracking.
        status = self.pg.get_indexing_status(file_path)
        if status is None:
            status = IndexingStatus(file_path=file_path)

        # Accumulate intermediate results for the final IndexedPhoto.
        metadata: Optional[PhotoMetadata] = None
        faces: list = []
        caption_obj: Optional[PhotoCaption] = None
        location_name: Optional[str] = None
        text_embedding: Optional[list[float]] = None

        def _needs(stage: str) -> bool:
            """Return True if *stage* should run."""
            if stage not in active_stages:
                return False
            field = _STAGE_TO_STATUS_FIELD[stage]
            return not getattr(status, field, False)

        try:
            # --- Stage 1: EXIF extraction ---
            if _needs("exif"):
                try:
                    metadata = extract_metadata(file_path)
                    status.exif_extracted = True
                    status.error = None
                    logger.debug("EXIF extracted: %s", file_path)
                except Exception as exc:
                    status.error = f"exif: {exc}"
                    logger.warning("EXIF extraction failed for %s: %s", file_path, exc)
                    status.last_updated = datetime.now(timezone.utc)
                    self.pg.upsert_indexing_status(status)
                    return status

            # --- Stage 2: Face detection + classification ---
            if _needs("faces"):
                try:
                    detected = self.face_detector.detect_faces(file_path)
                    faces = self.face_classifier.classify_faces(detected)
                    status.faces_extracted = True
                    status.faces_classified = True
                    status.error = None
                    logger.debug(
                        "Faces detected=%d classified=%d: %s",
                        len(detected),
                        len(faces),
                        file_path,
                    )
                except Exception as exc:
                    status.error = f"faces: {exc}"
                    logger.warning("Face processing failed for %s: %s", file_path, exc)
                    status.last_updated = datetime.now(timezone.utc)
                    self.pg.upsert_indexing_status(status)
                    return status

            # --- Stage 3: VLM captioning ---
            if _needs("caption"):
                try:
                    caption_obj = self.captioner.caption_photo(file_path)
                    status.captioned = True
                    status.error = None
                    logger.debug("Captioned: %s", file_path)
                except Exception as exc:
                    status.error = f"caption: {exc}"
                    logger.warning("Captioning failed for %s: %s", file_path, exc)
                    status.last_updated = datetime.now(timezone.utc)
                    self.pg.upsert_indexing_status(status)
                    return status

            # --- Stage 4: Geocode + embed ---
            if _needs("embed"):
                try:
                    # We may need metadata for geocoding / search text
                    # even if exif stage was already done in a prior run.
                    if metadata is None:
                        metadata = extract_metadata(file_path)

                    # Reverse geocode if GPS data is available.
                    if (
                        self.config.geocoding.enabled
                        and metadata.gps_lat is not None
                        and metadata.gps_lon is not None
                    ):
                        location_name = reverse_geocode(
                            metadata.gps_lat, metadata.gps_lon
                        )

                    # Retrieve caption text from the current run or from
                    # the previously-stored record.
                    caption_text: Optional[str] = None
                    if caption_obj is not None:
                        caption_text = caption_obj.caption
                    else:
                        existing = self.pg.get_photo(file_path)
                        if existing is not None:
                            caption_text = existing.get("caption")

                    # Gather face labels for the search text.
                    face_labels: list[str] = []
                    if faces:
                        face_labels = [
                            f.label for f in faces if f.label != "unknown"
                        ]
                    else:
                        existing = self.pg.get_photo(file_path)
                        if existing and existing.get("faces"):
                            face_labels = [
                                fl for fl in existing["faces"]
                                if fl != "unknown"
                            ]

                    search_text, embedding_vec = self.embedder.embed_photo(
                        caption=caption_text,
                        face_labels=face_labels,
                        location=location_name,
                        date_taken=metadata.date_taken,
                        camera=metadata.camera,
                    )
                    text_embedding = embedding_vec
                    status.embedded = True
                    status.error = None
                    logger.debug("Embedded: %s", file_path)
                except Exception as exc:
                    status.error = f"embed: {exc}"
                    logger.warning("Embedding failed for %s: %s", file_path, exc)
                    status.last_updated = datetime.now(timezone.utc)
                    self.pg.upsert_indexing_status(status)
                    return status

            # --- Persist the composite record ---
            if metadata is None:
                # Ensure we have metadata even if exif stage was skipped.
                try:
                    metadata = extract_metadata(file_path)
                except Exception:
                    metadata = PhotoMetadata(
                        file_path=file_path,
                        file_name=os.path.basename(file_path),
                        file_size_bytes=os.path.getsize(file_path),
                        file_type=os.path.splitext(file_path)[1].lstrip(".").upper(),
                    )

            indexed = IndexedPhoto(
                metadata=metadata,
                faces=faces if faces else [],
                caption=caption_obj,
                location_name=location_name,
                text_embedding=text_embedding,
            )

            self.pg.upsert_photo(indexed)
            if faces:
                self.pg.save_photo_faces(file_path, faces)
            if text_embedding is not None:
                self.qdrant.upsert_photo(indexed)

        except Exception as exc:
            # Catch-all for unexpected errors during persistence.
            status.error = f"unexpected: {exc}"
            logger.exception("Unexpected error processing %s", file_path)

        status.last_updated = datetime.now(timezone.utc)
        self.pg.upsert_indexing_status(status)
        return status

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        dry_run: bool = False,
        stages: set[str] | None = None,
        file_filter: str | None = None,
        errors_only: bool = False,
    ) -> dict:
        """Execute the indexing pipeline over all pending photos.

        Args:
            dry_run: When ``True`` scan and report counts without processing.
            stages: Restrict processing to these stages (``None`` = all).
            file_filter: Only process files whose path contains this
                substring.
            errors_only: Re-process only files that previously errored.

        Returns:
            A summary dict with keys ``processed``, ``succeeded``,
            ``failed``, ``skipped``, ``total_scanned``, and
            ``time_elapsed``.
        """
        # Register a graceful Ctrl+C handler.
        self._interrupted = False
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum: int, frame: object) -> None:
            console.print(
                "\n[yellow]Interrupt received — finishing current file "
                "and saving status...[/yellow]"
            )
            self._interrupted = True

        signal.signal(signal.SIGINT, _handle_sigint)

        try:
            return self._run_inner(dry_run, stages, file_filter, errors_only)
        finally:
            signal.signal(signal.SIGINT, prev_handler)

    def _run_inner(
        self,
        dry_run: bool,
        stages: set[str] | None,
        file_filter: str | None,
        errors_only: bool,
    ) -> dict:
        """Core run logic (separated for clean signal handling)."""
        start_time = time.monotonic()

        # 1. Discover files.
        all_files = self.scan_photos()

        # 2. Determine the relevant stage for filtering pending work.
        #    Use the *last* requested stage (in pipeline order) so that
        #    multi-stage runs like --faces-only (exif + faces) check whether
        #    the final stage of that group is already done.
        _STAGE_ORDER = ["exif", "faces", "caption", "embed"]
        filter_stage: str | None = None
        if stages:
            for s in reversed(_STAGE_ORDER):
                if s in stages:
                    filter_stage = s
                    break

        # 3. Get pending files.
        if errors_only:
            error_statuses = self.pg.get_files_with_errors()
            pending = [s.file_path for s in error_statuses]
            # Clear status so stages re-run.
            for s in error_statuses:
                self.pg.clear_indexing_status(s.file_path)
            console.print(
                f"[yellow]Re-processing {len(pending)} file(s) with errors[/yellow]"
            )
        else:
            pending = self.get_pending_files(all_files, stage=filter_stage)

        # 4. Apply optional substring filter.
        if file_filter:
            pending = [fp for fp in pending if file_filter in fp]

        total = len(pending)

        # 5. Dry-run: just report.
        if dry_run:
            elapsed = time.monotonic() - start_time
            stats = {
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped": total,
                "total_scanned": len(all_files),
                "time_elapsed": round(elapsed, 2),
            }
            console.print(
                Panel(
                    f"[bold]Dry-run summary[/bold]\n\n"
                    f"  Total photos scanned:  {len(all_files)}\n"
                    f"  Pending processing:    {total}\n"
                    f"  Stages:                {', '.join(sorted(stages)) if stages else 'all'}\n"
                    f"  Scan time:             {elapsed:.1f}s",
                    title="photo-search",
                    border_style="blue",
                )
            )
            return stats

        if total == 0:
            console.print("[green]Nothing to process — all files are up to date.[/green]")
            return {
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
                "total_scanned": len(all_files),
                "time_elapsed": round(time.monotonic() - start_time, 2),
            }

        # 6. Process with progress bar.
        processed = 0
        succeeded = 0
        failed = 0
        batch_log_interval = self.config.pipeline.batch_log_interval
        concurrency = max(1, self.config.pipeline.concurrency)

        if concurrency > 1:
            logger.info("Running pipeline with concurrency=%d", concurrency)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("Indexing photos", total=total)

            def _log_progress() -> None:
                """Emit the periodic batch-progress log line."""
                if not batch_log_interval or processed % batch_log_interval != 0:
                    return
                elapsed = time.monotonic() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f files/min) — %d ok, %d errors",
                    processed,
                    total,
                    rate * 60,
                    succeeded,
                    failed,
                )

            if concurrency <= 1:
                # --- Sequential path (unchanged behaviour) ---
                for file_path in pending:
                    if self._interrupted:
                        console.print(
                            f"[yellow]Stopped after {processed} files "
                            f"(interrupted)[/yellow]"
                        )
                        break

                    fname = os.path.basename(file_path)
                    progress.update(task_id, description=f"[cyan]{fname}[/cyan]")

                    status = self.process_photo(file_path, stages=stages)
                    processed += 1
                    if status.error:
                        failed += 1
                    else:
                        succeeded += 1
                    progress.advance(task_id)
                    _log_progress()
            else:
                # --- Parallel path (ThreadPoolExecutor) ---
                with ThreadPoolExecutor(
                    max_workers=concurrency,
                    thread_name_prefix="photo-index",
                ) as executor:
                    futures = {
                        executor.submit(
                            self.process_photo, fp, stages
                        ): fp
                        for fp in pending
                    }

                    try:
                        for future in as_completed(futures):
                            file_path = futures[future]
                            fname = os.path.basename(file_path)
                            try:
                                status = future.result()
                            except Exception as exc:
                                # process_photo already traps per-stage
                                # errors; this is only for unexpected crashes.
                                logger.exception(
                                    "Worker crashed on %s", file_path
                                )
                                status = IndexingStatus(
                                    file_path=file_path,
                                    error=f"worker: {exc}",
                                )

                            processed += 1
                            if status.error:
                                failed += 1
                            else:
                                succeeded += 1

                            progress.update(
                                task_id, description=f"[cyan]{fname}[/cyan]"
                            )
                            progress.advance(task_id)
                            _log_progress()

                            if self._interrupted:
                                console.print(
                                    f"[yellow]Stopped after {processed} "
                                    f"files (interrupted) — cancelling "
                                    f"remaining workers...[/yellow]"
                                )
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                break
                    finally:
                        # Ensure lingering futures are drained before we leave
                        # the executor context, so the progress bar doesn't
                        # lie about the final state on Ctrl+C.
                        for f in futures:
                            if not f.done():
                                f.cancel()

        elapsed = time.monotonic() - start_time
        rate = processed / elapsed if elapsed > 0 else 0

        stats = {
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": total - processed,
            "total_scanned": len(all_files),
            "time_elapsed": round(elapsed, 2),
        }

        # Summary panel.
        console.print(
            Panel(
                f"[bold]Indexing complete[/bold]\n\n"
                f"  Processed:   {processed}\n"
                f"  Succeeded:   {succeeded}\n"
                f"  Failed:      {failed}\n"
                f"  Skipped:     {total - processed}\n"
                f"  Throughput:  {rate * 60:.1f} files/min\n"
                f"  Elapsed:     {elapsed:.1f}s",
                title="photo-search",
                border_style="green" if failed == 0 else "yellow",
            )
        )

        return stats

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release storage connections."""
        try:
            self.pg.close()
        except Exception:
            logger.debug("Error closing Postgres connection", exc_info=True)
        logger.info("Pipeline resources released")
