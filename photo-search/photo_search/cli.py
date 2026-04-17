"""CLI interface for photo-search using Typer.

Provides commands for indexing photos, searching, managing face labels,
and inspecting pipeline status.  All output uses ``rich`` for readable,
colourful terminal formatting.

Entry-point (defined in pyproject.toml)::

    photo-search index --dry-run
    photo-search search "sunset at the beach"
    photo-search status --detailed
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from photo_search.config import load_config

app = typer.Typer(
    name="photo-search",
    help="AI-powered photo search system",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logging with sensible defaults."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# index
# ======================================================================

@app.command()
def index(
    faces_only: bool = typer.Option(
        False, "--faces-only", help="Only run face detection and classification"
    ),
    captions_only: bool = typer.Option(
        False, "--captions-only", help="Only run VLM captioning"
    ),
    embed_only: bool = typer.Option(
        False, "--embed-only", help="Only run embedding generation"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Scan and report counts without processing"
    ),
    file_filter: Optional[str] = typer.Option(
        None, "--filter", help="Only process files whose path contains this string"
    ),
    concurrency: Optional[int] = typer.Option(
        None,
        "--concurrency",
        "-j",
        help=(
            "Number of files to process in parallel. Overrides "
            "pipeline.concurrency in config.yaml. Use 1 for Ollama "
            "(GPU-bound) and 8-10 for Anthropic."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to config.yaml"
    ),
) -> None:
    """Run the photo indexing pipeline."""
    _setup_logging(verbose)

    # Determine stages from flags.
    stages: set[str] | None = None
    if faces_only:
        stages = {"exif", "faces"}
    elif captions_only:
        stages = {"caption"}
    elif embed_only:
        stages = {"embed"}

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # CLI --concurrency overrides config.
    if concurrency is not None:
        if concurrency < 1:
            console.print("[red]--concurrency must be >= 1[/red]")
            raise typer.Exit(code=1)
        config.pipeline.concurrency = concurrency

    from photo_search.pipeline import IndexingPipeline

    pipeline: IndexingPipeline | None = None
    try:
        pipeline = IndexingPipeline(config)
        stats = pipeline.run(
            dry_run=dry_run,
            stages=stages,
            file_filter=file_filter,
        )
        if stats.get("failed", 0) > 0:
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Pipeline error:[/red] {exc}")
        logger.exception("Pipeline error")
        raise typer.Exit(code=1) from exc
    finally:
        if pipeline is not None:
            pipeline.cleanup()


# ======================================================================
# label-faces
# ======================================================================

@app.command("label-faces")
def label_faces(
    label: str = typer.Option(..., prompt="Label for this person"),
    display_name: Optional[str] = typer.Option(
        None, "--display-name", help="Human-readable display name"
    ),
    samples: int = typer.Option(
        5, "--samples", help="Number of face samples to collect"
    ),
    photo_count: int = typer.Option(
        50, "--photo-count", help="Number of photos to scan for faces"
    ),
    seed_photo: Optional[str] = typer.Option(
        None, "--seed-photo",
        help="Bootstrap from a photo containing the target person. "
        "Shows similar unknown faces ranked by cosine similarity.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to config.yaml"
    ),
) -> None:
    """Interactively label faces to seed known identities."""
    _setup_logging(verbose)

    if display_name is None:
        display_name = label.title()

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    from photo_search.faces import FaceDetector, compute_centroid, crop_face
    from photo_search.storage import PostgresStorage

    pg = PostgresStorage(config.postgres.connection_string)

    try:
        pg.init_schema()
        detector = FaceDetector(
            model_pack=config.faces.model_pack,
            min_face_size=config.faces.min_face_size,
        )

        collected_embeddings: list[list[float]] = []

        # ===================================================================
        # Seed-photo workflow: bootstrap from a known photo
        # ===================================================================
        if seed_photo is not None:
            seed_path = Path(seed_photo)
            if not seed_path.is_file():
                console.print(f"[red]Seed photo not found: {seed_photo}[/red]")
                raise typer.Exit(code=1)

            console.print(f"\n[bold]Detecting faces in seed photo:[/bold] {seed_photo}")
            seed_faces = detector.detect_faces(str(seed_path))

            if not seed_faces:
                console.print("[red]No faces detected in seed photo.[/red]")
                raise typer.Exit(code=1)

            # If multiple faces, let user select which one is the target.
            seed_embedding: list[float]
            if len(seed_faces) == 1:
                seed_embedding = seed_faces[0].embedding
                console.print("[green]Found 1 face in seed photo.[/green]")
            else:
                console.print(
                    f"[yellow]Found {len(seed_faces)} faces in seed photo. "
                    f"Please select which one is {label}:[/yellow]\n"
                )
                for idx, face in enumerate(seed_faces, start=1):
                    cropped = crop_face(str(seed_path), face.bbox, padding=0.3)
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        cropped.save(tmp, format="PNG")
                        tmp_path = tmp.name
                    subprocess.run(["open", tmp_path], check=False)

                    while True:
                        answer = input(
                            f"  Is face #{idx} the target person {label}? (y/n): "
                        ).strip().lower()
                        if answer == "y":
                            seed_embedding = face.embedding
                            console.print(f"[green]Selected face #{idx} as seed.[/green]")
                            break
                        if answer == "n":
                            break
                        console.print("  Please enter y or n.")
                    if answer == "y":
                        break
                else:
                    console.print("[red]No face selected. Aborting.[/red]")
                    raise typer.Exit(code=1)

            # Fetch all unknown faces and compute similarities.
            console.print("\n[bold]Fetching unknown faces from database...[/bold]")
            unknown_faces = pg.get_unknown_faces()
            pg.reconnect()  # Close connection before interactive session
            if not unknown_faces:
                console.print(
                    "[yellow]No unknown faces found in database. "
                    "Run face detection first.[/yellow]"
                )
                raise typer.Exit(code=1)

            console.print(
                f"Found [bold]{len(unknown_faces)}[/bold] unknown faces. "
                "Computing similarities..."
            )

            # Compute cosine similarity between seed and all unknowns.
            from sklearn.metrics.pairwise import cosine_similarity

            seed_vec = np.array(seed_embedding, dtype=np.float32).reshape(1, -1)
            similarities: list[tuple[float, dict]] = []

            for unknown in unknown_faces:
                unknown_vec = unknown["embedding"].reshape(1, -1)
                sim = float(cosine_similarity(seed_vec, unknown_vec)[0, 0])
                similarities.append((sim, unknown))

            # Sort by similarity (descending).
            similarities.sort(key=lambda x: x[0], reverse=True)

            console.print(
                f"\n[bold]Presenting top matches for {label}:[/bold] "
                f"(target: {samples} samples)\n"
            )

            quit_requested = False
            for sim_score, face_data in similarities:
                if quit_requested or len(collected_embeddings) >= samples:
                    break

                photo_path = face_data["photo_file_path"]
                bbox = face_data["bbox"]

                # Show cropped face.
                cropped = crop_face(photo_path, bbox, padding=0.3)
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    cropped.save(tmp, format="PNG")
                    tmp_path = tmp.name
                subprocess.run(["open", tmp_path], check=False)

                answer = _prompt_yes_no_quit(
                    f"Is this {label}? (similarity: {sim_score:.3f}) "
                    f"(y/n/q) [{len(collected_embeddings)}/{samples}]: "
                )

                if answer == "y":
                    collected_embeddings.append(face_data["embedding"].tolist())
                    console.print(
                        f"  [green]Collected ({len(collected_embeddings)}/{samples})[/green]"
                    )
                elif answer == "q":
                    quit_requested = True

        # ===================================================================
        # Random sampling workflow (original behavior)
        # ===================================================================
        else:
            # Collect photos to scan.
            source_dir = Path(config.photos.source_dir)
            supported = {ext.lower() for ext in config.photos.supported_extensions}
            all_photos = [
                p for p in source_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in supported
            ]

            if not all_photos:
                console.print(f"[red]No photos found in {source_dir}[/red]")
                raise typer.Exit(code=1)

            sample_size = min(photo_count, len(all_photos))
            photos = random.sample(all_photos, sample_size)

            console.print(
                f"\nScanning [bold]{len(photos)}[/bold] photos for faces "
                f"to label as [cyan]'{label}'[/cyan]...\n"
            )

            quit_requested = False

            for photo in photos:
                if quit_requested or len(collected_embeddings) >= samples:
                    break

                faces = detector.detect_faces(str(photo))
                if not faces:
                    continue

                for face in faces:
                    if len(collected_embeddings) >= samples:
                        break

                    # Show cropped face via system viewer.
                    cropped = crop_face(str(photo), face.bbox, padding=0.3)
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        cropped.save(tmp, format="PNG")
                        tmp_path = tmp.name
                    subprocess.run(["open", tmp_path], check=False)

                    answer = _prompt_yes_no_quit(
                        f"Is this {label}? (y/n/q to quit) "
                        f"[{len(collected_embeddings)}/{samples} collected]: "
                    )

                    if answer == "y":
                        collected_embeddings.append(face.embedding)
                        console.print(
                            f"  [green]Collected "
                            f"({len(collected_embeddings)}/{samples})[/green]"
                        )
                    elif answer == "q":
                        quit_requested = True
                        break

        if not collected_embeddings:
            console.print("[red]No embeddings collected. Aborting.[/red]")
            raise typer.Exit(code=1)

        centroid = compute_centroid(collected_embeddings)
        pg.save_face_identity(
            label=label,
            display_name=display_name,
            centroid=centroid,
            sample_count=len(collected_embeddings),
        )

        new_count = len(collected_embeddings)
        console.print(
            f"\n[green]Saved identity [bold]'{label}'[/bold] "
            f"(+{new_count} sample(s), merged with any existing).[/green]"
        )
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error during face labeling:[/red] {exc}")
        logger.exception("Face labeling error")
        raise typer.Exit(code=1) from exc
    finally:
        pg.close()


def _prompt_yes_no_quit(prompt: str) -> str:
    """Prompt the user and return ``'y'``, ``'n'``, or ``'q'``."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "n", "q"):
            return answer
        console.print("  Please enter y, n, or q.")


# ======================================================================
# reclassify-faces
# ======================================================================

@app.command("reclassify-faces")
def reclassify_faces(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to config.yaml"
    ),
) -> None:
    """Re-classify all detected faces against current identities."""
    _setup_logging(verbose)

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    from photo_search.faces import FaceClassifier, FaceDetector
    from photo_search.storage import PostgresStorage

    pg = PostgresStorage(config.postgres.connection_string)
    try:
        pg.init_schema()

        identities = pg.get_face_identities()
        if not identities:
            console.print(
                "[yellow]No face identities found. Label some faces first.[/yellow]"
            )
            raise typer.Exit(code=0)

        classifier = FaceClassifier(
            similarity_threshold=config.faces.similarity_threshold,
        )
        classifier.load_identities(identities)

        detector = FaceDetector(
            model_pack=config.faces.model_pack,
            min_face_size=config.faces.min_face_size,
        )

        console.print(
            f"Loaded [bold]{len(identities)}[/bold] identity(ies). "
            "Re-classifying all detected faces..."
        )

        statuses = pg.get_all_statuses()
        console.print(
            f"  Total photos tracked:    {statuses.get('total', 0)}\n"
            f"  Faces extracted:         {statuses.get('faces_extracted', 0)}"
        )

        # Walk the source directory and reclassify every file that has
        # previously had faces extracted.
        reclassified = 0
        source_dir = config.photos.source_dir
        supported = {ext.lower() for ext in config.photos.supported_extensions}
        skipped = {ext.lower() for ext in config.photos.skip_extensions}

        for dirpath, _, filenames in os.walk(source_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in skipped or ext not in supported:
                    continue

                fp = os.path.join(dirpath, fname)
                file_status = pg.get_indexing_status(fp)
                if file_status is None or not file_status.faces_extracted:
                    continue

                detected = detector.detect_faces(fp)
                if not detected:
                    continue

                classified = classifier.classify_faces(detected)
                pg.save_photo_faces(fp, classified)

                file_status.faces_classified = True
                file_status.last_updated = datetime.now(tz=timezone.utc)
                pg.upsert_indexing_status(file_status)
                reclassified += 1

        console.print(
            f"\n[green]Re-classified faces in {reclassified} photo(s).[/green]"
        )
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error during reclassification:[/red] {exc}")
        logger.exception("Reclassification error")
        raise typer.Exit(code=1) from exc
    finally:
        pg.close()


# ======================================================================
# search
# ======================================================================

@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    person: Optional[str] = typer.Option(
        None, "--person", help="Filter by person label"
    ),
    year: Optional[int] = typer.Option(None, "--year", help="Filter by year"),
    after: Optional[str] = typer.Option(
        None, "--after", help="Filter after date (YYYY-MM-DD)"
    ),
    before: Optional[str] = typer.Option(
        None, "--before", help="Filter before date (YYYY-MM-DD)"
    ),
    top: int = typer.Option(10, "--top", help="Number of results"),
    open_result: bool = typer.Option(
        False, "--open", help="Open top result in Preview"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to config.yaml"
    ),
) -> None:
    """Search photos using natural language."""
    _setup_logging(verbose)

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    from photo_search.embed import TextEmbedder
    from photo_search.storage import QdrantStorage

    embedder: TextEmbedder | None = None
    qdrant: QdrantStorage | None = None

    try:
        embedder = TextEmbedder(
            base_url=config.ollama.base_url,
            model=config.ollama.embedding_model,
        )
        qdrant = QdrantStorage(
            url=config.qdrant.url,
            collection_name=config.qdrant.collection_name,
            vector_size=config.qdrant.vector_size,
        )

        console.print(f"Searching for: [cyan]{query}[/cyan]\n")

        query_vector = embedder.embed_text(query)

        # Build filters dict matching QdrantStorage._build_filter keys.
        filters: dict = {}
        if person:
            filters["person"] = person
        if year:
            filters["year"] = year
        if after:
            filters["date_from"] = after
        if before:
            filters["date_to"] = before

        results = qdrant.search(
            query_vector=query_vector,
            limit=top,
            filters=filters if filters else None,
        )

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            raise typer.Exit(code=0)

        # Build results table.
        table = Table(title=f"Search Results ({len(results)})", show_lines=True)
        table.add_column("Rank", style="bold", width=5, justify="right")
        table.add_column("Score", width=7, justify="right")
        table.add_column("File", style="cyan", max_width=30)
        table.add_column("Date", width=12)
        table.add_column("People", style="magenta", max_width=20)
        table.add_column("Caption", max_width=60)
        table.add_column("Location", max_width=20)

        for rank, result in enumerate(results, start=1):
            caption_text = result.caption or ""
            if len(caption_text) > 60:
                caption_text = caption_text[:57] + "..."

            date_str = ""
            if result.date_taken:
                date_str = result.date_taken.strftime("%Y-%m-%d")

            people_str = ", ".join(result.faces) if result.faces else ""

            table.add_row(
                str(rank),
                f"{result.score:.4f}",
                result.file_name,
                date_str,
                people_str,
                caption_text,
                result.location_name or "",
            )

        console.print(table)

        # Optionally open the top result.
        if open_result and results:
            top_path = results[0].file_path
            console.print(f"\nOpening: [cyan]{top_path}[/cyan]")
            subprocess.run(["open", top_path], check=False)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Search error:[/red] {exc}")
        logger.exception("Search error")
        raise typer.Exit(code=1) from exc


# ======================================================================
# status
# ======================================================================

@app.command()
def status(
    detailed: bool = typer.Option(
        False, "--detailed", help="Show per-file breakdown"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to config.yaml"
    ),
) -> None:
    """Show indexing pipeline status."""
    _setup_logging(verbose)

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    from photo_search.storage import PostgresStorage, QdrantStorage

    pg = PostgresStorage(config.postgres.connection_string)
    qdrant: QdrantStorage | None = None

    try:
        pg.init_schema()
        counts = pg.get_all_statuses()

        qdrant = QdrantStorage(
            url=config.qdrant.url,
            collection_name=config.qdrant.collection_name,
            vector_size=config.qdrant.vector_size,
        )
        qdrant_count = qdrant.count()

        total = counts.get("total", 0)
        exif = counts.get("exif_extracted", 0)
        faces_ex = counts.get("faces_extracted", 0)
        faces_cl = counts.get("faces_classified", 0)
        captioned = counts.get("captioned", 0)
        embedded = counts.get("embedded", 0)

        panel_text = (
            f"[bold]Indexing Status[/bold]\n\n"
            f"  Total files tracked:       {total}\n"
            f"  EXIF extracted:            {exif}\n"
            f"  Faces detected:            {faces_ex}\n"
            f"  Faces classified:          {faces_cl}\n"
            f"  Captioned:                 {captioned}\n"
            f"  Embedded:                  {embedded}\n"
            f"  Qdrant vectors:            {qdrant_count}\n"
            f"  Fully indexed:             {embedded}/{total}"
        )

        error_files = pg.get_files_with_errors()
        if error_files:
            panel_text += f"\n\n  [red]Files with errors:       {len(error_files)}[/red]"

        console.print(
            Panel(panel_text, title="photo-search status", border_style="blue")
        )

        if detailed:
            console.print()
            table = Table(title="Per-file Status", show_lines=False)
            table.add_column("File", style="cyan", max_width=50)
            table.add_column("EXIF", width=5, justify="center")
            table.add_column("Faces", width=6, justify="center")
            table.add_column("Caption", width=8, justify="center")
            table.add_column("Embed", width=6, justify="center")
            table.add_column("Error", style="red", max_width=40)

            incomplete = pg.get_incomplete_files()
            for s in incomplete:
                table.add_row(
                    os.path.basename(s.file_path),
                    _bool_icon(s.exif_extracted),
                    _bool_icon(s.faces_extracted),
                    _bool_icon(s.captioned),
                    _bool_icon(s.embedded),
                    _truncate(s.error, 40) if s.error else "",
                )

            if error_files:
                for s in error_files:
                    # Avoid duplicates with incomplete files.
                    incomplete_paths = {f.file_path for f in incomplete}
                    if s.file_path in incomplete_paths:
                        continue
                    table.add_row(
                        os.path.basename(s.file_path),
                        _bool_icon(s.exif_extracted),
                        _bool_icon(s.faces_extracted),
                        _bool_icon(s.captioned),
                        _bool_icon(s.embedded),
                        _truncate(s.error, 40) if s.error else "",
                    )

            console.print(table)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Status error:[/red] {exc}")
        logger.exception("Status error")
        raise typer.Exit(code=1) from exc
    finally:
        pg.close()


def _bool_icon(val: bool) -> str:
    """Return a check or cross for boolean display."""
    return "[green]Y[/green]" if val else "[dim]N[/dim]"


def _truncate(text: str | None, length: int) -> str:
    """Truncate text to *length* characters with ellipsis."""
    if text is None:
        return ""
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


# ======================================================================
# reindex
# ======================================================================

@app.command()
def reindex(
    file: Optional[str] = typer.Option(
        None, "--file", help="Specific file to re-index"
    ),
    all_errors: bool = typer.Option(
        False, "--all-errors", help="Re-process all errored files"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to config.yaml"
    ),
) -> None:
    """Re-index specific files or all errored files."""
    _setup_logging(verbose)

    if not file and not all_errors:
        console.print(
            "[yellow]Specify --file <path> or --all-errors to re-index.[/yellow]"
        )
        raise typer.Exit(code=1)

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    from photo_search.pipeline import IndexingPipeline

    pipeline: IndexingPipeline | None = None
    try:
        pipeline = IndexingPipeline(config)

        if file:
            if not os.path.isfile(file):
                console.print(f"[red]File not found:[/red] {file}")
                raise typer.Exit(code=1)
            pipeline.pg.clear_indexing_status(file)
            console.print(f"Re-indexing: [cyan]{file}[/cyan]")
            status = pipeline.process_photo(file)
            if status.error:
                console.print(f"[red]Error:[/red] {status.error}")
                raise typer.Exit(code=2)
            console.print("[green]Re-indexing complete.[/green]")
        elif all_errors:
            stats = pipeline.run(errors_only=True)
            if stats.get("failed", 0) > 0:
                raise typer.Exit(code=2)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Reindex error:[/red] {exc}")
        logger.exception("Reindex error")
        raise typer.Exit(code=1) from exc
    finally:
        if pipeline is not None:
            pipeline.cleanup()


# ======================================================================
# init-db
# ======================================================================

@app.command("init-db")
def init_db(
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to config.yaml"
    ),
) -> None:
    """Initialize Postgres tables and Qdrant collection."""
    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    from photo_search.storage import PostgresStorage, QdrantStorage

    try:
        pg = PostgresStorage(config.postgres.connection_string)
        pg.init_schema()
        console.print("[green]Postgres schema initialised.[/green]")

        qdrant = QdrantStorage(
            url=config.qdrant.url,
            collection_name=config.qdrant.collection_name,
            vector_size=config.qdrant.vector_size,
        )
        qdrant.ensure_collection()
        console.print("[green]Qdrant collection ensured.[/green]")

        pg.close()

        console.print(
            Panel(
                "[bold green]Database initialisation complete.[/bold green]\n\n"
                f"  Postgres: {config.postgres.connection_string}\n"
                f"  Qdrant:   {config.qdrant.url} / {config.qdrant.collection_name}",
                title="photo-search init-db",
                border_style="green",
            )
        )
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Database initialisation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ======================================================================
# Entry point for direct execution
# ======================================================================

if __name__ == "__main__":
    app()
