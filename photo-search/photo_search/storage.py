"""Storage backends for photo-search: PostgreSQL metadata and Qdrant vectors.

PostgresStorage manages relational data (photo metadata, face identities,
indexing status) while QdrantStorage handles vector similarity search over
text embeddings of photo captions.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Direction,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

def _sanitize(val: Any) -> Any:
    """Strip NUL bytes from strings before sending to Postgres."""
    if isinstance(val, str):
        return val.replace("\x00", "")
    return val


from photo_search.geo import split_location_name
from photo_search.models import (
    IdentifiedFace,
    IndexedPhoto,
    IndexingStatus,
    SearchResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INIT_SQL_PATH = Path(__file__).resolve().parent.parent / "scripts" / "init_db.sql"


def _file_path_to_point_id(file_path: str) -> int:
    """Derive a deterministic uint64 Qdrant point ID from a file path.

    Uses the first 16 hex characters of an MD5 hash interpreted as an
    unsigned 64-bit integer.  This is *not* used for security -- only for
    stable, collision-resistant ID generation.
    """
    return int(hashlib.md5(file_path.encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# PostgresStorage
# ---------------------------------------------------------------------------


class PostgresStorage:
    """Thin wrapper around psycopg2 for photo-search relational data.

    Connections are managed per-thread using :class:`threading.local`.
    Each thread lazily opens its own connection on first use -- psycopg2
    connections are not safe to share across threads -- and every opened
    connection is tracked so :meth:`close` can release them all.

    In single-threaded use (the default) there is effectively one
    connection, matching the previous behaviour.
    """

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._local = threading.local()
        self._all_conns: list[psycopg2.extensions.connection] = []
        self._lock = threading.Lock()

    # -- connection management ------------------------------------------------

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Return the calling thread's connection, creating one if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            logger.debug("Opening new Postgres connection (thread-local)")
            conn = psycopg2.connect(self._connection_string)
            conn.autocommit = False
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return conn

    def close(self) -> None:
        """Close every connection opened by any thread."""
        with self._lock:
            conns = self._all_conns
            self._all_conns = []
        for conn in conns:
            try:
                if not conn.closed:
                    conn.close()
            except Exception:
                logger.debug("Error closing Postgres connection", exc_info=True)
        # Clear current-thread reference (other threads' locals are GC'd
        # when their threads exit).
        if hasattr(self._local, "conn"):
            self._local.conn = None
        logger.debug("Postgres connection(s) closed")

    def reconnect(self) -> None:
        """Close the current thread's connection so the next call gets a fresh one.

        Call this before DB operations that follow a long idle period (e.g.
        after an interactive prompt) to avoid using a stale connection.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                if not conn.closed:
                    conn.close()
            except Exception:
                pass
            self._local.conn = None

    # -- schema ---------------------------------------------------------------

    def init_schema(self) -> None:
        """Execute scripts/init_db.sql to create tables and indexes.

        Raises:
            FileNotFoundError: If the SQL file cannot be located.
            psycopg2.Error: On any database error (transaction is rolled back).
        """
        if not _INIT_SQL_PATH.is_file():
            raise FileNotFoundError(f"Schema SQL not found at {_INIT_SQL_PATH}")

        sql = _INIT_SQL_PATH.read_text()
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Database schema initialized from %s", _INIT_SQL_PATH)
        except psycopg2.Error:
            conn.rollback()
            raise

    # -- photos ---------------------------------------------------------------

    def upsert_photo(self, photo: IndexedPhoto) -> None:
        """Insert or update a photo record in the photos table.

        On conflict (duplicate file_path) the row is updated with the new
        values.  Also persists associated faces via :meth:`save_photo_faces`.
        """
        meta = photo.metadata
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO photos (
                        file_path, file_name, caption, date_taken,
                        gps_lat, gps_lon, location_name, camera,
                        file_type, file_size_bytes, width, height,
                        caption_model, embedding_model, indexed_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, NOW()
                    )
                    ON CONFLICT (file_path) DO UPDATE SET
                        file_name       = EXCLUDED.file_name,
                        -- Preserve existing caption when the upsert is a
                        -- resumed embed-only run that doesn't carry caption
                        -- text.  Otherwise captions get nulled on every
                        -- subsequent pipeline pass.
                        caption         = COALESCE(EXCLUDED.caption, photos.caption),
                        date_taken      = EXCLUDED.date_taken,
                        gps_lat         = EXCLUDED.gps_lat,
                        gps_lon         = EXCLUDED.gps_lon,
                        location_name   = EXCLUDED.location_name,
                        camera          = EXCLUDED.camera,
                        file_type       = EXCLUDED.file_type,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        width           = EXCLUDED.width,
                        height          = EXCLUDED.height,
                        caption_model   = COALESCE(EXCLUDED.caption_model, photos.caption_model),
                        embedding_model = COALESCE(EXCLUDED.embedding_model, photos.embedding_model),
                        indexed_at      = NOW()
                    """,
                    tuple(_sanitize(v) for v in (
                        meta.file_path,
                        meta.file_name,
                        photo.caption.caption if photo.caption else None,
                        meta.date_taken,
                        meta.gps_lat,
                        meta.gps_lon,
                        photo.location_name,
                        meta.camera,
                        meta.file_type,
                        meta.file_size_bytes,
                        meta.width,
                        meta.height,
                        photo.caption.model if photo.caption else None,
                        "nomic-embed-text" if photo.text_embedding else None,
                    )),
                )
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            raise

        # Persist faces in the junction table
        if photo.faces:
            self.save_photo_faces(meta.file_path, photo.faces)

    def get_photo(self, file_path: str) -> Optional[dict[str, Any]]:
        """Fetch a single photo record as a dictionary, or None."""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM photos WHERE file_path = %s", (file_path,))
            row = cur.fetchone()
        return dict(row) if row else None

    # -- indexing status ------------------------------------------------------

    def upsert_indexing_status(self, status: IndexingStatus) -> None:
        """Insert or update a row in the indexing_status table."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO indexing_status (
                        file_path, exif_extracted, faces_extracted,
                        faces_classified, captioned, embedded,
                        error, last_updated
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (file_path) DO UPDATE SET
                        exif_extracted   = EXCLUDED.exif_extracted,
                        faces_extracted  = EXCLUDED.faces_extracted,
                        faces_classified = EXCLUDED.faces_classified,
                        captioned        = EXCLUDED.captioned,
                        embedded         = EXCLUDED.embedded,
                        error            = EXCLUDED.error,
                        last_updated     = NOW()
                    """,
                    (
                        status.file_path,
                        status.exif_extracted,
                        status.faces_extracted,
                        status.faces_classified,
                        status.captioned,
                        status.embedded,
                        status.error,
                    ),
                )
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            raise

    def get_indexing_status(self, file_path: str) -> Optional[IndexingStatus]:
        """Fetch the indexing status for a single file, or None."""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM indexing_status WHERE file_path = %s", (file_path,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return IndexingStatus(**row)

    def get_incomplete_files(self) -> list[IndexingStatus]:
        """Return all files that have not yet been fully embedded."""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM indexing_status WHERE embedded = FALSE ORDER BY file_path"
            )
            rows = cur.fetchall()
        return [IndexingStatus(**row) for row in rows]

    def get_all_statuses(self) -> dict[str, int]:
        """Return aggregate counts for each pipeline stage.

        Returns a dict like::

            {
                "total": 5000,
                "exif_extracted": 4800,
                "faces_extracted": 4500,
                "faces_classified": 4500,
                "captioned": 4200,
                "embedded": 4000,
                "errors": 12,
            }
        """
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                    AS total,
                    COUNT(*) FILTER (WHERE exif_extracted)      AS exif_extracted,
                    COUNT(*) FILTER (WHERE faces_extracted)     AS faces_extracted,
                    COUNT(*) FILTER (WHERE faces_classified)    AS faces_classified,
                    COUNT(*) FILTER (WHERE captioned)           AS captioned,
                    COUNT(*) FILTER (WHERE embedded)            AS embedded,
                    COUNT(*) FILTER (WHERE error IS NOT NULL)   AS errors
                FROM indexing_status
                """
            )
            row = cur.fetchone()
        if row is None:
            return {
                "total": 0,
                "exif_extracted": 0,
                "faces_extracted": 0,
                "faces_classified": 0,
                "captioned": 0,
                "embedded": 0,
                "errors": 0,
            }
        return {
            "total": row[0],
            "exif_extracted": row[1],
            "faces_extracted": row[2],
            "faces_classified": row[3],
            "captioned": row[4],
            "embedded": row[5],
            "errors": row[6],
        }

    def get_files_with_errors(self) -> list[IndexingStatus]:
        """Return all files that have a non-null error recorded."""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM indexing_status WHERE error IS NOT NULL ORDER BY file_path"
            )
            rows = cur.fetchall()
        return [IndexingStatus(**row) for row in rows]

    def clear_indexing_status(self, file_path: str) -> None:
        """Reset a file's indexing status so it can be re-processed.

        All boolean flags are set back to False and the error is cleared.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE indexing_status SET
                        exif_extracted   = FALSE,
                        faces_extracted  = FALSE,
                        faces_classified = FALSE,
                        captioned        = FALSE,
                        embedded         = FALSE,
                        error            = NULL,
                        last_updated     = NOW()
                    WHERE file_path = %s
                    """,
                    (file_path,),
                )
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            raise

    # -- face identities ------------------------------------------------------

    def save_face_identity(
        self,
        label: str,
        display_name: str,
        centroid: np.ndarray,
        sample_count: int,
    ) -> None:
        """Upsert a face identity, merging with existing samples if present.

        When the label already exists the new centroid is merged with the
        stored one using a weighted average (weighted by sample counts) so
        that running ``label-faces`` multiple times accumulates diversity
        instead of overwriting.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT centroid_embedding, sample_count "
                    "FROM face_identities WHERE label = %s",
                    (label,),
                )
                row = cur.fetchone()

                if row is not None and row[0] is not None:
                    old_centroid = np.frombuffer(
                        bytes(row[0]), dtype=np.float32
                    )
                    old_count = row[1] or 0
                    total = old_count + sample_count
                    merged = (
                        old_centroid * old_count + centroid * sample_count
                    ) / total
                    norm = np.linalg.norm(merged)
                    if norm > 0:
                        merged = merged / norm
                    final_centroid = merged.astype(np.float32)
                    final_count = total
                else:
                    final_centroid = centroid.astype(np.float32)
                    final_count = sample_count

                centroid_bytes = final_centroid.tobytes()
                cur.execute(
                    """
                    INSERT INTO face_identities (
                        label, display_name, centroid_embedding,
                        sample_count, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (label) DO UPDATE SET
                        display_name       = EXCLUDED.display_name,
                        centroid_embedding = EXCLUDED.centroid_embedding,
                        sample_count       = EXCLUDED.sample_count,
                        updated_at         = NOW()
                    """,
                    (label, display_name, psycopg2.Binary(centroid_bytes),
                     final_count),
                )
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            raise

    def get_face_identities(self) -> list[dict[str, Any]]:
        """Fetch all face identities with centroids decoded as numpy arrays.

        Returns a list of dicts with keys: label, display_name,
        centroid_embedding (np.ndarray | None), sample_count.
        """
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT label, display_name, centroid_embedding, sample_count "
                "FROM face_identities ORDER BY label"
            )
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            centroid: Optional[np.ndarray] = None
            raw = row.get("centroid_embedding")
            if raw is not None:
                centroid = np.frombuffer(bytes(raw), dtype=np.float32)
            results.append(
                {
                    "label": row["label"],
                    "display_name": row["display_name"],
                    "centroid_embedding": centroid,
                    "sample_count": row["sample_count"],
                }
            )
        return results

    # -- locations ------------------------------------------------------------

    def list_locations(
        self, prefix: str = "", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Suggest indexed location names for autocomplete.

        Case-insensitive substring match against the raw ``location_name``
        column (e.g. ``"Woodcrest, California, US"``).  Groups duplicates
        and returns the most common locations first.

        Returns a list of dicts with keys: ``location_name``, ``photo_count``.
        """
        pattern = f"%{prefix.strip()}%"
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT location_name, COUNT(*) AS photo_count
                FROM photos
                WHERE location_name IS NOT NULL
                  AND location_name <> ''
                  AND location_name ILIKE %s
                GROUP BY location_name
                ORDER BY photo_count DESC, location_name ASC
                LIMIT %s
                """,
                (pattern, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "location_name": r["location_name"],
                "photo_count": r["photo_count"],
            }
            for r in rows
        ]

    # -- photo faces ----------------------------------------------------------

    def save_photo_faces(
        self, file_path: str, faces: list[IdentifiedFace]
    ) -> None:
        """Replace all face records for a photo with the given list.

        Existing face rows for the file are deleted first so that
        re-indexing produces a clean slate.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM photo_faces WHERE photo_file_path = %s",
                    (file_path,),
                )
                for face in faces:
                    embedding_bytes = np.array(
                        face.embedding, dtype=np.float32
                    ).tobytes()
                    cur.execute(
                        """
                        INSERT INTO photo_faces (
                            photo_file_path, face_label, confidence,
                            similarity, bbox_x, bbox_y, bbox_w, bbox_h,
                            embedding, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """,
                        (
                            file_path,
                            face.label,
                            face.confidence,
                            face.similarity,
                            face.bbox[0],
                            face.bbox[1],
                            face.bbox[2],
                            face.bbox[3],
                            psycopg2.Binary(embedding_bytes),
                        ),
                    )
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            raise

    def get_photo_faces(self, file_path: str) -> list[IdentifiedFace]:
        """Load all identified faces for a photo, reconstructed from DB rows.

        Returns an empty list if the photo has no recorded faces.  Used by
        the pipeline's embed stage to repopulate ``IndexedPhoto.faces`` on
        resumed runs (where the faces stage was skipped) so that the Qdrant
        payload carries the correct ``faces`` array.
        """
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT face_label, confidence, similarity,
                       bbox_x, bbox_y, bbox_w, bbox_h, embedding
                FROM photo_faces
                WHERE photo_file_path = %s
                """,
                (file_path,),
            )
            rows = cur.fetchall()

        out: list[IdentifiedFace] = []
        for row in rows:
            emb_bytes = row["embedding"]
            if emb_bytes is not None:
                embedding = np.frombuffer(
                    bytes(emb_bytes), dtype=np.float32
                ).tolist()
            else:
                embedding = []
            out.append(
                IdentifiedFace(
                    bbox=(
                        row["bbox_x"] or 0.0,
                        row["bbox_y"] or 0.0,
                        row["bbox_w"] or 0.0,
                        row["bbox_h"] or 0.0,
                    ),
                    confidence=row["confidence"] or 0.0,
                    similarity=row["similarity"] or 0.0,
                    label=row["face_label"] or "unknown",
                    embedding=embedding,
                )
            )
        return out

    def get_unknown_faces(self, page_size: int = 500) -> list[dict[str, Any]]:
        """Fetch all faces labeled 'unknown' with their embeddings and metadata.

        Data is fetched in pages to avoid overwhelming kubectl port-forward
        with a single huge result set (~28 MB for 14K faces).

        Returns a list of dicts with keys: photo_file_path, bbox (tuple),
        confidence, embedding (np.ndarray).
        """
        conn = self._get_connection()
        results: list[dict[str, Any]] = []
        offset = 0

        while True:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT photo_file_path, bbox_x, bbox_y, bbox_w, bbox_h,
                           confidence, embedding
                    FROM photo_faces
                    WHERE face_label = 'unknown'
                    ORDER BY photo_file_path, bbox_x
                    LIMIT %s OFFSET %s
                    """,
                    (page_size, offset),
                )
                rows = cur.fetchall()
            conn.commit()

            if not rows:
                break

            for row in rows:
                embedding_bytes = row.get("embedding")
                if embedding_bytes is None:
                    continue
                embedding = np.frombuffer(bytes(embedding_bytes), dtype=np.float32)
                results.append(
                    {
                        "photo_file_path": row["photo_file_path"],
                        "bbox": (
                            row["bbox_x"],
                            row["bbox_y"],
                            row["bbox_w"],
                            row["bbox_h"],
                        ),
                        "confidence": row["confidence"],
                        "embedding": embedding,
                    }
                )

            offset += page_size

        return results

    def get_all_faces_paged(self, page_size: int = 500) -> list[dict[str, Any]]:
        """Fetch all face records with embeddings, paged for port-forward safety.

        Returns a list of dicts with keys: id, photo_file_path, face_label,
        confidence, embedding (np.ndarray).
        """
        conn = self._get_connection()
        results: list[dict[str, Any]] = []
        offset = 0

        while True:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, photo_file_path, face_label, confidence, embedding
                    FROM photo_faces
                    ORDER BY id
                    LIMIT %s OFFSET %s
                    """,
                    (page_size, offset),
                )
                rows = cur.fetchall()
            conn.commit()

            if not rows:
                break

            for row in rows:
                embedding_bytes = row.get("embedding")
                if embedding_bytes is None:
                    continue
                embedding = np.frombuffer(bytes(embedding_bytes), dtype=np.float32)
                results.append(
                    {
                        "id": row["id"],
                        "photo_file_path": row["photo_file_path"],
                        "face_label": row["face_label"],
                        "confidence": row["confidence"],
                        "embedding": embedding,
                    }
                )

            offset += page_size

        return results

    def batch_update_face_labels(
        self, updates: list[tuple[str, float, int]]
    ) -> None:
        """Batch-update face_label and similarity for face records by ID.

        Args:
            updates: List of (label, similarity, face_id) tuples.
        """
        if not updates:
            return
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    "UPDATE photo_faces SET face_label = %s, similarity = %s WHERE id = %s",
                    updates,
                    page_size=500,
                )
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            raise


# ---------------------------------------------------------------------------
# QdrantStorage
# ---------------------------------------------------------------------------


class QdrantStorage:
    """Vector storage backend using Qdrant for photo caption embeddings."""

    def __init__(self, url: str, collection_name: str, vector_size: int) -> None:
        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size
        # Parse URL to handle HTTPS ingress (port 443) vs direct access (port 6333).
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme == "https" and not parsed.port:
            # HTTPS ingress: connect to host:443, skip TLS verify for internal CAs.
            self._client = QdrantClient(
                host=parsed.hostname,
                port=443,
                https=True,
                prefer_grpc=False,
                verify=False,
                check_compatibility=False,
            )
        else:
            self._client = QdrantClient(url=url, prefer_grpc=False)

    # -- collection management ------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the Qdrant collection and payload indexes if needed.

        Uses cosine distance which is standard for normalized text embeddings.
        Also ensures keyword/integer payload indexes exist so that filter
        queries (e.g. ``match: faces == 'eva'`` or ``year == 2023``) actually
        match points.  Without an index on array fields Qdrant's ``MatchValue``
        silently returns zero hits.
        """
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (size=%d, cosine)",
                self._collection_name,
                self._vector_size,
            )
        else:
            logger.debug("Qdrant collection '%s' already exists", self._collection_name)

        # Ensure filter-supporting payload indexes.  ``create_payload_index``
        # is idempotent when the schema matches, so it's safe to call on
        # every startup.
        for field, schema in (
            ("faces", PayloadSchemaType.KEYWORD),
            ("year", PayloadSchemaType.INTEGER),
            ("file_type", PayloadSchemaType.KEYWORD),
            ("date_taken", PayloadSchemaType.DATETIME),
            ("city", PayloadSchemaType.KEYWORD),
            ("region", PayloadSchemaType.KEYWORD),
            ("country_code", PayloadSchemaType.KEYWORD),
        ):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=schema,
                )
                logger.info("Ensured payload index %s (%s)", field, schema)
            except Exception as exc:
                # Qdrant returns an error if the index already exists with a
                # matching schema; tolerate it.
                logger.debug("Payload index %s already present or skipped: %s",
                             field, exc)

    # -- CRUD -----------------------------------------------------------------

    def upsert_photo(self, photo: IndexedPhoto) -> None:
        """Upsert a photo's embedding and payload into Qdrant.

        The point ID is derived deterministically from the file path so that
        re-indexing the same photo overwrites the previous point.

        Raises:
            ValueError: If the photo has no text embedding.
        """
        if photo.text_embedding is None:
            raise ValueError(
                f"Cannot upsert photo without text_embedding: "
                f"{photo.metadata.file_path}"
            )

        point_id = _file_path_to_point_id(photo.metadata.file_path)
        meta = photo.metadata

        city, region, country_code = split_location_name(photo.location_name)

        # Build the payload dict that mirrors the Qdrant schema.
        payload: dict[str, Any] = {
            "file_path": meta.file_path,
            "file_name": meta.file_name,
            "caption": photo.caption.caption if photo.caption else None,
            "date_taken": meta.date_taken.isoformat() if meta.date_taken else None,
            "year": meta.date_taken.year if meta.date_taken else None,
            "gps_lat": meta.gps_lat,
            "gps_lon": meta.gps_lon,
            "location_name": photo.location_name,
            "city": city,
            "region": region,
            "country_code": country_code,
            "camera": meta.camera,
            "file_type": meta.file_type,
            "faces": [f.label for f in photo.faces if f.label != "unknown"],
            "width": meta.width,
            "height": meta.height,
        }

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=photo.text_embedding,
                    payload=payload,
                )
            ],
        )
        logger.debug("Upserted point %d for %s", point_id, meta.file_path)

    def delete_photo(self, file_path: str) -> None:
        """Delete a photo's point from Qdrant by its file path."""
        point_id = _file_path_to_point_id(file_path)
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=[point_id],
        )
        logger.debug("Deleted point %d for %s", point_id, file_path)

    def count(self) -> int:
        """Return the total number of points in the collection."""
        info = self._client.get_collection(self._collection_name)
        return info.points_count

    # -- search ---------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar photos using a text embedding vector.

        Args:
            query_vector: 768-dim embedding from nomic-embed-text.
            limit: Maximum number of results to return.
            filters: Optional filter dict supporting keys:
                - ``person`` (str or list[str]): match on the ``faces`` payload
                  field.  If a list is provided, ALL listed labels must be
                  present on the photo (AND semantics).
                - ``year`` (int): exact match on the ``year`` payload field.
                - ``date_from`` (str, ISO date): lower bound for date_taken.
                - ``date_to`` (str, ISO date): upper bound for date_taken.
                - ``city`` (str, lowercase): exact match on the ``city`` payload.
                - ``region`` (str, lowercase): exact match on the ``region``
                  payload (full name, e.g. ``"california"``).
                - ``country_code`` (str, ISO2 upper): exact match on
                  ``country_code`` (e.g. ``"IT"``).

        Returns:
            A list of :class:`SearchResult` ordered by descending score.
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
        ).points

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            date_taken = None
            if payload.get("date_taken"):
                try:
                    date_taken = datetime.fromisoformat(payload["date_taken"])
                except (ValueError, TypeError):
                    pass

            results.append(
                SearchResult(
                    file_path=payload.get("file_path", ""),
                    file_name=payload.get("file_name", ""),
                    score=hit.score,
                    caption=payload.get("caption"),
                    faces=payload.get("faces", []),
                    date_taken=date_taken,
                    location_name=payload.get("location_name"),
                    camera=payload.get("camera"),
                )
            )
        return results

    def browse(
        self,
        limit: int = 60,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """List photos matching *filters* without a text query.

        Uses Qdrant's scroll API and orders by ``date_taken`` descending so
        the most recent photos appear first.  Accepts the same filter keys
        as :meth:`search`.  Scores are set to ``0.0`` since there is no
        similarity ranking.
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            order_by=OrderBy(key="date_taken", direction=Direction.DESC),
        )

        results: list[SearchResult] = []
        for pt in points:
            payload = pt.payload or {}
            date_taken = None
            if payload.get("date_taken"):
                try:
                    date_taken = datetime.fromisoformat(payload["date_taken"])
                except (ValueError, TypeError):
                    pass

            results.append(
                SearchResult(
                    file_path=payload.get("file_path", ""),
                    file_name=payload.get("file_name", ""),
                    score=0.0,
                    caption=payload.get("caption"),
                    faces=payload.get("faces", []),
                    date_taken=date_taken,
                    location_name=payload.get("location_name"),
                    camera=payload.get("camera"),
                )
            )
        return results

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Translate a simple filter dict into a Qdrant Filter object."""
        must_conditions: list[FieldCondition] = []

        if "person" in filters:
            # Normalise to a list; each label becomes its own MatchValue so
            # that Qdrant ANDs them together (photo must contain ALL people).
            persons = filters["person"]
            if isinstance(persons, str):
                persons = [persons]
            for label in persons:
                must_conditions.append(
                    FieldCondition(
                        key="faces",
                        match=MatchValue(value=label),
                    )
                )

        if "year" in filters:
            must_conditions.append(
                FieldCondition(
                    key="year",
                    match=MatchValue(value=filters["year"]),
                )
            )

        # Location fields are exact-match, stored pre-normalised (city/region
        # lowercase, country_code uppercase).  Callers are expected to pass
        # values already in the stored shape.
        for key in ("city", "region", "country_code"):
            if key in filters and filters[key]:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=filters[key]),
                    )
                )

        date_range_kwargs: dict[str, Any] = {}
        if "date_from" in filters:
            date_range_kwargs["gte"] = filters["date_from"]
        if "date_to" in filters:
            date_range_kwargs["lte"] = filters["date_to"]
        if date_range_kwargs:
            must_conditions.append(
                FieldCondition(
                    key="date_taken",
                    range=Range(**date_range_kwargs),
                )
            )

        return Filter(must=must_conditions)
