"""Re-write Qdrant point payloads from current Postgres state.

Every point's vector stays untouched; only the payload gets overwritten so
that ``faces`` (and ``caption``, ``location_name`` etc.) reflect the
up-to-date Postgres records.  This fixes collections whose payloads were
poisoned by the resumed-embed bug where every re-embed pass wrote
``faces: []`` and ``caption: null`` on top of previously-correct payloads.

Usage::

    python scripts/repair_qdrant_payloads.py          # commit
    python scripts/repair_qdrant_payloads.py --dry-run

This script does **not** recompute embeddings.  The vectors will still encode
whatever ``search_text`` was in place at embed time.  Run a full
``photo-search index --embed-only`` afterwards to also refresh the vectors
themselves.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
import psycopg2.extras

from photo_search.config import load_config
from photo_search.storage import QdrantStorage


BATCH_SIZE = 200


def _fetch_postgres_rows(conn_str: str) -> dict[str, dict[str, Any]]:
    """Build a dict: file_path -> {photos row fields + faces list}."""
    records: dict[str, dict[str, Any]] = {}

    conn = psycopg2.connect(conn_str)
    try:
        conn.set_session(readonly=True)
        with conn.cursor(
            name="repair_photos", cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.itersize = 1000
            cur.execute(
                """
                SELECT file_path, file_name, caption, date_taken, gps_lat,
                       gps_lon, location_name, camera, file_type, width, height
                FROM photos
                """
            )
            while True:
                batch = cur.fetchmany(1000)
                if not batch:
                    break
                for row in batch:
                    records[row["file_path"]] = dict(row)
                    records[row["file_path"]]["faces"] = []
        conn.commit()

        # Second pass: join face labels (excluding 'unknown').
        with conn.cursor(
            name="repair_faces", cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.itersize = 2000
            cur.execute(
                """
                SELECT photo_file_path, face_label
                FROM photo_faces
                WHERE face_label IS NOT NULL AND face_label <> 'unknown'
                """
            )
            while True:
                batch = cur.fetchmany(2000)
                if not batch:
                    break
                for row in batch:
                    rec = records.get(row["photo_file_path"])
                    if rec is None:
                        continue
                    # Use a set to dedupe repeats of the same label on one photo.
                    rec.setdefault("_faces_set", set()).add(row["face_label"])
        conn.commit()
    finally:
        conn.close()

    # Materialise face sets into sorted lists for stable payloads.
    for rec in records.values():
        s = rec.pop("_faces_set", None)
        rec["faces"] = sorted(s) if s else []

    return records


def _build_payload(rec: dict[str, Any]) -> dict[str, Any]:
    date_taken = rec.get("date_taken")
    return {
        "file_path": rec["file_path"],
        "file_name": rec["file_name"],
        "caption": rec.get("caption"),
        "date_taken": date_taken.isoformat() if date_taken else None,
        "year": date_taken.year if date_taken else None,
        "gps_lat": rec.get("gps_lat"),
        "gps_lon": rec.get("gps_lon"),
        "location_name": rec.get("location_name"),
        "camera": rec.get("camera"),
        "file_type": rec.get("file_type"),
        "faces": rec.get("faces", []),
        "width": rec.get("width"),
        "height": rec.get("height"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rewrite Qdrant payloads from Postgres.")
    p.add_argument("--dry-run", action="store_true", help="Don't write to Qdrant")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    args = p.parse_args(argv)

    config = load_config(args.config)

    print("Connecting to Postgres + pulling photos/face data...", file=sys.stderr)
    records = _fetch_postgres_rows(config.postgres.connection_string)
    print(f"  {len(records)} photo rows loaded", file=sys.stderr)

    with_faces = sum(1 for r in records.values() if r["faces"])
    with_cap = sum(1 for r in records.values() if r.get("caption"))
    print(
        f"  {with_faces} have face labels, {with_cap} have captions",
        file=sys.stderr,
    )

    print("Connecting to Qdrant...", file=sys.stderr)
    qdrant = QdrantStorage(
        url=config.qdrant.url,
        collection_name=config.qdrant.collection_name,
        vector_size=config.qdrant.vector_size,
    )
    qdrant.ensure_collection()  # also creates the payload indexes

    client = qdrant._client  # pylint: disable=protected-access

    # Scroll existing points and match them to Postgres by file_path.
    print("Scrolling collection and rewriting payloads...", file=sys.stderr)
    next_offset = None
    total_points = 0
    updated = 0
    missing_in_pg = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=config.qdrant.collection_name,
            limit=500,
            with_payload=["file_path"],
            with_vectors=False,
            offset=next_offset,
        )
        if not points:
            break

        # Build (point_id, new_payload) pairs for this batch.
        batch_ops: list[tuple[int, dict[str, Any]]] = []
        for pt in points:
            total_points += 1
            fp = (pt.payload or {}).get("file_path")
            if not fp:
                continue
            rec = records.get(fp)
            if rec is None:
                missing_in_pg += 1
                continue
            batch_ops.append((pt.id, _build_payload(rec)))

        if not args.dry_run and batch_ops:
            # One overwrite_payload call per point is robust — Qdrant also
            # supports ``set_payload`` per-id, which fully replaces the
            # payload for that id.
            for pid, payload in batch_ops:
                client.overwrite_payload(
                    collection_name=config.qdrant.collection_name,
                    payload=payload,
                    points=[pid],
                    wait=False,
                )
        updated += len(batch_ops)
        print(
            f"  scrolled {total_points} points, queued {updated} payload rewrites...",
            file=sys.stderr,
            flush=True,
        )

        if next_offset is None:
            break

    print("=== repair summary ===", file=sys.stderr)
    print(f"  points in collection:      {total_points}", file=sys.stderr)
    print(f"  payloads rewritten:        {updated}", file=sys.stderr)
    print(f"  points with no PG record:  {missing_in_pg}", file=sys.stderr)
    if args.dry_run:
        print("  (dry-run: no writes performed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
