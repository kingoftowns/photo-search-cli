"""Portable, version-agnostic Postgres backup for photo-search.

Writes a gzipped newline-delimited JSON dump of the four data tables
(``photos``, ``photo_faces``, ``face_identities``, ``indexing_status``) using
psycopg2 — avoids ``pg_dump`` client/server version mismatches.

Usage::

    # Back up to ./backups/photo-search-YYYYMMDD-HHMMSS.json.gz
    python scripts/backup_db.py

    # Custom output path
    python scripts/backup_db.py --out /path/to/dump.json.gz

    # Include verification counts in stdout
    python scripts/backup_db.py --verify

The companion ``restore_db.py`` replays a dump back into an empty-or-compatible
schema.  The backup is deliberately plain JSON (one table per line prefixed
with a header record) so it can be inspected, diffed, or partially restored by
hand.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

# Tables to back up, in dependency-safe restore order (parents first).
TABLES: list[str] = [
    "face_identities",
    "photos",
    "photo_faces",
    "indexing_status",
]


def _default_connstr() -> str:
    """Resolve the connection string from config.yaml or env var."""
    env = os.environ.get("PHOTO_SEARCH_POSTGRES__CONNECTION_STRING")
    if env:
        return env

    # Fall back to config.yaml next to the script.
    try:
        # Local import to avoid eager dep.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from photo_search.config import load_config  # type: ignore
        return load_config().postgres.connection_string
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not resolve Postgres connection string.  Set "
            "PHOTO_SEARCH_POSTGRES__CONNECTION_STRING or ensure config.yaml "
            "is readable."
        ) from exc


def _serialise_cell(val):
    """JSON-safe cell encoder: bytes → b64, datetime → ISO, else passthrough."""
    if isinstance(val, (bytes, memoryview)):
        return {"__bytes__": base64.b64encode(bytes(val)).decode("ascii")}
    if isinstance(val, datetime):
        return {"__dt__": val.isoformat()}
    return val


def backup(conn_str: str, out_path: Path, verify: bool = False) -> dict[str, int]:
    """Dump the four data tables to *out_path* (gzipped NDJSON)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    counts: dict[str, int] = {}

    # Server-side cursor batch size.  Each ``photo_faces`` row carries a
    # ~2KB embedding; at 500 rows/batch that's ~1MB per fetch — sized so
    # kubectl port-forwards stay responsive.
    FETCH_BATCH = 500

    conn = psycopg2.connect(conn_str)
    try:
        conn.set_session(readonly=True, autocommit=False)
        with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
            header = {
                "__meta__": {
                    "format": "photo-search.dump.v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "tables": TABLES,
                }
            }
            fh.write(json.dumps(header) + "\n")

            for table in TABLES:
                # Discover column names via a zero-row probe on a plain cursor.
                with conn.cursor() as probe:
                    probe.execute(f'SELECT * FROM "{table}" LIMIT 0')
                    colnames = [d.name for d in probe.description]

                fh.write(json.dumps({"__table__": table, "columns": colnames}) + "\n")

                # Server-side named cursor — streams from Postgres in fixed
                # batches instead of buffering the whole result set.
                cur_name = f"backup_cursor_{table}"
                with conn.cursor(
                    name=cur_name,
                    cursor_factory=psycopg2.extras.RealDictCursor,
                ) as cur:
                    cur.itersize = FETCH_BATCH
                    cur.execute(f'SELECT * FROM "{table}"')
                    n = 0
                    while True:
                        batch = cur.fetchmany(FETCH_BATCH)
                        if not batch:
                            break
                        for row in batch:
                            encoded = {
                                k: _serialise_cell(v) for k, v in row.items()
                            }
                            fh.write(json.dumps(encoded) + "\n")
                        n += len(batch)
                        if verify:
                            print(
                                f"  [{table}] streamed {n} rows...",
                                file=sys.stderr,
                                flush=True,
                            )
                counts[table] = n
                conn.commit()  # close server cursor before next table
    finally:
        conn.close()

    # Atomic rename — either the dump is complete or it doesn't exist.
    tmp_path.replace(out_path)

    if verify:
        print("=== backup complete ===", file=sys.stderr)
        for t, n in counts.items():
            print(f"  {t:20s} {n:>8d} rows", file=sys.stderr)
        print(f"  -> {out_path}", file=sys.stderr)

    return counts


def _default_out() -> Path:
    repo = Path(__file__).resolve().parent.parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo / "backups" / f"photo-search-{stamp}.json.gz"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backup photo-search Postgres data.")
    p.add_argument("--out", type=Path, default=None, help="Output path (.json.gz)")
    p.add_argument("--verify", action="store_true", help="Print row counts")
    p.add_argument(
        "--connstr",
        default=None,
        help="Override connection string (else use env or config.yaml)",
    )
    args = p.parse_args(argv)

    conn_str = args.connstr or _default_connstr()
    out_path = args.out or _default_out()

    backup(conn_str, out_path, verify=args.verify)
    # Print the path to stdout so shells can `tee` or capture.
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
