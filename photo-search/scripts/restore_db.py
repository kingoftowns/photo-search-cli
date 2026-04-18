"""Restore a photo-search dump produced by ``backup_db.py``.

Usage::

    # Restore into the configured database (prompts for confirmation unless
    # --yes is passed).  Existing rows for the restored tables are TRUNCATED
    # first so the restore is exact.
    python scripts/restore_db.py path/to/dump.json.gz

    # Dry-run: parse and validate the dump without writing anything.
    python scripts/restore_db.py --dry-run path/to/dump.json.gz

    # Restore only specific tables.
    python scripts/restore_db.py --tables photos,photo_faces dump.json.gz

``photo_faces`` has a foreign key to ``photos``, so tables are restored in the
order declared by the dump header.  TRUNCATE uses ``CASCADE`` to clear
dependents automatically.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras


def _default_connstr() -> str:
    env = os.environ.get("PHOTO_SEARCH_POSTGRES__CONNECTION_STRING")
    if env:
        return env
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from photo_search.config import load_config  # type: ignore
    return load_config().postgres.connection_string


def _deserialise_cell(val):
    if isinstance(val, dict):
        if "__bytes__" in val:
            return psycopg2.Binary(base64.b64decode(val["__bytes__"]))
        if "__dt__" in val:
            return datetime.fromisoformat(val["__dt__"])
    return val


def _iter_dump(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def restore(
    conn_str: str,
    dump_path: Path,
    tables: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Replay *dump_path* into the database."""
    iterator = _iter_dump(dump_path)

    header = next(iterator)
    meta = header.get("__meta__")
    if not meta or meta.get("format") != "photo-search.dump.v1":
        raise ValueError(f"Not a photo-search dump: {dump_path}")

    wanted = set(tables) if tables else set(meta["tables"])
    order = [t for t in meta["tables"] if t in wanted]

    # Group rows by table.  Memory-efficient enough for our scale (<100k rows).
    current_table: str | None = None
    current_cols: list[str] | None = None
    rows_by_table: dict[str, tuple[list[str], list[list]]] = {}

    for rec in iterator:
        if "__table__" in rec:
            current_table = rec["__table__"]
            current_cols = rec["columns"]
            if current_table in wanted:
                rows_by_table[current_table] = (current_cols, [])
            continue
        if current_table not in wanted:
            continue
        # Row record: align to declared column order.
        row = [
            _deserialise_cell(rec.get(c)) for c in current_cols  # type: ignore[arg-type]
        ]
        rows_by_table[current_table][1].append(row)

    counts = {t: len(rows_by_table[t][1]) for t in order}

    if dry_run:
        print("=== dry-run: parsed dump OK ===", file=sys.stderr)
        for t, n in counts.items():
            print(f"  {t:20s} {n:>8d} rows", file=sys.stderr)
        return counts

    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Restore in declared (parent-first) order.  TRUNCATE CASCADE
            # ensures dependents are cleared when the parent is wiped.
            for table in order:
                cols, rows = rows_by_table[table]
                # Truncate atomically with the insert to avoid empty-window
                # failures mid-restore.
                cur.execute(f'TRUNCATE "{table}" CASCADE')
                if not rows:
                    continue
                collist = ", ".join(f'"{c}"' for c in cols)
                placeholders = ", ".join(["%s"] * len(cols))
                sql = (
                    f'INSERT INTO "{table}" ({collist}) VALUES ({placeholders})'
                )
                psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
                print(f"restored {len(rows):>8d} rows into {table}", file=sys.stderr)

    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Restore a photo-search dump.")
    p.add_argument("dump", type=Path, help="Path to a .json.gz dump")
    p.add_argument("--tables", default=None, help="Comma-separated tables to restore")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--connstr", default=None)
    args = p.parse_args(argv)

    if not args.dump.is_file():
        print(f"Dump file not found: {args.dump}", file=sys.stderr)
        return 2

    conn_str = args.connstr or _default_connstr()
    tables = [t.strip() for t in args.tables.split(",")] if args.tables else None

    if not args.dry_run and not args.yes:
        print(
            f"About to TRUNCATE and restore from {args.dump} into:\n  {conn_str}",
            file=sys.stderr,
        )
        ans = input("  type 'yes' to continue: ").strip().lower()
        if ans != "yes":
            print("aborted.", file=sys.stderr)
            return 1

    restore(conn_str, args.dump, tables=tables, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
