"""One-time migration: strip markdown headers/bullets from photos.caption.

Haiku previously captioned photos as markdown (``# Photo Description``,
``## People``, leading ``-`` bullets).  The updated CAPTION_PROMPT asks
for plain prose going forward, but this migration cleans up captions
that were generated before the prompt change.

Cleans:
  * ATX headers (``# ...``, ``## ...``, up to ``######``) — keep the
    heading *text*, drop the hashes.
  * Leading list markers (``- ``, ``* ``, numbered ``1. ``).
  * Runs of blank lines, collapsing to a single newline.
  * Leading/trailing whitespace.

Does NOT touch:
  * Captions that already look clean (no-op UPDATE still runs but the
    value is unchanged — negligible cost).
  * Stored vectors in Qdrant — those need a stage-4 re-embed after this
    migration, triggered separately via
    ``photo-search index --stage embed`` (plus clearing the ``embedded``
    flag in indexing_status first).

Usage::

    python scripts/strip_caption_markdown.py --dry-run     # print sample
    python scripts/strip_caption_markdown.py               # commit

Read config from ``config.yaml`` or PHOTO_SEARCH_* env vars.  Requires
postgres.connection_string to point at the live DB (likely via a
``kubectl port-forward svc/postgres 5432:5432`` from a workstation).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
import psycopg2.extras

from photo_search.config import load_config


_ATX_HEADER = re.compile(r"^#{1,6}\s+")
_LIST_MARKER = re.compile(r"^(?:[-*]\s+|\d+\.\s+)")
_MULTI_BLANK = re.compile(r"\n{2,}")


def strip_markdown(text: str) -> str:
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        s = _ATX_HEADER.sub("", line)
        s = _LIST_MARKER.sub("", s)
        lines.append(s)
    cleaned = "\n".join(lines)
    cleaned = _MULTI_BLANK.sub("\n\n", cleaned)
    return cleaned.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show before/after samples without writing.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of before/after samples to print.",
    )
    args = parser.parse_args()

    cfg = load_config()
    conn = psycopg2.connect(cfg.postgres.connection_string)
    conn.autocommit = False

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT file_path, caption FROM photos "
                "WHERE caption IS NOT NULL AND caption <> ''"
            )
            rows = cur.fetchall()

        print(f"Loaded {len(rows)} captions.")

        updates: list[tuple[str, str]] = []
        changed = 0
        for row in rows:
            original = row["caption"]
            cleaned = strip_markdown(original)
            if cleaned != original:
                changed += 1
                updates.append((cleaned, row["file_path"]))

        print(f"{changed} caption(s) will change, {len(rows) - changed} are already clean.")

        # Print a few samples.
        for original_row, (cleaned, file_path) in zip(
            [r for r in rows if strip_markdown(r["caption"]) != r["caption"]][: args.sample],
            updates[: args.sample],
        ):
            print()
            print(f"--- {file_path} ---")
            print("BEFORE:")
            print(original_row["caption"][:400])
            print("AFTER:")
            print(cleaned[:400])

        if args.dry_run:
            print("\n(dry-run — no writes)")
            return

        if not updates:
            print("Nothing to update.")
            return

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE photos SET caption = %s WHERE file_path = %s",
                updates,
                page_size=500,
            )
        conn.commit()
        print(f"\nCommitted {len(updates)} updates.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
