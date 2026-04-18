# Fix search + re-caption runbook

This runbook walks through:

1. Baseline-backing-up the current (broken) database.
2. Applying the code fixes.
3. Re-captioning every photo with Anthropic Haiku (the **~$30 step**).
4. **Taking a post-caption backup and proving the restore round-trips bit-perfectly**
   — so if anything breaks downstream you never have to pay for captions again.
5. Re-embedding with Ollama.
6. Repairing Qdrant payloads.
7. Smoke-testing `--person` search end-to-end.

Every step uses absolute paths so you can copy-paste into a fresh shell.

---

## 0. What the underlying bugs were

Audited and fixed in the working tree (not yet committed):

| Bug | File | Effect |
|---|---|---|
| Resumed `embed` stage upserts with `caption=None`, which overwrites the previously-stored caption via `caption = EXCLUDED.caption` on `ON CONFLICT`. | [photo-search/photo_search/storage.py](photo-search/photo_search/storage.py) | `photos.caption` was NULLed for 13,916 of 13,921 rows — the $30 of Anthropic captions went to waste. |
| Resumed `embed` stage builds `IndexedPhoto.faces=[]` when the faces stage was already done in a prior run. | [photo-search/photo_search/pipeline.py](photo-search/photo_search/pipeline.py) | Every Qdrant payload was written as `"faces": []` even though Postgres held 11,743 labeled face rows. |
| No keyword payload index on `faces` in the Qdrant collection. | [photo-search/photo_search/storage.py](photo-search/photo_search/storage.py) | `MatchValue` against array-typed payload fields silently returns zero hits — `--person eva` always returned nothing. |

Fixes applied:

- `PostgresStorage.upsert_photo` now uses `COALESCE(EXCLUDED.caption, photos.caption)` so a resumed embed can never wipe a caption.
- The pipeline's embed stage now reloads `photo_faces` and `photos.caption` from Postgres and reconstructs `IndexedPhoto.faces` + `IndexedPhoto.caption` so the Qdrant payload reflects current state.
- `QdrantStorage.ensure_collection()` now idempotently creates keyword/integer/datetime payload indexes on `faces`, `year`, `file_type`, `date_taken`.

---

## 1. Pre-flight checks

```bash
# Working dir — all commands below assume this is your cwd.
cd /Users/michael/_code/ai-photos/photo-search

# Postgres port-forward up and healthy.
PGPASSWORD=my-ai-photos-password psql -h localhost -p 5432 -U photouser -d photosearch \
  -c "SELECT 'pg ok' AS status;"

# Qdrant ingress reachable.
curl -s https://qdrant.k8s.blacktoaster.com/collections/photos \
  | python3 -c "import sys,json; d=json.load(sys.stdin)['result']; print('qdrant ok, points:', d['points_count'])"

# Anthropic key present (needed for step 5).
test -n "$ANTHROPIC_API_KEY" && echo "ANTHROPIC_API_KEY ok" || echo "!! ANTHROPIC_API_KEY missing"

# Ollama up (needed for step 7).
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print('ollama ok, models:', [m['name'] for m in json.load(sys.stdin).get('models', [])])"

# Virtualenv has all deps.
./.venv/bin/python -c "import psycopg2, anthropic, qdrant_client, PIL; print('deps ok')"
```

All five checks must print `ok`. If any fail, restart the relevant port-forward / Ollama / venv before proceeding.

---

## 2. Baseline backup (cheap insurance, takes ~1 minute)

Snapshot the *current* database before any writes happen.

```bash
mkdir -p backups
./.venv/bin/python scripts/backup_db.py --verify --out backups/pre-fix.json.gz
```

Expected tail output:

```
=== backup complete ===
  face_identities             5 rows
  photos                  13921 rows
  photo_faces             23996 rows
  indexing_status         13922 rows
  -> backups/pre-fix.json.gz
```

File size should be ~50 MB (gzipped). `photo_faces` dominates because each row carries a 512-dim ArcFace embedding.

---

## 3. Ensure the code fixes are live

The fixes are already on the working tree. Install the package in the venv (if not already):

```bash
./.venv/bin/pip install -e .
```

Sanity-check the Qdrant payload indexes get created (idempotent — safe to run any time):

```bash
./.venv/bin/python -c "
from photo_search.config import load_config
from photo_search.storage import QdrantStorage
c = load_config()
q = QdrantStorage(c.qdrant.url, c.qdrant.collection_name, c.qdrant.vector_size)
q.ensure_collection()
print('ok')
"

# Verify indexes exist.
curl -s https://qdrant.k8s.blacktoaster.com/collections/photos \
  | python3 -c "
import sys, json
print(json.dumps(json.load(sys.stdin)['result'].get('payload_schema', {}), indent=2))
"
```

Expected: the second command prints an object containing keys `faces`, `year`, `file_type`, `date_taken` — **not `{}` like before**.

---

## 4. Switch the captioner provider to Anthropic

Edit [photo-search/config.yaml](photo-search/config.yaml) so `captioner.provider: "anthropic"`:

```yaml
captioner:
  provider: "anthropic"   # was "ollama"
  anthropic:
    model: "claude-haiku-4-5"
    max_tokens: 300
```

Leave the Ollama embedding config as-is (`ollama.embedding_model: "nomic-embed-text"`); it's free and stays local.

---

## 5. Reset the `captioned` and `embedded` flags, then re-caption ($$$ step)

### 5a. Reset flags (keeps `exif_extracted`, `faces_extracted`, `faces_classified` = true)

```bash
PGPASSWORD=my-ai-photos-password psql -h localhost -p 5432 -U photouser -d photosearch -c "
BEGIN;
UPDATE indexing_status
   SET captioned = FALSE,
       embedded  = FALSE,
       error     = NULL,
       last_updated = NOW();
SELECT
  COUNT(*) FILTER (WHERE captioned)        AS captioned,
  COUNT(*) FILTER (WHERE NOT captioned)    AS to_caption,
  COUNT(*) FILTER (WHERE embedded)         AS embedded,
  COUNT(*) FILTER (WHERE NOT embedded)     AS to_embed
FROM indexing_status;
COMMIT;
"
```

Expected: `to_caption` and `to_embed` ≈ 13,922.

### 5b. Run captioning (Anthropic Haiku, concurrency 8)

```bash
./.venv/bin/python -m photo_search index --captions-only --concurrency 8 --verbose 2>&1 \
  | tee logs/caption-$(date +%Y%m%d-%H%M%S).log
```

Tail the log; the CLI will report progress and per-file timing. Expect **~$30** in Anthropic spend based on your previous run.

### 5c. Verify captions actually landed in Postgres

```bash
PGPASSWORD=my-ai-photos-password psql -h localhost -p 5432 -U photouser -d photosearch -c "
SELECT
  COUNT(*)                                    AS total_photos,
  COUNT(caption)                              AS with_caption,
  ROUND(100.0 * COUNT(caption) / COUNT(*), 1) AS pct,
  MIN(length(caption))                        AS min_len,
  MAX(length(caption))                        AS max_len
FROM photos;
"
```

✅ PASS: `with_caption` is > 13,800 and `pct` is ~99%.
❌ FAIL: `with_caption` is still near-zero → stop, investigate, don't proceed to step 7.

---

## 6. **Post-caption backup and restore-test — the gate between $$$ and everything else**

**Do not run step 7 until both sub-steps pass.** This is the insurance you asked for.

### 6a. Take a fresh backup

```bash
./.venv/bin/python scripts/backup_db.py --verify --out backups/post-caption.json.gz
```

Record the summary line counts (you'll match them below).

### 6b. Dry-run the restore against the dump (no writes)

```bash
./.venv/bin/python scripts/restore_db.py --dry-run backups/post-caption.json.gz
```

Expected output: four lines matching 6a's counts. Any parse error here means the dump is corrupt — re-take it before proceeding.

### 6c. Optional but recommended: full round-trip proof

This actually wipes the live DB and restores from the dump, then compares MD5 checksums of the content. It gives you 100% certainty the dump is sufficient to rebuild from scratch. Skip only if you're confident.

```bash
# --- capture pre-wipe checksum ---
PRE=$(PGPASSWORD=my-ai-photos-password psql -h localhost -p 5432 -U photouser -d photosearch -At -c "
SELECT MD5(string_agg(t||'|'||cnt::text||'|'||chk, ';' ORDER BY t)) FROM (
  SELECT 'photos' t, COUNT(*) cnt, MD5(string_agg(file_path||'|'||COALESCE(caption,''), ',' ORDER BY file_path)) chk FROM photos
  UNION ALL SELECT 'photo_faces', COUNT(*), MD5(string_agg(photo_file_path||'|'||face_label||'|'||COALESCE(confidence::text,''), ',' ORDER BY id)) FROM photo_faces
  UNION ALL SELECT 'face_identities', COUNT(*), MD5(string_agg(label||'|'||sample_count::text, ',' ORDER BY label)) FROM face_identities
  UNION ALL SELECT 'indexing_status', COUNT(*), MD5(string_agg(file_path||'|'||embedded::text||'|'||captioned::text, ',' ORDER BY file_path)) FROM indexing_status
) x;
")
echo "pre-wipe digest: $PRE"

# --- wipe the live DB ---
PGPASSWORD=my-ai-photos-password psql -h localhost -p 5432 -U photouser -d photosearch -c "
TRUNCATE indexing_status, photo_faces, photos, face_identities CASCADE;
"

# --- restore from the post-caption dump ---
./.venv/bin/python scripts/restore_db.py --yes backups/post-caption.json.gz

# --- capture post-restore checksum and compare ---
POST=$(PGPASSWORD=my-ai-photos-password psql -h localhost -p 5432 -U photouser -d photosearch -At -c "
SELECT MD5(string_agg(t||'|'||cnt::text||'|'||chk, ';' ORDER BY t)) FROM (
  SELECT 'photos' t, COUNT(*) cnt, MD5(string_agg(file_path||'|'||COALESCE(caption,''), ',' ORDER BY file_path)) chk FROM photos
  UNION ALL SELECT 'photo_faces', COUNT(*), MD5(string_agg(photo_file_path||'|'||face_label||'|'||COALESCE(confidence::text,''), ',' ORDER BY id)) FROM photo_faces
  UNION ALL SELECT 'face_identities', COUNT(*), MD5(string_agg(label||'|'||sample_count::text, ',' ORDER BY label)) FROM face_identities
  UNION ALL SELECT 'indexing_status', COUNT(*), MD5(string_agg(file_path||'|'||embedded::text||'|'||captioned::text, ',' ORDER BY file_path)) FROM indexing_status
) x;
")
echo "post-restore digest: $POST"

if [ "$PRE" = "$POST" ]; then echo "RESTORE OK"; else echo "!!! DIGEST MISMATCH !!!"; exit 1; fi
```

✅ PASS: `RESTORE OK` printed. You can now trust the dump; proceed to step 7.
❌ FAIL: `DIGEST MISMATCH` — **do not proceed**. The dump is incomplete. Stop and investigate.

> *This exact round-trip was already proven against the pre-fix database (2026-04-17). Running it again against the post-caption data just confirms the backup you rely on for the next six hours is in fact complete.*

---

## 7. Re-embed with Ollama (free, ~30 min)

```bash
./.venv/bin/python -m photo_search index --embed-only --concurrency 1 --verbose 2>&1 \
  | tee logs/embed-$(date +%Y%m%d-%H%M%S).log
```

Concurrency 1 because Ollama is GPU/CPU-bound and parallel calls just queue.

Verify the run stored non-empty faces + caption in Qdrant payloads:

```bash
./.venv/bin/python -c "
import json, urllib.request
req = urllib.request.Request(
    'https://qdrant.k8s.blacktoaster.com/collections/photos/points/scroll',
    data=json.dumps({'limit': 50, 'with_payload': True, 'with_vector': False}).encode(),
    headers={'Content-Type': 'application/json'},
)
pts = json.load(urllib.request.urlopen(req))['result']['points']
nonempty_faces = sum(1 for p in pts if p['payload'].get('faces'))
nonempty_cap   = sum(1 for p in pts if p['payload'].get('caption'))
print(f'of 50 sampled: {nonempty_faces} have faces, {nonempty_cap} have captions')
"
```

Expected: both counts > 0 (captions on almost every photo, faces on whichever photos had recognised people).

---

## 8. Repair stale Qdrant payloads (catches any points that sneaked through)

Idempotent re-upsert of payloads from Postgres. Run it once after step 7.

```bash
./.venv/bin/python scripts/repair_qdrant_payloads.py --dry-run     # preview
./.venv/bin/python scripts/repair_qdrant_payloads.py               # commit
```

Expected: `points in collection` ≈ 13,921, `payloads rewritten` ≈ 13,921, `points with no PG record` ≈ 0.

---

## 9. Smoke tests

### 9a. Filter by each person

```bash
for p in austin eva marcella michael mike; do
  count=$(curl -s -X POST https://qdrant.k8s.blacktoaster.com/collections/photos/points/count \
    -H 'Content-Type: application/json' \
    -d "{\"filter\":{\"must\":[{\"key\":\"faces\",\"match\":{\"value\":\"$p\"}}]},\"exact\":true}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['count'])")
  printf "  %-10s %s\n" "$p" "$count"
done
```

Expected: counts roughly matching Postgres `photo_faces` row counts per label (michael ~6,356; marcella ~2,950; mike ~1,729; austin ~560; eva ~132).

### 9b. End-to-end CLI search

```bash
./.venv/bin/python -m photo_search search "at the beach" --person eva --top 5
./.venv/bin/python -m photo_search search "birthday cake"          --top 5
./.venv/bin/python -m photo_search search "sitting outside" --person eva --top 5
```

Expected: the person-filtered queries return only photos that actually contain Eva.

---

## 10. Commit the fixes

After you've confirmed everything works, commit the code changes:

```bash
cd /Users/michael/_code/ai-photos
git status
git diff photo-search/photo_search/storage.py photo-search/photo_search/pipeline.py
git add photo-search/photo_search/storage.py \
        photo-search/photo_search/pipeline.py \
        photo-search/scripts/backup_db.py \
        photo-search/scripts/restore_db.py \
        photo-search/scripts/repair_qdrant_payloads.py \
        .gitignore \
        RUNBOOK.md
git commit -m "fix: preserve captions + populate faces in Qdrant payloads on resume"
```

---

## Backup cadence going forward

- Take a new `backups/YYYYMMDD-reason.json.gz` dump **before** any pipeline run that could touch `photos.caption` or `photo_faces`.
- Keep the last ~5 dumps locally. They're ~50 MB each.
- The directory is already in `.gitignore` so nothing accidentally commits.

When in doubt: `./.venv/bin/python scripts/backup_db.py --verify` takes under a minute and costs nothing.

---

## Appendix: restoring a single table

If you only want to recover `photos.caption` (say, after another bug) and leave faces + indexing_status alone:

```bash
./.venv/bin/python scripts/restore_db.py --tables photos --yes backups/post-caption.json.gz
```

TRUNCATE + insert happens only on the named tables.

## Appendix: file manifest

- [photo-search/scripts/backup_db.py](photo-search/scripts/backup_db.py) — portable psycopg2 NDJSON dumper (server-side cursor, batches of 500).
- [photo-search/scripts/restore_db.py](photo-search/scripts/restore_db.py) — replays a dump; TRUNCATE-CASCADE-then-INSERT.
- [photo-search/scripts/repair_qdrant_payloads.py](photo-search/scripts/repair_qdrant_payloads.py) — overwrites Qdrant payloads from current Postgres state without touching vectors.
