# photo-search

Local CLI tool that indexes photos from NAS storage and enables natural language search using AI. Runs on macOS, stores metadata in PostgreSQL (k8s) and vectors in Qdrant (k8s).

## Architecture

```
Mac (local)                          k3s cluster
+--------------------------+         +---------------------------+
| photo-search CLI         |         | PostgreSQL (ai-photos ns) |
|   - EXIF extraction      | ------> |   - photo metadata        |
|   - InsightFace (faces)  |   PG    |   - face identities       |
|   - Ollama VLM (captions)|  port   |   - indexing status       |
|   - nomic-embed (vectors)| forward +---------------------------+
+--------------------------+         | Qdrant (ai-photos ns)     |
         |                   ------> |   - 768-dim vectors       |
         v                  ingress  |   - cosine similarity     |
  /Volumes/<nas>/Photos              +---------------------------+
  (NFS mount from NAS)
```

- **InsightFace** (buffalo_l): Face detection + 512-dim ArcFace embeddings, runs locally via ONNX
- **Captioning**: Vision-language model for photo descriptions
  - **Ollama** (qwen2.5vl:8b): Local inference (~5-15 sec/photo)
  - **Anthropic** (claude-haiku-4-5): Cloud API (~0.3 sec/photo, $0.002/photo) - **70x faster**
- **Ollama** (nomic-embed-text): Text embedding for search vectors (768-dim), runs locally
- **PostgreSQL**: Photo metadata, face identities, per-file indexing status (resume tracking)
- **Qdrant**: Vector similarity search over caption embeddings

## Photo source

Photos were exported from Apple Photos via **File > Export > Export Unmodified Originals** to Desktop, then rsynced to the QNAP NAS:

```bash
rsync -rvh --progress --no-times ~/Desktop/Photos/ /Volumes/<nas-share>/Photos/
```

This produces a flat directory of ~14K files (mostly HEIC from iPhones, plus JPG/PNG). The `--no-times` flag means filesystem timestamps are rsync time, not photo time -- that's fine because we extract dates from EXIF data embedded in the files.

## Prerequisites

Before starting, ensure the following are online:

- **k3s cluster** with the `ai-photos` namespace, PostgreSQL, and Qdrant deployed (see Helm charts in `../infra/`)
- **NFS mount** at your configured `photos.source_dir` path (see `config.yaml`)
- **Python 3.12** (not 3.14 -- insightface/onnxruntime don't support it yet)
- **For captioning**, choose one:
  - **Anthropic API** (recommended for speed): Set `ANTHROPIC_API_KEY` and `captioner.provider: "anthropic"` in config
  - **Ollama** (local): Install and pull models: `ollama pull qwen3-vl:8b && ollama pull nomic-embed-text`
- **For search/embedding**: Ollama required: `ollama pull nomic-embed-text`

## Setup from scratch

### 1. Deploy infrastructure on k8s

NFS storage is already provisioned on the NAS. The Helm charts create PVs and PVCs pointing to NFS paths configured in each chart's `values.yaml`. Copy the `.example` files and fill in your NFS server IP and paths.

```bash
# Create namespace
kubectl create namespace ai-photos

# Deploy PostgreSQL and Qdrant (creates PV, PVC, Deployment, Service each)
helm install postgres ../infra/postgres/ -n ai-photos
helm install qdrant ../infra/qdrant/ -n ai-photos

# Verify pods are running
kubectl get pods -n ai-photos
```

### 2. Access services

PostgreSQL needs port-forwarding (TCP, can't go through HTTP ingress):
```bash
kubectl port-forward -n ai-photos svc/postgres 5432:5432
```

Qdrant is accessible via ingress at `https://<your-qdrant-ingress-host>` (configured in `../infra/qdrant/values.yaml`).

### 3. Install the CLI tool

```bash
cd photo-search
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Configure

Copy `config.yaml.example` to `config.yaml` and fill in your values:
- `photos.source_dir`: Path to your NFS-mounted photo directory
- `postgres.connection_string`: Postgres connection (default uses localhost via port-forward)
- `qdrant.url`: Qdrant endpoint (your ingress URL or localhost via port-forward)
- `ollama.base_url`: Ollama endpoint (default `localhost:11434`)

All settings can be overridden via environment variables with prefix `PHOTO_SEARCH_` and double-underscore nesting (e.g. `PHOTO_SEARCH_POSTGRES__CONNECTION_STRING`).

### 5. Initialize databases

```bash
photo-search init-db
```

Creates 4 Postgres tables (photos, face_identities, photo_faces, indexing_status) and the Qdrant `photos` collection.

### 6. Mount NAS

```bash
sudo mkdir -p /Volumes/<nas>
sudo mount -t nfs -o resvport,soft,timeo=10 <nas-host>:<nas-export-path> /Volumes/<nas>
```

The `soft,timeo=10` options prevent the CLI from hanging if NFS drops. Adjust the NFS host and export path to match your NAS configuration.

## Indexing workflow

Indexing runs in stages. Each stage is resumable -- Ctrl+C and restart anytime.

### Stage 1: Face Detection
**Services needed:** PostgreSQL only (no Ollama/Anthropic needed)

```bash
photo-search index --faces-only
```

Extracts EXIF metadata (date, GPS, camera) and runs InsightFace face detection on every photo. ~265 files/min on M3 Pro.

### Stage 2: Label People
**Services needed:** PostgreSQL only (no Ollama/Anthropic needed)

Label common people first (they appear in many photos):

```bash
# Random sampling - good for people who appear frequently
photo-search label-faces --label="Michael" --display-name="Michael" --samples=5
photo-search label-faces --label="Marcella" --display-name="Marcella" --samples=5

# After labeling, apply identities to all detected faces
photo-search reclassify-faces
```

For rare people (appear in <5% of photos), use **seed-photo bootstrapping** - provide a known photo and the system finds similar faces ranked by similarity:

```bash
# Bootstrap from a known photo - MUCH faster for rare people
photo-search label-faces --label="Austin" \
  --seed-photo="/Volumes/voyager2/Photos/IMG_1234.HEIC" \
  --samples=5

# System will:
# 1. Detect Austin's face in the seed photo
# 2. Compute similarity to ALL unknown faces (cosine similarity)
# 3. Show you the most similar faces first (highest confidence matches)
# 4. After confirming 5 samples, create the identity

# Apply the new identity to all faces
photo-search reclassify-faces
```

**Workflow summary:**
1. Run face detection once (`--faces-only`)
2. Label common people (random sampling works fine)
3. Run `reclassify-faces` to apply those identities
4. Label rare people using `--seed-photo` (finds them efficiently)
5. Run `reclassify-faces` again after each new identity

### Stage 3: Captioning
**Services needed:** Anthropic API OR Ollama

**Option A: Anthropic (Recommended - 70x faster)**
```bash
# Set API key in environment
export ANTHROPIC_API_KEY="your-key-here"

# Update config.yaml:
# captioner:
#   provider: "anthropic"
#   anthropic:
#     model: "claude-haiku-4-5"
#     max_tokens: 300

# Run with parallel processing (8-10 workers for Tier 1)
photo-search index --captions-only --concurrency=10
```

Performance: ~0.3 sec/photo, ~$0.002/photo. For 13K photos: ~1 hour, ~$30 total.

**Option B: Ollama (Local)**
```bash
# Make sure ollama is running: ollama serve
photo-search index --captions-only
```

Performance: ~5-15 sec/photo. For 13K photos: ~20-40 hours. Concurrency=1 (GPU-bound).

### Stage 4: Embedding
**Services needed:** Ollama (for nomic-embed-text), Qdrant

```bash
# Make sure ollama is running: ollama serve
photo-search index --embed-only
```

Builds combined search text (caption + face labels + location + date + camera), generates 768-dim vector, stores in Qdrant. Fast -- ~0.5 sec/photo.

### Alternative: All stages at once

```bash
photo-search index
```

Runs all stages sequentially. Use `--concurrency=N` to parallelize captioning (only helps with Anthropic).

## Service Requirements Summary

| Operation | PostgreSQL | Ollama | Anthropic API | Qdrant |
|-----------|-----------|---------|---------------|---------|
| `index --faces-only` | ✅ | ❌ | ❌ | ❌ |
| `label-faces` | ✅ | ❌ | ❌ | ❌ |
| `label-faces --seed-photo` | ✅ | ❌ | ❌ | ❌ |
| `reclassify-faces` | ✅ | ❌ | ❌ | ❌ |
| `index --captions-only` (Ollama) | ✅ | ✅ | ❌ | ❌ |
| `index --captions-only` (Anthropic) | ✅ | ❌ | ✅ | ❌ |
| `index --embed-only` | ✅ | ✅ | ❌ | ✅ |
| `search` | ✅ | ✅ | ❌ | ✅ |

**Key insights:**
- Face operations (detect, label, reclassify) only need PostgreSQL - no AI services required
- Captioning needs either Ollama OR Anthropic (pick one based on speed vs. local preference)
- Search and embedding always need Ollama (for nomic-embed-text)

## Searching

```bash
photo-search search "sunset at the beach"
photo-search search "kids playing in the park" --person alice
photo-search search "family dinner" --year 2023 --top 5
photo-search search "hiking" --after 2022-01-01 --before 2022-12-31
photo-search search "birthday party" --open  # opens top result in Preview
```

### Filtering by person

`--person` is a hard filter: only photos whose labeled faces contain that label are returned. Pass the flag multiple times **or** a comma-separated list to require ALL listed people on the same photo (AND semantics):

```bash
# Equivalent — photos containing BOTH Austin and Michael:
photo-search search "at the park" --person austin --person michael
photo-search search "at the park" --person austin,michael
```

Unrecognized labels return zero results (use `photo-search label-faces` to list existing identities).

### Filtering by location

`--location` accepts `"City, State"` (US) or `"City, Country"` (international). The city / state / country are stored as separate Qdrant payload fields at embed time, so matches are exact (not fuzzy text):

```bash
# US: state as abbreviation or full name
photo-search search "at the beach" --location "Laguna Beach, CA"
photo-search search "rock climbing"  --location "Zion National Park, Utah"

# International: country by name or ISO2 code
photo-search search "architecture"   --location "Florence, Italy"
photo-search search "canals"         --location "Amsterdam, Netherlands"

# Combine with person/date filters
photo-search search "dinner" \
  --person marcella,michael \
  --location "Florence, Italy" \
  --year 2023
```

Under the hood: ``photo_search/geo.py`` maps state abbreviations (`CA` → `california`) and country names (`Italy` → `IT`) to the values actually stored in Qdrant. The city token is matched case-insensitively against the ``city`` payload field. If the state/country token is unrecognized it falls back to a raw region match (useful for exotic admin regions like `Tuscany` or `Provence`).

## Adding new photos (routine imports)

Once the initial backfill is done, adding new photos is fully incremental. The pipeline scans `photos.source_dir`, skips files already tracked in `indexing_status`, and only runs the stages that are still incomplete for each file.

### 0. Bring up the services you'll need

Depending on which stages run, you need some mix of Postgres (always), Ollama (always, for embedding), Qdrant (always, for embedding), and Anthropic API (if captioning provider = anthropic):

```bash
# Terminal A — Postgres port-forward (required for every command).
kubectl port-forward -n ai-photos svc/postgres 5432:5432

# Terminal B — Ollama running locally (required for embedding; also for
# captioning if provider="ollama").
ollama serve
# First time only on this machine:
#   ollama pull nomic-embed-text
#   ollama pull qwen3-vl:8b   # only if you caption with Ollama

# Qdrant: already reachable via its ingress URL from config.yaml —
# no port-forward needed.

# Anthropic: export the API key if captioner.provider is "anthropic".
export ANTHROPIC_API_KEY="sk-ant-..."
```

NFS also has to be mounted (`mount | grep Photos` should show your Photos share). If it isn't:

```bash
sudo mount -t nfs -o resvport,soft,timeo=10 <nas-host>:<nas-export-path> /Volumes/<nas>
```

### 1. Export new photos from Apple Photos → local staging

In Apple Photos, select the new imports and use **File → Export → Export Unmodified Originals** to e.g. `~/Desktop/new-batch/`. This preserves EXIF (dates, GPS) which the pipeline depends on.

### 2. Sync the staging folder to the NAS

```bash
rsync -rvh --progress --no-times \
  ~/Desktop/new-batch/ \
  /Volumes/<nas-share>/Photos/
```

`--no-times` is fine because we extract dates from EXIF, not from filesystem mtime. Adjust the destination if your `photos.source_dir` in `config.yaml` points somewhere else.

### 3. Run the indexing pipeline

```bash
cd photo-search
source .venv/bin/activate

# Full pipeline (faces → caption → embed).  Skips files already complete.
# Concurrency=10 is safe for Anthropic Tier 1 — tune to your tier.
# Drop to --concurrency=1 if captioner.provider="ollama" (GPU-bound).
photo-search index --concurrency=10
```

That single command:
- Enumerates files under `source_dir`, creates `indexing_status` rows for anything new.
- Runs face detection on files without faces yet. Existing identities apply automatically at embed time via the `photo_faces` join, so photos of already-labeled people are immediately searchable by `--person`.
- Captions any file that doesn't have a caption yet (skips already-captioned photos).
- Embeds any file whose caption or face set has changed since last embed — upserts to Qdrant with the latest payload, including the structured `city` / `region` / `country_code` fields used by `--location`.

### 4. Verify

```bash
photo-search status                   # counts: total, with faces, with caption, embedded
photo-search search "something recent from the new batch" --top 5
```

### If the new batch contains people you haven't labeled yet

After step 3, check whether any new faces ended up as `unknown`:

```bash
# Lists every detected-but-unlabeled face cluster with sample counts.
photo-search status --detailed | grep unknown
```

Label the new people, then apply the identities and re-embed only the affected files:

```bash
# Label via random sampling (works for people appearing in many new photos)
photo-search label-faces --label="Eva" --display-name="Eva" --samples=5

# Or bootstrap from a known seed photo (better for rare people)
photo-search label-faces --label="Eva" \
  --seed-photo="/Volumes/<nas-share>/Photos/IMG_9876.HEIC" --samples=5

# Propagate the new identity to every matching face in the DB
photo-search reclassify-faces

# Re-embed so Qdrant payloads pick up the new face labels
photo-search index --embed-only
```

`reclassify-faces` prints which photos changed; the subsequent `--embed-only` only touches those.

### Running individual stages

If you want to interleave or parallelize steps (e.g. kick off captioning while manually labeling), each stage is safe to run alone:

```bash
photo-search index --faces-only       # detect faces on new files
photo-search index --captions-only    # caption any uncaptioned file
photo-search index --embed-only       # embed any file with a fresh caption/faces
```

All stages are resumable — Ctrl+C anywhere and just rerun.

### What you do NOT need to do on routine imports

- **No Qdrant payload repair.** `scripts/repair_qdrant_payloads.py` is for one-off disaster recovery (e.g. if the collection's payloads get out of sync with Postgres). New photos get correct payloads automatically at embed time.
- **No re-run of captions on old photos.** Captioning is skipped per-file when `captioned=true` in `indexing_status`.
- **No schema migrations.** The Postgres DDL and Qdrant collection are stable; adding new files doesn't touch either schema.

## Backup and disaster recovery

`scripts/backup_db.py` produces a portable gzipped NDJSON dump of the four Postgres tables. Recommended before any large batch operation (big captioning run, bulk reclassify, etc.):

```bash
# Create ./backups/photo-search-YYYYMMDD-HHMMSS.json.gz
./.venv/bin/python scripts/backup_db.py --verify

# Dry-run parse a dump without writing (sanity check)
./.venv/bin/python scripts/restore_db.py --dry-run backups/<file>.json.gz

# Full restore (TRUNCATEs target tables, prompts for confirmation)
./.venv/bin/python scripts/restore_db.py backups/<file>.json.gz
```

If Qdrant payloads drift out of sync with Postgres (rare — typically only after a bug or manual intervention), repair them from Postgres without re-embedding:

```bash
./.venv/bin/python scripts/repair_qdrant_payloads.py --dry-run
./.venv/bin/python scripts/repair_qdrant_payloads.py
```

This preserves the existing vectors and only rewrites payloads (captions, faces, location fields, etc.).

## Monitoring and recovery

```bash
photo-search status              # aggregate counts per stage
photo-search status --detailed   # per-file breakdown
photo-search reindex --all-errors  # retry all failed files
photo-search reindex --file /path/to/specific/photo.heic  # retry one file
```

## Project structure

```
photo_search/
  config.py      - YAML + env var config loading (Pydantic Settings)
  models.py      - Data models (PhotoMetadata, DetectedFace, IndexedPhoto, etc.)
  exif.py        - EXIF extraction (Pillow + pillow-heif for HEIC support)
  faces.py       - InsightFace detection + cosine similarity classification
  caption.py     - Ollama / Anthropic VLM captioning with image resize
  embed.py       - Text embedding via Ollama nomic-embed-text
  geocode.py     - Offline reverse geocoding (lat/lon to "City, Region, CC")
  geo.py         - US state / country lookup tables for the --location filter
  storage.py     - PostgreSQL + Qdrant client wrappers
  pipeline.py    - Orchestrator: scan, resume, per-file staged processing
  cli.py         - Typer CLI (index, search, label-faces, status, etc.)
scripts/
  init_db.sql                - PostgreSQL DDL (tables + indexes)
  backup_db.py               - Portable NDJSON dump of the 4 data tables
  restore_db.py              - Replay a dump (TRUNCATE + INSERT)
  repair_qdrant_payloads.py  - Rewrite Qdrant payloads from Postgres truth
config.yaml      - All configuration
```

## Key design decisions

- **Python 3.12**: Required for insightface/onnxruntime compatibility (not 3.14)
- **Pillow for image loading**: cv2.imread can't read HEIC -- all image loading goes through Pillow with pillow-heif registered
- **EXIF via getexif() + get_ifd()**: HEIC files store DateTimeOriginal and GPS in sub-IFDs that require explicit IFD access
- **Per-file, per-stage resume**: Each file has boolean flags in indexing_status tracking which stages completed. Pipeline checks before re-running.
- **Pluggable captioning backends**: Supports both Ollama (local) and Anthropic (cloud) via provider configuration
- **Parallel captioning for Anthropic**: ThreadPoolExecutor with thread-local PostgreSQL connections enables 8-10x parallelism (70x total speedup vs. Ollama)
- **Seed-photo face labeling**: For rare people, bootstrap from a known photo using cosine similarity to find similar faces efficiently
- **Deterministic Qdrant IDs**: MD5 hash of file_path ensures re-indexing overwrites the same point
- **NUL byte sanitization**: Some EXIF strings contain \x00 bytes that Postgres rejects -- stripped on insert
- **HTTPS ingress for Qdrant**: Uses REST API through nginx ingress with TLS verify disabled (internal CA)
- **Images resized before VLM**: Max 1536px (Ollama) or 1024px (Anthropic) to avoid context overflow and reduce costs
- **Structured location payload**: `location_name` (`"City, Region, CC"`) is also split into separate `city` / `region` / `country_code` Qdrant payload fields with keyword indexes — enables exact `--location` filters instead of relying on soft text-embedding matches
- **Keyword payload indexes**: `faces`, `city`, `region`, `country_code`, `file_type`, `year` (integer), `date_taken` (datetime) — created idempotently by `QdrantStorage.ensure_collection()` so `MatchValue` filters actually hit
- **COALESCE on upsert**: `PostgresStorage.upsert_photo` uses `COALESCE(EXCLUDED.col, photos.col)` for caption/caption_model/embedding_model so resumed pipeline runs never overwrite good values with NULLs
