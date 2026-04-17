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
  caption.py     - Ollama VLM captioning with image resize
  embed.py       - Text embedding via Ollama nomic-embed-text
  geocode.py     - Offline reverse geocoding (lat/lon to city name)
  storage.py     - PostgreSQL + Qdrant client wrappers
  pipeline.py    - Orchestrator: scan, resume, per-file staged processing
  cli.py         - Typer CLI (index, search, label-faces, status, etc.)
scripts/
  init_db.sql    - PostgreSQL DDL (tables + indexes)
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
