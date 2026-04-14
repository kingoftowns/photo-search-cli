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
- **Ollama** (qwen2.5vl:7b): Vision-language model for photo captioning, runs locally
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
- **Ollama** installed with models pulled: `ollama pull qwen2.5vl:7b && ollama pull nomic-embed-text`

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

### Stage 1: EXIF + Face detection (no Ollama needed)

```bash
photo-search index --faces-only
```

Extracts EXIF metadata (date, GPS, camera) and runs InsightFace detection on every photo. ~265 files/min on an M-series Mac.

### Stage 2: Label faces

```bash
# For each family member -- shows face crops in Preview, you confirm y/n
photo-search label-faces --label alice --display-name "Alice" --samples 5 --photo-count 100
photo-search label-faces --label bob --display-name "Bob" --samples 5 --photo-count 100
# etc.

# Reclassify all previously detected faces with the new identities
photo-search reclassify-faces
```

### Stage 3: Captioning (needs `ollama serve` running)

```bash
photo-search index --captions-only
```

Generates VLM scene descriptions. Slowest stage -- ~5-15 sec/photo. Fully resumable.

### Stage 4: Embedding (needs `ollama serve` running)

```bash
photo-search index --embed-only
```

Builds combined search text (caption + faces + location + date + camera), generates 768-dim vector, stores in Qdrant. Fast -- ~0.5 sec/photo.

### Alternative: All stages at once

```bash
photo-search index
```

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
- **Deterministic Qdrant IDs**: MD5 hash of file_path ensures re-indexing overwrites the same point
- **NUL byte sanitization**: Some EXIF strings contain \x00 bytes that Postgres rejects -- stripped on insert
- **HTTPS ingress for Qdrant**: Uses REST API through nginx ingress with TLS verify disabled (internal CA)
- **Images resized to max 1536px** before sending to VLM to avoid context overflow
