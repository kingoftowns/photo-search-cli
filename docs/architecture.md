# ai-photos — How it works

A self-hosted photo search tool. The goal: drop ~30 years of family photos onto
an NFS share, have a vision-language model describe each one, recognize who's
in each shot, and then let you search the library in plain English from a
browser (*"Eva blowing out birthday candles, 2021"*, *"Italy sunset 2023"*).

This doc is the "explain it to another engineer in 15 minutes" version. For
ops/install see [infra/README.md](../infra/README.md). For the Mac-Studio
deployment option see [docs/run-api-on-mac-studio.md](run-api-on-mac-studio.md).

## TL;DR

```
Photos on NFS ──► Indexer (offline) ──► Postgres (metadata + faces)
                                   └──► Qdrant (vector per photo)
                                                    ▲
                                                    │  query embedding
Browser ──► React/Vite SPA ──► FastAPI ─────────────┘
                                         └─► Postgres (metadata)
                                         └─► NFS       (thumbnails + originals)
```

- **NFS** is the source of truth for the photo bytes. Nothing writes to it.
- **Indexer** is a Python CLI (`photo-search/scripts/`) you run offline. It
  walks the NFS tree, processes each file through a 4-stage pipeline, and
  writes results to Postgres + Qdrant.
- **Postgres** holds relational data: per-photo metadata, per-face rows, known
  identities, and per-file pipeline status (so re-runs are resumable).
- **Qdrant** holds one 768-dim vector per photo plus a small payload used for
  metadata filtering (people, year, location).
- **FastAPI** is read-only at query time. It embeds the user's query with
  Ollama, asks Qdrant for nearest neighbors, hydrates metadata from Postgres,
  and serves thumbnails/originals off NFS via signed (base64-url) path tokens.
- **React SPA** is a static bundle served by an nginx sidecar. The API origin
  is baked in at build time (`VITE_API_BASE`).

## Components

| Chart (release) | Service | Image | Role |
|---|---|---|---|
| `infra/postgres` | `postgres:5432` | official | Metadata DB |
| `infra/qdrant` | `qdrant:6333` | qdrant/qdrant | Vector DB |
| `infra/ollama` | `ollama:11434` | ollama/ollama | Embeddings (+ optional VLM) |
| `infra/api` | `api:8000` | custom (FastAPI/uvicorn) | Backend |
| `infra/web` | `web:80` | custom (React + nginx) | Frontend |

Five NFS exports back the cluster:
`/voyager2/Photos` (the library, RO), and four RW exports under
`/k8s/ai-photos/{postgres,qdrant,ollama,thumbs}`.

## Ingest pipeline

Entrypoint: `photo-search/photo_search/pipeline.py`. A run does
`scan → filter to pending → process_photo() per file → write status`.

Each file moves through four **stages**, and each stage flips a boolean in
the `indexing_status` table in Postgres:

```
exif → faces → caption → embed
```

Every stage is idempotent at the file level. If a run crashes or is
`^C`-ed, the next run picks up whichever stage has `*_extracted = FALSE`
for each file. You can also re-run a single stage for everything (or just
files with errors) without redoing the earlier ones.

### Stage 1 — EXIF (`metadata.py`)

Opens the file with Pillow + `pillow-heif`. Extracts `file_size_bytes`,
`width`/`height`, `date_taken`, GPS lat/lon, `camera`, and a few optional
bits (focal length, aperture, ISO, orientation). These populate a
`PhotoMetadata` pydantic model.

Cheapest stage by far — it's just header reads.

### Stage 2 — Faces (`faces.py`)

Two components:

- **`FaceDetector`** — wraps InsightFace's `FaceAnalysis` with the
  `buffalo_l` model pack running on CPU. For each photo it returns a list
  of `DetectedFace(bbox, confidence, embedding[512])`.
  The 512-dim vector is an **ArcFace embedding** — a compact identity
  fingerprint.
- **`FaceClassifier`** — holds a dict of known identities
  (`label → centroid[512]`). For each detected face it computes cosine
  similarity against every centroid and picks the best match, or labels
  the face `"unknown"` if no score clears `similarity_threshold` (0.4 by
  default).

Identities are built separately by the `label-faces` CLI: you tag a small
set of faces with a name (e.g. `michael`, `eva`), the CLI averages their
ArcFace embeddings into a unit-normalized **centroid** via
`compute_centroid`, and stores it in `face_identities`. Running
`label-faces` again merges new samples into the existing centroid with a
sample-count-weighted average (see
[storage.py:389-450](../photo-search/photo_search/storage.py#L389-L450)) so
coverage improves over time without resetting.

Both detected and identified face records are written to `photo_faces`
(one row per face per photo) — bbox, confidence, similarity, label, and
the raw 512-dim embedding (as binary). Persisting the embedding is what
makes re-labeling possible without re-running detection.

### Stage 3 — Caption (`caption.py`)

A pluggable vision-language model describes the photo in a paragraph
aimed at a search index — who's in it, the setting, the activity,
visible text, the mood. See `CAPTION_PROMPT`.

The captioner is picked by config (`captioner.provider`):

- **`AnthropicCaptioner`** — Claude API (`claude-haiku-4-5` by default).
  Image is base64'd into the message. Fast, consistent, costs a few
  cents per 100 photos. API key lives in the `photosearch-anthropic`
  Secret in-cluster.
- **`OllamaCaptioner`** — local inference via Ollama. `qwen2.5vl:7b` /
  `qwen3-vl:8b` are the target VLMs. No external dependency, but on the
  Pi cluster without a GPU these are effectively unusable (minutes per
  photo). On a Mac with Metal acceleration they're practical.

Both share a preprocessing step (`BaseCaptioner._resize_image`): HEIC is
always transcoded to JPEG (no VLM reads HEIC natively) and images are
resized so the long edge fits within `pipeline.resize_max_dimension`
(1536 px). Output: a `PhotoCaption(caption, model, generation_time)`.

The choice is a cost / privacy / speed tradeoff. Today the cluster is
wired to Anthropic; the `create_captioner()` factory makes swapping
providers a single-value config change.

### Stage 4 — Embed (`embed.py` + geocoding + writes)

This is where everything comes together into one searchable vector. The
pipeline:

1. **Reverse-geocode** the GPS coordinates into a `location_name` like
   `"Woodcrest, California, US"` (see `photo_search/geo.py`). Cached
   per-photo in `photos.location_name` so subsequent runs don't re-query.
2. **Rehydrate** from Postgres on resumed runs — if the caption/faces
   stages ran in an earlier invocation, pull them back so the final
   record is complete.
3. **Compose a multi-line search text** — `TextEmbedder.build_search_text`
   concatenates:
   ```
   <caption>
   People: alice, bob
   Location: Woodcrest, California, US
   Date: June 12, 2023
   Camera: iPhone 13 Pro
   ```
   This is deliberately denormalized: a query like *"bob in California"*
   has to match a single embedding, and cramming every facet into one
   text gives the vector enough signal to do that without extra filters.
4. **Embed** the combined text via Ollama's `nomic-embed-text` model →
   768-dim float vector.
5. **Write both stores** (see `pipeline.py` finalize step):
   - `PostgresStorage.upsert_photo` — one row per photo in `photos`, plus
     face junction rows. Uses `COALESCE(EXCLUDED.caption, photos.caption)`
     so an embed-only rerun doesn't null out earlier captions.
   - `QdrantStorage.upsert_photo` — one point with:
     - **id**: deterministic uint64 from `MD5(file_path)` so re-indexing
       the same file overwrites (no orphan points).
     - **vector**: the 768-dim nomic embedding.
     - **payload**: a handful of filter-supporting fields — `file_path`,
       `caption`, `date_taken`, `year`, `gps_lat/lon`, `location_name`,
       split out `city`/`region`/`country_code`, `faces: [label, ...]`,
       `file_type`, `width`/`height`.

Qdrant's payload indexes (keyword for `faces`/`city`/`country_code`,
integer for `year`, datetime for `date_taken`) are ensured on every API
startup — without them filter queries silently return zero hits because
Qdrant's `MatchValue` against an unindexed array field doesn't match.

### Pipeline mechanics

- `ThreadPoolExecutor(max_workers=pipeline.concurrency)` runs photos in
  parallel. Each thread gets its own Postgres connection (see the
  `threading.local` pattern in `PostgresStorage`) because psycopg2
  connections aren't thread-safe.
- Per-stage status is persisted after each stage, so a crash loses at
  most one photo's worth of progress.
- The CLI supports `--stage`, `--errors-only`, `--filter`, `--dry-run`,
  and SIGINT graceful shutdown (finishes the in-flight photo, writes its
  status, then exits).

## Query path

Everything in the API is read-only and small:
`photo-search/api/` — four route files and a thumbnails cache.

### Text search — `GET /api/search?q=...`

1. FastAPI parses filters (`person`, `city`, `region`, `country_code`,
   `year`, `after`, `before`).
2. If `q` is non-empty, embed it with the same `TextEmbedder` that was
   used for indexing. *Same embedder → same vector space → meaningful
   cosine distances.*
3. `QdrantStorage.search(vec, limit=top, filters=...)` translates filters
   into a Qdrant `Filter(must=[...])`. Person filters with multiple names
   become multiple `MatchValue` conditions (AND semantics — photo must
   contain all named people).
4. If `q` is empty and filters exist, fall back to `QdrantStorage.browse`
   which uses `scroll` with `order_by: date_taken DESC` — nice for
   "everything from 2023" without a text query.
5. Each Qdrant hit is a `SearchResult` whose payload already contains
   enough to render a card (caption preview, people, date, camera,
   location). No Postgres round-trip needed for search — we stuffed the
   displayable fields into Qdrant on purpose.

### Rendering images — path tokens

Returning absolute paths to the browser would be a PII/security mess and
a portability problem (paths differ between the indexing host and the
API pod). Instead, each result carries a **path token** — a
`base64url(file_path)` — and the UI renders:

```html
<img src="https://api.photos.k8s.blacktoaster.com/thumbs/<token>?size=600">
```

On the API:

- `api/paths.py:resolve_safe(request, token)` base64-decodes the token,
  translates any known alias prefix (see **Deployment portability** below),
  calls `Path.resolve()`, and verifies the result is under
  `photos_root`. Fails closed with 403 on any traversal attempt.
- `api/routes.py:thumb` hands the resolved path to `ThumbnailCache`,
  which mmaps a cached JPEG from `/var/cache/photo-thumbs/<hash>/<size>.jpg`
  or generates one on miss. HEIC decodes happen here — in-process,
  gated by a `threading.Semaphore` so simultaneous cold-start decodes
  can't blow past the memory limit.
- `api/routes.py:original` serves the raw file unchanged (JPEG/PNG) or
  transcodes HEIC → JPEG on the fly, since browsers don't do HEIC.

The thumbnail cache sits on a dedicated NFS export
(`/k8s/ai-photos/thumbs`, mounted RWO) so it survives pod restarts —
without that, every rollout caused a wave of HEIC decodes that OOMKilled
the pod.

### Other API endpoints

- `GET /api/photos/{token}` → full metadata for one photo (Postgres
  lookup — the source of truth, not Qdrant).
- `GET /api/faces` → list of known identities for the people filter
  chip in the UI.
- `GET /api/locations?prefix=...` → autocomplete from distinct
  `photos.location_name` values.
- `GET /api/status` → counts per pipeline stage + Qdrant vector count.
  Used by a tiny admin view.

## Data shapes, quick reference

### Postgres (relational — `photo-search/scripts/init_db.sql`)

| Table | Purpose |
|---|---|
| `photos` | One row per file (keyed by absolute `file_path`). Caption, metadata, location, models used. |
| `photo_faces` | One row per detected face per photo. Keeps the 512-dim ArcFace embedding so re-labeling is cheap. |
| `face_identities` | Known people — label, display name, centroid, sample count. |
| `indexing_status` | Per-file stage flags. Drives resumability. |

### Qdrant (vector)

- One collection, `photos`. Cosine distance, 768-dim.
- Point id = deterministic hash of file path → upsert = overwrite.
- Payload: caption, date_taken/year, gps, split location, faces[], file
  type, dimensions.
- Payload indexes on `faces`, `year`, `file_type`, `date_taken`, `city`,
  `region`, `country_code`.

### Pydantic flow types (`photo_search/models.py`)

`PhotoMetadata` → `DetectedFace` → `IdentifiedFace` → `PhotoCaption` →
`IndexedPhoto` (the fully-assembled record written to both stores) →
`SearchResult` (what the API returns to the SPA).

## Deployment portability

The trickiest real-world wrinkle: indexing can run on a workstation
(Mac), but the API runs in k8s — so `file_path`s stored in Postgres/Qdrant
are absolute Mac paths (`/Volumes/voyager2/Photos/IMG_1234.HEIC`) while
the API pod sees them at `/photos/IMG_1234.HEIC`. Reindexing to rewrite
paths is a non-starter; instead `resolve_safe` holds a list of
`source_dir_aliases` (wired through `infra/api/values.yaml`) and rewrites
any decoded token that starts with an alias prefix to `photos_root`
*before* the traversal check runs. Same file, two mount points, one
code path — and the `Path.resolve().relative_to(root)` still enforces
traversal safety after the rewrite.

Same idea supports the "run the API on a Mac Studio" option documented
in [run-api-on-mac-studio.md](run-api-on-mac-studio.md): flip
`replicaCount` to 0 in `infra/api/values.yaml`, run the FastAPI app
natively (M-series HEIC decoding is ~5–8× faster than on the Pis),
and everything else — Postgres, Qdrant, optionally Ollama — keeps
running in the cluster.

## What's deliberately not here

- **No in-cluster indexing Job.** Indexing is CPU-heavy and we don't
  want it on the Pis. Runs on a workstation against cluster Postgres
  (kube-DNS) and Qdrant (public ingress).
- **No auth.** Public HTTPS + CORS-limited origins, but anyone with the
  URL can search. Behind a Cloudflare Access or Tailscale gate at the
  network layer if that matters.
- **No video.** `.mov`/`.mp4`/`.gif` are explicitly in `skip_extensions`.
  Could plug in a separate keyframe → same pipeline flow later.
- **No face clustering UI.** The `label-faces` CLI is how new identities
  get added; there's no in-app "group these unknown faces and name them"
  yet. All the data (per-face embeddings in `photo_faces`) is present —
  it's just a missing frontend feature.

## Key files to trace through

- [photo-search/photo_search/pipeline.py](../photo-search/photo_search/pipeline.py) — pipeline orchestration, stage dispatch, resume logic.
- [photo-search/photo_search/faces.py](../photo-search/photo_search/faces.py) — detector + classifier + centroid math.
- [photo-search/photo_search/caption.py](../photo-search/photo_search/caption.py) — provider abstraction, resize/transcode, `create_captioner`.
- [photo-search/photo_search/embed.py](../photo-search/photo_search/embed.py) — text composition + Ollama embed.
- [photo-search/photo_search/storage.py](../photo-search/photo_search/storage.py) — Postgres + Qdrant wrappers, payload schema, filter builder.
- [photo-search/photo_search/models.py](../photo-search/photo_search/models.py) — all flow types in ~100 lines.
- [photo-search/api/routes.py](../photo-search/api/routes.py) — every HTTP endpoint.
- [photo-search/api/paths.py](../photo-search/api/paths.py) — token encode/decode + alias-aware traversal safety.
- [photo-search/api/thumbnails.py](../photo-search/api/thumbnails.py) — disk-cached thumb generation + decode semaphore.
