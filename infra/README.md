# ai-photos — Helm charts

Everything runs in the `ai-photos` namespace. Five live charts:

| Chart dir | Release name (live) | Service | What |
|---|---|---|---|
| [postgres/](postgres/) | `postgres` ⚠️ | `postgres:5432` | PostgreSQL metadata DB |
| [qdrant/](qdrant/)     | `qdrant` ⚠️    | `qdrant:6333`  | Vector DB |
| [ollama/](ollama/)     | `photosearch-ollama` | `ollama:11434` | Embeddings + optional VLM |
| [api/](api/)           | `photosearch-api`    | `api:8000`     | FastAPI backend |
| [web/](web/)           | `photosearch-web`    | `web:80`       | React frontend + nginx |

⚠️ `postgres` and `qdrant` were initially deployed with short release names;
the others follow the `photosearch-<chart>` convention. Either is fine; just
don't mix them when running `helm upgrade` against an existing release or
you'll create a second release instead of upgrading the first.

The stale scaffolding dirs `photo-api/`, `photo-web/`, `photos-pv/` are
**not deployed** — they only contain `values.yaml.example`. Ignore or delete.

Public hosts:

| Host | Chart |
|---|---|
| `photos.k8s.blacktoaster.com` | `web` |
| `api.photos.k8s.blacktoaster.com` | `api` (CORS-restricted to the web host) |
| `qdrant.k8s.blacktoaster.com` | `qdrant` (ingress exposed for off-cluster indexing) |

## Cluster prereqs (one-time)

- ingress-nginx
- cert-manager + a `vault-issuer` ClusterIssuer (referenced by `api`/`web`/`qdrant` Certificate resources)
- external-dns pointed at `in.k8s.blacktoaster.com`
- `nfs-storage` StorageClass
- NFS server `192.168.22.250` with exports for:
  - `/voyager2/Photos` — library, mounted read-only into the api pod
  - `/k8s/ai-photos/postgres`
  - `/k8s/ai-photos/qdrant`
  - `/k8s/ai-photos/ollama`
  - `/k8s/ai-photos/thumbs` — api's persistent thumbnail cache
- Container registry at `registry.k8s.blacktoaster.com`

## Build + push images

Bump `image.tag` in the relevant `values.yaml` after pushing so the
Deployment picks it up on the next `helm upgrade`.

```sh
SHA=$(git rev-parse --short HEAD)

# API (arm64 — the cluster is Raspberry Pi nodes)
docker buildx build --platform linux/arm64 \
  -t registry.k8s.blacktoaster.com/ai-photos/api:$SHA \
  --push photo-search/

# Web bundle (VITE_API_BASE is baked in at build time)
docker buildx build --platform linux/arm64 \
  --build-arg VITE_API_BASE=https://api.photos.k8s.blacktoaster.com \
  -t registry.k8s.blacktoaster.com/ai-photos/web:$SHA \
  --push web/
```

## Pre-create namespace + secrets

Secrets are **not** in this repo. Create them once before the first install:

```sh
kubectl create namespace ai-photos

# Postgres password — must match what the chart's secret template expects.
# The postgres chart (infra/postgres/templates/secret.yaml) generates it
# from values on first install; if you need to set it explicitly:
kubectl -n ai-photos create secret generic photosearch-postgres \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)"

# Anthropic key (only needed when captioner.provider == anthropic)
kubectl -n ai-photos create secret generic photosearch-anthropic \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...
```

## Install

Order matters once — `api` waits for postgres/qdrant/ollama to be ready:

```sh
helm upgrade --install postgres          infra/postgres -n ai-photos
helm upgrade --install qdrant            infra/qdrant   -n ai-photos
helm upgrade --install photosearch-ollama infra/ollama  -n ai-photos
helm upgrade --install photosearch-api   infra/api      -n ai-photos
helm upgrade --install photosearch-web   infra/web      -n ai-photos
```

(Use `photosearch-postgres` / `photosearch-qdrant` as release names instead
if you're installing into a fresh cluster and want the naming consistent.)

Wait for readiness:

```sh
kubectl -n ai-photos rollout status deploy/postgres --timeout=120s
kubectl -n ai-photos rollout status deploy/qdrant   --timeout=120s
kubectl -n ai-photos rollout status deploy/ollama   --timeout=300s    # pulls model on first start
kubectl -n ai-photos rollout status deploy/photosearch-api --timeout=180s
kubectl -n ai-photos rollout status deploy/web      --timeout=120s
```

Pull the embedding model the first time ollama comes up:

```sh
kubectl -n ai-photos exec -it deploy/ollama -- ollama pull nomic-embed-text
```

## Smoke test

```sh
kubectl -n ai-photos get pods
curl -fsSL https://api.photos.k8s.blacktoaster.com/api/health
# → {"status":"ok"}

open https://photos.k8s.blacktoaster.com
```

## Upgrade (code or config change)

1. Build + push a new image with a fresh tag (see above).
2. Bump `image.tag` in [api/values.yaml](api/values.yaml) or [web/values.yaml](web/values.yaml).
3. `helm upgrade --install <release> infra/<chart> -n ai-photos`
4. `kubectl -n ai-photos rollout status deploy/<name>`

The api deployment has `strategy: Recreate` (RWO thumb PVC can't be shared
across pods), so expect a ~10 s blip during upgrade.

## Rollback

```sh
helm -n ai-photos history photosearch-api
helm -n ai-photos rollback photosearch-api <REV>
```

## Indexing

Indexing runs from a workstation against the cluster's Postgres + Qdrant
(Postgres is cluster-internal; Qdrant is reachable via its ingress host).
Not yet an in-cluster Job.

See [../photo-search/README.md](../photo-search/README.md) and the CLI
commands under `photo-search/scripts/`.
