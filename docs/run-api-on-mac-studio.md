# Move the photo-search API (and optionally Ollama) to a Mac Studio (keep everything else in k8s)

> Status: **not started — pick this up when you have time to test.**
> Current prod: API runs in k8s on Raspberry Pi nodes. It works but is sluggish
> on cold scrolls because HEIC decode on an ARM Cortex-A76 is 5–8× slower per
> image than on an Apple M-series. DB, Qdrant, Ollama, and the web bundle are
> all fine on the Pis.

## Goal

Run `postgres`, `qdrant`, and the React/nginx `web` chart in k8s on the Pi
cluster. Run the **FastAPI `api`** natively on a Mac Studio — and, once the
Mac Studio is on the Pi VLAN, optionally run **Ollama** there too (bigger
win: it's the only place `qwen2.5vl:7b` is actually practical, because the
Pis have no GPU). Keep all Helm charts intact and deployable so "run anywhere"
still works: `kubectl scale deploy/photosearch-api --replicas=1` or
`helm upgrade --install photosearch-api infra/api -n ai-photos` should revive
the in-cluster API untouched.

## Why

- **CPU (API):** 12 MP HEIC decode is ~600–900 ms on a Pi 5, ~80–150 ms on M2/M3.
- **Memory (API):** Mac Studio has no meaningful cap for this workload; `_DECODE_SEMAPHORE` can be raised to 12–16.
- **Thumb cache:** Local SSD instead of NFS-backed RWO PVC — big win for cache-hit latency.
- **GPU (Ollama):** `qwen2.5vl:7b` (VLM used for `captioner.provider: ollama` during indexing) is effectively unusable on a Pi — needs Metal/CUDA and unified memory. Mac Studio makes the "local VLM vs Anthropic" captioner choice a real one.
- The API is stateless and Ollama is stateless (models on disk); moving either is mechanically simple.

### What Ollama actually does (for reference)

1. **Query-time:** every `/api/search` calls `nomic-embed-text` once to vectorize the query string. Small, ~30 ms on a Pi, ~5 ms on M-series. Not a bottleneck; location doesn't matter much.
2. **Index-time:** if `captioner.provider: ollama`, runs `qwen2.5vl:7b` once per photo. Only triggered by the CLI indexer on the Mac. This is where colocating Ollama on the Mac pays off big.

So: colocating Ollama with the API on the Mac is a nice-to-have for query latency and a big deal for index-time VLM flexibility.

## Current cluster facts (for reference when you pick this up)

- Namespace: `ai-photos` on cluster accessed via default kubecontext.
- Services (all ClusterIP today): `postgres:5432`, `qdrant:6333`, `ollama:11434`, `photosearch-api:8000`.
- Photo library is on NFS at `192.168.22.250:/voyager2/Photos`. The Mac already mounts it at `/Volumes/voyager2/Photos` (that's where you indexed from — Postgres stores those exact paths as primary keys).
- The traversal-alias fix in [photo-search/api/paths.py](../photo-search/api/paths.py) + `photos.source_dir_aliases` in [photo_search/config.py](../photo-search/photo_search/config.py) lets the in-cluster API translate `/Volumes/voyager2/Photos/...` → `/photos/...`. On the Mac that alias is unused (paths match natively), but the code stays — the chart still sets it via [infra/api/values.yaml](../infra/api/values.yaml) for k8s fallback.
- Frontend bundle has `VITE_API_BASE=https://api.photos.k8s.blacktoaster.com` baked in. CORS on the API already allows `https://photos.k8s.blacktoaster.com`.
- Last good image tag: `35948b1-thumbfix3`. Helm release `photosearch-api` at revision ≥ 9 with `resources.limits = 3500m CPU / 6Gi mem`, `requests = 500m / 1Gi`.

## Plan (in order)

### 1. Expose the three backends outside the cluster

They're ClusterIP. Flip each to NodePort (or add NodePort siblings). Bake
this into the chart values when you're ready to make it permanent; for a
quick test, patch live:

```sh
kubectl -n ai-photos patch svc postgres -p '{"spec":{"type":"NodePort"}}'
kubectl -n ai-photos patch svc qdrant   -p '{"spec":{"type":"NodePort"}}'
kubectl -n ai-photos patch svc ollama   -p '{"spec":{"type":"NodePort"}}'
kubectl -n ai-photos get svc -o wide    # note the :3xxxx ports
```

Or, zero-touch, start with `kubectl port-forward` from the Mac:

```sh
kubectl -n ai-photos port-forward svc/postgres 5432:5432 &
kubectl -n ai-photos port-forward svc/qdrant   6333:6333 &
kubectl -n ai-photos port-forward svc/ollama   11434:11434 &
```

Port-forward is fine for a first smoke test but dies on laptop sleep; use
NodePort for a steady-state Mac Studio setup.

### 2. Scale the in-cluster API to zero (keep Helm release + PVCs)

```sh
kubectl -n ai-photos scale deploy/photosearch-api --replicas=0
```

Don't `helm uninstall` — you want the release, ConfigMap, thumb PVC, and
thumb-PV all to stick around so the fallback is `scale --replicas=1` away.
The NFS export at `/k8s/ai-photos/thumbs` survives regardless.

### 3. Run the API natively on the Mac

Prereqs on the Mac:
- `/Volumes/voyager2/Photos` mounted read-write (already is).
- Python 3.12 venv at `photo-search/.venv` with deps installed (already is — `pillow-heif`, `fastapi`, `uvicorn`, etc.).
- Network to the Pi cluster for postgres/qdrant/ollama.

Env the API needs (either via shell or `~/Library/LaunchAgents/...plist`):

```sh
export PHOTO_SEARCH_POSTGRES__CONNECTION_STRING="postgresql://photouser:${POSTGRES_PASSWORD}@<pi-nodeport-host>:<nodeport>/photosearch"
export PHOTO_SEARCH_QDRANT__URL="http://<pi-nodeport-host>:<nodeport>"
export PHOTO_SEARCH_OLLAMA__BASE_URL="http://<pi-nodeport-host>:<nodeport>"
export PHOTO_SEARCH_THUMB_CACHE="$HOME/Library/Caches/photo-search-thumbs"
export PHOTO_SEARCH_DECODE_CONCURRENCY=16
export PHOTO_SEARCH_CORS_ORIGINS="https://photos.k8s.blacktoaster.com"
# Only needed if captioning from the Mac instance (usually not — indexing job stays on the Pi or CLI).
export ANTHROPIC_API_KEY="..."  # optional
```

And a `config.yaml` on disk (or use the existing one from the repo with
overrides):

```yaml
photos:
  source_dir: /Volumes/voyager2/Photos
  source_dir_aliases: []          # unused on the Mac
  supported_extensions: [.heic, .jpg, .jpeg, .png, .tiff]
  skip_extensions: [.mov, .mp4, .aac, .m4a, .aae, .gif]
```

Start it:

```sh
cd ~/_code/ai-photos/photo-search
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. Make it persistent (launchd)

Write `~/Library/LaunchAgents/com.michael.photo-search-api.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>            <string>com.michael.photo-search-api</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/michael/_code/ai-photos/photo-search/.venv/bin/uvicorn</string>
    <string>api:app</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>8000</string>
  </array>
  <key>WorkingDirectory</key> <string>/Users/michael/_code/ai-photos/photo-search</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PHOTO_SEARCH_POSTGRES__CONNECTION_STRING</key><string>postgresql://photouser:PASS@PI:PORT/photosearch</string>
    <key>PHOTO_SEARCH_QDRANT__URL</key>                <string>http://PI:PORT</string>
    <key>PHOTO_SEARCH_OLLAMA__BASE_URL</key>           <string>http://PI:PORT</string>
    <key>PHOTO_SEARCH_THUMB_CACHE</key>                <string>/Users/michael/Library/Caches/photo-search-thumbs</string>
    <key>PHOTO_SEARCH_DECODE_CONCURRENCY</key>         <string>16</string>
    <key>PHOTO_SEARCH_CORS_ORIGINS</key>               <string>https://photos.k8s.blacktoaster.com</string>
  </dict>
  <key>RunAtLoad</key>        <true/>
  <key>KeepAlive</key>        <true/>
  <key>StandardOutPath</key>  <string>/Users/michael/Library/Logs/photo-search-api.out.log</string>
  <key>StandardErrorPath</key><string>/Users/michael/Library/Logs/photo-search-api.err.log</string>
</dict>
</plist>
```

Load it:

```sh
launchctl load -w ~/Library/LaunchAgents/com.michael.photo-search-api.plist
launchctl list | grep photo-search
tail -f ~/Library/Logs/photo-search-api.out.log
```

System Preferences → Battery/Energy Saver: "Prevent automatic sleeping when
display is off" + "Start up automatically after a power failure". Needed
because launchd won't help if macOS is asleep.

### 5. Point the frontend at the Mac's API

Two flavors, pick one:

**a. DNS swing (zero rebuild).** Add an A record (or split-horizon DNS via
your router) so `api.photos.k8s.blacktoaster.com` resolves to the Mac's LAN
IP. Terminate TLS on the Mac (caddy is two lines; or reverse-proxy through
the nginx already running on the web pod if you keep TLS there and route
`/api` upstream to the Mac — more moving parts). The baked-in `VITE_API_BASE`
keeps working; CORS already allows the web origin.

**b. Second hostname (rebuild bundle).** Set `VITE_API_BASE=https://api-local.photos.k8s.blacktoaster.com` (or `http://mac-studio.lan:8000`), rebuild + redeploy the web chart. Lets Pi-API and Mac-API coexist while you A/B.

### 6. Verify

```sh
# API health (local)
curl -s http://127.0.0.1:8000/api/health

# End-to-end thumb (production origin)
TOKEN=$(curl -s "https://api.photos.k8s.blacktoaster.com/api/search?q=tree&limit=1" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['results'][0]['path_token'])")
curl -sI -H "Origin: https://photos.k8s.blacktoaster.com" \
  "https://api.photos.k8s.blacktoaster.com/thumbs/$TOKEN?size=600" | head -5
# expect HTTP/2 200, content-type: image/jpeg

# Browser smoke: hard-reload https://photos.k8s.blacktoaster.com/, scroll a grid.
# Subjectively: cold-cache thumbs should feel ~5× snappier than the Pi-backed version.

# Regression: traversal attack still 403
TOK=$(printf /etc/passwd | base64 | tr '+/' '-_' | tr -d '=')
curl -sI "https://api.photos.k8s.blacktoaster.com/thumbs/$TOK" | head -3
```

### 7. Rollback

If anything misbehaves:

```sh
launchctl unload -w ~/Library/LaunchAgents/com.michael.photo-search-api.plist
# (undo DNS change if you did DNS swing)
kubectl -n ai-photos scale deploy/photosearch-api --replicas=1
kubectl -n ai-photos rollout status deploy/photosearch-api
```

The in-cluster API comes back to exactly its current (working, sluggish)
state.

## Also move Ollama? (optional extension)

Once the Mac Studio is on the Pi VLAN, running Ollama natively there is
worth considering:

1. `brew install ollama` (or run the native app). Launch at login.
2. `ollama pull nomic-embed-text` + (optional) `ollama pull qwen2.5vl:7b`.
3. Point the API at it: `PHOTO_SEARCH_OLLAMA__BASE_URL=http://localhost:11434`.
4. Scale the cluster Ollama to 0: `kubectl -n ai-photos scale deploy/ollama --replicas=0`. Keep the Helm release + PVC as a fallback.
5. If indexing runs from the same Mac with `captioner.provider: ollama`, point the indexer at the same local endpoint — no LAN hop, fast VLM inference.

The cluster Ollama chart stays in the repo and can be revived at any time
(same pattern as the API).

## Open decisions (resolve when you pick this up)

- **DNS swing vs second hostname?** The former is transparent to users but requires TLS termination on the Mac. The latter keeps TLS on the cluster ingress but requires a bundle rebuild.
- **Keep in-cluster API at `replicas=0` vs `helm uninstall`?** `replicas=0` is cheaper to revive, but leaves a dormant ReplicaSet and PVCs. I'd leave it at 0.
- **Are NodePorts acceptable on your home LAN, or do you want Ingress/LB for postgres?** Postgres exposure matters; it's trivially fingerprinted. If the Mac and cluster share a trusted LAN, NodePort is fine; otherwise consider `PHOTO_SEARCH_POSTGRES__CONNECTION_STRING=…?sslmode=require` and a TLS cert, or restrict the NodePort via `spec.loadBalancerSourceRanges` equivalent / firewall.
- **Move Ollama to the Mac too?** Recommend yes once it's on the VLAN. Biggest concrete win is making `qwen2.5vl:7b` usable for VLM captioning. Query-time `nomic-embed-text` gains are marginal but nice.
- **Captioner provider after the move?** With a beefy local Ollama, you could flip `captioner.provider: ollama` to avoid Anthropic costs on indexing. Trade-off: caption quality vs cost. Easy to A/B by running one indexing job each way on a small sample.
- **Should indexing also move to the Mac?** It already runs from the Mac via the CLI; nothing changes. Captioner with Anthropic works identically from either side.

## Files touched vs untouched

No code changes required for this migration — the alias-translation work is
already in place and harmless when unused. Only **deployment surface** moves:

- Untouched: all Helm charts under [infra/](../infra), all code under [photo-search/](../photo-search).
- Patched (reversible): Service types on `postgres`, `qdrant`, `ollama` (ClusterIP → NodePort).
- Scaled: `photosearch-api` Deployment replicas 1 → 0.
- New on the Mac: launchd plist + local `config.yaml` + env.
- Maybe new: DNS record / TLS cert for `api.photos.k8s.blacktoaster.com` pointing at the Mac.

## Pick-up prompt for Claude

When you resume this, paste the following at the start of a new session:

> I want to move the photo-search API off the Raspberry Pi k8s cluster and
> run it natively on my Mac Studio, keeping postgres/qdrant/ollama/web in
> k8s on the Pis. The plan is in
> [docs/run-api-on-mac-studio.md](docs/run-api-on-mac-studio.md). Help me
> execute it step by step, starting with exposing the backends and running
> a smoke test via `kubectl port-forward` before committing to NodePorts
> and launchd. Current cluster state: API is `replicas=1` on image
> `35948b1-thumbfix3`, working but sluggish; frontend at
> `https://photos.k8s.blacktoaster.com` points at
> `https://api.photos.k8s.blacktoaster.com`.
