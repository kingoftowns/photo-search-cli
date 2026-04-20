"""Diagnose the nomic-embed-text task-prefix hypothesis.

Pulls one stored vector from Qdrant (a point whose caption mentions a
keyword we know should match), then compares cosine similarity of that
vector against several query embeddings:

    1. raw "<query>"                         (current behavior)
    2. "search_query: <query>"               (proposed query prefix)

Also re-embeds the photo's caption-derived text two ways:

    3. raw combined text                     (current stored-vector recipe)
    4. "search_document: <combined text>"    (proposed document prefix)

Then computes the full 4x4 cosine matrix so we can see whether prefixing
both sides substantially lifts the query's score.

Read-only. Requires:
    * config.yaml (for qdrant.url, ollama.base_url, embedding_model)
    * local Ollama with nomic-embed-text pulled
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

from photo_search.config import load_config
from photo_search.embed import TextEmbedder


KEYWORD = "beavers"  # substring to look for in caption (case-insensitive)
QUERY_TERMS = ["beavers", "baseball"]


def _connect_qdrant(url: str) -> QdrantClient:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme == "https" and not parsed.port:
        return QdrantClient(
            host=parsed.hostname,
            port=443,
            https=True,
            prefer_grpc=False,
            verify=False,
            check_compatibility=False,
        )
    return QdrantClient(url=url, prefer_grpc=False)


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def main() -> None:
    cfg = load_config()
    print(f"Qdrant:  {cfg.qdrant.url}")
    print(f"Ollama:  {cfg.ollama.base_url}  model={cfg.ollama.embedding_model}")
    print()

    qc = _connect_qdrant(cfg.qdrant.url)
    # Substring scan via scroll (caption isn't full-text indexed).
    points = []
    next_offset = None
    scanned = 0
    print(f"Scanning Qdrant for a point with '{KEYWORD}' in caption...")
    while scanned < 20000 and not points:
        batch, next_offset = qc.scroll(
            collection_name=cfg.qdrant.collection_name,
            offset=next_offset,
            limit=500,
            with_payload=True,
            with_vectors=True,
        )
        if not batch:
            break
        for pt in batch:
            cap = (pt.payload or {}).get("caption") or ""
            if KEYWORD.lower() in cap.lower():
                points = [pt]
                break
        scanned += len(batch)
        if next_offset is None:
            break
    print(f"Scanned {scanned} points.")

    if not points:
        print(f"No point found with '{KEYWORD}' in caption. Aborting.")
        return

    pt = points[0]
    payload = pt.payload or {}
    stored_vec = list(pt.vector) if pt.vector else []
    caption = payload.get("caption") or ""
    people = payload.get("faces") or []
    location = payload.get("location_name")
    date_taken = payload.get("date_taken")
    camera = payload.get("camera")

    print("Found point:")
    print(f"  file_path : {payload.get('file_path')}")
    print(f"  date      : {date_taken}")
    print(f"  people    : {people}")
    print(f"  location  : {location}")
    print(f"  caption (first 240 chars):")
    print(f"    {caption[:240]!r}")
    print(f"  stored_vector dim: {len(stored_vec)}")
    print()

    embedder = TextEmbedder(
        base_url=cfg.ollama.base_url,
        model=cfg.ollama.embedding_model,
    )

    # Reconstruct the search text using the same shape that build_search_text
    # produces (faces need to be list[str]; date a datetime; camera a str).
    from datetime import datetime
    dt = None
    if date_taken:
        try:
            dt = datetime.fromisoformat(date_taken)
        except (TypeError, ValueError):
            dt = None
    combined = TextEmbedder.build_search_text(
        caption=caption,
        face_labels=people,
        location=location,
        date_taken=dt,
        camera=camera,
    )

    # --- Embeddings to compare -------------------------------------------------
    print("Embedding reference vectors...")
    doc_noprefix = embedder.embed_text(combined)
    doc_prefix = embedder.embed_text(f"search_document: {combined}")

    query_vecs: dict[str, dict[str, list[float]]] = {}
    for term in QUERY_TERMS:
        query_vecs[term] = {
            "raw": embedder.embed_text(term),
            "prefix": embedder.embed_text(f"search_query: {term}"),
        }

    # --- Cosine scores ---------------------------------------------------------
    def row(label: str, qv: list[float]) -> str:
        s_stored = _cosine(qv, stored_vec)
        s_dnopref = _cosine(qv, doc_noprefix)
        s_dpref = _cosine(qv, doc_prefix)
        return f"  {label:<40s}  stored={s_stored:+.4f}  doc_no_pref={s_dnopref:+.4f}  doc_pref={s_dpref:+.4f}"

    print()
    print("Cosine similarity of each query against three document vectors:")
    print("  - stored       : the vector currently in Qdrant (no prefixes)")
    print("  - doc_no_pref  : the combined text re-embedded now, no prefix")
    print("  - doc_pref     : the combined text re-embedded now with `search_document: `")
    print()
    for term in QUERY_TERMS:
        print(f"[{term!r}]")
        print(row(f"raw query (current behavior)", query_vecs[term]["raw"]))
        print(row(f"search_query: prefix",         query_vecs[term]["prefix"]))
        print()

    # Sanity: doc_no_pref should equal stored_vec if the stored vector was
    # computed from the same text with the same model.
    self_sim = _cosine(doc_noprefix, stored_vec)
    print(f"Sanity check — stored vs. freshly-embedded same recipe: {self_sim:+.4f}")
    print()

    # --- Markdown-stripped comparison -----------------------------------------
    import re
    def strip_md(text: str) -> str:
        # Drop ATX headers ('# ...' / '## ...') and leading hyphens; collapse blanks.
        lines = []
        for line in text.splitlines():
            s = re.sub(r"^#{1,6}\s+", "", line)
            s = re.sub(r"^-\s+", "", s)
            lines.append(s)
        cleaned = "\n".join(lines)
        return re.sub(r"\n{2,}", "\n", cleaned).strip()

    stripped_caption = strip_md(caption)
    combined_stripped = TextEmbedder.build_search_text(
        caption=stripped_caption,
        face_labels=people,
        location=location,
        date_taken=dt,
        camera=camera,
    )
    doc_stripped_pref = embedder.embed_text(f"search_document: {combined_stripped}")
    print("Stripped-markdown caption (first 240 chars):")
    print(f"  {stripped_caption[:240]!r}")
    print()
    print("Scores against markdown-stripped + `search_document:` prefixed vector:")
    for term in QUERY_TERMS:
        s_raw = _cosine(query_vecs[term]["raw"], doc_stripped_pref)
        s_pref = _cosine(query_vecs[term]["prefix"], doc_stripped_pref)
        print(f"  [{term!r}]  raw_query={s_raw:+.4f}  prefixed_query={s_pref:+.4f}")
    print()

    # --- Rank check ------------------------------------------------------------
    # Where does this photo currently rank for "beavers" in the live index?
    this_point_id = pt.id
    for term in QUERY_TERMS:
        hits = qc.query_points(
            collection_name=cfg.qdrant.collection_name,
            query=query_vecs[term]["raw"],
            limit=60,
            with_payload=True,
        ).points
        try:
            rank = next(i for i, h in enumerate(hits, 1) if h.id == this_point_id)
            print(f"[{term!r}]  this photo currently ranks #{rank} of top-60 "
                  f"(score={hits[rank-1].score:+.4f}); top score={hits[0].score:+.4f}")
        except StopIteration:
            print(f"[{term!r}]  this photo is NOT in top 60 "
                  f"(top score={hits[0].score:+.4f}, floor={hits[-1].score:+.4f})")
        # Show the top 5 results for visual inspection.
        print(f"       top 5 for '{term}':")
        for i, h in enumerate(hits[:5], 1):
            p = h.payload or {}
            cap_snippet = (p.get("caption") or "")[:140].replace("\n", " ")
            has_kw = " ← contains 'beavers'" if "beavers" in (p.get("caption") or "").lower() else ""
            print(f"         #{i} {h.score:+.4f} {p.get('file_name', '?')}{has_kw}")
            print(f"             {cap_snippet!r}")


if __name__ == "__main__":
    main()
