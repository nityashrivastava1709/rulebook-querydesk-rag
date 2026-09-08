"""
Combines embedding-based semantic retrieval with BM25 lexical retrieval into
one ranked list. Both score types are normalized per-query before combining,
since raw cosine similarity and raw BM25 scores live on completely different,
incomparable scales.
"""

from src.retrieval import embedder, bm25_scorer
from src.vectorstore import chroma_client

# Named, tunable constants -- deliberately not inline magic numbers. Semantic
# similarity is weighted higher by default because meaning matters more than
# exact wording for most student questions, but BM25 still gets enough
# weight to surface an exact clause-number or percentage match.
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3

TOP_K_CANDIDATES = 10


def _normalize(scores: dict) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rank(question: str, top_k: int = TOP_K_CANDIDATES) -> list[dict]:
    """Returns the top_k chunks ranked by combined semantic + BM25 score.
    Each result includes both raw sub-scores and the final combined_score.

    IMPORTANT: the semantic (cosine) score is used in its RAW form, not
    per-query min-max normalized. Cosine similarity is already on a
    comparable 0-1 scale across different queries, so normalizing it per
    query would rescale the top candidate of every query up to ~1.0
    regardless of whether it's actually a good match -- which would quietly
    defeat threshold-based "not_covered" abstention entirely. Only BM25
    (which is unbounded and not naturally comparable across queries) gets
    per-query normalization.
    """
    query_vector = embedder.embed_text(question)

    # Semantic candidates from ChromaDB.
    semantic_results = chroma_client.query(query_vector, top_k=top_k)
    semantic_by_id = {r["chunk_id"]: r for r in semantic_results}
    raw_semantic_scores = {cid: r["semantic_score"] for cid, r in semantic_by_id.items()}

    # BM25 scores across the FULL corpus (not just the semantic candidates),
    # so a strong keyword match that embeddings ranked low still surfaces.
    bm25_scores_all = bm25_scorer.score_all(question)
    # Union of candidates: everything the semantic pass found, plus the
    # top BM25 matches, so neither retrieval method's unique finds are lost.
    top_bm25_ids = sorted(bm25_scores_all, key=bm25_scores_all.get, reverse=True)[:top_k]
    candidate_ids = set(semantic_by_id.keys()) | set(top_bm25_ids)

    bm25_scores_subset = {cid: bm25_scores_all.get(cid, 0.0) for cid in candidate_ids}
    norm_bm25 = _normalize(bm25_scores_subset)

    combined = []
    for cid in candidate_ids:
        sem = raw_semantic_scores.get(cid, 0.0)  # raw cosine similarity, NOT renormalized
        bm = norm_bm25.get(cid, 0.0)
        combined_score = SEMANTIC_WEIGHT * sem + BM25_WEIGHT * bm

        if cid in semantic_by_id:
            base = dict(semantic_by_id[cid])
        else:
            # Found only via BM25 -- fetch its metadata via a direct Chroma get.
            base = _fetch_metadata_only(cid)

        base["semantic_score_raw"] = sem
        base["bm25_score_norm"] = bm
        base["combined_score"] = combined_score
        combined.append(base)

    combined.sort(key=lambda r: r["combined_score"], reverse=True)
    return combined[:top_k]


def _fetch_metadata_only(chunk_id: str) -> dict:
    collection = chroma_client.get_collection()
    result = collection.get(ids=[chunk_id])
    if not result["ids"]:
        return {"chunk_id": chunk_id, "text": "", "section_reference": "", "topic_tag": "", "source_file": ""}
    return {
        "chunk_id": chunk_id,
        "text": result["documents"][0],
        "section_reference": result["metadatas"][0]["section_reference"],
        "topic_tag": result["metadatas"][0]["topic_tag"],
        "source_file": result["metadatas"][0]["source_file"],
    }
