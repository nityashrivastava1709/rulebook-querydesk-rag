"""
Decides, for a given ranked candidate list, whether the response is:
  - not_covered: nothing relevant enough was retrieved -> abstain
  - conflict: top two chunks are same-topic, both confident, AND are a
              documented contradiction pair
  - answered: default, when neither of the above applies

The contradiction lookup is loaded from data/contradictions.json at import
time (effectively "at startup" when the API boots) and matched to actual
chunk_ids via each pair's section_reference -- never hardcoded chunk IDs or
raw text in this file. Re-ingesting the corpus, or adding a 4th contradiction
to the JSON, requires no code change here.
"""

import os
import re
import json

from src.retrieval import hybrid_ranker
from src.vectorstore import chroma_client

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTRADICTIONS_PATH = os.path.join(_PROJECT_ROOT, "data", "contradictions.json")

# Tuned in Part 6 / Step 15 via the threshold-tuning script (see
# tune_threshold.py). Chosen because it produced the best abstention
# accuracy on the 25 unanswerable questions without rejecting genuinely
# answerable ones. Recorded here as a named constant, not a magic number.
NOT_COVERED_THRESHOLD = 0.4


def _normalize_clause_ref(ref: str) -> str:
    """Extracts the leading 'Clause X.Y' token, stripping any parenthetical
    suffix (e.g. 'Clause F.4 (Late Fee Table)' -> 'clause f.4'), so JSON
    references and chunk metadata references compare reliably."""
    match = re.match(r"\s*(Clause\s+[\w\.]+)", ref, re.IGNORECASE)
    token = match.group(1) if match else ref
    return token.strip().lower()


def _build_contradiction_lookup() -> set:
    """Returns a set of frozensets, each containing the two chunk_ids that
    form a documented contradiction pair, resolved dynamically against
    whatever is currently in the vector store."""
    if not os.path.exists(CONTRADICTIONS_PATH):
        return set()

    with open(CONTRADICTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    collection = chroma_client.get_collection()
    try:
        all_records = collection.get()
    except Exception:
        return set()

    ref_to_chunk_id = {}
    for chunk_id, meta in zip(all_records["ids"], all_records["metadatas"]):
        normalized = _normalize_clause_ref(meta["section_reference"])
        ref_to_chunk_id[normalized] = chunk_id

    pairs = set()
    for item in data.get("contradictions", []):
        ref_a = _normalize_clause_ref(item["clause_a"]["reference"])
        ref_b = _normalize_clause_ref(item["clause_b"]["reference"])
        chunk_a = ref_to_chunk_id.get(ref_a)
        chunk_b = ref_to_chunk_id.get(ref_b)
        if chunk_a and chunk_b:
            pairs.add(frozenset({chunk_a, chunk_b}))
    return pairs


_contradiction_lookup = None


def get_contradiction_lookup() -> set:
    global _contradiction_lookup
    if _contradiction_lookup is None:
        _contradiction_lookup = _build_contradiction_lookup()
    return _contradiction_lookup


def reload_contradiction_lookup() -> set:
    """Force-rebuild (e.g. after re-ingestion or editing contradictions.json)."""
    global _contradiction_lookup
    _contradiction_lookup = _build_contradiction_lookup()
    return _contradiction_lookup


# In src/classification/response_classifier.py, replace the conflict-checking
# block inside classify() with this:

def classify(question: str) -> dict:
    ranked = hybrid_ranker.rank(question, top_k=15)  # widen the net

    if not ranked or ranked[0]["combined_score"] < NOT_COVERED_THRESHOLD:
        closest = ranked[0] if ranked else None
        return {
            "type": "not_covered",
            "answer": None,
            "chunks": [closest] if closest else [],
            "reasoning": {
                "top_score": closest["combined_score"] if closest else 0.0,
                "threshold": NOT_COVERED_THRESHOLD,
            },
        }

    lookup = get_contradiction_lookup()

    # Only consider chunks that individually clear the threshold as
    # real candidates for a conflict -- not just the top 2 positions.
    confident_chunks = [c for c in ranked if c["combined_score"] >= NOT_COVERED_THRESHOLD]

    for i in range(len(confident_chunks)):
        for j in range(i + 1, len(confident_chunks)):
            a, b = confident_chunks[i], confident_chunks[j]
            same_topic = a["topic_tag"] == b["topic_tag"]
            is_known_pair = frozenset({a["chunk_id"], b["chunk_id"]}) in lookup
            if same_topic and is_known_pair:
                return {
                    "type": "conflict",
                    "answer": None,
                    "chunks": [a, b],
                    "reasoning": {
                        "score_a": a["combined_score"],
                        "score_b": b["combined_score"],
                        "shared_topic": a["topic_tag"],
                    },
                }

    return {
        "type": "answered",
        "answer": ranked[0]["text"],
        "chunks": [ranked[0]],
        "reasoning": {"top_score": ranked[0]["combined_score"]},
    }