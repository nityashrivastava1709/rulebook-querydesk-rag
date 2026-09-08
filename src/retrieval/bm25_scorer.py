"""
BM25 keyword/lexical scorer. Complements semantic embeddings: embeddings can
blur specific numbers or exact clause references (e.g. '75%' vs '65%') into
"roughly similar", while BM25 catches literal term overlap with high
precision. In a rulebook full of specific thresholds and clause numbers,
that precision matters.
"""

import os
import re
import pickle
from rank_bm25 import BM25Okapi

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BM25_PATH = os.path.join(_PROJECT_ROOT, "data", "bm25_index.pkl")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9%]+", text.lower())


def build_index(chunks: list[dict]) -> None:
    """Builds a BM25 index over chunk texts and persists it (with the
    aligned chunk_id list) to disk."""
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    chunk_ids = [c["chunk_id"] for c in chunks]

    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)


_bm25 = None
_chunk_ids = None


def _load():
    global _bm25, _chunk_ids
    if _bm25 is None:
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _chunk_ids = data["chunk_ids"]
    return _bm25, _chunk_ids


def score_all(question: str) -> dict:
    """Returns {chunk_id: bm25_score} for every chunk in the corpus."""
    bm25, chunk_ids = _load()
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)
    return dict(zip(chunk_ids, scores))
