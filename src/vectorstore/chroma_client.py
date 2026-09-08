"""
Persistent ChromaDB wrapper. Stores each chunk's embedding, raw text, and
metadata (section_reference, topic_tag, source_file) together as one record,
so a similarity query returns everything needed in one call.

Uses PersistentClient (writes to disk) so the index survives server restarts
and doesn't need to be rebuilt on every API call. The path is always resolved
relative to this file's location -- not the current working directory -- so
ingestion and the API server always agree on where the database lives,
regardless of which directory each is launched from.
"""

import os
import chromadb

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
COLLECTION_NAME = "rulebook_collection"

_client = None
_collection = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        # Explicit cosine space: with cosine distance, similarity = 1 - distance
        # is exact, unlike the default L2 space where that shortcut is wrong.
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def reset_collection():
    """Drops and recreates the collection. Call this at the start of
    ingestion so re-running ingestion doesn't duplicate or stack records."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return _collection


def add_chunks(chunks: list[dict], embeddings) -> None:
    """chunks: list of dicts with chunk_id, text, section_reference,
    topic_tag, source_file. embeddings: 2D array aligned with chunks."""
    collection = get_collection()
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "section_reference": c["section_reference"],
                "topic_tag": c["topic_tag"],
                "source_file": c["source_file"],
            }
            for c in chunks
        ],
    )


def query(query_embedding, top_k: int = 10) -> list[dict]:
    """Returns top_k most similar chunks with text, metadata, and a
    similarity score (1 - cosine_distance, since embeddings are normalized)."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, max(collection.count(), 1)),
    )

    output = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
        similarity = 1 - dist  # chroma default distance is squared L2 on normalized vecs; treat as similarity proxy
        output.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "section_reference": meta["section_reference"],
                "topic_tag": meta["topic_tag"],
                "source_file": meta["source_file"],
                "semantic_score": max(0.0, min(1.0, similarity)),
            }
        )
    return output


def count() -> int:
    return get_collection().count()
