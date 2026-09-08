"""
Run this once to build the searchable index from the corpus:
    python ingest.py

This is intentionally separate from the API server. Ingestion is slow
(embedding everything) and only needs to run when the corpus changes;
the API server should only ever READ from what this script builds.
"""

import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from src.ingestion.loaders import load_corpus
from src.ingestion.chunker import chunk_all
from src.retrieval import embedder, bm25_scorer
from src.vectorstore import chroma_client

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def main():
    print(f"[1/5] Loading corpus from {CORPUS_DIR} ...")
    documents = load_corpus(CORPUS_DIR)
    print(f"      Loaded {len(documents)} source documents: {[d['source_file'] for d in documents]}")

    print("[2/5] Chunking by clause boundary ...")
    chunks = chunk_all(documents)
    print(f"      Produced {len(chunks)} chunks.")
    by_topic = Counter(c["topic_tag"] for c in chunks)
    print(f"      Topic breakdown: {dict(by_topic)}")

    print("[3/5] Generating embeddings (all-MiniLM-L6-v2, local, first run downloads the model) ...")
    embeddings = embedder.embed_chunks(chunks)
    print(f"      Embedded {embeddings.shape[0]} chunks, dimension {embeddings.shape[1]}.")

    print("[4/5] Writing to ChromaDB ...")
    chroma_client.reset_collection()
    chroma_client.add_chunks(chunks, embeddings)
    print(f"      ChromaDB collection now holds {chroma_client.count()} vectors at {chroma_client.CHROMA_PATH}")

    print("[5/5] Building BM25 index ...")
    bm25_scorer.build_index(chunks)
    print(f"      BM25 index persisted to {bm25_scorer.BM25_PATH}")

    print("\nIngestion complete.")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
