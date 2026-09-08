"""
Wraps a local sentence-transformers model (downloaded once from Hugging Face,
cached locally, no API key required) for embedding both corpus chunks and
incoming questions.

CRITICAL: the exact same model/function must be used for corpus chunks (at
ingestion time) and for questions (at query time), or similarity scores
become meaningless.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str):
    """Embed a single string (e.g. an incoming question). Returns a 1D vector."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_chunks(chunks: list[dict]):
    """Batch-embed a list of chunk dicts (each with a 'text' field).
    Returns a 2D numpy array, one row per chunk, in the same order."""
    model = get_model()
    texts = [c["text"] for c in chunks]
    return model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
    )
