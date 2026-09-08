"""
FastAPI app exposing a single POST /ask endpoint. This file is intentionally
thin: it only orchestrates calls to the classifier and (optionally) the LLM
generation layer. No retrieval, ranking, or classification logic lives here.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.classification import response_classifier as classifier

# Resolve the project root directory dynamically (two levels up from main.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_PATH = FRONTEND_DIR / "index.html"

# Set to True once you've verified the LLM layer separately (Part 7.5).
# When False, the API returns pure extractive answers -- exactly the
# already-proven Part 6/7 behavior -- with zero LLM involvement.
USE_LLM_GENERATION = False


class AskRequest(BaseModel):
    question: str


class CitationItem(BaseModel):
    text: str
    section_reference: str
    score: float
    source_file: str


class AskResponse(BaseModel):
    type: str
    answer: str | None
    citations: list[CitationItem]
    generation_method: str = "extractive"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the contradiction lookup and warm the retrieval indexes ONCE at
    # startup, not per-request.
    classifier.reload_contradiction_lookup()
    yield


# 1. Instantiate FastAPI app
app = FastAPI(title="Rulebook Conflict-Aware QA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_citation(chunk: dict) -> CitationItem:
    return CitationItem(
        text=chunk["text"],
        section_reference=chunk["section_reference"],
        score=round(chunk.get("combined_score", 0.0), 4),
        source_file=chunk["source_file"],
    )


# 2. Define routes after 'app' is instantiated
@app.get("/")
def serve_frontend():
    if not INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frontend index.html not found at {INDEX_PATH}",
        )
    return FileResponse(INDEX_PATH)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        result = classifier.classify(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    citations = [_to_citation(c) for c in result["chunks"]]
    generation_method = "extractive"
    answer_text = result["answer"]

    if USE_LLM_GENERATION:
        from src.generation import llm_generator

        try:
            if result["type"] == "answered":
                gen = llm_generator.generate_answered(request.question, result["chunks"][0])
            elif result["type"] == "conflict":
                gen = llm_generator.generate_conflict(
                    request.question, result["chunks"][0], result["chunks"][1]
                )
            else:
                closest = result["chunks"][0] if result["chunks"] else None
                gen = llm_generator.generate_not_covered(request.question, closest)
            answer_text = gen["text"]
            generation_method = gen["generation_method"]
        except Exception:
            # LLM layer failing must never break the API -- fall back to
            # the already-proven extractive answer.
            generation_method = "extractive_fallback"

    if result["type"] == "not_covered" and answer_text is None:
        answer_text = "This rulebook does not address this question."

    return AskResponse(
        type=result["type"],
        answer=answer_text,
        citations=citations,
        generation_method=generation_method,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# 3. Mount static files after 'app' exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")