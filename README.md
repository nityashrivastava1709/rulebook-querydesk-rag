# Rulebook Conflict-Aware QA System

A question-answering service over a university rulebook that doesn't just answer confidently.
Every answer either (1) cites the exact clause it came from, (2) admits the rulebook is silent
on the question, or (3) flags that two clauses disagree — showing both sides.

## Aim

Naive RAG chatbots answer everything, including questions their source material doesn't cover
and questions where the source material contradicts itself. On a document like a university
regulation set — where a wrong attendance percentage told to a student has real consequences —
that's a genuine failure mode, not a cosmetic one. This system is built to know the difference
between "I found the answer," "I found nothing relevant," and "I found two things that
disagree," and to say so explicitly rather than papering over it.

Answers are **extractive by default**: the source of truth is always the retrieved clause text
itself, never a language model's free-form recollection. An optional LLM layer (Hugging Face,
local, no API key) can rephrase an already-selected answer into more natural language, but it
never decides *what* the answer is, and a guardrail check falls back to the raw extractive text
if the LLM's output mentions anything not present in what it was given.

## Corpus

~6,000 words across three formats, as required:
- `corpus/regulations.md` — main academic regulations (attendance, exams, medical leave,
  committees, scholarships, discipline, degree requirements, etc.), clause-numbered.
- `corpus/fee_deadlines.md` — fee schedule, including two markdown tables (escalating late-fee
  schedule, hostel fee deadlines).
- `corpus/hostel_policy.pdf` — hostel and residential life policy, generated as an actual PDF.

## The 3 planted contradictions

Documented in `data/contradictions.json`, matched to chunks dynamically by section reference
(never hardcoded chunk IDs — see "Design notes" below):

1. **Attendance threshold** — Clause 2.1 requires 75% attendance; Clause 4.3 says a
   medically-exempted student needs only 65%, explicitly overriding 2.1 ("Notwithstanding
   Clause 2.1...").
2. **Late fee calculation** — Clause F.3 says the late fee is a flat ₹500 regardless of delay;
   the Clause F.4 table describes an escalating percentage-based penalty instead.
3. **Waiver authority** — Clause 5.2 gives the Examination Eligibility Committee discretion to
   waive attendance shortfalls; Clause 8.4 says the Vice-Chancellor has *exclusive* authority
   over attendance waivers connected to disciplinary matters, with no other body permitted.

## The 25 unanswerable questions

Documented in `data/unanswerable_questions.json`, each with a note on the nearest-but-different
covered topic a naive system might wrongly match to (e.g. "family wedding" vs. the corpus's
medical-leave-only exemption).

## Threshold tuning

`NOT_COVERED_THRESHOLD` in `src/classification/response_classifier.py` is chosen empirically,
not guessed. Run `tune_threshold.py` after ingestion — it scores the 25 unanswerable questions
plus 5 known-answerable questions, reports the score distribution, and tests several candidate
thresholds, recommending the one that maximizes correct abstentions while minimizing false
rejections of real answers. Update the constant with whatever it recommends for your actual
embedding results (the shipped default of 0.55 is a starting point, not the final tuned value —
tune it on your machine after ingestion, since real embedding scores will differ from any
example run).

## Evaluation

Run `python -m src.evaluation.evaluator` after ingestion. It reports:
- **Abstention accuracy** — % of the 25 unanswerable questions correctly classified as `not_covered`.
- **Conflict detection accuracy** — % of the 3 contradictions' test questions correctly classified as `conflict`.
- **Answerable accuracy** — % of known-answerable questions correctly classified as `answered`.

Results are saved to `data/eval_results.json`.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the index (downloads the embedding model from Hugging Face the
#    first time only — no API key needed, it's a local model)
python ingest.py

# 3. (Recommended) Tune the not_covered threshold for your actual embeddings
python tune_threshold.py
# then update NOT_COVERED_THRESHOLD in src/classification/response_classifier.py

# 4. Run the evaluation suite
python -m src.evaluation.evaluator

# 5. Start the API server
uvicorn src.api.main:app --reload --port 8000

# 6. Open frontend/index.html in a browser (it calls http://localhost:8000/ask)
```

## Optional: LLM rephrasing layer

Set `USE_LLM_GENERATION = True` in `src/api/main.py` to enable natural-language rephrasing via
a local Hugging Face model (`google/flan-t5-base`, downloaded once, runs on CPU, no API key, no
cost). The LLM only rephrases chunks the classifier already selected — it never sees the full
corpus and never makes the answered/not_covered/conflict decision itself. Every generated
response passes a guardrail check that verifies it doesn't reference any clause not present in
its input; if it fails, the raw extractive chunk text is shown instead, and the response is
marked `"generation_method": "extractive_fallback"` so you can measure how often this happens.

## Design notes (for the interview)

- **Chunking by clause, not fixed token windows** — a clause is never split mid-sentence, since
  that would break both citation accuracy and conflict detection (a clause's contradiction with
  another only holds if its full meaning is intact in one chunk).
- **Hybrid retrieval (semantic + BM25)** — pure embedding similarity can blur specific numbers
  ("75%" vs "65%") into "roughly similar"; BM25 catches exact term/number matches embeddings
  miss. Combined with named, tunable weights (`SEMANTIC_WEIGHT = 0.7`, `BM25_WEIGHT = 0.3`),
  not inline magic numbers.
- **Semantic scores are NOT re-normalized per query** — cosine similarity is already on a
  comparable 0–1 scale across different questions, so per-query min-max normalization would
  force the top candidate of every query to ~1.0 regardless of actual match quality, silently
  breaking threshold-based abstention. Only BM25 (unbounded, not naturally comparable across
  queries) is normalized.
- **Contradiction lookup is data-driven** — `contradictions.json` references clauses by
  human-readable `section_reference` (e.g. "Clause 2.1"), never by chunk ID. At startup, the
  classifier resolves each reference to whatever chunk ID currently matches it in the vector
  store. Adding a 4th contradiction requires editing the JSON and re-running ingestion — zero
  code changes.
- **Extractive-first, LLM-optional** — the core pipeline never depends on the LLM to be correct;
  the LLM is a presentation-layer convenience with a hard guardrail and a safe fallback.

## Project structure

```
rulebook-qa/
├── corpus/                          # source documents (md, md+table, pdf)
├── data/                            # contradictions, test questions, generated indexes
├── src/
│   ├── ingestion/                   # loaders.py, chunker.py
│   ├── retrieval/                   # embedder.py, bm25_scorer.py, hybrid_ranker.py
│   ├── vectorstore/                 # chroma_client.py
│   ├── classification/              # response_classifier.py
│   ├── generation/                  # llm_generator.py (optional LLM layer)
│   ├── evaluation/                  # evaluator.py
│   └── api/                         # main.py (FastAPI, single POST /ask)
├── frontend/index.html              # plain HTML + fetch(), no framework
├── ingest.py                        # run once to build the index
├── tune_threshold.py                # run once to pick NOT_COVERED_THRESHOLD
└── requirements.txt
```
