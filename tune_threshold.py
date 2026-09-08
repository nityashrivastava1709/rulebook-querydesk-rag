"""
Run after ingest.py to empirically choose the not_covered threshold:
    python tune_threshold.py

Runs the 25 unanswerable questions plus a handful of known-answerable
questions through the hybrid ranker, reports the score distribution, and
tests candidate thresholds so the choice is evidence-based rather than a
guess. Update NOT_COVERED_THRESHOLD in src/classification/response_classifier.py
with whatever this recommends.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from src.retrieval import hybrid_ranker

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# A handful of questions we know ARE answerable, straight from the corpus.
KNOWN_ANSWERABLE = [
    "When is the semester fee due?",
    "What is the hostel curfew time on weekdays?",
    "How many credits are required for a minimum course load?",
    "What is the security deposit for new hostel students?",
    "What is the minimum CGPA required for a Merit Scholarship?",
]

CANDIDATE_THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]


def main():
    with open(os.path.join(DATA_DIR, "unanswerable_questions.json")) as f:
        unanswerable = [item["question"] for item in json.load(f)["unanswerable_questions"]]

    print("Scoring known-answerable questions...")
    answerable_scores = []
    for q in KNOWN_ANSWERABLE:
        ranked = hybrid_ranker.rank(q, top_k=1)
        score = ranked[0]["combined_score"] if ranked else 0.0
        answerable_scores.append(score)
        print(f"  [{score:.3f}] {q}")

    print("\nScoring unanswerable questions...")
    unanswerable_scores = []
    for q in unanswerable:
        ranked = hybrid_ranker.rank(q, top_k=1)
        score = ranked[0]["combined_score"] if ranked else 0.0
        unanswerable_scores.append(score)
        print(f"  [{score:.3f}] {q}")

    print("\n--- Score distribution ---")
    print(f"Answerable   -> min={min(answerable_scores):.3f} max={max(answerable_scores):.3f}")
    print(f"Unanswerable -> min={min(unanswerable_scores):.3f} max={max(unanswerable_scores):.3f}")

    print("\n--- Threshold candidates ---")
    best_threshold = None
    best_score = -1
    for t in CANDIDATE_THRESHOLDS:
        correct_abstentions = sum(1 for s in unanswerable_scores if s < t)
        false_rejections = sum(1 for s in answerable_scores if s < t)
        # Simple scoring: maximize correct abstentions, penalize false rejections heavily.
        tuning_score = correct_abstentions - (false_rejections * 3)
        print(
            f"  threshold={t:.2f} -> correct abstentions {correct_abstentions}/{len(unanswerable_scores)}, "
            f"false rejections on answerable {false_rejections}/{len(answerable_scores)}"
        )
        if tuning_score > best_score:
            best_score = tuning_score
            best_threshold = t

    print(f"\nRecommended NOT_COVERED_THRESHOLD: {best_threshold}")
    print("Update this value in src/classification/response_classifier.py")


if __name__ == "__main__":
    main()
