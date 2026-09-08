"""
Runs the full evaluation suite:
  1. All 25 unanswerable questions -> should classify as not_covered
  2. Test questions for each of the 3 documented contradictions -> should classify as conflict
  3. A handful of known-answerable questions -> should classify as answered

Saves results to data/eval_results.json and prints a summary, including the
headline abstention accuracy number for the README.

Run with:
    python -m src.evaluation.evaluator
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.classification import response_classifier as rc

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

KNOWN_ANSWERABLE = [
    "When is the semester fee due?",
    "What is the hostel curfew time on weekdays?",
    "How many credits are required for a minimum course load?",
    "What is the security deposit for new hostel students?",
    "What is the minimum CGPA required for a Merit Scholarship?",
]


def run_unanswerable_suite():
    with open(os.path.join(DATA_DIR, "unanswerable_questions.json")) as f:
        questions = json.load(f)["unanswerable_questions"]

    results = []
    correct = 0
    for item in questions:
        verdict = rc.classify(item["question"])
        is_correct = verdict["type"] == "not_covered"
        correct += is_correct
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "predicted_type": verdict["type"],
                "correct": is_correct,
            }
        )
    accuracy = correct / len(questions) if questions else 0.0
    return results, accuracy


def run_contradiction_suite():
    with open(os.path.join(DATA_DIR, "contradictions.json")) as f:
        contradictions = json.load(f)["contradictions"]

    results = []
    correct = 0
    total = 0
    for item in contradictions:
        for q in item["test_questions"]:
            total += 1
            verdict = rc.classify(q)
            is_correct = verdict["type"] == "conflict"
            correct += is_correct
            results.append(
                {
                    "contradiction_id": item["id"],
                    "question": q,
                    "predicted_type": verdict["type"],
                    "correct": is_correct,
                }
            )
    accuracy = correct / total if total else 0.0
    return results, accuracy


def run_answerable_suite():
    results = []
    correct = 0
    for q in KNOWN_ANSWERABLE:
        verdict = rc.classify(q)
        is_correct = verdict["type"] == "answered"
        correct += is_correct
        results.append({"question": q, "predicted_type": verdict["type"], "correct": is_correct})
    accuracy = correct / len(KNOWN_ANSWERABLE) if KNOWN_ANSWERABLE else 0.0
    return results, accuracy


def main():
    print("Running unanswerable question suite (25 questions)...")
    unanswerable_results, abstention_accuracy = run_unanswerable_suite()
    print(f"  Abstention accuracy: {abstention_accuracy:.1%}")

    print("\nRunning contradiction detection suite...")
    conflict_results, conflict_accuracy = run_contradiction_suite()
    print(f"  Conflict detection accuracy: {conflict_accuracy:.1%}")

    print("\nRunning known-answerable suite...")
    answerable_results, answerable_accuracy = run_answerable_suite()
    print(f"  Answerable accuracy: {answerable_accuracy:.1%}")

    output = {
        "abstention_accuracy": abstention_accuracy,
        "conflict_detection_accuracy": conflict_accuracy,
        "answerable_accuracy": answerable_accuracy,
        "unanswerable_results": unanswerable_results,
        "conflict_results": conflict_results,
        "answerable_results": answerable_results,
    }

    output_path = os.path.join(DATA_DIR, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
