# save as diagnose_conflict.py in your project root, then run: python diagnose_conflict.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.classification import response_classifier as rc
from src.retrieval import hybrid_ranker

# Step 1: confirm the lookup resolved
lookup = rc.get_contradiction_lookup()
print("Contradiction pairs resolved:", len(lookup))
for pair in lookup:
    print(" ", pair)
print()

# Step 2: pick a question that SHOULD trigger a conflict
question = "What is the minimum attendance percentage required to sit for exams?"
ranked = hybrid_ranker.rank(question, top_k=6)

print(f"Question: {question}\n")
for r in ranked:
    print(f"  {r['section_reference']:15s} topic={r['topic_tag']:20s} combined={r['combined_score']:.3f}")

print()
print("Current threshold:", rc.NOT_COVERED_THRESHOLD)