"""
Optional presentation layer: rephrases the ALREADY-DECIDED answer from the
classifier into natural language using a small local Hugging Face model
(no API key, no cost -- runs on CPU).

CRITICAL DESIGN RULE: this module never decides what the answer is or
whether something is covered. It only rephrases chunk text the classifier
already selected. If its output mentions any clause reference not present
in what it was given, that's treated as a hallucination and the raw
extractive chunk text is shown instead -- the system never trusts an
unverified LLM output as the final answer.
"""

import re
from transformers import pipeline

# Small, free, local, no API key. flan-t5-base is instruction-tuned and
# good at "rephrase only what I give you" style tasks without needing a
# much larger (and much slower-on-CPU) model.
MODEL_NAME = "google/flan-t5-base"

_generator = None


def get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline("text2text-generation", model=MODEL_NAME)
    return _generator


def _extract_clause_refs(text: str) -> set:
    return set(re.findall(r"Clause\s+[\w\.]+", text, re.IGNORECASE))


def _guardrail_ok(generated_text: str, allowed_refs: set) -> bool:
    """Fails the guardrail if the LLM mentions a clause reference that
    wasn't in the source material it was given."""
    mentioned = _extract_clause_refs(generated_text)
    normalized_allowed = {r.lower() for r in allowed_refs}
    normalized_mentioned = {r.lower() for r in mentioned}
    return normalized_mentioned.issubset(normalized_allowed)


def generate_answered(question: str, chunk: dict) -> dict:
    prompt = (
        "Using ONLY the exact content of the passage below, write a natural, "
        "conversational answer to the question. Do not add any information "
        "not present in the passage. Mention the section reference explicitly.\n\n"
        f"Question: {question}\n\n"
        f"Passage ({chunk['section_reference']}): {chunk['text']}\n\n"
        "Answer:"
    )
    generator = get_generator()
    output = generator(prompt, max_new_tokens=120, do_sample=False)[0]["generated_text"].strip()

    allowed_refs = {chunk["section_reference"]}
    if _guardrail_ok(output, allowed_refs) and output:
        return {"text": output, "generation_method": "llm_rephrased"}
    return {"text": chunk["text"], "generation_method": "extractive_fallback"}


def generate_conflict(question: str, chunk_a: dict, chunk_b: dict) -> dict:
    prompt = (
        "Two passages from a rulebook disagree with each other. Summarize what "
        "EACH source says, using only the text given, and explicitly state that "
        "they conflict. Do NOT decide which one is correct or more valid.\n\n"
        f"Question: {question}\n\n"
        f"Source A ({chunk_a['section_reference']}): {chunk_a['text']}\n\n"
        f"Source B ({chunk_b['section_reference']}): {chunk_b['text']}\n\n"
        "Summary of the conflict:"
    )
    generator = get_generator()
    output = generator(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"].strip()

    allowed_refs = {chunk_a["section_reference"], chunk_b["section_reference"]}
    if _guardrail_ok(output, allowed_refs) and output:
        return {"text": output, "generation_method": "llm_rephrased"}
    fallback_text = (
        f"{chunk_a['section_reference']} states: {chunk_a['text']}\n\n"
        f"However, {chunk_b['section_reference']} states: {chunk_b['text']}\n\n"
        "These two sections conflict."
    )
    return {"text": fallback_text, "generation_method": "extractive_fallback"}


def generate_not_covered(question: str, closest_chunk: dict | None) -> dict:
    if closest_chunk is None:
        return {
            "text": "This rulebook does not address this question.",
            "generation_method": "extractive_fallback",
        }

    prompt = (
        "The rulebook does not directly answer this question. Below is the "
        "closest related passage, which does NOT fully answer it. Write a "
        "short response stating plainly that this is not covered, and only "
        "optionally mention the related passage as something the user could "
        "check manually. Do not present it as an answer.\n\n"
        f"Question: {question}\n\n"
        f"Closest related passage ({closest_chunk['section_reference']}): {closest_chunk['text']}\n\n"
        "Response:"
    )
    generator = get_generator()
    output = generator(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"].strip()

    allowed_refs = {closest_chunk["section_reference"]}
    if _guardrail_ok(output, allowed_refs) and output:
        return {"text": output, "generation_method": "llm_rephrased"}
    return {
        "text": (
            "This rulebook does not address this question. "
            f"The closest related clause is {closest_chunk['section_reference']}, "
            "but it does not directly answer what was asked."
        ),
        "generation_method": "extractive_fallback",
    }
