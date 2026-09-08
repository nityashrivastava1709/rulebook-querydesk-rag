"""
Splits loaded documents into clause-level chunks (never mid-sentence, never
fixed token windows), and attaches metadata: chunk_id, section_reference,
topic_tag, source_file.

Chunking by clause matters here specifically because citations must point to
one complete rule, and conflict detection depends on a clause's full meaning
being intact in a single chunk.
"""

import re

# --- Topic tagging -----------------------------------------------------
# Keyword-priority classification. Order matters: more specific rules are
# checked first so that a clause mentioning both "waiver" and "attendance"
# (e.g. an attendance-waiver-authority clause) isn't mis-tagged as a plain
# attendance-threshold clause, and so on.


def infer_topic_tag(clause_text: str, heading_context: str = "") -> str:
    t = clause_text.lower()
    h = heading_context.lower()

    if "late fee" in t:
        return "late_fee"
    if "waiver" in t and "attendance" in t:
        return "attendance_waiver"
    if "attendance" in t and ("%" in clause_text or "percent" in t):
        return "attendance_threshold"
    if "medical leave" in t or "medical certificate" in t or "medical exemption" in t:
        return "medical_leave"
    if "examination" in t and "eligib" in t:
        return "exam_eligibility"
    if "scholarship" in t:
        return "scholarship"
    if "hostel" in t or "hostel" in h:
        return "hostel"
    if "disciplinary" in t or "misconduct" in t or "ragging" in t:
        return "discipline"
    if "thesis" in t or "research" in t or "postgraduate" in t:
        return "research"
    if "fee" in t:
        return "fees"
    return "general"


# --- Document-specific splitters ----------------------------------------
# The corpus has three distinct structural styles by design (bold inline
# clause markers, heading-as-clause, and PDF-flattened plain text). Each
# gets its own splitter; all three funnel into the same chunk record shape.


def _split_sections(text: str, pattern: str):
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    sections = []
    for idx, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def _split_clauses(section_text: str, clause_regex: str):
    pattern = re.compile(clause_regex, re.DOTALL)
    return [
        (f"Clause {cid.strip()}", body.strip())
        for cid, body in pattern.findall(section_text)
    ]


def chunk_regulations_md(text: str, source_file: str) -> list[dict]:
    """Style: '## Section N: Heading' followed by '**Clause X.Y** — body'."""
    sections = _split_sections(text, r"^## Section\s+\d+:\s*(.+)$")
    chunks = []
    for heading, section_text in sections:
        clauses = _split_clauses(
            section_text, r"\*\*Clause\s+([\w\.]+)\*\*\s*[\u2014-]\s*(.*?)(?=\*\*Clause\s+[\w\.]+\*\*|\Z)"
        )
        for section_reference, body in clauses:
            chunks.append(
                {
                    "text": f"{heading} - {section_reference}: {body}",
                    "section_reference": section_reference,
                    "topic_tag": infer_topic_tag(body, heading),
                    "source_file": source_file,
                }
            )
    return chunks


def chunk_fee_md(text: str, source_file: str) -> list[dict]:
    """Style: '## Clause F.N — Title' heading itself IS the clause; body follows."""
    pattern = re.compile(r"^## Clause\s+([\w\.]+)\s*[\u2014-]\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    chunks = []
    for idx, m in enumerate(matches):
        cid = m.group(1).strip()
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        full_text = f"{title}. {body}" if body else title
        section_reference = f"Clause {cid}"
        chunks.append(
            {
                "text": full_text,
                "section_reference": section_reference,
                "topic_tag": infer_topic_tag(full_text, title),
                "source_file": source_file,
            }
        )
    return chunks


def chunk_hostel_pdf(text: str, source_file: str) -> list[dict]:
    """Style (flattened PDF text): 'Section H1: Heading' then 'Clause H1.1 — body'."""
    sections = _split_sections(text, r"Section\s+(H\d+:\s*[^\n]+)")
    chunks = []
    for heading, section_text in sections:
        clauses = _split_clauses(
            section_text,
            r"Clause\s+(H[\w\.]+)\s*[\u2014-]\s*(.*?)(?=Clause\s+H[\w\.]+\s*[\u2014-]|\Z)",
        )
        for section_reference, body in clauses:
            chunks.append(
                {
                    "text": f"{heading} - {section_reference}: {body}",
                    "section_reference": section_reference,
                    "topic_tag": infer_topic_tag(body, heading),
                    "source_file": source_file,
                }
            )
    return chunks


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return value.strip("_")


def chunk_document(document: dict) -> list[dict]:
    """Dispatches to the correct splitter based on source_file, then assigns
    a stable, deterministic chunk_id to every resulting chunk."""
    text = document["text"]
    source_file = document["source_file"]

    if source_file.endswith(".pdf"):
        raw_chunks = chunk_hostel_pdf(text, source_file)
    elif source_file == "fee_deadlines.md":
        raw_chunks = chunk_fee_md(text, source_file)
    else:
        raw_chunks = chunk_regulations_md(text, source_file)

    for chunk in raw_chunks:
        chunk["chunk_id"] = _slugify(f"{source_file}__{chunk['section_reference']}")

    return raw_chunks


def chunk_all(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
