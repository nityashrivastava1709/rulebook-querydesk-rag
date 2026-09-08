"""
Loaders normalize every corpus source format (markdown, markdown-with-tables, PDF)
into one uniform shape: {"text": <raw text>, "source_file": <filename>}.

Downstream modules (chunker, embedder, etc.) never need to know or care which
format a document originally came from.
"""

import os
import re
import pdfplumber


def load_markdown(filepath: str) -> dict:
    """Reads a markdown file as-is. Markdown already carries structure
    (headings, clause markers) that the chunker will parse later."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return {"text": text, "source_file": os.path.basename(filepath)}


def load_markdown_with_tables(filepath: str) -> dict:
    """Reads a markdown file that contains one or more pipe-delimited tables.

    Table rows are converted into natural-language sentences in place, because
    embedding models are trained on prose, not tabular syntax -- a raw table
    row like '| 8-14 days | 5% | ... |' embeds poorly, while a full sentence
    describing the same row embeds and retrieves well.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    output_lines = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.strip()

        # Detect the start of a markdown table: a header row followed by a
        # separator row of dashes/pipes.
        is_table_header = (
            stripped.startswith("|")
            and i + 1 < len(raw_lines)
            and re.match(r"^\|?[\s\-\|:]+\|?$", raw_lines[i + 1].strip())
        )

        if is_table_header:
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2  # skip header + separator row
            sentences = []
            while i < len(raw_lines) and raw_lines[i].strip().startswith("|"):
                cells = [c.strip() for c in raw_lines[i].strip().strip("|").split("|")]
                if len(cells) == len(headers):
                    row_desc = "; ".join(
                        f"{h}: {c}" for h, c in zip(headers, cells) if c
                    )
                    sentences.append(f"Table entry - {row_desc}.")
                i += 1
            output_lines.append(" ".join(sentences))
            continue

        output_lines.append(line)
        i += 1

    return {"text": "\n".join(output_lines), "source_file": os.path.basename(filepath)}


def load_pdf(filepath: str) -> dict:
    """Extracts text from a PDF, page by page, with light cleanup of
    PDF-specific artifacts (extra blank lines, stray whitespace)."""
    pages_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

    full_text = "\n".join(pages_text)
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return {"text": full_text, "source_file": os.path.basename(filepath)}


def load_corpus(corpus_dir: str) -> list[dict]:
    """Loads every supported file in corpus_dir using the correct loader,
    returning a flat list of uniformly-shaped documents."""
    documents = []
    for filename in sorted(os.listdir(corpus_dir)):
        filepath = os.path.join(corpus_dir, filename)
        if filename.endswith(".pdf"):
            documents.append(load_pdf(filepath))
        elif filename == "fee_deadlines.md":
            # Known to contain markdown tables.
            documents.append(load_markdown_with_tables(filepath))
        elif filename.endswith(".md"):
            documents.append(load_markdown(filepath))
        # Other formats can be added here as needed.
    return documents
