"""chunker.py — Convert FinQA examples into retrievable text chunks.

Phase 1 Step 1.3 upgrades:
- Table chunks now include document context (company, year) and units in
  every row so the row is self-contained for retrieval.
- A ``parent_id`` field points back to the example_id so a parent-child
  retriever can later expand a matched row to the full table + context.
- Text chunks gain ``parent_id`` and ``company`` / ``year`` fields for
  consistency with table chunks.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_company_year(example_id: str) -> tuple[str, str]:
    """Extract (company_ticker, fiscal_year) from a FinQA ``example_id``.

    FinQA IDs have the form ``"TICKER/YEAR/page_N.pdf-M"``.

    Returns
    -------
    tuple[str, str]
        ``(ticker, year)`` or ``("", "")`` if the ID is malformed.
    """
    parts = example_id.split("/")

    ticker = parts[0].upper() if len(parts) > 0 else ""
    year = parts[1] if len(parts) > 1 else ""

    return ticker, year


def _infer_units(pre_text: list[str]) -> str:
    """Try to detect the reporting unit from the pre-text narratives.

    Scans sentences for patterns like "in millions", "in thousands",
    "in billions".  Returns the matched unit string or an empty string
    when nothing is found.

    Parameters
    ----------
    pre_text :
        The ``pre_text`` list from a FinQA example.
    """
    unit_pattern = re.compile(
        r"in\s+(millions?|thousands?|billions?)\s+of\s+\w+|"
        r"(millions?|thousands?|billions?)\s+of\s+(?:us\s+)?dollars?",
        re.IGNORECASE,
    )

    for sentence in pre_text:
        match = unit_pattern.search(sentence)
        if match:
            return match.group(0).strip()

    return ""


# ---------------------------------------------------------------------------
# Text chunk creation
# ---------------------------------------------------------------------------

def create_text_chunks(example: dict[str, Any]) -> list[dict]:
    """Create chunks from pre_text and post_text.

    Each chunk gains ``parent_id``, ``company``, and ``year`` metadata
    fields that downstream filters and parent-child expansion can use.
    """
    chunks = []

    example_id = str(example["id"])
    company, year = _parse_company_year(example_id)

    # Pre-text chunks
    for i, text in enumerate(example.get("pre_text", [])):
        if text.strip() and any(char.isalnum() for char in text):
            chunks.append(
                {
                    "example_id": example_id,
                    "chunk_id": f"text_{i}",
                    "source": "pre_text",
                    "position": i,
                    "company": company,
                    "year": year,
                    "parent_id": example_id,
                    "text": text.strip(),
                }
            )

    # Post-text chunks
    for i, text in enumerate(example.get("post_text", [])):
        if text.strip() and any(char.isalnum() for char in text):
            chunks.append(
                {
                    "example_id": example_id,
                    "chunk_id": f"post_text_{i}",
                    "source": "post_text",
                    "position": i,
                    "company": company,
                    "year": year,
                    "parent_id": example_id,
                    "text": text.strip(),
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# Table chunk creation (Phase 1 enriched)
# ---------------------------------------------------------------------------

def create_table_chunks(example: dict[str, Any]) -> list[dict]:
    """Create one self-contained chunk per table row.

    Phase 1 enhancements
    --------------------
    * Every row chunk is prefixed with ``[DOC: TICKER YEAR]`` and a
      ``[UNIT: ...]`` tag when the reporting unit can be inferred from
      ``pre_text``.  This makes the chunk independently interpretable
      without surrounding context, which is critical for BM25 and dense
      retrieval.

    * Column headers are included in every row (unchanged from the
      original implementation).

    * A ``parent_id`` field points back to the ``example_id`` so a
      parent-child retriever can re-fetch the full table plus surrounding
      narrative for the LLM.

    * ``company`` and ``year`` metadata are stored as top-level keys for
      metadata pre-filtering.
    """
    table = example.get("table", [])

    if not table:
        return []

    headers = table[0]
    chunks = []

    example_id = str(example["id"])
    company, year = _parse_company_year(example_id)

    # Infer the reporting unit from the pre-text (best-effort)
    pre_text = example.get("pre_text", [])
    units = _infer_units(pre_text)

    # Build a concise document-context prefix, e.g. "[DOC: UNP 2009]"
    inner_parts = [p for p in [company, year] if p]
    doc_prefix = "[DOC: " + " ".join(inner_parts) + "]" if inner_parts else ""

    unit_prefix = f"[UNIT: {units}]" if units else ""

    for i, row in enumerate(table[1:]):
        values = []

        for header, value in zip(headers, row):
            if header:
                values.append(f"{header}: {value}")
            else:
                values.append(str(value))

        row_text = " | ".join(values)

        # Prepend doc context so every row chunk is self-contained
        prefix_parts = [p for p in [doc_prefix, unit_prefix] if p]
        if prefix_parts:
            full_text = " ".join(prefix_parts) + " " + row_text.strip()
        else:
            full_text = row_text.strip()

        chunks.append(
            {
                "example_id": example_id,
                "chunk_id": f"table_{i}",
                "source": "table",
                "position": i,
                "company": company,
                "year": year,
                "parent_id": example_id,
                "units": units,
                "text": full_text,
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def create_chunks(example: dict[str, Any]) -> list[dict]:
    """Create all retrievable chunks for a FinQA example."""
    text_chunks = create_text_chunks(example)
    table_chunks = create_table_chunks(example)

    return text_chunks + table_chunks


if __name__ == "__main__":
    from finqa_loader import load_finqa

    data = load_finqa()

    example = data[0]

    chunks = create_chunks(example)

    print(f"Number of chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n---")
        print(f"ID      : {chunk['chunk_id']}")
        print(f"Source  : {chunk['source']}")
        print(f"Company : {chunk.get('company', '')}")
        print(f"Year    : {chunk.get('year', '')}")
        print(f"Units   : {chunk.get('units', '')}")
        print(f"Text    : {chunk['text'][:200]}")
