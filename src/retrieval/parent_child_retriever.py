"""parent_child_retriever.py — Parent-child context expansion for Phase 1 Step 1.3.

When dense or BM25 retrieval matches a fine-grained *child* chunk (a single
table row or a short text sentence), passing only that chunk to the LLM
often loses the surrounding context needed for accurate reasoning.

This module expands each matched child chunk to its *parent* context:
the full table plus the surrounding pre/post narrative paragraphs from the
same FinQA example.

Two retrieval strategies are supported:

1. ``expand_chunk(chunk, corpus_index)`` — expand one chunk to its parent.
2. ``expand_chunks(chunks, corpus_index)`` — expand a list, deduplicating
   by parent so the same table is not included twice.

The *corpus_index* is a dict keyed by ``example_id`` that maps to all
chunks from that example.  Build it once with ``build_corpus_index`` and
reuse it across queries.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_corpus_index(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group all corpus chunks by their ``example_id``.

    Parameters
    ----------
    chunks :
        The full list of chunks loaded from ``train_chunks.jsonl``.

    Returns
    -------
    dict[str, list[dict]]
        Mapping ``example_id -> [chunk, chunk, ...]`` in original order.
    """
    index: dict[str, list[dict]] = {}

    for chunk in chunks:
        example_id = chunk.get("example_id", "")

        if example_id not in index:
            index[example_id] = []

        index[example_id].append(chunk)

    return index


# ---------------------------------------------------------------------------
# Single-chunk expansion
# ---------------------------------------------------------------------------

def expand_chunk(
    chunk: dict,
    corpus_index: dict[str, list[dict]],
    include_pre_text: bool = True,
    include_post_text: bool = True,
    include_table: bool = True,
    max_text_sentences: int = 5,
) -> dict:
    """Expand a single retrieved chunk to its full parent context.

    The returned dict is a new dict containing:

    ``child_chunk``
        The original retrieved chunk (unchanged).
    ``parent_text``
        The combined text of the parent context (for passing to the LLM).
    ``table_chunks``
        All table row chunks from the same example (may be empty).
    ``text_chunks``
        Up to *max_text_sentences* surrounding text chunks.
    ``example_id``
        The common document identifier.

    Parameters
    ----------
    chunk :
        The child chunk to expand.
    corpus_index :
        Output of ``build_corpus_index``.
    include_pre_text :
        Include pre_text chunks from the parent example.
    include_post_text :
        Include post_text chunks from the parent example.
    include_table :
        Include all table row chunks from the parent example.
    max_text_sentences :
        Maximum number of text chunks (pre + post combined) to include
        in the parent context.  Keeps the context window manageable.
    """
    example_id = chunk.get("parent_id") or chunk.get("example_id", "")

    siblings = corpus_index.get(example_id, [])

    table_chunks = []
    pre_chunks = []
    post_chunks = []

    for sibling in siblings:
        source = sibling.get("source", "")

        if source == "table":
            if include_table:
                table_chunks.append(sibling)

        elif source == "pre_text":
            if include_pre_text:
                pre_chunks.append(sibling)

        elif source == "post_text":
            if include_post_text:
                post_chunks.append(sibling)

    # Limit text chunks to avoid overwhelming the LLM context window
    text_chunks = pre_chunks + post_chunks
    text_chunks = text_chunks[:max_text_sentences]

    # Build a single combined parent context string
    parts = []

    if text_chunks:
        parts.append(
            " ".join(c["text"] for c in text_chunks)
        )

    if table_chunks:
        # Reconstruct a readable table block from individual row chunks
        table_lines = [c["text"] for c in table_chunks]
        parts.append("\n".join(table_lines))

    parent_text = "\n\n".join(parts)

    return {
        "child_chunk": chunk,
        "parent_text": parent_text,
        "table_chunks": table_chunks,
        "text_chunks": text_chunks,
        "example_id": example_id,
    }


# ---------------------------------------------------------------------------
# Batch expansion with deduplication
# ---------------------------------------------------------------------------

def expand_chunks(
    chunks: list[dict],
    corpus_index: dict[str, list[dict]],
    deduplicate_by_parent: bool = True,
    **expand_kwargs,
) -> list[dict]:
    """Expand a list of retrieved chunks to their parent contexts.

    Parameters
    ----------
    chunks :
        Retrieved child chunks (e.g. top-5 reranked results).
    corpus_index :
        Output of ``build_corpus_index``.
    deduplicate_by_parent :
        If ``True``, when multiple chunks share the same ``example_id``,
        only the first one's parent context is included.  This prevents
        the same table and narrative from being passed to the LLM
        multiple times.
    **expand_kwargs :
        Forwarded to ``expand_chunk``.

    Returns
    -------
    list[dict]
        List of expanded parent-context dicts, one per unique parent.
    """
    seen_parents: set[str] = set()
    expanded = []

    for chunk in chunks:
        example_id = chunk.get("parent_id") or chunk.get("example_id", "")

        if deduplicate_by_parent and example_id in seen_parents:
            continue

        seen_parents.add(example_id)

        expanded.append(
            expand_chunk(chunk, corpus_index, **expand_kwargs)
        )

    return expanded


# ---------------------------------------------------------------------------
# Convenience: build context string for LLM
# ---------------------------------------------------------------------------

def build_llm_context(
    expanded_results: list[dict],
    max_results: int = 3,
) -> str:
    """Assemble a ready-to-use context block for the LLM from expanded results.

    Parameters
    ----------
    expanded_results :
        Output of ``expand_chunks``.
    max_results :
        Maximum number of parent contexts to include.

    Returns
    -------
    str
        A formatted string with numbered evidence sections.
    """
    sections = []

    for i, result in enumerate(expanded_results[:max_results], start=1):
        example_id = result["example_id"]
        parent_text = result["parent_text"]

        section = (
            f"[Evidence {i}] Source: {example_id}\n"
            f"{parent_text}"
        )
        sections.append(section)

    return "\n\n---\n\n".join(sections)
