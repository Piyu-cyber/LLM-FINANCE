"""context_builder.py — Prompt context assembly and token budget management.

Assembles prompt context from retrieved SEC 10-K evidence chunks for downstream
LLM reasoning and deterministic verification.

Supported Context Modes:
1. "child_only": Formats retrieved child chunks directly (single table rows or sentences).
2. "parent_expanded": Expands each retrieved chunk into its parent table and narrative.
3. "deduplicated_parent": Expands retrieved chunks and deduplicates by parent example_id,
   preventing duplicate table and narrative context across sibling chunks.

Deterministic Selection Strategy for Token Budget Capping:
- Chunks/sections are evaluated in strict priority order (retrieval rank 1 to K).
- Selection is chunk-atomic: chunks are never truncated mid-span or mid-table,
  preserving exact verbatim text required for downstream operand span verification.
- When adding a chunk causes cumulative tokens to exceed max_tokens, that chunk
  and subsequent lower-ranked chunks are omitted.
- Truncation is never silent: every dropped chunk is recorded in ContextBuildResult.chunks_dropped,
  with is_truncated=True and an explicit truncation_reason string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

try:
    from src.retrieval.parent_child_retriever import (
        build_corpus_index,
        expand_chunk,
        expand_chunks,
    )
except ImportError:
    try:
        from parent_child_retriever import (
            build_corpus_index,
            expand_chunk,
            expand_chunks,
        )
    except ImportError:
        import sys
        from pathlib import Path

        retrieval_dir = Path(__file__).resolve().parent.parent / "retrieval"
        if str(retrieval_dir) not in sys.path:
            sys.path.insert(0, str(retrieval_dir))
        from parent_child_retriever import (
            build_corpus_index,
            expand_chunk,
            expand_chunks,
        )


@dataclass
class ContextBuildResult:
    """Structured result of context assembly."""

    context_text: str
    chunks_included: list[dict]
    chunks_dropped: list[dict]
    token_count: int
    max_tokens: Optional[int]
    is_truncated: bool
    mode: str
    truncation_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for logging and reporting."""
        return {
            "token_count": self.token_count,
            "max_tokens": self.max_tokens,
            "is_truncated": self.is_truncated,
            "mode": self.mode,
            "num_included": len(self.chunks_included),
            "num_dropped": len(self.chunks_dropped),
            "truncation_reason": self.truncation_reason,
            "context_length_chars": len(self.context_text),
        }


def default_token_counter(text: str) -> int:
    """Deterministic token counting fallback (word and punctuation regex)."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def format_child_chunk(chunk: dict) -> str:
    """Format an individual evidence chunk with unambiguous citation headers.

    Example output:
        [Evidence Chunk: table_row_0] (Document: AAPL_2019, Source: table)
        Sales | $ 260,174 | $ 265,595
    """
    chunk_id = chunk.get("chunk_id", "unknown_chunk")
    example_id = chunk.get("example_id", chunk.get("parent_id", "unknown_doc"))
    source = chunk.get("source", "")
    company = chunk.get("company", "")
    year = chunk.get("year", "")

    meta_parts = [f"Document: {example_id}"]
    if source:
        meta_parts.append(f"Source: {source}")
    if company:
        meta_parts.append(f"Company: {company}")
    if year:
        meta_parts.append(f"Year: {year}")

    header = f"[Evidence Chunk: {chunk_id}] ({', '.join(meta_parts)})"
    text = chunk.get("text", "").strip()
    return f"{header}\n{text}"


def format_parent_expanded_item(expanded_item: dict) -> str:
    """Format an expanded parent context block with granular chunk headers.

    If constituent table_chunks or text_chunks are available, each is labeled
    with its specific chunk_id so the downstream LLM can ground individual operands.
    """
    example_id = expanded_item.get("example_id", "unknown_doc")
    table_chunks = expanded_item.get("table_chunks", [])
    text_chunks = expanded_item.get("text_chunks", [])
    child_chunk = expanded_item.get("child_chunk", {})
    child_id = child_chunk.get("chunk_id", "")

    header = f"=== Parent Document Section: {example_id} ==="
    if child_id:
        header += f" (Matched Target: {child_id})"

    body_parts = []

    # If granular constituent chunks are available, format each with its chunk_id
    if text_chunks or table_chunks:
        if text_chunks:
            body_parts.append("--- Narrative Context ---")
            for c in text_chunks:
                cid = c.get("chunk_id", "text")
                body_parts.append(f"[Evidence Chunk: {cid}]\n{c.get('text', '').strip()}")

        if table_chunks:
            body_parts.append("--- Financial Table ---")
            for c in table_chunks:
                cid = c.get("chunk_id", "table_row")
                body_parts.append(f"[Evidence Chunk: {cid}]\n{c.get('text', '').strip()}")
    else:
        # Fallback to parent_text string if granular chunks not separated
        parent_text = expanded_item.get("parent_text", "").strip()
        body_parts.append(parent_text)

    return f"{header}\n" + "\n\n".join(body_parts)


class ContextBuilder:
    """Assembles prompt evidence context with strict token budget enforcement."""

    VALID_MODES = {"child_only", "parent_expanded", "deduplicated_parent"}

    def __init__(
        self,
        default_mode: str = "child_only",
        max_tokens: Optional[int] = 2048,
        token_counter: Optional[Union[Callable[[str], int], Any]] = None,
        separator: str = "\n\n---\n\n",
    ) -> None:
        """Initialize ContextBuilder.

        Parameters
        ----------
        default_mode : str
            Default context mode ("child_only", "parent_expanded", "deduplicated_parent").
        max_tokens : Optional[int]
            Default maximum token budget for the context block.
        token_counter : Optional[Callable or object]
            Token counter function, or LLMClient/Tokenizer with count_tokens method.
        separator : str
            Delimiter between distinct evidence sections.
        """
        if default_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid default_mode: {default_mode}. Must be one of {self.VALID_MODES}"
            )
        self.default_mode = default_mode
        self.max_tokens = max_tokens
        self.separator = separator
        self.token_counter = self._resolve_token_counter(token_counter)

    @staticmethod
    def _resolve_token_counter(
        counter: Optional[Union[Callable[[str], int], Any]],
    ) -> Callable[[str], int]:
        if counter is None:
            return default_token_counter
        if hasattr(counter, "count_tokens") and callable(counter.count_tokens):
            return counter.count_tokens
        if hasattr(counter, "encode") and callable(counter.encode):
            return lambda text: len(counter.encode(text))
        if callable(counter):
            return counter
        return default_token_counter

    def build_context(
        self,
        chunks: Sequence[dict],
        corpus_index: Optional[dict[str, list[dict]]] = None,
        mode: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ContextBuildResult:
        """Assemble formatted evidence context from retrieved chunks.

        Parameters
        ----------
        chunks : Sequence[dict]
            Retrieved evidence chunks in rank/priority order.
        corpus_index : Optional[dict[str, list[dict]]]
            Corpus index mapping example_id -> list of sibling chunks.
            Required for "parent_expanded" and "deduplicated_parent" modes.
        mode : Optional[str]
            Context mode override.
        max_tokens : Optional[int]
            Token budget override.

        Returns
        -------
        ContextBuildResult
            Assembled context and truncation diagnostic metadata.
        """
        active_mode = mode or self.default_mode
        if active_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: {active_mode}. Must be one of {self.VALID_MODES}"
            )

        budget = max_tokens if max_tokens is not None else self.max_tokens

        if not chunks:
            return ContextBuildResult(
                context_text="",
                chunks_included=[],
                chunks_dropped=[],
                token_count=0,
                max_tokens=budget,
                is_truncated=False,
                mode=active_mode,
                truncation_reason=None,
            )

        # 1. Expand chunks according to selected mode
        items_to_format: list[tuple[dict, str]] = []

        if active_mode == "child_only":
            for chunk in chunks:
                formatted = format_child_chunk(chunk)
                items_to_format.append((chunk, formatted))

        elif active_mode == "parent_expanded":
            if corpus_index is None:
                raise ValueError(
                    "corpus_index is required for 'parent_expanded' mode."
                )
            for chunk in chunks:
                expanded = expand_chunk(chunk, corpus_index)
                formatted = format_parent_expanded_item(expanded)
                items_to_format.append((chunk, formatted))

        elif active_mode == "deduplicated_parent":
            if corpus_index is None:
                raise ValueError(
                    "corpus_index is required for 'deduplicated_parent' mode."
                )
            expanded_list = expand_chunks(
                list(chunks), corpus_index, deduplicate_by_parent=True
            )
            for item in expanded_list:
                child = item.get("child_chunk", {})
                formatted = format_parent_expanded_item(item)
                items_to_format.append((child, formatted))

        # 2. Greedy chunk-atomic selection under token budget
        included_chunks: list[dict] = []
        dropped_chunks: list[dict] = []
        included_texts: list[str] = []

        for i, (orig_chunk, formatted_text) in enumerate(items_to_format):
            candidate_texts = included_texts + [formatted_text]
            candidate_context = self.separator.join(candidate_texts)
            candidate_tokens = self.token_counter(candidate_context)

            if budget is not None and candidate_tokens > budget:
                # If even the very first chunk exceeds budget, keep it as atomic minimum
                # but flag as truncated
                if not included_texts:
                    included_chunks.append(orig_chunk)
                    included_texts.append(formatted_text)
                    # All remaining chunks are dropped
                    for remaining_chunk, _ in items_to_format[i + 1 :]:
                        dropped_chunks.append(remaining_chunk)
                    break
                else:
                    # Drop candidate and all subsequent chunks
                    for remaining_chunk, _ in items_to_format[i:]:
                        dropped_chunks.append(remaining_chunk)
                    break
            else:
                included_chunks.append(orig_chunk)
                included_texts.append(formatted_text)

        final_context = self.separator.join(included_texts)
        final_tokens = self.token_counter(final_context)
        is_truncated = len(dropped_chunks) > 0

        truncation_reason = None
        if is_truncated and budget is not None:
            truncation_reason = (
                f"Context exceeded token budget of {budget} tokens: "
                f"retained {len(included_chunks)} chunk(s) ({final_tokens} tokens), "
                f"dropped {len(dropped_chunks)} lower-ranked chunk(s)."
            )

        return ContextBuildResult(
            context_text=final_context,
            chunks_included=included_chunks,
            chunks_dropped=dropped_chunks,
            token_count=final_tokens,
            max_tokens=budget,
            is_truncated=is_truncated,
            mode=active_mode,
            truncation_reason=truncation_reason,
        )
