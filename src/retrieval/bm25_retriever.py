import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """Load chunk records from JSONL."""
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks for a query."""

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            result = dict(self.chunks[index])
            result["score"] = float(scores[index])
            results.append(result)

        return results


def main() -> None:
    print("Loading corpus...")

    chunks = load_corpus()

    print(f"Loaded {len(chunks)} chunks.")

    print("Building BM25 index...")

    retriever = BM25Retriever(chunks)

    query = "libor annual interest expense 3.8 million"

    results = retriever.retrieve(query, top_k=100)

    gold_chunk_id = "text_1"
    gold_example_id = "ADI/2009/page_49.pdf-1"

    print("\nQuery:")
    print(query)

    for rank, result in enumerate(results, start=1):
        if (
            result["example_id"] == gold_example_id
            and result["chunk_id"] == gold_chunk_id
        ):
            print(f"\nGOLD EVIDENCE FOUND AT RANK {rank}")
            print(result)
            break
    else:
        print("\nGold evidence not found in top 100.")


if __name__ == "__main__":
    main()