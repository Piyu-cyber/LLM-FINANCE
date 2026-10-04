import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")
EMBEDDINGS_PATH = Path("data/finqa/train_embeddings.npy")

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_corpus() -> list[dict]:
    chunks = []

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


def main() -> None:
    print("Loading corpus...")

    chunks = load_corpus()

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Embedding shape: {embeddings.shape}")

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    query = "what is the the interest expense in 2009?"

    print("\nQuery:")
    print(query)

    results = retrieve(
        query,
        chunks,
        embeddings,
        model,
        top_k=5,
    )

    print("\nTop results:")

    for rank, result in enumerate(results, start=1):
        print("\n---")
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Example ID: {result['example_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()