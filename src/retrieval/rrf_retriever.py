import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from reranker import ColBERTReranker, rerank


CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")
EMBEDDINGS_PATH = Path("data/finqa/train_embeddings.npy")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BM25_TOP_K = 100
DENSE_TOP_K = 100
RRF_K = 60

RERANK_TOP_K = 5


def load_corpus():
    chunks = []

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    return text.lower().split()


def build_bm25(chunks):
    tokenized = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized)


def bm25_search(query, bm25, top_k):

    scores = bm25.get_scores(
        tokenize(query)
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(index), float(scores[index]))
        for index in top_indices
    ]


def dense_search(
    query,
    model,
    embeddings,
    top_k,
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(index), float(scores[index]))
        for index in top_indices
    ]


def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    k=60,
):

    rrf_scores = {}

    # BM25 ranking
    for rank, (index, _) in enumerate(
        bm25_results,
        start=1,
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0.0)
            + 1.0 / (k + rank)
        )

    # Dense ranking
    for rank, (index, _) in enumerate(
        dense_results,
        start=1,
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0.0)
            + 1.0 / (k + rank)
        )

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked


def main():

    query = input(
        "\nQuery:\n"
    ).strip()

    # --------------------------------------------------
    # Load corpus
    # --------------------------------------------------

    print("\nLoading corpus...")

    chunks = load_corpus()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

    # --------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------

    print("\nLoading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"\nLoading dense model: {MODEL_NAME}"
    )

    dense_model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------

    print("\nRunning BM25...")

    bm25_results = bm25_search(
        query,
        bm25,
        BM25_TOP_K,
    )

    print(
        f"BM25 candidates: "
        f"{len(bm25_results)}"
    )

    print("\nRunning dense retrieval...")

    dense_results = dense_search(
        query,
        dense_model,
        embeddings,
        DENSE_TOP_K,
    )

    print(
        f"Dense candidates: "
        f"{len(dense_results)}"
    )

    # --------------------------------------------------
    # RRF
    # --------------------------------------------------

    print("\nApplying RRF...")

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        RRF_K,
    )

    print(
        f"RRF candidate pool: "
        f"{len(fused_results)}"
    )

    # --------------------------------------------------
    # Convert RRF candidates to chunks
    # --------------------------------------------------

    candidates = []

    for index, rrf_score in fused_results:

        chunk = chunks[index].copy()

        chunk["rrf_score"] = float(
            rrf_score
        )

        candidates.append(chunk)

    # --------------------------------------------------
    # ColBERT-style reranking
    # --------------------------------------------------

    print("\nLoading ColBERT-style reranker...")

    reranker = ColBERTReranker()

    print("\nReranking RRF candidates...")

    reranked_results = rerank(
        query,
        candidates,
        reranker,
        top_k=RERANK_TOP_K,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RRF + COLBERT-STYLE RERANKING")
    print("=" * 60)

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        print("\n---")

        print(
            f"Rank: {rank}"
        )

        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Example ID: "
            f"{result['example_id']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )


if __name__ == "__main__":
    main()