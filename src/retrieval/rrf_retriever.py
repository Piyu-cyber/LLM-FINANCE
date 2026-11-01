import torch
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from reranker import ColBERTReranker, rerank as colbert_rerank
from query_parser import parse_query, metadata_boost
from parent_child_retriever import build_corpus_index, expand_chunks


CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")
EMBEDDINGS_PATH = Path("data/finqa/train_embeddings.npy")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BM25_TOP_K = 100
DENSE_TOP_K = 100
RRF_K = 60

RERANK_TOP_K = 5


def load_corpus(path: Path = CORPUS_PATH) -> List[Dict[str, Any]]:
    chunks = []

    if not path.exists():
        return chunks

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized)


def bm25_search(query: str, bm25: BM25Okapi, top_k: int = BM25_TOP_K) -> List[tuple[int, float]]:

    scores = bm25.get_scores(
        tokenize(query)
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(index), float(scores[index]))
        for index in top_indices
    ]


def dense_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    top_k: int = DENSE_TOP_K,
) -> List[tuple[int, float]]:

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
    bm25_results: List[tuple[int, float]],
    dense_results: List[tuple[int, float]],
    k: int = 60,
) -> List[tuple[int, float]]:

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


def retrieve(
    query: str,
    chunks: List[Dict[str, Any]],
    bm25: BM25Okapi,
    dense_model: SentenceTransformer,
    embeddings: np.ndarray,
    top_k: int = RERANK_TOP_K,
    pool_size: int = 50,
    reranker_type: str = "cross_encoder",
    apply_metadata_boosting: bool = True,
    expand_parents: bool = False,
    corpus_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Complete hybrid retrieval pipeline: BM25 + Dense + RRF + Metadata Boost + Reranker.

    Parameters
    ----------
    query : str
        User's financial question.
    chunks : list[dict]
        Corpus chunks.
    bm25 : BM25Okapi
        Indexed BM25 corpus.
    dense_model : SentenceTransformer
        Dense embedder.
    embeddings : np.ndarray
        Pre-computed corpus embeddings.
    top_k : int
        Final number of chunks to return.
    pool_size : int
        Candidate pool size passed from RRF to the reranker.
    reranker_type : str
        'cross_encoder' (default BAAI/bge-reranker-base), 'colbert', or 'none'.
    apply_metadata_boosting : bool
        Whether to boost candidate scores based on company and fiscal-year matches.
    expand_parents : bool
        Whether to expand matched chunks to their full parent table and narrative context.
    corpus_index : dict, optional
        Pre-computed parent index for parent expansion.

    Returns
    -------
    list[dict]
        Top-k ranked evidence chunks (or expanded parent contexts).
    """
    # 1. Dual retrieval
    bm25_res = bm25_search(query, bm25, BM25_TOP_K)
    dense_res = dense_search(query, dense_model, embeddings, DENSE_TOP_K)

    # 2. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(bm25_res, dense_res, RRF_K)

    # 3. Candidate pool extraction & optional metadata boosting
    meta = parse_query(query) if apply_metadata_boosting else None
    company_ids = meta["company_ids"] if meta else []
    fiscal_years = meta["fiscal_years"] if meta else []

    candidate_pool = []
    for index, rrf_score in fused_results[:pool_size]:
        chunk = chunks[index].copy()
        chunk["rrf_score"] = float(rrf_score)

        if apply_metadata_boosting:
            boost = metadata_boost(chunk, company_ids, fiscal_years)
            chunk["rrf_score_boosted"] = chunk["rrf_score"] * boost
        else:
            chunk["rrf_score_boosted"] = chunk["rrf_score"]

        candidate_pool.append(chunk)

    if apply_metadata_boosting:
        candidate_pool.sort(key=lambda x: x["rrf_score_boosted"], reverse=True)

    # 4. Reranking
    if reranker_type == "cross_encoder":
        from cross_encoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        final_results = reranker.rerank(query, candidate_pool, top_k=top_k)
    elif reranker_type == "colbert":
        reranker = ColBERTReranker()
        final_results = colbert_rerank(query, candidate_pool, reranker, top_k=top_k)
    else:
        final_results = candidate_pool[:top_k]

    # 5. Optional parent-child context expansion
    if expand_parents:
        if corpus_index is None:
            corpus_index = build_corpus_index(chunks)
        final_results = expand_chunks(final_results, corpus_index)

    return final_results


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
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # --------------------------------------------------
    # Retrieve with Hybrid Pipeline
    # --------------------------------------------------
    print("\nExecuting hybrid retrieval with Cross-Encoder...")
    results = retrieve(
        query=query,
        chunks=chunks,
        bm25=bm25,
        dense_model=dense_model,
        embeddings=embeddings,
        top_k=RERANK_TOP_K,
        reranker_type="cross_encoder",
        apply_metadata_boosting=True,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("HYBRID RETRIEVAL + CROSS-ENCODER RESULTS")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print("\n---")
        print(f"Rank: {rank}")
        if "rerank_score" in result:
            print(f"Rerank Score: {result['rerank_score']:.4f}")
        if "rrf_score_boosted" in result:
            print(f"RRF Boosted Score: {result['rrf_score_boosted']:.6f}")
        elif "rrf_score" in result:
            print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Example ID: {result['example_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
