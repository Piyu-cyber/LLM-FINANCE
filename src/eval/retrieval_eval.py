import torch
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")
FINQA_PATH = Path("data/finqa/dataset/train.json")
EMBEDDINGS_PATH = Path("data/finqa/train_embeddings.npy")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 100
QUERY_BATCH_SIZE = 64
RRF_K = 60

# Top-N candidates passed to cross-encoder reranker
RERANK_POOL_SIZE = 50
RERANK_TOP_K = 5


# ============================================================
# Loading
# ============================================================

def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                data.append(json.loads(line))

    return data


def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Gold evidence
# ============================================================

def get_gold_evidence(example):

    qa = example.get("qa", {})
    gold_inds = qa.get("gold_inds", {})

    evidence = []

    if isinstance(gold_inds, dict):

        for value in gold_inds.values():

            if isinstance(value, list):
                evidence.extend(value)

            elif isinstance(value, str):
                evidence.append(value)

    elif isinstance(gold_inds, list):
        evidence.extend(gold_inds)

    elif isinstance(gold_inds, str):
        evidence.append(gold_inds)

    return evidence


def get_example_id(example):
    """Return the document-level ID for a FinQA example.

    FinQA stores the filing identifier in example["id"], e.g.
    "UNP/2009/page_34.pdf-1".  This is the *page-level* key that
    every chunk for that example carries in chunk["example_id"].
    """
    return str(example.get("id", ""))


def evidence_matches(retrieved_text, gold_evidence):

    retrieved_text = retrieved_text.strip().lower()

    for gold_text in gold_evidence:

        if not isinstance(gold_text, str):
            continue

        gold_text = gold_text.strip().lower()

        if not gold_text:
            continue

        if gold_text in retrieved_text:
            return True

        if retrieved_text in gold_text:
            return True

    return False


def page_hit(chunk, gold_example_id):
    """Return True when the chunk belongs to the same document as the gold example.

    A chunk is a page-level hit when its ``example_id`` exactly matches
    the gold document identifier, regardless of which specific row or
    sentence was retrieved.
    """
    if not gold_example_id:
        return False

    return chunk.get("example_id", "") == gold_example_id


# ============================================================
# Metrics
# ============================================================

def update_metrics(found_span_rank, found_page_rank, metrics):
    """Update both span-level and page-level recall / MRR counters.

    Parameters
    ----------
    found_span_rank : int or None
        1-based rank of the first retrieved chunk whose text matches
        the gold evidence text.  ``None`` if not found in top-K.
    found_page_rank : int or None
        1-based rank of the first retrieved chunk that belongs to the
        correct document (page-level hit).  ``None`` if not found.
    metrics : dict
        The running metrics dictionary (mutated in place).
    """
    metrics["evaluated"] += 1

    # ------------------------------------------------------------------
    # Span-level (text match)
    # ------------------------------------------------------------------
    if found_span_rank is not None:

        if found_span_rank <= 1:
            metrics["span_recall_at_1"] += 1

        if found_span_rank <= 5:
            metrics["span_recall_at_5"] += 1

        if found_span_rank <= 10:
            metrics["span_recall_at_10"] += 1

        metrics["span_mrr"] += 1.0 / found_span_rank

    # ------------------------------------------------------------------
    # Page-level (document match)
    # ------------------------------------------------------------------
    if found_page_rank is not None:

        if found_page_rank <= 1:
            metrics["page_recall_at_1"] += 1

        if found_page_rank <= 5:
            metrics["page_recall_at_5"] += 1

        if found_page_rank <= 10:
            metrics["page_recall_at_10"] += 1

        metrics["page_mrr"] += 1.0 / found_page_rank


def create_metrics():
    """Create a fresh dual-level metrics dictionary."""
    return {
        "evaluated": 0,
        "no_gold": 0,
        # Span-level: exact text match with gold evidence
        "span_recall_at_1": 0,
        "span_recall_at_5": 0,
        "span_recall_at_10": 0,
        "span_mrr": 0.0,
        # Page-level: retrieved chunk belongs to correct document
        "page_recall_at_1": 0,
        "page_recall_at_5": 0,
        "page_recall_at_10": 0,
        "page_mrr": 0.0,
    }


def print_metrics(method, metrics):
    """Print a formatted dual-level recall table."""
    evaluated = metrics["evaluated"]

    print()
    print("=" * 60)
    print(f"{method} RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Examples evaluated : {evaluated}")
    print(f"No gold evidence   : {metrics['no_gold']}")
    print(f"Top-K evaluated    : {TOP_K}")

    if evaluated == 0:
        return

    print()
    print(f"{'Metric':<20} {'Span-level':>12} {'Page-level':>12}")
    print("-" * 46)

    print(
        f"{'Recall@1':<20}"
        f" {metrics['span_recall_at_1'] / evaluated:>12.4f}"
        f" {metrics['page_recall_at_1'] / evaluated:>12.4f}"
    )

    print(
        f"{'Recall@5':<20}"
        f" {metrics['span_recall_at_5'] / evaluated:>12.4f}"
        f" {metrics['page_recall_at_5'] / evaluated:>12.4f}"
    )

    print(
        f"{'Recall@10':<20}"
        f" {metrics['span_recall_at_10'] / evaluated:>12.4f}"
        f" {metrics['page_recall_at_10'] / evaluated:>12.4f}"
    )

    print(
        f"{'MRR':<20}"
        f" {metrics['span_mrr'] / evaluated:>12.4f}"
        f" {metrics['page_mrr'] / evaluated:>12.4f}"
    )

    print("=" * 60)


def metrics_to_dict(method, metrics):
    """Serialise metrics to a plain dict for JSON export."""
    evaluated = metrics["evaluated"]

    if evaluated == 0:
        return {"method": method, "evaluated": 0}

    return {
        "method": method,
        "evaluated": evaluated,
        "no_gold": metrics["no_gold"],
        "span_recall_at_1": round(metrics["span_recall_at_1"] / evaluated, 4),
        "span_recall_at_5": round(metrics["span_recall_at_5"] / evaluated, 4),
        "span_recall_at_10": round(metrics["span_recall_at_10"] / evaluated, 4),
        "span_mrr": round(metrics["span_mrr"] / evaluated, 4),
        "page_recall_at_1": round(metrics["page_recall_at_1"] / evaluated, 4),
        "page_recall_at_5": round(metrics["page_recall_at_5"] / evaluated, 4),
        "page_recall_at_10": round(metrics["page_recall_at_10"] / evaluated, 4),
        "page_mrr": round(metrics["page_mrr"] / evaluated, 4),
    }


# ============================================================
# BM25
# ============================================================

def tokenize(text):

    return text.lower().split()


def build_bm25(chunks):

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_corpus)


def evaluate_bm25(chunks, examples):

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

    metrics = create_metrics()

    print("\nEvaluating BM25...")

    for example_number, example in enumerate(
        examples,
        start=1
    ):

        question = example.get(
            "qa",
            {}
        ).get("question")

        if not question:
            continue

        gold_evidence = get_gold_evidence(example)
        gold_example_id = get_example_id(example)

        if not gold_evidence:
            metrics["no_gold"] += 1
            continue

        scores = bm25.get_scores(
            tokenize(question)
        )

        top_indices = np.argsort(
            scores
        )[::-1][:TOP_K]

        found_span_rank = None
        found_page_rank = None

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            chunk = chunks[index]

            if found_span_rank is None and evidence_matches(
                chunk.get("text", ""),
                gold_evidence
            ):
                found_span_rank = rank

            if found_page_rank is None and page_hit(
                chunk,
                gold_example_id
            ):
                found_page_rank = rank

            # Stop scanning once both ranks are found
            if found_span_rank is not None and found_page_rank is not None:
                break

        update_metrics(
            found_span_rank,
            found_page_rank,
            metrics
        )

        if example_number % 500 == 0:

            print(
                f"Processed "
                f"{example_number}/{len(examples)} examples..."
            )

    print_metrics(
        "BM25",
        metrics
    )

    return metrics


# ============================================================
# Dense
# ============================================================

def load_dense_model():

    print(
        f"\nLoading model: {MODEL_NAME}"
    )

    return SentenceTransformer(
        MODEL_NAME,
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )


def prepare_valid_examples(examples):

    valid_examples = []

    for example in examples:

        question = example.get(
            "qa",
            {}
        ).get("question")

        if not question:
            continue

        gold_evidence = get_gold_evidence(example)

        if not gold_evidence:
            continue

        valid_examples.append(
            (
                example,
                question,
                gold_evidence
            )
        )

    return valid_examples


def encode_queries(
    model,
    questions
):

    print("\nEncoding queries...")

    embeddings = model.encode(
        questions,
        batch_size=QUERY_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print(
        f"Query embedding shape: "
        f"{embeddings.shape}"
    )

    return embeddings


def evaluate_dense(
    chunks,
    valid_examples,
    embeddings,
    query_embeddings
):

    metrics = create_metrics()

    print(
        "\nEvaluating Dense Retrieval..."
    )

    for example_number, (
        example,
        question,
        gold_evidence
    ) in enumerate(
        valid_examples,
        start=1
    ):

        query_embedding = query_embeddings[
            example_number - 1
        ]

        scores = embeddings @ query_embedding

        top_indices = np.argsort(
            scores
        )[::-1][:TOP_K]

        gold_example_id = get_example_id(example)

        found_span_rank = None
        found_page_rank = None

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            chunk = chunks[index]

            if found_span_rank is None and evidence_matches(
                chunk.get("text", ""),
                gold_evidence
            ):
                found_span_rank = rank

            if found_page_rank is None and page_hit(
                chunk,
                gold_example_id
            ):
                found_page_rank = rank

            if found_span_rank is not None and found_page_rank is not None:
                break

        update_metrics(
            found_span_rank,
            found_page_rank,
            metrics
        )

        if example_number % 500 == 0:

            print(
                f"Processed "
                f"{example_number}/{len(valid_examples)} examples..."
            )

    print_metrics(
        "DENSE",
        metrics
    )

    return metrics


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    bm25_indices,
    dense_indices,
    k=60
):

    rrf_scores = {}

    for rank, index in enumerate(
        bm25_indices,
        start=1
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0.0)
            + 1.0 / (k + rank)
        )

    for rank, index in enumerate(
        dense_indices,
        start=1
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0.0)
            + 1.0 / (k + rank)
        )

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


def evaluate_rrf(
    chunks,
    valid_examples,
    bm25,
    embeddings,
    query_embeddings
):

    metrics = create_metrics()

    print(
        "\nEvaluating RRF..."
    )

    for example_number, (
        example,
        question,
        gold_evidence
    ) in enumerate(
        valid_examples,
        start=1
    ):

        # ----------------------------
        # BM25
        # ----------------------------

        bm25_scores = bm25.get_scores(
            tokenize(question)
        )

        bm25_indices = np.argsort(
            bm25_scores
        )[::-1][:TOP_K]

        # ----------------------------
        # Dense
        # ----------------------------

        query_embedding = query_embeddings[
            example_number - 1
        ]

        dense_scores = (
            embeddings @ query_embedding
        )

        dense_indices = np.argsort(
            dense_scores
        )[::-1][:TOP_K]

        # ----------------------------
        # RRF
        # ----------------------------

        fused = reciprocal_rank_fusion(
            bm25_indices,
            dense_indices,
            RRF_K
        )

        top_indices = [
            index
            for index, score in fused[:TOP_K]
        ]

        # ----------------------------
        # Check gold evidence
        # ----------------------------

        gold_example_id = get_example_id(example)

        found_span_rank = None
        found_page_rank = None

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            chunk = chunks[index]

            if found_span_rank is None and evidence_matches(
                chunk.get("text", ""),
                gold_evidence
            ):
                found_span_rank = rank

            if found_page_rank is None and page_hit(
                chunk,
                gold_example_id
            ):
                found_page_rank = rank

            if found_span_rank is not None and found_page_rank is not None:
                break

        update_metrics(
            found_span_rank,
            found_page_rank,
            metrics
        )

        if example_number % 500 == 0:

            print(
                f"Processed "
                f"{example_number}/{len(valid_examples)} examples..."
            )

    print_metrics(
        "RRF",
        metrics
    )

    return metrics


# ============================================================
# RRF + Cross-Encoder Reranker (Phase 1 Step 1.4)
# ============================================================

def evaluate_rrf_reranked(
    chunks,
    valid_examples,
    bm25,
    embeddings,
    query_embeddings
):
    """Evaluate RRF followed by cross-encoder reranking.

    Takes the top-RERANK_POOL_SIZE RRF candidates, reranks them with
    BAAI/bge-reranker-base, and evaluates Recall/MRR over the reranked
    top-RERANK_TOP_K results.
    """
    # Import here to avoid a hard dependency when not used
    try:
        from cross_encoder_reranker import CrossEncoderReranker
    except ImportError:
        print(
            "\n[ERROR] cross_encoder_reranker not found. "
            "Run from src/retrieval on PYTHONPATH."
        )
        return create_metrics()

    reranker = CrossEncoderReranker()

    metrics = create_metrics()

    print(
        "\nEvaluating RRF + Cross-Encoder Reranker..."
    )
    print(
        f"  Pool size : {RERANK_POOL_SIZE}  ->  keep top {RERANK_TOP_K}"
    )

    for example_number, (
        example,
        question,
        gold_evidence
    ) in enumerate(
        valid_examples,
        start=1
    ):

        # ----------------------------
        # BM25
        # ----------------------------

        bm25_scores = bm25.get_scores(
            tokenize(question)
        )

        bm25_indices = np.argsort(
            bm25_scores
        )[::-1][:TOP_K]

        # ----------------------------
        # Dense
        # ----------------------------

        query_embedding = query_embeddings[
            example_number - 1
        ]

        dense_scores = embeddings @ query_embedding

        dense_indices = np.argsort(
            dense_scores
        )[::-1][:TOP_K]

        # ----------------------------
        # RRF
        # ----------------------------

        fused = reciprocal_rank_fusion(
            bm25_indices,
            dense_indices,
            RRF_K
        )

        pool_indices = [
            index
            for index, score in fused[:RERANK_POOL_SIZE]
        ]

        # ----------------------------
        # Cross-encoder reranking
        # ----------------------------

        pool_chunks = [chunks[i] for i in pool_indices]

        reranked = reranker.rerank(
            question,
            pool_chunks,
            top_k=RERANK_TOP_K
        )

        # ----------------------------
        # Check gold evidence
        # ----------------------------

        gold_example_id = get_example_id(example)

        found_span_rank = None
        found_page_rank = None

        for rank, chunk in enumerate(
            reranked,
            start=1
        ):

            if found_span_rank is None and evidence_matches(
                chunk.get("text", ""),
                gold_evidence
            ):
                found_span_rank = rank

            if found_page_rank is None and page_hit(
                chunk,
                gold_example_id
            ):
                found_page_rank = rank

            if found_span_rank is not None and found_page_rank is not None:
                break

        update_metrics(
            found_span_rank,
            found_page_rank,
            metrics
        )

        if example_number % 500 == 0:

            print(
                f"Processed "
                f"{example_number}/{len(valid_examples)} examples..."
            )

    print_metrics(
        "RRF + Cross-Encoder",
        metrics
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    print("Loading corpus...")

    chunks = load_jsonl(
        CORPUS_PATH
    )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    print(
        "\nLoading FinQA train set..."
    )

    examples = load_json(
        FINQA_PATH
    )

    print(
        f"Loaded {len(examples)} examples."
    )

    print("\nLoading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    model = load_dense_model()

    valid_examples = prepare_valid_examples(
        examples
    )

    print(
        f"\nValid examples: "
        f"{len(valid_examples)}"
    )

    questions = [
        item[1]
        for item in valid_examples
    ]

    query_embeddings = encode_queries(
        model,
        questions
    )

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(
        chunks
    )

    print("\nChoose evaluation method:")
    print("1. BM25")
    print("2. Dense")
    print("3. RRF")
    print("4. RRF + Cross-Encoder Reranker")
    print("5. All (ablation table)")

    choice = input(
        "\nEnter choice: "
    ).strip()

    all_results = []

    if choice == "1":

        m = evaluate_bm25(chunks, examples)
        all_results.append(metrics_to_dict("BM25", m))

    elif choice == "2":

        m = evaluate_dense(
            chunks,
            valid_examples,
            embeddings,
            query_embeddings
        )
        all_results.append(metrics_to_dict("Dense", m))

    elif choice == "3":

        m = evaluate_rrf(
            chunks,
            valid_examples,
            bm25,
            embeddings,
            query_embeddings
        )
        all_results.append(metrics_to_dict("RRF", m))

    elif choice == "4":

        m = evaluate_rrf_reranked(
            chunks,
            valid_examples,
            bm25,
            embeddings,
            query_embeddings
        )
        all_results.append(metrics_to_dict("RRF+CE", m))

    elif choice == "5":

        print("\n[Ablation] Running all methods...\n")

        m1 = evaluate_bm25(chunks, examples)
        all_results.append(metrics_to_dict("BM25", m1))

        m2 = evaluate_dense(
            chunks, valid_examples, embeddings, query_embeddings
        )
        all_results.append(metrics_to_dict("Dense", m2))

        m3 = evaluate_rrf(
            chunks, valid_examples, bm25, embeddings, query_embeddings
        )
        all_results.append(metrics_to_dict("RRF", m3))

        m4 = evaluate_rrf_reranked(
            chunks, valid_examples, bm25, embeddings, query_embeddings
        )
        all_results.append(metrics_to_dict("RRF+CE", m4))

    else:

        print("Invalid choice.")

    # ------------------------------------------------------------------
    # Save results to JSON for the ablation table
    # ------------------------------------------------------------------

    if all_results:

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        output_path = results_dir / "retrieval_ablation.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(
            f"\n[Results saved to {output_path}]"
        )


if __name__ == "__main__":
    main()
