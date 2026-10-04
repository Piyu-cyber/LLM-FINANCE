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


# ============================================================
# Metrics
# ============================================================

def update_metrics(found_rank, metrics):

    metrics["evaluated"] += 1

    if found_rank is None:
        return

    if found_rank <= 1:
        metrics["recall_at_1"] += 1

    if found_rank <= 5:
        metrics["recall_at_5"] += 1

    if found_rank <= 10:
        metrics["recall_at_10"] += 1

    metrics["mrr"] += 1.0 / found_rank


def create_metrics():

    return {
        "evaluated": 0,
        "no_gold": 0,
        "recall_at_1": 0,
        "recall_at_5": 0,
        "recall_at_10": 0,
        "mrr": 0.0,
    }


def print_metrics(method, metrics):

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

    print(
        f"Recall@1           : "
        f"{metrics['recall_at_1'] / evaluated:.4f}"
    )

    print(
        f"Recall@5           : "
        f"{metrics['recall_at_5'] / evaluated:.4f}"
    )

    print(
        f"Recall@10          : "
        f"{metrics['recall_at_10'] / evaluated:.4f}"
    )

    print(
        f"MRR                : "
        f"{metrics['mrr'] / evaluated:.4f}"
    )

    print("=" * 60)


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

        if not gold_evidence:
            metrics["no_gold"] += 1
            continue

        scores = bm25.get_scores(
            tokenize(question)
        )

        top_indices = np.argsort(
            scores
        )[::-1][:TOP_K]

        found_rank = None

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            if evidence_matches(
                chunks[index].get("text", ""),
                gold_evidence
            ):

                found_rank = rank
                break

        update_metrics(
            found_rank,
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


# ============================================================
# Dense
# ============================================================

def load_dense_model():

    print(
        f"\nLoading model: {MODEL_NAME}"
    )

    return SentenceTransformer(
        MODEL_NAME,
        device="cuda"
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

        found_rank = None

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            if evidence_matches(
                chunks[index].get("text", ""),
                gold_evidence
            ):

                found_rank = rank
                break

        update_metrics(
            found_rank,
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

        found_rank = None

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            if evidence_matches(
                chunks[index].get("text", ""),
                gold_evidence
            ):

                found_rank = rank
                break

        update_metrics(
            found_rank,
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

    choice = input(
        "\nEnter choice: "
    ).strip()

    if choice == "1":

        evaluate_bm25(
            chunks,
            examples
        )

    elif choice == "2":

        evaluate_dense(
            chunks,
            valid_examples,
            embeddings,
            query_embeddings
        )

    elif choice == "3":

        evaluate_rrf(
            chunks,
            valid_examples,
            bm25,
            embeddings,
            query_embeddings
        )

    else:

        print("Invalid choice.")


if __name__ == "__main__":
    main()