import json
import numpy as np

from rank_bm25 import BM25Okapi


CORPUS_PATH = "data/finqa/train_chunks.jsonl"
FINQA_PATH = "data/finqa/dataset/train.json"

TOP_K = 100


def load_jsonl(path):
    """Load a JSONL file: one JSON object per line."""
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                data.append(json.loads(line))

    return data


def load_json(path):
    """Load a normal JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    """Simple whitespace tokenizer."""
    return text.lower().split()


def build_bm25(chunks):
    """Build BM25 index over all corpus chunks."""

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_corpus)


def get_gold_evidence(example):
    """
    Extract gold evidence from a FinQA example.

    FinQA stores evidence references in qa["gold_inds"].
    The values identify evidence from pre_text, post_text,
    or table content.
    """

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
    """
    Check whether the retrieved chunk contains one of
    the gold evidence strings.
    """

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


def evaluate_bm25(chunks, examples):

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    reciprocal_rank_sum = 0.0

    evaluated = 0
    no_gold_evidence = 0

    print("\nEvaluating BM25...")

    for example_number, example in enumerate(examples, start=1):

        qa = example.get("qa", {})

        question = qa.get("question")

        if not question:
            continue

        gold_evidence = get_gold_evidence(example)

        if not gold_evidence:
            no_gold_evidence += 1
            continue

        scores = bm25.get_scores(
            tokenize(question)
        )

        top_indices = np.argsort(scores)[::-1][:TOP_K]

        found_rank = None

        for rank, index in enumerate(top_indices, start=1):

            retrieved_chunk = chunks[index]

            retrieved_text = retrieved_chunk.get("text", "")

            if evidence_matches(
                retrieved_text,
                gold_evidence
            ):
                found_rank = rank
                break

        evaluated += 1

        if found_rank is not None:

            if found_rank <= 1:
                recall_at_1 += 1

            if found_rank <= 5:
                recall_at_5 += 1

            if found_rank <= 10:
                recall_at_10 += 1

            reciprocal_rank_sum += 1.0 / found_rank

        if example_number % 500 == 0:
            print(
                f"Processed {example_number}/{len(examples)} examples..."
            )

    if evaluated == 0:

        print("\nNo examples could be evaluated.")

        print(
            "Examples without gold evidence:",
            no_gold_evidence
        )

        return

    recall_1 = recall_at_1 / evaluated
    recall_5 = recall_at_5 / evaluated
    recall_10 = recall_at_10 / evaluated
    mrr = reciprocal_rank_sum / evaluated

    print("\n")
    print("=" * 55)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 55)

    print(f"Examples evaluated : {evaluated}")
    print(f"No gold evidence   : {no_gold_evidence}")
    print(f"Top-K evaluated    : {TOP_K}")

    print()

    print(f"Recall@1           : {recall_1:.4f}")
    print(f"Recall@5           : {recall_5:.4f}")
    print(f"Recall@10          : {recall_10:.4f}")
    print(f"MRR                : {mrr:.4f}")

    print("=" * 55)


def main():

    print("Loading corpus...")

    chunks = load_jsonl(CORPUS_PATH)

    print(f"Loaded {len(chunks)} chunks.")

    print("\nLoading FinQA train set...")

    examples = load_json(FINQA_PATH)

    print(f"Loaded {len(examples)} examples.")

    evaluate_bm25(
        chunks,
        examples
    )


if __name__ == "__main__":
    main()