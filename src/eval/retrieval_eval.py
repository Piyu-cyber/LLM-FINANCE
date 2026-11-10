"""retrieval_eval.py — Dual-level retrieval evaluation and benchmark ablation for FinQA.

Phase 1 Upgrades:
- Distinguishes Hit@K from true multi-span Recall@K.
- Implements true multi-span Recall@K when multiple gold spans exist.
- Removes unsafe bidirectional substring matching ('retrieved in gold' inflation).
- Headless execution with argparse CLI (split, method, limit, output, device).
- Supports execution from repo root without manual PYTHONPATH manipulation.
- Preserves old benchmark result files by saving to results/retrieval_ablation_phase1.json.
"""

import sys
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Bootstrap import paths for direct execution from repository root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _subdir in ["src", "src/retrieval", "src/verification", "src/eval"]:
    _p = str(_REPO_ROOT / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from query_parser import parse_query, metadata_boost


DEFAULT_CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")
DEFAULT_EMBEDDINGS_PATH = Path("data/finqa/train_embeddings.npy")
DATASET_DIR = Path("data/finqa/dataset")

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"

TOP_K = 100
QUERY_BATCH_SIZE = 64
RRF_K = 60
RERANK_POOL_SIZE = 50
RERANK_TOP_K = 10


# ============================================================
# Loading
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Gold Evidence and Span Normalization
# ============================================================

def get_gold_evidence(example: Dict[str, Any]) -> List[str]:
    """Extract gold evidence text spans from a FinQA example."""
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

    # Filter non-string or empty spans
    return [str(e).strip() for e in evidence if e and isinstance(e, str) and str(e).strip()]


def get_example_id(example: Dict[str, Any]) -> str:
    """Return the document-level page ID for a FinQA example."""
    return str(example.get("id", "")).strip()


def normalize_evidence_text(text: str) -> str:
    """Normalize text for evidence comparison (lowercase, trimmed, collapsed whitespace)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def span_matches(retrieved_text: str, gold_text: str) -> bool:
    """Check whether retrieved text properly matches gold evidence span.

    Removes unsafe bidirectional substring matching ('retrieved in gold').
    A retrieved text matches gold if:
    1. The normalized gold text is contained in the retrieved chunk.
    2. Or substantial multi-token overlap exists (>= 4 tokens, Jaccard >= 0.8).
    """
    if not retrieved_text or not gold_text:
        return False

    r_clean = normalize_evidence_text(retrieved_text)
    g_clean = normalize_evidence_text(gold_text)

    if not r_clean or not g_clean:
        return False

    # 1. Gold span is fully contained in retrieved chunk
    if g_clean in r_clean:
        return True

    r_tokens = re.findall(r"\b\w+\b", r_clean)
    g_tokens = re.findall(r"\b\w+\b", g_clean)

    if not r_tokens or not g_tokens:
        return False

    # 2. If retrieved text is shorter than gold, require substantial phrase match (>= 4 tokens)
    if r_clean in g_clean and len(r_tokens) >= 4:
        return True

    # 3. High token overlap (for slightly different whitespace/formatting in tables)
    r_set = set(r_tokens)
    g_set = set(g_tokens)
    if len(g_set) >= 4 and len(r_set) >= 4:
        common = r_set.intersection(g_set)
        if len(common) / len(g_set) >= 0.8:
            return True

    return False


def evidence_matches(retrieved_text: str, gold_evidence: List[str]) -> bool:
    """Check whether retrieved chunk satisfies at least one gold evidence item."""
    if not retrieved_text or not gold_evidence:
        return False
    for gold in gold_evidence:
        if span_matches(retrieved_text, gold):
            return True
    return False


def page_hit(chunk: Dict[str, Any], gold_example_id: str) -> bool:
    """Return True when chunk belongs to the same document page as gold example."""
    if not gold_example_id:
        return False
    return str(chunk.get("example_id", "")).strip() == gold_example_id


# ============================================================
# Metrics Implementation: Hit@K vs Multi-Span Recall@K
# ============================================================

def create_metrics() -> Dict[str, Any]:
    """Create a new metric accumulation dictionary."""
    return {
        "evaluated": 0,
        "no_gold": 0,
        # Hit@K (at least 1 gold span found)
        "span_hit_at_1": 0.0,
        "span_hit_at_5": 0.0,
        "span_hit_at_10": 0.0,
        # True multi-span Recall@K (fraction of all gold spans found)
        "span_recall_at_1": 0.0,
        "span_recall_at_5": 0.0,
        "span_recall_at_10": 0.0,
        "span_mrr": 0.0,
        # Page-level metrics
        "page_hit_at_1": 0.0,
        "page_hit_at_5": 0.0,
        "page_hit_at_10": 0.0,
        "page_recall_at_1": 0.0,
        "page_recall_at_5": 0.0,
        "page_recall_at_10": 0.0,
        "page_mrr": 0.0,
    }


def compute_example_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    gold_evidence: List[str],
    gold_example_id: str,
    top_ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, Any]:
    """Compute exact per-example retrieval metrics distinguishing Hit@K from true Recall@K."""
    num_gold = len(gold_evidence)
    res: Dict[str, Any] = {}

    first_span_rank = None
    first_page_rank = None

    matched_gold_indices = set()
    spans_matched_at_k = {k: 0 for k in top_ks}
    page_hit_at_k = {k: False for k in top_ks}

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        c_text = chunk.get("text", "")
        is_page = (gold_example_id and page_hit(chunk, gold_example_id))

        if is_page and first_page_rank is None:
            first_page_rank = rank

        chunk_matched_any = False
        for g_idx, gold_span in enumerate(gold_evidence):
            if span_matches(c_text, gold_span):
                chunk_matched_any = True
                matched_gold_indices.add(g_idx)

        if chunk_matched_any and first_span_rank is None:
            first_span_rank = rank

        for k in top_ks:
            if rank <= k:
                spans_matched_at_k[k] = len(matched_gold_indices)
                if is_page:
                    page_hit_at_k[k] = True

    for k in top_ks:
        covered = spans_matched_at_k[k]
        res[f"span_recall_at_{k}"] = (covered / num_gold) if num_gold > 0 else 0.0
        res[f"span_hit_at_{k}"] = 1.0 if covered > 0 else 0.0
        p_hit = 1.0 if page_hit_at_k[k] else 0.0
        res[f"page_hit_at_{k}"] = p_hit
        res[f"page_recall_at_{k}"] = p_hit

    res["span_mrr"] = (1.0 / first_span_rank) if first_span_rank is not None else 0.0
    res["page_mrr"] = (1.0 / first_page_rank) if first_page_rank is not None else 0.0
    res["first_span_rank"] = first_span_rank
    res["first_page_rank"] = first_page_rank

    return res


def update_metrics(found_span_rank, found_page_rank, metrics: Dict[str, Any]) -> None:
    """Backward-compatible helper updating metrics from first found ranks."""
    metrics["evaluated"] += 1

    # Span-level
    if found_span_rank is not None:
        if found_span_rank <= 1:
            metrics["span_recall_at_1"] += 1.0
            metrics["span_hit_at_1"] += 1.0
        if found_span_rank <= 5:
            metrics["span_recall_at_5"] += 1.0
            metrics["span_hit_at_5"] += 1.0
        if found_span_rank <= 10:
            metrics["span_recall_at_10"] += 1.0
            metrics["span_hit_at_10"] += 1.0
        metrics["span_mrr"] += 1.0 / found_span_rank

    # Page-level
    if found_page_rank is not None:
        if found_page_rank <= 1:
            metrics["page_recall_at_1"] += 1.0
            metrics["page_hit_at_1"] += 1.0
        if found_page_rank <= 5:
            metrics["page_recall_at_5"] += 1.0
            metrics["page_hit_at_5"] += 1.0
        if found_page_rank <= 10:
            metrics["page_recall_at_10"] += 1.0
            metrics["page_hit_at_10"] += 1.0
        metrics["page_mrr"] += 1.0 / found_page_rank


def update_example_metrics(metrics: Dict[str, Any], example_metrics: Dict[str, Any]) -> None:
    """Accumulate per-example multi-span and page metrics into the aggregate metrics."""
    metrics["evaluated"] += 1
    for k in (1, 5, 10):
        metrics[f"span_hit_at_{k}"] += example_metrics[f"span_hit_at_{k}"]
        metrics[f"span_recall_at_{k}"] += example_metrics[f"span_recall_at_{k}"]
        metrics[f"page_hit_at_{k}"] += example_metrics[f"page_hit_at_{k}"]
        metrics[f"page_recall_at_{k}"] += example_metrics[f"page_recall_at_{k}"]

    metrics["span_mrr"] += example_metrics["span_mrr"]
    metrics["page_mrr"] += example_metrics["page_mrr"]


def metrics_to_dict(method: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert accumulated metrics to a clean serializable dictionary."""
    evaluated = metrics["evaluated"]
    if evaluated == 0:
        return {"method": method, "evaluated": 0}

    return {
        "method": method,
        "evaluated": evaluated,
        "no_gold": metrics.get("no_gold", 0),
        # Hit@K
        "span_hit_at_1": round(metrics["span_hit_at_1"] / evaluated, 4),
        "span_hit_at_5": round(metrics["span_hit_at_5"] / evaluated, 4),
        "span_hit_at_10": round(metrics["span_hit_at_10"] / evaluated, 4),
        # True Multi-Span Recall@K
        "span_recall_at_1": round(metrics["span_recall_at_1"] / evaluated, 4),
        "span_recall_at_5": round(metrics["span_recall_at_5"] / evaluated, 4),
        "span_recall_at_10": round(metrics["span_recall_at_10"] / evaluated, 4),
        "span_mrr": round(metrics["span_mrr"] / evaluated, 4),
        # Page-Level Recall / Hit
        "page_hit_at_1": round(metrics["page_hit_at_1"] / evaluated, 4),
        "page_hit_at_5": round(metrics["page_hit_at_5"] / evaluated, 4),
        "page_hit_at_10": round(metrics["page_hit_at_10"] / evaluated, 4),
        "page_recall_at_1": round(metrics["page_recall_at_1"] / evaluated, 4),
        "page_recall_at_5": round(metrics["page_recall_at_5"] / evaluated, 4),
        "page_recall_at_10": round(metrics["page_recall_at_10"] / evaluated, 4),
        "page_mrr": round(metrics["page_mrr"] / evaluated, 4),
    }


def print_metrics(method: str, metrics: Dict[str, Any]) -> None:
    evaluated = metrics["evaluated"]
    print()
    print("=" * 66)
    print(f" {method} RETRIEVAL EVALUATION")
    print("=" * 66)
    print(f"Examples evaluated : {evaluated}")
    print(f"No gold evidence   : {metrics.get('no_gold', 0)}")
    print(f"Top-K evaluated    : {TOP_K}")

    if evaluated == 0:
        return

    print("-" * 66)
    print(f"{'Metric':<18} {'Span Hit':>12} {'Span Recall':>12} {'Page Hit':>12}")
    print("-" * 66)

    for k in (1, 5, 10):
        s_hit = metrics[f"span_hit_at_{k}"] / evaluated
        s_rec = metrics[f"span_recall_at_{k}"] / evaluated
        p_hit = metrics[f"page_hit_at_{k}"] / evaluated
        print(f"{f'Rank@{k}':<18} {s_hit:>12.4f} {s_rec:>12.4f} {p_hit:>12.4f}")

    s_mrr = metrics["span_mrr"] / evaluated
    p_mrr = metrics["page_mrr"] / evaluated
    print("-" * 66)
    print(f"{'MRR':<18} {s_mrr:>12.4f} {'-':>12} {p_mrr:>12.4f}")
    print("=" * 66)


# ============================================================
# BM25 Helper
# ============================================================

def tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized_corpus = [tokenize(chunk.get("text", "")) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


# ============================================================
# Evaluators
# ============================================================

def prepare_valid_examples(examples: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str, List[str]]]:
    valid = []
    for example in examples:
        question = example.get("qa", {}).get("question")
        if not question:
            continue
        gold = get_gold_evidence(example)
        if not gold:
            continue
        valid.append((example, question, gold))
    return valid


def evaluate_bm25(chunks: List[Dict[str, Any]], valid_examples: List[Tuple[Dict[str, Any], str, List[str]]], bm25: BM25Okapi) -> Dict[str, Any]:
    metrics = create_metrics()
    print("\nEvaluating BM25...")

    for ex_num, (example, question, gold_evidence) in enumerate(valid_examples, start=1):
        scores = bm25.get_scores(tokenize(question))
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        retrieved_chunks = [chunks[i] for i in top_indices]

        ex_m = compute_example_metrics(retrieved_chunks, gold_evidence, get_example_id(example))
        update_example_metrics(metrics, ex_m)

        if ex_num % 500 == 0:
            print(f"Processed {ex_num}/{len(valid_examples)} examples...")

    print_metrics("BM25", metrics)
    return metrics


def evaluate_dense(
    chunks: List[Dict[str, Any]],
    valid_examples: List[Tuple[Dict[str, Any], str, List[str]]],
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> Dict[str, Any]:
    metrics = create_metrics()
    print("\nEvaluating Dense Retrieval...")

    for ex_num, (example, question, gold_evidence) in enumerate(valid_examples, start=1):
        query_embedding = query_embeddings[ex_num - 1]
        scores = embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        retrieved_chunks = [chunks[i] for i in top_indices]

        ex_m = compute_example_metrics(retrieved_chunks, gold_evidence, get_example_id(example))
        update_example_metrics(metrics, ex_m)

        if ex_num % 500 == 0:
            print(f"Processed {ex_num}/{len(valid_examples)} examples...")

    print_metrics("Dense", metrics)
    return metrics


def reciprocal_rank_fusion(bm25_indices, dense_indices, k=60):
    rrf_scores = {}
    for rank, index in enumerate(bm25_indices, start=1):
        rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (k + rank)
    for rank, index in enumerate(dense_indices, start=1):
        rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (k + rank)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def evaluate_rrf(
    chunks: List[Dict[str, Any]],
    valid_examples: List[Tuple[Dict[str, Any], str, List[str]]],
    bm25: BM25Okapi,
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> Dict[str, Any]:
    metrics = create_metrics()
    print("\nEvaluating RRF (Vanilla)...")

    for ex_num, (example, question, gold_evidence) in enumerate(valid_examples, start=1):
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K]

        query_embedding = query_embeddings[ex_num - 1]
        dense_scores = embeddings @ query_embedding
        dense_indices = np.argsort(dense_scores)[::-1][:TOP_K]

        fused = reciprocal_rank_fusion(bm25_indices, dense_indices, RRF_K)
        retrieved_chunks = [chunks[idx] for idx, _ in fused[:TOP_K]]

        ex_m = compute_example_metrics(retrieved_chunks, gold_evidence, get_example_id(example))
        update_example_metrics(metrics, ex_m)

        if ex_num % 500 == 0:
            print(f"Processed {ex_num}/{len(valid_examples)} examples...")

    print_metrics("RRF", metrics)
    return metrics


def evaluate_rrf_metadata(
    chunks: List[Dict[str, Any]],
    valid_examples: List[Tuple[Dict[str, Any], str, List[str]]],
    bm25: BM25Okapi,
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> Dict[str, Any]:
    metrics = create_metrics()
    print("\nEvaluating RRF + Metadata Boost...")

    for ex_num, (example, question, gold_evidence) in enumerate(valid_examples, start=1):
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K]

        query_embedding = query_embeddings[ex_num - 1]
        dense_scores = embeddings @ query_embedding
        dense_indices = np.argsort(dense_scores)[::-1][:TOP_K]

        fused = reciprocal_rank_fusion(bm25_indices, dense_indices, RRF_K)

        meta = parse_query(question)
        c_ids = meta["company_ids"]
        y_ids = meta["fiscal_years"]

        boosted = []
        for index, score in fused[:TOP_K]:
            b = metadata_boost(chunks[index], c_ids, y_ids)
            boosted.append((index, score * b))
        boosted.sort(key=lambda x: x[1], reverse=True)
        retrieved_chunks = [chunks[idx] for idx, _ in boosted[:TOP_K]]

        ex_m = compute_example_metrics(retrieved_chunks, gold_evidence, get_example_id(example))
        update_example_metrics(metrics, ex_m)

        if ex_num % 500 == 0:
            print(f"Processed {ex_num}/{len(valid_examples)} examples...")

    print_metrics("RRF + Metadata Boost", metrics)
    return metrics


def evaluate_rrf_reranked(
    chunks: List[Dict[str, Any]],
    valid_examples: List[Tuple[Dict[str, Any], str, List[str]]],
    bm25: BM25Okapi,
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    apply_metadata_boosting: bool = False,
) -> Dict[str, Any]:
    try:
        from cross_encoder_reranker import get_cross_encoder
        reranker = get_cross_encoder(model_name=CROSS_ENCODER_MODEL_NAME)
    except ImportError:
        print("\n[ERROR] cross_encoder_reranker not found.")
        return create_metrics()

    metrics = create_metrics()
    method_name = "RRF + Meta + Cross-Encoder" if apply_metadata_boosting else "RRF + Cross-Encoder"
    print(f"\nEvaluating {method_name}...")
    print(f"  Pool size : {RERANK_POOL_SIZE}  ->  keep top {RERANK_TOP_K}")

    for ex_num, (example, question, gold_evidence) in enumerate(valid_examples, start=1):
        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K]

        query_embedding = query_embeddings[ex_num - 1]
        dense_scores = embeddings @ query_embedding
        dense_indices = np.argsort(dense_scores)[::-1][:TOP_K]

        fused = reciprocal_rank_fusion(bm25_indices, dense_indices, RRF_K)

        if apply_metadata_boosting:
            meta = parse_query(question)
            c_ids = meta["company_ids"]
            y_ids = meta["fiscal_years"]

            boosted = []
            for index, score in fused[:RERANK_POOL_SIZE]:
                b = metadata_boost(chunks[index], c_ids, y_ids)
                boosted.append((index, score * b))
            boosted.sort(key=lambda x: x[1], reverse=True)
            pool_indices = [idx for idx, _ in boosted]
        else:
            pool_indices = [index for index, score in fused[:RERANK_POOL_SIZE]]

        pool_chunks = [chunks[i] for i in pool_indices]
        reranked = reranker.rerank(question, pool_chunks, top_k=RERANK_TOP_K)

        ex_m = compute_example_metrics(reranked, gold_evidence, get_example_id(example), top_ks=(1, 5, 10))
        update_example_metrics(metrics, ex_m)

        if ex_num % 500 == 0:
            print(f"Processed {ex_num}/{len(valid_examples)} examples...")

    print_metrics(method_name, metrics)
    return metrics


# ============================================================
# Main (Headless CLI with Argument Parsing)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval models on FinQA (Phase 1).")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev", help="Dataset split to evaluate on (default: dev)")
    parser.add_argument("--method", choices=["bm25", "dense", "rrf", "rrf_meta", "rrf_ce", "rrf_meta_ce", "all"], default="all", help="Method to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries evaluated")
    parser.add_argument("--output", type=str, default="results/retrieval_ablation_phase1.json", help="Path to save result JSON")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('cuda' or 'cpu')")
    parser.add_argument("--corpus-path", type=str, default=str(DEFAULT_CORPUS_PATH), help="Corpus JSONL path")
    parser.add_argument("--embeddings-path", type=str, default=str(DEFAULT_EMBEDDINGS_PATH), help="Precomputed embeddings path")
    return parser.parse_args()


def main():
    args = parse_args()

    split_file = DATASET_DIR / f"{args.split}.json"
    corpus_path = Path(args.corpus_path)
    embeddings_path = Path(args.embeddings_path)

    print(f"LLM-FINANCE Retrieval Benchmark Evaluation (Phase 1)")
    print(f"Split: {args.split} ({split_file}) | Method: {args.method} | Limit: {args.limit}")

    if not split_file.exists():
        print(f"[ERROR] Split file {split_file} not found.")
        sys.exit(1)
    if not corpus_path.exists():
        print(f"[ERROR] Corpus file {corpus_path} not found.")
        sys.exit(1)

    print(f"Loading corpus from {corpus_path}...")
    chunks = load_jsonl(corpus_path)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading dataset split from {split_file}...")
    examples = load_json(split_file)
    print(f"Loaded {len(examples)} examples in {args.split} split.")

    valid_examples = prepare_valid_examples(examples)
    print(f"Valid examples with gold evidence: {len(valid_examples)}")

    if args.limit and args.limit > 0:
        valid_examples = valid_examples[:args.limit]
        print(f"Evaluating on limited subset of {len(valid_examples)} examples.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    # Build BM25 index if needed
    bm25 = None
    if args.method in ("bm25", "rrf", "rrf_meta", "rrf_ce", "rrf_meta_ce", "all"):
        print("Building BM25 index...")
        bm25 = build_bm25(chunks)

    # Load dense model and embeddings if needed
    embeddings = None
    query_embeddings = None
    if args.method in ("dense", "rrf", "rrf_meta", "rrf_ce", "rrf_meta_ce", "all"):
        if not embeddings_path.exists():
            print(f"[ERROR] Embeddings file {embeddings_path} not found.")
            sys.exit(1)
        print(f"Loading corpus embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
        print(f"Corpus embeddings shape: {embeddings.shape}")

        print(f"Loading dense model {DENSE_MODEL_NAME} on {device}...")
        dense_model = SentenceTransformer(DENSE_MODEL_NAME, device=device)

        questions = [item[1] for item in valid_examples]
        print(f"Encoding {len(questions)} queries...")
        query_embeddings = dense_model.encode(
            questions,
            batch_size=QUERY_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    all_results = []

    if args.method in ("bm25", "all"):
        m = evaluate_bm25(chunks, valid_examples, bm25)
        all_results.append(metrics_to_dict("BM25", m))

    if args.method in ("dense", "all"):
        m = evaluate_dense(chunks, valid_examples, embeddings, query_embeddings)
        all_results.append(metrics_to_dict("Dense", m))

    if args.method in ("rrf", "all"):
        m = evaluate_rrf(chunks, valid_examples, bm25, embeddings, query_embeddings)
        all_results.append(metrics_to_dict("RRF", m))

    if args.method in ("rrf_meta", "all"):
        m = evaluate_rrf_metadata(chunks, valid_examples, bm25, embeddings, query_embeddings)
        all_results.append(metrics_to_dict("RRF+Meta", m))

    if args.method in ("rrf_ce", "all"):
        m = evaluate_rrf_reranked(chunks, valid_examples, bm25, embeddings, query_embeddings, apply_metadata_boosting=False)
        all_results.append(metrics_to_dict("RRF+CE", m))

    if args.method in ("rrf_meta_ce", "all"):
        m = evaluate_rrf_reranked(chunks, valid_examples, bm25, embeddings, query_embeddings, apply_metadata_boosting=True)
        all_results.append(metrics_to_dict("RRF+Meta+CE", m))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split": args.split,
            "queries_evaluated": len(valid_examples),
            "query_ids": [get_example_id(item[0]) for item in valid_examples],
            "corpus_path": str(corpus_path),
            "dense_model": DENSE_MODEL_NAME,
            "cross_encoder_model": CROSS_ENCODER_MODEL_NAME,
            "parameters": {
                "top_k": TOP_K,
                "rrf_k": RRF_K,
                "pool_size": RERANK_POOL_SIZE,
                "rerank_top_k": RERANK_TOP_K,
            },
        },
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Successfully saved {len(all_results)} method results to {output_path}]")


if __name__ == "__main__":
    main()
