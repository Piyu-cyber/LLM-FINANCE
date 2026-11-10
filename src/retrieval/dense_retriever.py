"""dense_retriever.py — Dense embedding generator and retriever for FinQA using BGE-small.

Phase 2 Enhancements:
- Added batch-wise GPU embedding generation with CUDA profiling.
- Configurable CLI (--chunks, --output, --device, --batch-size).
- Exact normalization, finiteness check, and peak VRAM tracking.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 64


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def generate_embeddings(
    chunks_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Generate dense embeddings for a corpus of chunks on GPU/CPU with profiling."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Dense Retriever] Loading corpus from {chunks_path}...")
    start_total = time.perf_counter()
    chunks = load_corpus(chunks_path)
    print(f"[Dense Retriever] Loaded {len(chunks)} chunks.")

    texts = [c.get("text", "") for c in chunks]

    # Model loading
    print(f"[Dense Retriever] Loading model {model_name} on {device}...")
    start_model_load = time.perf_counter()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model = SentenceTransformer(model_name, device=device)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_time = time.perf_counter() - start_model_load
    print(f"[Dense Retriever] Model loaded in {model_load_time:.2f}s")

    # Embedding generation
    print(f"[Dense Retriever] Encoding {len(texts)} chunks (batch_size={batch_size})...")
    start_encode = time.perf_counter()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_time = time.perf_counter() - start_encode

    # Verification checks
    assert len(embeddings) == len(chunks), f"Row mismatch: {len(embeddings)} vs {len(chunks)}"
    assert np.all(np.isfinite(embeddings)), "Non-finite values detected in generated embeddings!"
    
    # Check L2 normalization (within float32 precision)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit-normalized!"

    # Save embeddings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    total_time = time.perf_counter() - start_total
    throughput = len(chunks) / encode_time if encode_time > 0 else 0.0

    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device == "cuda" and torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    profile_stats = {
        "corpus_path": str(chunks_path),
        "output_path": str(output_path),
        "model_name": model_name,
        "device": device,
        "total_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape),
        "embedding_dimension": int(embeddings.shape[1]),
        "batch_size": batch_size,
        "model_load_time_sec": round(model_load_time, 4),
        "embedding_time_sec": round(encode_time, 4),
        "total_processing_time_sec": round(total_time, 4),
        "chunks_per_second": round(throughput, 2),
        "peak_allocated_vram_mb": round(peak_allocated_mb, 2),
        "peak_reserved_vram_mb": round(peak_reserved_mb, 2),
    }

    print(f"[Dense Retriever] Saved {embeddings.shape} embeddings to {output_path}")
    print(f"[Dense Retriever] Generation time: {encode_time:.2f}s | Throughput: {throughput:.1f} chunks/sec")
    if device == "cuda":
        print(f"[Dense Retriever] Peak Allocated VRAM: {peak_allocated_mb:.1f} MB | Peak Reserved: {peak_reserved_mb:.1f} MB")

    return profile_stats


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
    parser = argparse.ArgumentParser(description="Dense retriever and embedding generator for FinQA.")
    parser.add_argument("--chunks", type=Path, default=None, help="Path to chunks JSONL file")
    parser.add_argument("--output", type=Path, default=None, help="Path to save output embeddings (.npy)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for encoding")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--query", type=str, default=None, help="Query string for interactive search demo")
    args = parser.parse_args()

    if args.chunks and args.output:
        generate_embeddings(
            chunks_path=args.chunks,
            output_path=args.output,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        # Default demo behavior
        corpus_path = args.chunks or Path("data/finqa/train_chunks.jsonl")
        emb_path = args.output or Path("data/finqa/train_embeddings.npy")
        if not corpus_path.exists() or not emb_path.exists():
            print(f"Error: {corpus_path} or {emb_path} does not exist. Specify --chunks and --output.")
            sys.exit(1)

        chunks = load_corpus(corpus_path)
        embeddings = np.load(emb_path)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(args.model, device=device)

        query = args.query or "what is the interest expense in 2009?"
        results = retrieve(query, chunks, embeddings, model, top_k=5)
        for rank, res in enumerate(results, start=1):
            print(f"Rank {rank} (score {res['score']:.4f}): {res.get('text', '')[:100]}")


if __name__ == "__main__":
    main()
