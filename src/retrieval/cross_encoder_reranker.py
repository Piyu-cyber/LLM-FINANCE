"""cross_encoder_reranker.py — Cross-encoder reranking for Phase 1 Step 1.4.

Wraps BAAI/bge-reranker-base (or any compatible HuggingFace cross-encoder)
from the sentence-transformers library.  Accepts the top-N RRF candidates
and returns a re-scored, re-ranked short-list.

Usage (standalone)
------------------
>>> from cross_encoder_reranker import CrossEncoderReranker
>>> reranker = CrossEncoderReranker()
>>> reranked = reranker.rerank(query, candidates, top_k=5)

Each element of *candidates* should be a dict with at least a ``"text"``
key.  The reranker adds a ``"rerank_score"`` key and returns a list sorted
in descending score order.
"""

import torch
from sentence_transformers import CrossEncoder


# Default model — lightweight cross-encoder tuned for retrieval
DEFAULT_MODEL = "BAAI/bge-reranker-base"

# Batch size for cross-encoder inference
BATCH_SIZE = 32


class CrossEncoderReranker:
    """Cross-encoder reranker backed by ``sentence_transformers.CrossEncoder``.

    Parameters
    ----------
    model_name :
        HuggingFace model identifier.  ``BAAI/bge-reranker-base`` is a
        strong, fast baseline; ``BAAI/bge-reranker-v2-m3`` is larger but
        better.  Swap freely for ablation experiments.
    batch_size :
        Number of (query, document) pairs to score in one forward pass.
        Reduce if you hit out-of-memory errors on GPU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[CrossEncoderReranker] Loading {model_name} on {device}")

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score all *candidates* against *query* and return the top-k.

        Parameters
        ----------
        query :
            The natural-language question.
        candidates :
            List of chunk dicts, each with a ``"text"`` field.  Typically
            the top-50 to top-100 RRF results.
        top_k :
            How many reranked results to return.

        Returns
        -------
        list[dict]
            Reranked candidates (descending ``rerank_score``), length
            ``min(top_k, len(candidates))``.  Each dict retains all
            original fields and gains a ``"rerank_score"`` float.
        """
        if not candidates:
            return []

        # Build (query, document_text) pairs for cross-encoder
        pairs = [
            [query, candidate.get("text", "")]
            for candidate in candidates
        ]

        # score() returns a numpy array of floats
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach scores and sort
        scored = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            scored.append(result)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]
