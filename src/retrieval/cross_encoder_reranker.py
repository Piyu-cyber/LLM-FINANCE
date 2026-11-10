"""cross_encoder_reranker.py — Cross-encoder reranking for Phase 1 Step 1.4.

Wraps BAAI/bge-reranker-base (or any compatible HuggingFace cross-encoder)
from the sentence-transformers library. Accepts the top-N RRF candidates
and returns a re-scored, re-ranked short-list.

Phase 1 Enhancements:
- Reuses singleton model instance across retrieval calls to prevent repeated
  CUDA memory allocation and disk loading overhead.
- Supports dependency injection for testing and model swapping.
- Preserves descending score ordering.
"""

from typing import Any, Optional
import torch
from sentence_transformers import CrossEncoder


# Default model — lightweight cross-encoder tuned for retrieval
DEFAULT_MODEL = "BAAI/bge-reranker-base"

# Batch size for cross-encoder inference
BATCH_SIZE = 32

# Global singleton cache keyed by (model_name, batch_size)
_GLOBAL_RERANKER_CACHE: dict[tuple[str, int], "CrossEncoderReranker"] = {}


def get_cross_encoder(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> "CrossEncoderReranker":
    """Retrieve or create a cached singleton CrossEncoderReranker.

    Ensures the model weights are allocated on GPU only once per session.
    """
    key = (model_name, batch_size)
    if key not in _GLOBAL_RERANKER_CACHE:
        _GLOBAL_RERANKER_CACHE[key] = CrossEncoderReranker(
            model_name=model_name,
            batch_size=batch_size,
        )
    return _GLOBAL_RERANKER_CACHE[key]


def reset_cross_encoder_cache() -> None:
    """Clear all cached CrossEncoderReranker singletons (useful in testing)."""
    global _GLOBAL_RERANKER_CACHE
    _GLOBAL_RERANKER_CACHE.clear()


class CrossEncoderReranker:
    """Cross-encoder reranker backed by ``sentence_transformers.CrossEncoder``.

    Parameters
    ----------
    model_name :
        HuggingFace model identifier.
    batch_size :
        Number of (query, document) pairs to score in one forward pass.
    model :
        Optional pre-instantiated model for dependency injection.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = BATCH_SIZE,
        model: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if model is not None:
            self.model = model
        else:
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
        """Score all *candidates* against *query* and return the top-k sorted by score descending.

        Parameters
        ----------
        query :
            The natural-language question.
        candidates :
            List of chunk dicts, each with a ``"text"`` field.
        top_k :
            How many reranked results to return.

        Returns
        -------
        list[dict]
            Reranked candidates (descending ``rerank_score``), length
            ``min(top_k, len(candidates))``. Each dict retains all
            original fields and gains a ``"rerank_score"`` float.
        """
        if not candidates:
            return []

        pairs = [
            [query, candidate.get("text", "")]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored = []
        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            scored.append(result)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
