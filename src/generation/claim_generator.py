"""claim_generator.py — End-to-end financial claim generation and verification orchestrator.

Coordinates the complete verifiable financial RAG pipeline:
1. Retrieval execution (Dense + Parent-Child or Weighted RRF + Parent-Child).
2. Evidence context assembly with chunk-atomic token budget management.
3. Constrained financial prompt generation (accounting polarity, verbatim span citations).
4. Model inference via LLMClient (Mock, Local Qwen, or API).
5. Strict output parsing and schema validation against CLAIM_SCHEMA.
6. Symbolic verification via run_verification_pipeline (the final authority).
7. Structured, immutable ClaimGenerationResult with detailed diagnostics and latencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

# Generation components
try:
    from .context_builder import ContextBuildResult, ContextBuilder
    from .generation_config import GenerationConfig
    from .llm_client import GenerationResult, LLMClient, MockLLMClient
    from .output_parser import OutputParser, ParseResult
    from .prompts import build_generation_prompt
    from .repair_engine import (
        RepairAttemptAudit,
        RepairEngine,
        RepairabilityClassification,
        classify_repairability,
        compute_repair_metrics,
    )
except (ImportError, ValueError):
    from context_builder import ContextBuildResult, ContextBuilder
    from generation_config import GenerationConfig
    from llm_client import GenerationResult, LLMClient, MockLLMClient
    from output_parser import OutputParser, ParseResult
    from prompts import build_generation_prompt
    from repair_engine import (
        RepairAttemptAudit,
        RepairEngine,
        RepairabilityClassification,
        classify_repairability,
        compute_repair_metrics,
    )

# Verification components
try:
    from src.verification.pipeline import run_verification_pipeline
    from src.verification.validator import build_chunk_lookup, validate_operand
except Exception:
    try:
        from pipeline import run_verification_pipeline
        from validator import build_chunk_lookup, validate_operand
    except Exception:
        import sys
        from pathlib import Path

        v_dir = Path(__file__).resolve().parent.parent / "verification"
        if str(v_dir) not in sys.path:
            sys.path.insert(0, str(v_dir))
        from pipeline import run_verification_pipeline
        from validator import build_chunk_lookup, validate_operand

# Retrieval components
try:
    from src.retrieval.parent_child_retriever import (
        build_corpus_index,
        expand_chunk,
        expand_chunks,
    )
except Exception:
    try:
        from parent_child_retriever import (
            build_corpus_index,
            expand_chunk,
            expand_chunks,
        )
    except Exception:
        import sys
        from pathlib import Path

        r_dir = Path(__file__).resolve().parent.parent / "retrieval"
        if str(r_dir) not in sys.path:
            sys.path.insert(0, str(r_dir))
        from parent_child_retriever import (
            build_corpus_index,
            expand_chunk,
            expand_chunks,
        )

def get_chunk_uid(chunk: dict) -> str:
    """Return a deterministic stable UID for a chunk (example_id#chunk_id)."""
    ex = str(chunk.get("example_id", chunk.get("parent_id", ""))).strip()
    cid = str(chunk.get("chunk_id", "")).strip()
    return f"{ex}#{cid}" if ex and cid else str(chunk.get("text", "")[:64])


def dense_search(
    query: str,
    model: Any,
    embeddings: Any,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """Compute dense inner-product scores against pre-computed corpus embeddings."""
    try:
        if hasattr(model, "encode"):
            raw_emb = model.encode([query], normalize_embeddings=True)
            query_emb = raw_emb[0]
        else:
            query_emb = [0.0] * len(embeddings[0])

        scores = []
        for vec in embeddings:
            dot = sum(float(a) * float(b) for a, b in zip(vec, query_emb))
            scores.append(dot)

        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(int(i), float(scores[i])) for i in indices]
    except Exception:
        try:
            scores = embeddings @ query_emb
            indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
            return [(int(i), float(scores[i])) for i in indices]
        except Exception:
            return []


def bm25_search(
    query: str,
    bm25: Any,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """Compute BM25 lexical scores for query tokens against indexed corpus."""
    try:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        return [(int(i), float(scores[i])) for i in indices]
    except Exception:
        return []


@dataclass
class ClaimGenerationResult:
    """Documented structured result of end-to-end generation and verification."""

    question: str
    example_id: Optional[str]
    retrieval_config: str
    retrieved_chunk_ids: list[str]
    context_mode: str
    context_token_count: int
    context_is_truncated: bool
    generation_result: Optional[GenerationResult]
    parsed_claim: Optional[dict[str, Any]]
    verification_result: Optional[dict[str, Any]]
    status: str
    detailed_status: str
    verified_answer: Optional[float]
    is_verified: bool
    error_diagnostics: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    context_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    initial_status: Optional[str] = None
    initial_detailed_status: Optional[str] = None
    initial_diagnostics: list[str] = field(default_factory=list)
    repair_attempted: bool = False
    repair_eligible: bool = False
    repair_attempts: int = 0
    repair_audit: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a telemetry and logging dictionary."""
        return {
            "question": self.question,
            "example_id": self.example_id,
            "retrieval_config": self.retrieval_config,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "context_mode": self.context_mode,
            "context_token_count": self.context_token_count,
            "context_is_truncated": self.context_is_truncated,
            "status": self.status,
            "detailed_status": self.detailed_status,
            "verified_answer": self.verified_answer,
            "is_verified": self.is_verified,
            "error_diagnostics": self.error_diagnostics,
            "repair": {
                "attempted": self.repair_attempted,
                "eligible": self.repair_eligible,
                "attempts": self.repair_attempts,
                "initial_status": self.initial_status,
                "initial_detailed_status": self.initial_detailed_status,
                "initial_diagnostics": self.initial_diagnostics,
                "audit": self.repair_audit,
            },
            "latencies_ms": {
                "retrieval": round(self.retrieval_latency_ms, 2),
                "context": round(self.context_latency_ms, 2),
                "generation": round(self.generation_latency_ms, 2),
                "verification": round(self.verification_latency_ms, 2),
                "total": round(self.total_latency_ms, 2),
            },
            "generation": self.generation_result.to_dict() if self.generation_result else None,
            "parsed_claim": self.parsed_claim,
            "verification_result": self.verification_result,
        }


def weighted_reciprocal_rank_fusion(
    bm25_results: list[tuple[Any, float]],
    dense_results: list[tuple[Any, float]],
    k: int = 60,
    w_dense: float = 0.75,
    w_bm25: float = 0.25,
) -> list[tuple[Any, float]]:
    """Merge ranked results using asymmetric Weighted RRF (75% Dense / 25% BM25)."""
    scores: dict[Any, float] = {}

    for rank, (item_id, _) in enumerate(dense_results, start=1):
        scores[item_id] = scores.get(item_id, 0.0) + (w_dense / (k + rank))

    for rank, (item_id, _) in enumerate(bm25_results, start=1):
        scores[item_id] = scores.get(item_id, 0.0) + (w_bm25 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class ClaimGenerator:
    """Orchestrates end-to-end financial question answering and deterministic verification."""

    VALID_RETRIEVAL_CONFIGS = {"dense_parent_child", "weighted_rrf_parent_child", "direct"}

    def __init__(
        self,
        llm_client: LLMClient,
        context_builder: Optional[ContextBuilder] = None,
        output_parser: Optional[OutputParser] = None,
        retrieval_config: str = "dense_parent_child",
        context_mode: str = "deduplicated_parent",
        corpus_index: Optional[dict[str, list[dict]]] = None,
        dense_model: Optional[Any] = None,
        dense_embeddings: Optional[np.ndarray] = None,
        bm25_index: Optional[Any] = None,
        corpus_chunks: Optional[list[dict]] = None,
        repair_engine: Optional[RepairEngine] = None,
        enable_repair: bool = False,
        max_repair_budget: int = 1,
    ) -> None:
        """Initialize ClaimGenerator.

        Parameters
        ----------
        llm_client : LLMClient
            Configured inference client (MockLLMClient, HuggingFaceLocalClient, or ConfigurableAPIClient).
        context_builder : Optional[ContextBuilder]
            Context assembler. If None, instantiated with default settings.
        output_parser : Optional[OutputParser]
            Strict CLAIM_SCHEMA parser. If None, instantiated with default schema.
        retrieval_config : str
            Retrieval strategy ("dense_parent_child", "weighted_rrf_parent_child", "direct").
        context_mode : str
            Context expansion mode ("deduplicated_parent", "parent_expanded", "child_only").
        corpus_index : Optional[dict]
            Mapping example_id -> list of sibling chunks for parent expansion.
        dense_model : Optional[Any]
            Dense embedding model (SentenceTransformer).
        dense_embeddings : Optional[np.ndarray]
            Pre-computed dense corpus embeddings.
        bm25_index : Optional[Any]
            Indexed BM25 corpus (BM25Okapi).
        corpus_chunks : Optional[list[dict]]
            Full corpus chunks list.
        """
        if retrieval_config not in self.VALID_RETRIEVAL_CONFIGS:
            raise ValueError(
                f"Invalid retrieval_config: '{retrieval_config}'. "
                f"Must be one of {self.VALID_RETRIEVAL_CONFIGS}"
            )

        self.llm_client = llm_client
        self.context_builder = context_builder or ContextBuilder(
            default_mode=context_mode,
            token_counter=getattr(llm_client, "count_tokens", None),
        )
        self.output_parser = output_parser or OutputParser()
        self.retrieval_config = retrieval_config
        self.context_mode = context_mode

        self.corpus_chunks = corpus_chunks or []
        self.corpus_index = corpus_index
        if self.corpus_index is None and self.corpus_chunks:
            self.corpus_index = build_corpus_index(self.corpus_chunks)

        self.dense_model = dense_model
        self.dense_embeddings = dense_embeddings
        self.bm25_index = bm25_index

        self.repair_engine = repair_engine or RepairEngine(
            llm_client=self.llm_client,
            output_parser=self.output_parser,
            max_repair_budget=max_repair_budget,
        )
        self.enable_repair = enable_repair
        self.max_repair_budget = max_repair_budget

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        config: Optional[str] = None,
    ) -> list[dict]:
        """Execute candidate retrieval according to the configured strategy."""
        active_config = config or self.retrieval_config

        if active_config == "direct":
            return []

        if not self.corpus_chunks:
            return []

        # 1. Primary: Dense Retrieval
        if active_config == "dense_parent_child":
            if self.dense_model is None or self.dense_embeddings is None:
                return []
            query_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
            scores = self.dense_embeddings @ query_emb
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                chunk = self.corpus_chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                results.append(chunk)
            return results

        # 2. Comparison: Weighted RRF (75% Dense / 25% BM25)
        elif active_config == "weighted_rrf_parent_child":
            if (
                self.dense_model is None
                or self.dense_embeddings is None
                or self.bm25_index is None
            ):
                return []

            dense_res = dense_search(query, self.dense_model, self.dense_embeddings, top_k=100)
            bm25_res = bm25_search(query, self.bm25_index, top_k=100)

            uid_to_chunk = {get_chunk_uid(c): c for c in self.corpus_chunks}
            dense_uids = [
                (get_chunk_uid(self.corpus_chunks[idx]), s)
                for idx, s in dense_res
                if idx < len(self.corpus_chunks)
            ]
            bm25_uids = [
                (get_chunk_uid(self.corpus_chunks[idx]), s)
                for idx, s in bm25_res
                if idx < len(self.corpus_chunks)
            ]

            fused = weighted_reciprocal_rank_fusion(
                bm25_results=bm25_uids,
                dense_results=dense_uids,
                k=60,
                w_dense=0.75,
                w_bm25=0.25,
            )

            results = []
            for uid, score in fused[:top_k]:
                if uid in uid_to_chunk:
                    c = uid_to_chunk[uid].copy()
                    c["score"] = float(score)
                    results.append(c)
            return results

        return []

    def _build_verification_chunk_pool(
        self,
        retrieved_chunks: Sequence[dict],
        corpus_index: Optional[dict[str, list[dict]]],
    ) -> list[dict]:
        """Assemble the complete pool of visible chunks for the verifier.

        Ensures that if parent expansion exposed constituent table rows or narrative
        lines to the model in the prompt, the verifier's lookup can find them.
        """
        pool: list[dict] = list(retrieved_chunks)
        seen_keys = {
            (c.get("example_id", c.get("parent_id", "")), c.get("chunk_id", ""))
            for c in pool
        }

        if corpus_index:
            example_ids = {
                c.get("example_id", c.get("parent_id", "")) for c in retrieved_chunks
            }
            for ex_id in example_ids:
                if ex_id and ex_id in corpus_index:
                    for sibling in corpus_index[ex_id]:
                        key = (ex_id, sibling.get("chunk_id", ""))
                        if key not in seen_keys:
                            pool.append(sibling)
                            seen_keys.add(key)

        return pool

    def generate_and_verify(
        self,
        question: str,
        example_id: Optional[str] = None,
        retrieved_chunks: Optional[Sequence[dict]] = None,
        context_mode: Optional[str] = None,
        resolve_economic_sign: bool = False,
        enable_repair: Optional[bool] = None,
    ) -> ClaimGenerationResult:
        """Execute end-to-end generation and verification for a financial question."""
        start_total = time.perf_counter()
        error_diagnostics: list[str] = []

        active_context_mode = context_mode or self.context_mode
        should_repair = self.enable_repair if enable_repair is None else bool(enable_repair)

        # -------------------------------------------------------------
        # 1. Retrieval Stage
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if retrieved_chunks is not None:
            chunks = list(retrieved_chunks)
            active_retrieval_config = "direct"
        else:
            chunks = self.retrieve(question, top_k=5)
            active_retrieval_config = self.retrieval_config
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_chunk_ids = [c.get("chunk_id", "") for c in chunks]

        # Guard: Empty Retrieval Results
        if not chunks:
            total_lat = (time.perf_counter() - start_total) * 1000
            diag = "No evidence chunks retrieved for query."
            return ClaimGenerationResult(
                question=question,
                example_id=example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=[],
                context_mode=active_context_mode,
                context_token_count=0,
                context_is_truncated=False,
                generation_result=None,
                parsed_claim=None,
                verification_result=None,
                status="GENERATION_ERROR",
                detailed_status="EMPTY_RETRIEVAL_RESULTS",
                verified_answer=None,
                is_verified=False,
                error_diagnostics=[diag],
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=total_lat,
            )

        # Infer example_id from top chunk if not explicitly provided
        resolved_example_id = example_id
        if not resolved_example_id and chunks:
            resolved_example_id = chunks[0].get("example_id") or chunks[0].get("parent_id")

        # -------------------------------------------------------------
        # 2. Context Building Stage
        # -------------------------------------------------------------
        t1 = time.perf_counter()
        try:
            # If in parent expansion mode and corpus_index not preloaded, build on the fly
            active_corpus_index = self.corpus_index
            if active_context_mode in ("parent_expanded", "deduplicated_parent") and active_corpus_index is None:
                active_corpus_index = build_corpus_index(chunks)

            context_res = self.context_builder.build_context(
                chunks=chunks,
                corpus_index=active_corpus_index,
                mode=active_context_mode,
            )
        except Exception as exc:
            total_lat = (time.perf_counter() - start_total) * 1000
            diag = f"Context building failed: {exc}"
            return ClaimGenerationResult(
                question=question,
                example_id=resolved_example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_mode=active_context_mode,
                context_token_count=0,
                context_is_truncated=False,
                generation_result=None,
                parsed_claim=None,
                verification_result=None,
                status="GENERATION_ERROR",
                detailed_status="CONTEXT_BUILD_FAILURE",
                verified_answer=None,
                is_verified=False,
                error_diagnostics=[diag],
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=total_lat,
            )
        context_latency_ms = (time.perf_counter() - t1) * 1000

        # Build candidate verification pool representing all visible evidence
        visible_chunk_pool = self._build_verification_chunk_pool(chunks, active_corpus_index)
        permitted_chunk_ids = frozenset(
            c.get("chunk_id", "").strip()
            for c in visible_chunk_pool
            if c.get("chunk_id")
        )

        # -------------------------------------------------------------
        # 3. Prompt Construction & LLM Generation Stage
        # -------------------------------------------------------------
        prompt = build_generation_prompt(
            question=question,
            context_text=context_res.context_text,
            example_id=resolved_example_id,
            include_few_shot=True,
        )

        t2 = time.perf_counter()
        gen_result = self.llm_client.generate(prompt)
        generation_latency_ms = (time.perf_counter() - t2) * 1000

        # Guard: Generation Failures (Timeout, API error, OOM)
        if gen_result.status != "SUCCESS":
            total_lat = (time.perf_counter() - start_total) * 1000
            detailed_status = "GENERATION_TIMEOUT" if gen_result.status == "TIMEOUT" else "MODEL_INFERENCE_FAILURE"
            diag = gen_result.error_message or f"Generation failed with status {gen_result.status}"
            return ClaimGenerationResult(
                question=question,
                example_id=resolved_example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_mode=active_context_mode,
                context_token_count=context_res.token_count,
                context_is_truncated=context_res.is_truncated,
                generation_result=gen_result,
                parsed_claim=None,
                verification_result=None,
                status="GENERATION_ERROR",
                detailed_status=detailed_status,
                verified_answer=None,
                is_verified=False,
                error_diagnostics=[diag],
                retrieval_latency_ms=retrieval_latency_ms,
                context_latency_ms=context_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_lat,
            )

        # -------------------------------------------------------------
        # 4. Strict Output Parsing Stage
        # -------------------------------------------------------------
        t3 = time.perf_counter()
        parse_result: ParseResult = self.output_parser.parse(gen_result.text)

        # Case A: Model explicitly abstained (insufficient_evidence)
        if parse_result.status == "INSUFFICIENT_EVIDENCE":
            total_lat = (time.perf_counter() - start_total) * 1000
            verif_lat = (time.perf_counter() - t3) * 1000
            return ClaimGenerationResult(
                question=question,
                example_id=resolved_example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_mode=active_context_mode,
                context_token_count=context_res.token_count,
                context_is_truncated=context_res.is_truncated,
                generation_result=gen_result,
                parsed_claim=parse_result.claim,
                verification_result=None,
                status="INSUFFICIENT_EVIDENCE",
                detailed_status="MODEL_DECLARED_ABSTENTION",
                verified_answer=None,
                is_verified=False,
                error_diagnostics=[],
                retrieval_latency_ms=retrieval_latency_ms,
                context_latency_ms=context_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verif_lat,
                total_latency_ms=total_lat,
                initial_status="INSUFFICIENT_EVIDENCE",
                initial_detailed_status="MODEL_DECLARED_ABSTENTION",
                repair_attempted=False,
                repair_eligible=False,
                repair_attempts=0,
            )

        # Case B: Parsing or Schema / Referential Failure
        if not parse_result.is_valid:
            total_lat = (time.perf_counter() - start_total) * 1000
            verif_lat = (time.perf_counter() - t3) * 1000
            init_status = "PARSING_ERROR"
            init_detailed = parse_result.status
            init_diags = list(parse_result.errors)

            if should_repair and self.max_repair_budget > 0:
                is_verif, outcome, audit = self.repair_engine.attempt_repair(
                    question=question,
                    context_text=context_res.context_text,
                    example_id=resolved_example_id,
                    previous_raw_output=gen_result.text,
                    previous_claim=None,
                    initial_status=init_status,
                    initial_detailed_status=init_detailed,
                    initial_diagnostics=init_diags,
                    visible_chunk_pool=visible_chunk_pool,
                    permitted_chunk_ids=permitted_chunk_ids,
                    resolve_economic_sign=resolve_economic_sign,
                )
                if audit.is_repairable:
                    return ClaimGenerationResult(
                        question=question,
                        example_id=resolved_example_id,
                        retrieval_config=active_retrieval_config,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        context_mode=active_context_mode,
                        context_token_count=context_res.token_count,
                        context_is_truncated=context_res.is_truncated,
                        generation_result=outcome.get("generation_result") or gen_result,
                        parsed_claim=outcome.get("parsed_claim"),
                        verification_result=outcome.get("verification_result"),
                        status=outcome["status"],
                        detailed_status=outcome["detailed_status"],
                        verified_answer=outcome.get("verified_answer"),
                        is_verified=is_verif,
                        error_diagnostics=outcome.get("diagnostics", []),
                        retrieval_latency_ms=retrieval_latency_ms,
                        context_latency_ms=context_latency_ms,
                        generation_latency_ms=generation_latency_ms + outcome.get("generation_latency_ms", 0.0),
                        verification_latency_ms=verif_lat + outcome.get("verification_latency_ms", 0.0),
                        total_latency_ms=(time.perf_counter() - start_total) * 1000,
                        initial_status=init_status,
                        initial_detailed_status=init_detailed,
                        initial_diagnostics=init_diags,
                        repair_attempted=True,
                        repair_eligible=True,
                        repair_attempts=1,
                        repair_audit=audit.to_dict(),
                    )
                else:
                    return ClaimGenerationResult(
                        question=question,
                        example_id=resolved_example_id,
                        retrieval_config=active_retrieval_config,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        context_mode=active_context_mode,
                        context_token_count=context_res.token_count,
                        context_is_truncated=context_res.is_truncated,
                        generation_result=gen_result,
                        parsed_claim=None,
                        verification_result=None,
                        status="PARSING_ERROR",
                        detailed_status=parse_result.status,
                        verified_answer=None,
                        is_verified=False,
                        error_diagnostics=parse_result.errors,
                        retrieval_latency_ms=retrieval_latency_ms,
                        context_latency_ms=context_latency_ms,
                        generation_latency_ms=generation_latency_ms,
                        verification_latency_ms=verif_lat,
                        total_latency_ms=total_lat,
                        initial_status=init_status,
                        initial_detailed_status=init_detailed,
                        initial_diagnostics=init_diags,
                        repair_attempted=False,
                        repair_eligible=False,
                        repair_attempts=0,
                        repair_audit=audit.to_dict(),
                    )

            return ClaimGenerationResult(
                question=question,
                example_id=resolved_example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_mode=active_context_mode,
                context_token_count=context_res.token_count,
                context_is_truncated=context_res.is_truncated,
                generation_result=gen_result,
                parsed_claim=None,
                verification_result=None,
                status="PARSING_ERROR",
                detailed_status=parse_result.status,
                verified_answer=None,
                is_verified=False,
                error_diagnostics=parse_result.errors,
                retrieval_latency_ms=retrieval_latency_ms,
                context_latency_ms=context_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verif_lat,
                total_latency_ms=total_lat,
                initial_status=init_status,
                initial_detailed_status=init_detailed,
                initial_diagnostics=init_diags,
                repair_attempted=False,
                repair_eligible=False,
                repair_attempts=0,
                repair_audit=None,
            )

        # -------------------------------------------------------------
        # 5. Symbolic Verification Stage (Authoritative Verifier)
        # -------------------------------------------------------------
        context_passages = [c.get("text", "") for c in visible_chunk_pool if c.get("text")]
        if context_res.context_text:
            context_passages.append(context_res.context_text)

        try:
            verif_result = run_verification_pipeline(
                claim=parse_result.claim,
                example_id=resolved_example_id,
                chunks=visible_chunk_pool,
                context_passages=context_passages,
                resolve_economic_sign=resolve_economic_sign,
            )
        except Exception as exc:
            total_lat = (time.perf_counter() - start_total) * 1000
            verif_lat = (time.perf_counter() - t3) * 1000
            diag = f"Verification engine raised unexpected exception: {exc}"
            return ClaimGenerationResult(
                question=question,
                example_id=resolved_example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_mode=active_context_mode,
                context_token_count=context_res.token_count,
                context_is_truncated=context_res.is_truncated,
                generation_result=gen_result,
                parsed_claim=parse_result.claim,
                verification_result=None,
                status="VERIFICATION_FAILED",
                detailed_status="VERIFICATION_EXCEPTION",
                verified_answer=None,
                is_verified=False,
                error_diagnostics=[diag],
                retrieval_latency_ms=retrieval_latency_ms,
                context_latency_ms=context_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verif_lat,
                total_latency_ms=total_lat,
            )

        verification_latency_ms = (time.perf_counter() - t3) * 1000
        total_latency_ms = (time.perf_counter() - start_total) * 1000

        # -------------------------------------------------------------
        # 6. Map Verification Outcome
        # -------------------------------------------------------------
        if verif_result.get("valid", False):
            calc_val = verif_result.get("verified_value")
            if calc_val is None and verif_result.get("calculation"):
                calc_val = verif_result["calculation"].get("result")

            return ClaimGenerationResult(
                question=question,
                example_id=resolved_example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_mode=active_context_mode,
                context_token_count=context_res.token_count,
                context_is_truncated=context_res.is_truncated,
                generation_result=gen_result,
                parsed_claim=parse_result.claim,
                verification_result=verif_result,
                status="VERIFIED",
                detailed_status="VERIFIED",
                verified_answer=float(calc_val) if calc_val is not None else None,
                is_verified=True,
                error_diagnostics=[],
                retrieval_latency_ms=retrieval_latency_ms,
                context_latency_ms=context_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verification_latency_ms,
                total_latency_ms=total_latency_ms,
                initial_status="VERIFIED",
                initial_detailed_status="VERIFIED",
                repair_attempted=False,
                repair_eligible=False,
                repair_attempts=0,
            )
        else:
            reason = verif_result.get("reason", "Verification rejected claim.")
            raw_status = verif_result.get("status", "VERIFICATION_FAILED")
            detailed_status = raw_status
            diagnostics = [reason]

            # When INVALID_EVIDENCE is returned, inspect individual operands to identify
            # the exact grounding failure (chunk, span, value, unit, period)
            if raw_status == "INVALID_EVIDENCE":
                lookup = build_chunk_lookup(visible_chunk_pool)
                target_claim = None
                for c in parse_result.claim.get("claims", []):
                    if c.get("id") == parse_result.claim.get("final_claim_id"):
                        target_claim = c
                        break
                if target_claim:
                    referent = target_claim.get("referent")
                    for op in target_claim.get("operands", []):
                        op_res = validate_operand(op, lookup, resolved_example_id, referent)
                        if not op_res["valid"]:
                            op_reason = op_res.get("reason", "")
                            diagnostics.append(f"Operand '{op.get('id')}': {op_reason}")
                            op_r_lower = op_reason.lower()
                            if "chunk" in op_r_lower:
                                detailed_status = "EVIDENCE_GROUNDING_ERROR"
                            elif "span" in op_r_lower:
                                detailed_status = "EVIDENCE_GROUNDING_ERROR"
                            elif "period" in op_r_lower:
                                detailed_status = "PERIOD_VALIDATION_ERROR"
                            elif "unit" in op_r_lower:
                                detailed_status = "UNIT_VALIDATION_ERROR"
                            elif "value" in op_r_lower or "was not found in evidence" in op_r_lower:
                                detailed_status = "EVIDENCE_GROUNDING_ERROR"
                            else:
                                detailed_status = "EVIDENCE_GROUNDING_ERROR"
                            break

            elif raw_status == "DIRECTIONAL_CONTRADICTION" or "directional contradiction" in reason.lower():
                detailed_status = "DIRECTIONAL_CONTRADICTION"
            elif raw_status in ("CALCULATION_ERROR", "INVALID_PROGRAM", "NO_PROGRAM"):
                detailed_status = "ARITHMETIC_ERROR"
            elif "not found" in reason.lower() and ("chunk" in reason.lower() or "span" in reason.lower() or "value" in reason.lower()):
                detailed_status = "EVIDENCE_GROUNDING_ERROR"
            elif "unit" in reason.lower() and "not supported" in reason.lower():
                detailed_status = "UNIT_VALIDATION_ERROR"
            elif "period" in reason.lower() and "not found" in reason.lower():
                detailed_status = "PERIOD_VALIDATION_ERROR"

            init_status = "VERIFICATION_FAILED"
            init_detailed = detailed_status
            init_diags = list(diagnostics)

            if should_repair and self.max_repair_budget > 0:
                is_verif, outcome, audit = self.repair_engine.attempt_repair(
                    question=question,
                    context_text=context_res.context_text,
                    example_id=resolved_example_id,
                    previous_raw_output=gen_result.text,
                    previous_claim=parse_result.claim,
                    initial_status=init_status,
                    initial_detailed_status=init_detailed,
                    initial_diagnostics=init_diags,
                    visible_chunk_pool=visible_chunk_pool,
                    permitted_chunk_ids=permitted_chunk_ids,
                    resolve_economic_sign=resolve_economic_sign,
                )
                if audit.is_repairable:
                    return ClaimGenerationResult(
                        question=question,
                        example_id=resolved_example_id,
                        retrieval_config=active_retrieval_config,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        context_mode=active_context_mode,
                        context_token_count=context_res.token_count,
                        context_is_truncated=context_res.is_truncated,
                        generation_result=outcome.get("generation_result") or gen_result,
                        parsed_claim=outcome.get("parsed_claim"),
                        verification_result=outcome.get("verification_result"),
                        status=outcome["status"],
                        detailed_status=outcome["detailed_status"],
                        verified_answer=outcome.get("verified_answer"),
                        is_verified=is_verif,
                        error_diagnostics=outcome.get("diagnostics", []),
                        retrieval_latency_ms=retrieval_latency_ms,
                        context_latency_ms=context_latency_ms,
                        generation_latency_ms=generation_latency_ms + outcome.get("generation_latency_ms", 0.0),
                        verification_latency_ms=verification_latency_ms + outcome.get("verification_latency_ms", 0.0),
                        total_latency_ms=(time.perf_counter() - start_total) * 1000,
                        initial_status=init_status,
                        initial_detailed_status=init_detailed,
                        initial_diagnostics=init_diags,
                        repair_attempted=True,
                        repair_eligible=True,
                        repair_attempts=1,
                        repair_audit=audit.to_dict(),
                    )
                else:
                    return ClaimGenerationResult(
                        question=question,
                        example_id=resolved_example_id,
                        retrieval_config=active_retrieval_config,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        context_mode=active_context_mode,
                        context_token_count=context_res.token_count,
                        context_is_truncated=context_res.is_truncated,
                        generation_result=gen_result,
                        parsed_claim=parse_result.claim,
                        verification_result=verif_result,
                        status="VERIFICATION_FAILED",
                        detailed_status=detailed_status,
                        verified_answer=None,
                        is_verified=False,
                        error_diagnostics=diagnostics,
                        retrieval_latency_ms=retrieval_latency_ms,
                        context_latency_ms=context_latency_ms,
                        generation_latency_ms=generation_latency_ms,
                        verification_latency_ms=verification_latency_ms,
                        total_latency_ms=total_latency_ms,
                        initial_status=init_status,
                        initial_detailed_status=init_detailed,
                        initial_diagnostics=init_diags,
                        repair_attempted=False,
                        repair_eligible=False,
                        repair_attempts=0,
                        repair_audit=audit.to_dict(),
                    )

            return ClaimGenerationResult(
                question=question,
                example_id=resolved_example_id,
                retrieval_config=active_retrieval_config,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_mode=active_context_mode,
                context_token_count=context_res.token_count,
                context_is_truncated=context_res.is_truncated,
                generation_result=gen_result,
                parsed_claim=parse_result.claim,
                verification_result=verif_result,
                status="VERIFICATION_FAILED",
                detailed_status=detailed_status,
                verified_answer=None,
                is_verified=False,
                error_diagnostics=diagnostics,
                retrieval_latency_ms=retrieval_latency_ms,
                context_latency_ms=context_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verification_latency_ms,
                total_latency_ms=total_latency_ms,
                initial_status=init_status,
                initial_detailed_status=init_detailed,
                initial_diagnostics=init_diags,
                repair_attempted=False,
                repair_eligible=False,
                repair_attempts=0,
                repair_audit=None,
            )
