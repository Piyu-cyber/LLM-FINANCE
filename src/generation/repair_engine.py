"""repair_engine.py — Deterministic financial claim repair and abstention engine.

Coordinates single-attempt diagnostic repair for LLM-generated claims:
1. Deterministic repairability classification (syntax, schema, referential, arithmetic,
   and narrative direction vs non-repairable grounding, abstention, and infrastructure errors).
2. Immutable evidence registry enforcement (preventing hallucinated chunk IDs or external evidence).
3. Diagnostic feedback repair prompt construction.
4. Single-attempt repair generation with strict budget enforcement (max_budget = 1).
5. Complete, non-shortened re-verification through the authoritative verification pipeline.
6. Explicit terminal statuses and detailed audit logging.
7. Mathematical metrics calculation with transparent denominators.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Local generation imports
try:
    from .generation_config import GenerationConfig
    from .llm_client import GenerationResult, LLMClient
    from .output_parser import OutputParser, ParseResult
    from .prompts import build_repair_prompt
except (ImportError, ValueError):
    from generation_config import GenerationConfig
    from llm_client import GenerationResult, LLMClient
    from output_parser import OutputParser, ParseResult
    from prompts import build_repair_prompt

# Verification imports
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


# ---------------------------------------------------------------------------
# Classification Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairabilityClassification:
    """Deterministic classification of claim failure repairability."""

    is_repairable: bool
    category: str
    reason: str


# ---------------------------------------------------------------------------
# Audit Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepairAttemptAudit:
    """Detailed audit record of a single repair iteration."""

    attempt_number: int = 1
    is_repairable: bool = False
    repairability_category: str = ""
    repairability_reason: str = ""
    diagnostic_feedback: list[str] = field(default_factory=list)
    repair_prompt_preview: str = ""
    repair_raw_completion: str = ""
    repaired_claim: Optional[dict[str, Any]] = None
    reverification_status: str = ""
    reverification_detailed_status: str = ""
    reverification_passed: bool = False
    generation_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    total_repair_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert audit log to dictionary format."""
        return {
            "attempt_number": self.attempt_number,
            "is_repairable": self.is_repairable,
            "repairability_category": self.repairability_category,
            "repairability_reason": self.repairability_reason,
            "diagnostic_feedback": self.diagnostic_feedback,
            "repair_prompt_preview": self.repair_prompt_preview[:256],
            "has_repaired_claim": self.repaired_claim is not None,
            "reverification_status": self.reverification_status,
            "reverification_detailed_status": self.reverification_detailed_status,
            "reverification_passed": self.reverification_passed,
            "generation_latency_ms": round(self.generation_latency_ms, 3),
            "verification_latency_ms": round(self.verification_latency_ms, 3),
            "total_repair_latency_ms": round(self.total_repair_latency_ms, 3),
        }


# ---------------------------------------------------------------------------
# Classification Function
# ---------------------------------------------------------------------------

def classify_repairability(
    status: str,
    detailed_status: str,
    diagnostics: Sequence[str],
    claim: Optional[dict[str, Any]] = None,
) -> RepairabilityClassification:
    """Deterministically classify whether a failed claim is eligible for repair.

    Parameters
    ----------
    status : str
        High-level status (e.g. 'VERIFICATION_FAILED', 'PARSING_ERROR', 'INSUFFICIENT_EVIDENCE').
    detailed_status : str
        Detailed status code (e.g. 'ARITHMETIC_ERROR', 'EVIDENCE_GROUNDING_ERROR', etc.).
    diagnostics : Sequence[str]
        List of diagnostic error strings from the parser or verifier.
    claim : Optional[dict]
        Parsed claim structure if available.

    Returns
    -------
    RepairabilityClassification
        Frozen dataclass indicating is_repairable, category, and explanatory reason.
    """
    diag_str = " ".join(diagnostics).lower()

    # 1. Non-repairable: Model Abstention
    if status == "INSUFFICIENT_EVIDENCE" or detailed_status == "MODEL_DECLARED_ABSTENTION":
        return RepairabilityClassification(
            is_repairable=False,
            category="NON_REPAIRABLE_ABSTENTION",
            reason="Model explicitly declared insufficient evidence; repair is not permitted.",
        )

    # 2. Non-repairable: Retrieval or Context Failures
    if detailed_status in ("EMPTY_RETRIEVAL_RESULTS", "CONTEXT_BUILD_FAILURE"):
        return RepairabilityClassification(
            is_repairable=False,
            category="NON_REPAIRABLE_RETRIEVAL",
            reason="No evidence retrieved or context assembly failed; no grounding evidence exists for repair.",
        )

    # 3. Non-repairable: Infrastructure Failures (Timeout, Client Exception)
    if detailed_status in ("GENERATION_TIMEOUT", "MODEL_INFERENCE_FAILURE"):
        return RepairabilityClassification(
            is_repairable=False,
            category="NON_REPAIRABLE_INFRASTRUCTURE",
            reason="Model generation timed out or encountered client exception; repair cannot proceed.",
        )

    # 4. Non-repairable: Verification Pipeline Internal Exception
    if detailed_status == "VERIFICATION_EXCEPTION":
        return RepairabilityClassification(
            is_repairable=False,
            category="NON_REPAIRABLE_EXCEPTION",
            reason="Deterministic verifier raised an unexpected exception; unsafe to attempt repair.",
        )

    # 5. Non-repairable: Evidence Grounding Errors
    # (Hallucinated numerical values, non-existent chunk IDs, missing/fabricated evidence spans)
    if detailed_status == "EVIDENCE_GROUNDING_ERROR" or (
        "not found in evidence" in diag_str
        or "chunk not found" in diag_str
        or "span not found" in diag_str
        or "empty evidence span" in diag_str
    ):
        return RepairabilityClassification(
            is_repairable=False,
            category="NON_REPAIRABLE_GROUNDING",
            reason="Evidence grounding failure (unsupported operand value, missing span, or fabricated chunk ID). "
            "Hallucinated evidence cannot be repaired without external sources.",
        )

    # 6. Repairable: Syntax & JSON Decode Errors
    if detailed_status == "JSON_DECODE_ERROR":
        return RepairabilityClassification(
            is_repairable=True,
            category="SYNTAX",
            reason="JSON formatting or decoding error is repairable via constrained re-prompting.",
        )

    # 7. Repairable: Schema Validation Errors (Missing required fields)
    if detailed_status == "SCHEMA_VALIDATION_ERROR":
        return RepairabilityClassification(
            is_repairable=True,
            category="SCHEMA",
            reason="Schema validation error (missing fields or type mismatch) is repairable without new evidence.",
        )

    # 8. Repairable: Referential Integrity Errors
    if detailed_status == "REFERENTIAL_INTEGRITY_ERROR":
        return RepairabilityClassification(
            is_repairable=True,
            category="REFERENTIAL",
            reason="Program referential integrity error (undefined operand reference or step order) is repairable.",
        )

    # 9. Repairable: Arithmetic Errors (When operands are already grounded)
    if detailed_status == "ARITHMETIC_ERROR":
        return RepairabilityClassification(
            is_repairable=True,
            category="ARITHMETIC",
            reason="Calculation program error over grounded operands is repairable via operation correction.",
        )

    # 10. Repairable: Directional Contradiction
    if detailed_status == "DIRECTIONAL_CONTRADICTION":
        return RepairabilityClassification(
            is_repairable=True,
            category="DIRECTION",
            reason="Narrative direction contradiction is repairable by selecting an accounting polarity operation (e.g. economic_change).",
        )

    # 11. Repairable: Unit Validation Error
    if detailed_status == "UNIT_VALIDATION_ERROR":
        return RepairabilityClassification(
            is_repairable=True,
            category="METADATA",
            reason="Unit mismatch is repairable if the evidence unambiguously contains the true unit.",
        )

    # 12. Repairable: Period Validation Error
    if detailed_status == "PERIOD_VALIDATION_ERROR":
        return RepairabilityClassification(
            is_repairable=True,
            category="METADATA",
            reason="Period mismatch is repairable if the evidence unambiguously contains the true fiscal period.",
        )

    # Default fallback: Non-repairable
    return RepairabilityClassification(
        is_repairable=False,
        category="NON_REPAIRABLE_DEFAULT",
        reason=f"Error status '{detailed_status}' is non-repairable by default.",
    )


# ---------------------------------------------------------------------------
# Repair Engine Class
# ---------------------------------------------------------------------------

class RepairEngine:
    """Safe, single-attempt claim repair engine enforcing deterministic re-verification."""

    def __init__(
        self,
        llm_client: LLMClient,
        output_parser: Optional[OutputParser] = None,
        max_repair_budget: int = 1,
    ) -> None:
        """Initialize RepairEngine.

        Parameters
        ----------
        llm_client : LLMClient
            Configured inference client for repair generation.
        output_parser : Optional[OutputParser]
            Strict CLAIM_SCHEMA parser.
        max_repair_budget : int
            Maximum number of repair attempts permitted (default = 1).
        """
        self.llm_client = llm_client
        self.output_parser = output_parser or OutputParser()
        self.max_repair_budget = max(0, int(max_repair_budget))

    def verify_permitted_evidence(
        self,
        claim: dict[str, Any],
        permitted_chunk_ids: frozenset[str],
    ) -> Tuple[bool, Optional[str]]:
        """Verify that every cited operand in the claim belongs to the immutable evidence registry."""
        for c in claim.get("claims", []):
            for op in c.get("operands", []):
                cid = op.get("chunk_id", "").strip()
                if cid not in permitted_chunk_ids:
                    return False, (
                        f"Repaired operand '{op.get('id')}' references chunk_id '{cid}' "
                        f"which is not in the permitted evidence registry {sorted(list(permitted_chunk_ids))}."
                    )
        return True, None

    def attempt_repair(
        self,
        question: str,
        context_text: str,
        example_id: Optional[str],
        previous_raw_output: str,
        previous_claim: Optional[dict[str, Any]],
        initial_status: str,
        initial_detailed_status: str,
        initial_diagnostics: Sequence[str],
        visible_chunk_pool: Sequence[dict[str, Any]],
        permitted_chunk_ids: Optional[frozenset[str]] = None,
        resolve_economic_sign: bool = False,
    ) -> Tuple[bool, dict[str, Any], RepairAttemptAudit]:
        """Execute a single repair attempt and perform complete re-verification.

        Returns
        -------
        Tuple[bool, dict, RepairAttemptAudit]
            (is_verified, repair_outcome_dict, audit_log)
        """
        start_repair_total = time.perf_counter()

        # Step 1: Immutable evidence registry resolution
        if permitted_chunk_ids is None:
            permitted_chunk_ids = frozenset(
                c.get("chunk_id", "").strip()
                for c in visible_chunk_pool
                if c.get("chunk_id")
            )

        # Step 2: Classify repairability
        classification = classify_repairability(
            status=initial_status,
            detailed_status=initial_detailed_status,
            diagnostics=initial_diagnostics,
            claim=previous_claim,
        )

        audit = RepairAttemptAudit(
            attempt_number=1,
            is_repairable=classification.is_repairable,
            repairability_category=classification.category,
            repairability_reason=classification.reason,
            diagnostic_feedback=list(initial_diagnostics),
        )

        if not classification.is_repairable:
            total_lat = (time.perf_counter() - start_repair_total) * 1000
            audit.total_repair_latency_ms = total_lat
            audit.reverification_status = initial_status
            audit.reverification_detailed_status = "REPAIR_NOT_ATTEMPTED"
            return False, {
                "status": initial_status,
                "detailed_status": initial_detailed_status,
                "diagnostics": list(initial_diagnostics),
                "parsed_claim": previous_claim,
                "verification_result": None,
                "verified_answer": None,
                "generation_result": None,
                "generation_latency_ms": 0.0,
                "verification_latency_ms": 0.0,
                "total_repair_latency_ms": total_lat,
            }, audit

        # Step 3: Construct targeted single-turn repair prompt
        repair_prompt = build_repair_prompt(
            question=question,
            context_text=context_text,
            previous_output=previous_raw_output,
            errors=initial_diagnostics,
            example_id=example_id,
        )
        audit.repair_prompt_preview = repair_prompt[:256]

        # Step 4: Execute repair generation
        t_gen_start = time.perf_counter()
        repair_gen = self.llm_client.generate(repair_prompt)
        gen_lat_ms = (time.perf_counter() - t_gen_start) * 1000
        audit.generation_latency_ms = gen_lat_ms
        audit.repair_raw_completion = repair_gen.text

        # Guard: Repair generation timeout or inference failure
        if repair_gen.status != "SUCCESS":
            total_lat = (time.perf_counter() - start_repair_total) * 1000
            detailed_status = (
                "GENERATION_TIMEOUT"
                if repair_gen.status == "TIMEOUT"
                else "MODEL_INFERENCE_FAILURE"
            )
            audit.reverification_status = "REPAIR_FAILED"
            audit.reverification_detailed_status = detailed_status
            audit.total_repair_latency_ms = total_lat
            return False, {
                "status": "REPAIR_FAILED",
                "detailed_status": detailed_status,
                "diagnostics": [repair_gen.error_message or f"Repair generation failed with status {repair_gen.status}"],
                "parsed_claim": None,
                "verification_result": None,
                "verified_answer": None,
                "generation_result": repair_gen,
                "generation_latency_ms": gen_lat_ms,
                "verification_latency_ms": 0.0,
                "total_repair_latency_ms": total_lat,
            }, audit

        # Step 5: Strict re-parsing and schema validation
        t_verif_start = time.perf_counter()
        parse_res = self.output_parser.parse(repair_gen.text)
        audit.repaired_claim = parse_res.claim

        # Case 5A: Model abstained during repair
        if parse_res.status == "INSUFFICIENT_EVIDENCE":
            verif_lat_ms = (time.perf_counter() - t_verif_start) * 1000
            total_lat = (time.perf_counter() - start_repair_total) * 1000
            audit.verification_latency_ms = verif_lat_ms
            audit.total_repair_latency_ms = total_lat
            audit.reverification_status = "INSUFFICIENT_EVIDENCE"
            audit.reverification_detailed_status = "MODEL_DECLARED_ABSTENTION"
            return False, {
                "status": "INSUFFICIENT_EVIDENCE",
                "detailed_status": "MODEL_DECLARED_ABSTENTION",
                "diagnostics": ["Model explicitly abstained with insufficient_evidence during repair."],
                "parsed_claim": parse_res.claim,
                "verification_result": None,
                "verified_answer": None,
                "generation_result": repair_gen,
                "generation_latency_ms": gen_lat_ms,
                "verification_latency_ms": verif_lat_ms,
                "total_repair_latency_ms": total_lat,
            }, audit

        # Case 5B: Parsing, syntax, or schema failure on repaired output
        if not parse_res.is_valid:
            verif_lat_ms = (time.perf_counter() - t_verif_start) * 1000
            total_lat = (time.perf_counter() - start_repair_total) * 1000
            audit.verification_latency_ms = verif_lat_ms
            audit.total_repair_latency_ms = total_lat
            audit.reverification_status = "REPAIR_FAILED"
            audit.reverification_detailed_status = parse_res.status
            return False, {
                "status": "REPAIR_FAILED",
                "detailed_status": parse_res.status,
                "diagnostics": parse_res.errors,
                "parsed_claim": None,
                "verification_result": None,
                "verified_answer": None,
                "generation_result": repair_gen,
                "generation_latency_ms": gen_lat_ms,
                "verification_latency_ms": verif_lat_ms,
                "total_repair_latency_ms": total_lat,
            }, audit

        # Step 6: Evidence registry check (no fabricated chunk IDs allowed)
        is_evidence_valid, evidence_err = self.verify_permitted_evidence(
            claim=parse_res.claim,
            permitted_chunk_ids=permitted_chunk_ids,
        )
        if not is_evidence_valid:
            verif_lat_ms = (time.perf_counter() - t_verif_start) * 1000
            total_lat = (time.perf_counter() - start_repair_total) * 1000
            audit.verification_latency_ms = verif_lat_ms
            audit.total_repair_latency_ms = total_lat
            audit.reverification_status = "REPAIR_FAILED"
            audit.reverification_detailed_status = "EVIDENCE_GROUNDING_ERROR"
            return False, {
                "status": "REPAIR_FAILED",
                "detailed_status": "EVIDENCE_GROUNDING_ERROR",
                "diagnostics": [evidence_err],
                "parsed_claim": parse_res.claim,
                "verification_result": None,
                "verified_answer": None,
                "generation_result": repair_gen,
                "generation_latency_ms": gen_lat_ms,
                "verification_latency_ms": verif_lat_ms,
                "total_repair_latency_ms": total_lat,
            }, audit

        # Step 7: Complete deterministic re-verification (Stages 1 through 5)
        context_passages = [c.get("text", "") for c in visible_chunk_pool if c.get("text")]
        if context_text:
            context_passages.append(context_text)

        try:
            verif_res = run_verification_pipeline(
                claim=parse_res.claim,
                example_id=example_id,
                chunks=visible_chunk_pool,
                context_passages=context_passages,
                resolve_economic_sign=resolve_economic_sign,
            )
        except Exception as exc:
            verif_lat_ms = (time.perf_counter() - t_verif_start) * 1000
            total_lat = (time.perf_counter() - start_repair_total) * 1000
            audit.verification_latency_ms = verif_lat_ms
            audit.total_repair_latency_ms = total_lat
            audit.reverification_status = "REPAIR_FAILED"
            audit.reverification_detailed_status = "VERIFICATION_EXCEPTION"
            return False, {
                "status": "REPAIR_FAILED",
                "detailed_status": "VERIFICATION_EXCEPTION",
                "diagnostics": [f"Re-verification raised unexpected exception: {exc}"],
                "parsed_claim": parse_res.claim,
                "verification_result": None,
                "verified_answer": None,
                "generation_result": repair_gen,
                "generation_latency_ms": gen_lat_ms,
                "verification_latency_ms": verif_lat_ms,
                "total_repair_latency_ms": total_lat,
            }, audit

        verif_lat_ms = (time.perf_counter() - t_verif_start) * 1000
        total_lat = (time.perf_counter() - start_repair_total) * 1000
        audit.verification_latency_ms = verif_lat_ms
        audit.total_repair_latency_ms = total_lat

        # Step 8: Final re-verification evaluation
        if verif_res.get("valid", False):
            calc_val = verif_res.get("verified_value")
            if calc_val is None and verif_res.get("calculation"):
                calc_val = verif_res["calculation"].get("result")

            audit.reverification_passed = True
            audit.reverification_status = "VERIFIED"
            audit.reverification_detailed_status = "VERIFIED_AFTER_REPAIR"

            return True, {
                "status": "VERIFIED",
                "detailed_status": "VERIFIED_AFTER_REPAIR",
                "diagnostics": [],
                "parsed_claim": parse_res.claim,
                "verification_result": verif_res,
                "verified_answer": float(calc_val) if calc_val is not None else None,
                "generation_result": repair_gen,
                "generation_latency_ms": gen_lat_ms,
                "verification_latency_ms": verif_lat_ms,
                "total_repair_latency_ms": total_lat,
            }, audit
        else:
            reason = verif_res.get("reason", "Re-verification rejected repaired claim.")
            raw_status = verif_res.get("status", "VERIFICATION_FAILED")
            detailed_status = raw_status
            diagnostics = [reason]

            # Diagnostic drill-down for operand grounding failures
            if raw_status == "INVALID_EVIDENCE":
                lookup = build_chunk_lookup(visible_chunk_pool)
                target_claim = None
                for c in parse_res.claim.get("claims", []):
                    if c.get("id") == parse_res.claim.get("final_claim_id"):
                        target_claim = c
                        break
                if target_claim:
                    referent = target_claim.get("referent")
                    for op in target_claim.get("operands", []):
                        op_res = validate_operand(op, lookup, example_id, referent)
                        if not op_res["valid"]:
                            op_reason = op_res.get("reason", "")
                            diagnostics.append(f"Operand '{op.get('id')}': {op_reason}")
                            op_r_lower = op_reason.lower()
                            if "period" in op_r_lower:
                                detailed_status = "PERIOD_VALIDATION_ERROR"
                            elif "unit" in op_r_lower:
                                detailed_status = "UNIT_VALIDATION_ERROR"
                            elif "chunk" in op_r_lower or "span" in op_r_lower or "value" in op_r_lower:
                                detailed_status = "EVIDENCE_GROUNDING_ERROR"
                            else:
                                detailed_status = "EVIDENCE_GROUNDING_ERROR"
                            break

            elif raw_status == "DIRECTIONAL_CONTRADICTION" or "directional contradiction" in reason.lower():
                detailed_status = "DIRECTIONAL_CONTRADICTION"
            elif raw_status in ("CALCULATION_ERROR", "INVALID_PROGRAM", "NO_PROGRAM"):
                detailed_status = "ARITHMETIC_ERROR"

            audit.reverification_passed = False
            audit.reverification_status = "REPAIR_FAILED"
            audit.reverification_detailed_status = detailed_status

            return False, {
                "status": "REPAIR_FAILED",
                "detailed_status": detailed_status,
                "diagnostics": diagnostics,
                "parsed_claim": parse_res.claim,
                "verification_result": verif_res,
                "verified_answer": None,
                "generation_result": repair_gen,
                "generation_latency_ms": gen_lat_ms,
                "verification_latency_ms": verif_lat_ms,
                "total_repair_latency_ms": total_lat,
            }, audit


# ---------------------------------------------------------------------------
# Metrics Computation Function
# ---------------------------------------------------------------------------

def compute_repair_metrics(results: Sequence[Any]) -> dict[str, Any]:
    """Compute comprehensive repair and verification metrics across query executions.

    Formulas and Denominators:
    -------------------------
    - N (total_queries): len(results)
    - initial_verification_success_rate: count(initial_status == 'VERIFIED') / N
    - repair_eligibility_rate: count(repair_eligible) / N
    - repair_eligibility_among_failures: count(repair_eligible) / count(initial_status != 'VERIFIED')
    - repair_attempt_rate: count(repair_attempted) / N
    - repair_success_rate: count(repair_attempted and status == 'VERIFIED') / count(repair_attempted)
    - final_verification_success_rate: count(status == 'VERIFIED') / N
    - abstention_rate: count(status == 'INSUFFICIENT_EVIDENCE') / N
    - repair_failure_rate: count(status == 'REPAIR_FAILED') / count(repair_attempted)
    - avg_repair_generation_latency_ms: mean(repair_gen_lat) across repair attempts
    - avg_repair_verification_latency_ms: mean(repair_verif_lat) across repair attempts
    - avg_total_repair_latency_ms: mean(total_repair_lat) across repair attempts
    """
    total = len(results)
    if total == 0:
        return {
            "total_queries": 0,
            "initial_verification_success_rate": 0.0,
            "repair_eligibility_rate": 0.0,
            "repair_attempt_rate": 0.0,
            "repair_success_rate": 0.0,
            "final_verification_success_rate": 0.0,
            "abstention_rate": 0.0,
        }

    initial_verified = sum(
        1 for r in results if getattr(r, "initial_status", r.status) == "VERIFIED"
    )
    initial_failed = total - initial_verified
    eligible = sum(1 for r in results if getattr(r, "repair_eligible", False))
    attempted = sum(1 for r in results if getattr(r, "repair_attempted", False))
    repaired_and_verified = sum(
        1 for r in results if getattr(r, "repair_attempted", False) and r.status == "VERIFIED"
    )
    final_verified = sum(1 for r in results if r.status == "VERIFIED")
    abstained = sum(1 for r in results if r.status == "INSUFFICIENT_EVIDENCE")
    repair_failed = sum(1 for r in results if r.status == "REPAIR_FAILED")

    repair_gen_lats: list[float] = []
    repair_verif_lats: list[float] = []
    repair_total_lats: list[float] = []

    for r in results:
        audit = getattr(r, "repair_audit", None)
        if audit and isinstance(audit, dict):
            if "generation_latency_ms" in audit:
                repair_gen_lats.append(audit["generation_latency_ms"])
            if "verification_latency_ms" in audit:
                repair_verif_lats.append(audit["verification_latency_ms"])
            if "total_repair_latency_ms" in audit:
                repair_total_lats.append(audit["total_repair_latency_ms"])

    return {
        "total_queries": total,
        "initial_verified_count": initial_verified,
        "initial_failed_count": initial_failed,
        "repair_eligible_count": eligible,
        "repair_attempted_count": attempted,
        "repaired_and_verified_count": repaired_and_verified,
        "final_verified_count": final_verified,
        "abstained_count": abstained,
        "repair_failed_count": repair_failed,
        "initial_verification_success_rate": round(initial_verified / total, 4),
        "repair_eligibility_rate": round(eligible / total, 4),
        "repair_eligibility_among_failures": round(eligible / initial_failed, 4) if initial_failed > 0 else 0.0,
        "repair_attempt_rate": round(attempted / total, 4),
        "repair_success_rate": round(repaired_and_verified / attempted, 4) if attempted > 0 else 0.0,
        "repair_failure_rate": round(repair_failed / attempted, 4) if attempted > 0 else 0.0,
        "final_verification_success_rate": round(final_verified / total, 4),
        "abstention_rate": round(abstained / total, 4),
        "avg_repair_generation_latency_ms": round(sum(repair_gen_lats) / len(repair_gen_lats), 3) if repair_gen_lats else 0.0,
        "avg_repair_verification_latency_ms": round(sum(repair_verif_lats) / len(repair_verif_lats), 3) if repair_verif_lats else 0.0,
        "avg_total_repair_latency_ms": round(sum(repair_total_lats) / len(repair_total_lats), 3) if repair_total_lats else 0.0,
    }
