# LLM-FINANCE — Phase 3.4: Pre-Implementation Audit and Repair Engine Design

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents using FinQA  
**Phase**: Phase 3.4 — Repair and Abstention Engine  
**Author**: Senior ML Systems Engineer & Financial NLP Research Engineer  
**Date**: November 4, 2026  
**Status**: Pre-Implementation Audit Complete  

---

## 1. Executive Summary

Phase 3.3 established the end-to-end execution path from query to deterministic verification. In Phase 3.3, any verification or parsing failure immediately terminated in rejection.

Phase 3.4 introduces a safe, deterministic, single-attempt claim repair and abstention mechanism. The overarching design principle is **strict verification non-negotiability**:
> An LLM-generated claim that fails verification may receive at most one repair attempt, but the repaired claim must undergo complete, unshortened re-verification from Stage 1 through Stage 5. The LLM can never self-certify answers, bypass checks, or expand the retrieved evidence pool.

This audit inspects existing modules (`claim_generator.py`, `output_parser.py`, `prompts.py`, `validator.py`, `pipeline.py`, `calculator.py`), specifies the diagnostic error taxonomy, classifies repairability, establishes evidence preservation rules, models all state transitions, and outlines the 20 required integration tests.

---

## 2. Audit of Existing Modules and Diagnostic Structures

### 2.1 Error Codes in `src/verification/pipeline.py` & `src/verification/validator.py`
The verification pipeline returns a structured dictionary:
```python
{
    "valid": bool,
    "status": str,
    "reason": str,
    "calculation": Optional[dict],
    "direction_check": Optional[dict],
}
```
Observed status codes and reasons:
1. `SCHEMA_INVALID`: Claim structure fails `CLAIM_SCHEMA` jsonschema validation.
2. `INVALID_FINAL_CLAIM`: `final_claim_id` does not match any claim object in `claims`.
3. `INVALID_EVIDENCE`: One or more operands failed `validate_operand`:
   - `Evidence chunk not found` (chunk ID absent from lookup table).
   - `Empty evidence span` (span_text is empty or whitespace).
   - `Evidence span not found inside cited chunk` (verbatim substring check failed).
   - `Claimed value <val> was not found in evidence` (numerical value absent from span text).
   - `Claimed period <period> was not found in evidence` (period not in span, context, or chunk).
   - `Claimed unit '<unit>' is not supported by evidence` (incompatible unit category).
4. `NO_PROGRAM`: Final claim does not contain an executable calculation program.
5. `INVALID_PROGRAM`: Step lacks operation, or unknown operand ID referenced in step.
6. `CALCULATION_ERROR`: Arithmetic failure in `calculate` (e.g. division by zero, invalid operation name, operands length mismatch).
7. `DIRECTIONAL_CONTRADICTION`: Mismatch between computed direction (`calc_dir`) and narrative evidence disclosure (`narr_dir`), such as naive negative subtraction on expenses when disclosures state growth.

### 2.2 Error Codes in `src/generation/output_parser.py`
The parser returns `ParseResult`:
```python
@dataclass
class ParseResult:
    is_valid: bool
    status: str
    claim: Optional[dict[str, Any]] = None
    errors: list[str] = field(default_factory=list)
    raw_text: str = ""
    extracted_json_text: str = ""
```
Observed status codes:
1. `SUCCESS`: Valid JSON matching `CLAIM_SCHEMA` and referential checks.
2. `INSUFFICIENT_EVIDENCE`: Explicit model abstention (`status: "insufficient_evidence"`).
3. `JSON_DECODE_ERROR`: Syntax error, unclosed braces, or non-JSON completion.
4. `SCHEMA_VALIDATION_ERROR`: Missing required top-level or child properties.
5. `REFERENTIAL_INTEGRITY_ERROR`: Undefined operand reference, forward reference in execution graph, invalid operation name, or duplicate operand IDs.

### 2.3 Status Codes in `src/generation/claim_generator.py`
`ClaimGenerationResult` categorizes outcomes into high-level and detailed statuses:
- High-level: `VERIFIED`, `INSUFFICIENT_EVIDENCE`, `GENERATION_ERROR`, `PARSING_ERROR`, `VERIFICATION_FAILED`.
- Detailed statuses: `VERIFIED`, `MODEL_DECLARED_ABSTENTION`, `EMPTY_RETRIEVAL_RESULTS`, `GENERATION_TIMEOUT`, `MODEL_INFERENCE_FAILURE`, `JSON_DECODE_ERROR`, `SCHEMA_VALIDATION_ERROR`, `REFERENTIAL_INTEGRITY_ERROR`, `EVIDENCE_GROUNDING_ERROR`, `PERIOD_VALIDATION_ERROR`, `UNIT_VALIDATION_ERROR`, `ARITHMETIC_ERROR`, `DIRECTIONAL_CONTRADICTION`, `VERIFICATION_EXCEPTION`.

---

## 3. Deterministic Repairability Classification

An error is repairable **only if** the problem stems from formatting, referential consistency, operation selection, or misattributed metadata that is unambiguously resolvable from the *already-retrieved* evidence. It is non-repairable if resolving it requires information not present in the original evidence context.

### 3.1 Taxonomy of Errors

| Diagnostic Status | Sub-Category / Condition | Repairable? | Default Action | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `JSON_DECODE_ERROR` | Syntax error, markdown fence glitch | **Yes** | Attempt Repair | LLM can re-emit valid JSON without modifying facts. |
| `SCHEMA_VALIDATION_ERROR` | Missing `final_claim_id`, missing `referent` | **Yes** | Attempt Repair | Omission of required structural wrapper is rectifiable. |
| `REFERENTIAL_INTEGRITY_ERROR` | Typo in operand ID reference, program ordering | **Yes** | Attempt Repair | Execution graph can be ordered topologically. |
| `ARITHMETIC_ERROR` | Unsupported operation name, inverted operand order | **Yes** | Attempt Repair | Correct operation can be selected over grounded operands. |
| `DIRECTIONAL_CONTRADICTION` | Naive subtraction on negative expense item | **Yes** | Attempt Repair | LLM can switch to `economic_change` / magnitude change. |
| `UNIT_VALIDATION_ERROR` | Unit declared as `USD` instead of `million USD` | **Conditional** | Attempt Repair if Evidence Unambiguous | Evidence chunk text clearly states the unit. |
| `PERIOD_VALIDATION_ERROR` | Period declared as `2018` instead of `2019` | **Conditional** | Attempt Repair if Evidence Unambiguous | Evidence chunk text clearly contains the true period. |
| `EVIDENCE_GROUNDING_ERROR` | Cited chunk ID does not exist in evidence | **No** | **Abstain / Reject** | Fabricated evidence identities cannot be repaired safely. |
| `EVIDENCE_GROUNDING_ERROR` | Span text not found in cited chunk | **No** | **Abstain / Reject** | Hallucinated text indicates ungrounded reasoning. |
| `EVIDENCE_GROUNDING_ERROR` | Numerical value not found in span text | **No** | **Abstain / Reject** | Hallucinated numerical value indicates lack of source support. |
| `INSUFFICIENT_EVIDENCE` | Model declared `status: "insufficient_evidence"` | **No** | **Abstain Immediately** | Model correctly identified evidence insufficiency; no repair. |
| `EMPTY_RETRIEVAL_RESULTS` | Retrieval found no documents | **No** | **Early Stop** | Zero context cannot answer question. |
| `GENERATION_TIMEOUT` | Client timed out | **No** | **Fail Generation** | Infrastructure / latency failure. |
| `MODEL_INFERENCE_FAILURE` | Client raised exception / API error | **No** | **Fail Generation** | Upstream failure. |
| `VERIFICATION_EXCEPTION` | Verifier code crashed | **No** | **Fail Verification** | Internal system defect. |

---

## 4. Evidence Integrity & Preservation Architecture

To prevent data drift, context leakage, or post-hoc hallucinations during repair:

1. **Immutable Evidence Registry**:
   When initial retrieval and parent expansion run, the set of allowed chunk identifiers is recorded in an immutable set:
   ```python
   permitted_chunk_ids = frozenset(c.get("chunk_id") for c in visible_chunk_pool if c.get("chunk_id"))
   ```
2. **Zero Additional Retrieval**:
   The repair engine is strictly prohibited from invoking `retrieve()`, query expansion, or fetching external documents.
3. **Identical Context Passages**:
   The repair prompt is provided the exact same `context_text` and `example_id` used in the initial generation call.
4. **Strict Operand Guarding in Re-Verification**:
   When the repaired claim is parsed, every operand is checked against `permitted_chunk_ids`. If the repaired claim introduces an operand citing an ID not in `permitted_chunk_ids`, it is immediately rejected as an ungrounded hallucination (`EVIDENCE_GROUNDING_ERROR`).

---

## 5. State Machine & Execution Flow

```mermaid
flowchart TD
    START["Input Query & Retrieved Evidence"] --> GEN1["Initial Generation (Attempt 0)"]
    GEN1 --> PARSE1["Parse JSON & Schema"]
    
    PARSE1 -->|insufficient_evidence| TERM_ABSTAIN["INSUFFICIENT_EVIDENCE<br>(MODEL_DECLARED_ABSTENTION)"]
    PARSE1 -->|Generation Timeout/Error| TERM_GEN_ERR["GENERATION_ERROR"]
    
    PARSE1 -->|Valid JSON Claim| VERIF1["Authoritative Verification Pipeline"]
    VERIF1 -->|All Stages Pass| TERM_VERIFIED["VERIFIED (Answer Certified)"]
    
    PARSE1 -->|Parse / Schema Error| CHECK_REP{"Classify Repairability"}
    VERIF1 -->|Verification Failed| CHECK_REP
    
    CHECK_REP -->|Non-Repairable<br>(e.g. Hallucinated Value, Fabricated Chunk ID, Abstention)| TERM_REPAIR_NOT_ATTEMPTED["VERIFICATION_FAILED / PARSING_ERROR<br>(REPAIR_NOT_ATTEMPTED)"]
    
    CHECK_REP -->|Repairable<br>(e.g. Arithmetic, Direction, Syntax, Schema)| DO_REP["Construct Repair Prompt<br>(Diagnostics + Evidence)"]
    
    DO_REP --> GEN2["Repair Generation (Attempt 1 / Budget=1)"]
    
    GEN2 -->|Timeout / Inference Error| TERM_REP_FAIL_GEN["REPAIR_FAILED<br>(GENERATION_TIMEOUT)"]
    
    GEN2 --> PARSE2["Strict Re-Parsing"]
    PARSE2 -->|Malformed JSON / Schema Error| TERM_REP_FAIL_PARSE["REPAIR_FAILED<br>(PARSING_ERROR)"]
    
    PARSE2 -->|Valid Claim| VERIF2["Complete Re-Verification Pipeline<br>(Stages 1 to 5)"]
    
    VERIF2 -->|All Stages Pass| TERM_VERIFIED_REP["VERIFIED<br>(VERIFIED_AFTER_REPAIR)"]
    VERIF2 -->|Any Stage Fails| TERM_REP_FAIL_VERIF["REPAIR_FAILED<br>(VERIFICATION_FAILED_AFTER_REPAIR)"]
```

---

## 6. Audit Logging & Telemetry Specifications

The repair engine must capture complete diagnostic traces without storing API keys or credentials:

```python
@dataclass
class RepairAttemptAudit:
    attempt_number: int  # 1
    prompt_tokens: int
    completion_tokens: int
    is_repairable: bool
    repairability_reason: str
    repair_prompt_preview: str  # First 256 chars
    diagnostic_feedback: list[str]
    raw_completion: str
    repaired_claim: Optional[dict[str, Any]]
    reverification_status: str
    reverification_detailed_status: str
    reverification_passed: bool
    generation_latency_ms: float
    verification_latency_ms: float
    total_repair_latency_ms: float
```

The resulting audit record is attached to `ClaimGenerationResult.repair_audit` for complete scientific traceability.

---

## 7. Integration Test Strategy (`tests/test_repair_engine.py`)

To ensure research-grade stability and 100% determinism, all 20 tests will use `MockLLMClient` with targeted response maps or queues:

1. `test_valid_claim_no_repair_needed`: Valid claim verifies on initial pass; repair budget remains unused.
2. `test_repairable_arithmetic_error_succeeds`: Initial generation outputs `subtract` on positive numbers yielding negative; repair changes to valid subtraction; verifier certifies answer.
3. `test_repairable_narrative_contradiction_succeeds`: Initial generation uses naive subtraction on negative interest expense; verifier rejects with `DIRECTIONAL_CONTRADICTION`; repair switches to `operation: "economic_change"`; re-verification certifies.
4. `test_invalid_json_repair_succeeds`: Initial generation outputs malformed JSON with unclosed braces; repair outputs valid JSON; verifies.
5. `test_missing_required_field_repair_succeeds`: Initial generation omits `final_claim_id`; repair adds `final_claim_id`; verifies.
6. `test_unsupported_operand_repair_not_attempted`: Initial generation hallucinates `$999,999`; classified as non-repairable; no repair call made; status remains `VERIFICATION_FAILED` (`REPAIR_NOT_ATTEMPTED`).
7. `test_fabricated_chunk_id_repair_not_attempted`: Initial generation cites `chunk_999`; non-repairable; repair budget unused.
8. `test_insufficient_evidence_no_repair`: Initial generation outputs `status: "insufficient_evidence"`; immediately abstains without repair.
9. `test_repair_response_contains_another_unsupported_operand_rejected`: Repaired claim tries to fix arithmetic by inventing a new number; re-verification rejects with `REPAIR_FAILED`.
10. `test_repair_response_contains_fabricated_evidence_identity_rejected`: Repaired claim cites non-existent chunk; re-verification rejects with `REPAIR_FAILED`.
11. `test_repair_produces_incorrect_arithmetic_rejected`: Repaired claim still contains division by zero; re-verification rejects with `REPAIR_FAILED`.
12. `test_repair_produces_contradictory_narrative_rejected`: Repaired claim fails narrative direction check; rejects with `REPAIR_FAILED`.
13. `test_repair_returns_malformed_json_fails`: Repair output itself is malformed JSON; terminates with `REPAIR_FAILED` (`JSON_DECODE_ERROR`).
14. `test_repair_timeout_fails`: Repair generation times out; terminates with `REPAIR_FAILED` (`GENERATION_TIMEOUT`).
15. `test_repair_attempts_strictly_limited_to_one`: Even if repair fails, exactly 1 repair attempt was made; no subsequent call occurs.
16. `test_original_evidence_registry_remains_unchanged`: Permitted chunk IDs before and after repair are identical.
17. `test_original_verification_diagnostics_preserved`: Initial verification diagnostics remain accessible in result audit.
18. `test_repair_cannot_bypass_schema_validation`: Repaired JSON missing schema requirements is rejected.
19. `test_repair_cannot_bypass_unit_or_period_validation`: Repaired claim with invalid period or unit is rejected.
20. `test_explicit_model_abstention_triggers_no_repair`: Repaired claim outputting `insufficient_evidence` halts without further repair.

---

## 8. Conclusion

This audit establishes the explicit boundary conditions, error classification rules, and re-verification guarantees for Phase 3.4. We now proceed to implementing `src/generation/repair_engine.py` and updating `ClaimGenerator`.
