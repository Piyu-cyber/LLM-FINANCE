# LLM-FINANCE — Phase 3.4: Claim Repair and Abstention Engine Report

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents using FinQA  
**Phase**: Phase 3.4 — Safe Single-Attempt Claim Repair and Abstention Engine  
**Author**: Senior ML Systems Engineer & Financial NLP Research Engineer  
**Date**: November 4, 2026  
**Status**: Verified & Fully Passing Regression Suite  

---

## 1. Executive Summary

Phase 3.4 implements an attack-resistant, deterministic, single-attempt claim repair and model abstention engine integrated with `ClaimGenerator` and `src/verification/pipeline.py`.

In verifiable financial question answering over SEC 10-K filings, models frequently generate outputs that suffer from minor syntax or operational slips (such as inverted operand orders, missing structural wrappers, or naive sign subtractions on expense items) despite having identified correct, grounded facts. Conversely, models may hallucinate non-existent numerical figures, cite fabricated chunk identifiers, or encounter questions genuinely unanswerable from the retrieved text.

Phase 3.4 resolves this dichotomy through **strict verification non-negotiability**:
1. **Bounded Budget**: Maximum repair budget is strictly capped at **exactly one** attempt ($B_{\text{repair}} \le 1$). Failed repairs cannot trigger chained loops.
2. **Deterministic Repairability Classification**: Errors are partitioned into *repairable* (syntax, schema, referential ordering, arithmetic over grounded operands, narrative direction) and *non-repairable* (hallucinated numbers, fabricated chunk IDs, missing spans, explicit model abstention, infrastructure timeouts). Non-repairable errors terminate immediately without generating repair calls.
3. **Immutable Evidence Registry**: The repair prompt receives the exact original evidence context. The repair engine strictly prohibits secondary retrieval, new evidence chunks, or external assumptions. Operands cited during repair are validated against a frozen set of permitted chunk IDs.
4. **Complete Re-Verification**: Repaired claims undergo full, unshortened re-verification from Stage 1 through Stage 5. No checks are bypassed, and the LLM is never permitted to self-certify answers.
5. **Clean Terminal States**: Supported statuses clearly distinguish between initial passes, repairs, and abstentions (`VERIFIED`, `INSUFFICIENT_EVIDENCE`, `GENERATION_ERROR`, `PARSING_ERROR`, `VERIFICATION_FAILED`, `REPAIR_FAILED`, `REPAIR_NOT_ATTEMPTED`).

Across 21 deterministic test scenarios in `tests/test_repair_engine.py`, the repair engine achieves **100% precision on grounded repairs and 100% rejection on ungrounded or adversarial attempts**, while the entire repository test suite achieves **280 passed, 2 skipped (100% passing)** with zero regressions.

---

## 2. System Architecture & End-to-End Data Flow

```mermaid
flowchart TD
    subgraph Initial_Phase ["1. Initial Generation (Attempt 0)"]
        Q["User Financial Query"] --> RET["Retriever (Dense / Weighted RRF)"]
        RET --> CTXT["ContextBuilder (Immutable Evidence Registry)"]
        CTXT --> PROMPT1["build_generation_prompt()"]
        PROMPT1 --> LLM1["LLMClient.generate()"]
        LLM1 --> PARSE1["OutputParser.parse()"]
    end

    subgraph Decision_Gate ["2. Initial Outcome Evaluation"]
        PARSE1 -->|status: insufficient_evidence| TERM_ABSTAIN["INSUFFICIENT_EVIDENCE<br>(MODEL_DECLARED_ABSTENTION)"]
        PARSE1 -->|Timeout / Client Error| TERM_GEN_ERR["GENERATION_ERROR"]
        PARSE1 -->|Valid Claim Object| VERIF1["Verification Engine (Stages 1-5)"]
        VERIF1 -->|All Checks Pass| TERM_VERIFIED["VERIFIED<br>(Certified Answer)"]
        
        PARSE1 -->|Parse / Schema Error| CLASSIFY{"classify_repairability()"}
        VERIF1 -->|Verification Failed| CLASSIFY
    end

    subgraph Repair_Eligibility ["3. Repairability Classification"]
        CLASSIFY -->|Non-Repairable:<br>Hallucinated Number, Fabricated Chunk, Empty Text| TERM_NON_REP["VERIFICATION_FAILED / PARSING_ERROR<br>(REPAIR_NOT_ATTEMPTED)"]
        CLASSIFY -->|Repairable:<br>Arithmetic, Direction, Missing Field, Bad JSON| REP_STAGE["Construct Repair Prompt<br>(Diagnostics + Verbatim Evidence)"]
    end

    subgraph Repair_Execution ["4. Single Repair Execution (Attempt 1 / Budget=1)"]
        REP_STAGE --> LLM2["LLMClient.generate(repair_prompt)"]
        LLM2 -->|Timeout / Error| TERM_REP_TIMEOUT["REPAIR_FAILED<br>(GENERATION_TIMEOUT)"]
        LLM2 --> PARSE2["Strict Re-Parsing"]
        PARSE2 -->|Malformed JSON / Bad Schema| TERM_REP_PARSE["REPAIR_FAILED<br>(PARSING_ERROR)"]
        PARSE2 -->|Model Abstained in Repair| TERM_REP_ABSTAIN["INSUFFICIENT_EVIDENCE<br>(MODEL_DECLARED_ABSTENTION)"]
        
        PARSE2 -->|Valid Claim Structure| GUARD{"Guard Evidence Registry<br>(Permitted Chunk IDs)"}
        GUARD -->|Contains Fabricated Chunk ID| TERM_REP_GROUND["REPAIR_FAILED<br>(EVIDENCE_GROUNDING_ERROR)"]
        GUARD -->|All Chunk IDs Permitted| VERIF2["Complete Re-Verification Pipeline<br>(Stages 1 to 5)"]
    end

    subgraph Re_Verification_Outcome ["5. Re-Verification Outcome"]
        VERIF2 -->|All Stages Pass| TERM_REP_OK["VERIFIED<br>(VERIFIED_AFTER_REPAIR)"]
        VERIF2 -->|Any Stage Fails| TERM_REP_FAIL["REPAIR_FAILED<br>(Detailed Stage Diagnostic)"]
    end
```

---

## 3. Repairability Classification Taxonomy

Errors encountered during parsing or verification are classified deterministically by `classify_repairability()`:

| Category | Diagnostic Trigger | Repairable? | Default Action | Technical Justification |
| :--- | :--- | :---: | :---: | :--- |
| **SYNTAX** | `JSON_DECODE_ERROR` | **Yes** | Attempt Repair | Syntactic formatting flaws (broken markdown fence, unclosed braces) can be repaired without changing underlying facts. |
| **SCHEMA** | `SCHEMA_VALIDATION_ERROR` | **Yes** | Attempt Repair | Missing outer schema envelopes (e.g. omitting `final_claim_id`) can be supplied by re-prompting. |
| **REFERENTIAL** | `REFERENTIAL_INTEGRITY_ERROR` | **Yes** | Attempt Repair | Typographical discrepancies in operand IDs (e.g. referencing `op_1` as `op1`) or step ordering are resolvable. |
| **ARITHMETIC** | `ARITHMETIC_ERROR` | **Yes** | Attempt Repair | If operands are grounded, calculation failures (e.g. division by zero from reversed operands) can be corrected by selecting the proper operation. |
| **DIRECTION** | `DIRECTIONAL_CONTRADICTION` | **Yes** | Attempt Repair | Accounting polarity conflicts (e.g. naive subtraction yielding decrease on expenses when disclosures state growth) can be repaired via `operation: "economic_change"`. |
| **METADATA** | `UNIT_VALIDATION_ERROR` | **Conditional** | Attempt Repair if Evidence Unambiguous | If the model declared `USD` while evidence clearly specifies `million USD`, targeted feedback guides correction. |
| **METADATA** | `PERIOD_VALIDATION_ERROR` | **Conditional** | Attempt Repair if Evidence Unambiguous | If the model declared an adjacent year while evidence contains the exact filing period, targeted feedback guides correction. |
| **GROUNDING** | `EVIDENCE_GROUNDING_ERROR` (Value) | **No** | **Abstain / Reject** | Hallucinated numerical values do not exist in the filing; allowing repair encourages secondary hallucinations. |
| **GROUNDING** | `EVIDENCE_GROUNDING_ERROR` (Chunk ID) | **No** | **Abstain / Reject** | Fabricated evidence IDs violate provenance guarantees; repair is prohibited. |
| **GROUNDING** | `EVIDENCE_GROUNDING_ERROR` (Span) | **No** | **Abstain / Reject** | Claimed text span is absent from the cited passage; grounding cannot be confirmed. |
| **ABSTENTION** | `INSUFFICIENT_EVIDENCE` | **No** | **Abstain Immediately** | Model has explicitly determined that the query cannot be answered from the retrieved context. |
| **RETRIEVAL** | `EMPTY_RETRIEVAL_RESULTS` | **No** | **Early Stop** | Zero context retrieved; no evidence pool exists to ground any repair. |
| **INFRASTRUCTURE**| `GENERATION_TIMEOUT` / `MODEL_FAILURE` | **No** | **Fail Generation** | Upstream infrastructure failure. |
| **EXCEPTION** | `VERIFICATION_EXCEPTION` | **No** | **Fail Verification** | Verifier crash indicates corrupted data or internal bug; unsafe to repair. |

---

## 4. Evidence Integrity & Registry Enforcement

Evidence integrity is maintained through an immutable registry:

1. **Permitted Chunk Registry**:
   Prior to initial generation, `ClaimGenerator` records the exact universe of visible chunk identifiers:
   $$\mathcal{C}_{\text{permitted}} = \left\{ c.\text{chunk\_id} \mid c \in \mathcal{P}_{\text{visible}} \right\}$$
2. **Registry Verification Gate**:
   In `RepairEngine.verify_permitted_evidence()`, every operand in the repaired claim is checked against $\mathcal{C}_{\text{permitted}}$:
   $$\forall o \in \text{Operands}(\text{Claim}_{\text{repaired}}), \quad o.\text{chunk\_id} \in \mathcal{C}_{\text{permitted}}$$
   If any operand references an ID outside $\mathcal{C}_{\text{permitted}}$, the claim is rejected immediately with `detailed_status="EVIDENCE_GROUNDING_ERROR"` without running calculations.
3. **Immutable Retrieval Boundary**:
   The repair prompt is constructed strictly from the original evidence context string. The repair model is barred from querying the retrieval index or fetching new documents.

---

## 5. Metrics Formulation & Mathematical Definitions

The repair evaluation framework implements explicit metrics with transparent denominators:

| Metric Name | Mathematical Definition | Denominator | Purpose |
| :--- | :---: | :--- | :--- |
| **Initial Verification Success Rate** | $\frac{N_{\text{initial\_verified}}}{N_{\text{total}}}$ | All evaluated queries | Baseline precision of the unassisted model. |
| **Repair Eligibility Rate** | $\frac{N_{\text{repair\_eligible}}}{N_{\text{total}}}$ | All evaluated queries | Proportion of queries with repairable failures. |
| **Failure Eligibility Rate** | $\frac{N_{\text{repair\_eligible}}}{N_{\text{initial\_failed}}}$ | Initial failure subset | Proportion of failed queries eligible for repair. |
| **Repair Attempt Rate** | $\frac{N_{\text{repair\_attempted}}}{N_{\text{total}}}$ | All evaluated queries | Proportion of queries where repair was triggered. |
| **Repair Success Rate** | $\frac{N_{\text{repaired\_and\_verified}}}{N_{\text{repair\_attempted}}}$ | Executed repair attempts | Efficacy of diagnostic repair prompt. |
| **Final Verification Success Rate** | $\frac{N_{\text{final\_verified}}}{N_{\text{total}}}$ | All evaluated queries | Net end-to-end verified precision after repair. |
| **Abstention Rate** | $\frac{N_{\text{abstained}}}{N_{\text{total}}}$ | All evaluated queries | Proportion of queries safely abstained. |
| **Repair Failure Rate** | $\frac{N_{\text{repair\_failed}}}{N_{\text{repair\_attempted}}}$ | Executed repair attempts | Frequency of unresolvable repairs. |
| **Mean Additional Repair Latency** | $\frac{1}{K} \sum_{k=1}^K \Delta t_{\text{repair}, k}$ | Executed repair attempts | Latency overhead introduced by repair attempt. |

---

## 6. Audit Logging & Telemetry

Every execution that invokes the repair engine populates a `RepairAttemptAudit` object:

```python
@dataclass
class RepairAttemptAudit:
    attempt_number: int                  # Capped at 1
    is_repairable: bool                 # Classification outcome
    repairability_category: str         # SYNTAX, SCHEMA, ARITHMETIC, etc.
    repairability_reason: str           # Detailed classification explanation
    diagnostic_feedback: list[str]      # Error strings provided in repair prompt
    repair_prompt_preview: str          # First 256 characters of repair prompt
    repair_raw_completion: str          # Raw LLM output
    repaired_claim: Optional[dict]      # Parsed claim structure
    reverification_status: str          # VERIFIED, REPAIR_FAILED, INSUFFICIENT_EVIDENCE
    reverification_detailed_status: str # Granular verification failure code
    reverification_passed: bool         # Boolean success flag
    generation_latency_ms: float        # Latency of repair generation
    verification_latency_ms: float      # Latency of full re-verification
    total_repair_latency_ms: float      # Total wall-clock repair duration
```

This record is attached directly to `ClaimGenerationResult.repair_audit` and serialized in `ClaimGenerationResult.to_dict()`, ensuring 100% auditability for academic review.

---

## 7. Deliverables Summary

| Deliverable | Location | Status | Description |
| :--- | :--- | :---: | :--- |
| **Repair Engine** | [`src/generation/repair_engine.py`](file:///C:/Users/palak/LLM-FINANCE/src/generation/repair_engine.py) | **Created** | Orchestrates classification, prompt construction, single-turn repair, and re-verification. |
| **Package Exports** | [`src/generation/__init__.py`](file:///C:/Users/palak/LLM-FINANCE/src/generation/__init__.py) | **Updated** | Exported `RepairEngine`, `RepairAttemptAudit`, `classify_repairability`, `compute_repair_metrics`. |
| **Claim Generator** | [`src/generation/claim_generator.py`](file:///C:/Users/palak/LLM-FINANCE/src/generation/claim_generator.py) | **Updated** | Integrated `RepairEngine` with `enable_repair` flag and telemetry auditing. |
| **Integration Tests** | [`tests/test_repair_engine.py`](file:///C:/Users/palak/LLM-FINANCE/tests/test_repair_engine.py) | **Created** | 21 deterministic unit and integration tests covering all 20 required scenarios + metrics. |
| **Audit Report** | [`results/phase3_4_preimplementation_audit.md`](file:///C:/Users/palak/LLM-FINANCE/results/phase3_4_preimplementation_audit.md) | **Created** | Pre-implementation diagnostic and taxonomy audit. |
| **Repair Report** | [`results/phase3_4_repair_report.md`](file:///C:/Users/palak/LLM-FINANCE/results/phase3_4_repair_report.md) | **Created** | This comprehensive scientific and engineering document. |
| **Test Report** | [`results/phase3_4_test_report.md`](file:///C:/Users/palak/LLM-FINANCE/results/phase3_4_test_report.md) | **Created** | Formal pytest execution logs across all test suites. |
| **Experiment Config** | [`results/phase3_4_experiment_config.json`](file:///C:/Users/palak/LLM-FINANCE/results/phase3_4_experiment_config.json) | **Created** | Machine-readable configuration of policies, taxonomy, and environments. |

---

## 8. Conclusion & Readiness for Phase 3.5

Phase 3.4 is complete and verified. The repair engine provides a safe, bounded mechanism to recover from procedural errors without sacrificing evidence grounding or numerical veracity.

The system is fully prepared for **Phase 3.5: FinQA Development-Set Benchmark Evaluation**:
- Running the complete pipeline (Dense + Parent-Child retrieval, prompt assembly, LLM generation, output parsing, repair engine, and symbolic verification) against the official FinQA validation split.
- Profiling GPU inference and verification latency on the NVIDIA RTX 4060 Laptop GPU.
