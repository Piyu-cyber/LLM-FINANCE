"""tests/test_repair_engine.py — Comprehensive tests for claim repair and abstention engine.

Validates the 20 mandatory Phase 3.4 integration scenarios:
1. Valid claim: no repair needed.
2. Repairable arithmetic error: repair succeeds.
3. Repairable narrative contradiction: repair succeeds when evidence is unambiguous.
4. Invalid JSON: repair succeeds.
5. Missing required field: repair succeeds.
6. Unsupported operand: repair is not attempted.
7. Fabricated chunk ID: repair is not attempted by default.
8. Insufficient evidence: abstention without repair.
9. Repair response contains another unsupported operand: reject.
10. Repair response contains fabricated evidence identity: reject.
11. Repair produces incorrect arithmetic: reject.
12. Repair produces contradictory narrative: reject.
13. Repair returns malformed JSON: REPAIR_FAILED.
14. Repair timeout: REPAIR_FAILED.
15. Repair attempts remain limited to one.
16. Original evidence registry remains unchanged.
17. Original verification diagnostics are preserved.
18. Repair cannot bypass schema validation.
19. Repair cannot bypass unit or period validation.
20. Explicit model abstention triggers no repair.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.generation.claim_generator import ClaimGenerationResult, ClaimGenerator
from src.generation.llm_client import MockLLMClient
from src.generation.output_parser import OutputParser
from src.generation.repair_engine import (
    RepairAttemptAudit,
    RepairEngine,
    classify_repairability,
    compute_repair_metrics,
)


# ---------------------------------------------------------------------------
# Test Fixtures: SEC 10-K Grounded Evidence
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_aapl_evidence():
    return [
        {
            "example_id": "AAPL_2019",
            "chunk_id": "table_row_0",
            "source": "table",
            "company": "AAPL",
            "year": "2019",
            "text": "[DOC: AAPL 2019] [UNIT: in millions of dollars] Year 2019: $ 260,174 | Year 2018: $ 265,595",
        },
        {
            "example_id": "AAPL_2019",
            "chunk_id": "text_0",
            "source": "pre_text",
            "company": "AAPL",
            "year": "2019",
            "text": "Amounts are in millions of dollars. Net sales decreased by 5,421 million in fiscal 2019.",
        },
    ]


@pytest.fixture
def sample_msft_expense_evidence():
    return [
        {
            "example_id": "MSFT_2009",
            "chunk_id": "table_row_4",
            "source": "table",
            "company": "MSFT",
            "year": "2009",
            "text": "[DOC: MSFT 2009] [UNIT: millions of dollars] Interest expense | 2009: $ (600) | 2008: $ (511)",
        },
        {
            "example_id": "MSFT_2009",
            "chunk_id": "text_1",
            "source": "pre_text",
            "company": "MSFT",
            "year": "2009",
            "text": "in millions of dollars. Interest expense increased in 2009 due to senior note issuance.",
        },
    ]


@pytest.fixture
def sample_corpus_index(sample_aapl_evidence, sample_msft_expense_evidence):
    return {
        "AAPL_2019": sample_aapl_evidence,
        "MSFT_2009": sample_msft_expense_evidence,
    }


def make_valid_aapl_claim():
    return {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales decreased by 5,421 million USD from 2018 to 2019.",
                "value": -5421.0,
                "unit": "million USD",
                "citations": ["table_row_0", "text_0"],
                "operands": [
                    {
                        "id": "op_2019",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 260,174",
                        "context_text": "Year 2019: $ 260,174 | Year 2018: $ 265,595",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2019",
                    },
                    {
                        "id": "op_2018",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 265,595",
                        "context_text": "Year 2019: $ 260,174 | Year 2018: $ 265,595",
                        "normalized_value": 265595.0,
                        "unit": "million USD",
                        "period": "2018",
                    },
                ],
                "program": [
                    {
                        "id": "step_1",
                        "operation": "subtract",
                        "operands": ["op_2019", "op_2018"],
                    }
                ],
                "referent": {
                    "entity": "net sales",
                    "period": "2019",
                    "unit": "million USD",
                },
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }


# =========================================================================
# Scenario 1: Valid claim — no repair needed
# =========================================================================
def test_valid_claim_no_repair_needed(sample_aapl_evidence, sample_corpus_index):
    valid_claim = make_valid_aapl_claim()
    mock_llm = MockLLMClient(default_response=json.dumps(valid_claim))

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was the change in net sales from 2018 to 2019?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.verified_answer == -5421.0
    assert result.repair_attempted is False
    assert result.repair_attempts == 0
    assert result.initial_status == "VERIFIED"


# =========================================================================
# Scenario 2: Repairable arithmetic error — repair succeeds
# =========================================================================
def test_repairable_arithmetic_error_succeeds(sample_aapl_evidence, sample_corpus_index):
    # Initial: division by zero
    bad_claim = make_valid_aapl_claim()
    bad_claim["claims"][0]["program"] = [
        {"id": "step_zero", "operation": "subtract", "operands": ["op_2019", "op_2019"]},
        {"id": "step_div", "operation": "divide", "operands": ["op_2019", "step_zero"]},
    ]

    # Repaired: correct subtraction
    good_claim = make_valid_aapl_claim()

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim), json.dumps(good_claim)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was the change in net sales from 2018 to 2019?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.detailed_status == "VERIFIED_AFTER_REPAIR"
    assert result.verified_answer == -5421.0
    assert result.repair_attempted is True
    assert result.repair_attempts == 1
    assert result.initial_status == "VERIFICATION_FAILED"
    assert result.initial_detailed_status == "ARITHMETIC_ERROR"
    assert result.repair_audit is not None
    assert result.repair_audit["reverification_passed"] is True


# =========================================================================
# Scenario 3: Repairable narrative contradiction — repair succeeds
# =========================================================================
def test_repairable_narrative_contradiction_succeeds(sample_msft_expense_evidence, sample_corpus_index):
    # Initial: naive subtraction yields negative (-89) on expense, contradicting "increased"
    bad_msft_claim = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Interest expense change.",
                "value": -89.0,
                "unit": "million USD",
                "citations": ["table_row_4", "text_1"],
                "operands": [
                    {
                        "id": "op_2009",
                        "chunk_id": "table_row_4",
                        "span_text": "$ (600)",
                        "context_text": "Interest expense | 2009: $ (600) | 2008: $ (511)",
                        "normalized_value": -600.0,
                        "unit": "million USD",
                        "period": "2009",
                    },
                    {
                        "id": "op_2008",
                        "chunk_id": "table_row_4",
                        "span_text": "$ (511)",
                        "context_text": "Interest expense | 2009: $ (600) | 2008: $ (511)",
                        "normalized_value": -511.0,
                        "unit": "million USD",
                        "period": "2008",
                    },
                ],
                "program": [{"id": "step_1", "operation": "subtract", "operands": ["op_2009", "op_2008"]}],
                "referent": {"entity": "interest expense", "period": "2009", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    # Repaired: operation: economic_change yields +89.0
    good_msft_claim = json.loads(json.dumps(bad_msft_claim))
    good_msft_claim["claims"][0]["program"][0]["operation"] = "economic_change"
    good_msft_claim["claims"][0]["value"] = 89.0

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_msft_claim), json.dumps(good_msft_claim)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="By how much did interest expense increase?",
        example_id="MSFT_2009",
        retrieved_chunks=sample_msft_expense_evidence,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.detailed_status == "VERIFIED_AFTER_REPAIR"
    assert result.verified_answer == 89.0
    assert result.repair_attempted is True
    assert result.initial_detailed_status == "DIRECTIONAL_CONTRADICTION"


# =========================================================================
# Scenario 4: Invalid JSON — repair succeeds
# =========================================================================
def test_invalid_json_repair_succeeds(sample_aapl_evidence, sample_corpus_index):
    # Malformed JSON with trailing broken brace
    broken_json = '{"claims": [{"id": "c1", "value": 100, ... broken JSON'
    good_claim = make_valid_aapl_claim()

    mock_llm = MockLLMClient(
        response_queue=[broken_json, json.dumps(good_claim)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.detailed_status == "VERIFIED_AFTER_REPAIR"
    assert result.repair_attempted is True
    assert result.initial_status == "PARSING_ERROR"
    assert result.initial_detailed_status == "JSON_DECODE_ERROR"


# =========================================================================
# Scenario 5: Missing required field — repair succeeds
# =========================================================================
def test_missing_required_field_repair_succeeds(sample_aapl_evidence, sample_corpus_index):
    # Missing 'final_claim_id'
    bad_claim = make_valid_aapl_claim()
    del bad_claim["final_claim_id"]

    good_claim = make_valid_aapl_claim()

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim), json.dumps(good_claim)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.repair_attempted is True
    assert result.initial_detailed_status == "SCHEMA_VALIDATION_ERROR"


# =========================================================================
# Scenario 6: Unsupported operand — repair is not attempted
# =========================================================================
def test_unsupported_operand_repair_not_attempted(sample_aapl_evidence, sample_corpus_index):
    bad_claim = make_valid_aapl_claim()
    # Hallucinated number 999999.0 not in evidence
    bad_claim["claims"][0]["operands"][0]["span_text"] = "$ 999,999"
    bad_claim["claims"][0]["operands"][0]["normalized_value"] = 999999.0

    mock_llm = MockLLMClient(default_response=json.dumps(bad_claim))

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "EVIDENCE_GROUNDING_ERROR"
    assert result.repair_attempted is False
    assert result.repair_eligible is False
    assert result.repair_audit is not None
    assert result.repair_audit["is_repairable"] is False


# =========================================================================
# Scenario 7: Fabricated chunk ID — repair is not attempted by default
# =========================================================================
def test_fabricated_chunk_id_repair_not_attempted(sample_aapl_evidence, sample_corpus_index):
    bad_claim = make_valid_aapl_claim()
    bad_claim["claims"][0]["operands"][0]["chunk_id"] = "table_row_fake_999"

    mock_llm = MockLLMClient(default_response=json.dumps(bad_claim))

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "EVIDENCE_GROUNDING_ERROR"
    assert result.repair_attempted is False
    assert result.repair_eligible is False


# =========================================================================
# Scenario 8: Insufficient evidence — abstention without repair
# =========================================================================
def test_insufficient_evidence_no_repair(sample_aapl_evidence, sample_corpus_index):
    abstain_claim = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Evidence does not disclose this metric.",
                "value": None,
                "unit": "million USD",
                "citations": [],
                "operands": [],
                "program": [],
                "referent": {"entity": "unknown", "period": "2019", "unit": "million USD"},
                "status": "insufficient_evidence",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(abstain_claim))

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was the tax rate?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.detailed_status == "MODEL_DECLARED_ABSTENTION"
    assert result.repair_attempted is False
    assert result.verified_answer is None


# =========================================================================
# Scenario 9: Repair response contains another unsupported operand — reject
# =========================================================================
def test_repair_response_contains_another_unsupported_operand_rejected(sample_aapl_evidence, sample_corpus_index):
    # Initial: arithmetic error (repairable)
    bad_claim_1 = make_valid_aapl_claim()
    bad_claim_1["claims"][0]["program"] = [
        {"id": "step_zero", "operation": "subtract", "operands": ["op_2019", "op_2019"]},
        {"id": "step_div", "operation": "divide", "operands": ["op_2019", "step_zero"]},
    ]

    # Repaired: attempts to fix arithmetic by inventing 777777.0
    bad_claim_2 = make_valid_aapl_claim()
    bad_claim_2["claims"][0]["operands"][0]["span_text"] = "$ 777,777"
    bad_claim_2["claims"][0]["operands"][0]["normalized_value"] = 777777.0

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), json.dumps(bad_claim_2)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was the change?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "EVIDENCE_GROUNDING_ERROR"
    assert result.repair_attempted is True


# =========================================================================
# Scenario 10: Repair response contains fabricated evidence identity — reject
# =========================================================================
def test_repair_response_contains_fabricated_evidence_identity_rejected(sample_aapl_evidence, sample_corpus_index):
    # Initial: missing final_claim_id (repairable)
    bad_claim_1 = make_valid_aapl_claim()
    del bad_claim_1["final_claim_id"]

    # Repaired: adds final_claim_id, but invents chunk_id 'table_row_phantom'
    bad_claim_2 = make_valid_aapl_claim()
    bad_claim_2["claims"][0]["operands"][0]["chunk_id"] = "table_row_phantom"

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), json.dumps(bad_claim_2)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "EVIDENCE_GROUNDING_ERROR"
    assert result.repair_attempted is True


# =========================================================================
# Scenario 11: Repair produces incorrect arithmetic — reject
# =========================================================================
def test_repair_produces_incorrect_arithmetic_rejected(sample_aapl_evidence, sample_corpus_index):
    # Initial: schema error (repairable)
    bad_claim_1 = make_valid_aapl_claim()
    del bad_claim_1["final_claim_id"]

    # Repaired: adds final_claim_id, but uses division by zero
    bad_claim_2 = make_valid_aapl_claim()
    bad_claim_2["claims"][0]["program"] = [
        {"id": "step_zero", "operation": "subtract", "operands": ["op_2019", "op_2019"]},
        {"id": "step_div", "operation": "divide", "operands": ["op_2019", "step_zero"]},
    ]

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), json.dumps(bad_claim_2)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "ARITHMETIC_ERROR"


# =========================================================================
# Scenario 12: Repair produces contradictory narrative — reject
# =========================================================================
def test_repair_produces_contradictory_narrative_rejected(sample_msft_expense_evidence, sample_corpus_index):
    # Initial: schema error
    bad_claim_1 = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Interest expense.",
                "value": -89.0,
                "unit": "million USD",
                "citations": ["table_row_4", "text_1"],
                "operands": [
                    {
                        "id": "op_2009",
                        "chunk_id": "table_row_4",
                        "span_text": "$ (600)",
                        "context_text": "Interest expense | 2009: $ (600) | 2008: $ (511)",
                        "normalized_value": -600.0,
                        "unit": "million USD",
                        "period": "2009",
                    },
                    {
                        "id": "op_2008",
                        "chunk_id": "table_row_4",
                        "span_text": "$ (511)",
                        "context_text": "Interest expense | 2009: $ (600) | 2008: $ (511)",
                        "normalized_value": -511.0,
                        "unit": "million USD",
                        "period": "2008",
                    },
                ],
                "program": [{"id": "step_1", "operation": "subtract", "operands": ["op_2009", "op_2008"]}],
                "referent": {"entity": "interest expense", "period": "2009", "unit": "million USD"},
                "status": "answered",
            }
        ],
        # Omits final_claim_id
    }

    # Repaired adds final_claim_id, but still calculates -89 (decrease) vs narrative "increased"
    bad_claim_2 = json.loads(json.dumps(bad_claim_1))
    bad_claim_2["final_claim_id"] = "c1"

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), json.dumps(bad_claim_2)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="By how much did interest expense increase?",
        example_id="MSFT_2009",
        retrieved_chunks=sample_msft_expense_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "DIRECTIONAL_CONTRADICTION"


# =========================================================================
# Scenario 13: Repair returns malformed JSON — REPAIR_FAILED
# =========================================================================
def test_repair_returns_malformed_json_fails(sample_aapl_evidence, sample_corpus_index):
    # Initial: missing final_claim_id
    bad_claim_1 = make_valid_aapl_claim()
    del bad_claim_1["final_claim_id"]

    # Repair completion: broken JSON syntax
    broken_repair = '{"claims": [broken JSON...'

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), broken_repair]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "JSON_DECODE_ERROR"


# =========================================================================
# Scenario 14: Repair timeout — REPAIR_FAILED
# =========================================================================
def test_repair_timeout_fails(sample_aapl_evidence, sample_corpus_index):
    bad_claim_1 = make_valid_aapl_claim()
    del bad_claim_1["final_claim_id"]

    # LLM returns timeout on repair
    mock_llm = MockLLMClient(
        response_queue=[
            json.dumps(bad_claim_1),
            "Generation timed out after 30.0s.",
        ]
    )

    # Inject timeout simulator on second call
    call_count = 0
    orig_generate = mock_llm.generate

    def timeout_on_second(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            from src.generation.llm_client import GenerationResult
            return GenerationResult(
                text="",
                model_name="mock-financial-llm",
                status="TIMEOUT",
                error_message="Simulated repair timeout after 60s",
            )
        return orig_generate(prompt)

    mock_llm.generate = timeout_on_second

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "GENERATION_TIMEOUT"


# =========================================================================
# Scenario 15: Repair attempts remain limited to one
# =========================================================================
def test_repair_attempts_remain_limited_to_one(sample_aapl_evidence, sample_corpus_index):
    bad_claim = make_valid_aapl_claim()
    bad_claim["claims"][0]["program"] = [
        {"id": "step_zero", "operation": "subtract", "operands": ["op_2019", "op_2019"]},
        {"id": "step_div", "operation": "divide", "operands": ["op_2019", "step_zero"]},
    ]

    mock_llm = MockLLMClient(default_response=json.dumps(bad_claim))

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
        max_repair_budget=1,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.repair_attempts == 1
    assert result.repair_attempted is True
    assert result.status == "REPAIR_FAILED"
    # Verify no second repair was made
    assert len(mock_llm.call_history) == 2


# =========================================================================
# Scenario 16: Original evidence registry remains unchanged
# =========================================================================
def test_original_evidence_registry_remains_unchanged(sample_aapl_evidence, sample_corpus_index):
    engine = RepairEngine(llm_client=MockLLMClient())
    permitted = frozenset(["table_row_0", "text_0"])

    claim_valid = make_valid_aapl_claim()
    is_valid, err = engine.verify_permitted_evidence(claim_valid, permitted)
    assert is_valid is True
    assert err is None

    claim_tampered = make_valid_aapl_claim()
    claim_tampered["claims"][0]["operands"][0]["chunk_id"] = "table_row_injected"
    is_valid, err = engine.verify_permitted_evidence(claim_tampered, permitted)
    assert is_valid is False
    assert "table_row_injected" in err


# =========================================================================
# Scenario 17: Original verification diagnostics are preserved
# =========================================================================
def test_original_verification_diagnostics_preserved(sample_aapl_evidence, sample_corpus_index):
    bad_claim_1 = make_valid_aapl_claim()
    bad_claim_1["claims"][0]["program"] = [
        {"id": "step_zero", "operation": "subtract", "operands": ["op_2019", "op_2019"]},
        {"id": "step_div", "operation": "divide", "operands": ["op_2019", "step_zero"]},
    ]
    good_claim_2 = make_valid_aapl_claim()

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), json.dumps(good_claim_2)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is True
    assert result.initial_detailed_status == "ARITHMETIC_ERROR"
    assert len(result.initial_diagnostics) > 0
    assert any("division by zero" in d.lower() or "calculation" in d.lower() for d in result.initial_diagnostics)


# =========================================================================
# Scenario 18: Repair cannot bypass schema validation
# =========================================================================
def test_repair_cannot_bypass_schema_validation(sample_aapl_evidence, sample_corpus_index):
    bad_claim_1 = make_valid_aapl_claim()
    bad_claim_1["claims"][0]["program"] = [
        {"id": "step_zero", "operation": "subtract", "operands": ["op_2019", "op_2019"]},
        {"id": "step_div", "operation": "divide", "operands": ["op_2019", "step_zero"]},
    ]

    # Repaired claim leaves out 'claims' list
    bad_claim_2 = {"final_claim_id": "c1"}

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), json.dumps(bad_claim_2)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "SCHEMA_VALIDATION_ERROR"


# =========================================================================
# Scenario 19: Repair cannot bypass unit or period validation
# =========================================================================
def test_repair_cannot_bypass_unit_or_period_validation(sample_aapl_evidence, sample_corpus_index):
    bad_claim_1 = make_valid_aapl_claim()
    bad_claim_1["claims"][0]["program"] = [
        {"id": "step_zero", "operation": "subtract", "operands": ["op_2019", "op_2019"]},
        {"id": "step_div", "operation": "divide", "operands": ["op_2019", "step_zero"]},
    ]

    # Repaired claim fixes arithmetic but sets unit to 'shares' (unit error)
    bad_claim_2 = make_valid_aapl_claim()
    bad_claim_2["claims"][0]["operands"][0]["unit"] = "shares"

    mock_llm = MockLLMClient(
        response_queue=[json.dumps(bad_claim_1), json.dumps(bad_claim_2)]
    )

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "REPAIR_FAILED"
    assert result.detailed_status == "UNIT_VALIDATION_ERROR"


# =========================================================================
# Scenario 20: Explicit model abstention triggers no repair
# =========================================================================
def test_explicit_model_abstention_triggers_no_repair(sample_aapl_evidence, sample_corpus_index):
    abstain_claim = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Insufficient evidence provided.",
                "value": None,
                "unit": "million USD",
                "citations": [],
                "operands": [],
                "program": [],
                "referent": {"entity": "metric", "period": "2019", "unit": "million USD"},
                "status": "insufficient_evidence",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(abstain_claim))

    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        enable_repair=True,
    )

    result = generator.generate_and_verify(
        question="What was the tax rate?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.detailed_status == "MODEL_DECLARED_ABSTENTION"
    assert result.repair_attempted is False
    assert len(mock_llm.call_history) == 1


# =========================================================================
# Additional Test: Metrics computation verification
# =========================================================================
def test_compute_repair_metrics_calculation():
    r1 = ClaimGenerationResult(
        question="q1",
        example_id="AAPL_2019",
        retrieval_config="direct",
        retrieved_chunk_ids=["c1"],
        context_mode="child_only",
        context_token_count=100,
        context_is_truncated=False,
        generation_result=None,
        parsed_claim=None,
        verification_result=None,
        status="VERIFIED",
        detailed_status="VERIFIED",
        verified_answer=100.0,
        is_verified=True,
        initial_status="VERIFIED",
        repair_attempted=False,
        repair_eligible=False,
    )

    r2 = ClaimGenerationResult(
        question="q2",
        example_id="AAPL_2019",
        retrieval_config="direct",
        retrieved_chunk_ids=["c1"],
        context_mode="child_only",
        context_token_count=100,
        context_is_truncated=False,
        generation_result=None,
        parsed_claim=None,
        verification_result=None,
        status="VERIFIED",
        detailed_status="VERIFIED_AFTER_REPAIR",
        verified_answer=200.0,
        is_verified=True,
        initial_status="VERIFICATION_FAILED",
        repair_attempted=True,
        repair_eligible=True,
        repair_audit={
            "generation_latency_ms": 15.0,
            "verification_latency_ms": 5.0,
            "total_repair_latency_ms": 20.0,
        },
    )

    r3 = ClaimGenerationResult(
        question="q3",
        example_id="AAPL_2019",
        retrieval_config="direct",
        retrieved_chunk_ids=["c1"],
        context_mode="child_only",
        context_token_count=100,
        context_is_truncated=False,
        generation_result=None,
        parsed_claim=None,
        verification_result=None,
        status="REPAIR_FAILED",
        detailed_status="ARITHMETIC_ERROR",
        verified_answer=None,
        is_verified=False,
        initial_status="VERIFICATION_FAILED",
        repair_attempted=True,
        repair_eligible=True,
        repair_audit={
            "generation_latency_ms": 10.0,
            "verification_latency_ms": 4.0,
            "total_repair_latency_ms": 14.0,
        },
    )

    r4 = ClaimGenerationResult(
        question="q4",
        example_id="AAPL_2019",
        retrieval_config="direct",
        retrieved_chunk_ids=["c1"],
        context_mode="child_only",
        context_token_count=100,
        context_is_truncated=False,
        generation_result=None,
        parsed_claim=None,
        verification_result=None,
        status="INSUFFICIENT_EVIDENCE",
        detailed_status="MODEL_DECLARED_ABSTENTION",
        verified_answer=None,
        is_verified=False,
        initial_status="INSUFFICIENT_EVIDENCE",
        repair_attempted=False,
        repair_eligible=False,
    )

    metrics = compute_repair_metrics([r1, r2, r3, r4])

    assert metrics["total_queries"] == 4
    assert metrics["initial_verified_count"] == 1
    assert metrics["initial_verification_success_rate"] == 0.25
    assert metrics["repair_eligible_count"] == 2
    assert metrics["repair_attempted_count"] == 2
    assert metrics["repaired_and_verified_count"] == 1
    assert metrics["repair_success_rate"] == 0.5
    assert metrics["final_verified_count"] == 2
    assert metrics["final_verification_success_rate"] == 0.5
    assert metrics["abstained_count"] == 1
    assert metrics["abstention_rate"] == 0.25
    assert metrics["avg_repair_generation_latency_ms"] == 12.5
    assert metrics["avg_repair_verification_latency_ms"] == 4.5
    assert metrics["avg_total_repair_latency_ms"] == 17.0
