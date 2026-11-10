"""test_claim_generator.py — End-to-end integration tests for ClaimGenerator.

Covers all 17 required scenarios:
1. Valid grounded revenue calculation.
2. Valid expense magnitude calculation.
3. Ungrounded numerical operand.
4. Fabricated chunk ID.
5. Incorrect financial period.
6. Incorrect financial unit.
7. Incorrect arithmetic.
8. Contradictory narrative direction.
9. Invalid JSON.
10. Missing required claim fields.
11. Explicit insufficient-evidence response.
12. Empty retrieval results.
13. Generation timeout.
14. Verification exception.
15. Parent-expanded context grounding.
16. Deduplicated parent context grounding.
17. Weighted RRF retrieval integration.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Stubs for heavy dependencies if running in minimal system Python (non-virtualenv)
_IN_VENV = (sys.prefix != getattr(sys, "base_prefix", sys.prefix))
if not _IN_VENV:
    if "rank_bm25" not in sys.modules:
        class _MockBM25:
            def __init__(self, corpus=None):
                self.corpus = corpus or []
            def get_scores(self, query_tokens):
                return np.zeros(len(self.corpus))
        _bm_mock = MagicMock()
        _bm_mock.BM25Okapi = _MockBM25
        sys.modules["rank_bm25"] = _bm_mock

try:
    from src.generation.claim_generator import (
        ClaimGenerationResult,
        ClaimGenerator,
        weighted_reciprocal_rank_fusion,
    )
    from src.generation.context_builder import ContextBuilder
    from src.generation.llm_client import MockLLMClient
except ImportError:
    from claim_generator import (
        ClaimGenerationResult,
        ClaimGenerator,
        weighted_reciprocal_rank_fusion,
    )
    from context_builder import ContextBuilder
    from llm_client import MockLLMClient


@pytest.fixture
def sample_aapl_evidence():
    return [
        {
            "example_id": "AAPL_2019",
            "chunk_id": "table_row_0",
            "source": "table",
            "company": "AAPL",
            "year": "2019",
            "text": "millions of US dollars: Net sales | 2019: $ 260,174 | 2018: $ 265,595",
        },
        {
            "example_id": "AAPL_2019",
            "chunk_id": "text_0",
            "source": "pre_text",
            "company": "AAPL",
            "year": "2019",
            "text": "All monetary amounts are in millions of US dollars for fiscal 2019 and 2018.",
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
            "text": "millions of dollars: Interest expense | 2009: $ (600) | 2008: $ (511)",
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


# =========================================================================
# Scenario 1: Valid Grounded Revenue Calculation
# =========================================================================
def test_valid_grounded_revenue_calculation(sample_aapl_evidence, sample_corpus_index):
    claim_dict = {
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
                        "context_text": "Net sales | $ 260,174 | $ 265,595",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2019",
                    },
                    {
                        "id": "op_2018",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 265,595",
                        "context_text": "Net sales | $ 260,174 | $ 265,595",
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

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was the change in net sales from 2018 to 2019?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.detailed_status == "VERIFIED"
    assert result.verified_answer == -5421.0
    assert result.error_diagnostics == []
    assert result.verification_result["valid"] is True


# =========================================================================
# Scenario 2: Valid Expense Magnitude Calculation (Accounting Polarity)
# =========================================================================
def test_valid_expense_magnitude_calculation(sample_msft_expense_evidence, sample_corpus_index):
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Interest expense increased by 89 million USD from 2008 to 2009.",
                "value": 89.0,
                "unit": "million USD",
                "citations": ["table_row_4", "text_1"],
                "operands": [
                    {
                        "id": "op_2009",
                        "chunk_id": "table_row_4",
                        "span_text": "(600)",
                        "context_text": "Interest expense | $ (600) | $ (511)",
                        "normalized_value": -600.0,
                        "unit": "million USD",
                        "period": "2009",
                    },
                    {
                        "id": "op_2008",
                        "chunk_id": "table_row_4",
                        "span_text": "(511)",
                        "context_text": "Interest expense | $ (600) | $ (511)",
                        "normalized_value": -511.0,
                        "unit": "million USD",
                        "period": "2008",
                    },
                ],
                "program": [
                    {
                        "id": "step_1",
                        "operation": "economic_change",
                        "operands": ["op_2009", "op_2008"],
                    }
                ],
                "referent": {
                    "entity": "interest expense",
                    "period": "2009",
                    "unit": "million USD",
                },
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="By how much did interest expense increase from 2008 to 2009?",
        example_id="MSFT_2009",
        retrieved_chunks=sample_msft_expense_evidence,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.verified_answer == 89.0


# =========================================================================
# Scenario 3: Ungrounded Numerical Operand
# =========================================================================
def test_ungrounded_numerical_operand(sample_aapl_evidence, sample_corpus_index):
    # Model fabricates number $999,999 not in chunk text
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales decreased.",
                "value": -100.0,
                "unit": "million USD",
                "citations": ["table_row_0"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 999,999",
                        "context_text": "Net sales | $ 260,174",
                        "normalized_value": 999999.0,
                        "unit": "million USD",
                        "period": "2019",
                    }
                ],
                "program": [{"id": "step_1", "operation": "abs", "operands": ["op_1"]}],
                "referent": {"entity": "net sales", "period": "2019", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "EVIDENCE_GROUNDING_ERROR"
    assert any("not found" in err.lower() for err in result.error_diagnostics)


# =========================================================================
# Scenario 4: Fabricated Chunk ID
# =========================================================================
def test_fabricated_chunk_id(sample_aapl_evidence, sample_corpus_index):
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales decreased.",
                "value": 260174.0,
                "unit": "million USD",
                "citations": ["table_row_999"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_999",  # Non-existent chunk
                        "span_text": "$ 260,174",
                        "context_text": "Net sales | $ 260,174",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2019",
                    }
                ],
                "program": [{"id": "step_1", "operation": "abs", "operands": ["op_1"]}],
                "referent": {"entity": "net sales", "period": "2019", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "EVIDENCE_GROUNDING_ERROR"
    assert any("chunk not found" in err.lower() for err in result.error_diagnostics)


# =========================================================================
# Scenario 5: Incorrect Financial Period
# =========================================================================
def test_incorrect_financial_period(sample_aapl_evidence, sample_corpus_index):
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales in 2012.",
                "value": 260174.0,
                "unit": "million USD",
                "citations": ["table_row_0"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 260,174",
                        "context_text": "Net sales | $ 260,174 | $ 265,595",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2012",  # Evidence only mentions 2019 and 2018
                    }
                ],
                "program": [{"id": "step_1", "operation": "abs", "operands": ["op_1"]}],
                "referent": {"entity": "net sales", "period": "2012", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was net sales in 2012?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "PERIOD_VALIDATION_ERROR"
    assert any("period" in err.lower() for err in result.error_diagnostics)


# =========================================================================
# Scenario 6: Incorrect Financial Unit
# =========================================================================
def test_incorrect_financial_unit(sample_aapl_evidence, sample_corpus_index):
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales in shares.",
                "value": 260174.0,
                "unit": "shares",
                "citations": ["table_row_0"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 260,174",
                        "context_text": "Net sales | $ 260,174 | $ 265,595",
                        "normalized_value": 260174.0,
                        "unit": "shares",  # Table specifies monetary dollars, not shares
                        "period": "2019",
                    }
                ],
                "program": [{"id": "step_1", "operation": "abs", "operands": ["op_1"]}],
                "referent": {"entity": "net sales", "period": "2019", "unit": "shares"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "UNIT_VALIDATION_ERROR"
    assert any("unit" in err.lower() for err in result.error_diagnostics)


# =========================================================================
# Scenario 7: Incorrect Arithmetic (Division by zero / bad calculation)
# =========================================================================
def test_incorrect_arithmetic_calculation(sample_aapl_evidence, sample_corpus_index):
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Calculation error.",
                "value": 1.0,
                "unit": "million USD",
                "citations": ["table_row_0"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 260,174",
                        "context_text": "Net sales | $ 260,174",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2019",
                    }
                ],
                "program": [
                    {"id": "step_zero", "operation": "subtract", "operands": ["op_1", "op_1"]},
                    {"id": "step_div", "operation": "divide", "operands": ["op_1", "step_zero"]},
                ],
                "referent": {"entity": "net sales", "period": "2019", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="Calculate division",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "ARITHMETIC_ERROR"
    assert any("division by zero" in err.lower() for err in result.error_diagnostics)


# =========================================================================
# Scenario 8: Contradictory Narrative Direction
# =========================================================================
def test_contradictory_narrative_direction(sample_msft_expense_evidence, sample_corpus_index):
    # Narrative says "increased", but model performs naive subtraction yielding negative change (-89)
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Interest expense decreased by 89 million USD.",
                "value": -89.0,
                "unit": "million USD",
                "citations": ["table_row_4", "text_1"],
                "operands": [
                    {
                        "id": "op_2009",
                        "chunk_id": "table_row_4",
                        "span_text": "(600)",
                        "context_text": "Interest expense | $ (600) | $ (511)",
                        "normalized_value": -600.0,
                        "unit": "million USD",
                        "period": "2009",
                    },
                    {
                        "id": "op_2008",
                        "chunk_id": "table_row_4",
                        "span_text": "(511)",
                        "context_text": "Interest expense | $ (600) | $ (511)",
                        "normalized_value": -511.0,
                        "unit": "million USD",
                        "period": "2008",
                    },
                ],
                "program": [
                    {
                        "id": "step_1",
                        "operation": "subtract",  # Naive signed subtraction: -600 - (-511) = -89 (decrease)
                        "operands": ["op_2009", "op_2008"],
                    }
                ],
                "referent": {
                    "entity": "interest expense",
                    "period": "2009",
                    "unit": "million USD",
                },
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was the change in interest expense?",
        example_id="MSFT_2009",
        retrieved_chunks=sample_msft_expense_evidence,
    )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "DIRECTIONAL_CONTRADICTION"
    assert any("contradiction" in err.lower() for err in result.error_diagnostics)


# =========================================================================
# Scenario 9: Invalid JSON
# =========================================================================
def test_invalid_json_generation(sample_aapl_evidence, sample_corpus_index):
    mock_llm = MockLLMClient(default_response='{"claims": [ unclosed syntax ...')
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "PARSING_ERROR"
    assert result.detailed_status == "JSON_DECODE_ERROR"
    assert len(result.error_diagnostics) > 0


# =========================================================================
# Scenario 10: Missing Required Claim Fields
# =========================================================================
def test_missing_required_claim_fields(sample_aapl_evidence, sample_corpus_index):
    # Valid JSON, but missing "final_claim_id"
    bad_schema = {"claims": []}
    mock_llm = MockLLMClient(default_response=json.dumps(bad_schema))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "PARSING_ERROR"
    assert result.detailed_status == "SCHEMA_VALIDATION_ERROR"
    assert any("final_claim_id" in err for err in result.error_diagnostics)


# =========================================================================
# Scenario 11: Explicit Insufficient-Evidence Response (Abstention)
# =========================================================================
def test_explicit_insufficient_evidence_response(sample_aapl_evidence, sample_corpus_index):
    abstention_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "The provided evidence does not contain capital expenditures.",
                "value": None,
                "unit": "million USD",
                "citations": [],
                "operands": [],
                "program": [],
                "referent": {
                    "entity": "capital expenditure",
                    "period": "2019",
                    "unit": "million USD",
                },
                "status": "insufficient_evidence",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(abstention_dict))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was capital expenditure in 2019?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.detailed_status == "MODEL_DECLARED_ABSTENTION"
    assert result.verified_answer is None


# =========================================================================
# Scenario 12: Empty Retrieval Results
# =========================================================================
def test_empty_retrieval_results():
    mock_llm = MockLLMClient()
    generator = ClaimGenerator(llm_client=mock_llm, corpus_chunks=[])

    result = generator.generate_and_verify(
        question="What was revenue?",
        retrieved_chunks=[],
    )

    assert result.is_verified is False
    assert result.status == "GENERATION_ERROR"
    assert result.detailed_status == "EMPTY_RETRIEVAL_RESULTS"
    assert any("No evidence chunks retrieved" in err for err in result.error_diagnostics)


# =========================================================================
# Scenario 13: Generation Timeout
# =========================================================================
def test_generation_timeout(sample_aapl_evidence, sample_corpus_index):
    mock_llm = MockLLMClient(simulate_timeout=True)
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=sample_aapl_evidence,
    )

    assert result.is_verified is False
    assert result.status == "GENERATION_ERROR"
    assert result.detailed_status == "GENERATION_TIMEOUT"
    assert any("timed out" in err.lower() or "timeout" in err.lower() for err in result.error_diagnostics)


# =========================================================================
# Scenario 14: Verification Exception
# =========================================================================
def test_verification_exception_handling(sample_aapl_evidence, sample_corpus_index):
    valid_claim = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales.",
                "value": 260174.0,
                "unit": "million USD",
                "citations": ["table_row_0"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 260,174",
                        "context_text": "$ 260,174",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2019",
                    }
                ],
                "program": [{"id": "s1", "operation": "abs", "operands": ["op_1"]}],
                "referent": {"entity": "sales", "period": "2019", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }
    mock_llm = MockLLMClient(default_response=json.dumps(valid_claim))
    generator = ClaimGenerator(llm_client=mock_llm, corpus_index=sample_corpus_index)

    with patch(
        "src.generation.claim_generator.run_verification_pipeline",
        side_effect=RuntimeError("Simulated unexpected verifier fault"),
    ):
        result = generator.generate_and_verify(
            question="What was net sales?",
            example_id="AAPL_2019",
            retrieved_chunks=sample_aapl_evidence,
        )

    assert result.is_verified is False
    assert result.status == "VERIFICATION_FAILED"
    assert result.detailed_status == "VERIFICATION_EXCEPTION"
    assert any("Simulated unexpected verifier fault" in err for err in result.error_diagnostics)


# =========================================================================
# Scenario 15: Parent-Expanded Context Grounding
# =========================================================================
def test_parent_expanded_context_grounding():
    # Retrieved chunk is ONLY the table row
    child_chunk = {
        "example_id": "DOC_1",
        "chunk_id": "table_row_1",
        "source": "table",
        "text": "millions of USD: Gross Profit | 2020: $ 800",
    }
    # Parent context includes the narrative paragraph with additional facts
    parent_corpus_index = {
        "DOC_1": [
            child_chunk,
            {
                "example_id": "DOC_1",
                "chunk_id": "text_0",
                "source": "pre_text",
                "text": "All monetary amounts in millions of USD. In 2020, operating costs were 300.",
            },
        ]
    }

    # Model cites text_0 (from parent expansion, not child chunk) and table_row_1
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net income was 500 million USD in 2020.",
                "value": 500.0,
                "unit": "million USD",
                "citations": ["table_row_1", "text_0"],
                "operands": [
                    {
                        "id": "op_gp",
                        "chunk_id": "table_row_1",
                        "span_text": "$ 800",
                        "context_text": "Gross Profit | 2020: $ 800",
                        "normalized_value": 800.0,
                        "unit": "million USD",
                        "period": "2020",
                    },
                    {
                        "id": "op_cost",
                        "chunk_id": "text_0",  # Introduced via parent expansion!
                        "span_text": "300",
                        "context_text": "operating costs were 300.",
                        "normalized_value": 300.0,
                        "unit": "million USD",
                        "period": "2020",
                    },
                ],
                "program": [{"id": "step_1", "operation": "subtract", "operands": ["op_gp", "op_cost"]}],
                "referent": {"entity": "net income", "period": "2020", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=parent_corpus_index,
        context_mode="parent_expanded",
    )

    result = generator.generate_and_verify(
        question="What was net income in 2020?",
        example_id="DOC_1",
        retrieved_chunks=[child_chunk],
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.verified_answer == 500.0


# =========================================================================
# Scenario 16: Deduplicated Parent Context Grounding
# =========================================================================
def test_deduplicated_parent_context_grounding(sample_aapl_evidence, sample_corpus_index):
    # Pass duplicate child chunks from AAPL_2019
    duplicate_chunks = [sample_aapl_evidence[0], sample_aapl_evidence[0]]

    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales.",
                "value": 260174.0,
                "unit": "million USD",
                "citations": ["table_row_0"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 260,174",
                        "context_text": "$ 260,174",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2019",
                    }
                ],
                "program": [{"id": "s1", "operation": "abs", "operands": ["op_1"]}],
                "referent": {"entity": "sales", "period": "2019", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    generator = ClaimGenerator(
        llm_client=mock_llm,
        corpus_index=sample_corpus_index,
        context_mode="deduplicated_parent",
    )

    result = generator.generate_and_verify(
        question="What was net sales?",
        example_id="AAPL_2019",
        retrieved_chunks=duplicate_chunks,
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.context_mode == "deduplicated_parent"


# =========================================================================
# Scenario 17: Weighted RRF Retrieval Integration
# =========================================================================
def test_weighted_rrf_retrieval_integration(sample_aapl_evidence, sample_corpus_index):
    # Test the weighted RRF mathematical fusion directly
    dense_uids = [("AAPL_2019#table_row_0", 0.95), ("AAPL_2019#text_0", 0.80)]
    bm25_uids = [("AAPL_2019#text_0", 12.5), ("AAPL_2019#table_row_0", 5.2)]

    fused = weighted_reciprocal_rank_fusion(
        bm25_results=bm25_uids,
        dense_results=dense_uids,
        k=60,
        w_dense=0.75,
        w_bm25=0.25,
    )

    assert len(fused) == 2
    # Verify rank scores are floats and ordered
    assert fused[0][1] >= fused[1][1]

    # Now verify end-to-end integration via ClaimGenerator with mock Dense/BM25
    claim_dict = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales.",
                "value": 260174.0,
                "unit": "million USD",
                "citations": ["table_row_0"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "table_row_0",
                        "span_text": "$ 260,174",
                        "context_text": "$ 260,174",
                        "normalized_value": 260174.0,
                        "unit": "million USD",
                        "period": "2019",
                    }
                ],
                "program": [{"id": "s1", "operation": "abs", "operands": ["op_1"]}],
                "referent": {"entity": "sales", "period": "2019", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }

    mock_llm = MockLLMClient(default_response=json.dumps(claim_dict))
    mock_dense_model = MagicMock()
    mock_dense_model.encode.return_value = [[0.0] * 384]
    mock_embeddings = [[0.0] * 384, [0.0] * 384]
    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [5.0, 10.0]

    generator = ClaimGenerator(
        llm_client=mock_llm,
        retrieval_config="weighted_rrf_parent_child",
        corpus_chunks=sample_aapl_evidence,
        corpus_index=sample_corpus_index,
        dense_model=mock_dense_model,
        dense_embeddings=mock_embeddings,
        bm25_index=mock_bm25,
    )

    result = generator.generate_and_verify(
        question="What was net sales in 2019?",
        example_id="AAPL_2019",
    )

    assert result.is_verified is True
    assert result.status == "VERIFIED"
    assert result.retrieval_config == "weighted_rrf_parent_child"
    assert result.verified_answer == 260174.0
