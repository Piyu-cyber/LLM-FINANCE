import pytest
from pipeline import run_verification_pipeline


EXAMPLE_ID = "UNP/2009/page_34.pdf-1"

NARRATIVE_CONTEXT = (
    "interest expense increased in 2009 versus 2008 due primarily to "
    "higher weighted-average debt levels."
)

SAMPLE_CHUNKS = [
    {
        "example_id": EXAMPLE_ID,
        "chunk_id": "table_1",
        "text": (
            "millions of dollars: interest expense | "
            "2009: -600 ( 600 ) | "
            "2008: -511 ( 511 ) | "
            "2007: -482 ( 482 ) | "
            "% change 2009 v 2008: 17 | % change 2008 v 2007: 6"
        ),
    },
    {
        "example_id": EXAMPLE_ID,
        "chunk_id": "post_text_0",
        "text": NARRATIVE_CONTEXT,
    },
]


def build_base_claim(operation="subtract", value=-89, direction=None):
    return {
        "claims": [
            {
                "id": "claim_1",
                "claim_text": (
                    f"The change in interest expense from 2008 to 2009 was {value} million USD."
                ),
                "value": value,
                "unit": "million USD",
                "direction": direction,
                "citations": ["table_1", "post_text_0"],
                "operands": [
                    {
                        "id": "op_2009",
                        "chunk_id": "table_1",
                        "span_text": "2009: -600 ( 600 )",
                        "context_text": "millions of dollars: interest expense | 2009: -600 ( 600 )",
                        "normalized_value": -600,
                        "unit": "million USD",
                        "period": "2009",
                    },
                    {
                        "id": "op_2008",
                        "chunk_id": "table_1",
                        "span_text": "2008: -511 ( 511 )",
                        "context_text": "millions of dollars: interest expense | 2008: -511 ( 511 )",
                        "normalized_value": -511,
                        "unit": "million USD",
                        "period": "2008",
                    },
                ],
                "program": [
                    {
                        "id": "step_1",
                        "operation": operation,
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
        "final_claim_id": "claim_1",
    }


def test_pipeline_detects_directional_contradiction():
    # Demonstrating the core flaw:
    # Operands are grounded (-600, -511), arithmetic is deterministic (-600 - (-511) = -89),
    # BUT narrative text says 'interest expense increased'!
    # The pipeline must catch this and reject it.
    naive_claim = build_base_claim(operation="subtract", value=-89)

    result = run_verification_pipeline(
        naive_claim,
        EXAMPLE_ID,
        chunks=SAMPLE_CHUNKS,
        context_passages=[NARRATIVE_CONTEXT],
        resolve_economic_sign=False,
    )

    assert result["valid"] is False
    assert result["status"] == "DIRECTIONAL_CONTRADICTION"
    assert "Directional contradiction" in result["reason"]
    assert result["direction_check"]["narrative_direction"] == "increase"
    assert result["direction_check"]["calculated_direction"] == "decrease"


def test_pipeline_resolves_economic_magnitude():
    # When resolve_economic_sign=True, the pipeline detects the accounting polarity
    # and resolves expense subtraction to economic magnitude: |-600| - |-511| = +89 million USD
    naive_claim = build_base_claim(operation="subtract", value=-89)

    result = run_verification_pipeline(
        naive_claim,
        EXAMPLE_ID,
        chunks=SAMPLE_CHUNKS,
        context_passages=[NARRATIVE_CONTEXT],
        resolve_economic_sign=True,
    )

    assert result["valid"] is True
    assert result["status"] == "VERIFIED_RESULT_ECONOMIC_RESOLVED"
    assert result["calculation"]["result"] == 89.0
    assert result["calculation"]["direction"] == "increase"


def test_pipeline_with_economic_change_operation():
    # When the claim explicitly uses the accounting-aware 'economic_change' operation
    economic_claim = build_base_claim(operation="economic_change", value=89, direction="increase")

    result = run_verification_pipeline(
        economic_claim,
        EXAMPLE_ID,
        chunks=SAMPLE_CHUNKS,
        context_passages=[NARRATIVE_CONTEXT],
    )

    assert result["valid"] is True
    assert result["status"] == "VERIFIED_RESULT"
    assert result["calculation"]["result"] == 89.0
    assert result["calculation"]["direction"] == "increase"


if __name__ == "__main__":
    pytest.main([__file__])
