import pytest
from validator import (
    load_corpus,
    build_chunk_lookup,
    validate_claim,
    extract_narrative_direction,
    validate_narrative_direction,
)


@pytest.fixture
def sample_lookup():
    chunks = [
        {
            "example_id": "UNP/2009/page_34.pdf-1",
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
            "example_id": "UNP/2009/page_34.pdf-1",
            "chunk_id": "post_text_0",
            "text": (
                "interest expense increased in 2009 versus 2008 due primarily "
                "to higher weighted-average debt levels."
            ),
        },
    ]
    return build_chunk_lookup(chunks)


def test_extract_narrative_direction():
    text = "Interest expense increased in 2009 versus 2008 due to debt levels."
    res = extract_narrative_direction(text, metric="interest expense")
    assert res["direction"] == "increase"
    assert res["keyword"] == "increased"

    dec_text = "Operating costs decreased significantly during the fiscal period."
    dec_res = extract_narrative_direction(dec_text, metric="operating costs")
    assert dec_res["direction"] == "decrease"
    assert dec_res["keyword"] == "decreased"


def test_validate_narrative_direction_contradiction():
    # Naive signed -89 for an expense vs "increased" narrative text
    evidence_text = [
        "interest expense increased in 2009 versus 2008 due to higher debt."
    ]
    res = validate_narrative_direction(
        claimed_value=-89,
        evidence_texts=evidence_text,
        referent={"entity": "interest expense", "period": "2009", "unit": "million USD"},
    )
    assert res["valid"] is False
    assert res["status"] == "DIRECTIONAL_CONTRADICTION"
    assert "suggested_fix" in res


def test_validate_narrative_direction_economic_match():
    # Economic magnitude +89 matches narrative "increased"
    evidence_text = [
        "interest expense increased in 2009 versus 2008 due to higher debt."
    ]
    res = validate_narrative_direction(
        claimed_value=89,
        evidence_texts=evidence_text,
        referent={"entity": "interest expense", "period": "2009", "unit": "million USD"},
    )
    assert res["valid"] is True
    assert res["status"] == "VERIFIED_DIRECTION"
    assert res["narrative_direction"] == "increase"


def test_validate_claim_with_sample_lookup(sample_lookup):
    claim = {
        "claims": [
            {
                "id": "claim_1",
                "claim_text": "The interest expense in 2009 was 600 million USD.",
                "value": 600,
                "unit": "million USD",
                "citations": ["table_1"],
                "operands": [
                    {
                        "id": "operand_1",
                        "chunk_id": "table_1",
                        "span_text": "2009: -600 ( 600 )",
                        "context_text": "millions of dollars: interest expense | 2009: -600 ( 600 )",
                        "normalized_value": 600,
                        "unit": "million USD",
                        "period": "2009",
                    }
                ],
                "program": [],
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

    result = validate_claim(
        claim,
        sample_lookup,
        example_id="UNP/2009/page_34.pdf-1",
        context_texts=["interest expense increased in 2009 versus 2008"],
    )
    assert result["valid"] is True
    assert result["status"] == "VERIFIED_EVIDENCE"


if __name__ == "__main__":
    pytest.main([__file__])
