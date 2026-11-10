"""test_output_parser.py — Comprehensive tests for strict OutputParser."""

import json
import pytest

try:
    from src.generation.output_parser import (
        OutputParser,
        ParseResult,
        extract_json_string,
        is_numeric_value,
    )
except ImportError:
    from output_parser import (
        OutputParser,
        ParseResult,
        extract_json_string,
        is_numeric_value,
    )


@pytest.fixture
def parser():
    return OutputParser()


@pytest.fixture
def valid_claim_dict():
    return {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Net sales decreased by 5421 million USD from 2018 to 2019.",
                "value": -5421.0,
                "unit": "million USD",
                "citations": ["table_row_0"],
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


def test_is_numeric_value():
    assert is_numeric_value(100)
    assert is_numeric_value(100.5)
    assert is_numeric_value("100.5")
    assert is_numeric_value("1,234.56")
    assert is_numeric_value("-511")
    assert not is_numeric_value(None)
    assert not is_numeric_value(True)
    assert not is_numeric_value(False)
    assert not is_numeric_value("abc")
    assert not is_numeric_value([])


def test_extract_json_string_clean():
    raw = '{"claims": [], "final_claim_id": "c1"}'
    assert extract_json_string(raw) == raw


def test_extract_json_string_markdown_fence():
    raw = "```json\n{\"claims\": [], \"final_claim_id\": \"c1\"}\n```"
    assert extract_json_string(raw) == '{"claims": [], "final_claim_id": "c1"}'


def test_extract_json_string_prose_surrounding():
    raw = (
        "Here is the requested output:\n"
        '{"claims": [], "final_claim_id": "c1"}\n'
        "Hope this helps!"
    )
    assert extract_json_string(raw) == '{"claims": [], "final_claim_id": "c1"}'


def test_extract_json_string_empty():
    assert extract_json_string("") is None
    assert extract_json_string("   ") is None
    assert extract_json_string("No JSON here!") is None


def test_parse_valid_claim_success(parser, valid_claim_dict):
    raw = json.dumps(valid_claim_dict)
    res = parser.parse(raw)

    assert isinstance(res, ParseResult)
    assert res.is_valid is True
    assert res.status == "SUCCESS"
    assert res.errors == []
    assert res.claim == valid_claim_dict


def test_parse_markdown_fence_and_prose(parser, valid_claim_dict):
    raw = (
        "Based on the SEC 10-K filing:\n"
        "```json\n"
        f"{json.dumps(valid_claim_dict, indent=2)}\n"
        "```\n"
        "This completes the calculation."
    )
    res = parser.parse(raw)
    assert res.is_valid is True
    assert res.status == "SUCCESS"
    assert res.claim["final_claim_id"] == "c1"


def test_parse_insufficient_evidence_status(parser):
    abstention = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "The provided evidence does not report revenue for 2010.",
                "value": None,
                "unit": "million USD",
                "citations": [],
                "operands": [],
                "program": [],
                "referent": {
                    "entity": "revenue",
                    "period": "2010",
                    "unit": "million USD",
                },
                "status": "insufficient_evidence",
            }
        ],
        "final_claim_id": "c1",
    }
    res = parser.parse(json.dumps(abstention))
    assert res.is_valid is True
    assert res.status == "INSUFFICIENT_EVIDENCE"
    assert res.errors == []


def test_parse_syntax_error(parser):
    # Trailing comma before closing brace
    raw = '{"claims": [], "final_claim_id": "c1",}'
    res = parser.parse(raw)
    assert res.is_valid is False
    assert res.status == "JSON_DECODE_ERROR"
    assert any("JSON syntax error" in err for err in res.errors)


def test_parse_no_json_object(parser):
    raw = "The company reported 500 million in revenue but no JSON was provided."
    res = parser.parse(raw)
    assert res.is_valid is False
    assert res.status == "JSON_DECODE_ERROR"
    assert "No JSON object found" in res.errors[0]


def test_parse_top_level_list_fails_schema(parser):
    raw = json.dumps([{"id": "c1"}])
    res = parser.parse(raw)
    assert res.is_valid is False
    assert res.status == "SCHEMA_VALIDATION_ERROR"
    assert any("must be an object" in err for err in res.errors)


def test_parse_missing_required_schema_field(parser):
    # Missing final_claim_id
    bad_dict = {"claims": []}
    res = parser.parse(json.dumps(bad_dict))
    assert res.is_valid is False
    assert res.status == "SCHEMA_VALIDATION_ERROR"
    assert any("final_claim_id" in err for err in res.errors)


def test_parse_missing_operand_required_field(parser, valid_claim_dict):
    # Delete 'unit' from an operand
    del valid_claim_dict["claims"][0]["operands"][0]["unit"]
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "SCHEMA_VALIDATION_ERROR"
    assert any("unit" in err for err in res.errors)


def test_parse_unmatched_final_claim_id(parser, valid_claim_dict):
    valid_claim_dict["final_claim_id"] = "non_existent_claim_id"
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("does not match any claim" in err for err in res.errors)


def test_parse_empty_operands_when_answered(parser, valid_claim_dict):
    valid_claim_dict["claims"][0]["operands"] = []
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("operands list is empty" in err for err in res.errors)


def test_parse_empty_program_when_answered(parser, valid_claim_dict):
    valid_claim_dict["claims"][0]["program"] = []
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("program calculation steps are empty" in err for err in res.errors)


def test_parse_empty_chunk_id_in_operand(parser, valid_claim_dict):
    valid_claim_dict["claims"][0]["operands"][0]["chunk_id"] = "   "
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("non-empty chunk_id" in err for err in res.errors)


def test_parse_empty_span_text_in_operand(parser, valid_claim_dict):
    valid_claim_dict["claims"][0]["operands"][0]["span_text"] = ""
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("non-empty span_text" in err for err in res.errors)


def test_parse_duplicate_operand_id(parser, valid_claim_dict):
    valid_claim_dict["claims"][0]["operands"][1]["id"] = "op_2019"  # Same as first
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("Duplicate operand id" in err for err in res.errors)


def test_parse_unsupported_operation(parser, valid_claim_dict):
    valid_claim_dict["claims"][0]["program"][0]["operation"] = "integrate_polynomial"
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("unsupported operation" in err for err in res.errors)


def test_parse_program_undefined_operand_reference(parser, valid_claim_dict):
    valid_claim_dict["claims"][0]["program"][0]["operands"] = ["op_2019", "ghost_operand_id"]
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("references undefined operand 'ghost_operand_id'" in err for err in res.errors)


def test_parse_program_forward_reference_fails(parser, valid_claim_dict):
    # Step 1 references step 2 output
    valid_claim_dict["claims"][0]["program"] = [
        {"id": "step_1", "operation": "add", "operands": ["op_2019", "step_2"]},
        {"id": "step_2", "operation": "subtract", "operands": ["op_2018", "op_2019"]},
    ]
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("references undefined operand 'step_2'" in err for err in res.errors)


def test_parse_chained_program_steps_succeeds(parser, valid_claim_dict):
    # Step 1 calculates difference, Step 2 calculates percentage
    valid_claim_dict["claims"][0]["program"] = [
        {"id": "step_1", "operation": "subtract", "operands": ["op_2019", "op_2018"]},
        {"id": "step_2", "operation": "divide", "operands": ["step_1", "op_2018"]},
    ]
    res = parser.parse(json.dumps(valid_claim_dict))
    assert res.is_valid is True
    assert res.status == "SUCCESS"
    assert len(res.claim["claims"][0]["program"]) == 2


def test_no_silent_coercion(parser):
    # Verify that malformed numbers are not silently converted into 0.0
    bad_claim = {
        "claims": [
            {
                "id": "c1",
                "claim_text": "Test claim",
                "value": "not_a_number",
                "unit": "USD",
                "citations": ["chunk_1"],
                "operands": [
                    {
                        "id": "op_1",
                        "chunk_id": "chunk_1",
                        "span_text": "$100",
                        "context_text": "Sales were $100",
                        "normalized_value": "invalid_number_str",
                        "unit": "USD",
                        "period": "2020",
                    }
                ],
                "program": [{"id": "s1", "operation": "add", "operands": ["op_1", "op_1"]}],
                "referent": {"entity": "sales", "period": "2020", "unit": "USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "c1",
    }
    res = parser.parse(json.dumps(bad_claim))
    assert res.is_valid is False
    assert res.status == "REFERENTIAL_INTEGRITY_ERROR"
    assert any("not a valid numeric value" in err for err in res.errors)
