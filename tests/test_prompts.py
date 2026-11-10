"""test_prompts.py — Comprehensive tests for financial prompt construction and schema validation."""

import jsonschema
import pytest

try:
    from src.generation.prompts import (
        FEW_SHOT_EXAMPLES,
        SYSTEM_PROMPT,
        build_generation_prompt,
        build_repair_prompt,
        build_system_prompt,
        build_user_prompt,
    )
    from src.verification.claim_schema import CLAIM_SCHEMA
except ImportError:
    from prompts import (
        FEW_SHOT_EXAMPLES,
        SYSTEM_PROMPT,
        build_generation_prompt,
        build_repair_prompt,
        build_system_prompt,
        build_user_prompt,
    )
    from claim_schema import CLAIM_SCHEMA


def test_system_prompt_critical_invariants():
    sys_prompt = build_system_prompt()
    assert "CRITICAL INVARIANTS" in sys_prompt
    assert "Grounding" in sys_prompt
    assert "Exact Span Citation" in sys_prompt
    assert "economic_change" in sys_prompt
    assert "insufficient_evidence" in sys_prompt
    assert "referential integrity" in sys_prompt.lower()


def test_few_shot_examples_comply_with_claim_schema():
    assert len(FEW_SHOT_EXAMPLES) >= 3
    for idx, ex in enumerate(FEW_SHOT_EXAMPLES):
        claim_output = ex["output"]
        # Validate against authoritative CLAIM_SCHEMA
        jsonschema.validate(instance=claim_output, schema=CLAIM_SCHEMA)
        assert "claims" in claim_output
        assert "final_claim_id" in claim_output
        final_id = claim_output["final_claim_id"]
        matching = [c for c in claim_output["claims"] if c["id"] == final_id]
        assert len(matching) == 1, f"Example {idx} final_claim_id not matched"


def test_build_user_prompt():
    prompt = build_user_prompt(
        question="What was net income in 2019?",
        context_text="[Evidence Chunk: c1] Net Income | $ 500",
        example_id="TEST_2019",
        include_few_shot=False,
    )
    assert "### RETRIEVED SEC 10-K EVIDENCE ###" in prompt
    assert "Target Document ID: TEST_2019" in prompt
    assert "[Evidence Chunk: c1] Net Income | $ 500" in prompt
    assert "### FINANCIAL QUESTION ###" in prompt
    assert "What was net income in 2019?" in prompt
    assert "Generate the structured claim JSON:" in prompt


def test_build_generation_prompt():
    prompt = build_generation_prompt(
        question="What was total sales in 2020?",
        context_text="[Evidence Chunk: c1] Sales | $ 100",
        include_few_shot=True,
    )
    assert SYSTEM_PROMPT in prompt
    assert "### EXAMPLES OF VALID CLAIMS AND PROGRAMS ###" in prompt
    assert "What was total sales in 2020?" in prompt


def test_build_repair_prompt():
    errors = [
        "Operand 'op_1' normalized_value '-600.0' not found in chunk 'table_row_1'.",
        "Program step 'step_1' references undefined operand 'op_99'.",
    ]
    repair = build_repair_prompt(
        question="What was the change in interest expense?",
        context_text="[Evidence Chunk: table_row_1] Interest expense | (600)",
        previous_output='{"claims": []}',
        errors=errors,
    )
    assert "### REPAIR REQUEST: VERIFICATION FAILED ###" in repair
    assert "- Operand 'op_1' normalized_value '-600.0' not found" in repair
    assert "- Program step 'step_1' references undefined operand 'op_99'" in repair
    assert "### PREVIOUS GENERATION ###" in repair
    assert '{"claims": []}' in repair
    assert "What was the change in interest expense?" in repair
