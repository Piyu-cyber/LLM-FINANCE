"""output_parser.py — Strict JSON and schema parser for LLM-generated financial claims.

Parses raw LLM completion text into validated CLAIM_SCHEMA structures with:
- Code-fence and prose stripping.
- Strict JSON syntax error diagnostics.
- CLAIM_SCHEMA schema validation via jsonschema.
- Internal referential integrity validation (operand ID existence, program DAG order,
  chunk_id non-emptiness, valid financial operations).
- Explicit abstention detection (status: "insufficient_evidence").
- Guarantee against silent coercion of malformed data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Sequence

import jsonschema
from jsonschema import ValidationError

try:
    from src.verification.claim_schema import CLAIM_SCHEMA
except ImportError:
    try:
        from claim_schema import CLAIM_SCHEMA
    except ImportError:
        import sys
        from pathlib import Path

        v_dir = Path(__file__).resolve().parent.parent / "verification"
        if str(v_dir) not in sys.path:
            sys.path.insert(0, str(v_dir))
        from claim_schema import CLAIM_SCHEMA


ALLOWED_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "economic_change",
    "magnitude_change",
    "abs_change",
    "percentage_change",
    "abs",
}


@dataclass
class ParseResult:
    """Structured result of output parsing and validation."""

    is_valid: bool
    status: str
    claim: Optional[dict[str, Any]] = None
    errors: list[str] = field(default_factory=list)
    raw_text: str = ""
    extracted_json_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert parse result to a serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "errors": self.errors,
            "has_claim": self.claim is not None,
            "extracted_length": len(self.extracted_json_text),
        }


def extract_json_string(raw_text: str) -> Optional[str]:
    """Extract the candidate JSON substring from raw model completion text.

    Handles:
    - Markdown fenced blocks (```json ... ``` or ``` ... ```)
    - Preamble and postscript conversational text
    - Outer balanced curly braces or square brackets
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # 1. Check for markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = fence_pattern.findall(text)
    for match in matches:
        match_str = match.strip()
        if (match_str.startswith("{") and match_str.endswith("}")) or (
            match_str.startswith("[") and match_str.endswith("]")
        ):
            return match_str

    # 2. Check outermost container: compare '{' vs '['
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    # If bracket starts before brace (e.g. top-level array)
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        if last_bracket != -1 and last_bracket > first_bracket:
            return text[first_bracket : last_bracket + 1].strip()

    # Otherwise if brace exists
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    # Fallback to bracket if brace wasn't valid
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket : last_bracket + 1].strip()

    return None


def is_numeric_value(val: Any) -> bool:
    """Check if a value is a valid numeric representation."""
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, (int, float, Decimal)):
        return True
    if isinstance(val, str):
        try:
            Decimal(val.replace(",", "").strip())
            return True
        except (InvalidOperation, ValueError):
            return False
    return False


class OutputParser:
    """Strict JSON and financial claim validator."""

    def __init__(self, schema: Optional[dict] = None) -> None:
        """Initialize OutputParser with schema."""
        self.schema = schema if schema is not None else CLAIM_SCHEMA

    def parse(self, raw_text: str) -> ParseResult:
        """Parse raw model text into a validated claim dictionary.

        Parameters
        ----------
        raw_text : str
            Raw text response from LLM generation.

        Returns
        -------
        ParseResult
            Detailed parsing and validation outcome.
        """
        extracted = extract_json_string(raw_text)

        if not extracted:
            return ParseResult(
                is_valid=False,
                status="JSON_DECODE_ERROR",
                claim=None,
                errors=["No JSON object found in model output."],
                raw_text=raw_text,
                extracted_json_text="",
            )

        # 1. JSON Syntax Validation
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            return ParseResult(
                is_valid=False,
                status="JSON_DECODE_ERROR",
                claim=None,
                errors=[f"JSON syntax error: {exc.msg} at line {exc.lineno} col {exc.colno}"],
                raw_text=raw_text,
                extracted_json_text=extracted,
            )

        if not isinstance(parsed, dict):
            return ParseResult(
                is_valid=False,
                status="SCHEMA_VALIDATION_ERROR",
                claim=None,
                errors=[f"Top-level JSON structure must be an object (dict), got {type(parsed).__name__}."],
                raw_text=raw_text,
                extracted_json_text=extracted,
            )

        # 2. JSON Schema Validation against CLAIM_SCHEMA
        try:
            jsonschema.validate(instance=parsed, schema=self.schema)
        except ValidationError as val_err:
            path_str = " -> ".join(str(p) for p in val_err.path) if val_err.path else "root"
            return ParseResult(
                is_valid=False,
                status="SCHEMA_VALIDATION_ERROR",
                claim=None,
                errors=[f"Schema validation error at '{path_str}': {val_err.message}"],
                raw_text=raw_text,
                extracted_json_text=extracted,
            )

        # 3. Referential and Semantic Integrity Validation
        integrity_errors: list[str] = []

        final_claim_id = parsed.get("final_claim_id", "")
        claims_list = parsed.get("claims", [])

        # Match final_claim_id
        target_claim = None
        for c in claims_list:
            if c.get("id") == final_claim_id:
                target_claim = c
                break

        if target_claim is None:
            integrity_errors.append(
                f"final_claim_id '{final_claim_id}' does not match any claim in the claims array."
            )
            return ParseResult(
                is_valid=False,
                status="REFERENTIAL_INTEGRITY_ERROR",
                claim=None,
                errors=integrity_errors,
                raw_text=raw_text,
                extracted_json_text=extracted,
            )

        status_val = target_claim.get("status")

        # Handle abstention case
        if status_val == "insufficient_evidence":
            return ParseResult(
                is_valid=True,
                status="INSUFFICIENT_EVIDENCE",
                claim=parsed,
                errors=[],
                raw_text=raw_text,
                extracted_json_text=extracted,
            )

        # If status is "answered", enforce complete grounded program requirements
        if status_val == "answered":
            claim_val = target_claim.get("value")
            if claim_val is not None and not is_numeric_value(claim_val):
                integrity_errors.append(
                    f"Claim value '{claim_val}' must be numeric or null, got {type(claim_val).__name__}."
                )

            operands = target_claim.get("operands", [])
            if not operands:
                integrity_errors.append(
                    "Claim status is 'answered' but operands list is empty."
                )

            defined_operand_ids: set[str] = set()

            for idx, op in enumerate(operands):
                op_id = op.get("id")
                if not op_id or not isinstance(op_id, str):
                    integrity_errors.append(f"Operand at index {idx} has invalid or missing id.")
                elif op_id in defined_operand_ids:
                    integrity_errors.append(f"Duplicate operand id '{op_id}' in operands list.")
                else:
                    defined_operand_ids.add(op_id)

                chunk_id = op.get("chunk_id")
                if not chunk_id or not isinstance(chunk_id, str) or not chunk_id.strip():
                    integrity_errors.append(
                        f"Operand '{op_id or idx}' must specify a non-empty chunk_id."
                    )

                span_text = op.get("span_text")
                if not span_text or not isinstance(span_text, str) or not span_text.strip():
                    integrity_errors.append(
                        f"Operand '{op_id or idx}' must specify a non-empty span_text."
                    )

                norm_val = op.get("normalized_value")
                if not is_numeric_value(norm_val):
                    integrity_errors.append(
                        f"Operand '{op_id or idx}' normalized_value '{norm_val}' is not a valid numeric value."
                    )

                unit = op.get("unit")
                if not unit or not isinstance(unit, str) or not unit.strip():
                    integrity_errors.append(
                        f"Operand '{op_id or idx}' must specify a non-empty unit string."
                    )

                period = op.get("period")
                if not period or not isinstance(period, str) or not period.strip():
                    integrity_errors.append(
                        f"Operand '{op_id or idx}' must specify a non-empty period string."
                    )

            # Program validation
            program = target_claim.get("program", [])
            if not program:
                integrity_errors.append(
                    "Claim status is 'answered' but program calculation steps are empty."
                )

            for step_idx, step in enumerate(program):
                if not isinstance(step, dict):
                    integrity_errors.append(
                        f"Program step {step_idx} must be an object (dict)."
                    )
                    continue

                step_id = step.get("id", f"step_{step_idx + 1}")
                op_name = step.get("operation")
                if not op_name or not isinstance(op_name, str):
                    integrity_errors.append(
                        f"Program step '{step_id}' is missing a valid operation string."
                    )
                else:
                    op_clean = op_name.lower().strip()
                    if op_clean not in ALLOWED_OPERATIONS:
                        integrity_errors.append(
                            f"Program step '{step_id}' specifies unsupported operation '{op_name}'. "
                            f"Allowed operations are: {sorted(ALLOWED_OPERATIONS)}"
                        )

                step_operands = step.get("operands")
                if not isinstance(step_operands, list) or len(step_operands) == 0:
                    integrity_errors.append(
                        f"Program step '{step_id}' must specify a non-empty operands list."
                    )
                else:
                    for ref_id in step_operands:
                        if not isinstance(ref_id, str):
                            integrity_errors.append(
                                f"Program step '{step_id}' operand reference must be a string, got {ref_id}."
                            )
                        elif ref_id not in defined_operand_ids:
                            integrity_errors.append(
                                f"Program step '{step_id}' references undefined operand '{ref_id}'. "
                                f"Must refer to an operand in operands list or a prior step id."
                            )

                # Add this step's output id to defined operands for subsequent steps
                if step_id:
                    defined_operand_ids.add(step_id)

            # Referent validation
            referent = target_claim.get("referent", {})
            if not referent.get("entity") or not str(referent.get("entity")).strip():
                integrity_errors.append("Claim referent entity must be a non-empty string.")

        if integrity_errors:
            return ParseResult(
                is_valid=False,
                status="REFERENTIAL_INTEGRITY_ERROR",
                claim=None,
                errors=integrity_errors,
                raw_text=raw_text,
                extracted_json_text=extracted,
            )

        return ParseResult(
            is_valid=True,
            status="SUCCESS",
            claim=parsed,
            errors=[],
            raw_text=raw_text,
            extracted_json_text=extracted,
        )
