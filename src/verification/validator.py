import json
import re
from pathlib import Path

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from claim_schema import CLAIM_SCHEMA


CORPUS_PATH = Path(
    "data/finqa/train_chunks.jsonl"
)


def load_corpus():
    """
    Load the FinQA retrieval corpus.

    Returns:
        list[dict]: All corpus chunks.
    """

    chunks = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            chunks.append(
                json.loads(line)
            )

    return chunks


def build_chunk_lookup(chunks):
    """
    Build a lookup table using:

        (example_id, chunk_id)

    Chunk IDs are not necessarily globally unique,
    so example_id is included in the key.
    """

    lookup = {}

    for chunk in chunks:

        key = (
            chunk["example_id"],
            chunk["chunk_id"]
        )

        lookup[key] = chunk

    return lookup


def validate_claim_schema(claim):
    """
    Validate the structure of a generated claim
    against CLAIM_SCHEMA.
    """

    try:

        validate(
            instance=claim,
            schema=CLAIM_SCHEMA
        )

        return True, "Schema valid"

    except ValidationError as e:

        return False, str(e)


def find_chunk(
    lookup,
    example_id,
    chunk_id
):
    """
    Find a specific evidence chunk.
    """

    return lookup.get(
        (
            example_id,
            chunk_id
        )
    )


def normalize_number(value):
    """
    Convert common financial number formats
    into floats.

    Examples:

        "-600"      -> -600.0
        "600"       -> 600.0
        "(600)"     -> -600.0
        "$600"      -> 600.0
        "3.8"       -> 3.8
    """

    if isinstance(
        value,
        (int, float)
    ):
        return float(value)

    if not isinstance(
        value,
        str
    ):
        return None

    value = value.strip()

    if not value:
        return None

    # Parentheses usually represent negative
    # financial values.
    negative = (
        value.startswith("(")
        and value.endswith(")")
    )

    value = (
        value
        .replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    value = value.strip("()")

    try:

        number = float(value)

    except ValueError:

        return None

    if negative:
        number = -number

    return number


def extract_numbers(text):
    """
    Extract numerical values from evidence text.

    Examples:

        "interest expense 2009: -600"

        -> [2009.0, -600.0]
    """

    if not isinstance(
        text,
        str
    ):
        return []

    pattern = r"""
        (?<![\w.])
        \(?\$?
        -?
        \d+(?:,\d{3})*(?:\.\d+)?
        %?
        \)?
        (?![\w.])
    """

    matches = re.findall(
        pattern,
        text,
        flags=re.VERBOSE
    )

    numbers = []

    for match in matches:

        value = normalize_number(
            match
        )

        if value is not None:
            numbers.append(value)

    return numbers


def value_matches_evidence(
    claimed_value,
    evidence_text,
    tolerance=1e-6
):
    """
    Check whether the claimed numerical value
    occurs in the evidence.
    """

    claimed_value = normalize_number(
        claimed_value
    )

    if claimed_value is None:
        return False

    evidence_values = extract_numbers(
        evidence_text
    )

    for value in evidence_values:

        if abs(
            value - claimed_value
        ) <= tolerance:

            return True

    return False


def period_matches_evidence(
    claimed_period,
    evidence_text
):
    """
    Check whether the claimed period/year
    occurs in the evidence.
    """

    if claimed_period is None:
        return False

    period = str(
        claimed_period
    ).strip()

    if not period:
        return False

    return period in evidence_text


def unit_matches_evidence(
    claimed_unit,
    evidence_text
):
    """
    Perform a basic unit consistency check.

    This is intentionally conservative.
    More sophisticated unit normalization
    can be added later.
    """

    if not claimed_unit:
        return False

    unit = claimed_unit.lower().strip()

    text = evidence_text.lower()

    unit_aliases = {

        "million usd": [
        "million",
        "millions",
        "million dollars",
        "millions of dollars",
        "usd",
        "$"
    ],

    "billion usd": [
        "billion",
        "billions",
        "billion dollars",
        "billions of dollars",
        "usd",
        "$"
    ],

    "percent": [
        "%",
        "percent",
        "percentage"
    ],
    }

    aliases = unit_aliases.get(
        unit,
        [unit]
    )

    return any(
        alias in text
        for alias in aliases
    )


def validate_operand(
    operand,
    lookup,
    example_id,
    referent=None
):
    """
    Validate one operand against its cited evidence.

    Checks:

        1. Evidence chunk exists.
        2. Evidence span exists.
        3. Numerical value occurs in evidence.
        4. Operand's claimed period occurs in evidence.
        5. Claimed unit is compatible with evidence.
    """

    chunk_id = operand["chunk_id"]

    chunk = find_chunk(
        lookup,
        example_id,
        chunk_id
    )

    # --------------------------------------------------
    # 1. Check that the cited chunk exists.
    # --------------------------------------------------

    if chunk is None:

        return {
            "valid": False,
            "reason": "Evidence chunk not found"
        }

    chunk_text = chunk.get(
        "text",
        ""
    )

    # --------------------------------------------------
    # 2. Check that the evidence span exists.
    # --------------------------------------------------

    span_text = operand[
        "span_text"
    ].strip()

    if not span_text:

        return {
            "valid": False,
            "reason": "Empty evidence span"
        }

    if span_text.lower() not in chunk_text.lower():

        return {
            "valid": False,
            "reason": (
                "Evidence span not found "
                "inside cited chunk"
            )
        }

    # --------------------------------------------------
    # 3. Check numerical value.
    # --------------------------------------------------

    claimed_value = operand.get(
        "normalized_value"
    )

    if not value_matches_evidence(
        claimed_value,
        span_text
    ):

        return {
            "valid": False,
            "reason": (
                f"Claimed value "
                f"{claimed_value} "
                "was not found in evidence"
            )
        }

    # --------------------------------------------------
    # 4. Check operand-specific period.
    #
    # IMPORTANT:
    # Each operand can belong to a different period.
    #
    # Example:
    #   op_2009 -> 2009
    #   op_2008 -> 2008
    #
    # Do NOT use referent["period"] here.
    # --------------------------------------------------

    operand_period = operand.get(
        "period"
    )

    if operand_period:

        if not period_matches_evidence(
            operand_period,
            span_text
        ):

            return {
                "valid": False,
                "reason": (
                    f"Claimed period "
                    f"{operand_period} "
                    "was not found in evidence"
                )
            }

    # --------------------------------------------------
    # 5. Check unit.
    # --------------------------------------------------

    claimed_unit = operand.get(
        "unit"
    )

    if claimed_unit:

        if not unit_matches_evidence(
            claimed_unit,
            chunk_text
        ):

            return {
                "valid": False,
                "reason": (
                    f"Claimed unit "
                    f"'{claimed_unit}' "
                    "is not supported by evidence"
                )
            }

    # --------------------------------------------------
    # Everything passed.
    # --------------------------------------------------

    return {
        "valid": True,
        "reason": (
            "Evidence span, value, period, "
            "and unit verified"
        )
    }

def validate_claim(
    claim,
    lookup,
    example_id
):
    """
    Validate a complete structured claim.

    Returns a verification result containing:

        - overall validity
        - verification status
        - explanation
        - individual operand results
    """

    # --------------------------------------------------
    # 1. Validate JSON structure.
    # --------------------------------------------------

    schema_valid, schema_message = (
        validate_claim_schema(
            claim
        )
    )

    if not schema_valid:

        return {
            "valid": False,
            "status": "INVALID_SCHEMA",
            "reason": schema_message,
            "operands": []
        }

    # --------------------------------------------------
    # 2. Validate every claim.
    # --------------------------------------------------

    operand_results = []

    all_valid = True

    for current_claim in claim["claims"]:

        referent = current_claim.get(
            "referent"
        )

        for operand in current_claim[
            "operands"
        ]:

            result = validate_operand(
                operand,
                lookup,
                example_id,
                referent
            )

            operand_results.append(
                result
            )

            if not result["valid"]:

                all_valid = False

    # --------------------------------------------------
    # 3. Produce final verification status.
    # --------------------------------------------------

    if all_valid:

        status = "VERIFIED_EVIDENCE"

        reason = (
            "All operands are anchored to "
            "valid evidence spans, values, "
            "periods, and units."
        )

    else:

        status = "INVALID_EVIDENCE"

        reason = (
            "One or more operands could not "
            "be verified against the cited evidence."
        )

    return {
        "valid": all_valid,
        "status": status,
        "reason": reason,
        "operands": operand_results
    }


if __name__ == "__main__":

    print(
        "validator.py loaded successfully."
    )