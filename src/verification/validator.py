import json
import re
from pathlib import Path

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from claim_schema import CLAIM_SCHEMA
from calculator import (
    to_decimal,
    is_expense_like,
    infer_direction,
    CalculationError,
)


CORPUS_PATH = Path(
    "data/finqa/train_chunks.jsonl"
)


# ---------------------------------------------------------------------------
# Directional financial keywords
# ---------------------------------------------------------------------------

INCREASE_KEYWORDS = {
    "increase",
    "increased",
    "increases",
    "increasing",
    "higher",
    "rose",
    "rising",
    "grew",
    "growth",
    "jumped",
    "up from",
    "surpassed",
    "gain",
    "expanded",
    "improvement",
    "improved",
    "expansion",
}

DECREASE_KEYWORDS = {
    "decrease",
    "decreased",
    "decreases",
    "decreasing",
    "lower",
    "fell",
    "falling",
    "declined",
    "decline",
    "dropped",
    "drop",
    "down from",
    "reduction",
    "reduced",
    "loss",
    "contracted",
    "contraction",
}


def load_corpus(path=CORPUS_PATH):
    """
    Load the FinQA retrieval corpus.

    Returns:
        list[dict]: All corpus chunks.
    """

    chunks = []
    p = Path(path)

    if not p.exists():
        return chunks

    with p.open(
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


class UnitCategory:
    USD = "usd"
    THOUSAND_USD = "thousand_usd"
    MILLION_USD = "million_usd"
    BILLION_USD = "billion_usd"
    PERCENT = "percent"
    SHARES = "shares"
    THOUSAND_SHARES = "thousand_shares"
    MILLION_SHARES = "million_shares"
    USD_PER_SHARE = "usd_per_share"
    UNKNOWN = "unknown"


def normalize_unit(unit_str: str) -> str:
    """Normalize user or claim unit string to canonical financial unit category."""
    if not unit_str or not isinstance(unit_str, str):
        return UnitCategory.UNKNOWN

    u = unit_str.lower().strip()

    if ("per share" in u or "/ share" in u or "/share" in u) and any(c in u for c in ["$", "usd", "dollar"]):
        return UnitCategory.USD_PER_SHARE

    if "%" in u or "percent" in u or "percentage" in u or "basis point" in u or "bps" in u:
        return UnitCategory.PERCENT

    if "share" in u or "shares" in u:
        if "billion" in u:
            return "billion_shares"
        if "million" in u:
            return UnitCategory.MILLION_SHARES
        if "thousand" in u:
            return UnitCategory.THOUSAND_SHARES
        return UnitCategory.SHARES

    if "billion" in u:
        return UnitCategory.BILLION_USD

    if "million" in u or "mm" in u.split():
        return UnitCategory.MILLION_USD

    if "thousand" in u or "k" in u.split():
        return UnitCategory.THOUSAND_USD

    if any(c in u for c in ["$", "usd", "dollar", "dollars", "us dollar", "us dollars", "us$"]):
        return UnitCategory.USD

    return u


def period_matches_evidence(
    claimed_period,
    evidence_text
) -> bool:
    """Check whether the claimed period/year occurs in the evidence using
    exact boundary-aware matching."""
    if claimed_period is None or not evidence_text:
        return False

    period_str = str(claimed_period).strip()
    if not period_str:
        return False

    evidence = str(evidence_text)

    # 1. Year range: e.g. "2008-2009", "2008 to 2009", "2008/2009"
    range_match = re.search(r"\b(19\d{2}|20\d{2})\s*(?:-|to|/|–)\s*(19\d{2}|20\d{2})\b", period_str, re.IGNORECASE)
    if range_match:
        y1, y2 = range_match.group(1), range_match.group(2)
        direct_range = rf"\b{re.escape(y1)}\s*(?:-|to|/|–)\s*{re.escape(y2)}\b"
        if re.search(direct_range, evidence, re.IGNORECASE):
            return True
        p1 = rf"\b{re.escape(y1)}\b"
        p2 = rf"\b{re.escape(y2)}\b"
        return bool(re.search(p1, evidence) and re.search(p2, evidence))

    # 2. 4-digit year: e.g. "2008", "2009", "1999"
    year_4digit = re.search(r"^(19\d{2}|20\d{2})$", period_str)
    if year_4digit:
        year = year_4digit.group(1)
        pattern = rf"(?:\bFY\s*|\b){re.escape(year)}\b"
        return bool(re.search(pattern, evidence, re.IGNORECASE))

    # 3. Fiscal year with prefix: e.g. "FY2009", "FY 2009", "FY09"
    fy_match = re.search(r"^FY\s*(\d{2,4})$", period_str, re.IGNORECASE)
    if fy_match:
        digits = fy_match.group(1)
        if len(digits) == 4:
            pattern = rf"(?:\bFY\s*|\b){re.escape(digits)}\b"
            return bool(re.search(pattern, evidence, re.IGNORECASE))
        elif len(digits) == 2:
            full_year = f"20{digits}"
            pattern = rf"(?:\bFY\s*|\b|\'|’){re.escape(digits)}\b|\b{re.escape(full_year)}\b"
            return bool(re.search(pattern, evidence, re.IGNORECASE))

    # 4. Quarterly period: e.g. "Q1 2009", "1Q 2009", "2009 Q1"
    q_match = re.search(r"^(?:(Q[1-4]|[1-4]Q)\s*(\d{2,4})|(\d{4})\s*(Q[1-4]|[1-4]Q))$", period_str, re.IGNORECASE)
    if q_match:
        pattern = rf"\b{re.escape(period_str)}\b"
        if re.search(pattern, evidence, re.IGNORECASE):
            return True

    # 5. Fallback for numeric periods: reject partial digits
    if period_str.isdigit():
        pattern = rf"(?<!\d){re.escape(period_str)}(?!\d)"
        return bool(re.search(pattern, evidence))

    # General word-boundary match
    pattern = rf"\b{re.escape(period_str)}\b"
    return bool(re.search(pattern, evidence, re.IGNORECASE))


def unit_matches_evidence(
    claimed_unit,
    evidence_text
) -> bool:
    """Perform robust financial unit consistency check."""
    if not claimed_unit or not evidence_text:
        return False

    cat = normalize_unit(claimed_unit)
    text = evidence_text.lower()

    has_usd_symbol = bool(re.search(r"(\$|usd\b|dollars?\b)", text))
    has_percent = bool(re.search(r"(%|percent\b|percentage\b|pct\b)", text))
    has_shares = bool(re.search(r"\bshares?\b", text))
    has_per_share = bool(re.search(r"(\bper\s+share|/\s*share)", text))

    has_billion = bool(re.search(r"\b(billions?|in\s+billions?|billions?\s+of\s+dollars?)\b", text))
    has_million = bool(re.search(r"\b(millions?|in\s+millions?|millions?\s+of\s+dollars?)\b", text))
    has_thousand = bool(re.search(r"\b(thousands?|in\s+thousands?|thousands?\s+of\s+dollars?)\b", text))

    if cat == UnitCategory.BILLION_USD:
        return has_billion and not (has_million and not has_billion) and not (has_shares and not has_billion)

    elif cat == UnitCategory.MILLION_USD:
        return has_million and not (has_billion and not has_million) and not (has_shares and not has_million)

    elif cat == UnitCategory.THOUSAND_USD:
        return has_thousand and not (has_million or has_billion) and not (has_shares and not has_thousand)

    elif cat == UnitCategory.USD:
        if not (has_usd_symbol or "$" in text):
            return False
        if has_billion or has_million or has_thousand:
            return False
        if has_percent or has_shares:
            return False
        return True

    elif cat == UnitCategory.PERCENT:
        return has_percent and not (has_shares and not has_percent)

    elif cat == UnitCategory.SHARES:
        if has_percent or (has_usd_symbol and not has_shares):
            return False
        return has_shares and not (has_million or has_thousand or has_billion)

    elif cat == UnitCategory.MILLION_SHARES:
        return has_shares and has_million

    elif cat == UnitCategory.THOUSAND_SHARES:
        return has_shares and has_thousand

    elif cat == UnitCategory.USD_PER_SHARE:
        return has_per_share and (has_usd_symbol or "$" in text)

    else:
        return bool(re.search(rf"\b{re.escape(str(claimed_unit).lower().strip())}\b", text))


def extract_narrative_direction(text, metric=None):
    """
    Extract financial direction from narrative evidence.

    If metric is provided (e.g. "interest expense"), prioritize sentences
    that explicitly discuss that metric.
    """
    if not text:
        return {"direction": "unknown", "keyword": None, "sentence": None}

    # Split into candidate sentences/clauses
    sentences = [s.strip() for s in re.split(r"[.!?\n|]", text) if s.strip()]

    metric_terms = []
    if metric:
        metric_terms = [m.lower().strip() for m in metric.split() if len(m) > 2]

    relevant_sentences = []
    other_sentences = []

    for s in sentences:
        s_lower = s.lower()
        if metric_terms and any(t in s_lower for t in metric_terms):
            relevant_sentences.append(s)
        else:
            other_sentences.append(s)

    ordered_sentences = relevant_sentences + other_sentences

    for s in ordered_sentences:
        tokens = set(re.findall(r"\b\w+\b", s.lower()))
        inc_hits = tokens.intersection(INCREASE_KEYWORDS)
        dec_hits = tokens.intersection(DECREASE_KEYWORDS)

        if inc_hits and not dec_hits:
            return {
                "direction": "increase",
                "keyword": sorted(list(inc_hits))[0],
                "sentence": s,
            }
        elif dec_hits and not inc_hits:
            return {
                "direction": "decrease",
                "keyword": sorted(list(dec_hits))[0],
                "sentence": s,
            }

    return {"direction": "unknown", "keyword": None, "sentence": None}


def validate_narrative_direction(
    claimed_value,
    evidence_texts,
    referent=None,
    claimed_direction=None,
    item_type=None,
):
    """
    Check consistency between calculated/claimed direction and narrative text.

    Catches the classic 'grounded-but-wrong' accounting inversion:
    e.g. naive -600 - (-511) = -89 (decrease) vs narrative 'interest expense increased'.
    """
    metric = referent.get("entity", "") if referent else ""
    if not item_type and referent:
        item_type = referent.get("entity", "")

    combined_text = " . ".join(evidence_texts)
    narrative_info = extract_narrative_direction(combined_text, metric=metric)
    narr_dir = narrative_info["direction"]

    if narr_dir == "unknown":
        return {
            "valid": True,
            "status": "DIRECTION_UNCONSTRAINED",
            "narrative_direction": "unknown",
            "calculated_direction": infer_direction(claimed_value),
            "reason": "No directional keywords found in narrative evidence.",
            "supporting_sentence": None,
            "keyword": None,
        }

    calc_dir = infer_direction(claimed_value)
    effective_claimed_dir = claimed_direction.lower() if claimed_direction else calc_dir

    if effective_claimed_dir == narr_dir:
        return {
            "valid": True,
            "status": "VERIFIED_DIRECTION",
            "narrative_direction": narr_dir,
            "calculated_direction": calc_dir,
            "reason": f"Narrative direction ('{narr_dir}') matches claimed/calculated change ({claimed_value}).",
            "supporting_sentence": narrative_info["sentence"],
            "keyword": narrative_info["keyword"],
        }

    # Detect accounting sign inversion on expenses/costs
    is_expense = is_expense_like(item_type) or (referent and is_expense_like(referent.get("entity", "")))

    if is_expense and calc_dir == "decrease" and narr_dir == "increase":
        return {
            "valid": False,
            "status": "DIRECTIONAL_CONTRADICTION",
            "narrative_direction": narr_dir,
            "calculated_direction": calc_dir,
            "reason": (
                f"Directional contradiction: Narrative evidence explicitly states that '{metric}' {narr_dir}d "
                f"('{narrative_info['sentence']}'), but the naive signed calculation yields a negative change "
                f"({claimed_value}), indicating a decrease. In accounting statements, expenses are recorded as negative outflows; "
                f"the economic change in expense magnitude is positive (e.g. +89 million USD)."
            ),
            "supporting_sentence": narrative_info["sentence"],
            "keyword": narrative_info["keyword"],
            "suggested_fix": "Use economic magnitude change (|curr| - |prev|) for expense referents.",
        }

    return {
        "valid": False,
        "status": "DIRECTIONAL_CONTRADICTION",
        "narrative_direction": narr_dir,
        "calculated_direction": calc_dir,
        "reason": (
            f"Directional contradiction: Narrative evidence indicates '{narr_dir}' "
            f"('{narrative_info['sentence']}'), but claimed/calculated result is {calc_dir} ({claimed_value})."
        ),
        "supporting_sentence": narrative_info["sentence"],
        "keyword": narrative_info["keyword"],
    }


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
    # --------------------------------------------------

    operand_period = operand.get(
        "period"
    )

    if operand_period:
        period_found = (
            period_matches_evidence(operand_period, span_text)
            or (operand.get("context_text") and period_matches_evidence(operand_period, operand["context_text"]))
            or (chunk_text and period_matches_evidence(operand_period, chunk_text))
        )

        if not period_found:

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
    example_id,
    context_texts=None,
    check_direction=True,
):
    """
    Validate a complete structured claim.

    Returns a verification result containing:

        - overall validity
        - verification status
        - explanation
        - individual operand results
        - directional check result (if applicable)
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
            "operands": [],
            "direction_check": None,
        }

    # --------------------------------------------------
    # 2. Validate every claim and operand.
    # --------------------------------------------------

    operand_results = []
    all_valid = True

    final_claim_id = claim.get("final_claim_id")
    final_claim = None

    for current_claim in claim["claims"]:
        if current_claim.get("id") == final_claim_id:
            final_claim = current_claim

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

    if not all_valid:
        return {
            "valid": False,
            "status": "INVALID_EVIDENCE",
            "reason": (
                "One or more operands could not "
                "be verified against the cited evidence."
            ),
            "operands": operand_results,
            "direction_check": None,
        }

    # --------------------------------------------------
    # 3. Optional Directional & Narrative Consistency Check
    # --------------------------------------------------
    direction_result = None

    if check_direction and final_claim:
        # Collect narrative passages from context_texts and cited chunks
        collected_texts = []
        if context_texts:
            if isinstance(context_texts, list):
                collected_texts.extend(context_texts)
            elif isinstance(context_texts, str):
                collected_texts.append(context_texts)

        for cid in final_claim.get("citations", []):
            chunk = find_chunk(lookup, example_id, cid)
            if chunk and chunk.get("text"):
                collected_texts.append(chunk["text"])

        for op in final_claim.get("operands", []):
            if op.get("context_text"):
                collected_texts.append(op["context_text"])

        if collected_texts:
            direction_result = validate_narrative_direction(
                claimed_value=final_claim.get("value"),
                evidence_texts=collected_texts,
                referent=final_claim.get("referent"),
                claimed_direction=final_claim.get("direction"),
            )

            if not direction_result["valid"]:
                return {
                    "valid": False,
                    "status": direction_result["status"],
                    "reason": direction_result["reason"],
                    "operands": operand_results,
                    "direction_check": direction_result,
                }

    # --------------------------------------------------
    # 4. Produce final verification status.
    # --------------------------------------------------

    status = "VERIFIED_EVIDENCE"
    reason = (
        "All operands are anchored to valid evidence spans, values, "
        "periods, and units"
    )
    if direction_result and direction_result["valid"]:
        reason += f", and narrative direction is verified ({direction_result['status']})."
    else:
        reason += "."

    return {
        "valid": True,
        "status": status,
        "reason": reason,
        "operands": operand_results,
        "direction_check": direction_result,
    }


if __name__ == "__main__":
    print("validator.py loaded successfully.")
