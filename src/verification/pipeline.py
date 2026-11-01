from calculator import calculate, CalculationError, is_expense_like
from validator import (
    load_corpus,
    build_chunk_lookup,
    validate_claim,
    validate_narrative_direction,
    find_chunk,
)


def run_verification_pipeline(
    claim,
    example_id,
    chunks=None,
    context_passages=None,
    resolve_economic_sign=False,
):
    """
    Complete verification pipeline with narrative direction check.

    Steps:
        1. Validate claim schema.
        2. Validate evidence operands (spans, values, periods, units).
        3. Extract verified operands.
        4. Resolve and execute calculation program deterministically.
        5. Verify narrative direction consistency (e.g. 'increased' vs negative change).
        6. Return verified result or actionable contradiction diagnostic.

    Parameters
    ----------
    claim : dict
        Structured claim object.
    example_id : str
        Target FinQA example identifier.
    chunks : list[dict] | None
        Optional corpus chunks. If None, loaded from disk or synthesized from operands.
    context_passages : list[str] | str | None
        Optional narrative passages for directional cross-checking.
    resolve_economic_sign : bool
        If True, automatically resolves expense negative signs to economic magnitude.
    """

    # --------------------------------------------------
    # Load corpus or synthesize lookup
    # --------------------------------------------------

    if chunks is None:
        chunks = load_corpus()

    lookup = build_chunk_lookup(chunks)

    # Fallback: if corpus not available on disk, synthesize from operand context_text
    if not lookup:
        for c in claim.get("claims", []):
            for op in c.get("operands", []):
                cid = op.get("chunk_id")
                ctxt = op.get("context_text") or op.get("span_text")
                if cid and ctxt:
                    lookup[(example_id, cid)] = {
                        "example_id": example_id,
                        "chunk_id": cid,
                        "text": ctxt,
                        "source": "claim_context",
                    }

    # --------------------------------------------------
    # Step 1 + 2:
    # Validate schema and evidence
    # --------------------------------------------------

    validation_result = validate_claim(
        claim,
        lookup,
        example_id,
        context_texts=context_passages,
        check_direction=False,  # Checked after calculation against final result
    )

    if not validation_result["valid"]:

        return {
            "valid": False,
            "status": validation_result["status"],
            "reason": validation_result["reason"],
            "calculation": None,
            "direction_check": None,
        }

    # --------------------------------------------------
    # Find final claim
    # --------------------------------------------------

    final_claim_id = claim[
        "final_claim_id"
    ]

    final_claim = None

    for current_claim in claim["claims"]:

        if current_claim["id"] == final_claim_id:

            final_claim = current_claim
            break

    if final_claim is None:

        return {
            "valid": False,
            "status": "INVALID_FINAL_CLAIM",
            "reason": (
                "final_claim_id does not "
                "refer to an existing claim."
            ),
            "calculation": None,
            "direction_check": None,
        }

    # --------------------------------------------------
    # Build operand lookup
    # --------------------------------------------------

    operands = {}

    for operand in final_claim["operands"]:

        operands[
            operand["id"]
        ] = operand["normalized_value"]

    # --------------------------------------------------
    # Get calculation program
    # --------------------------------------------------

    program = final_claim.get(
        "program",
        []
    )

    if not program:

        return {
            "valid": False,
            "status": "NO_PROGRAM",
            "reason": (
                "The final claim does not "
                "contain a calculation program."
            ),
            "calculation": None,
            "direction_check": None,
        }

    # --------------------------------------------------
    # Execute program
    # --------------------------------------------------

    intermediate_results = {}
    last_calculation_info = None

    for step_number, step in enumerate(
        program,
        start=1
    ):

        operation = step.get(
            "operation"
        )

        operand_ids = step.get(
            "operands",
            []
        )

        if not operation:

            return {
                "valid": False,
                "status": "INVALID_PROGRAM",
                "reason": (
                    f"Step {step_number} "
                    "does not specify an operation."
                ),
                "calculation": None,
                "direction_check": None,
            }

        values = []

        for operand_id in operand_ids:

            if operand_id in operands:
                values.append(operands[operand_id])
            elif operand_id in intermediate_results:
                values.append(intermediate_results[operand_id])
            else:
                return {
                    "valid": False,
                    "status": "INVALID_PROGRAM",
                    "reason": (
                        f"Unknown operand '{operand_id}' in step {step_number}."
                    ),
                    "calculation": None,
                    "direction_check": None,
                }

        # --------------------------------------------------
        # Deterministic calculation
        # --------------------------------------------------

        referent = final_claim.get("referent", {})
        item_type = referent.get("entity", "")

        try:
            calculation = calculate(
                operation,
                values,
                item_type=item_type,
            )
            last_calculation_info = calculation

        except CalculationError as e:

            return {
                "valid": False,
                "status": "CALCULATION_ERROR",
                "reason": str(e),
                "calculation": None,
                "direction_check": None,
            }

        result_id = step.get(
            "id",
            f"step_{step_number}"
        )

        intermediate_results[result_id] = calculation["result"]

    # --------------------------------------------------
    # Final calculation result
    # --------------------------------------------------

    final_step_id = program[-1].get(
        "id",
        f"step_{len(program)}"
    )

    final_result = intermediate_results[final_step_id]

    # --------------------------------------------------
    # Step 5: Narrative Direction & Economic Check
    # --------------------------------------------------
    collected_texts = []
    if context_passages:
        if isinstance(context_passages, list):
            collected_texts.extend(context_passages)
        elif isinstance(context_passages, str):
            collected_texts.append(context_passages)

    for cid in final_claim.get("citations", []):
        c = find_chunk(lookup, example_id, cid)
        if c and c.get("text"):
            collected_texts.append(c["text"])

    for op in final_claim.get("operands", []):
        if op.get("context_text"):
            collected_texts.append(op["context_text"])

    direction_check = validate_narrative_direction(
        claimed_value=final_result,
        evidence_texts=collected_texts,
        referent=final_claim.get("referent"),
        claimed_direction=final_claim.get("direction"),
    )

    if not direction_check["valid"]:
        # If directional contradiction is detected, check if accounting polarity resolution is requested
        if resolve_economic_sign and direction_check.get("suggested_fix"):
            if len(values) == 2 and (values[0] < 0 and values[1] < 0):
                economic_result = float(abs(values[0]) - abs(values[1]))
                recheck = validate_narrative_direction(
                    claimed_value=economic_result,
                    evidence_texts=collected_texts,
                    referent=final_claim.get("referent"),
                    claimed_direction="increase",
                )

                if recheck["valid"]:
                    return {
                        "valid": True,
                        "status": "VERIFIED_RESULT_ECONOMIC_RESOLVED",
                        "reason": (
                            f"Accounting sign resolved: Naive signed calculation yielded {final_result}, "
                            f"contradicting narrative statement ('{direction_check['supporting_sentence']}'). "
                            f"Resolved to economic magnitude change: +{economic_result:.0f} {final_claim.get('unit', '')}."
                        ),
                        "calculation": {
                            "result": economic_result,
                            "naive_signed_result": final_result,
                            "steps": intermediate_results,
                            "direction": "increase",
                        },
                        "direction_check": recheck,
                    }

        return {
            "valid": False,
            "status": direction_check["status"],
            "reason": direction_check["reason"],
            "calculation": {
                "result": final_result,
                "steps": intermediate_results,
            },
            "direction_check": direction_check,
        }

    return {
        "valid": True,
        "status": "VERIFIED_RESULT",
        "reason": (
            "Evidence was verified, calculation program was executed "
            f"deterministically, and narrative direction '{direction_check['narrative_direction']}' was confirmed."
        ),
        "calculation": {
            "result": final_result,
            "steps": intermediate_results,
            "direction": direction_check["calculated_direction"],
        },
        "direction_check": direction_check,
    }
