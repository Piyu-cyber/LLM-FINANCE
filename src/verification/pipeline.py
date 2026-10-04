from calculator import calculate, CalculationError
from validator import (
    load_corpus,
    build_chunk_lookup,
    validate_claim,
)


def run_verification_pipeline(
    claim,
    example_id,
):
    """
    Complete verification pipeline.

    Steps:

        1. Validate claim schema.
        2. Validate evidence.
        3. Extract verified operands.
        4. Resolve the calculation program.
        5. Execute calculation deterministically.
        6. Return verified result.
    """

    # --------------------------------------------------
    # Load corpus
    # --------------------------------------------------

    chunks = load_corpus()

    lookup = build_chunk_lookup(
        chunks
    )

    # --------------------------------------------------
    # Step 1 + 2:
    # Validate schema and evidence
    # --------------------------------------------------

    validation_result = validate_claim(
        claim,
        lookup,
        example_id,
    )

    if not validation_result["valid"]:

        return {
            "valid": False,
            "status": validation_result["status"],
            "reason": validation_result["reason"],
            "calculation": None,
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
        }

    # --------------------------------------------------
    # Execute program
    # --------------------------------------------------

    intermediate_results = {}

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
            }

        values = []

        for operand_id in operand_ids:

            # First look for an original operand.
            if operand_id in operands:

                values.append(
                    operands[operand_id]
                )

            # Then look for an intermediate result.
            elif operand_id in intermediate_results:

                values.append(
                    intermediate_results[
                        operand_id
                    ]
                )

            else:

                return {
                    "valid": False,
                    "status": "INVALID_PROGRAM",
                    "reason": (
                        f"Unknown operand "
                        f"'{operand_id}' "
                        f"in step {step_number}."
                    ),
                    "calculation": None,
                }

        # --------------------------------------------------
        # Deterministic calculation
        # --------------------------------------------------

        try:

            calculation = calculate(
                operation,
                values,
            )

        except CalculationError as e:

            return {
                "valid": False,
                "status": "CALCULATION_ERROR",
                "reason": str(e),
                "calculation": None,
            }

        # Store result.
        result_id = step.get(
            "id",
            f"step_{step_number}"
        )

        intermediate_results[
            result_id
        ] = calculation["result"]

    # --------------------------------------------------
    # Final calculation
    # --------------------------------------------------

    final_step_id = program[-1].get(
        "id",
        f"step_{len(program)}"
    )

    final_result = intermediate_results[
        final_step_id
    ]

    return {
        "valid": True,
        "status": "VERIFIED_RESULT",
        "reason": (
            "Evidence was verified and "
            "the calculation program was "
            "executed deterministically."
        ),
        "calculation": {
            "result": final_result,
            "steps": intermediate_results,
        },
    }